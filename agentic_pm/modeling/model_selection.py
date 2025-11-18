"""
agentic_pm/modeling/model_selection.py

Final, corrected utilities for:
- time-aware splits (per-unit relative-cycle)
- correct leakage-free baselines (persistence, moving-average linear map)
- model training wrappers (RF, ElasticNet, LightGBM)
- evaluation (MAE/RMSE/R2, Precision@k, Early-warning)
- Optuna tuning helper (LightGBM)
- lightweight time-aware randomized search for sklearn estimators

Notes / contracts:
- Input df must contain at least columns: ['unit','cycle','RUL'] + feature columns (numeric).
- X passed to training functions must be a numpy array aligned to df rows and must NOT contain unit/cycle/RUL.
"""

from __future__ import annotations
import math
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
import joblib

# optional dependencies
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import optuna
except Exception:
    optuna = None

# logger
logger = logging.getLogger("model_selection")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# ---------------------------
# Time-aware splitting helpers
# ---------------------------
def add_relative_cycle(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    max_cycle = df2.groupby("unit")["cycle"].transform("max")
    df2["rel_cycle"] = df2["cycle"] / (max_cycle + 1e-9)
    return df2


def per_unit_holdout(df: pd.DataFrame, holdout_frac: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a train/validation split by holding out the last `holdout_frac` portion
    of cycles **per unit** (common approach for CMAPSS papers).

    Returns:
        train_idx (np.ndarray of row indices), val_idx (np.ndarray)
    """
    assert 0.0 < holdout_frac < 1.0
    df2 = add_relative_cycle(df)
    train_mask = df2["rel_cycle"] <= (1 - holdout_frac)
    val_mask = df2["rel_cycle"] > (1 - holdout_frac)
    train_idx = df2.index[train_mask].to_numpy()
    val_idx = df2.index[val_mask].to_numpy()
    logger.info("Per-unit holdout: train rows=%d val rows=%d (holdout_frac=%.2f)",
                len(train_idx), len(val_idx), holdout_frac)
    return train_idx, val_idx


def make_time_splits_rel(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time-aware folds based on relative cycle (expanding-window-like).
    Uses thresholds spaced in (0.2, 0.95). Returns list of (train_idx, val_idx).
    """
    df2 = add_relative_cycle(df)
    rel = df2["rel_cycle"].values
    idxs = np.arange(len(df2))
    thresholds = np.linspace(0.2, 0.95, n_splits + 1)
    folds = []
    for i in range(n_splits):
        t_train = thresholds[i]
        t_val = thresholds[i + 1]
        train_idx = idxs[rel <= t_train]
        val_idx = idxs[(rel > t_train) & (rel <= t_val)]
        if len(train_idx) == 0 or len(val_idx) == 0:
            q_train = np.quantile(rel, max(0.01, (i + 1) / (n_splits + 2)))
            q_val = np.quantile(rel, min(0.999, (i + 2) / (n_splits + 2)))
            train_idx = idxs[rel <= q_train]
            val_idx = idxs[(rel > q_train) & (rel <= q_val)]
        folds.append((train_idx, val_idx))
    logger.info("Created %d time-aware folds", len(folds))
    return folds


# ---------------------------
# Baselines (fixed: no leakage)
# ---------------------------
def baseline_persistence_shift(df: pd.DataFrame) -> np.ndarray:
    """
    Persistence baseline (leakage-free): predict RUL_t = RUL_{t-1} per unit.
    For the first row of each unit, forward/backfill to avoid NaN.
    """
    r = df.groupby("unit")["RUL"].shift(1)
    r = r.fillna(method="bfill").fillna(method="ffill")
    return r.values


def baseline_ma_linear_map(train_df: pd.DataFrame, test_df: pd.DataFrame, sensor_col: str = "sensor_1", window: int = 5) -> np.ndarray:
    """
    Learn a linear map from rolling MA of a sensor to RUL using TRAIN rows,
    then apply the same MA transform on test rows to produce predictions.
    This avoids leakage because the regression is fit only on train.
    """
    ma_train = train_df.groupby("unit")[sensor_col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    lr = LinearRegression()
    lr.fit(ma_train.values.reshape(-1, 1), train_df["RUL"].values)
    ma_test = test_df.groupby("unit")[sensor_col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    preds = lr.predict(ma_test.values.reshape(-1, 1))
    return preds


# ---------------------------
# Evaluation metrics
# ---------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def precision_at_k_rul(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> float:
    k = min(k, len(y_pred))
    pred_topk = np.argsort(y_pred)[:k]
    true_topk = np.argsort(y_true)[:k]
    inter = len(set(pred_topk).intersection(set(true_topk)))
    return inter / k


def early_warning_rate(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, lead: int = 7) -> float:
    df2 = df.copy().reset_index(drop=True)
    df2["y_true"] = y_true
    df2["y_pred"] = y_pred
    units = df2["unit"].unique()
    success = 0
    total = 0
    for u in units:
        udf = df2[df2.unit == u].reset_index(drop=True)
        if udf.shape[0] == 0:
            continue
        if (udf["y_true"] == 0).any():
            failure_cycle = int(udf.loc[udf["y_true"] == 0, "cycle"].iloc[0])
        else:
            failure_cycle = int(udf["cycle"].max())
        cand = udf[udf["y_pred"] <= lead]
        total += 1
        if cand.shape[0] == 0:
            continue
        earliest_pred_cycle = int(cand["cycle"].iloc[0])
        if earliest_pred_cycle <= (failure_cycle - lead):
            success += 1
    return success / (total + 1e-9)


# ---------------------------
# Model training wrappers
# ---------------------------
def fit_random_forest(X_train: np.ndarray, y_train: np.ndarray, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
    if params is None:
        params = {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 2}
    rf = RandomForestRegressor(random_state=random_state, **params)
    rf.fit(X_train, y_train)
    return rf


def fit_elasticnet(X_train: np.ndarray, y_train: np.ndarray, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
    if params is None:
        params = {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 10000}
    en = ElasticNet(random_state=random_state, **params)
    en.fit(X_train, y_train)
    return en


def fit_lightgbm(
    X_train, y_train,
    X_val=None, y_val=None,
    params=None,
    num_boost_round=1000,
    early_stopping_rounds=50
):
    """
    Universal LightGBM trainer compatible with ALL LightGBM versions.
    Uses only callbacks → no deprecated parameters like verbose_eval / evals_result / early_stopping_rounds.
    """
    if lgb is None:
        raise ImportError("LightGBM is not installed.")

    if params is None:
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1
        }

    dtrain = lgb.Dataset(X_train, label=y_train)

    # No validation set → simple training
    if X_val is None or y_val is None:
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
        )
        return model, {}

    # With validation set
    dval = lgb.Dataset(X_val, label=y_val)

    # Version-safe early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds)
    ]

    # ⚠️ IMPORTANT:
    # No verbose_eval, no evals_result, no early_stopping_rounds kwarg.
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks
    )

    # safe output
    evals_result = {
        "best_iteration": getattr(model, "best_iteration", None)
    }

    return model, evals_result


# ---------------------------
# Optuna helpers (LightGBM)
# ---------------------------
def optuna_objective_lgb(trial, X: np.ndarray, y: np.ndarray, df: pd.DataFrame, n_splits: int = 3):
    if optuna is None:
        raise ImportError("optuna is not installed.")
    param = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.2),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
    }
    folds = make_time_splits_rel(df, n_splits=n_splits)
    maes = []
    for train_idx, val_idx in folds:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        bst, _ = fit_lightgbm(X_tr, y_tr, X_va, y_va, params=param, num_boost_round=500, early_stopping_rounds=30)
        y_pred = bst.predict(X_va)
        maes.append(mean_absolute_error(y_va, y_pred))
    return float(np.mean(maes))


def tune_lightgbm_optuna(X: np.ndarray, y: np.ndarray, df: pd.DataFrame, n_trials: int = 40):
    if optuna is None:
        raise ImportError("optuna is not installed.")
    study = optuna.create_study(direction="minimize")
    func = lambda trial: optuna_objective_lgb(trial, X, y, df)
    study.optimize(func, n_trials=n_trials)
    logger.info("Optuna tuning completed. Best value: %s", study.best_value)
    return study


# ---------------------------
# Nested CV utility
# ---------------------------
def nested_cv_evaluate(X: np.ndarray, y: np.ndarray, df: pd.DataFrame, outer_splits: int = 3, inner_trials: int = 20):
    if optuna is None:
        raise ImportError("optuna is not installed.")
    outer_folds = make_time_splits_rel(df, n_splits=outer_splits)
    results = []
    for i, (train_idx, val_idx) in enumerate(outer_folds):
        logger.info("Outer fold %d: train=%d val=%d", i, len(train_idx), len(val_idx))
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        study = optuna.create_study(direction="minimize")
        func = lambda trial: optuna_objective_lgb(trial, X_tr, y_tr, df_tr, n_splits=3)
        study.optimize(func, n_trials=inner_trials)
        best_params = study.best_params
        logger.info("Fold %d best params: %s", i, best_params)
        bst, _ = fit_lightgbm(X_tr, y_tr, X_va, y_va, params=best_params, num_boost_round=1000, early_stopping_rounds=50)
        y_pred = bst.predict(X_va)
        metrics = regression_metrics(y_va, y_pred)
        ew = early_warning_rate(df.iloc[val_idx], y_va, y_pred, lead=7)
        results.append({"outer_fold": i, "metrics": metrics, "early_warning_7": float(ew), "best_params": best_params})
    return results


# ---------------------------
# Randomized search light wrapper (time-aware)
# ---------------------------
def time_aware_random_search(estimator_cls, base_params: Dict[str, Any], param_distributions: Dict[str, List[Any]], df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                             n_iter: int = 20, n_splits: int = 5, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=rng))
    folds = make_time_splits_rel(df, n_splits=n_splits)
    best_score = float("inf")
    best_model = None
    results = []
    for params in param_list:
        scores = []
        for train_idx, val_idx in folds:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            model = estimator_cls(**{**base_params, **params})
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            scores.append(mean_absolute_error(y_va, y_pred))
        mean_score = float(np.mean(scores))
        results.append({"params": params, "mean_mae": mean_score})
        if mean_score < best_score:
            best_score = mean_score
            best_model = model
    logger.info("Time-aware random search done. best MAE=%.4f", best_score)
    return best_model, results


# ---------------------------
# Save/load helpers
# ---------------------------
def save_sklearn_model(model, path: str):
    joblib.dump(model, path)
    logger.info("Saved sklearn model to %s", path)


def load_sklearn_model(path: str):
    return joblib.load(path)


# ---------------------------
# Quick workflow helper
# ---------------------------
def quick_train_eval_rf(X: np.ndarray, y: np.ndarray, df: pd.DataFrame, holdout_frac: float = 0.3):
    train_idx, val_idx = per_unit_holdout(df, holdout_frac=holdout_frac)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    rf = fit_random_forest(X_tr, y_tr)
    y_pred = rf.predict(X_va)
    metrics = regression_metrics(y_va, y_pred)
    return rf, metrics, val_idx, y_va, y_pred
