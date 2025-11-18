"""
agentic_pm/modeling/nested_cv.py

Nested Time-Series Cross-Validation for CMAPSS RUL
- Outer folds → unbiased generalization estimate
- Inner folds → Optuna hyperparameter tuning
- Metrics: MAE, RMSE, R2, Precision@K, Early-warning
"""

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from typing import List, Tuple, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------------------
# 1) Time-aware Relative Cycle Splitting
# -------------------------------------------------------------
def add_rel_cycle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    max_c = df.groupby("unit")["cycle"].transform("max")
    df["rel_cycle"] = df["cycle"] / (max_c + 1e-9)
    return df


def make_outer_splits(df: pd.DataFrame, n_splits=3):
    """
    Outer splits:
    Example for 3 folds:
        train: <= 0.40   val: 0.40–0.60
        train: <= 0.60   val: 0.60–0.80
        train: <= 0.80   val: 0.80–1.00
    """
    df2 = add_rel_cycle(df)
    rel = df2["rel_cycle"].values
    idxs = np.arange(len(df2))

    thresholds = np.linspace(0.4, 1.0, n_splits + 1)

    folds = []
    for i in range(n_splits):
        t_train = thresholds[i]
        t_val = thresholds[i + 1]

        train_idx = idxs[rel <= t_train]
        val_idx = idxs[(rel > t_train) & (rel <= t_val)]

        if len(val_idx) == 0:
            continue

        folds.append((train_idx, val_idx))

    return folds


def make_inner_splits(df: pd.DataFrame, n_splits=3):
    """
    Inner folds for Optuna tuning.
    Same structure but smaller ranges.
    """
    df2 = add_rel_cycle(df)
    rel = df2["rel_cycle"].values
    idxs = np.arange(len(df2))

    thresholds = np.linspace(0.2, 1.0, n_splits + 1)

    folds = []
    for i in range(n_splits):
        t_train = thresholds[i]
        t_val = thresholds[i + 1]

        train_idx = idxs[rel <= t_train]
        val_idx = idxs[(rel > t_train) & (rel <= t_val)]

        if len(val_idx) == 0:
            continue

        folds.append((train_idx, val_idx))

    return folds


# -------------------------------------------------------------
# 2) Metrics
# -------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def precision_at_k(y_true, y_pred, k=100):
    k = min(len(y_pred), k)
    pred_top = np.argsort(y_pred)[:k]
    true_top = np.argsort(y_true)[:k]
    return len(set(pred_top).intersection(true_top)) / k


def early_warning_rate(df, y_true, y_pred, lead=7):
    df2 = df.copy().reset_index(drop=True)
    df2["true"] = y_true
    df2["pred"] = y_pred

    units = df2["unit"].unique()
    success = 0
    total = 0

    for u in units:
        tmp = df2[df2.unit == u].reset_index()

        if (tmp["true"] == 0).any():
            fail_cycle = int(tmp.loc[tmp["true"] == 0, "cycle"].iloc[0])
        else:
            fail_cycle = int(tmp["cycle"].max())

        cand = tmp[tmp["pred"] <= lead]
        if len(cand) == 0:
            total += 1
            continue

        earliest = int(cand["cycle"].iloc[0])
        if earliest <= fail_cycle - lead:
            success += 1
        total += 1

    return success / (total + 1e-9)


# -------------------------------------------------------------
# 3) Optuna objective for LightGBM
# -------------------------------------------------------------
def optuna_objective(trial, X, y, df, inner_splits):
    # [FIXED 1] suggest_loguniform is deprecated
    params = {
        "objective": "regression",
        "metric": "l1",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 16, 200),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 80),
    }

    maes = []
    
    for train_idx, val_idx in inner_splits:
        Xtr, Xv = X[train_idx], X[val_idx]
        ytr, yv = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(Xtr, label=ytr)
        dval = lgb.Dataset(Xv, label=yv)
        
        # Callbacks for logging and early stopping
        callbacks = [lgb.log_evaluation(period=0)] 
        callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))
        
        # [FIXED 2] Removed deprecated arguments from lgb.train
        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=800,
            valid_sets=[dval],
            callbacks=callbacks
        )

        yp = bst.predict(Xv)
        mae = mean_absolute_error(yv, yp)
        maes.append(mae)

    return np.mean(maes)

# -------------------------------------------------------------
# 4) Nested CV Main Function
# -------------------------------------------------------------
def nested_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RUL",
    outer_splits: int = 3,
    inner_splits: int = 3,
    inner_trials: int = 20,
):
    """
    Runs the full Nested Time-Aware Cross-Validation pipeline.
    """

    results = []
    outer = make_outer_splits(df, n_splits=outer_splits)

    X_full = df[feature_cols].values
    y_full = df[target_col].values

    print(f"🔵 Starting Nested CV with {outer_splits} outer folds")

    for fold_id, (tr_idx, va_idx) in enumerate(outer):
        print(f"\n==============================")
        print(f"🟣 Outer Fold {fold_id+1}/{outer_splits}")
        print("==============================")

        X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
        X_va, y_va = X_full[va_idx], y_full[va_idx]

        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        # ---- INNER LOOP TUNING ----
        inner_fold_idxs = make_inner_splits(df_tr, n_splits=inner_splits)

        print("   🔍 Inner Optuna tuning running...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: optuna_objective(trial, X_tr, y_tr, df_tr, inner_fold_idxs),
            n_trials=inner_trials,
        )

        best_params = study.best_params
        print(f"   ✔ Best inner params: {best_params}")

        # ---- TRAIN FINAL MODEL ON OUTER TRAIN [FIXED] ----
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va)
        
        # Callbacks for final training
        callbacks = [lgb.log_evaluation(period=0)]
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

        bst = lgb.train(
            {**best_params, "objective": "regression", "metric": "l1"},
            dtrain,
            num_boost_round=1500,
            valid_sets=[dval],
            callbacks=callbacks # [FIXED] Use callbacks
        )

        y_pred = bst.predict(X_va)

        # ---- METRICS ----
        mets = regression_metrics(y_va, y_pred)
        p100 = precision_at_k(y_va, y_pred, k=100)
        ew7 = early_warning_rate(df_va, y_va, y_pred, lead=7)

        res = {
            "outer_fold": fold_id,
            "MAE": mets["MAE"],
            "RMSE": mets["RMSE"],
            "R2": mets["R2"],
            "Precision@100": p100,
            "EarlyWarning@7": ew7,
            "best_params": best_params,
        }
        results.append(res)

    # ---- Final Summary ----
    df_res = pd.DataFrame(results)
    avg = df_res.mean(numeric_only=True).to_dict()

    print("\n==============================")
    print("📌 Nested CV Completed")
    print("==============================")
    print(df_res)
    print("\n📌 Average metrics across outer folds:")
    print(avg)

    return results, avg