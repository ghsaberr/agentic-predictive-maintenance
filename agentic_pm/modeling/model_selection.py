# agentic_pm/modeling/model_selection.py
"""
Core utilities for model selection and training (CMAPSS-friendly).

Contains:
- Splitting helpers (relative-cycle and unit-based)
- Baselines (persistence, MA->linear)
- Metrics (MAE/RMSE/R2, Precision@k, Early-warning)
- Model wrappers: RandomForest, ElasticNet, LightGBM (version-safe)
- LSTM glue (PyTorch): windowing, Dataset, model, train loop, optuna objective
- Save/load helpers

Notes:
- Optional dependencies: lightgbm, optuna, torch. Code checks availability.
- Functions are written to be importable and used by orchestration code (nested_cv.py).
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

# Optional deps
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import optuna
except Exception:
    optuna = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None

# Logger
logger = logging.getLogger("agentic_pm.model_selection")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# ---------------------------
# Splitting helpers
# ---------------------------
def add_rel_cycle(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    max_cycle = df2.groupby("unit")["cycle"].transform("max")
    df2["rel_cycle"] = df2["cycle"] / (max_cycle + 1e-9)
    return df2


def make_time_splits_rel(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Time-aware folds based on relative cycle (expanding-window style).
    Returns list of (train_idx, val_idx) row-index arrays.
    """
    df2 = add_rel_cycle(df)
    rel = df2["rel_cycle"].values
    idxs = np.arange(len(df2))
    thresholds = np.linspace(0.2, 0.95, n_splits + 1)
    folds = []
    for i in range(n_splits):
        t_train = thresholds[i]
        t_val = thresholds[i + 1]
        train_idx = idxs[rel <= t_train]
        val_idx = idxs[(rel > t_train) & (rel <= t_val)]
        # fallback if empty
        if len(train_idx) == 0 or len(val_idx) == 0:
            q_train = np.quantile(rel, max(0.01, (i + 1) / (n_splits + 2)))
            q_val = np.quantile(rel, min(0.999, (i + 2) / (n_splits + 2)))
            train_idx = idxs[rel <= q_train]
            val_idx = idxs[(rel > q_train) & (rel <= q_val)]
        folds.append((train_idx, val_idx))
    logger.info("Created %d time-aware folds", len(folds))
    return folds


def make_unit_folds(df: pd.DataFrame, n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create outer folds *by unit* to avoid leakage:
    - Split unit IDs into n_splits segments (by ordering or randomly if desired).
    Returns list of (train_idx, val_idx) row-index arrays.
    """
    units = sorted(df["unit"].unique())
    n_units = len(units)
    folds = []
    # split units into contiguous blocks for val (to mimic temporal holdout)
    step = int(np.ceil(n_units / n_splits))
    for i in range(n_splits):
        val_units = units[i * step:(i + 1) * step]
        if len(val_units) == 0:
            continue
        train_units = [u for u in units if u not in val_units]
        train_idx = df[df["unit"].isin(train_units)].index.to_numpy()
        val_idx = df[df["unit"].isin(val_units)].index.to_numpy()
        folds.append((train_idx, val_idx))
    logger.info("Created %d unit-based folds (units=%d)", len(folds), n_units)
    return folds


def per_unit_holdout(df: pd.DataFrame, holdout_frac: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hold out the last `holdout_frac` of cycles per unit (leakage-free).
    Returns train_idx, val_idx (row indices).
    """
    assert 0.0 < holdout_frac < 1.0
    df2 = add_rel_cycle(df)
    train_mask = df2["rel_cycle"] <= (1 - holdout_frac)
    val_mask = df2["rel_cycle"] > (1 - holdout_frac)
    train_idx = df2.index[train_mask].to_numpy()
    val_idx = df2.index[val_mask].to_numpy()
    logger.info("Per-unit holdout: train rows=%d val rows=%d (holdout_frac=%.2f)",
                len(train_idx), len(val_idx), holdout_frac)
    return train_idx, val_idx


# ---------------------------
# Baselines
# ---------------------------
def baseline_persistence_shift(df: pd.DataFrame) -> np.ndarray:
    """
    Leakage-free persistence: predict RUL_t = previous RUL (shift(1) per-unit).
    Fill the first entry per unit by forward/backfill.
    """
    r = df.groupby("unit")["RUL"].shift(1)
    r = r.fillna(method="bfill").fillna(method="ffill")
    return r.values


def baseline_ma_linear_map(train_df: pd.DataFrame, test_df: pd.DataFrame, sensor_col: str = "sensor_1", window: int = 5) -> np.ndarray:
    """
    Fit linear regression mapping rolling mean(sensor_col) -> RUL on train, apply to test.
    """
    ma_train = train_df.groupby("unit")[sensor_col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    lr = LinearRegression()
    lr.fit(ma_train.values.reshape(-1, 1), train_df["RUL"].values)
    ma_test = test_df.groupby("unit")[sensor_col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    preds = lr.predict(ma_test.values.reshape(-1, 1))
    return preds


# ---------------------------
# Metrics
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
    """
    For each unit: success if the model predicts RUL <= lead at least
    'lead' cycles before actual failure.
    """
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
# Model wrappers
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
    Version-safe LightGBM trainer using callbacks only.
    Returns (model, info_dict).
    """
    if lgb is None:
        raise ImportError("lightgbm is not installed.")

    if params is None:
        params = {"objective": "regression", "metric": "mae", "verbosity": -1}

    dtrain = lgb.Dataset(X_train, label=y_train)

    if X_val is None or y_val is None:
        model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
        return model, {}

    dval = lgb.Dataset(X_val, label=y_val)
    callbacks = []
    try:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
    except Exception:
        # older versions may have different signature
        callbacks.append(lambda env: None)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=callbacks
    )
    info = {"best_iteration": getattr(model, "best_iteration", None)}
    return model, info


# ---------------------------
# Optuna for LightGBM (helper)
# ---------------------------
def optuna_objective_lgb(trial, X: np.ndarray, y: np.ndarray, df: pd.DataFrame, n_splits: int = 3):
    if optuna is None:
        raise ImportError("optuna is not installed.")
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
    folds = make_time_splits_rel(df, n_splits=n_splits)
    maes = []
    for train_idx, val_idx in folds:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        bst, _ = fit_lightgbm(X_tr, y_tr, X_va, y_va, params=params, num_boost_round=800, early_stopping_rounds=30)
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
# LSTM utilities (PyTorch)
# ---------------------------
def make_windows(df: pd.DataFrame, feature_cols: List[str], seq_len: int = 50, stride: int = 1):
    """
    Convert dataframe to sliding windows per unit.
    Returns X (N, seq_len, F), y (N,), units (N,)
    """
    X_list, y_list, u_list = [], [], []
    for unit, g in df.groupby("unit"):
        arr = g[feature_cols].values
        rul = g["RUL"].values
        T = arr.shape[0]
        if T < seq_len:
            continue
        for start in range(0, T - seq_len + 1, stride):
            end = start + seq_len
            X_list.append(arr[start:end])
            y_list.append(rul[end - 1])
            u_list.append(unit)
    if len(X_list) == 0:
        return np.zeros((0, seq_len, len(feature_cols))), np.zeros((0,)), np.zeros((0,))
    X = np.stack(X_list)
    y = np.array(y_list)
    units = np.array(u_list)
    return X, y, units


if torch is not None:
    class SequenceDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class LSTMRegressor(nn.Module):
        def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x: (B, T, F)
            _, (h_n, _) = self.lstm(x)
            last = h_n[-1]  # (B, hidden)
            return self.fc(last).squeeze(1)

    def train_lstm(model: nn.Module, train_loader, val_loader,
                   lr: float = 1e-3, epochs: int = 30, device: str = "cpu", patience: int = 6):
        """
        Train a PyTorch LSTM with L1 loss (MAE) and simple early stopping.
        Returns trained model and best validation MAE.
        """
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        best_val = float("inf")
        best_state = None
        cur_pat = 0
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = F.l1_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_losses.append(loss.item())
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_losses.append(F.l1_loss(pred, yb).item())
            val_mae = float(np.mean(val_losses)) if len(val_losses) else float("inf")
            logger.info("Epoch %d train_mae %.4f val_mae %.4f", epoch, np.mean(train_losses) if train_losses else np.nan, val_mae)
            if val_mae < best_val:
                best_val = val_mae
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                cur_pat = 0
            else:
                cur_pat += 1
            if cur_pat >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_val

    def optuna_objective_lstm(trial, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]):
        """
        Optuna objective for LSTM: suggests seq_len, hidden_size, layers, lr.
        Builds datasets and runs a short train.
        """
        if optuna is None:
            raise ImportError("optuna not installed")
        seq_len = trial.suggest_int("seq_len", 50, 200)
        hidden = trial.suggest_int("hidden_size", 32, 128)
        layers = trial.suggest_int("num_layers", 1, 2)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

        Xtr, ytr, _ = make_windows(train_df, feature_cols, seq_len=seq_len)
        Xv, yv, _ = make_windows(val_df, feature_cols, seq_len=seq_len)
        if Xtr.shape[0] == 0 or Xv.shape[0] == 0:
            return float("inf")

        batch = 64 if Xtr.shape[0] >= 64 else 16
        train_loader = DataLoader(SequenceDataset(Xtr, ytr), batch_size=batch, shuffle=True)
        val_loader = DataLoader(SequenceDataset(Xv, yv), batch_size=batch, shuffle=False)

        model = LSTMRegressor(n_features=len(feature_cols), hidden_size=hidden, num_layers=layers)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, best_val = train_lstm(model, train_loader, val_loader, lr=lr, epochs=25, device=device, patience=5)
        return best_val

else:
    # Placeholders if torch not installed
    SequenceDataset = None
    LSTMRegressor = None
    def train_lstm(*args, **kwargs):
        raise ImportError("PyTorch is not installed; LSTM functions are unavailable.")
    def optuna_objective_lstm(*args, **kwargs):
        raise ImportError("PyTorch/Optuna not installed; LSTM optuna objective is unavailable.")


# ---------------------------
# Utilities: save/load
# ---------------------------
def save_sklearn_model(model, path: str):
    joblib.dump(model, path)
    logger.info("Saved sklearn model to %s", path)


def load_sklearn_model(path: str):
    return joblib.load(path)
