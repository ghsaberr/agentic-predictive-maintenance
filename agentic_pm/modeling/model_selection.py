# ============================================================
#  agentic_pm/modeling/model_selection.py   (PART 1 — CLASSICAL)
# ============================================================

from __future__ import annotations
import math
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, IsolationForest
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
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import optuna
except Exception:
    optuna = None

# Torch (sequence + autoencoder)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None

# -------------------------------------------------------------------
# LOGGER
# -------------------------------------------------------------------
logger = logging.getLogger("agentic_pm.model_selection")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------
# 1. Splitting helpers
# -------------------------------------------------------------------
def add_rel_cycle(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    max_cycle = df2.groupby("unit")["cycle"].transform("max")
    df2["rel_cycle"] = df2["cycle"] / (max_cycle + 1e-9)
    return df2


def make_time_splits_rel(df: pd.DataFrame, n_splits: int = 5):
    """
    Time-aware splits based on relative cycle.
    Produces leakage-free expanding-window folds.
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

        if len(train_idx) == 0 or len(val_idx) == 0:
            q_train = np.quantile(rel, (i + 1) / (n_splits + 2))
            q_val = np.quantile(rel, (i + 2) / (n_splits + 2))
            train_idx = idxs[rel <= q_train]
            val_idx = idxs[(rel > q_train) & (rel <= q_val)]

        folds.append((train_idx, val_idx))

    logger.info(f"Created {len(folds)} time-aware folds.")
    return folds


def per_unit_holdout(df: pd.DataFrame, holdout_frac: float = 0.3):
    """
    Hold out last fraction of cycles per unit.
    """
    df2 = add_rel_cycle(df)
    train_mask = df2["rel_cycle"] <= (1 - holdout_frac)
    val_mask = df2["rel_cycle"] > (1 - holdout_frac)
    return df2.index[train_mask].to_numpy(), df2.index[val_mask].to_numpy()


# -------------------------------------------------------------------
# 2. Baselines
# -------------------------------------------------------------------
def baseline_persistence_shift(df: pd.DataFrame):
    """
    Persistence RUL_t = previous RUL.
    """
    r = df.groupby("unit")["RUL"].shift(1)
    return r.fillna(method="bfill").fillna(method="ffill").values


def baseline_ma_linear_map(train_df, test_df, sensor_col="sensor_1", window=5):
    rolling_train = train_df.groupby("unit")[sensor_col].rolling(window, min_periods=1).mean()
    rolling_train = rolling_train.reset_index(level=0, drop=True)

    lr = LinearRegression()
    lr.fit(rolling_train.values.reshape(-1, 1), train_df["RUL"].values)

    rolling_test = test_df.groupby("unit")[sensor_col].rolling(window, min_periods=1).mean()
    rolling_test = rolling_test.reset_index(level=0, drop=True)

    return lr.predict(rolling_test.values.reshape(-1, 1))


# -------------------------------------------------------------------
# 3. Metrics
# -------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def precision_at_k_rul(y_true, y_pred, k=100):
    k = min(k, len(y_pred))
    pred_top = np.argsort(y_pred)[:k]
    true_top = np.argsort(y_true)[:k]
    return len(set(pred_top).intersection(set(true_top))) / k


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


# -------------------------------------------------------------------
# 4. Classical Models (RF, ElasticNet, LightGBM, XGBoost)
# -------------------------------------------------------------------
def fit_random_forest(X_train, y_train, params=None, rs=42):
    if params is None:
        params = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2}
    model = RandomForestRegressor(random_state=rs, **params)
    model.fit(X_train, y_train)
    return model


def fit_elasticnet(X_train, y_train, params=None):
    if params is None:
        params = {"alpha": 0.5, "l1_ratio": 0.6, "max_iter": 10000}
    model = ElasticNet(**params)
    model.fit(X_train, y_train)
    return model


def fit_lightgbm(
    X_train, y_train,
    X_val=None, y_val=None,
    params=None,
    rounds=1000,
    early=50
):
    if lgb is None:
        raise ImportError("LightGBM not installed")

    if params is None:
        params = {"objective": "regression", "metric": "l1", "verbosity": -1}

    dtrain = lgb.Dataset(X_train, label=y_train)

    if X_val is None:
        model = lgb.train(params, dtrain, num_boost_round=rounds)
        return model, {}

    dval = lgb.Dataset(X_val, label=y_val)

    callbacks = [lgb.early_stopping(stopping_rounds=early)]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dval],
        callbacks=callbacks
    )

    return model, {"best_iter": getattr(model, "best_iteration", None)}


def fit_xgboost(X_train, y_train, X_val=None, y_val=None, params=None):
    if xgb is None:
        raise ImportError("XGBoost not installed")

    if params is None:
        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    dtrain = xgb.DMatrix(X_train, label=y_train)

    if X_val is None:
        bst = xgb.train(params, dtrain)
        return bst

    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dval, "val")]

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=800,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False
    )
    return bst


# ============================================================
#  PART 2 — SEQUENCE MODELS (LSTM / GRU / MINI-TCN)
# ============================================================

# Only available if torch is installed
if torch is not None:

    # -------------------------------------------------------
    # 1) Window Builder (per-unit sliding windows)
    # -------------------------------------------------------
    def make_windows(df: pd.DataFrame, feature_cols: List[str],
                     seq_len: int = 50, stride: int = 1):
        """
        Convert dataframe → sliding windows per unit.

        Returns:
            X : (N, seq_len, F)
            y : (N,)
            units : (N,)
        """
        X_list, y_list, u_list = [], [], []

        for unit, g in df.groupby("unit"):
            arr = g[feature_cols].values
            rul = g["RUL"].values
            T = arr.shape[0]

            if T < seq_len:
                continue

            for start in range(0, T-seq_len+1, stride):
                end = start + seq_len
                X_list.append(arr[start:end])
                y_list.append(rul[end-1])   # single-step RUL
                u_list.append(unit)

        if len(X_list) == 0:
            return (np.zeros((0, seq_len, len(feature_cols))),
                    np.zeros((0,)),
                    np.zeros((0,)))

        X = np.stack(X_list)
        y = np.array(y_list)
        units = np.array(u_list)
        return X, y, units


    # -------------------------------------------------------
    # 2) PyTorch Dataset
    # -------------------------------------------------------
    class SequenceDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    # -------------------------------------------------------
    # 3) Lightweight LSTM
    # -------------------------------------------------------
    class LSTMRegressor(nn.Module):
        def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            h_last = h_n[-1]      # (B, hidden)
            return self.fc(h_last).squeeze(1)


    # -------------------------------------------------------
    # 4) Lightweight GRU
    # -------------------------------------------------------
    class GRURegressor(nn.Module):
        def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.1):
            super().__init__()
            self.gru = nn.GRU(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, h_n = self.gru(x)
            h_last = h_n[-1]
            return self.fc(h_last).squeeze(1)


    # -------------------------------------------------------
    # 5) Mini-TCN — Temporal Convolutional Network (very light)
    # -------------------------------------------------------
    class MiniTCN(nn.Module):
        """
        A tiny causal TCN:
        Conv1D → ReLU → Conv1D → GlobalAvgPool → FC
        """
        def __init__(self, n_features, hidden=32, kernel=3):
            super().__init__()
            self.conv1 = nn.Conv1d(n_features, hidden, kernel, padding=kernel-1)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel-1)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            # x : (B, T, F)
            x = x.transpose(1, 2)  # → (B, F, T)

            h = torch.relu(self.conv1(x))
            h = torch.relu(self.conv2(h))

            # Global average pooling across time
            h = h.mean(dim=2)     # (B, hidden)

            return self.fc(h).squeeze(1)


    # -------------------------------------------------------
    # 6) Generic Training Loop (MAE + Early Stopping)
    # -------------------------------------------------------
    def train_sequence_model(
        model, train_loader, val_loader,
        lr=1e-3, epochs=30, patience=5, device="cpu"
    ):
        """
        Generic trainer for LSTM / GRU / TCN.
        Uses MAE loss + early stopping.
        """
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_mae = float("inf")
        best_state = None
        no_improve = 0

        for ep in range(epochs):
            # ---- train
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = F.l1_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # ---- validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_losses.append(F.l1_loss(pred, yb).item())

            mae = float(np.mean(val_losses))
            logger.info(f"[SEQ] Epoch {ep:02d}  val_MAE={mae:.4f}")

            # early stopping
            if mae < best_mae:
                best_mae = mae
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, best_mae


    # -------------------------------------------------------
    # 7) Optuna objective for LSTM / GRU / TCN
    # -------------------------------------------------------
    def optuna_objective_sequence(
        trial,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        model_type: str = "lstm"
    ):
        """
        Optuna objective for sequence models.
        Supports: lstm, gru, tcn
        """

        seq_len = trial.suggest_int("seq_len", 30, 120)
        hidden = trial.suggest_int("hidden", 16, 128)
        layers = trial.suggest_int("layers", 1, 2)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # ---- build windows
        Xtr, ytr, _ = make_windows(train_df, feature_cols, seq_len=seq_len)
        Xva, yva, _ = make_windows(val_df, feature_cols, seq_len=seq_len)

        if len(Xtr) == 0 or len(Xva) == 0:
            return float("inf")

        train_loader = DataLoader(SequenceDataset(Xtr, ytr), batch_size=32, shuffle=True)
        val_loader   = DataLoader(SequenceDataset(Xva, yva), batch_size=32, shuffle=False)

        # ---- choose model
        n_features = len(feature_cols)
        if model_type == "lstm":
            model = LSTMRegressor(n_features, hidden, layers)
        elif model_type == "gru":
            model = GRURegressor(n_features, hidden, layers)
        elif model_type == "tcn":
            model = MiniTCN(n_features, hidden)
        else:
            raise ValueError("Unknown model_type")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        _, best_mae = train_sequence_model(
            model, train_loader, val_loader,
            lr=lr, epochs=20, patience=4, device=device
        )

        return best_mae

else:
    # Torch not available - placeholders
    def make_windows(*args, **kwargs):
        raise ImportError("Torch is required for sequence models")
