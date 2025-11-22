"""
Nested cross-validation workflow for CMAPSS predictive maintenance.

This file:
- Does NOT implement any model internally.
- Uses model_selection.py as the single source of truth.
- Supports classical models, LSTM, anomaly-based predictors.
- Laptop-friendly, research-grade, no leakage.

Usage:
    from agentic_pm.modeling.nested_cv import nested_cv

"""

from __future__ import annotations
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from .model_selection import (
    ModelRegistry,
    make_unit_folds,
    regression_metrics,
    precision_at_k_rul,
    early_warning_rate,
    make_windows,          # For LSTM
    SequenceDataset,
)

# Optional deps
try:
    import optuna
except Exception:
    optuna = None

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:
    torch = None


logger = logging.getLogger("agentic_pm.nested_cv")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)



# ---------------------------------------------------------------------
# INNER TUNING: dynamic based on model_type
# ---------------------------------------------------------------------
def run_inner_tuning(
    model_type: str,
    train_df: pd.DataFrame,
    feature_cols: List[str],
    inner_splits: int = 3,
    inner_trials: int = 20,
) -> Dict:
    """
    Wrapper that calls appropriate optuna tuner from ModelRegistry.

    Model types:
        - "lgbm"
        - "rf"
        - "elasticnet"
        - "lstm"
        - "iforest"
    """

    registry = ModelRegistry()

    if model_type == "lgbm":
        logger.info("   🔍 Inner Optuna tuning for LightGBM ...")
        study = registry.tune(model_type, train_df=train_df,
                              feature_cols=feature_cols,
                              n_trials=inner_trials,
                              n_splits=inner_splits)
        return study.best_params

    elif model_type == "lstm":
        logger.info("   🔍 Inner Optuna tuning for LSTM ...")
        study = registry.tune(model_type, train_df=train_df,
                              feature_cols=feature_cols,
                              n_trials=inner_trials,
                              n_splits=inner_splits)
        return study.best_params

    elif model_type in ("rf", "elasticnet", "iforest"):
        # light tuning using random search (sklearn models)
        logger.info(f"   🔍 Inner tuning (random search) for {model_type}")
        return registry.random_search(model_type, train_df=train_df,
                                      feature_cols=feature_cols, n_iter=inner_trials)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")



# ---------------------------------------------------------------------
# OUTER LOOP
# ---------------------------------------------------------------------
def nested_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RUL",
    outer_splits: int = 3,
    inner_splits: int = 3,
    inner_trials: int = 20,
    model_type: str = "lgbm",     # lgbm | rf | elasticnet | lstm | iforest
) -> Tuple[pd.DataFrame, Dict]:
    """
    Top-level nested CV:
        - Outer folds created per unit to avoid leakage.
        - Inner folds model-specific tuning.
        - Supports sequence models (LSTM) and tabular models.

    Returns:
        - df_results: per-fold metrics + best_params
        - avg_metrics: aggregated performance
    """

    logger.info(f"🔵 Starting Nested CV with {outer_splits} outer folds (model={model_type})")

    registry = ModelRegistry()
    outer_folds = make_unit_folds(df, n_splits=outer_splits)

    results = []

    for fold_id, (train_idx, val_idx) in enumerate(outer_folds):
        logger.info("\n" + "="*30)
        logger.info(f"🟣 Outer Fold {fold_id+1}/{outer_splits}")
        logger.info("="*30)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        # ---------------- INNER OPTUNA TUNING ----------------
        best_params = run_inner_tuning(
            model_type,
            train_df=train_df,
            feature_cols=feature_cols,
            inner_splits=inner_splits,
            inner_trials=inner_trials,
        )
        logger.info(f"   ✔ Best inner params: {best_params}")

        # ---------------- TRAIN FINAL MODEL ----------------
        if model_type == "lstm":
            if torch is None:
                raise ImportError("PyTorch is required for LSTM model.")

            seq_len = best_params.get("seq_len", 50)
            logger.info(f"   📦 Building LSTM windows (seq_len={seq_len})")

            Xtr, ytr, _ = make_windows(train_df, feature_cols, seq_len=seq_len)
            Xva, yva, _ = make_windows(val_df,   feature_cols, seq_len=seq_len)

            if len(Xtr) == 0 or len(Xva) == 0:
                logger.warning("Sequence windowing produced empty dataset. Skipping fold.")
                continue

            batch = 64 if len(Xtr) >= 64 else 16
            train_loader = DataLoader(SequenceDataset(Xtr, ytr),
                                      batch_size=batch, shuffle=True)
            val_loader   = DataLoader(SequenceDataset(Xva, yva),
                                      batch_size=batch, shuffle=False)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, best_val = registry.fit(
                model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                best_params=best_params,
                n_features=len(feature_cols),
                device=device
            )

            y_pred = registry.predict(model_type, Xva, trained_model=model)

        else:
            # TABULAR MODELS
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_val   = val_df[feature_cols].values
            y_val   = val_df[target_col].values

            model = registry.fit(
                model_type,
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                params=best_params
            )
            y_pred = registry.predict(model_type, X_val, trained_model=model)
            yva    = y_val  # for metrics

        # ---------------- METRICS ----------------
        met = regression_metrics(yva, y_pred)
        prec = precision_at_k_rul(yva, y_pred, k=100)
        warn = early_warning_rate(val_df, yva, y_pred, lead=7)

        row = {
            "outer_fold": fold_id,
            **met,
            "Precision@100": prec,
            "EarlyWarning@7": warn,
            "best_params": best_params,
        }
        results.append(row)

    df_results = pd.DataFrame(results)
    avg = df_results.drop(columns=["best_params"]).mean(numeric_only=True).to_dict()

    logger.info("\n" + "="*30)
    logger.info("📌 Nested CV Completed")
    logger.info("="*30)
    logger.info(df_results)
    logger.info(f"\n📌 Average metrics across outer folds:\n{avg}\n")

    return df_results, avg
