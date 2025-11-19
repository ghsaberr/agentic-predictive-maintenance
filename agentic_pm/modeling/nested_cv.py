# agentic_pm/modeling/nested_cv.py
"""
Workflow: Nested cross-validation (unit-based) for CMAPSS RUL.
Uses primitives from model_selection.py (imported).
Supports nested CV for LightGBM (default) or LSTM (if torch available).
"""

from __future__ import annotations
import pprint
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from agentic_pm.modeling import model_selection as ms

try:
    import optuna
except Exception:
    optuna = None

def nested_cv_workflow(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RUL",
    outer_splits: int = 3,
    inner_splits: int = 3,
    inner_trials: int = 20,
    model_type: str = "lightgbm",  # "lightgbm" or "lstm"
    unit_based: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Run nested CV:
    - outer: unit-based folds (recommended, unit leakage-free)
    - inner: optuna tuning (LightGBM or LSTM)
    Returns (results_list_per_outer_fold, averaged_metrics)
    """
    if unit_based:
        outer_folds = ms.make_unit_folds(df, n_splits=outer_splits)
    else:
        outer_folds = ms.make_time_splits_rel(df, n_splits=outer_splits)

    X_full = df[feature_cols].values
    y_full = df[target_col].values

    results = []

    print(f"Starting nested CV: model_type={model_type}, outer_splits={len(outer_folds)}, inner_trials={inner_trials}")

    for i, (train_idx, val_idx) in enumerate(outer_folds):
        print("\n" + "=" * 60)
        print(f"Outer fold {i+1}/{len(outer_folds)} | train rows {len(train_idx)} val rows {len(val_idx)}")
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_va = df.iloc[val_idx].reset_index(drop=True)
        X_tr, y_tr = X_full[train_idx], y_full[train_idx]
        X_va, y_va = X_full[val_idx], y_full[val_idx]

        # ---- INNER tuning ----
        best_params = None
        if model_type.lower() == "lightgbm":
            if optuna is None:
                raise ImportError("optuna is required for LightGBM tuning but not available.")
            study = optuna.create_study(direction="minimize")
            func = lambda trial: ms.optuna_objective_lgb(trial, X_tr, y_tr, df_tr, n_splits=inner_splits)
            study.optimize(func, n_trials=inner_trials)
            best_params = study.best_params
            print("Inner best params (LGBM):")
            pprint.pprint(best_params)
            # train final on outer train
            bst, info = ms.fit_lightgbm(X_tr, y_tr, X_va, y_va, params={**best_params, "objective":"regression","metric":"l1"}, num_boost_round=1500, early_stopping_rounds=50)
            y_pred = bst.predict(X_va)
        elif model_type.lower() == "lstm":
            if ms.torch is None:
                raise ImportError("PyTorch is required for LSTM but not available.")
            # inner tuning using optuna_objective_lstm on df_tr with internal inner splits
            if optuna is None:
                raise ImportError("optuna is required for LSTM tuning but not available.")
            # create unit-level inner train/val split from df_tr
            inner_units = sorted(df_tr["unit"].unique())
            n_inner = max(2, min(inner_splits, len(inner_units)//2))
            # simple split: last 20% units as val for inner tuning
            n_val_units = max(1, int(0.2 * len(inner_units)))
            def inner_obj(trial):
                # split df_tr into train/val for inner tuning
                # (here we use per-unit holdout within df_tr)
                tr_idx_inner, va_idx_inner = ms.per_unit_holdout(df_tr, holdout_frac=0.2)
                tr_df_inner = df_tr.iloc[tr_idx_inner].reset_index(drop=True)
                va_df_inner = df_tr.iloc[va_idx_inner].reset_index(drop=True)
                return ms.optuna_objective_lstm(trial, tr_df_inner, va_df_inner, feature_cols)
            study = optuna.create_study(direction="minimize")
            study.optimize(inner_obj, n_trials=inner_trials)
            best_params = study.best_params
            print("Inner best params (LSTM):")
            pprint.pprint(best_params)
            # Build final train/val windows with best seq_len
            seq_len = best_params["seq_len"]
            Xtr_win, ytr_win, _ = ms.make_windows(df_tr, feature_cols, seq_len=seq_len)
            Xva_win, yva_win, _ = ms.make_windows(df_va, feature_cols, seq_len=seq_len)
            if Xtr_win.shape[0] == 0 or Xva_win.shape[0] == 0:
                print("Not enough windows for LSTM in this fold -> skipping")
                y_pred = np.full_like(y_va, fill_value=np.mean(y_tr))
            else:
                batch = 64 if Xtr_win.shape[0] >= 64 else 16
                train_loader = DataLoader(ms.SequenceDataset(Xtr_win, ytr_win), batch_size=batch, shuffle=True)
                val_loader = DataLoader(ms.SequenceDataset(Xva_win, yva_win), batch_size=batch, shuffle=False)
                model = ms.LSTMRegressor(n_features=len(feature_cols), hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"])
                device = "cuda" if ms.torch and ms.torch.cuda.is_available() else "cpu"
                model, best_val = ms.train_lstm(model, train_loader, val_loader, lr=best_params["lr"], epochs=40, device=device, patience=6)
                # predict per-window; map back to df_va by choosing predictions corresponding to last window rows
                y_pred = model(torch.tensor(Xva_win, dtype=ms.torch.float32).to(device)).detach().cpu().numpy()
                # Align y_pred to rows in df_va: currently each window corresponds to last-row RUL in that window.
                # For simplicity return predictions directly for evaluation against y_va_win
                y_va = yva_win
        else:
            raise ValueError("Unsupported model_type. Choose 'lightgbm' or 'lstm'.")

        # ---- Metrics ----
        # If model_type == 'lstm' we may have replaced y_va above.
        try:
            metrics = ms.regression_metrics(y_va, y_pred)
            p100 = ms.precision_at_k_rul(y_va, y_pred, k=100)
            ew7 = ms.early_warning_rate(df.iloc[val_idx].reset_index(drop=True), y_va, y_pred, lead=7)
        except Exception as e:
            print("Error computing metrics:", e)
            metrics = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
            p100 = 0.0
            ew7 = 0.0

        res = {
            "outer_fold": i,
            "metrics": metrics,
            "Precision@100": float(p100),
            "EarlyWarning@7": float(ew7),
            "best_params": best_params,
            "val_rows": int(len(val_idx))
        }
        results.append(res)

    # summary
    df_res = pd.DataFrame([{
        "outer_fold": r["outer_fold"],
        "MAE": r["metrics"]["MAE"],
        "RMSE": r["metrics"]["RMSE"],
        "R2": r["metrics"]["R2"],
        "Precision@100": r["Precision@100"],
        "EarlyWarning@7": r["EarlyWarning@7"]
    } for r in results])

    avg = df_res.mean(numeric_only=True).to_dict()
    print("\nNested CV complete. Per-fold results:")
    print(df_res)
    print("\nAveraged metrics:")
    print(avg)
    return results, avg
