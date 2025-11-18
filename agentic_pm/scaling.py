"""
agentic_pm/scaling.py

Provides:
- global_standardize(train_df, test_df, feature_cols, subset_name)
- conditional_standardize(train_df, test_df, feature_cols, op_cols, n_clusters, subset_name)

Intended usage:
1) Run feature engineering to produce feature columns (on RAW).
2) Choose feature_cols (which columns to feed to model).
3) Call one of these functions to obtain scaled train/test and saved scalers.

Saves scalers in artifacts/scalers/{subset_name}_*.pkl
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

SCALER_DIR = Path("artifacts/scalers")
SCALER_DIR.mkdir(parents=True, exist_ok=True)


def global_standardize(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list, subset_name: str = "FD"):
    """
    Fit a StandardScaler on train_df[feature_cols] and transform both train and test.
    Returns (train_scaled_df, test_scaled_df, scaler)
    """
    scaler = StandardScaler()
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    scaler.fit(X_train)
    train_t = train_df.copy()
    test_t = test_df.copy()
    train_t[feature_cols] = scaler.transform(X_train)
    test_t[feature_cols] = scaler.transform(X_test)
    scaler_path = SCALER_DIR / f"{subset_name}_global_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[scaling] Saved global scaler: {scaler_path}")
    return train_t, test_t, scaler


def conditional_standardize(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list, op_cols: list, n_clusters: int = 6, subset_name: str = "FD"):
    """
    Cluster units by operating settings (per-unit mean of op_cols), then fit standard scaler per cluster
    using training rows in that cluster. Transform both train and test accordingly.

    Returns: (train_scaled_df, test_scaled_df, dict_of_scalers, unit_cluster_map)
    """
    # compute per-unit op means (from train_df)
    unit_op_means = train_df.groupby("unit")[op_cols].mean().reset_index()
    X = unit_op_means[op_cols].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    unit_op_means["cluster"] = labels
    unit_cluster = dict(zip(unit_op_means["unit"], unit_op_means["cluster"]))

    train_t = train_df.copy()
    test_t = test_df.copy()

    train_t["cluster"] = train_t["unit"].map(unit_cluster)
    test_t["cluster"] = test_t["unit"].map(unit_cluster)

    scalers = {}
    for c in sorted(train_t["cluster"].dropna().unique()):
        mask_tr = train_t["cluster"] == c
        mask_te = test_t["cluster"] == c
        Xc_tr = train_t.loc[mask_tr, feature_cols]
        # fallback: if cluster has few/no rows in train, fit on full train
        if Xc_tr.shape[0] < 2:
            scaler = StandardScaler()
            scaler.fit(train_df[feature_cols].values)
        else:
            scaler = StandardScaler()
            scaler.fit(Xc_tr.values)
        if mask_tr.sum() > 0:
            train_t.loc[mask_tr, feature_cols] = scaler.transform(train_t.loc[mask_tr, feature_cols].values)
        if mask_te.sum() > 0:
            test_t.loc[mask_te, feature_cols] = scaler.transform(test_t.loc[mask_te, feature_cols].values)
        scalers[c] = scaler
        joblib.dump(scaler, SCALER_DIR / f"{subset_name}_cluster_{c}_scaler.pkl")

    joblib.dump(unit_cluster, SCALER_DIR / f"{subset_name}_unit_cluster_map.pkl")
    print(f"[scaling] Saved {len(scalers)} cluster scalers and unit->cluster map for {subset_name}")
    return train_t, test_t, scalers, unit_cluster
