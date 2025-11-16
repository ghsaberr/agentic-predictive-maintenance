"""
agentic_pm/data_ingest.py

Purpose:
- Load CMAPSS FD001-FD004 (expects files already downloaded into data/raw/CMAPSS)
- Parse files into pandas DataFrames with meaningful column names
- Compute RUL for train and test sets (per-row RUL for test by merging RUL_FD00X.txt)
- Provide EDA helpers (summary stats, sensor/time plots)
- Implement normalization strategies:
    - global_standardize (global StandardScaler)
    - conditional_standardize (cluster by operating-settings and standardize per-cluster)
- Save processed CSVs and fitted scalers for later pipeline stages

Usage (example):
>>> python -m agentic_pm.data_ingest

Note: This module does NOT download CMAPSS automatically. Place files like
  data/raw/CMAPSS/train_FD001.txt
  data/raw/CMAPSS/test_FD001.txt
  data/raw/CMAPSS/RUL_FD001.txt

Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Config
RAW_BASE = Path("data/raw/CMAPSS")
PROCESSED_BASE = Path("data/processed/CMAPSS")
SCALER_DIR = Path("artifacts/scalers")
PROCESSED_BASE.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)

# Column names for CMAPSS files (26 cols)
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COL_NAMES = ["unit", "cycle"] + OP_COLS + SENSOR_COLS

# Mapping for normalization strategy per subset
# FD001, FD003 -> global; FD002, FD004 -> conditional
NORMALIZATION_MAP = {
    "FD001": "global",
    "FD002": "conditional",
    "FD003": "global",
    "FD004": "conditional",
}


# ------------------------
# Core loading / parsing
# ------------------------

def read_cmapps_file(filepath: Path) -> pd.DataFrame:
    """Read a CMAPSS train or test txt file into DataFrame with proper column names."""
    df = pd.read_csv(filepath, sep="\s+", header=None, names=COL_NAMES)
    # ensure dtypes
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def read_rul_file(filepath: Path) -> pd.Series:
    """Read RUL file (single column) and return a Series indexed by unit (1-based)."""
    rul = pd.read_csv(filepath, header=None).squeeze("columns")
    rul.index = np.arange(1, len(rul) + 1)  # unit ids are 1..N
    rul.name = "RUL"
    return rul


def compute_train_rul(df_train: pd.DataFrame) -> pd.DataFrame:
    """Add RUL column to train DataFrame: RUL = max_cycle_for_unit - cycle"""
    max_cycle = df_train.groupby("unit")["cycle"].transform("max")
    df = df_train.copy()
    df["RUL"] = max_cycle - df["cycle"]
    return df


def compute_test_rul(df_test: pd.DataFrame, rul_series: pd.Series) -> pd.DataFrame:
    """Compute per-row RUL for test set by merging per-unit RUL (remaining after last observed cycle).

    For each unit:
        final_cycle = max(cycle observed in test for that unit)
        RUL_at_final = rul_series.loc[unit]
    For a row with cycle c:
        row_RUL = RUL_at_final + (final_cycle - c)

    Returns a new DataFrame with RUL column.
    """
    df = df_test.copy()
    final_cycles = df.groupby("unit")["cycle"].transform("max")
    # Map per-unit remaining life at last observed cycle
    df = df.merge(rul_series.rename("RUL_unit"), left_on="unit", right_index=True)
    df["RUL"] = df["RUL_unit"] + (final_cycles - df["cycle"])
    df.drop(columns=["RUL_unit"], inplace=True)
    return df


# ------------------------
# Normalization methods
# ------------------------

def global_standardize(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list = None, subset_name: str = "FD"):
    """Fit StandardScaler on train_df[cols] (global) and transform both train and test.
    Saves scaler to artifacts/scalers/{subset_name}_global_scaler.pkl
    Returns transformed (train, test) and scaler.
    """
    if cols is None:
        cols = SENSOR_COLS + OP_COLS
    scaler = StandardScaler()
    X_train = train_df[cols].values
    X_test = test_df[cols].values
    scaler.fit(X_train)
    train_t = train_df.copy()
    test_t = test_df.copy()
    train_t[cols] = scaler.transform(X_train)
    test_t[cols] = scaler.transform(X_test)
    scaler_path = SCALER_DIR / f"{subset_name}_global_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved global scaler to {scaler_path}")
    return train_t, test_t, scaler


def conditional_standardize(train_df: pd.DataFrame, test_df: pd.DataFrame, n_clusters: int = 6, subset_name: str = "FD"):
    """Cluster units by their operating settings (mean across cycles) and standardize sensor+op features per-cluster.

    Strategy:
    - Compute per-unit mean of OP_COLS across cycles
    - Fit KMeans on these per-unit means (n_clusters default 6)
    - Assign cluster labels to each row (by unit)
    - For each cluster, fit StandardScaler on train rows in that cluster and transform both train/test rows in that cluster

    Returns (train_t, test_t, dict_of_scalers, unit_cluster_map)
    """
    # compute unit-level op means
    unit_means = train_df.groupby("unit")[OP_COLS].mean().reset_index()
    X = unit_means[OP_COLS].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    unit_means["cluster"] = labels

    # map unit -> cluster
    unit_cluster = dict(zip(unit_means["unit"], unit_means["cluster"]))

    # create copies
    train_t = train_df.copy()
    test_t = test_df.copy()

    scalers = {}

    # assign cluster column
    train_t["cluster"] = train_t["unit"].map(unit_cluster)
    test_t["cluster"] = test_t["unit"].map(unit_cluster)

    cols = SENSOR_COLS + OP_COLS

    for c in sorted(train_t["cluster"].unique()):
        mask_train = train_t["cluster"] == c
        mask_test = test_t["cluster"] == c
        Xc_train = train_t.loc[mask_train, cols]
        # If cluster has no train rows (very unlikely), skip
        if Xc_train.shape[0] < 2:
            # fallback to global scaler behavior for this cluster
            scaler = StandardScaler()
            scaler.fit(train_df[cols].values)
        else:
            scaler = StandardScaler()
            scaler.fit(Xc_train.values)
        # transform
        if mask_train.sum() > 0:
            train_t.loc[mask_train, cols] = scaler.transform(Xc_train.values)
        if mask_test.sum() > 0:
            Xc_test = test_t.loc[mask_test, cols]
            test_t.loc[mask_test, cols] = scaler.transform(Xc_test.values)
        scalers[c] = scaler
        joblib.dump(scaler, SCALER_DIR / f"{subset_name}_cluster_{c}_scaler.pkl")

    # also save unit->cluster mapping
    joblib.dump(unit_cluster, SCALER_DIR / f"{subset_name}_unit_cluster_map.pkl")
    print(f"Saved {len(scalers)} cluster scalers and unit->cluster map for {subset_name}")
    return train_t, test_t, scalers, unit_cluster


# ------------------------
# EDA / Visualization
# ------------------------

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for sensor columns (mean, std, min, max)"""
    stats = df[SENSOR_COLS].agg(["mean", "std", "min", "max"]).T
    stats.columns = ["mean", "std", "min", "max"]
    return stats


def plot_unit_sensors(df: pd.DataFrame, unit_id: int, sensors: list = None, nrows: int = None):
    """Plot time-series for a given unit for selected sensors."""
    if sensors is None:
        sensors = SENSOR_COLS[:6]
    unit_df = df[df["unit"] == unit_id]
    n = len(sensors)
    if nrows is None:
        nrows = int(np.ceil(n / 2))
    plt.figure(figsize=(12, 3 * nrows))
    for i, s in enumerate(sensors, 1):
        plt.subplot(nrows, 2, i)
        plt.plot(unit_df["cycle"], unit_df[s])
        plt.title(f"Unit {unit_id} - {s}")
        plt.xlabel("cycle")
        plt.ylabel(s)
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sensor_over_units(df: pd.DataFrame, sensor: str, sample_units: list = None, ncols: int = 2):
    """Overlay sensor traces from several units to see variability."""
    if sample_units is None:
        sample_units = df["unit"].unique()[:6]
    n = len(sample_units)
    plt.figure(figsize=(12, 3 * int(np.ceil(n / ncols))))
    for i, u in enumerate(sample_units, 1):
        plt.subplot(int(np.ceil(n / ncols)), ncols, i)
        tmp = df[df["unit"] == u]
        plt.plot(tmp["cycle"], tmp[sensor])
        plt.title(f"Unit {u}")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, sensors: list = None):
    if sensors is None:
        sensors = SENSOR_COLS[:12]
    corr = df[sensors].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="vlag")
    plt.title("Sensor correlation matrix")
    plt.show()


# ------------------------
# Top-level processing per subset
# ------------------------

def process_subset(subset: str, raw_base: Path = RAW_BASE, processed_base: Path = PROCESSED_BASE, n_clusters: int = 6):
    """Load, compute RUL, normalize according to NORMALIZATION_MAP, run basic EDA plots, and save CSVs/scalers.

    subset: e.g., "FD001"
    Expects files: train_FD001.txt, test_FD001.txt, RUL_FD001.txt
    """
    subset = subset.upper()
    print(f"Processing {subset} ...")
    train_f = raw_base / f"train_{subset}.txt"
    test_f = raw_base / f"test_{subset}.txt"
    rul_f = raw_base / f"RUL_{subset}.txt"

    if not (train_f.exists() and test_f.exists() and rul_f.exists()):
        raise FileNotFoundError(f"Missing files for {subset} in {raw_base}. Expected {train_f}, {test_f}, {rul_f}.")

    train_df = read_cmapps_file(train_f)
    test_df = read_cmapps_file(test_f)
    rul_series = read_rul_file(rul_f)

    train_df = compute_train_rul(train_df)
    test_df = compute_test_rul(test_df, rul_series)

    # Basic EDA: show shape and sensor stats
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    stats = summary_stats(train_df)
    print("Sensor summary stats (train):")
    print(stats.head())

    # Choose normalization
    norm = NORMALIZATION_MAP.get(subset, "global")
    if norm == "global":
        train_t, test_t, scaler = global_standardize(train_df, test_df, cols=None, subset_name=subset)
    else:
        train_t, test_t, scalers, unit_cluster_map = conditional_standardize(train_df, test_df, n_clusters=n_clusters, subset_name=subset)

    # Save processed
    processed_dir = processed_base / subset
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_out = processed_dir / f"train_{subset}.csv"
    test_out = processed_dir / f"test_{subset}.csv"
    train_t.to_csv(train_out, index=False)
    test_t.to_csv(test_out, index=False)
    print(f"Saved processed files: {train_out}, {test_out}")

    # Quick plots (show first unit)
    first_unit = train_df["unit"].unique()[0]
    try:
        plot_unit_sensors(train_df, first_unit, sensors=SENSOR_COLS[:6])
        plot_correlation_heatmap(train_df)
    except Exception as e:
        print("Plotting failed:", e)

    return {
        "train": train_t,
        "test": test_t,
        "stats": stats,
        "normalization": norm,
    }


# ------------------------
# Helper: run pipeline for all FD001-FD004
# ------------------------

def run_all(subsets=None):
    if subsets is None:
        subsets = ["FD001", "FD002", "FD003", "FD004"]
    results = {}
    for s in subsets:
        results[s] = process_subset(s)
    return results


# ------------------------
# CLI entrypoint
# ------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CMAPSS ingestion & preprocess pipeline (FD001-FD004)")
    parser.add_argument("--raw_dir", type=str, default=str(RAW_BASE), help="path to raw CMAPSS files")
    parser.add_argument("--out_dir", type=str, default=str(PROCESSED_BASE), help="path to save processed csvs")
    parser.add_argument("--subsets", type=str, default="FD001,FD002,FD003,FD004", help="comma-separated subsets")
    parser.add_argument("--clusters", type=int, default=6, help="n clusters for conditional normalization")
    args = parser.parse_args()

    RAW_BASE = Path(args.raw_dir)
    PROCESSED_BASE = Path(args.out_dir)
    PROCESSED_BASE.mkdir(parents=True, exist_ok=True)
    SCALER_DIR.mkdir(parents=True, exist_ok=True)

    subsets = [s.strip().upper() for s in args.subsets.split(",")]
    run_all(subsets)
