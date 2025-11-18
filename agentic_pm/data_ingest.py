"""
agentic_pm/data_ingest.py  (REDESIGNED)

Purpose:
- Load CMAPSS FD001-FD004 (expects files already downloaded into data/raw/CMAPSS)
- Parse files into pandas DataFrames with meaningful column names
- Compute RUL for train and test sets (per-row RUL for test by merging RUL_FD00X.txt)
- Save "raw" processed CSVs (intermediate) for subsequent cleaning/feature-engineering
- Does NOT perform scaling. Scaling is handled in scaling.py after feature engineering.

Usage:
>>> from agentic_pm.data_ingest import process_subset, run_all
>>> run_all()
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Config paths
RAW_BASE = Path("data/raw/CMAPSS")
INTERMEDIATE_BASE = Path("data/intermediate/CMAPSS")
INTERMEDIATE_BASE.mkdir(parents=True, exist_ok=True)

# Columns: 2 meta + 3 op settings + 21 sensors = 26 total
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COL_NAMES = ["unit", "cycle"] + OP_COLS + SENSOR_COLS


# ------------------------
# Core loading / parsing
# ------------------------
def read_cmapps_file(filepath: Path) -> pd.DataFrame:
    """Read a CMAPSS train or test txt file into DataFrame with proper column names."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COL_NAMES)
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def read_rul_file(filepath: Path) -> pd.Series:
    """Read RUL file (single column) and return a Series indexed by unit (1-based)."""
    rul = pd.read_csv(filepath, header=None).squeeze("columns")
    rul.index = np.arange(1, len(rul) + 1)
    rul.name = "RUL"
    return rul


def compute_train_rul(df_train: pd.DataFrame) -> pd.DataFrame:
    """Add RUL column to train DataFrame: RUL = max_cycle_for_unit - cycle"""
    df = df_train.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df


def compute_test_rul(df_test: pd.DataFrame, rul_series: pd.Series) -> pd.DataFrame:
    """
    Compute per-row RUL for test set by merging per-unit RUL (remaining after last observed cycle).
    RUL(row) = RUL_unit_at_final + (final_cycle - row_cycle)
    """
    df = df_test.copy()
    final_cycles = df.groupby("unit")["cycle"].transform("max")
    df = df.merge(rul_series.rename("RUL_unit"), left_on="unit", right_index=True)
    df["RUL"] = df["RUL_unit"] + (final_cycles - df["cycle"])
    df.drop(columns=["RUL_unit"], inplace=True)
    return df


# ------------------------
# Top-level processing per subset
# ------------------------
def process_subset(subset: str, raw_base: Path = RAW_BASE, out_base: Path = INTERMEDIATE_BASE):
    """
    Load files, compute RUL for train and test, save intermediate CSVs.
    Does NOT perform cleaning/feature-engineering/scaling.
    """
    subset = subset.upper()
    print(f"[data_ingest] Processing {subset} ...")
    train_f = raw_base / f"train_{subset}.txt"
    test_f = raw_base / f"test_{subset}.txt"
    rul_f = raw_base / f"RUL_{subset}.txt"

    if not (train_f.exists() and test_f.exists() and rul_f.exists()):
        raise FileNotFoundError(f"Missing files for {subset} in {raw_base}.")

    train_df = read_cmapps_file(train_f)
    test_df = read_cmapps_file(test_f)
    rul_series = read_rul_file(rul_f)

    train_df = compute_train_rul(train_df)
    test_df = compute_test_rul(test_df, rul_series)

    # Save intermediate (raw) CSVs for downstream cleaning & FE
    out_dir = out_base / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / f"train_{subset}_raw.csv"
    test_out = out_dir / f"test_{subset}_raw.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"[data_ingest] Saved: {train_out}, {test_out}")

    return {"train_raw": train_df, "test_raw": test_df}


def run_all(subsets=None):
    if subsets is None:
        subsets = ["FD001", "FD002", "FD003", "FD004"]
    out = {}
    for s in subsets:
        out[s] = process_subset(s)
    return out


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", type=str, default="FD001,FD002,FD003,FD004")
    args = parser.parse_args()
    subset_list = [s.strip().upper() for s in args.subsets.split(",")]
    run_all(subset_list)
