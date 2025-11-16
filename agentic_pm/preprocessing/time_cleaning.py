"""
Time Alignment & Cleaning Module for CMAPSS
Includes:
- align_cycles(df)
- detect_gaps(df)
- impute_missing(df)
- cap_outliers(df) using rolling Z-score
- plot_before_after(df_raw, df_clean, unit, sensor)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

# -----------------------------
# 1. ALIGN CYCLES
# -----------------------------
def align_cycles(df):
    out = []
    for unit, grp in df.groupby("unit"):
        full_cycles = pd.DataFrame({"cycle": range(grp.cycle.min(), grp.cycle.max()+1)})
        full_cycles["unit"] = unit
        merged = full_cycles.merge(grp, on=["unit","cycle"], how="left")
        out.append(merged)
    return pd.concat(out).reset_index(drop=True)

# -----------------------------
# 2. DETECT GAPS
# -----------------------------
def detect_gaps(df):
    df = df.copy()
    df["gap_flag"] = 0
    for unit, grp in df.groupby("unit"):
        cycles = grp.cycle
        missing = cycles.isna()
        df.loc[missing.index, "gap_flag"] = 1
    return df

# -----------------------------
# 3. IMPUTE MISSING
# -----------------------------
def impute_missing(df):
    df = df.copy()
    for col in SENSOR_COLS:
        df[col] = df.groupby("unit")[col].apply(lambda x: x.interpolate(limit=2).ffill())
    return df

# -----------------------------
# 4. OUTLIER CAPPING (Rolling Z-score)
# -----------------------------
def cap_outliers(df, window=30, z_thresh=4):
    df = df.copy()
    for sensor in SENSOR_COLS:
        rolling_mean = df.groupby("unit")[sensor].rolling(window).mean().reset_index(level=0,drop=True)
        rolling_std  = df.groupby("unit")[sensor].rolling(window).std().reset_index(level=0,drop=True)
        z = (df[sensor] - rolling_mean) / (rolling_std + 1e-6)
        upper = rolling_mean + z_thresh * rolling_std
        lower = rolling_mean - z_thresh * rolling_std
        df[sensor] = np.where(df[sensor] > upper, upper, df[sensor])
        df[sensor] = np.where(df[sensor] < lower, lower, df[sensor])
    return df

# -----------------------------
# 5. BEFORE/AFTER PLOTS
# -----------------------------
def plot_before_after(df_raw, df_clean, unit, sensor):
    r = df_raw[df_raw.unit == unit]
    c = df_clean[df_clean.unit == unit]
    plt.figure(figsize=(12,4))
    plt.plot(r.cycle, r[sensor], label="raw", alpha=0.6)
    plt.plot(c.cycle, c[sensor], label="cleaned", alpha=0.8)
    plt.title(f"Unit {unit} - {sensor} before/after cleaning")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
