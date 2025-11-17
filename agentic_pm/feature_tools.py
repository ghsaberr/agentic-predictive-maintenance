"""
agentic_pm/feature_tools.py

Feature engineering & deterministic tools for CMAPSS:

- create_temporal_features()
- create_frequency_features()
- create_anomaly_indicators()
- compute_health_index()
- diagnostic_checker()
- maintenance_simulator()
"""

import numpy as np
import pandas as pd
from scipy import fftpack
from sklearn.linear_model import LinearRegression

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


# -------------------------------------------------------------------
# 1) TEMPORAL FEATURES
# -------------------------------------------------------------------
def create_temporal_features(df, windows=[5,15,60], lags=[1,3,6]):
    df = df.copy()

    for w in windows:
        for s in SENSOR_COLS:
            df[f"{s}_rm_{w}"] = (
                df.groupby("unit")[s]
                .rolling(w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"{s}_rstd_{w}"] = (
                df.groupby("unit")[s]
                .rolling(w, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0)
            )

            # trend (slope)
            def slope(x):
                if len(x) < 2:
                    return 0.0
                X = np.arange(len(x)).reshape(-1, 1)
                lr = LinearRegression().fit(X, x)
                return lr.coef_[0]

            df[f"{s}_slope_{w}"] = (
                df.groupby("unit")[s]
                .rolling(w, min_periods=2)
                .apply(lambda x: slope(np.array(x)), raw=False)
                .reset_index(level=0, drop=True)
            )

            # EMA (uses entire unit history, not per-window)
            df[f"{s}_ema_{w}"] = df.groupby("unit")[s].apply(lambda x: x.ewm(span=w).mean())

    # lag features
    for lag in lags:
        for s in SENSOR_COLS:
            df[f"{s}_lag_{lag}"] = df.groupby("unit")[s].shift(lag)

    return df


# -------------------------------------------------------------------
# 2) FREQUENCY FEATURES (FFT peaks)
# -------------------------------------------------------------------
def create_frequency_features(df, sensors=None, n_peaks=3, window=64):
    df = df.copy()
    if sensors is None:
        sensors = SENSOR_COLS[:6]  # subset for speed

    for s in sensors:
        df[f"{s}_fft_peak1"] = np.nan
        df[f"{s}_fft_peak2"] = np.nan
        df[f"{s}_fft_peak3"] = np.nan

        for unit, grp in df.groupby("unit"):
            values = grp[s].values
            if len(values) < window:
                continue

            for i in range(window, len(values) + 1):
                window_vals = values[i-window : i]
                window_vals -= np.mean(window_vals)
                freqs = fftpack.fft(window_vals)
                mags = np.abs(freqs)[: len(freqs)//2]

                peak_idx = np.argsort(mags)[-n_peaks:][::-1]
                peak_mags = mags[peak_idx]

                idx = grp.index[i - 1]
                for k, mag in enumerate(peak_mags):
                    df.at[idx, f"{s}_fft_peak{1+k}"] = mag

    return df


# -------------------------------------------------------------------
# 3) ANOMALY INDICATORS
# -------------------------------------------------------------------
def create_anomaly_indicators(df, window=30, z_thresh=4):
    df = df.copy()

    for s in SENSOR_COLS:
        rm = (
            df.groupby("unit")[s]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        rs = (
            df.groupby("unit")[s]
            .rolling(window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

        z = (df[s] - rm) / (rs + 1e-9)
        df[f"{s}_z"] = z
        df[f"{s}_anom_flag"] = (z.abs() > z_thresh).astype(int)

        # change-points
        delta = df.groupby("unit")[s].diff().fillna(0)
        std_delta = delta.rolling(window, min_periods=1).std().fillna(1)
        df[f"{s}_changepoint"] = (delta.abs() > 3 * std_delta).astype(int)

    # composite anomaly score
    df["anom_score"] = df[[f"{s}_anom_flag" for s in SENSOR_COLS]].mean(axis=1)

    return df


# -------------------------------------------------------------------
# 4) HEALTH INDEX
# -------------------------------------------------------------------
def compute_health_index(df, sensor_weights=None):
    """
    - Health index = weighted sum of standardized sensor values
    """

    df = df.copy()

    if sensor_weights is None:
        sensor_weights = {s: 1.0 for s in SENSOR_COLS}

    weights = np.array([sensor_weights[s] for s in SENSOR_COLS])

    # Use the standardized features directly
    df["health_index"] = (
        df[SENSOR_COLS].mul(weights, axis=1).sum(axis=1) /
        (weights.sum() + 1e-9)
    )

    return df


# -------------------------------------------------------------------
# 5) DIAGNOSTIC CHECKER (DETERMINISTIC TOOL A)
# -------------------------------------------------------------------
def diagnostic_checker(window_df, thresholds=None):
    if thresholds is None:
        thresholds = {
            "sensor_3": 800,   # temp
            "sensor_4": 0.5,   # vibration
            "sensor_7": 0.2,   # leak drop
        }

    last = window_df.iloc[-1]

    flags = {}
    scores = {}

    flags["over_temp"] = int(last["sensor_3"] > thresholds["sensor_3"])
    scores["temp_margin"] = float(
        (last["sensor_3"] - thresholds["sensor_3"]) / (thresholds["sensor_3"] + 1e-9)
    )

    vib_median = window_df["sensor_4"].median()
    flags["vibration_exceeded"] = int(vib_median > thresholds["sensor_4"])
    scores["vib_median"] = float(vib_median)

    deltas = window_df["sensor_7"].diff().fillna(0)
    leak_events = (deltas < -thresholds["sensor_7"]).sum()
    flags["leak_suspected"] = int(leak_events >= 2)
    scores["leak_events"] = int(leak_events)

    flags["any_flag"] = int(
        flags["over_temp"] or flags["vibration_exceeded"] or flags["leak_suspected"]
    )

    scores["anom_score_mean"] = (
        float(window_df["anom_score"].mean()) if "anom_score" in window_df else 0.0
    )

    return {"flags": flags, "scores": scores}


# -------------------------------------------------------------------
# 6) MAINTENANCE SIMULATOR (DETERMINISTIC TOOL B)
# -------------------------------------------------------------------
def maintenance_simulator(risk_series, effectiveness=0.5):
    """
    Reduce last-step risk by given effectiveness (0.5 = 50% reduction).
    Enables "what-if" analysis for the agent.
    """
    rs = risk_series.copy().astype(float)
    if len(rs) == 0:
        return rs

    rs.iloc[-1] *= (1.0 - effectiveness)
    return rs
