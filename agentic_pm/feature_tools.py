"""
agentic_pm/feature_tools.py

Feature engineering & deterministic tools for CMAPSS (operate on RAW/cleaned data).

Functions:
- create_temporal_features(df, windows=[5,15,60], lags=[1,3,6])
- create_frequency_features(df, sensors=None, n_peaks=3, window=64)
- create_anomaly_indicators(df, window=30, z_thresh=4)
- compute_health_index(df, sensor_weights=None)   # uses per-unit min-max on RAW sensors
- diagnostic_checker(window_df, thresholds=None)  # uses RAW sensors and physical thresholds
- maintenance_simulator(risk_series, effectiveness=0.5)
"""
import numpy as np
import pandas as pd
from scipy import fftpack
from sklearn.linear_model import LinearRegression

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


# ----------------------------
# Temporal features
# ----------------------------
def create_temporal_features(df: pd.DataFrame, windows=[5, 15, 60], lags=[1, 3, 6]) -> pd.DataFrame:
    """
    Compute rolling mean/std, slope (trend), EMA and lag features per unit.
    Operates on RAW (not scaled) sensor columns.
    """
    df = df.copy()
    for w in windows:
        for s in SENSOR_COLS:
            # rolling mean/std
            df[f"{s}_rm_{w}"] = df.groupby("unit")[s].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"{s}_rstd_{w}"] = df.groupby("unit")[s].rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

            # slope: linear regression coefficient over the window
            def slope(arr):
                if len(arr) < 2:
                    return 0.0
                X = np.arange(len(arr)).reshape(-1, 1)
                lr = LinearRegression().fit(X, arr)
                return float(lr.coef_[0])

            df[f"{s}_slope_{w}"] = df.groupby("unit")[s].rolling(window=w, min_periods=2).apply(lambda x: slope(np.array(x)), raw=False).reset_index(level=0, drop=True)

            # EMA (exponential moving average) per unit
            df[f"{s}_ema_{w}"] = df.groupby("unit")[s].transform(lambda x: x.ewm(span=w, adjust=False).mean())

    # lag features
    for lag in lags:
        for s in SENSOR_COLS:
            df[f"{s}_lag_{lag}"] = df.groupby("unit")[s].shift(lag)

    return df


# ----------------------------
# Frequency-domain features (FFT peaks)
# ----------------------------
def create_frequency_features(df: pd.DataFrame, sensors=None, n_peaks=3, window=64) -> pd.DataFrame:
    """
    Compute FFT peak magnitudes on sliding windows and assign to the last cycle of each window.
    Warning: relatively heavy compute; use on subset of sensors or downsample.
    """
    df = df.copy()
    if sensors is None:
        sensors = SENSOR_COLS[:6]  # default subset

    for s in sensors:
        # prepare columns
        for k in range(1, n_peaks + 1):
            df[f"{s}_fft_peak{k}"] = np.nan

        for unit, grp in df.groupby("unit"):
            vals = grp[s].values
            if len(vals) < window:
                continue
            # sliding windows
            for i in range(window, len(vals) + 1):
                win = vals[i - window:i].astype(float)
                win = win - np.mean(win)  # detrend window
                freqs = fftpack.fft(win)
                mags = np.abs(freqs)[: len(freqs)//2]
                peak_idx = np.argsort(mags)[-n_peaks:][::-1]
                peak_mags = mags[peak_idx]
                idx = grp.index[i - 1]
                for k, mag in enumerate(peak_mags, start=1):
                    df.at[idx, f"{s}_fft_peak{k}"] = mag
    return df


# ----------------------------
# Anomaly indicators
# ----------------------------
def create_anomaly_indicators(df: pd.DataFrame, window=30, z_thresh=4) -> pd.DataFrame:
    """
    Rolling z-score per unit and change-point flags. Produces:
    - {sensor}_z, {sensor}_anom_flag, {sensor}_changepoint and a composite anom_score
    """
    df = df.copy()
    for s in SENSOR_COLS:
        rm = df.groupby("unit")[s].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        rs = df.groupby("unit")[s].rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        z = (df[s] - rm) / (rs + 1e-9)
        df[f"{s}_z"] = z
        df[f"{s}_anom_flag"] = (z.abs() > z_thresh).astype(int)

        # change-point based on delta vs rolling std of delta
        delta = df.groupby("unit")[s].diff().fillna(0)
        std_delta = delta.rolling(window, min_periods=1).std().fillna(1)
        df[f"{s}_changepoint"] = (delta.abs() > 3 * std_delta).astype(int)

    # composite anomaly score: fraction of sensors flagged
    df["anom_score"] = df[[f"{s}_anom_flag" for s in SENSOR_COLS]].mean(axis=1)
    return df


# ----------------------------
# Health index (per-unit min-max on RAW)
# ----------------------------
def compute_health_index(df: pd.DataFrame, sensor_weights: dict = None) -> pd.DataFrame:
    """
    Compute a health_index in [0,1] per row based on per-unit min-max normalization
    of raw sensors and a weighted sum. This is intended for agent/tool use (and can
    also be included in model features, but will be later scaled with scaling.py).
    """
    df = df.copy()
    if sensor_weights is None:
        sensor_weights = {s: 1.0 for s in SENSOR_COLS}
    # per-unit min-max normalize each sensor
    norm_cols = []
    for s in SENSOR_COLS:
        col_n = f"{s}_norm_unit"
        norm_cols.append(col_n)
        df[col_n] = df.groupby("unit")[s].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
    weights = np.array([sensor_weights[s] for s in SENSOR_COLS])
    norm_matrix = df[[f"{s}_norm_unit" for s in SENSOR_COLS]].values
    df["health_index"] = (norm_matrix * weights).sum(axis=1) / (weights.sum() + 1e-9)
    return df


# ----------------------------
# Deterministic tools
# ----------------------------
def diagnostic_checker(window_df: pd.DataFrame, thresholds: dict = None) -> dict:
    """
    Rule-based diagnostics that expect RAW sensor values (physical thresholds).
    Returns dict: {'flags': {...}, 'scores': {...}}
    Example thresholds:
       {'sensor_3': 800, 'sensor_4': 0.5, 'sensor_7': 0.2}
    """
    if thresholds is None:
        thresholds = {
            "sensor_3": 800.0,   # example: temperature threshold (physical units)
            "sensor_4": 0.5,     # example: vibration
            "sensor_7": 0.2,     # example: leak delta threshold
        }
    last = window_df.iloc[-1]
    flags = {}
    scores = {}

    flags["over_temp"] = int(last["sensor_3"] > thresholds["sensor_3"])
    scores["temp_margin"] = float((last["sensor_3"] - thresholds["sensor_3"]) / (thresholds["sensor_3"] + 1e-9))

    vib_median = window_df["sensor_4"].median()
    flags["vibration_exceeded"] = int(vib_median > thresholds["sensor_4"])
    scores["vib_median"] = float(vib_median)

    deltas = window_df["sensor_7"].diff().fillna(0)
    leak_events = (deltas < -thresholds["sensor_7"]).sum()
    flags["leak_suspected"] = int(leak_events >= 2)
    scores["leak_events"] = int(leak_events)

    flags["any_flag"] = int(flags["over_temp"] or flags["vibration_exceeded"] or flags["leak_suspected"])

    scores["anom_score_mean"] = float(window_df["anom_score"].mean()) if "anom_score" in window_df.columns else 0.0

    return {"flags": flags, "scores": scores}


def maintenance_simulator(risk_series: pd.Series, effectiveness: float = 0.5) -> pd.Series:
    """
    Simple preventive maintenance simulator that reduces the last-step risk by effectiveness.
    """
    rs = risk_series.copy().astype(float)
    if len(rs) == 0:
        return rs
    rs.iloc[-1] *= (1.0 - effectiveness)
    return rs
