import pandas as pd
import numpy as np
from agentic_pm.feature_tools import create_temporal_features, create_anomaly_indicators, compute_health_index, diagnostic_checker, maintenance_simulator

def make_dummy_df():
    units = []
    for u in range(1,4):
        cycles = np.arange(1,101)
        for c in cycles:
            row = {'unit':u, 'cycle':int(c)}
            for i in range(1,22):
                row[f'sensor_{i}'] = 0.01*c + (i*0.001) + np.random.normal(0,0.05)
            units.append(row)
    return pd.DataFrame(units)

def test_temporal_and_anomaly():
    df = make_dummy_df()
    tf = create_temporal_features(df, windows=[5,15], lags=[1,3])
    assert any(col.endswith('_rm_5') for col in tf.columns)
    ad = create_anomaly_indicators(tf, window=10, z_thresh=3)
    assert 'anom_score' in ad.columns

def test_health_and_tools():
    df = make_dummy_df()
    df = create_temporal_features(df, windows=[5], lags=[1])
    df = create_anomaly_indicators(df, window=10, z_thresh=3)
    df = compute_health_index(df)
    assert 'health_index' in df.columns
    w = df[df.unit==1].tail(10).copy()
    out = diagnostic_checker(w)
    assert 'flags' in out and 'scores' in out
    risk = pd.Series([0.8,0.7,0.6,0.4])
    sim = maintenance_simulator(risk, effectiveness=0.5)
    assert sim.iloc[-1] <= risk.iloc[-1]
