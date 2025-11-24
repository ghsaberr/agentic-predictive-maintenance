import numpy as np
from agentic_pm import feature_tools

def test_rolling_mean(mini_feature_df):
    x = np.array([1,2,3,4,5])
    rm = feature_tools._rolling_mean(x, 3)
    assert rm.tolist() == [1, 1.5, 2, 3, 4]

def test_create_temporal_features(synthetic_series):
    df = feature_tools.create_temporal_features(synthetic_series.copy())
    assert any(col.startswith("sensor_1_rm") for col in df.columns)
    assert any(col.startswith("sensor_1_rstd") for col in df.columns)
