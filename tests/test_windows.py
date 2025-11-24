import numpy as np
from agentic_pm.modeling.model_selection import make_windows

def test_windows_shape(synthetic_series):
    features = ["sensor_1", "sensor_2", "sensor_3"]
    X, y, u = make_windows(synthetic_series, features, seq_len=5)
    assert X.shape[1:] == (5, 3)
    assert len(X) == len(y)
    assert len(u) == len(y)
