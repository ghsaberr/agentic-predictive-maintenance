import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def synthetic_series():
    """Tiny synthetic sensor series for feature tests."""
    df = pd.DataFrame({
        "unit": [1]*20,
        "cycle": np.arange(1,21),
        "sensor_1": np.linspace(10, 20, 20),
        "sensor_2": np.sin(np.arange(20)),
        "sensor_3": np.random.randn(20),
        "RUL": np.arange(20)[::-1],
    })
    return df

@pytest.fixture
def mini_feature_df():
    """A small DataFrame after feature engineering."""
    df = pd.DataFrame({
        "unit": [1]*10,
        "cycle": np.arange(1,11),
        "sensor_1": np.linspace(5, 10, 10),
        "sensor_1_rm_5": np.linspace(5, 10, 10).rolling(5, min_periods=1).mean(),
        "sensor_1_rstd_5": np.linspace(5, 10, 10).rolling(5, min_periods=1).std(),
        "health_index": np.linspace(0.2, 0.8, 10),
        "anom_score": np.zeros(10)
    })
    return df
