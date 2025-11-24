from agentic_pm.agent.diagnostics import diagnostic_checker
import pandas as pd

def test_diagnostic_rules():
    df = pd.DataFrame({
        "sensor_1": [10, 20, 40, 80],   # rising fast
        "sensor_2": [0.1, 0.2, 10, 40], # anomaly
        "cycle": [1,2,3,4],
    })
    out = diagnostic_checker(df)
    assert "overheat_risk" in out
    assert out["overheat_risk"] == True
