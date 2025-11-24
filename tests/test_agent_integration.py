from agentic_pm.agent.agent_core import agent_recommend
import pandas as pd

def dummy_llm(prompt):
    return "Recommendation: Do X.\n- because A (doc_1)\n- because B (doc_2)"

def dummy_retrieve(q, top_k=3):
    return [
        {"doc_id": "doc_1", "text": "compressor overheating"},
        {"doc_id": "doc_2", "text": "bearing wear guidance"}
    ]

def test_agent_integration(monkeypatch):
    # mock components
    monkeypatch.setattr("agentic_pm.agent.agent_core.call_local_llm", lambda p, m: dummy_llm(p))
    monkeypatch.setattr("agentic_pm.agent.agent_core.retrieve", dummy_retrieve)

    # fake mini window
    df = pd.DataFrame({
        "unit": [1]*120,
        "cycle": range(120),
        "sensor_1": range(120),
        "sensor_2": range(120),
    })

    out = agent_recommend(
        asset_id=1,
        recent_df=df,
        model=None,
        meta={"type": "sequence", "features": ["sensor_1","sensor_2"], "seq_len": 10},
        llm_model_path="dummy"
    )

    assert "Recommendation" in out["llm_response"]
    assert len(out["retrieved"]) == 2
