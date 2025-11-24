import numpy as np
from agentic_pm.agent.retriever import embed_query, retrieve

def test_retriever_returns_results():
    qvec = embed_query("compressor overheat")
    assert qvec.shape[1] > 100   # embedding dim sanity check

    out = retrieve("compressor overheat", top_k=3)
    assert len(out) > 0
    assert "doc_id" in out[0]
    assert "text" in out[0]
