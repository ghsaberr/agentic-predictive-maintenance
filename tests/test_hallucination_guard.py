from agentic_pm.agent.guardrails import validate_citations

def test_hallucination_guard_pass():
    retrieved_docs = {
        "doc_1": "compressor overheating due to blockage",
        "doc_2": "bearing wear guidance"
    }

    llm_text = """
Recommendation: Inspect compressor.
- overheating observed (doc_1)
- vibration trending abnormal (doc_2)
"""

    ok, msg = validate_citations(llm_text, retrieved_docs)
    assert ok == True

def test_hallucination_guard_fail():
    retrieved_docs = {"doc_1": "compressor overheating"}

    llm_text = """
Recommendation: Inspect bearing.
- bearing melted spontaneously (doc_99)
"""

    ok, msg = validate_citations(llm_text, retrieved_docs)
    assert ok == False
    assert "doc_99" in msg
