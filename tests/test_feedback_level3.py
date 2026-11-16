import os
import pytest
from feedback.level3_system_memory import FeedbackRegistry, GoldenCache, SOPArchitect
from retrieval.vector_store import LocalVectorStore
from core.state import DocumentChunk, AgenticRAGState
from core.llm_provider import LocalMockLLM

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_feedback.db"
    return str(db_file)

def test_feedback_registry_log_and_retrieve(temp_db):
    registry = FeedbackRegistry(db_path=temp_db)
    state: AgenticRAGState = {
        "query": "Test question?",
        "current_query": "Test question?",
        "session_id": "test_sess",
        "retrieved_docs": [],
        "graded_docs": [{"id": "d1", "content": "Sample", "source": "t", "score": 0.8, "dynamic_weight": 0.0, "metadata": {}}],
        "retrieval_retries": 1,
        "retrieval_fallback": False,
        "generation": "Test answer",
        "generation_retries": 0,
        "critique": "",
        "groundedness_score": 0.95,
        "completeness_score": 0.90,
        "is_grounded": True,
        "is_complete": True,
        "cached_answer": False,
        "golden_cache_hit_id": None,
        "system_log_id": None,
        "execution_trace": []
    }
    log_id = registry.log_inference(state)
    assert log_id is not None

    record = registry.get_inference(log_id)
    assert record is not None
    assert record["query"] == "Test question?"
    assert record["generation"] == "Test answer"
    assert "d1" in record["doc_ids"]

    # Test Human feedback logging
    fid = registry.log_human_feedback(log_id, rating=1, comment="Excellent answer")
    assert fid is not None

def test_golden_cache_and_lookup(temp_db):
    registry = FeedbackRegistry(db_path=temp_db)
    cache = GoldenCache(registry=registry)

    # Store exemplar
    gid = cache.store_exemplar(
        query="What is Level 1 feedback?",
        answer="Level 1 feedback performs document grading and query rewriting."
    )
    assert gid is not None

    # Lookup identical query
    match = cache.lookup("What is Level 1 feedback?", threshold=0.85)
    assert match is not None
    matched_id, answer, sim = match
    assert matched_id == gid
    assert "Level 1" in answer
    assert sim >= 0.85

def test_vector_store_dynamic_weighting():
    vstore = LocalVectorStore()
    doc: DocumentChunk = {
        "id": "doc_w",
        "content": "Deep learning architectures.",
        "source": "manual",
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {}
    }
    vstore.add_documents([doc])

    # Initial search
    res1 = vstore.search("Deep learning", top_k=1)
    base_score = res1[0]["score"]

    # Apply positive Level 3 feedback
    vstore.update_weight("doc_w", delta=0.20)
    assert vstore.documents["doc_w"]["dynamic_weight"] == 0.20

    res2 = vstore.search("Deep learning", top_k=1)
    boosted_score = res2[0]["score"]
    assert boosted_score >= base_score

def test_sop_architect_analysis(temp_db):
    registry = FeedbackRegistry(db_path=temp_db)
    # Log a dummy inference to test telemetry
    registry.log_inference({
        "query": "Q", "current_query": "Q", "session_id": "s",
        "retrieved_docs": [], "graded_docs": [], "retrieval_retries": 1,
        "generation_retries": 0, "generation": "A", "groundedness_score": 0.9,
        "completeness_score": 0.9, "is_grounded": True, "is_complete": True,
        "execution_trace": []
    })
    sop = SOPArchitect(registry, llm=LocalMockLLM())
    analysis = sop.analyze_system_performance()
    assert "telemetry" in analysis
    assert analysis["telemetry"]["total_inferences"] == 1
    assert "diagnosis" in analysis

