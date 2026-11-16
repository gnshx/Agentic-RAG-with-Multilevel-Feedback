import pytest
from graph.workflow import AgenticRAGWorkflow
from retrieval.vector_store import LocalVectorStore
from feedback.level3_system_memory import FeedbackRegistry, GoldenCache
from core.state import DocumentChunk

@pytest.fixture
def workflow_setup(tmp_path):
    db_path = str(tmp_path / "graph_test.db")
    registry = FeedbackRegistry(db_path=db_path)
    golden_cache = GoldenCache(registry=registry)
    vstore = LocalVectorStore()

    doc: DocumentChunk = {
        "id": "agentic_1",
        "content": "Agentic RAG architectures use multi-level feedback loops: retrieval grading, hallucination auditing, and system memory.",
        "source": "knowledge_base",
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {}
    }
    vstore.add_documents([doc])

    wf = AgenticRAGWorkflow(vector_store=vstore, registry=registry, golden_cache=golden_cache)
    return wf, registry, golden_cache

def test_full_agentic_workflow_success(workflow_setup):
    wf, registry, golden_cache = workflow_setup

    final_state = wf.run(query="How does Agentic RAG use multi-level feedback?")

    assert final_state["generation"] != ""
    assert final_state["is_grounded"] is True
    assert final_state["is_complete"] is True
    assert final_state["system_log_id"] is not None
    assert len(final_state["execution_trace"]) >= 4

    # Verify step names recorded in trace
    node_names = [step["node"] for step in final_state["execution_trace"]]
    assert "check_golden_cache" in node_names
    assert "retrieve" in node_names
    assert "grade_documents" in node_names
    assert "generate" in node_names
    assert "audit_generation" in node_names
    assert "log_system_feedback" in node_names

def test_workflow_golden_cache_short_circuit(workflow_setup):
    wf, registry, golden_cache = workflow_setup

    # Pre-populate golden cache
    golden_cache.store_exemplar(
        query="What is the speed of light?",
        answer="The speed of light in vacuum is approximately 299,792,458 meters per second."
    )

    final_state = wf.run(query="What is the speed of light?")

    assert final_state["cached_answer"] is True
    assert "299,792,458" in final_state["generation"]
    # Fast path: should not need vector retrieval
    node_names = [step["node"] for step in final_state["execution_trace"]]
    assert "check_golden_cache" in node_names
    assert "retrieve" not in node_names

