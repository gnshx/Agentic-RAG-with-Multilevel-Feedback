import pytest
from feedback.level1_retrieval import DocumentGrader, QueryRewriter
from core.state import DocumentChunk
from core.llm_provider import LocalMockLLM

def test_document_grader_relevant():
    grader = DocumentGrader(llm=LocalMockLLM(), threshold=0.60)
    doc: DocumentChunk = {
        "id": "doc_1",
        "content": "Agentic RAG uses multi-level feedback loops to ensure grounding.",
        "source": "manual",
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {}
    }
    passed, score, reason = grader.grade("How do feedback loops work in Agentic RAG?", doc)
    assert passed is True
    assert score >= 0.60
    assert len(reason) > 0

def test_document_grader_irrelevant():
    grader = DocumentGrader(llm=LocalMockLLM(), threshold=0.60)
    doc: DocumentChunk = {
        "id": "doc_2",
        "content": "A banana is an elongated, edible fruit produced by several kinds of large herbaceous flowering plants.",
        "source": "manual",
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {}
    }
    passed, score, reason = grader.grade("Explain neural attention mechanisms", doc)
    assert passed is False
    assert score < 0.60

def test_query_rewriter():
    rewriter = QueryRewriter(llm=LocalMockLLM())
    new_query, reasoning = rewriter.rewrite("RAG feedback", failure_reason="Low similarity")
    assert "rag feedback" in new_query.lower() or "principles" in new_query.lower()
    assert len(reasoning) > 0

