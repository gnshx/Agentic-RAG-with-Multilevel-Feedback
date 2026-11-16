import pytest
from feedback.level2_generation import GeneratorAgent, HallucinationAuditor, CompletenessAuditor
from core.state import DocumentChunk
from core.llm_provider import LocalMockLLM

def test_hallucination_auditor_grounded():
    auditor = HallucinationAuditor(llm=LocalMockLLM(), threshold=0.70)
    context: list[DocumentChunk] = [{
        "id": "c1",
        "content": "Level 1 feedback evaluates document relevance prior to synthesis.",
        "source": "test",
        "score": 0.9,
        "dynamic_weight": 0.0,
        "metadata": {}
    }]
    generation = "Level 1 feedback evaluates document relevance before synthesis."
    passed, score, unsupported, critique = auditor.audit(context, generation)
    assert passed is True
    assert score >= 0.70
    assert len(unsupported) == 0

def test_hallucination_auditor_detects_unsupported_claim():
    auditor = HallucinationAuditor(llm=LocalMockLLM(), threshold=0.70)
    context: list[DocumentChunk] = [{
        "id": "c1",
        "content": "The system runs on Linux.",
        "source": "test",
        "score": 0.9,
        "dynamic_weight": 0.0,
        "metadata": {}
    }]
    # Triggering the mock simulator's unsupported claim branch
    generation = "The system runs on Linux and makes unsupported hallucination claims."
    passed, score, unsupported, critique = auditor.audit(context, generation)
    assert passed is False
    assert score < 0.70
    assert len(unsupported) > 0

def test_completeness_auditor():
    auditor = CompletenessAuditor(llm=LocalMockLLM(), threshold=0.70)
    passed, score, missing, critique = auditor.audit(
        query="What are the three feedback levels?",
        generation="The three levels are Level 1 (retrieval), Level 2 (generation), and Level 3 (memory)."
    )
    assert passed is True
    assert score >= 0.70
    assert len(missing) == 0

def test_generator_agent_with_critique():
    generator = GeneratorAgent(llm=LocalMockLLM())
    context: list[DocumentChunk] = [{
        "id": "c1",
        "content": "Context information on Agentic RAG.",
        "source": "manual",
        "score": 0.9,
        "dynamic_weight": 0.0,
        "metadata": {}
    }]
    answer = generator.generate(
        query="Explain Agentic RAG",
        context_docs=context,
        critique="Ensure Level 3 memory details are highlighted."
    )
    assert len(answer) > 20

