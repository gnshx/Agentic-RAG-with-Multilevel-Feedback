import pytest
import os
import json
from src.llm import LLM
from src.signals import consistency, retrieval_grade, groundedness, completeness, _json
from src.retriever import HybridRetriever
from src.planner import QueryPlanner
from src.generator import Generator
from src.levels import run_level_0, run_level_1, run_level_2, run_level_3
from src.controller import run_controller, passes
from eval.metrics import em, f1

# ==============================================================================
# SECTION 1: UNIT TESTS (U1 - U12 from TEST_VERIFICATION.md)
# ==============================================================================

def test_u1_em_normalization():
    assert em("The Eiffel Tower", "eiffel tower") == 1.0
    assert em("  Paris, France!  ", "paris france") == 1.0

def test_u2_f1_partial_overlap():
    # "eiffel tower paris" (3 tokens) vs "eiffel tower" (2 tokens) -> 2 common tokens
    # P = 2/3, R = 2/2 = 1.0 -> F1 = 2 * (2/3) * 1 / (2/3 + 1) = (4/3) / (5/3) = 0.8
    score = f1("The Eiffel Tower Paris", "Eiffel Tower")
    assert round(score, 4) == 0.8000

def test_u3_llm_cache(tmp_path):
    cache_file = str(tmp_path / "test_cache.json")
    llm = LLM(backend=lambda p, t, s=0: "response", cache_path=cache_file)
    r1 = llm.call("test prompt", temperature=0.0)
    assert llm.calls == 1
    r2 = llm.call("test prompt", temperature=0.0)
    assert r1 == r2
    # Verify call counter didn't increment on cache hit
    assert llm.calls == 1

def test_u4_llm_retry_and_failure():
    def failing_backend(p, t, s=0):
        raise ConnectionError("Simulated network timeout")

    llm = LLM(backend=failing_backend, cache_path=":memory:", max_retries=2)
    with pytest.raises(RuntimeError) as exc_info:
        llm.call("prompt")
    assert "LLM call failed after 2 retries" in str(exc_info.value)

def test_u5_json_parse_fallback():
    # Clean JSON
    assert _json('{"relevant": [true, false]}', {}) == {"relevant": [True, False]}
    # Extra surrounding prose
    noisy = "Here is your output:\n```json\n{\"relevant\": [true]}\n```\nHope this helps!"
    assert _json(noisy, {}) == {"relevant": [True]}
    # Malformed JSON defaults safely
    malformed = "Not a json string at all"
    assert _json(malformed, {"relevant": []}) == {"relevant": []}

def test_u6_controller_level_0_pass():
    cfg = {
        "budget": {"max_calls_per_query": 20},
        "levels": {
            "L0": {"samples": 3, "tau": 0.60},
            "L1": {"tau_retrieval": 0.5, "tau_ground": 0.8},
            "L2": {"tau_ground": 0.8, "tau_complete": 0.8},
            "L3": {"drafts": 3, "max_revisions": 2}
        }
    }
    llm = LLM(backend=lambda p, t, s=0: "Paris", cache_path=":memory:")
    levels = {
        0: lambda s: dict(s, answer="Paris", signals={"S1": 1.0}),
        1: lambda s: s
    }
    state = run_controller("What is capital of France?", levels, cfg, llm)
    assert state["level"] == 0
    assert state["answer"] == "Paris"
    assert len(state["trace"]) == 1

def test_u7_controller_escalates_to_l2():
    cfg = {
        "budget": {"max_calls_per_query": 20},
        "levels": {
            "L0": {"tau": 0.90},
            "L1": {"tau_retrieval": 0.80, "tau_ground": 0.80},
            "L2": {"tau_ground": 0.80, "tau_complete": 0.80},
            "L3": {"drafts": 3, "max_revisions": 2}
        }
    }
    llm = LLM(backend=lambda p, t, s=0: "ans", cache_path=":memory:")
    levels = {
        0: lambda s: dict(s, signals={"S1": 0.33}),  # Fails L0
        1: lambda s: dict(s, signals={"S2": 0.40, "S3": 0.50}),  # Fails L1
        2: lambda s: dict(s, signals={"S3": 0.90, "S4": 0.90}, answer="Final L2 Ans"),  # Passes L2
        3: lambda s: s
    }
    state = run_controller("Complex query", levels, cfg, llm)
    assert state["level"] == 2
    assert state["answer"] == "Final L2 Ans"
    assert len(state["trace"]) == 3

def test_u8_controller_stops_at_l3():
    cfg = {
        "budget": {"max_calls_per_query": 20},
        "levels": {
            "L0": {"tau": 0.99},
            "L1": {"tau_retrieval": 0.99, "tau_ground": 0.99},
            "L2": {"tau_ground": 0.99, "tau_complete": 0.99},
            "L3": {"drafts": 3, "max_revisions": 2}
        }
    }
    llm = LLM(backend=lambda p, t, s=0: "ans", cache_path=":memory:")
    levels = {
        0: lambda s: dict(s, signals={"S1": 0.1}),
        1: lambda s: dict(s, signals={"S2": 0.1, "S3": 0.1}),
        2: lambda s: dict(s, signals={"S3": 0.1, "S4": 0.1}),
        3: lambda s: dict(s, signals={"S3": 0.1, "S4": 0.1}, answer="L3 Ans")
    }
    state = run_controller("Hardest query", levels, cfg, llm)
    assert state["level"] == 3
    assert state["answer"] == "L3 Ans"

def test_u9_budget_guard():
    cfg = {
        "budget": {"max_calls_per_query": 5},
        "levels": {
            "L0": {"tau": 0.99},
            "L1": {"tau_retrieval": 0.99, "tau_ground": 0.99},
            "L2": {"tau_ground": 0.99, "tau_complete": 0.99},
            "L3": {"drafts": 3, "max_revisions": 2}
        }
    }
    llm = LLM(backend=lambda p, t, s=0: "ans", cache_path=":memory:")
    def expensive_level_0(s):
        # Consume 6 calls
        for i in range(6):
            llm.call("prompt", sample_id=i)
        return dict(s, signals={"S1": 0.1})

    levels = {0: expensive_level_0, 1: lambda s: s}
    state = run_controller("Query", levels, cfg, llm)
    assert state["status"] == "budget_exhausted"

def test_u10_retriever_search_returns_k():
    corpus = [{"id": f"d_{i}", "title": f"T{i}", "text": f"Content {i} on machine learning"} for i in range(10)]
    retriever = HybridRetriever(corpus=corpus)
    res = retriever.search("machine learning", k=5)
    assert len(res) == 5
    # Verify sorted descending by fused score
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)

def test_u11_rrf_fusion_rankings():
    corpus = [
        {"id": "docA", "title": "A", "text": "cat feline"},
        {"id": "docB", "title": "B", "text": "dog canine"},
        {"id": "docC", "title": "C", "text": "cat dog animal"}
    ]
    retriever = HybridRetriever(corpus=corpus)
    res = retriever.search("cat animal", k=3, rrf_k=60)
    assert len(res) == 3
    # Check that highest ranking document has non-zero score
    assert res[0]["score"] > 0.0

def test_u12_signal_ranges():
    llm = LLM(cache_path=":memory:")
    docs = [{"id": "d1", "text": "Paris is the capital of France."}]
    s1 = consistency(["Paris", "Paris", "Paris"])
    assert 0.0 <= s1 <= 1.0
    s2 = retrieval_grade(llm, "What is capital of France?", docs)
    assert 0.0 <= s2 <= 1.0
    s3 = groundedness(llm, "Paris is capital of France [1].", docs)
    assert 0.0 <= s3 <= 1.0
    s4 = completeness(llm, "What is capital of France?", "Paris [1].")
    assert 0.0 <= s4 <= 1.0

# ==============================================================================
# SECTION 2: INTEGRATION TESTS (I1 - I6 from TEST_VERIFICATION.md)
# ==============================================================================

def test_i1_retrieval_sanity_recall():
    corpus = [
        {"id": "gold_doc", "title": "Paris", "text": "The river Seine flows through Paris France."},
        {"id": "distractor1", "title": "Danube", "text": "The Danube flows through Vienna and Budapest."},
        {"id": "distractor2", "title": "Thames", "text": "The Thames flows through London England."}
    ]
    retriever = HybridRetriever(corpus=corpus)
    res = retriever.search("river Seine Paris", k=2)
    ids = [d["id"] for d in res]
    assert "gold_doc" in ids

def test_i2_level1_end_to_end():
    cfg = {"retrieval": {"k": 2}, "levels": {"L1": {"tau_retrieval": 0.5, "tau_ground": 0.8}}}
    corpus = [{"id": "d_paris", "title": "France", "text": "The capital of France is Paris."}]
    llm = LLM(cache_path=":memory:")
    retriever = HybridRetriever(corpus=corpus)
    generator = Generator(llm=llm)

    state = {"question": "What is the capital city of France?", "level": 1, "signals": {}}
    res = run_level_1(state, llm, retriever, generator, cfg)
    assert "Paris" in res["answer"]
    assert "S2" in res["signals"]
    assert "S3" in res["signals"]
    assert len(res["citations"]) > 0

def test_i3_level2_decomposition():
    llm = LLM(cache_path=":memory:")
    planner = QueryPlanner(llm=llm)
    subs = planner.decompose("Which river flows through the European capital city where the Eiffel Tower is situated?")
    assert len(subs) >= 2
    assert all(len(s) > 0 for s in subs)

def test_i4_full_controller_trace():
    cfg = {
        "budget": {"max_calls_per_query": 20},
        "retrieval": {"k": 3, "k_broad": 5},
        "levels": {
            "L0": {"samples": 3, "tau": 0.67},
            "L1": {"tau_retrieval": 0.5, "tau_ground": 0.8},
            "L2": {"tau_ground": 0.8, "tau_complete": 0.8},
            "L3": {"drafts": 3, "max_revisions": 2}
        }
    }
    corpus = [
        {"id": "doc_eiffel", "title": "Eiffel Tower", "text": "The Eiffel Tower is in Paris France."},
        {"id": "doc_paris", "title": "Paris", "text": "The river Seine flows through Paris."}
    ]
    llm = LLM(cache_path=":memory:")
    retriever = HybridRetriever(corpus=corpus)
    planner = QueryPlanner(llm=llm)
    generator = Generator(llm=llm)

    levels = {
        0: lambda s: run_level_0(s, llm, cfg),
        1: lambda s: run_level_1(s, llm, retriever, generator, cfg),
        2: lambda s: run_level_2(s, llm, retriever, planner, generator, cfg),
        3: lambda s: run_level_3(s, llm, retriever, generator, cfg)
    }

    state = run_controller("Which river flows through the European capital city where the Eiffel Tower is situated?", levels, cfg, llm)
    assert len(state["trace"]) > 0
    assert "calls" in state
    assert "latency" in state
    for step in state["trace"]:
        assert "level" in step
        assert "signals" in step
        assert "calls_used_level" in step

def test_i5_citations_refer_to_real_docs():
    corpus = [
        {"id": "real_doc_1", "title": "A", "text": "Real content A"},
        {"id": "real_doc_2", "title": "B", "text": "Real content B"}
    ]
    llm = LLM(cache_path=":memory:")
    gen = Generator(llm=llm)
    res = gen.answer("Question", corpus)
    for c in res["citations"]:
        assert c in ["real_doc_1", "real_doc_2"]

def test_i6_cached_rerun_zero_calls(tmp_path):
    cache_file = str(tmp_path / "rerun_cache.json")
    cfg = {
        "budget": {"max_calls_per_query": 20},
        "retrieval": {"k": 2, "k_broad": 4},
        "levels": {"L0": {"samples": 3, "tau": 0.67}, "L1": {"tau_retrieval": 0.5, "tau_ground": 0.8}, "L2": {"tau_ground": 0.8, "tau_complete": 0.8}, "L3": {"drafts": 3, "max_revisions": 2}}
    }
    corpus = [{"id": "d1", "title": "T", "text": "Content"}]
    llm = LLM(cache_path=cache_file)
    retriever = HybridRetriever(corpus=corpus)
    planner = QueryPlanner(llm=llm)
    generator = Generator(llm=llm)
    levels = {
        0: lambda s: run_level_0(s, llm, cfg),
        1: lambda s: run_level_1(s, llm, retriever, generator, cfg),
        2: lambda s: run_level_2(s, llm, retriever, planner, generator, cfg),
        3: lambda s: run_level_3(s, llm, retriever, generator, cfg)
    }

    # First run
    s1 = run_controller("What is capital of France?", levels, cfg, llm)
    calls_first = llm.calls
    llm.save()

    # Second run with same cache
    llm2 = LLM(cache_path=cache_file)
    s2 = run_controller("What is capital of France?", levels, cfg, llm2)
    assert s1["answer"] == s2["answer"]
    # Re-run from cache costs zero new backend calls
    assert llm2.calls == 0

