import os
import subprocess
import json
import pytest
from core.embeddings import LocalEmbeddingEngine
from feedback.level3_system_memory import FeedbackRegistry, GoldenCache
from src.llm import LLM, get_llm_instance
from src.generator import Generator
from src.retriever import HybridRetriever
from src.planner import QueryPlanner
from src.levels import run_level_0, run_level_1, run_level_2, run_level_3
from src.controller import run_controller
from src.signals import groundedness

def test_task1_deterministic_embeddings_across_processes():
    cmd = [
        ".venv/bin/python", "-c",
        "from core.embeddings import LocalEmbeddingEngine; "
        "import json; print(json.dumps(LocalEmbeddingEngine().embed_text('What is the capital of France?').tolist()[:5]))"
    ]
    proc1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    proc2 = subprocess.run(cmd, capture_output=True, text=True, check=True)

    vec1 = json.loads(proc1.stdout.strip())
    vec2 = json.loads(proc2.stdout.strip())
    assert vec1 == vec2, "Embeddings across two independent Python processes must be 100% identical"

def test_task1_different_text_produces_different_embeddings():
    engine = LocalEmbeddingEngine()
    v1 = engine.embed_text("What is the capital of France?")
    v2 = engine.embed_text("Who directed the movie Inception?")
    sim = engine.cosine_similarity(v1, v2)
    assert sim < 0.80, f"Expected low similarity for different topics, got {sim}"

def test_task1_goldencache_retrieval_across_engine_restart(tmp_path):
    db_path = str(tmp_path / "test_restart_cache.db")
    reg1 = FeedbackRegistry(db_path=db_path)
    engine1 = LocalEmbeddingEngine()
    cache1 = GoldenCache(reg1, embedding_engine=engine1)

    gid = cache1.store_exemplar(
        query="What is the speed of light in vacuum?",
        answer="Approximately 299,792,458 m/s."
    )
    assert gid is not None

    # Simulate process restart by instantiating new registry and new engine
    reg2 = FeedbackRegistry(db_path=db_path)
    engine2 = LocalEmbeddingEngine()
    cache2 = GoldenCache(reg2, embedding_engine=engine2)

    match = cache2.lookup("What is the speed of light in vacuum?", threshold=0.90)
    assert match is not None, "Failed to retrieve exemplar after restarting engine"
    matched_id, answer, sim = match
    assert matched_id == gid
    assert "299,792,458" in answer
    assert sim >= 0.99

def test_task2_seeded_benchmark_french_capital():
    llm = LLM(backend=None, cache_path=":memory:")
    cfg = {"levels": {"L0": {"samples": 3, "tau": 0.67}}}
    state = {"question": "What is the capital city of France?", "signals": {}}
    res = run_level_0(state, llm, cfg)
    assert res["answer"] == "Paris"
    assert res["signals"]["S1"] == 1.0

def test_task2_unsupported_swallow_airspeed():
    llm = LLM(backend=None, cache_path=":memory:")
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
    corpus = [{"id": "d1", "title": "Puccini", "text": "Turandot is an opera by Giacomo Puccini."}]
    retriever = HybridRetriever(corpus=corpus)
    planner = QueryPlanner(llm=llm)
    generator = Generator(llm=llm)
    levels = {
        0: lambda s: run_level_0(s, llm, cfg),
        1: lambda s: run_level_1(s, llm, retriever, generator, cfg),
        2: lambda s: run_level_2(s, llm, retriever, planner, generator, cfg),
        3: lambda s: run_level_3(s, llm, retriever, generator, cfg)
    }

    state = run_controller("What is the airspeed velocity of an unladen swallow?", levels, cfg, llm)
    assert "Puccini" not in state["answer"], "Unsupported question must never return Puccini"
    assert "Turandot" not in state["answer"]
    assert "insufficient evidence" in state["answer"].lower() or "cannot find" in state["answer"].lower()

def test_task2_contaminated_passage_does_not_trigger_unrelated_answer():
    llm = LLM(backend=None, cache_path=":memory:")
    gen = Generator(llm=llm)
    passages = [
        {"id": "doc_turandot", "title": "Turandot", "text": "Turandot is an opera by Giacomo Puccini featuring the aria Nessun dorma."}
    ]
    res = gen.answer("What is the airspeed velocity of an unladen swallow?", passages)
    assert "Puccini" not in res["answer"], "Passage content must not trigger false question matching"
    assert "Giacomo" not in res["answer"]
    assert "cannot find sufficient evidence" in res["answer"].lower()

def test_task2_unsupported_groundedness_score_is_zero():
    llm = LLM(backend=None, cache_path=":memory:")
    passages = [
        {"id": "doc_turandot", "title": "Turandot", "text": "Turandot is an opera by Giacomo Puccini featuring the aria Nessun dorma."}
    ]
    unsupported_ans = "Based on the provided passages, I cannot find sufficient evidence to answer this question."
    score = groundedness(llm, unsupported_ans, passages)
    assert score == 0.0, f"Unsupported answer must receive groundedness 0.0, got {score}"

def test_task3_persistent_document_weights_across_restarts(tmp_path):
    from retrieval.vector_store import LocalVectorStore
    from core.state import DocumentChunk
    db_file = str(tmp_path / "weights_test.db")

    # Instance 1: Add document and update weight
    vstore1 = LocalVectorStore(db_path=db_file)
    doc: DocumentChunk = {
        "id": "doc_persistent_1",
        "content": "Reinforcement learning from human feedback and dynamic document weighting.",
        "source": "knowledge_base",
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {}
    }
    vstore1.add_documents([doc])
    assert vstore1.documents["doc_persistent_1"]["dynamic_weight"] == 0.0

    # Apply feedback delta (+0.35)
    updated = vstore1.update_weight("doc_persistent_1", delta=0.35)
    assert updated is True
    assert vstore1.documents["doc_persistent_1"]["dynamic_weight"] == 0.35
    assert vstore1.get_weight("doc_persistent_1") == 0.35

    # Instance 2 (Simulating process restart): New instance pointing to same SQLite database
    vstore2 = LocalVectorStore(db_path=db_file)
    # Documents dictionary is initially empty in new instance
    assert "doc_persistent_1" not in vstore2.documents
    # Directly querying persisted weight succeeds
    assert vstore2.get_weight("doc_persistent_1") == 0.35

    # When doc is added to Instance 2, its dynamic_weight is automatically hydrated from SQLite
    fresh_doc: DocumentChunk = {
        "id": "doc_persistent_1",
        "content": "Reinforcement learning from human feedback and dynamic document weighting.",
        "source": "knowledge_base",
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {}
    }
    vstore2.add_documents([fresh_doc])
    assert vstore2.documents["doc_persistent_1"]["dynamic_weight"] == 0.35

def test_task4_explicit_ollama_failure_handling():
    from core.llm_provider import OllamaLLM
    import pytest

    # Bad port: must raise RuntimeError, not silently fall back to LocalMockLLM
    bad_ollama = OllamaLLM(base_url="http://localhost:59999", model="nonexistent", allow_fallback=False)
    with pytest.raises(RuntimeError) as excinfo:
        bad_ollama.invoke("Test prompt")
    assert "Ollama invocation failed" in str(excinfo.value)
    assert "http://localhost:59999" in str(excinfo.value)

    # With allow_fallback=True, it falls back to mock without raising
    fallback_ollama = OllamaLLM(base_url="http://localhost:59999", model="nonexistent", allow_fallback=True)
    res = fallback_ollama.invoke("Test prompt")
    assert res.content != ""

def test_task5_controller_golden_cache_and_registry_integration(tmp_path):
    from src.controller import run_controller
    from src.llm import LLM
    from feedback.level3_system_memory import FeedbackRegistry, GoldenCache
    import yaml

    db_path = str(tmp_path / "unified_test.db")
    reg = FeedbackRegistry(db_path=db_path)
    cache = GoldenCache(registry=reg)
    llm = LLM(backend=None, cache_path=":memory:")

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Pre-store an exemplar into GoldenCache
    cache.store_exemplar(
        query="What is the unified theory of Agentic RAG?",
        answer="The unified theory combines multi-hop decomposition, calibration, and multilevel memory."
    )

    levels_map = {
        0: lambda s: s,
        1: lambda s: s,
        2: lambda s: s,
        3: lambda s: s
    }

    # 2. Run controller on the cached query: should short-circuit with 0 calls
    state = run_controller(
        question="What is the unified theory of Agentic RAG?",
        levels=levels_map,
        cfg=cfg,
        llm=llm,
        golden_cache=cache,
        registry=reg
    )

    assert state["cached_answer"] is True
    assert "unified theory combines" in state["answer"]
    assert state["calls"] == 0
    assert state["system_log_id"] is not None

    # Verify log entry in registry
    inf = reg.get_inference(state["system_log_id"])
    assert inf is not None
    assert inf["query"] == "What is the unified theory of Agentic RAG?"

def test_task5_api_thresholds_mode_execution():
    from fastapi.testclient import TestClient
    from api.app import app, registry

    with TestClient(app) as c:
        payload = {
            "query": "What is the capital city of France?",
            "session_id": "test_unified_api",
            "mode": "thresholds"
        }
        res = c.post("/api/query", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["generation"] != ""
        assert "Paris" in data["generation"]
        assert data["system_log_id"] is not None

        # Clean up test entry
        with registry._get_connection() as conn:
            conn.cursor().execute("DELETE FROM inferences WHERE id = ?", (data["system_log_id"],))
            conn.commit()


def test_task6_raw_text_ingestion_updates_both_stores_and_deduplicates():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    corpus = []
    retriever = HybridRetriever(corpus=corpus)
    vector_store = LocalVectorStore()

    text_content = "The James Webb Space Telescope operates at the Sun-Earth L2 Lagrange point."
    res = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="user_input",
        content=text_content,
        title="JWST Orbit"
    )

    assert res["added_count"] == 1
    assert res["skipped_count"] == 0
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1

    doc_id = res["chunk_ids"][0]
    assert doc_id.startswith("custom_")
    assert corpus[0]["id"] == doc_id
    assert vector_store.documents[doc_id]["id"] == doc_id
    assert vector_store.documents[doc_id]["metadata"]["title"] == "JWST Orbit"

    # Search through HybridRetriever
    h_results = retriever.search("Where does the James Webb Space Telescope operate?", k=3)
    assert len(h_results) > 0
    assert h_results[0]["id"] == doc_id
    assert "Lagrange point" in h_results[0]["text"]

    # Search through LocalVectorStore
    v_results = vector_store.search("James Webb Space Telescope Lagrange", top_k=3)
    assert len(v_results) > 0
    assert v_results[0]["id"] == doc_id
    assert "Lagrange point" in v_results[0]["content"]

    # Re-ingesting identical content must prevent duplicates
    res2 = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="user_input",
        content=text_content,
        title="JWST Orbit"
    )
    assert res2["added_count"] == 0
    assert res2["skipped_count"] == 1
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1


def test_task6_txt_and_md_file_ingestion_updates_both_stores_and_retrieves():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    corpus = []
    retriever = HybridRetriever(corpus=corpus)
    vector_store = LocalVectorStore()

    # 1. Ingest .txt file with multiple paragraphs
    p1 = "Alpha Centauri is a triple star system consisting of Rigil Kentaurus, Toliman, and Proxima Centauri."
    p2 = "Proxima Centauri is a red dwarf star located 4.2465 light-years from Earth."
    txt_content = f"{p1}\n\n{p2}"

    res_txt = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="astronomy_guide.txt",
        content=txt_content
    )
    assert res_txt["added_count"] == 2
    assert res_txt["skipped_count"] == 0
    assert len(corpus) == 2
    assert len(vector_store.documents) == 2

    for cid in res_txt["chunk_ids"]:
        assert cid in vector_store.documents
        assert any(c["id"] == cid for c in corpus)

    # Retrieval assertions on both stores
    h_res = retriever.search("red dwarf star distance to Earth", k=2)
    assert any("Proxima Centauri" in doc["text"] for doc in h_res)
    v_res = vector_store.search("Proxima Centauri red dwarf", top_k=2)
    assert any("4.2465 light-years" in doc["content"] for doc in v_res)

    # 2. Ingest .md file
    md_content = "# Quantum Computing\n\nQuantum superposition enables qubits to exist in linear combinations of basis states.\n\nQuantum entanglement creates non-local correlations between distinct particles."
    res_md = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="quantum.md",
        content=md_content
    )
    assert res_md["added_count"] >= 2
    assert len(corpus) == 2 + res_md["added_count"]
    assert len(vector_store.documents) == 2 + res_md["added_count"]

    # Duplicate prevention on re-upload of .md file
    res_md_again = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="quantum.md",
        content=md_content
    )
    assert res_md_again["added_count"] == 0
    assert res_md_again["skipped_count"] == res_md["added_count"]
    assert len(corpus) == 2 + res_md["added_count"]
    assert len(vector_store.documents) == 2 + res_md["added_count"]


def test_task6_json_file_ingestion_formats_and_deduplication():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    corpus = []
    retriever = HybridRetriever(corpus=corpus)
    vector_store = LocalVectorStore()

    # 1. JSON list of structured dicts with explicit IDs
    json_data = json.dumps([
        {
            "id": "doc_bio_1",
            "title": "DNA Structure",
            "text": "DNA is composed of two polynucleotide chains that coil around each other to form a double helix.",
            "metadata": {"field": "genetics"}
        },
        {
            "id": "doc_bio_2",
            "title": "CRISPR-Cas9",
            "text": "CRISPR-Cas9 is an RNA-guided targeted genome editing tool derived from bacterial adaptive immune systems.",
            "metadata": {"field": "biotechnology"}
        }
    ])

    res_json = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="biology.json",
        content=json_data
    )
    assert res_json["added_count"] == 2
    assert res_json["chunk_ids"] == ["doc_bio_1", "doc_bio_2"]
    assert "doc_bio_1" in vector_store.documents
    assert "doc_bio_2" in vector_store.documents
    assert vector_store.documents["doc_bio_1"]["metadata"]["field"] == "genetics"
    assert vector_store.documents["doc_bio_2"]["metadata"]["field"] == "biotechnology"

    # Search in both stores
    h_res = retriever.search("RNA-guided genome editing", k=2)
    assert h_res[0]["id"] == "doc_bio_2"
    v_res = vector_store.search("double helix polynucleotide chains", top_k=2)
    assert v_res[0]["id"] == "doc_bio_1"

    # Re-ingest biology.json -> all skipped as duplicates
    res_json2 = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="biology.json",
        content=json_data
    )
    assert res_json2["added_count"] == 0
    assert res_json2["skipped_count"] == 2
    assert len(corpus) == 2
    assert len(vector_store.documents) == 2

    # 2. JSON list of raw strings
    json_strings = json.dumps([
        "Photosynthesis converts solar light energy into chemical energy stored in carbohydrates.",
        "Cellular respiration produces ATP molecules from biochemical energy in oxygenated cells."
    ])
    res_str_json = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="cellular.json",
        content=json_strings
    )
    assert res_str_json["added_count"] == 2
    for cid in res_str_json["chunk_ids"]:
        assert cid in vector_store.documents
        assert any(c["id"] == cid for c in corpus)

    # 3. Single JSON dict with 'content' instead of 'text'
    single_json = json.dumps({
        "title": "Mitochondria Powerhouse",
        "content": "Mitochondria are membrane-bound cell organelles that generate most chemical energy."
    })
    res_single = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="mitochondria.json",
        content=single_json
    )
    assert res_single["added_count"] == 1
    cid = res_single["chunk_ids"][0]
    assert cid in vector_store.documents
    assert "membrane-bound cell organelles" in vector_store.documents[cid]["content"]
    assert any(c["id"] == cid for c in corpus)


def test_task6_store_split_brain_recovery_and_empty_handling():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    # Simulating legacy split-brain where a doc exists in corpus but not in vector_store
    legacy_doc = {
        "id": "legacy_doc_split_1",
        "title": "Legacy Split Brain",
        "text": "This document was previously indexed into corpus but missed by vector store.",
        "source": "split_test.txt"
    }
    corpus = [legacy_doc]
    retriever = HybridRetriever(corpus=corpus)
    vector_store = LocalVectorStore()

    # Vector store is initially missing legacy_doc_split_1
    assert "legacy_doc_split_1" not in vector_store.documents

    # Ingesting split_test.txt synchronizes the missing store without duplicating corpus
    res = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="split_test.txt",
        content="This document was previously indexed into corpus but missed by vector store."
    )
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1
    assert vector_store.documents["legacy_doc_split_1"]["content"] == legacy_doc["text"]

    # Empty content handling
    res_empty = ingest_document_to_stores(
        corpus=corpus,
        retriever=retriever,
        vector_store=vector_store,
        source_name="empty.txt",
        content="   \n\n   "
    )
    assert res_empty["added_count"] == 0
    assert res_empty["skipped_count"] == 0

    # Invalid JSON error handling
    with pytest.raises(Exception):
        ingest_document_to_stores(
            corpus=corpus,
            retriever=retriever,
            vector_store=vector_store,
            source_name="corrupted.json",
            content="invalid json {"
        )


def test_task6_collision_and_deduplication_matrix():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    corpus = []
    retriever = HybridRetriever(corpus=corpus)
    vector_store = LocalVectorStore()

    # Base document: ID "doc_shared_100"
    base_json = json.dumps([
        {
            "id": "doc_shared_100",
            "title": "Speed of Light",
            "text": "The speed of light in vacuum is exactly 299,792,458 m/s.",
            "source": "physics.json"
        }
    ])
    res1 = ingest_document_to_stores(corpus, retriever, vector_store, source_name="physics.json", content=base_json)
    assert res1["added_count"] == 1
    assert res1["skipped_count"] == 0
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1
    assert "doc_shared_100" in vector_store.documents

    # 1. Same ID + same content -> skip duplicate
    res_same_same = ingest_document_to_stores(corpus, retriever, vector_store, source_name="physics.json", content=base_json)
    assert res_same_same["added_count"] == 0
    assert res_same_same["skipped_count"] == 1
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1

    # 2. Same ID + different content -> handle safely without silent discard
    colliding_json = json.dumps([
        {
            "id": "doc_shared_100",
            "title": "Gravitational Constant",
            "text": "The Newtonian gravitational constant G is approximately 6.67430e-11 m^3 kg^-1 s^-2.",
            "source": "physics.json"
        }
    ])
    # 2a. on_conflict="error" raises ValueError with clear message
    with pytest.raises(ValueError) as excinfo:
        ingest_document_to_stores(corpus, retriever, vector_store, source_name="physics.json", content=colliding_json, on_conflict="error")
    assert "collision" in str(excinfo.value).lower()
    assert "doc_shared_100" in str(excinfo.value)

    # 2b. default on_conflict="rename" generates deterministic alternative ID and indexes both
    res_collision = ingest_document_to_stores(corpus, retriever, vector_store, source_name="physics.json", content=colliding_json)
    assert res_collision["added_count"] == 1
    assert res_collision["skipped_count"] == 0
    assert len(corpus) == 2
    assert len(vector_store.documents) == 2
    alt_id = res_collision["chunk_ids"][0]
    assert alt_id != "doc_shared_100"
    assert alt_id.startswith("doc_shared_100_")
    # Both documents must exist in both stores with identical IDs
    assert corpus[0]["id"] == "doc_shared_100"
    assert corpus[1]["id"] == alt_id
    assert vector_store.documents["doc_shared_100"]["content"].startswith("The speed of light")
    assert vector_store.documents[alt_id]["content"].startswith("The Newtonian gravitational constant")
    # Verify retrieval finds both
    h1 = retriever.search("speed of light", k=2)
    assert any(d["id"] == "doc_shared_100" for d in h1)
    h2 = retriever.search("gravitational constant G", k=2)
    assert any(d["id"] == alt_id for d in h2)

    # 3. Same content + different IDs -> deduplicate correctly
    diff_id_same_content = json.dumps([
        {
            "id": "doc_completely_different_id_999",
            "title": "Speed of Light Duplicate",
            "text": "The speed of light in vacuum is exactly 299,792,458 m/s.",
            "source": "physics.json"
        }
    ])
    res_diff_id = ingest_document_to_stores(corpus, retriever, vector_store, source_name="physics.json", content=diff_id_same_content)
    assert res_diff_id["added_count"] == 0
    assert res_diff_id["skipped_count"] == 1
    assert len(corpus) == 2
    assert len(vector_store.documents) == 2

    # 4. Existing document present in only one store -> synchronize safely
    only_in_vstore = {
        "id": "doc_vstore_only",
        "content": "Special relativity was published by Albert Einstein in 1905.",
        "source": "relativity.json",
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {"title": "Special Relativity"}
    }
    vector_store.add_documents([only_in_vstore])
    assert "doc_vstore_only" in vector_store.documents
    assert not any(c.get("id") == "doc_vstore_only" for c in corpus)

    sync_json = json.dumps([
        {
            "id": "doc_vstore_only",
            "title": "Special Relativity",
            "text": "Special relativity was published by Albert Einstein in 1905.",
            "source": "relativity.json"
        }
    ])
    res_sync_v = ingest_document_to_stores(corpus, retriever, vector_store, source_name="relativity.json", content=sync_json)
    assert res_sync_v["added_count"] == 1
    # Corpus has now received the document with the identical ID
    assert any(c.get("id") == "doc_vstore_only" for c in corpus)
    assert len(vector_store.documents) == 3
    assert len(corpus) == 3

    # 5. Repeated uploads remain strictly idempotent
    for _ in range(3):
        res_repeat = ingest_document_to_stores(corpus, retriever, vector_store, source_name="relativity.json", content=sync_json)
        assert res_repeat["added_count"] == 0
        assert res_repeat["skipped_count"] == 1
        assert len(corpus) == 3
        assert len(vector_store.documents) == 3
