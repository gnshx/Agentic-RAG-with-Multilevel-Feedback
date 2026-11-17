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


def test_task6_same_id_same_content_different_source():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    corpus = []
    retriever = HybridRetriever(corpus=corpus)
    vector_store = LocalVectorStore()

    text = "The sun is a G-type main-sequence star."
    doc_a = json.dumps([{"id": "doc_sun", "title": "Sun (Textbook)", "text": text, "source": "textbook.json"}])
    res_a = ingest_document_to_stores(corpus, retriever, vector_store, source_name="textbook.json", content=doc_a)
    assert res_a["added_count"] == 1
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1

    # Ingest same ID, same text, but DIFFERENT source -> must NOT be skipped as duplicate
    doc_b = json.dumps([{"id": "doc_sun", "title": "Sun (NASA)", "text": text, "source": "nasa.json"}])

    # With on_conflict="error" raises ValueError identifying collision on source
    with pytest.raises(ValueError) as excinfo:
        ingest_document_to_stores(corpus, retriever, vector_store, source_name="nasa.json", content=doc_b, on_conflict="error")
    assert "doc_sun" in str(excinfo.value)

    # With on_conflict="rename", indexes safely without discarding either document
    res_b = ingest_document_to_stores(corpus, retriever, vector_store, source_name="nasa.json", content=doc_b)
    assert res_b["added_count"] == 1
    assert res_b["skipped_count"] == 0
    assert len(corpus) == 2
    assert len(vector_store.documents) == 2

    # Verify both sources exist in both stores
    corpus_sources = {c["source"] for c in corpus}
    vstore_sources = {d["source"] for d in vector_store.documents.values()}
    assert corpus_sources == {"textbook.json", "nasa.json"}
    assert vstore_sources == {"textbook.json", "nasa.json"}


def test_task6_same_source_content_different_ids_across_stores():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    text = "Superconductivity occurs below a characteristic critical temperature."
    source = "physics_lab.json"

    # Simulate existing mismatched stores: corpus has 'corpus_id_101', vector_store has 'vstore_id_202'
    corpus = [{"id": "corpus_id_101", "title": "Superconductivity", "text": text, "source": source}]
    retriever = HybridRetriever(corpus=corpus)
    vector_store = LocalVectorStore()
    vstore_doc = {
        "id": "vstore_id_202",
        "content": text,
        "source": source,
        "score": 0.0,
        "dynamic_weight": 0.25,
        "metadata": {"title": "Superconductivity"}
    }
    vector_store.add_documents([vstore_doc])

    assert "corpus_id_101" in [c["id"] for c in corpus]
    assert "vstore_id_202" in vector_store.documents
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1

    # Ingest document with same source and content
    ingest_json = json.dumps([{"title": "Superconductivity", "text": text, "source": source}])
    res = ingest_document_to_stores(corpus, retriever, vector_store, source_name=source, content=ingest_json)

    # Must reconcile stores without creating duplicate entries
    assert res["added_count"] == 0
    assert res["skipped_count"] == 1
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1

    canonical_id = res["chunk_ids"][0]
    assert canonical_id == "corpus_id_101"
    assert corpus[0]["id"] == canonical_id
    assert canonical_id in vector_store.documents
    assert "vstore_id_202" not in vector_store.documents
    # Preserved dynamic weight and content
    assert vector_store.documents[canonical_id]["content"] == text
    assert vector_store.documents[canonical_id]["dynamic_weight"] == 0.25


def test_task6_same_source_content_same_id_across_both_stores():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    corpus = []
    retriever = HybridRetriever(corpus=corpus)
    vector_store = LocalVectorStore()

    doc_json = json.dumps([{
        "id": "canonical_qed_1",
        "title": "QED",
        "text": "Quantum electrodynamics describes how light and matter interact.",
        "source": "qed.json"
    }])

    # First ingestion
    res1 = ingest_document_to_stores(corpus, retriever, vector_store, source_name="qed.json", content=doc_json)
    assert res1["added_count"] == 1
    assert res1["skipped_count"] == 0
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1

    # Second ingestion with same source, same content, same ID -> must skip duplicate
    res2 = ingest_document_to_stores(corpus, retriever, vector_store, source_name="qed.json", content=doc_json)
    assert res2["added_count"] == 0
    assert res2["skipped_count"] == 1
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1
    assert corpus[0]["id"] == "canonical_qed_1"
    assert "canonical_qed_1" in vector_store.documents


def test_task6_repeated_ingestion_after_reconciliation():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    text = "Bose-Einstein condensation occurs at temperatures close to absolute zero."
    source = "bec.json"

    # Start with misaligned IDs across stores
    corpus = [{"id": "bec_corpus_id", "title": "BEC", "text": text, "source": source}]
    retriever = HybridRetriever(corpus=corpus)
    vector_store = LocalVectorStore()
    vector_store.add_documents([{
        "id": "bec_vstore_id",
        "content": text,
        "source": source,
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {"title": "BEC"}
    }])

    doc_json = json.dumps([{"id": "bec_corpus_id", "title": "BEC", "text": text, "source": source}])

    # Reconcile on first ingestion
    res_reconcile = ingest_document_to_stores(corpus, retriever, vector_store, source_name=source, content=doc_json)
    assert len(corpus) == 1
    assert len(vector_store.documents) == 1
    assert corpus[0]["id"] == "bec_corpus_id"
    assert "bec_corpus_id" in vector_store.documents

    # Repeated ingestions (3 iterations) must remain strictly idempotent
    for _ in range(3):
        res_repeat = ingest_document_to_stores(corpus, retriever, vector_store, source_name=source, content=doc_json)
        assert res_repeat["added_count"] == 0
        assert res_repeat["skipped_count"] == 1
        assert len(corpus) == 1
        assert len(vector_store.documents) == 1
        assert corpus[0]["id"] == "bec_corpus_id"
        assert "bec_corpus_id" in vector_store.documents


def test_task6_reconciliation_persistence_across_store_reopen(tmp_path):
    import sqlite3
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    db_file = str(tmp_path / "recon_persistence.db")
    text = "High-temperature superconductors exhibit zero electrical resistance above 77 Kelvin."
    source = "materials_science.json"

    # 1. Pre-populate SQLite with inferences table and initial vector store state
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inferences (
                id TEXT PRIMARY KEY,
                doc_ids TEXT
            )
        """)
        conn.execute(
            "INSERT INTO inferences (id, doc_ids) VALUES (?, ?)",
            ("inf_sample_1", json.dumps(["vstore_beta_2", "doc_other"]))
        )
        conn.commit()

    vstore1 = LocalVectorStore(db_path=db_file)
    vstore1.add_documents([{
        "id": "vstore_beta_2",
        "content": text,
        "source": source,
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {"category": "condensed_matter"}
    }])
    vstore1.update_weight("vstore_beta_2", 0.45)
    assert vstore1.get_weight("vstore_beta_2") == 0.45

    # 2. Setup corpus with mismatched ID
    corpus = [{
        "id": "corpus_alpha_1",
        "title": "Superconductors",
        "text": text,
        "source": source,
        "metadata": {"category": "condensed_matter"}
    }]
    retriever = HybridRetriever(corpus=corpus)

    # 3. Ingest document -> triggers cross-store ID reconciliation
    doc_json = json.dumps([{
        "id": "corpus_alpha_1",
        "title": "Superconductors",
        "text": text,
        "source": source
    }])
    res = ingest_document_to_stores(corpus, retriever, vstore1, source_name=source, content=doc_json)

    assert res["added_count"] == 0
    assert res["skipped_count"] == 1
    canonical_id = res["chunk_ids"][0]
    assert canonical_id == "corpus_alpha_1"
    assert "corpus_alpha_1" in vstore1.documents
    assert "vstore_beta_2" not in vstore1.documents
    assert vstore1.documents["corpus_alpha_1"]["dynamic_weight"] == 0.45
    assert vstore1.get_weight("corpus_alpha_1") == 0.45
    assert vstore1.get_weight("vstore_beta_2") == 0.0

    # 4. Verify SQLite persistence tables directly
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT doc_id, dynamic_weight FROM document_weights WHERE doc_id = ?", ("corpus_alpha_1",))
        row = cur.fetchone()
        assert row is not None
        assert float(row["dynamic_weight"]) == 0.45

        cur.execute("SELECT doc_id FROM document_weights WHERE doc_id = ?", ("vstore_beta_2",))
        assert cur.fetchone() is None

        cur.execute("SELECT doc_ids FROM inferences WHERE id = ?", ("inf_sample_1",))
        inf_row = cur.fetchone()
        assert inf_row is not None
        reconciled_doc_ids = json.loads(inf_row["doc_ids"])
        assert reconciled_doc_ids == ["corpus_alpha_1", "doc_other"]

    # 5. Simulate closing store and reopening new instance from db_file
    vstore2 = LocalVectorStore(db_path=db_file)
    assert vstore2.get_weight("corpus_alpha_1") == 0.45
    assert vstore2.get_weight("vstore_beta_2") == 0.0

    # Hydrate documents in reopened store
    vstore2.add_documents([{
        "id": "corpus_alpha_1",
        "content": text,
        "source": source,
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {"category": "condensed_matter"}
    }])
    assert vstore2.documents["corpus_alpha_1"]["dynamic_weight"] == 0.45

    # Verify search leverages reconciled vector and dynamic weight
    search_res = vstore2.search("superconductors electrical resistance", top_k=1)
    assert len(search_res) == 1
    assert search_res[0]["id"] == "corpus_alpha_1"
    assert search_res[0]["dynamic_weight"] == 0.45

    # 6. Re-ingestion against reopened store must strictly skip duplicates
    res_reopen = ingest_document_to_stores(corpus, retriever, vstore2, source_name=source, content=doc_json)
    assert res_reopen["added_count"] == 0
    assert res_reopen["skipped_count"] == 1
    assert len(corpus) == 1
    assert len(vstore2.documents) == 1
    assert corpus[0]["id"] == "corpus_alpha_1"
    assert "corpus_alpha_1" in vstore2.documents


def test_task6_reconciliation_persistence_failure_and_rollback():
    from retrieval.document_loader import ingest_document_to_stores

    text = "Topological insulators have conducting surface states protected by time-reversal symmetry."
    source = "topological.json"

    corpus = [{"id": "corpus_topo_1", "title": "Topological", "text": text, "source": source}]
    retriever = HybridRetriever(corpus=corpus)

    class FailingVectorStore:
        def __init__(self):
            self.documents = {"vstore_topo_2": {"id": "vstore_topo_2", "content": text, "source": source}}
            self.vectors = {}

        def reconcile_id(self, old_id: str, new_id: str):
            raise RuntimeError("Simulated SQLite disk I/O failure during ID reconciliation")

    failing_vstore = FailingVectorStore()

    doc_json = json.dumps([{"id": "corpus_topo_1", "title": "Topological", "text": text, "source": source}])

    # Must raise RuntimeError without swallowing and without corrupting corpus
    with pytest.raises(RuntimeError) as excinfo:
        ingest_document_to_stores(corpus, retriever, failing_vstore, source_name=source, content=doc_json)

    assert "Simulated SQLite disk I/O failure" in str(excinfo.value)
    # Corpus remains intact
    assert len(corpus) == 1
    assert corpus[0]["id"] == "corpus_topo_1"


def test_task6_indexing_failure_rolls_back_corpus_and_retriever():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    text = "Graphene is a single layer of carbon atoms arranged in a two-dimensional honeycomb lattice."
    source = "nanomaterials.json"

    # Initial state: corpus has corpus_id_init, vector_store has vstore_canonical_id
    corpus = [{"id": "corpus_id_init", "title": "Graphene", "text": text, "source": source}]
    retriever = HybridRetriever(corpus=corpus)
    orig_tokenized = list(retriever.tokenized_corpus)
    orig_vectors = [v.copy() for v in retriever.dense_vectors]

    vector_store = LocalVectorStore()
    vector_store.add_documents([{
        "id": "vstore_canonical_id",
        "content": text,
        "source": source,
        "score": 0.0,
        "dynamic_weight": 0.35,
        "metadata": {"title": "Graphene"}
    }])

    # Ingest document where cand_id matches vstore_canonical_id, but retriever indexing fails
    doc_json = json.dumps([{"id": "vstore_canonical_id", "title": "Graphene", "text": text, "source": source}])

    def failing_index_corpus(c):
        raise RuntimeError("Simulated memory allocation error during retriever index_corpus")

    retriever.index_corpus = failing_index_corpus

    with pytest.raises(RuntimeError) as excinfo:
        ingest_document_to_stores(corpus, retriever, vector_store, source_name=source, content=doc_json)

    assert "Simulated memory allocation error" in str(excinfo.value)

    # Verify corpus entry ID was rolled back
    assert len(corpus) == 1
    assert corpus[0]["id"] == "corpus_id_init"

    # Verify retriever internal state was restored to snapshot
    assert retriever.corpus[0]["id"] == "corpus_id_init"
    assert retriever.tokenized_corpus == orig_tokenized
    assert len(retriever.dense_vectors) == len(orig_vectors)

    # Verify vector store remains intact
    assert "vstore_canonical_id" in vector_store.documents
    assert vector_store.documents["vstore_canonical_id"]["dynamic_weight"] == 0.35


def test_task6_malformed_telemetry_inferences_json_raises_and_rolls_back_sqlite(tmp_path):
    import sqlite3
    from retrieval.vector_store import LocalVectorStore

    db_file = str(tmp_path / "corrupt_telemetry.db")

    # 1. Initialize SQLite database with an inferences record containing malformed JSON
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE inferences (
                id TEXT PRIMARY KEY,
                doc_ids TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE document_weights (
                doc_id TEXT PRIMARY KEY,
                dynamic_weight REAL,
                updated_at TIMESTAMP
            )
        """)
        # Malformed JSON string that triggers json.loads failure
        conn.execute(
            "INSERT INTO inferences (id, doc_ids) VALUES (?, ?)",
            ("inf_corrupt_99", '{"malformed_json_missing_brace": ["old_chunk_id"')
        )
        conn.commit()

    vstore = LocalVectorStore(db_path=db_file)
    vstore.add_documents([{
        "id": "old_chunk_id",
        "content": "Quantum decoherence causes quantum states to lose phase coherence.",
        "source": "quantum.json",
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {}
    }])
    vstore.update_weight("old_chunk_id", 0.75)
    assert vstore.get_weight("old_chunk_id") == 0.75

    # 2. Reconcile ID must fail loudly on malformed telemetry JSON and roll back transaction
    with pytest.raises(RuntimeError) as excinfo:
        vstore.reconcile_id("old_chunk_id", "new_canonical_id")

    err_msg = str(excinfo.value)
    assert "Malformed telemetry JSON in inferences record 'inf_corrupt_99'" in err_msg

    # 3. Verify SQLite rollback: old_chunk_id still exists with 0.75; new_canonical_id was NOT created
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT doc_id, dynamic_weight FROM document_weights WHERE doc_id = ?", ("old_chunk_id",))
        row_old = cur.fetchone()
        assert row_old is not None
        assert float(row_old["dynamic_weight"]) == 0.75

        cur.execute("SELECT doc_id FROM document_weights WHERE doc_id = ?", ("new_canonical_id",))
        assert cur.fetchone() is None

        # Corrupt inferences record still holds original raw content without partial edit
        cur.execute("SELECT doc_ids FROM inferences WHERE id = ?", ("inf_corrupt_99",))
        inf_row = cur.fetchone()
        assert inf_row["doc_ids"] == '{"malformed_json_missing_brace": ["old_chunk_id"'

    # 4. Verify in-memory state: old_chunk_id is preserved, new_canonical_id does not exist
    assert "old_chunk_id" in vstore.documents
    assert "new_canonical_id" not in vstore.documents
    assert "old_chunk_id" in vstore.vectors
    assert "new_canonical_id" not in vstore.vectors
    assert vstore.documents["old_chunk_id"]["dynamic_weight"] == 0.75


def test_task6_retriever_rollback_preserves_shared_corpus_identity_and_sync():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    text_a = "Spintronics utilizes electron spin in addition to charge."
    source = "physics_spin.json"

    corpus = [{"id": "corpus_spin_1", "title": "Spintronics", "text": text_a, "source": source}]
    retriever = HybridRetriever(corpus=corpus)

    # Verify shared list identity
    assert retriever.corpus is corpus

    vector_store = LocalVectorStore()
    vector_store.add_documents([{
        "id": "vstore_spin_target",
        "content": text_a,
        "source": source,
        "score": 0.0,
        "dynamic_weight": 0.20,
        "metadata": {"title": "Spintronics"}
    }])

    doc_json = json.dumps([{"id": "vstore_spin_target", "title": "Spintronics", "text": text_a, "source": source}])

    # Inject failure into index_corpus
    def failing_index(c):
        raise RuntimeError("Transient indexing GPU/memory bus failure")

    retriever.index_corpus = failing_index

    with pytest.raises(RuntimeError) as excinfo:
        ingest_document_to_stores(corpus, retriever, vector_store, source_name=source, content=doc_json)

    assert "Transient indexing GPU/memory bus failure" in str(excinfo.value)

    # Crucial assertion: shared list identity must be strictly preserved
    assert retriever.corpus is corpus
    assert len(corpus) == 1
    assert corpus[0]["id"] == "corpus_spin_1"

    # Restore retriever indexing capability
    retriever.index_corpus = HybridRetriever.index_corpus.__get__(retriever, HybridRetriever)
    retriever.index_corpus(corpus)

    # Perform a subsequent valid ingestion and verify both lists stay strictly synchronized
    text_b = "Magnons are quasiparticles associated with spin waves."
    doc_b_json = json.dumps([{"id": "corpus_spin_2", "title": "Magnons", "text": text_b, "source": source}])
    res_b = ingest_document_to_stores(corpus, retriever, vector_store, source_name=source, content=doc_b_json)

    assert res_b["added_count"] == 1
    assert len(corpus) == 2
    assert len(retriever.corpus) == 2
    assert retriever.corpus is corpus
    assert any(c["id"] == "corpus_spin_2" for c in retriever.corpus)


def test_task6_reconcile_id_target_collision_documents_and_vectors_protection():
    import numpy as np
    from retrieval.vector_store import LocalVectorStore

    vstore = LocalVectorStore()
    vstore.add_documents([
        {
            "id": "doc_source_old",
            "content": "Source document to migrate.",
            "source": "source.json",
            "score": 0.0,
            "dynamic_weight": 0.10,
            "metadata": {}
        },
        {
            "id": "doc_target_existing",
            "content": "Target document that already exists with different data.",
            "source": "target.json",
            "score": 0.0,
            "dynamic_weight": 0.90,
            "metadata": {"protect": True}
        }
    ])

    # 1. Target ID collision in self.documents
    with pytest.raises(ValueError) as excinfo:
        vstore.reconcile_id("doc_source_old", "doc_target_existing")

    err_text = str(excinfo.value)
    assert "target ID 'doc_target_existing' already exists in vector store documents" in err_text

    # Verify target document was NOT overwritten and source doc is unchanged
    assert vstore.documents["doc_target_existing"]["content"] == "Target document that already exists with different data."
    assert vstore.documents["doc_target_existing"]["dynamic_weight"] == 0.90
    assert vstore.documents["doc_source_old"]["content"] == "Source document to migrate."

    # 2. Target ID collision in self.vectors
    vstore.vectors["orphan_vector_id"] = np.ones(128, dtype=np.float32)
    with pytest.raises(ValueError) as excinfo_vec:
        vstore.reconcile_id("doc_source_old", "orphan_vector_id")

    assert "target ID 'orphan_vector_id' already exists in vector store vectors" in str(excinfo_vec.value)
    assert "doc_source_old" in vstore.vectors


def test_task6_reconcile_id_target_collision_sqlite_weights_protection(tmp_path):
    import sqlite3
    from retrieval.vector_store import LocalVectorStore

    db_file = str(tmp_path / "target_weights_collision.db")
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE document_weights (
                doc_id TEXT PRIMARY KEY,
                dynamic_weight REAL,
                updated_at TIMESTAMP
            )
        """)
        # Pre-seed target ID with existing high dynamic weight
        conn.execute(
            "INSERT INTO document_weights (doc_id, dynamic_weight, updated_at) VALUES (?, ?, ?)",
            ("target_canonical_id", 0.85, "2026-11-17T00:00:00")
        )
        conn.commit()

    vstore = LocalVectorStore(db_path=db_file)
    vstore.add_documents([{
        "id": "old_source_id",
        "content": "Sample content to migrate.",
        "source": "sample.json",
        "score": 0.0,
        "dynamic_weight": 0.0,
        "metadata": {}
    }])
    vstore.update_weight("old_source_id", 0.15)
    assert vstore.get_weight("old_source_id") == 0.15

    # Reconcile toward target_canonical_id must be rejected before modifying SQLite
    with pytest.raises(ValueError) as excinfo:
        vstore.reconcile_id("old_source_id", "target_canonical_id")

    assert "target ID 'target_canonical_id' already exists in persisted document weights" in str(excinfo.value)

    # Verify SQLite table integrity: target_canonical_id retains 0.85, old_source_id retains 0.15
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT doc_id, dynamic_weight FROM document_weights WHERE doc_id = ?", ("target_canonical_id",))
        target_row = cur.fetchone()
        assert target_row is not None
        assert float(target_row["dynamic_weight"]) == 0.85

        cur.execute("SELECT doc_id, dynamic_weight FROM document_weights WHERE doc_id = ?", ("old_source_id",))
        source_row = cur.fetchone()
        assert source_row is not None
        assert float(source_row["dynamic_weight"]) == 0.15


def test_task6_corpus_target_collision_protection():
    from retrieval.document_loader import ingest_document_to_stores
    from retrieval.vector_store import LocalVectorStore

    source = "physics_notes.json"
    text_a = "Alpha particles consist of two protons and two neutrons."
    text_b = "Beta particles are high-energy, high-speed electrons or positrons."

    # Corpus has 'doc_alpha' with text_a and 'doc_beta' with text_b
    corpus = [
        {"id": "doc_alpha", "title": "Alpha Decay", "text": text_a, "source": source},
        {"id": "doc_beta", "title": "Beta Decay", "text": text_b, "source": source}
    ]
    retriever = HybridRetriever(corpus=corpus)
    vector_store = LocalVectorStore()

    # Vector store has 'doc_alpha' holding text_b (a cross-store ID collision scenario)
    vector_store.add_documents([
        {
            "id": "doc_alpha",
            "content": text_b,
            "source": source,
            "score": 0.0,
            "dynamic_weight": 0.30,
            "metadata": {"title": "Beta Decay"}
        }
    ])

    # Now ingest text_b with explicit candidate id="doc_alpha"
    # cand_id is "doc_alpha" == vstore_matched_id ("doc_alpha")
    # But target canonical ID "doc_alpha" already exists in corpus as doc_alpha (with text_a != text_b)!
    cand_doc = json.dumps([{"id": "doc_alpha", "title": "Beta Decay", "text": text_b, "source": source}])

    with pytest.raises(ValueError) as excinfo:
        ingest_document_to_stores(corpus, retriever, vector_store, source_name=source, content=cand_doc)

    err_msg = str(excinfo.value)
    assert "target canonical ID 'doc_alpha' already exists in corpus as a different document." in err_msg

    # Verify both corpus documents remain completely unchanged
    assert len(corpus) == 2
    assert corpus[0]["id"] == "doc_alpha"
    assert corpus[0]["text"] == text_a
    assert corpus[1]["id"] == "doc_beta"
    assert corpus[1]["text"] == text_b

    # Verify vector store remains completely unchanged
    assert len(vector_store.documents) == 1
    assert "doc_alpha" in vector_store.documents
    assert vector_store.documents["doc_alpha"]["content"] == text_b
    assert vector_store.documents["doc_alpha"]["dynamic_weight"] == 0.30


def test_task6_reconcile_id_in_memory_failure_rolls_back_sqlite(tmp_path):
    import sqlite3
    from retrieval.vector_store import LocalVectorStore

    db_file = str(tmp_path / "reconcile_rollback.db")
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE document_weights (
                doc_id TEXT PRIMARY KEY,
                dynamic_weight REAL,
                updated_at TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT INTO document_weights (doc_id, dynamic_weight, updated_at) VALUES (?, ?, ?)",
            ("old_persistent_id", 0.60, "2026-11-17T00:00:00")
        )
        conn.commit()

    vstore = LocalVectorStore(db_path=db_file)
    vstore.add_documents([{
        "id": "old_persistent_id",
        "content": "Original document content before simulated failure.",
        "source": "sim.json",
        "score": 0.0,
        "dynamic_weight": 0.60,
        "metadata": {"state": "original"}
    }])

    # Simulate in-memory failure during in-memory migration
    class FailingVectors(dict):
        def pop(self, key, *args):
            if key == "old_persistent_id":
                raise RuntimeError("Simulated in-memory crash during ID reconciliation")
            return super().pop(key, *args)

    vstore.vectors = FailingVectors(vstore.vectors)

    with pytest.raises(RuntimeError) as excinfo:
        vstore.reconcile_id("old_persistent_id", "new_migrated_id")

    assert "Simulated in-memory crash" in str(excinfo.value)

    # Verify SQLite transaction rolled back completely:
    # 1. 'old_persistent_id' STILL has its original dynamic_weight (0.60)
    # 2. 'new_migrated_id' does NOT exist in document_weights
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT doc_id, dynamic_weight FROM document_weights WHERE doc_id = ?", ("old_persistent_id",))
        old_row = cur.fetchone()
        assert old_row is not None
        assert float(old_row["dynamic_weight"]) == 0.60

        cur.execute("SELECT 1 FROM document_weights WHERE doc_id = ?", ("new_migrated_id",))
        assert cur.fetchone() is None

    # Verify in-memory state is rolled back intact
    assert "old_persistent_id" in vstore.documents
    assert vstore.documents["old_persistent_id"]["id"] == "old_persistent_id"
    assert "new_migrated_id" not in vstore.documents
    assert "old_persistent_id" in vstore.vectors
    assert "new_migrated_id" not in vstore.vectors


def test_task6_reconcile_id_atomic_write_lock_same_transaction_and_concurrency(tmp_path):
    import sqlite3
    from retrieval.vector_store import LocalVectorStore

    db_file = str(tmp_path / "atomic_concurrency.db")
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE document_weights (
                doc_id TEXT PRIMARY KEY,
                dynamic_weight REAL,
                updated_at TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT INTO document_weights (doc_id, dynamic_weight, updated_at) VALUES (?, ?, ?)",
            ("doc_src_orig", 0.45, "2026-11-17T00:00:00")
        )
        conn.commit()

    vstore = LocalVectorStore(db_path=db_file)
    vstore.add_documents([{
        "id": "doc_src_orig",
        "content": "Source text for atomic reconciliation test.",
        "source": "source.json",
        "score": 0.0,
        "dynamic_weight": 0.45,
        "metadata": {}
    }])

    # 1. Track executed statements and verify exact single-connection usage
    real_get_conn = vstore._get_connection
    conns = []
    executed_ops = []
    concurrent_write_blocked = []

    class TrackedConn:
        def __init__(self, raw):
            self.raw = raw
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
        def execute(self, sql, *args):
            executed_ops.append(("conn.execute", sql.strip()))
            return self.raw.execute(sql, *args)
        def cursor(self):
            cur = self.raw.cursor()
            class TrackedCur:
                def __init__(self, c):
                    self.c = c
                def execute(self, sql, *args):
                    clean_sql = sql.strip()
                    executed_ops.append(("cur.execute", clean_sql))
                    # Probe concurrent access during target-ID check under BEGIN IMMEDIATE
                    if "SELECT 1 FROM document_weights WHERE doc_id = ?" in clean_sql:
                        try:
                            # Secondary connection with low timeout to verify immediate write lock holds
                            conn_probe = sqlite3.connect(db_file, timeout=0.05)
                            conn_probe.execute(
                                "INSERT INTO document_weights VALUES (?, ?, ?)",
                                ("doc_tgt_new", 0.99, "2026-11-17T00:00:00")
                            )
                            conn_probe.commit()
                            concurrent_write_blocked.append(False)
                        except sqlite3.OperationalError:
                            concurrent_write_blocked.append(True)
                        finally:
                            conn_probe.close()
                    return self.c.execute(sql, *args)
                def fetchone(self):
                    return self.c.fetchone()
                def fetchall(self):
                    return self.c.fetchall()
            return TrackedCur(cur)
        def commit(self):
            executed_ops.append(("commit", ""))
            return self.raw.commit()
        def rollback(self):
            executed_ops.append(("rollback", ""))
            return self.raw.rollback()
        def close(self):
            executed_ops.append(("close", ""))
            return self.raw.close()

    vstore._get_connection = lambda: (conns.append(1), TrackedConn(real_get_conn()))[1]

    # Reconcile from doc_src_orig to doc_tgt_new
    res = vstore.reconcile_id("doc_src_orig", "doc_tgt_new")
    assert res is True

    # 2. Assertions: Single connection used (no separate conn_check)
    assert len(conns) == 1, "Collision check and migration must use the exact same SQLite connection."

    # 3. Assertions: BEGIN IMMEDIATE ran before collision check and migration
    op_names = [op[0] for op in executed_ops]
    op_sqls = [op[1] for op in executed_ops]
    assert executed_ops[0] == ("conn.execute", "BEGIN IMMEDIATE")

    check_idx = next(i for i, sql in enumerate(op_sqls) if "SELECT 1 FROM document_weights WHERE doc_id = ?" in sql)
    insert_idx = next(i for i, sql in enumerate(op_sqls) if "INSERT INTO document_weights" in sql)
    commit_idx = op_names.index("commit")

    assert check_idx < insert_idx < commit_idx, "Collision check must precede INSERT and both occur within the same transaction."

    # 4. Assertions: Concurrent write was blocked by the BEGIN IMMEDIATE lock
    assert concurrent_write_blocked == [True], "Concurrent write must be locked out by BEGIN IMMEDIATE."

    # 5. Assertions: Final database state has correct dynamic_weight and source doc was cleaned up
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT doc_id, dynamic_weight FROM document_weights WHERE doc_id = ?", ("doc_tgt_new",))
        target_row = cur.fetchone()
        assert target_row is not None
        assert float(target_row["dynamic_weight"]) == 0.45

        cur.execute("SELECT 1 FROM document_weights WHERE doc_id = ?", ("doc_src_orig",))
        assert cur.fetchone() is None

    # 6. Verify existing target weight is never overwritten under collision
    # Clear in-memory docs and vectors to directly exercise the persisted document_weights collision check under write lock
    vstore.documents.clear()
    vstore.vectors.clear()
    vstore.add_documents([{
        "id": "another_doc",
        "content": "Another text",
        "source": "source2.json",
        "score": 0.0,
        "dynamic_weight": 0.10,
        "metadata": {}
    }])
    with pytest.raises(ValueError) as excinfo:
        vstore.reconcile_id("another_doc", "doc_tgt_new")
    assert "target ID 'doc_tgt_new' already exists in persisted document weights" in str(excinfo.value)

    # Verify existing target weight (0.45) was not overwritten
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT dynamic_weight FROM document_weights WHERE doc_id = ?", ("doc_tgt_new",))
        row = cur.fetchone()
        assert row is not None
        assert float(row["dynamic_weight"]) == 0.45
