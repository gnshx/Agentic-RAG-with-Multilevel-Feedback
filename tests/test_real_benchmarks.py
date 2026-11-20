import json
import os
import pytest
from eval.prepare_real_benchmarks import generate_doc_id, clean_sentence_join

def test_clean_sentence_join():
    assert clean_sentence_join(["Hello ", " world! "]) == "Hello world!"
    assert clean_sentence_join("A raw string ") == "A raw string"
    assert clean_sentence_join([]) == ""

def test_generate_doc_id_deterministic():
    id1 = generate_doc_id("Sample Title", "Sample body text.")
    id2 = generate_doc_id("Sample Title", "Sample body text.")
    id3 = generate_doc_id("Other Title", "Sample body text.")
    assert id1 == id2
    assert id1.startswith("doc_")
    assert id1 != id3

def test_real_benchmarks_files_exist():
    expected_files = [
        "data/real_dev.json",
        "data/real_test.json",
        "data/real_2wiki_test.json",
        "data/real_corpus.json"
    ]
    for path in expected_files:
        assert os.path.exists(path), f"Expected file {path} to exist"
        assert os.path.getsize(path) > 0, f"Expected file {path} to be non-empty"

def test_real_benchmark_schemas_and_counts():
    with open("data/real_dev.json", "r", encoding="utf-8") as f:
        dev = json.load(f)
    with open("data/real_test.json", "r", encoding="utf-8") as f:
        test = json.load(f)
    with open("data/real_2wiki_test.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
    with open("data/real_corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Question Counts
    assert len(dev) == 100, f"Expected 100 dev questions, got {len(dev)}"
    assert len(test) == 100, f"Expected 100 test questions, got {len(test)}"
    assert len(wiki) == 100, f"Expected 100 2Wiki questions, got {len(wiki)}"

    # Corpus Count > 1000
    assert len(corpus) > 1000, f"Expected >1,000 paragraphs in corpus, got {len(corpus)}"
    corpus_dict = {doc["id"]: doc for doc in corpus}
    assert len(corpus_dict) == len(corpus), "Duplicate document IDs found in corpus"

    # Schema Verification for Corpus
    for doc in corpus[:50]:
        assert "id" in doc and doc["id"].startswith("doc_")
        assert "title" in doc and len(doc["title"]) > 0
        assert "text" in doc and len(doc["text"]) > 0

    # Cross-referencing Questions to Corpus
    for split_name, dataset_items in [("dev", dev), ("test", test), ("wiki", wiki)]:
        for item in dataset_items:
            assert "id" in item
            assert "question" in item and len(item["question"]) > 0
            assert "answer" in item
            assert "gold_paragraphs" in item and len(item["gold_paragraphs"]) > 0
            assert "context" in item and len(item["context"]) > 0
            assert "doc_ids" in item and len(item["doc_ids"]) > 0

            # Verify all referenced paragraphs exist in real_corpus.json
            for p in item["context"]:
                assert p["id"] in corpus_dict, f"Document {p['id']} referenced in {split_name} missing from corpus"
                assert corpus_dict[p["id"]]["title"] == p["title"]


def test_multimodel_benchmark_artifact():
    path = "results/tables/multimodel_benchmark_results.json"
    assert os.path.exists(path), f"Expected {path} to exist"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_models = [
        "llama3.2:3b",
        "mistral:latest",
        "llama3.1:latest",
        "qwen3:8b",
        "qwen3:14b",
        "llama3.1:70b"
    ]
    for m in expected_models:
        assert m in data, f"Model {m} missing from multi-model benchmark results"
        m_res = data[m]
        assert "Always-L0" in m_res
        assert "Always-L1" in m_res
        assert "FeedbackRAG" in m_res
        assert "escalation_rate" in m_res

        for sname in ["Always-L0", "Always-L1", "FeedbackRAG"]:
            s = m_res[sname]
            assert 0.0 <= s["em"] <= 100.0
            assert 0.0 <= s["f1"] <= 100.0
            assert s["avg_lat_ms"] > 0
            assert s["avg_calls"] >= 1.0


def test_multimodel_markdown_report_exists():
    path = "results/tables/multimodel_benchmark_results.md"
    assert os.path.exists(path), f"Expected {path} to exist"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "NVIDIA RTX A6000 48GB" in content
    for m in ["llama3.2:3b", "mistral:latest", "llama3.1:latest", "qwen3:8b", "qwen3:14b", "llama3.1:70b"]:
        assert m in content

