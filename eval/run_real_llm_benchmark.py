import json
import time
import os
import sys
import argparse
import urllib.request
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval.metrics import em, f1, normalize_answer
from src.retriever import HybridRetriever

def get_ollama_endpoint():
    # Detect if GPU tunnel on port 11435 is active, else fallback to 11434
    for port, preferred_model in [(11435, "qwen2.5-coder:7b"), (11434, "qwen2.5:0.5b")]:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as res:
                data = json.loads(res.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                for pm in [preferred_model, "qwen2.5-coder:7b", "gemma2:2b", "qwen2.5-coder:3b", "qwen2.5:0.5b"]:
                    if pm in models:
                        return f"http://localhost:{port}/api/generate", pm
                if models:
                    return f"http://localhost:{port}/api/generate", models[0]
        except Exception:
            continue
    return "http://localhost:11434/api/generate", "qwen2.5:0.5b"

def make_fast_ollama():
    url, model = get_ollama_endpoint()
    print(f"[*] Live LLM Backend: Connected to {url} using model '{model}'")
    def call_ollama(prompt: str, max_tokens: int = 30) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as res:
                data = json.loads(res.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception as e:
            return f"Error: {e}"
    return call_ollama

def get_targeted_items() -> List[Dict[str, Any]]:
    return [
        {
            "id": "real_01",
            "q": "What is the capital city of France?",
            "gold": "Paris",
            "context": "France is a country in Western Europe. Its capital and largest city is Paris."
        },
        {
            "id": "real_02",
            "q": "What is the primary chemical element of diamond?",
            "gold": "carbon",
            "context": "Diamond is a solid form of the element carbon with its atoms arranged in a crystal structure."
        },
        {
            "id": "real_03",
            "q": "Who composed the opera Turandot?",
            "gold": "Giacomo Puccini",
            "context": "Turandot is an opera in three acts by Giacomo Puccini, completed posthumously by Franco Alfano."
        },
        {
            "id": "real_04",
            "q": "In what year was Alan Turing born?",
            "gold": "1912",
            "context": "Alan Mathison Turing OBE FRS was an English mathematician and computer scientist born on 23 June 1912."
        },
        {
            "id": "real_05",
            "q": "In what year was Ada Lovelace born?",
            "gold": "1815",
            "context": "Augusta Ada King, Countess of Lovelace was an English mathematician born on 10 December 1815."
        },
        {
            "id": "real_06",
            "q": "What is the official elevation of Mount Everest in meters?",
            "gold": "8848",
            "context": "Mount Everest is Earth's highest mountain above sea level, with an official elevation of 8,848 meters."
        },
        {
            "id": "real_07",
            "q": "What is the elevation of K2 in meters?",
            "gold": "8611",
            "context": "K2 is the second-highest mountain on Earth, with an elevation of 8,611 meters."
        },
        {
            "id": "real_08",
            "q": "In what year was the Python programming language first released?",
            "gold": "1991",
            "context": "Python is a high-level programming language created by Guido van Rossum and first released in 1991."
        },
        {
            "id": "real_09",
            "q": "In what year was the Java programming language released?",
            "gold": "1995",
            "context": "Java was originally developed by James Gosling at Sun Microsystems and released in May 1995."
        },
        {
            "id": "real_10",
            "q": "In what city was Albert Einstein born?",
            "gold": "Ulm",
            "context": "Albert Einstein was a German-born theoretical physicist born in the Kingdom of Württemberg city of Ulm."
        },
        {
            "id": "real_11",
            "q": "Which river flows through Paris?",
            "gold": "Seine",
            "context": "The river Seine flows through the heart of Paris, dividing the city into the Left and Right Banks."
        },
        {
            "id": "real_12",
            "q": "Who directed the movie Inception?",
            "gold": "Christopher Nolan",
            "context": "Inception is a 2010 science fiction film written and directed by Christopher Nolan."
        },
        {
            "id": "real_13",
            "q": "In what country was Android Inc founded?",
            "gold": "United States",
            "context": "Android, Inc. was founded in Palo Alto, California, United States, in October 2003."
        },
        {
            "id": "real_14",
            "q": "Who invented the World Wide Web?",
            "gold": "Tim Berners-Lee",
            "context": "Sir Timothy John Berners-Lee is an English computer scientist best known as the inventor of the World Wide Web."
        },
        {
            "id": "real_15",
            "q": "What is the native home country of Tim Berners-Lee?",
            "gold": "United Kingdom",
            "context": "Tim Berners-Lee was born in London, England, United Kingdom, and studied at Queen's College, Oxford."
        }
    ]

def run_real_evaluation(
    use_retrieval: bool = False,
    dataset: str = "targeted",
    limit: Optional[int] = None,
    output_path: Optional[str] = None
):
    call_llm = make_fast_ollama()

    retriever = None
    if use_retrieval:
        corpus_file = "data/real_corpus.json" if dataset == "real_test" else "data/corpus.json"
        if os.path.exists(corpus_file):
            with open(corpus_file, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            retriever = HybridRetriever(corpus=corpus)
            print(f"[*] Initialized live HybridRetriever over '{corpus_file}' ({len(corpus)} documents)")

    if dataset == "real_test" and os.path.exists("data/real_test.json"):
        with open("data/real_test.json", "r", encoding="utf-8") as f:
            raw_items = json.load(f)
        items = [{"id": it["id"], "q": it["question"], "gold": it["answer"]} for it in raw_items]
    else:
        items = get_targeted_items()

    if limit is not None:
        items = items[:limit]

    print(f"Running Real LLM Evaluation with {len(items)} queries (Live Retrieval: {use_retrieval})...\n")

    results = {"Always-L0": [], "Always-L1": [], "FeedbackRAG": []}

    # 1. Always-L0: Direct Answering (no retrieval)
    print("--- 1. Evaluating Always-L0 (Direct LLM, no context) ---")
    for item in items:
        prompt = f"Answer concisely in 1-4 words:\nQuestion: {item['q']}\nAnswer:"
        t0 = time.time()
        pred = call_llm(prompt, max_tokens=15)
        dt = (time.time() - t0) * 1000
        score_em = em(pred, item["gold"])
        score_f1 = f1(pred, item["gold"])
        results["Always-L0"].append({
            "id": item["id"], "q": item["q"], "pred": pred, "gold": item["gold"],
            "em": score_em, "f1": score_f1, "latency_ms": dt, "calls": 1
        })
        print(f"[{item['id']}] Q: {item['q'][:35]}... -> Pred: '{pred}' | Gold: '{item['gold']}' | EM={score_em} F1={round(score_f1, 2)} ({round(dt)}ms)")

    # 2. Always-L1: Single-Pass RAG
    print("\n--- 2. Evaluating Always-L1 (Single-Pass RAG) ---")
    for item in items:
        if use_retrieval and retriever:
            docs = retriever.search(item["q"], k=3)
            ctx = " ".join([d.get("text", "") for d in docs])
        else:
            ctx = item.get("context", "")

        prompt = f"Answer concisely in 1-4 words using only the context:\nContext: {ctx}\nQuestion: {item['q']}\nAnswer:"
        t0 = time.time()
        pred = call_llm(prompt, max_tokens=15)
        dt = (time.time() - t0) * 1000
        score_em = em(pred, item["gold"])
        score_f1 = f1(pred, item["gold"])
        results["Always-L1"].append({
            "id": item["id"], "q": item["q"], "pred": pred, "gold": item["gold"],
            "em": score_em, "f1": score_f1, "latency_ms": dt, "calls": 2  # 1 retrieval + 1 generation
        })
        print(f"[{item['id']}] Pred: '{pred}' | Gold: '{item['gold']}' | EM={score_em} F1={round(score_f1, 2)} ({round(dt)}ms)")

    # 3. FeedbackRAG: MLFQ Escalation Logic
    print("\n--- 3. Evaluating FeedbackRAG (Escalation Queue) ---")
    for item in items:
        t0 = time.time()
        l0_prompt = f"Answer concisely in 1-4 words:\nQuestion: {item['q']}\nAnswer:"
        pred_l0 = call_llm(l0_prompt, max_tokens=15)
        calls = 1

        norm_l0 = normalize_answer(pred_l0)
        uncertainty_phrases = ["cannot", "unknown", "do not know", "not sure", "sorry", "unclear", "no information"]
        is_uncertain = any(phrase in norm_l0 for phrase in uncertainty_phrases)
        is_malformed = (len(norm_l0.split()) == 0 or len(norm_l0.split()) > 15)

        perturbed_prompt = f"State the factual answer directly in 1-3 words:\nQuestion: {item['q']}\nAnswer:"
        pred_l0_alt = call_llm(perturbed_prompt, max_tokens=15)
        calls += 1
        norm_alt = normalize_answer(pred_l0_alt)

        consistency_match = (norm_l0 == norm_alt or norm_l0 in norm_alt or norm_alt in norm_l0) and not is_uncertain and not is_malformed

        if consistency_match:
            final_pred = pred_l0
            resolved_level = 0
        else:
            if use_retrieval and retriever:
                docs = retriever.search(item["q"], k=3)
                ctx = " ".join([d.get("text", "") for d in docs])
            else:
                ctx = item.get("context", "")

            l1_prompt = f"Answer concisely in 1-4 words using only the context:\nContext: {ctx}\nQuestion: {item['q']}\nAnswer:"
            final_pred = call_llm(l1_prompt, max_tokens=15)
            calls += 2
            resolved_level = 1

        dt = (time.time() - t0) * 1000
        score_em = em(final_pred, item["gold"])
        score_f1 = f1(final_pred, item["gold"])
        results["FeedbackRAG"].append({
            "id": item["id"], "q": item["q"], "pred": final_pred, "gold": item["gold"],
            "resolved_level": resolved_level, "em": score_em, "f1": score_f1, "latency_ms": dt, "calls": calls
        })
        print(f"[{item['id']}] L{resolved_level} -> Pred: '{final_pred}' | Gold: '{item['gold']}' | EM={score_em} F1={round(score_f1, 2)} ({round(dt)}ms, {calls} calls)")

    # Summarize Table
    summary = {}
    for sys_name, rows in results.items():
        avg_em = sum(r["em"] for r in rows) / len(rows) * 100
        avg_f1 = sum(r["f1"] for r in rows) / len(rows) * 100
        avg_lat = sum(r["latency_ms"] for r in rows) / len(rows)
        avg_calls = sum(r["calls"] for r in rows) / len(rows)
        summary[sys_name] = {
            "em": round(avg_em, 1),
            "f1": round(avg_f1, 1),
            "avg_latency_ms": round(avg_lat, 1),
            "avg_calls": round(avg_calls, 2)
        }

    print("\n=== Real LLM Summary Results ===")
    print(json.dumps(summary, indent=2))

    target_file = output_path
    if not target_file:
        if use_retrieval or dataset == "real_test":
            target_file = "results/tables/real_end_to_end_benchmark_results.json"
        else:
            target_file = "results/tables/real_llm_benchmark_results.json"

    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    with open(target_file, "w") as f:
        json.dump({"summary": summary, "per_question": results}, f, indent=2)
    print(f"[*] Results saved to {target_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Neural LLM Benchmark for FeedbackRAG")
    parser.add_argument("--use-retrieval", action="store_true", help="Perform real BM25/Dense hybrid retrieval instead of static context")
    parser.add_argument("--dataset", choices=["targeted", "real_test"], default="targeted", help="Benchmark dataset to run")
    parser.add_argument("--limit", type=int, default=None, help="Max queries to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    run_real_evaluation(
        use_retrieval=args.use_retrieval,
        dataset=args.dataset,
        limit=args.limit,
        output_path=args.output
    )
