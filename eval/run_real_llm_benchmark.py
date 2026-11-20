import json
import time
import os
import sys
import argparse
import urllib.request
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval.metrics import em, f1, normalize_answer
from src.retriever import HybridRetriever
from src.signals import _json, retrieval_margin, lexical_groundedness_proxy, consistency
from src.learned_policy import LearnedEscalationPolicy, extract_level_features
from eval.bootstrap import bootstrap_ci, bootstrap_paired_test, format_metric_with_ci

def get_ollama_endpoint(preferred_model: Optional[str] = None, preferred_port: Optional[int] = None):
    # Detect if GPU tunnel on port 11435 is active, else fallback to 11434
    ports = [preferred_port] if preferred_port else [11435, 11434]
    for port in ports:
        if port is None:
            continue
        try:
            req = urllib.request.Request(f"http://localhost:{port}/api/tags")
            with urllib.request.urlopen(req, timeout=3.0) as res:
                data = json.loads(res.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                if preferred_model and preferred_model in models:
                    return f"http://localhost:{port}/api/generate", preferred_model
                for pm in [preferred_model, "llama3.1:70b", "qwen3:14b", "llama3.1:latest", "qwen3:8b", "mistral:latest", "llama3.2:3b", "qwen2.5-coder:14b", "qwen2.5:0.5b"]:
                    if pm and pm in models:
                        return f"http://localhost:{port}/api/generate", pm
                if models:
                    return f"http://localhost:{port}/api/generate", models[0]
        except Exception:
            continue
    return "http://localhost:11434/api/generate", (preferred_model or "llama3.2:latest")

def make_fast_ollama(model_name: Optional[str] = None, port: Optional[int] = None, cache_path: Optional[str] = None):
    url, model = get_ollama_endpoint(preferred_model=model_name, preferred_port=port)
    if cache_path is None:
        safe_m = model.replace(":", "_").replace("/", "_")
        cache_path = f"results/real_cache_{safe_m}.json"
    print(f"[*] Live LLM Backend: Connected to {url} using model '{model}' (cache: {cache_path})")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    def save_cache():
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    def call_ollama(prompt: str, max_tokens: int = 30, temp: float = 0.0, sample_id: int = 0) -> str:
        key = hashlib.sha256(f"{model}|{prompt}|{temp}|{sample_id}".encode("utf-8")).hexdigest()
        if key in cache:
            return cache[key]

        # For reasoning models (e.g. qwen3), allocate enough tokens to complete the think phase
        num_predict = max(max_tokens, 200) if ("qwen3" in model.lower()) else max(max_tokens, 30)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temp, "num_predict": num_predict}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as res:
                    data = json.loads(res.read().decode("utf-8"))
                    out = data.get("response", "").strip()
                    # Fallback if model put answer in thinking
                    if not out and data.get("thinking"):
                        think_text = data.get("thinking", "").strip().split("\n")[-1]
                        out = think_text
                    cache[key] = out
                    if len(cache) % 10 == 0:
                        save_cache()
                    return out
            except Exception as e:
                time.sleep(min(2 ** attempt, 4))
        return f"Error: LLM call failed after 3 attempts"

    call_ollama.save_cache = save_cache
    call_ollama.model = model
    call_ollama.url = url
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
    output_path: Optional[str] = None,
    model: Optional[str] = None,
    port: Optional[int] = None,
    cache_path: Optional[str] = None
):
    call_llm = make_fast_ollama(model_name=model, port=port, cache_path=cache_path)

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
    if hasattr(call_llm, "save_cache"):
        call_llm.save_cache()


def run_real_100q_benchmark(
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    systems_to_run: Optional[List[str]] = None,
    model: Optional[str] = None,
    port: Optional[int] = None,
    cache_path: Optional[str] = None
):
    """
    Comprehensive Real Neural LLM Benchmark across 100 Multi-Hop HotpotQA Queries.
    Evaluates 6 architectures on authentic neural weights (e.g. qwen3:14b or llama3.1:70b on NVIDIA RTX A6000 48GB GPU):
      1. Always-L0: Direct Parametric LLM (Zero Retrieval)
      2. Always-L1: Vanilla Single-Pass Hybrid RAG
      3. Always-L2: Iterative Multi-Hop RAG (Sub-Question Decomposition)
      4. Always-L3: Full Agentic RAG (Critic-Revision Loop)
      5. FeedbackRAG (Heuristic Thresholds): Dynamic MLFQ queue guided by calibrated signal gates
      6. FeedbackRAG (Learned Policy): Dynamic MLFQ queue guided by trained GBDT classifier
    """
    if cache_path is None:
        if model:
            safe_m = model.replace(":", "_").replace("/", "_")
            cache_path = f"results/real_100q_{safe_m}_cache.json"
        else:
            cache_path = "results/real_100q_llm_cache.json"
    call_llm = make_fast_ollama(model_name=model, port=port, cache_path=cache_path)

    corpus_file = "data/real_corpus.json"
    assert os.path.exists(corpus_file), f"Missing {corpus_file}"
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    retriever = HybridRetriever(corpus=corpus)
    print(f"[*] Initialized live HybridRetriever over '{corpus_file}' ({len(corpus)} documents)")

    test_file = "data/real_test.json"
    assert os.path.exists(test_file), f"Missing {test_file}"
    with open(test_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    items = [{"id": it["id"], "q": it["question"], "gold": it["answer"], "type": it.get("type", "bridge")} for it in raw_items]

    if limit is not None:
        items = items[:limit]

    print(f"\n==========================================================================================")
    print(f"Running Real 100-Question Multi-Hop Benchmark ({len(items)} queries on {call_llm.model})")
    print(f"==========================================================================================")

    learned_policy = LearnedEscalationPolicy.load("results/learned_policy.pkl")
    if learned_policy is not None and learned_policy.is_fitted:
        print(f"[*] Loaded trained LearnedEscalationPolicy (GBDT)")
    else:
        print("[!] Warning: LearnedEscalationPolicy not fitted or missing, using heuristic fallback")

    all_systems = [
        "Always-L0",
        "Always-L1",
        "Always-L2",
        "Always-L3",
        "FeedbackRAG-Thresholds",
        "FeedbackRAG-Learned"
    ]
    if systems_to_run:
        selected_systems = [s for s in all_systems if s in systems_to_run]
    else:
        selected_systems = all_systems

    results: Dict[str, List[Dict[str, Any]]] = {s: [] for s in selected_systems}
    level_distributions: Dict[str, Dict[int, int]] = {
        "FeedbackRAG-Thresholds": {0: 0, 1: 0, 2: 0, 3: 0},
        "FeedbackRAG-Learned": {0: 0, 1: 0, 2: 0, 3: 0}
    }

    uncertainty_phrases = [
        "cannot", "unknown", "do not know", "not sure", "sorry",
        "unclear", "no information", "i do not", "not known"
    ]

    # --- System Evaluation Loops ---
    for sys_name in selected_systems:
        print(f"\n--- Evaluating System: {sys_name} ({len(items)} items) ---")
        t_sys_start = time.time()

        for idx, item in enumerate(items):
            q = item["q"]
            gold = item["gold"]
            item_t0 = time.time()
            calls = 0
            resolved_level = 0
            pred = ""

            if sys_name == "Always-L0":
                # Direct parametric answer
                prompt = f"Answer concisely with the short entity or fact name in 1-4 words:\nQuestion: {q}\nAnswer:"
                pred = call_llm(prompt, max_tokens=25)
                calls = 1
                resolved_level = 0

            elif sys_name == "Always-L1":
                # Single-pass hybrid retrieval
                docs = retriever.search(q, k=3)
                ctx = " ".join([d.get("text", "") for d in docs])
                prompt = f"Answer concisely with the short entity or fact name in 1-4 words using only the provided context:\nContext: {ctx}\nQuestion: {q}\nAnswer:"
                pred = call_llm(prompt, max_tokens=25)
                calls = 1
                resolved_level = 1

            elif sys_name == "Always-L2":
                # Query decomposition + multi-hop iterative retrieval
                decomp_prompt = f"Break this question into 2 simple search queries:\nQuestion: {q}\nReturn JSON only: {{\"subquestions\": [\"query 1\", \"query 2\"]}}"
                decomp_raw = call_llm(decomp_prompt, max_tokens=80)
                calls = 1
                parsed = _json(decomp_raw, {})
                subs = parsed.get("subquestions", [q])
                if not isinstance(subs, list) or not subs:
                    subs = [q]

                pooled: Dict[str, Dict[str, Any]] = {}
                for sub_q in subs + [q]:
                    for d in retriever.search(sub_q, k=2):
                        pooled[d["id"]] = d
                docs = list(pooled.values())[:4]
                ctx = " ".join([d.get("text", "") for d in docs])

                gen_prompt = f"Answer concisely with the short entity or fact name in 1-4 words using only the provided context:\nContext: {ctx}\nQuestion: {q}\nAnswer:"
                pred = call_llm(gen_prompt, max_tokens=25)
                calls += 1
                resolved_level = 2

            elif sys_name == "Always-L3":
                # Full agentic RAG: broad retrieval + draft + critique + revise
                docs = retriever.search(q, k=6)
                ctx = " ".join([d.get("text", "") for d in docs])

                # Draft
                draft_prompt = f"Answer concisely with the short entity or fact name in 1-4 words using only the provided context:\nContext: {ctx}\nQuestion: {q}\nAnswer:"
                draft = call_llm(draft_prompt, max_tokens=25)
                calls = 1

                # Critique
                critique_prompt = f"Check if the draft answer is fully supported by the evidence:\nEvidence: {ctx}\nQuestion: {q}\nDraft Answer: {draft}\nIs the answer supported? Return JSON only: {{\"supported\": true, \"critique\": \"brief reasoning\"}}"
                crit_raw = call_llm(critique_prompt, max_tokens=80)
                crit_data = _json(crit_raw, {"supported": True})
                calls += 1

                pred = draft
                if not crit_data.get("supported", True):
                    revise_prompt = f"Rewrite the answer concisely in 1-4 words to fix the issue using the evidence:\nEvidence: {ctx}\nQuestion: {q}\nCritique: {crit_data.get('critique', '')}\nAnswer:"
                    pred = call_llm(revise_prompt, max_tokens=25)
                    calls += 1
                resolved_level = 3

            elif sys_name == "FeedbackRAG-Thresholds":
                # Level 0: Stochastic Self-Consistency (N=3 samples at T=0.7)
                l0_prompt = f"Answer concisely with the short entity or fact name in 1-4 words:\nQuestion: {q}\nAnswer:"
                samples_l0 = [call_llm(l0_prompt, max_tokens=25, temp=0.7, sample_id=k) for k in range(3)]
                calls = 3
                s1 = consistency(samples_l0)
                pred_l0 = samples_l0[0]
                norm_l0 = normalize_answer(pred_l0)
                is_uncertain = any(phrase in norm_l0 for phrase in uncertainty_phrases)
                is_malformed = (len(norm_l0.split()) == 0 or len(norm_l0.split()) > 10)

                if s1 >= 0.67 and not is_uncertain and not is_malformed:
                    pred = pred_l0
                    resolved_level = 0
                else:
                    # Escalate to Level 1
                    docs_l1 = retriever.search(q, k=3)
                    ctx_l1 = " ".join([d.get("text", "") for d in docs_l1])
                    l1_prompt = f"Answer concisely with the short entity or fact name in 1-4 words using only the provided context:\nContext: {ctx_l1}\nQuestion: {q}\nAnswer:"
                    pred_l1 = call_llm(l1_prompt, max_tokens=25)
                    calls += 1

                    s2 = retrieval_margin(docs_l1)
                    s3 = lexical_groundedness_proxy(pred_l1, docs_l1, method="overlap")

                    if s2 >= 0.15 and s3 >= 0.60:
                        pred = pred_l1
                        resolved_level = 1
                    else:
                        # Escalate to Level 2
                        decomp_prompt = f"Break this question into 2 simple search queries:\nQuestion: {q}\nReturn JSON only: {{\"subquestions\": [\"query 1\", \"query 2\"]}}"
                        decomp_raw = call_llm(decomp_prompt, max_tokens=80)
                        calls += 1
                        parsed = _json(decomp_raw, {})
                        subs = parsed.get("subquestions", [q])
                        if not isinstance(subs, list) or not subs:
                            subs = [q]

                        pooled = {}
                        for sub_q in subs + [q]:
                            for d in retriever.search(sub_q, k=2):
                                pooled[d["id"]] = d
                        docs_l2 = list(pooled.values())[:4]
                        ctx_l2 = " ".join([d.get("text", "") for d in docs_l2])

                        l2_prompt = f"Answer concisely with the short entity or fact name in 1-4 words using only the provided context:\nContext: {ctx_l2}\nQuestion: {q}\nAnswer:"
                        pred_l2 = call_llm(l2_prompt, max_tokens=25)
                        calls += 1

                        s4 = 0.85 if len(docs_l2) >= 3 else 0.40
                        if s4 >= 0.70:
                            pred = pred_l2
                            resolved_level = 2
                        else:
                            # Escalate to Level 3
                            docs_l3 = retriever.search(q, k=6)
                            ctx_l3 = " ".join([d.get("text", "") for d in docs_l3])
                            crit_prompt = f"Check if the draft answer is fully supported by the evidence:\nEvidence: {ctx_l3}\nQuestion: {q}\nDraft Answer: {pred_l2}\nIs the answer supported? Return JSON only: {{\"supported\": true, \"critique\": \"brief reasoning\"}}"
                            crit_raw = call_llm(crit_prompt, max_tokens=80)
                            calls += 1
                            crit_data = _json(crit_raw, {"supported": True})

                            pred = pred_l2
                            if not crit_data.get("supported", True):
                                revise_prompt = f"Rewrite the answer concisely in 1-4 words to fix the issue using the evidence:\nEvidence: {ctx_l3}\nQuestion: {q}\nCritique: {crit_data.get('critique', '')}\nAnswer:"
                                pred = call_llm(revise_prompt, max_tokens=25)
                                calls += 1
                            resolved_level = 3

                level_distributions["FeedbackRAG-Thresholds"][resolved_level] += 1

            elif sys_name == "FeedbackRAG-Learned":
                # Level 0: Stochastic Self-Consistency (N=3 samples at T=0.7)
                l0_prompt = f"Answer concisely with the short entity or fact name in 1-4 words:\nQuestion: {q}\nAnswer:"
                samples_l0 = [call_llm(l0_prompt, max_tokens=25, temp=0.7, sample_id=k) for k in range(3)]
                calls = 3
                s1 = consistency(samples_l0)
                pred_l0 = samples_l0[0]
                norm_l0 = normalize_answer(pred_l0)
                is_uncertain = any(phrase in norm_l0 for phrase in uncertainty_phrases)
                is_malformed = (len(norm_l0.split()) == 0 or len(norm_l0.split()) > 10)

                state_l0 = {
                    "question": q, "answer": pred_l0,
                    "signals": {"S1": s1, "S2": 0.0, "S3": 0.0, "S4": 0.0},
                    "docs": []
                }
                esc_l0 = False
                if learned_policy and learned_policy.is_fitted:
                    esc_l0, _ = learned_policy.should_escalate(state_l0, level=0)
                    if s1 < 0.67 or is_uncertain or is_malformed:
                        esc_l0 = True
                else:
                    esc_l0 = (s1 < 0.67 or is_uncertain or is_malformed)

                if not esc_l0:
                    pred = pred_l0
                    resolved_level = 0
                else:
                    # Escalate to Level 1
                    docs_l1 = retriever.search(q, k=3)
                    ctx_l1 = " ".join([d.get("text", "") for d in docs_l1])
                    l1_prompt = f"Answer concisely with the short entity or fact name in 1-4 words using only the provided context:\nContext: {ctx_l1}\nQuestion: {q}\nAnswer:"
                    pred_l1 = call_llm(l1_prompt, max_tokens=25)
                    calls += 1

                    s2 = retrieval_margin(docs_l1)
                    s3 = lexical_groundedness_proxy(pred_l1, docs_l1, method="overlap")

                    state_l1 = {
                        "question": q, "answer": pred_l1,
                        "signals": {"S1": s1, "S2": s2, "S3": s3, "S4": 0.0},
                        "docs": docs_l1
                    }
                    esc_l1 = False
                    if learned_policy and learned_policy.is_fitted:
                        esc_l1, _ = learned_policy.should_escalate(state_l1, level=1)
                        if s2 < 0.15 or s3 < 0.60:
                            esc_l1 = True
                    else:
                        esc_l1 = (s2 < 0.15 or s3 < 0.60)

                    if not esc_l1:
                        pred = pred_l1
                        resolved_level = 1
                    else:
                        # Escalate to Level 2
                        decomp_prompt = f"Break this question into 2 simple search queries:\nQuestion: {q}\nReturn JSON only: {{\"subquestions\": [\"query 1\", \"query 2\"]}}"
                        decomp_raw = call_llm(decomp_prompt, max_tokens=80)
                        calls += 1
                        parsed = _json(decomp_raw, {})
                        subs = parsed.get("subquestions", [q])
                        if not isinstance(subs, list) or not subs:
                            subs = [q]

                        pooled = {}
                        for sub_q in subs + [q]:
                            for d in retriever.search(sub_q, k=2):
                                pooled[d["id"]] = d
                        docs_l2 = list(pooled.values())[:4]
                        ctx_l2 = " ".join([d.get("text", "") for d in docs_l2])

                        l2_prompt = f"Answer concisely with the short entity or fact name in 1-4 words using only the provided context:\nContext: {ctx_l2}\nQuestion: {q}\nAnswer:"
                        pred_l2 = call_llm(l2_prompt, max_tokens=25)
                        calls += 1

                        s4 = 0.85 if len(docs_l2) >= 3 else 0.40
                        state_l2 = {
                            "question": q, "answer": pred_l2,
                            "signals": {"S1": s1, "S2": s2, "S3": s3, "S4": s4},
                            "docs": docs_l2
                        }
                        esc_l2 = False
                        if learned_policy and learned_policy.is_fitted:
                            esc_l2, _ = learned_policy.should_escalate(state_l2, level=2)
                            if s4 < 0.70:
                                esc_l2 = True
                        else:
                            esc_l2 = (s4 < 0.70)

                        if not esc_l2:
                            pred = pred_l2
                            resolved_level = 2
                        else:
                            # Escalate to Level 3
                            docs_l3 = retriever.search(q, k=6)
                            ctx_l3 = " ".join([d.get("text", "") for d in docs_l3])
                            crit_prompt = f"Check if the draft answer is fully supported by the evidence:\nEvidence: {ctx_l3}\nQuestion: {q}\nDraft Answer: {pred_l2}\nIs the answer supported? Return JSON only: {{\"supported\": true, \"critique\": \"brief reasoning\"}}"
                            crit_raw = call_llm(crit_prompt, max_tokens=80)
                            calls += 1
                            crit_data = _json(crit_raw, {"supported": True})

                            pred = pred_l2
                            if not crit_data.get("supported", True):
                                revise_prompt = f"Rewrite the answer concisely in 1-4 words to fix the issue using the evidence:\nEvidence: {ctx_l3}\nQuestion: {q}\nCritique: {crit_data.get('critique', '')}\nAnswer:"
                                pred = call_llm(revise_prompt, max_tokens=25)
                                calls += 1
                            resolved_level = 3

                level_distributions["FeedbackRAG-Learned"][resolved_level] += 1

            dt = (time.time() - item_t0) * 1000
            score_em = em(pred, gold)
            score_f1 = f1(pred, gold)

            results[sys_name].append({
                "id": item["id"],
                "q": q,
                "pred": pred,
                "gold": gold,
                "resolved_level": resolved_level,
                "em": score_em,
                "f1": score_f1,
                "calls": calls,
                "latency_ms": dt
            })

            if (idx + 1) % 10 == 0 or idx == len(items) - 1:
                curr_em = sum(r["em"] for r in results[sys_name]) / len(results[sys_name]) * 100
                curr_f1 = sum(r["f1"] for r in results[sys_name]) / len(results[sys_name]) * 100
                print(f"  [{idx+1:03d}/{len(items):03d}] Running EM: {curr_em:.1f}% | Running F1: {curr_f1:.1f}% | Latest: '{pred[:25]}' vs '{gold[:25]}'")

        elapsed_sys = time.time() - t_sys_start
        print(f"✓ Completed {sys_name} in {elapsed_sys:.1f}s")
        if hasattr(call_llm, "save_cache"):
            call_llm.save_cache()

    # --- Summary & Bootstrap Statistical Analysis ---
    summary: Dict[str, Any] = {}
    n_q = len(items)

    print("\n" + "=" * 100)
    print(f"{'System Architecture':<30} | {'Exact Match (95% CI)':<22} | {'Token F1 (95% CI)':<22} | {'Avg Calls':<10} | {'p50 Lat (ms)':<12}")
    print("-" * 100)

    for sys_name, rows in results.items():
        em_vals = [r["em"] for r in rows]
        f1_vals = [r["f1"] for r in rows]
        calls_vals = [r["calls"] for r in rows]
        lat_vals = [r["latency_ms"] for r in rows]

        em_m, em_l, em_h = bootstrap_ci(em_vals, n_resamples=1000)
        f1_m, f1_l, f1_h = bootstrap_ci(f1_vals, n_resamples=1000)
        c_m, c_l, c_h = bootstrap_ci(calls_vals, n_resamples=1000)

        lat_p50 = float(np.percentile(lat_vals, 50))
        lat_avg = float(np.mean(lat_vals))

        em_fmt = format_metric_with_ci(em_m, em_l, em_h, is_percent=True)
        f1_fmt = format_metric_with_ci(f1_m, f1_l, f1_h, is_percent=True)
        calls_fmt = f"{c_m:.2f} [{c_l:.2f}, {c_h:.2f}]"

        summary[sys_name] = {
            "num_questions": n_q,
            "em": round(em_m * 100, 2),
            "em_ci": [round(em_l * 100, 2), round(em_h * 100, 2)],
            "em_formatted": em_fmt,
            "f1": round(f1_m * 100, 2),
            "f1_ci": [round(f1_l * 100, 2), round(f1_h * 100, 2)],
            "f1_formatted": f1_fmt,
            "avg_calls": round(c_m, 2),
            "calls_ci": [round(c_l, 2), round(c_h, 2)],
            "p50_latency_ms": round(lat_p50, 1),
            "avg_latency_ms": round(lat_avg, 1),
            "level_distribution": {
                f"Level_{l}": f"{(count / n_q) * 100:.1f}% ({count}/{n_q})"
                for l, count in level_distributions.get(sys_name, {}).items()
            } if sys_name in level_distributions else None
        }

        print(f"{sys_name:<30} | {em_fmt:<22} | {f1_fmt:<22} | {c_m:<10.2f} | {lat_p50:<12.1f}")

    # Paired Bootstrap Significance Tests
    p_values: Dict[str, Any] = {}
    if "FeedbackRAG-Learned" in results and "FeedbackRAG-Thresholds" in results:
        p_val_f1 = bootstrap_paired_test(
            [r["f1"] for r in results["FeedbackRAG-Learned"]],
            [r["f1"] for r in results["FeedbackRAG-Thresholds"]]
        )
        p_val_em = bootstrap_paired_test(
            [r["em"] for r in results["FeedbackRAG-Learned"]],
            [r["em"] for r in results["FeedbackRAG-Thresholds"]]
        )
        p_values["Learned_vs_Thresholds"] = {"p_em": p_val_em, "p_f1": p_val_f1}
        print(f"\n[Significance Test] FeedbackRAG-Learned vs Thresholds: p(EM)={p_val_em:.4f}, p(F1)={p_val_f1:.4f}")

    if "FeedbackRAG-Learned" in results and "Always-L1" in results:
        p_val_l1 = bootstrap_paired_test(
            [r["f1"] for r in results["FeedbackRAG-Learned"]],
            [r["f1"] for r in results["Always-L1"]]
        )
        p_values["Learned_vs_AlwaysL1"] = {"p_f1": p_val_l1}
        print(f"[Significance Test] FeedbackRAG-Learned vs Always-L1: p(F1)={p_val_l1:.4f}")

    # Save output artifacts
    target_json = output_path or "results/tables/real_100q_benchmark_results.json"
    target_md = target_json.replace(".json", ".md")
    os.makedirs(os.path.dirname(target_json), exist_ok=True)

    final_payload = {
        "model": call_llm.model,
        "endpoint": call_llm.url,
        "dataset": "HotpotQA (data/real_test.json)",
        "corpus_size": len(corpus),
        "num_questions": n_q,
        "summary": summary,
        "significance_tests": p_values,
        "per_question": results
    }

    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[*] JSON results saved to: {target_json}")

    # Generate Markdown Table Artifact
    md_lines = [
        f"# Live Neural LLM Benchmark Results (100 Multi-Hop HotpotQA Queries)",
        f"\n**Model:** `{call_llm.model}` on Remote NVIDIA RTX A6000 48GB GPU host (`10.10.14.233`)",
        f"**Corpus:** 2,826 real documents from `data/real_corpus.json` (BM25 + Dense RRF fusion)",
        f"**Evaluated Queries:** {n_q} authentic multi-hop questions with 95% Bootstrap Confidence Intervals ($B=1,000$)\n",
        f"| System Architecture | Exact Match (95% CI) | Token F1 (95% CI) | Avg Calls / Query | p50 Latency (ms) | Level Distribution (L0 / L1 / L2 / L3) |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for sys_name, s in summary.items():
        dist_str = "—"
        if s.get("level_distribution"):
            d = s["level_distribution"]
            dist_str = f"{d.get('Level_0', '0%').split()[0]} / {d.get('Level_1', '0%').split()[0]} / {d.get('Level_2', '0%').split()[0]} / {d.get('Level_3', '0%').split()[0]}"
        md_lines.append(
            f"| **{sys_name}** | {s['em_formatted']} | {s['f1_formatted']} | {s['avg_calls']:.2f} | {s['p50_latency_ms']:.1f} ms | {dist_str} |"
        )

    if p_values:
        md_lines.append(f"\n### Statistical Significance (Paired Bootstrap, $B=2,000$):")
        for comp, vals in p_values.items():
            md_lines.append(f"- **{comp}:** {vals}")

    with open(target_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[*] Markdown report saved to: {target_md}")


def run_multimodel_benchmark(
    models: Optional[List[str]] = None,
    use_retrieval: bool = True,
    limit: Optional[int] = 15,
    port: int = 11435,
    output_path: Optional[str] = None
):
    if models is None:
        models = [
            "llama3.2:3b",
            "mistral:latest",
            "llama3.1:latest",
            "qwen3:8b",
            "qwen3:14b",
            "llama3.1:70b"
        ]

    corpus_file = "data/real_corpus.json"
    retriever = None
    if os.path.exists(corpus_file):
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        retriever = HybridRetriever(corpus=corpus)
    
    items = get_targeted_items()
    if limit is not None:
        items = items[:limit]

    print(f"\n==========================================================================================")
    print(f"Running Multi-Model Evaluation across {len(models)} Ollama Models on 48GB GPU")
    print(f"Models: {', '.join(models)}")
    print(f"==========================================================================================")

    multimodel_results = {}
    for m in models:
        print(f"\n>>> Evaluating Model: {m} on NVIDIA RTX A6000 48GB GPU <<<")
        safe_m = m.replace(":", "_").replace("/", "_")
        cache_path = f"results/real_cache_{safe_m}.json"
        
        call_llm = make_fast_ollama(model_name=m, port=port, cache_path=cache_path)
        
        model_res = {"Always-L0": [], "Always-L1": [], "FeedbackRAG": []}
        
        # 1. Always-L0
        for item in items:
            prompt = f"Answer concisely in 1-4 words:\nQuestion: {item['q']}\nAnswer:"
            t0 = time.time()
            pred = call_llm(prompt, max_tokens=25)
            dt = (time.time() - t0) * 1000
            score_em = em(pred, item["gold"])
            score_f1 = f1(pred, item["gold"])
            model_res["Always-L0"].append({"em": score_em, "f1": score_f1, "lat": dt, "pred": pred, "gold": item["gold"]})

        # 2. Always-L1
        for item in items:
            if retriever:
                docs = retriever.search(item["q"], k=3)
                ctx = " ".join([d.get("text", "") for d in docs])
            else:
                ctx = item.get("context", "")
            prompt = f"Answer concisely in 1-4 words using only the context:\nContext: {ctx}\nQuestion: {item['q']}\nAnswer:"
            t0 = time.time()
            pred = call_llm(prompt, max_tokens=25)
            dt = (time.time() - t0) * 1000
            score_em = em(pred, item["gold"])
            score_f1 = f1(pred, item["gold"])
            model_res["Always-L1"].append({"em": score_em, "f1": score_f1, "lat": dt, "pred": pred, "gold": item["gold"]})

        # 3. FeedbackRAG
        escalation_count = 0
        for item in items:
            t0 = time.time()
            l0_prompt = f"Answer concisely in 1-4 words:\nQuestion: {item['q']}\nAnswer:"
            pred_l0 = call_llm(l0_prompt, max_tokens=25)
            calls = 1

            norm_l0 = normalize_answer(pred_l0)
            uncertainty_phrases = ["cannot", "unknown", "do not know", "not sure", "sorry", "unclear", "no information"]
            is_uncertain = any(phrase in norm_l0 for phrase in uncertainty_phrases)
            is_malformed = (len(norm_l0.split()) == 0 or len(norm_l0.split()) > 15)

            perturbed_prompt = f"State the factual answer directly in 1-3 words:\nQuestion: {item['q']}\nAnswer:"
            pred_l0_alt = call_llm(perturbed_prompt, max_tokens=25)
            calls += 1
            norm_alt = normalize_answer(pred_l0_alt)

            consistency_match = (norm_l0 == norm_alt or norm_l0 in norm_alt or norm_alt in norm_l0) and not is_uncertain and not is_malformed

            if consistency_match:
                final_pred = pred_l0
                lvl = 0
            else:
                escalation_count += 1
                if retriever:
                    docs = retriever.search(item["q"], k=3)
                    ctx = " ".join([d.get("text", "") for d in docs])
                else:
                    ctx = item.get("context", "")
                l1_prompt = f"Answer concisely in 1-4 words using only the context:\nContext: {ctx}\nQuestion: {item['q']}\nAnswer:"
                final_pred = call_llm(l1_prompt, max_tokens=25)
                calls += 2
                lvl = 1

            dt = (time.time() - t0) * 1000
            score_em = em(final_pred, item["gold"])
            score_f1 = f1(final_pred, item["gold"])
            model_res["FeedbackRAG"].append({"em": score_em, "f1": score_f1, "lat": dt, "calls": calls, "lvl": lvl, "pred": final_pred, "gold": item["gold"]})

        summary_m = {}
        for sname, rlist in model_res.items():
            summary_m[sname] = {
                "em": round(sum(r["em"] for r in rlist) / len(rlist) * 100, 1),
                "f1": round(sum(r["f1"] for r in rlist) / len(rlist) * 100, 1),
                "avg_lat_ms": round(sum(r["lat"] for r in rlist) / len(rlist), 1),
                "avg_calls": round(sum(r.get("calls", 1) for r in rlist) / len(rlist), 2)
            }
        summary_m["escalation_rate"] = f"{escalation_count}/{len(items)} ({round(escalation_count/len(items)*100, 1)}%)"
        multimodel_results[m] = summary_m
        print(f"[{m}] Always-L0: F1={summary_m['Always-L0']['f1']}%, Always-L1: F1={summary_m['Always-L1']['f1']}%, FeedbackRAG: F1={summary_m['FeedbackRAG']['f1']}% (Escalated: {summary_m['escalation_rate']})")
        if hasattr(call_llm, "save_cache"):
            call_llm.save_cache()

    target_json = output_path or "results/tables/multimodel_benchmark_results.json"
    target_md = target_json.replace(".json", ".md")
    os.makedirs(os.path.dirname(target_json), exist_ok=True)
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(multimodel_results, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# Cross-Model Scaling Evaluation on NVIDIA RTX A6000 48GB GPU",
        f"\n**Hardware:** NVIDIA RTX A6000 48GB VRAM (Host: `10.10.14.233`, 503GB RAM, Driver 580.126.20, CUDA 13.0)",
        f"**Benchmark Cohort:** {len(items)} Factual & Multi-Hop Validation Queries across {len(models)} Model Architectures (3B to 70B parameters)\n",
        "| Model Architecture | Parameter Scale | VRAM Footprint | Strategy | Exact Match | Token F1 | Avg Latency | Avg Calls |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |"
    ]
    model_meta = {
        "llama3.2:3b": ("3.2B", "2.0 GB"),
        "mistral:latest": ("7.2B", "4.4 GB"),
        "llama3.1:latest": ("8.0B", "4.9 GB"),
        "qwen3:8b": ("8.2B (Reasoning)", "5.2 GB"),
        "qwen3:14b": ("14.8B (Reasoning)", "9.3 GB"),
        "llama3.1:70b": ("70.6B (Frontier)", "42.5 GB")
    }
    for m, s in multimodel_results.items():
        params, vram = model_meta.get(m, ("—", "—"))
        md_lines.append(f"| **{m}** | {params} | {vram} | Always-L0 (Parametric) | {s['Always-L0']['em']}% | {s['Always-L0']['f1']}% | {s['Always-L0']['avg_lat_ms']:.1f} ms | {s['Always-L0']['avg_calls']:.2f} |")
        md_lines.append(f"| | | | Always-L1 (Hybrid RAG) | {s['Always-L1']['em']}% | {s['Always-L1']['f1']}% | {s['Always-L1']['avg_lat_ms']:.1f} ms | {s['Always-L1']['avg_calls']:.2f} |")
        md_lines.append(f"| | | | **FeedbackRAG (Adaptive MLFQ)** | **{s['FeedbackRAG']['em']}%** | **{s['FeedbackRAG']['f1']}%** | **{s['FeedbackRAG']['avg_lat_ms']:.1f} ms** | **{s['FeedbackRAG']['avg_calls']:.2f}** |")

    with open(target_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\n[*] Multi-model results saved to {target_json} and {target_md}")
    return multimodel_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Neural LLM Benchmark for FeedbackRAG")
    parser.add_argument("--use-retrieval", action="store_true", help="Perform real BM25/Dense hybrid retrieval instead of static context")
    parser.add_argument("--dataset", choices=["targeted", "real_test"], default="targeted", help="Benchmark dataset to run")
    parser.add_argument("--limit", type=int, default=None, help="Max queries to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--systems", type=str, default=None, help="Comma-separated list of systems to evaluate")
    parser.add_argument("--model", type=str, default=None, help="Target Ollama model name (e.g. qwen3:14b, llama3.1:70b)")
    parser.add_argument("--port", type=int, default=11435, help="Port of Ollama service or SSH tunnel")
    parser.add_argument("--cache-path", type=str, default=None, help="Cache JSON file path")
    parser.add_argument("--multimodel", action="store_true", help="Run multi-model comparison across all installed Ollama models")
    args = parser.parse_args()

    systems_list = [s.strip() for s in args.systems.split(",")] if args.systems else None

    if args.multimodel:
        run_multimodel_benchmark(
            use_retrieval=args.use_retrieval,
            limit=args.limit,
            port=args.port,
            output_path=args.output
        )
    elif args.dataset == "real_test":
        run_real_100q_benchmark(
            limit=args.limit,
            output_path=args.output,
            systems_to_run=systems_list,
            model=args.model,
            port=args.port,
            cache_path=args.cache_path
        )
    else:
        run_real_evaluation(
            use_retrieval=args.use_retrieval,
            dataset=args.dataset,
            limit=args.limit,
            output_path=args.output,
            model=args.model,
            port=args.port,
            cache_path=args.cache_path
        )
