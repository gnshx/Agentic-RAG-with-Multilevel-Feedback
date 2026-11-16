import json
import time
import os
import sys
import yaml
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm import get_llm_instance
from src.retriever import HybridRetriever
from src.planner import QueryPlanner
from src.generator import Generator
from src.levels import run_level_0, run_level_1, run_level_2, run_level_3
from src.baselines.crag import CRAGBaseline
from src.baselines.self_rag import SelfRAGBaseline
from src.cost_tracker import CostTracker
from eval.metrics import em, f1
from eval.bootstrap import bootstrap_ci, format_metric_with_ci

def load_data(split_name: str = "test"):
    path = f"data/{split_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open("data/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return data, corpus, cfg

def evaluate_system(system_name: str, questions: List[Dict[str, Any]], corpus: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_llm_instance(cfg)
    retriever = HybridRetriever(corpus=corpus)
    planner = QueryPlanner(llm=llm)
    generator = Generator(llm=llm)
    cost_tracker = CostTracker()

    crag = CRAGBaseline(llm=llm, retriever=retriever, generator=generator)
    self_rag = SelfRAGBaseline(llm=llm, retriever=retriever, generator=generator)

    em_list: List[float] = []
    f1_list: List[float] = []
    calls_list: List[float] = []
    latency_list: List[float] = []
    cost_list: List[float] = []

    predictions = []
    t_start = time.time()

    for item in questions:
        q = item["question"]
        gold_ans = item["answer"]
        item_t0 = time.time()
        start_calls = llm.logical_calls
        start_cost = getattr(llm, "total_cost_usd", 0.0)

        pred_ans = ""
        citations = []

        if system_name == "Always-L0":
            s = {"question": q, "level": 0, "signals": {}, "drafts": [], "answer": "", "citations": []}
            s = run_level_0(s, llm, cfg)
            pred_ans = s["answer"]
        elif system_name == "Always-L1":
            s = {"question": q, "level": 1, "signals": {}, "docs": [], "answer": "", "citations": []}
            s = run_level_1(s, llm, retriever, generator, cfg)
            pred_ans = s["answer"]
            citations = s["citations"]
        elif system_name == "Always-L2":
            s = {"question": q, "level": 2, "signals": {}, "docs": [], "subquestions": [], "answer": "", "citations": []}
            s = run_level_2(s, llm, retriever, planner, generator, cfg)
            pred_ans = s["answer"]
            citations = s["citations"]
        elif system_name == "Always-L3":
            s = {"question": q, "level": 3, "signals": {}, "docs": [], "drafts": [], "answer": "", "citations": []}
            s = run_level_3(s, llm, retriever, generator, cfg)
            pred_ans = s["answer"]
            citations = s["citations"]
        elif system_name == "CRAG-Reimpl":
            res = crag.run(q)
            pred_ans = res["answer"]
            citations = res["citations"]
        elif system_name == "Self-RAG-Reimpl":
            res = self_rag.run(q)
            pred_ans = res["answer"]
            citations = res["citations"]
        else:
            raise ValueError(f"Unknown system name: {system_name}")

        latency = time.time() - item_t0
        calls_used = float(llm.logical_calls - start_calls)
        item_cost = float(getattr(llm, "total_cost_usd", 0.0) - start_cost)

        score_em = em(pred_ans, gold_ans)
        score_f1 = f1(pred_ans, gold_ans)

        em_list.append(score_em)
        f1_list.append(score_f1)
        calls_list.append(calls_used)
        latency_list.append(latency)
        cost_list.append(item_cost)

        predictions.append({
            "id": item["id"],
            "dataset": item.get("dataset", "HotpotQA"),
            "question": q,
            "prediction": pred_ans,
            "ground_truth": gold_ans,
            "em": score_em,
            "f1": score_f1,
            "calls": calls_used,
            "latency": latency,
            "citations": citations
        })

    total_time = time.time() - t_start
    llm.save()

    # 95% Bootstrap Confidence Intervals
    em_mean, em_low, em_high = bootstrap_ci(em_list, n_resamples=1000)
    f1_mean, f1_low, f1_high = bootstrap_ci(f1_list, n_resamples=1000)
    calls_mean, calls_low, calls_high = bootstrap_ci(calls_list, n_resamples=1000)
    cost_mean, cost_low, cost_high = bootstrap_ci(cost_list, n_resamples=1000)
    lat_stats = cost_tracker.summarize_latencies(latency_list)

    return {
        "system": system_name,
        "num_questions": len(questions),
        "em": round(em_mean * 100, 2),
        "em_ci": [round(em_low * 100, 2), round(em_high * 100, 2)],
        "em_formatted": format_metric_with_ci(em_mean, em_low, em_high, is_percent=True),
        "f1": round(f1_mean * 100, 2),
        "f1_ci": [round(f1_low * 100, 2), round(f1_high * 100, 2)],
        "f1_formatted": format_metric_with_ci(f1_mean, f1_low, f1_high, is_percent=True),
        "avg_calls": round(calls_mean, 2),
        "calls_ci": [round(calls_low, 2), round(calls_high, 2)],
        "avg_cost_usd": round(cost_mean, 6),
        "total_cost_usd": round(sum(cost_list), 5),
        "latency_stats": lat_stats,
        "total_time_s": round(total_time, 2),
        "raw_scores": {"em": em_list, "f1": f1_list, "calls": calls_list}
    }

def main():
    os.makedirs("results/tables", exist_ok=True)
    test_data, corpus, cfg = load_data("test")

    systems_to_evaluate = [
        "Always-L0",
        "Always-L1",
        "Always-L2",
        "Always-L3",
        "CRAG-Reimpl",
        "Self-RAG-Reimpl"
    ]

    print("==========================================================================================")
    print(f"Running Full Baseline Evaluation with 95% Bootstrap CIs ({len(test_data)} Test Questions)")
    print("==========================================================================================")
    print(f"{'System':<20} | {'Exact Match (95% CI)':<22} | {'F1 Score (95% CI)':<22} | {'Avg Calls':<10} | {'p50 Lat (s)':<10}")
    print("-" * 96)

    all_results = {}
    for sys_name in systems_to_evaluate:
        res = evaluate_system(sys_name, test_data, corpus, cfg)
        all_results[sys_name] = res
        print(f"{sys_name:<20} | {res['em_formatted']:<22} | {res['f1_formatted']:<22} | {res['avg_calls']:<10.2f} | {res['latency_stats']['p50']:<10.4f}")

    with open("results/tables/baselines_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n✓ Baseline evaluation with 95% CIs saved to results/tables/baselines_results.json")

if __name__ == "__main__":
    main()
