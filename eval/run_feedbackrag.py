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
from src.controller import run_controller
from src.learned_policy import LearnedEscalationPolicy
from src.cost_tracker import CostTracker
from src.trace import TraceLogger
from eval.metrics import em, f1
from eval.bootstrap import bootstrap_ci, bootstrap_paired_test, format_metric_with_ci

def run_feedbackrag_eval(split_name: str = "test", mode: str = "thresholds", policy: LearnedEscalationPolicy = None) -> Dict[str, Any]:
    with open(f"data/{split_name}.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open("data/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    llm = get_llm_instance(cfg)
    retriever = HybridRetriever(corpus=corpus)
    planner = QueryPlanner(llm=llm)
    generator = Generator(llm=llm)
    trace_logger = TraceLogger()
    cost_tracker = CostTracker()

    levels_map = {
        0: lambda s: run_level_0(s, llm, cfg),
        1: lambda s: run_level_1(s, llm, retriever, generator, cfg),
        2: lambda s: run_level_2(s, llm, retriever, planner, generator, cfg),
        3: lambda s: run_level_3(s, llm, retriever, generator, cfg)
    }

    em_list: List[float] = []
    f1_list: List[float] = []
    calls_list: List[float] = []
    latency_list: List[float] = []
    cost_list: List[float] = []
    level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    t0 = time.time()
    records = []

    for item in questions:
        q = item["question"]
        gold_ans = item["answer"]

        # Run Adaptive Escalation Controller
        state = run_controller(q, levels_map, cfg, llm, policy=policy, mode=mode)

        final_lvl = state["level"]
        level_counts[final_lvl] = level_counts.get(final_lvl, 0) + 1
        pred_ans = state["answer"]
        calls_used = float(state["calls"])
        cost_used = float(state.get("cost_usd", 0.0))
        lat = float(state.get("latency", 0.0))

        score_em = em(pred_ans, gold_ans)
        score_f1 = f1(pred_ans, gold_ans)

        em_list.append(score_em)
        f1_list.append(score_f1)
        calls_list.append(calls_used)
        latency_list.append(lat)
        cost_list.append(cost_used)

        record = {
            "id": item["id"],
            "dataset": item.get("dataset", "HotpotQA"),
            "question": q,
            "ground_truth": gold_ans,
            "prediction": pred_ans,
            "resolved_level": final_lvl,
            "em": score_em,
            "f1": score_f1,
            "calls": calls_used,
            "cost_usd": cost_used,
            "latency": lat,
            "status": state["status"],
            "trace": state["trace"]
        }
        records.append(record)
        trace_logger.log_query_trace(record)

        # Log by_level signals for policy training and offline threshold sweeping (Dev only)
        if split_name == "dev":
            by_level = {}
            for lvl in range(4):
                lvl_state = {
                    "question": q, "level": lvl, "docs": [], "subquestions": [],
                    "drafts": [], "answer": "", "citations": [], "signals": {}, "status": "ok"
                }
                c_before = getattr(llm, "logical_calls", llm.calls)
                lvl_state = levels_map[lvl](lvl_state)
                c_used = float(getattr(llm, "logical_calls", llm.calls) - c_before)
                by_level[lvl] = {
                    "question": q,
                    "answer": lvl_state.get("answer", ""),
                    "signals": dict(lvl_state.get("signals", {})),
                    "docs": lvl_state.get("docs", []),
                    "calls": c_used,
                    "em": em(lvl_state.get("answer", ""), gold_ans),
                    "f1": f1(lvl_state.get("answer", ""), gold_ans)
                }
            trace_logger.log_all_level_signals(item["id"], q, by_level)

    n = len(questions)
    elapsed = time.time() - t0
    llm.save()

    # 95% Bootstrap CIs
    em_m, em_l, em_h = bootstrap_ci(em_list, n_resamples=1000)
    f1_m, f1_l, f1_h = bootstrap_ci(f1_list, n_resamples=1000)
    c_m, c_l, c_h = bootstrap_ci(calls_list, n_resamples=1000)
    cost_m, cost_l, cost_h = bootstrap_ci(cost_list, n_resamples=1000)
    lat_stats = cost_tracker.summarize_latencies(latency_list)

    summary = {
        "system": f"FeedbackRAG ({mode})",
        "split": split_name,
        "mode": mode,
        "num_questions": n,
        "em": round(em_m * 100, 2),
        "em_ci": [round(em_l * 100, 2), round(em_h * 100, 2)],
        "em_formatted": format_metric_with_ci(em_m, em_l, em_h, is_percent=True),
        "f1": round(f1_m * 100, 2),
        "f1_ci": [round(f1_l * 100, 2), round(f1_h * 100, 2)],
        "f1_formatted": format_metric_with_ci(f1_m, f1_l, f1_h, is_percent=True),
        "avg_calls": round(c_m, 2),
        "calls_ci": [round(c_l, 2), round(c_h, 2)],
        "avg_cost_usd": round(cost_m, 6),
        "total_cost_usd": round(sum(cost_list), 5),
        "latency_stats": lat_stats,
        "total_time_s": round(elapsed, 2),
        "level_distribution": {
            f"Level_{lvl}": f"{(count / n) * 100:.1f}% ({count}/{n})"
            for lvl, count in level_counts.items()
        },
        "raw_scores": {"em": em_list, "f1": f1_list, "calls": calls_list}
    }

    os.makedirs("results/tables", exist_ok=True)
    suffix = f"{split_name}_{mode}"
    with open(f"results/tables/feedbackrag_{suffix}_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

def main():
    print("==========================================================================================")
    print("Running FeedbackRAG Evaluation with Bootstrap CIs & Learned Policy Training")
    print("==========================================================================================")

    # Phase 1: Dev split logging
    print("\n[Phase 1] Processing Dev Split (100 questions) to extract signals...")
    dev_res = run_feedbackrag_eval(split_name="dev", mode="thresholds")
    print(f"Dev Performance -> EM: {dev_res['em_formatted']} | F1: {dev_res['f1_formatted']} | Avg Calls: {dev_res['avg_calls']}")

    # Phase 2: Train Learned Escalation Policy strictly on Dev signals
    from eval.sweep_thresholds import load_signals_records
    dev_signals = load_signals_records("results/signals.jsonl")
    policy = LearnedEscalationPolicy(model_type="gradient_boosting")
    train_report = policy.train_on_dev_signals(dev_signals, split_name="dev")
    policy.save("results/learned_policy.pkl")
    print(f"\n[Phase 2] Trained Learned Policy (Gradient Boosting) on Dev:")
    print(f"CV Accuracy: {train_report['cv_accuracy_mean']:.4f} ± {train_report['cv_accuracy_std']:.4f} (Samples: {train_report['training_samples']})")

    # Phase 3: Evaluate FeedbackRAG (Threshold Mode) on 500 Test Questions
    print("\n[Phase 3] Evaluating FeedbackRAG (Thresholds) on 500 Test Questions...")
    test_thresh = run_feedbackrag_eval(split_name="test", mode="thresholds")
    print(f"Thresholds -> EM: {test_thresh['em_formatted']} | F1: {test_thresh['f1_formatted']} | Avg Calls: {test_thresh['avg_calls']}")
    print(f"Level Distribution: {test_thresh['level_distribution']}")

    # Phase 4: Evaluate FeedbackRAG (Learned Policy Mode) on 500 Test Questions
    print("\n[Phase 4] Evaluating FeedbackRAG (Learned Policy) on 500 Test Questions...")
    test_learned = run_feedbackrag_eval(split_name="test", mode="learned", policy=policy)
    print(f"Learned Policy -> EM: {test_learned['em_formatted']} | F1: {test_learned['f1_formatted']} | Avg Calls: {test_learned['avg_calls']}")
    print(f"Level Distribution: {test_learned['level_distribution']}")

    # Phase 5: Paired Bootstrap Significance Test
    p_val_em = bootstrap_paired_test(test_learned["raw_scores"]["em"], test_thresh["raw_scores"]["em"])
    p_val_f1 = bootstrap_paired_test(test_learned["raw_scores"]["f1"], test_thresh["raw_scores"]["f1"])
    print(f"\n[Phase 5] Significance Test (Learned vs Thresholds): p(EM)={p_val_em:.4f}, p(F1)={p_val_f1:.4f}")

    print("\n✓ Full FeedbackRAG evaluation completed successfully.")

if __name__ == "__main__":
    main()
