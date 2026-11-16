import json
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List
from eval.sweep_thresholds import load_signals_records
from eval.metrics import em, f1

def run_ablation_simulation(
    records: Dict[str, Dict[int, Dict[str, Any]]],
    dataset: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    drop_signal: str = None
) -> Dict[str, Any]:
    total_calls = 0
    total_em = 0.0
    total_f1 = 0.0
    n = len(dataset)
    gold_map = {item["id"]: item["answer"] for item in dataset}

    L = cfg.get("levels", {})

    for item in dataset:
        qid = item["id"]
        if qid not in records:
            continue
        by_level = records[qid]
        gold_ans = gold_map[qid]

        final_ans = ""
        used_calls = 0

        for lvl in range(4):
            lvl_data = by_level[lvl]
            used_calls += lvl_data.get("calls", 0)
            final_ans = lvl_data.get("answer", "")
            s = dict(lvl_data.get("signals", {}))

            # Simulate dropping signal
            passed = False
            if lvl == 0:
                if drop_signal == "S1":
                    # Dropping S1 means never passing L0 -> always escalate
                    passed = False
                else:
                    passed = s.get("S1", 0.0) >= L["L0"]["tau"]
            elif lvl == 1:
                s2_pass = True if drop_signal == "S2" else (s.get("S2", 0.0) >= L["L1"]["tau_retrieval"])
                s3_pass = True if drop_signal == "S3" else (s.get("S3", 0.0) >= L["L1"]["tau_ground"])
                passed = s2_pass and s3_pass
            elif lvl == 2:
                s3_pass = True if drop_signal == "S3" else (s.get("S3", 0.0) >= L["L2"]["tau_ground"])
                s4_pass = True if drop_signal == "S4" else (s.get("S4", 0.0) >= L["L2"]["tau_complete"])
                passed = s3_pass and s4_pass
            elif lvl == 3:
                passed = True

            if passed or lvl == 3:
                break

        total_calls += used_calls
        total_em += em(final_ans, gold_ans)
        total_f1 += f1(final_ans, gold_ans)

    return {
        "dropped_signal": drop_signal or "None (Full FeedbackRAG)",
        "em": round((total_em / max(n, 1)) * 100, 2),
        "f1": round((total_f1 / max(n, 1)) * 100, 2),
        "avg_calls": round(total_calls / max(n, 1), 2)
    }

def main():
    os.makedirs("results/tables", exist_ok=True)
    with open("data/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    records = load_signals_records("results/signals.jsonl")
    if not records:
        print("[Ablation] Error: results/signals.jsonl not found.")
        return

    print("==================================================================")
    print("Running Feedback Signal Ablation Study on Test Set (100 Questions)")
    print("==================================================================")

    # 1. Full FeedbackRAG baseline
    full = run_ablation_simulation(records, test_data, cfg, drop_signal=None)
    base_f1 = full["f1"]
    base_calls = full["avg_calls"]

    ablations = [full]
    signals_to_drop = [
        ("S1", "Drop S1 (Self-Consistency)"),
        ("S2", "Drop S2 (Retrieval Quality Grader)"),
        ("S3", "Drop S3 (Groundedness Checker)"),
        ("S4", "Drop S4 (Completeness Checker)")
    ]

    print(f"{'Ablation Variant':<36} | {'F1 (%)':<8} | {'Δ F1':<8} | {'Avg Calls':<10} | {'Δ Calls':<8}")
    print("-" * 80)
    print(f"{'Full FeedbackRAG (All S1-S4)':<36} | {full['f1']:<8} | {'0.00':<8} | {full['avg_calls']:<10} | {'0.00':<8}")

    for sig, desc in signals_to_drop:
        res = run_ablation_simulation(records, test_data, cfg, drop_signal=sig)
        delta_f1 = round(res["f1"] - base_f1, 2)
        delta_calls = round(res["avg_calls"] - base_calls, 2)
        res["delta_f1"] = delta_f1
        res["delta_calls"] = delta_calls
        ablations.append(res)
        print(f"{desc:<36} | {res['f1']:<8} | {delta_f1:<+8.2f} | {res['avg_calls']:<10} | {delta_calls:<+8.2f}")

    with open("results/tables/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablations, f, indent=2)

    print("\n✓ Ablation table saved to results/tables/ablation_results.json")

if __name__ == "__main__":
    main()
