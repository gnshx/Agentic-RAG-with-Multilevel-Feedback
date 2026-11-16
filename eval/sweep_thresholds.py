import json
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from itertools import product
from typing import Dict, Any, List, Tuple
from src.controller import passes
from eval.metrics import em, f1

def load_signals_records(signals_path: str = "results/signals.jsonl") -> Dict[str, Dict[int, Dict[str, Any]]]:
    records: Dict[str, Dict[int, Dict[str, Any]]] = {}
    if not os.path.exists(signals_path):
        return records
    with open(signals_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = entry["id"]
            # Convert string keys to int levels
            by_level = {int(k): v for k, v in entry["by_level"].items()}
            records[qid] = by_level
    return records

def replay_thresholds(
    records: Dict[str, Dict[int, Dict[str, Any]]],
    dataset: List[Dict[str, Any]],
    cfg: Dict[str, Any]
) -> Tuple[float, float, float]:
    total_calls = 0
    total_em = 0.0
    total_f1 = 0.0
    n = len(dataset)

    gold_map = {item["id"]: item["answer"] for item in dataset}

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

            # Check if passes at this level
            if lvl == 3 or passes(lvl, lvl_data.get("signals", {}), cfg):
                break

        total_calls += used_calls
        total_em += em(final_ans, gold_ans)
        total_f1 += f1(final_ans, gold_ans)

    avg_calls = round(total_calls / max(n, 1), 2)
    avg_em = round((total_em / max(n, 1)) * 100, 2)
    avg_f1 = round((total_f1 / max(n, 1)) * 100, 2)
    return avg_em, avg_f1, avg_calls

def run_threshold_sweep():
    os.makedirs("results/tables", exist_ok=True)
    with open("data/dev.json", "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    with open("data/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open("config.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    records = load_signals_records("results/signals.jsonl")
    if not records:
        print("[Sweep] Error: results/signals.jsonl not found. Run eval/run_feedbackrag.py first.")
        return

    print("==================================================================")
    print("Running Offline Multi-Threshold Sweep on Dev Set (Zero LLM calls)")
    print("==================================================================")

    tau0_vals = [0.50, 0.67, 0.85]
    tau1_ret_vals = [0.40, 0.50, 0.70]
    tau1_gnd_vals = [0.60, 0.80, 0.90]
    tau2_gnd_vals = [0.70, 0.80, 0.90]
    tau2_comp_vals = [0.70, 0.80, 0.90]

    sweep_points = []
    best_config = None
    best_efficiency_ratio = -1.0  # F1 / avg_calls

    for t0, t1_r, t1_g, t2_g, t2_c in product(tau0_vals, tau1_ret_vals, tau1_gnd_vals, tau2_gnd_vals, tau2_comp_vals):
        cand_cfg = {
            "budget": {"max_calls_per_query": 20},
            "levels": {
                "L0": {"tau": t0},
                "L1": {"tau_retrieval": t1_r, "tau_ground": t1_g},
                "L2": {"tau_ground": t2_g, "tau_complete": t2_c},
                "L3": {"drafts": 3, "max_revisions": 2}
            }
        }
        dev_em, dev_f1, dev_calls = replay_thresholds(records, dev_data, cand_cfg)

        ratio = dev_f1 / max(dev_calls, 1.0)
        point = {
            "thresholds": {
                "L0_tau": t0, "L1_tau_retrieval": t1_r, "L1_tau_ground": t1_g,
                "L2_tau_ground": t2_g, "L2_tau_complete": t2_c
            },
            "dev_em": dev_em,
            "dev_f1": dev_f1,
            "dev_calls": dev_calls,
            "f1_per_call": round(ratio, 3)
        }
        sweep_points.append(point)

        if ratio > best_efficiency_ratio:
            best_efficiency_ratio = ratio
            best_config = cand_cfg

    # Evaluate best config on Test set
    test_em, test_f1, test_calls = replay_thresholds(records, test_data, best_config)

    print(f"\n✓ Evaluated {len(sweep_points)} threshold combinations on Dev split.")
    print(f"Optimal Thresholds chosen on Dev: {best_config['levels']}")
    print(f"Test Set Performance at Tuned Thresholds -> EM: {test_em}% | F1: {test_f1}% | Avg Calls: {test_calls}")

    output_summary = {
        "best_config": best_config,
        "test_performance": {
            "em": test_em,
            "f1": test_f1,
            "avg_calls": test_calls
        },
        "all_sweep_points": sweep_points[:50]  # Store top sampled points
    }

    with open("results/tables/pareto_sweep_points.json", "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

if __name__ == "__main__":
    run_threshold_sweep()
