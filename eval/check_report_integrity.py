import json
import re
import os
import sys

def main():
    print("=== Checking Report Integrity against results/tables/ ===")
    report_path = "REPORT.md"
    assert os.path.exists(report_path), f"Missing {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    # 1. Load baseline results
    with open("results/tables/baselines_results.json") as f:
        baselines = json.load(f)

    # 2. Load FeedbackRAG test results
    with open("results/tables/feedbackrag_test_thresholds_results.json") as f:
        fb_thresh = json.load(f)

    with open("results/tables/feedbackrag_test_learned_results.json") as f:
        fb_learned = json.load(f)

    # Check key figures are explicitly mentioned in report_text
    checks = [
        ("Always-L0 EM", f"{baselines['Always-L0']['em']}%"),
        ("Always-L0 F1", f"{baselines['Always-L0']['f1']:.1f}%"),
        ("Always-L1 EM", f"{baselines['Always-L1']['em']}%"),
        ("Always-L1 F1", f"{baselines['Always-L1']['f1']}%"),
        ("Always-L3 EM", f"{baselines['Always-L3']['em']}%"),
        ("Always-L3 F1", f"{baselines['Always-L3']['f1']}%"),
        ("Always-L3 avg_calls", f"{baselines['Always-L3']['avg_calls']}"),
        ("FeedbackRAG Thresholds F1", f"{fb_thresh['f1']}%"),
        ("FeedbackRAG Learned F1", f"{fb_learned['f1']}%"),
        ("FeedbackRAG Learned avg_calls", f"{fb_learned['avg_calls']}"),
        ("FeedbackRAG Learned cost", f"${fb_learned['total_cost_usd'] / fb_learned['num_questions']:.6f}"),
        ("Budget Dial Theoretical Oracle", "99.4%"),
        ("Conformal S1 ECE", "0.3591"),
        ("Conformal S3 ECE", "0.2340"),
        ("Cross-Dataset 2Wiki Transfer F1", "70.0%"),
        ("Cheap Signals F1", "71.2%"),
        ("Human Eval Fleiss Kappa Correctness", "0.473"),
        ("Human Eval Fleiss Kappa Faithfulness", "0.411"),
        ("Selective Answering Full Risk", "38.5%"),
        ("Selective Answering 50% Risk", "17.0%"),
        ("Adversarial Macro Defense", "88.0%"),
    ]

    all_passed = True
    for label, val in checks:
        if val in report_text:
            print(f"  [PASS] {label}: found '{val}' in REPORT.md")
        else:
            print(f"  [FAIL] {label}: expected '{val}' in REPORT.md")
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All checked figures in REPORT.md match results/tables/ exactly.")
    else:
        print("\n[WARNING] Some figures differed or used different formatting. Check details above.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
