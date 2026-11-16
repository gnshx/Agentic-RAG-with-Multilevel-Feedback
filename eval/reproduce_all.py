import os
import sys
import json
import subprocess
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_command_logged(cmd: str, desc: str):
    print(f"\n=======================================================")
    print(f"[*] Running: {desc}")
    print(f"[*] Command: {cmd}")
    print(f"=======================================================")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, check=True)
    dt = time.time() - t0
    print(f"[+] Completed {desc} in {dt:.2f}s")

def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def main():
    print("=== FeedbackRAG Complete Pipeline Reproduction ===")
    start_total = time.time()

    # 1. Run baselines
    run_command_logged(
        f"{sys.executable} eval/run_baselines.py",
        "Baselines Evaluation (Always-L0, L1, L2, L3, CRAG, Self-RAG)"
    )

    # 2. Run FeedbackRAG
    run_command_logged(
        f"{sys.executable} eval/run_feedbackrag.py",
        "FeedbackRAG Pipeline (Dev training, Thresholds & Learned Policy on Test)"
    )

    # 3. Sweep thresholds
    run_command_logged(
        f"{sys.executable} eval/sweep_thresholds.py",
        "Dev Threshold Parameter Sweep"
    )

    # 4. Signal Ablation
    run_command_logged(
        f"{sys.executable} eval/ablation.py",
        "Feedback Signal Ablation Study"
    )

    # 5. Diagnostic Evaluations
    run_command_logged(
        f"{sys.executable} eval/judge_validation.py",
        "Judge Validation / Consistency Probes"
    )
    run_command_logged(
        f"{sys.executable} eval/abstention_test.py",
        "Abstention Evaluation on Unanswerable Queries"
    )
    run_command_logged(
        f"{sys.executable} eval/adversarial_test.py",
        "Adversarial Distractor Probing"
    )
    run_command_logged(
        f"{sys.executable} eval/error_analysis.py",
        "Failure Analysis Breakdown"
    )

    # 6. Real Local LLM Validation (if Ollama is available)
    try:
        run_command_logged(
            f"{sys.executable} eval/run_real_llm_benchmark.py",
            "Live Local LLM Benchmark (Qwen 2.5 0.5B via Ollama)"
        )
    except Exception as e:
        print(f"[!] Warning: Live Ollama evaluation skipped or failed: {e}")

    # 7. Generate publication plots
    run_command_logged(
        f"{sys.executable} eval/plots.py",
        "Pareto Frontier Plot Generation"
    )

    # 8. Save Run Metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit_hash(),
        "random_seed": 42,
        "pricing_reference": {
            "source": "OpenAI / Gemini lightweight standard pricing (Nov 2024)",
            "prompt_tokens_per_1m": 0.150,
            "completion_tokens_per_1m": 0.600
        },
        "datasets": {
            "dev_items": 100,
            "test_items": 500,
            "corpus_paragraphs": 48,
            "provenance": "Synthetic multi-hop benchmark generated from 20 seed templates (data/README.md)"
        },
        "execution_backends": {
            "regime_a": "Deterministic reference simulator (_mock_backend in src/llm.py)",
            "regime_b": "Qwen 2.5 0.5B via local Ollama API (eval/run_real_llm_benchmark.py)"
        }
    }

    os.makedirs("results", exist_ok=True)
    with open("results/run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_total
    print(f"\n[SUCCESS] Full reproduction completed in {total_time:.2f}s. Run metadata saved to results/run_metadata.json.")

if __name__ == "__main__":
    main()

