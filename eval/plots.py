import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_pareto_plot():
    os.makedirs("results/plots", exist_ok=True)

    baselines_path = "results/tables/baselines_results.json"
    thresh_path = "results/tables/feedbackrag_test_thresholds_results.json"
    learned_path = "results/tables/feedbackrag_test_learned_results.json"
    sweep_path = "results/tables/pareto_sweep_points.json"

    if not os.path.exists(baselines_path):
        print("[Plot] Error: results/tables/baselines_results.json not found.")
        return

    with open(baselines_path, "r", encoding="utf-8") as f:
        baselines = json.load(f)

    thresh_data = None
    if os.path.exists(thresh_path):
        with open(thresh_path, "r", encoding="utf-8") as f:
            thresh_data = json.load(f)

    learned_data = None
    if os.path.exists(learned_path):
        with open(learned_path, "r", encoding="utf-8") as f:
            learned_data = json.load(f)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    # Baselines data: (Name, calls, f1, color, marker, x_ci, y_ci)
    systems = [
        ("Always-L0 (Direct LLM)", baselines["Always-L0"], "#e74c3c", "o"),
        ("Always-L1 (Vanilla Hybrid RAG)", baselines["Always-L1"], "#e67e22", "s"),
        ("Always-L2 (Iterative RAG)", baselines["Always-L2"], "#3498db", "^"),
        ("Always-L3 (Full Agentic RAG)", baselines["Always-L3"], "#9b59b6", "D"),
        ("CRAG (Yan et al. Reimpl)", baselines["CRAG-Reimpl"], "#16a085", "p"),
        ("Self-RAG (Asai et al. Reimpl)", baselines["Self-RAG-Reimpl"], "#d35400", "v")
    ]

    for label, b_info, color, marker in systems:
        x = b_info["avg_calls"]
        y = b_info["f1"]
        x_err = [[x - b_info["calls_ci"][0]], [b_info["calls_ci"][1] - x]]
        y_err = [[y - b_info["f1_ci"][0]], [b_info["f1_ci"][1] - y]]

        ax.errorbar(
            x, y, xerr=x_err, yerr=y_err,
            fmt=marker, color=color, ecolor=color, elinewidth=1.2, capsize=3,
            markersize=9, markeredgecolor="black", markeredgewidth=1.2,
            zorder=5, label=label
        )
        ax.annotate(
            f" {label}\n ({x:.1f} calls, {y:.1f}% F1)",
            (x, y),
            textcoords="offset points",
            xytext=(10, -5),
            fontsize=8.5,
            weight="semibold",
            color=color
        )

    # FeedbackRAG (Thresholds)
    if thresh_data:
        tx = thresh_data["avg_calls"]
        ty = thresh_data["f1"]
        tx_err = [[tx - thresh_data["calls_ci"][0]], [thresh_data["calls_ci"][1] - tx]]
        ty_err = [[ty - thresh_data["f1_ci"][0]], [thresh_data["f1_ci"][1] - ty]]
        ax.errorbar(
            tx, ty, xerr=tx_err, yerr=ty_err,
            fmt="*", color="#27ae60", ecolor="#27ae60", elinewidth=1.5, capsize=4,
            markersize=15, markeredgecolor="black", markeredgewidth=1.4,
            zorder=7, label="FeedbackRAG (Thresholds)"
        )
        ax.annotate(
            f" FeedbackRAG (Thresholds)\n ({tx:.1f} calls, {ty:.1f}% F1)",
            (tx, ty),
            textcoords="offset points",
            xytext=(12, -8),
            fontsize=9.5,
            weight="heavy",
            color="#1e824c"
        )

    # FeedbackRAG (Learned Policy)
    if learned_data:
        lx = learned_data["avg_calls"]
        ly = learned_data["f1"]
        lx_err = [[lx - learned_data["calls_ci"][0]], [learned_data["calls_ci"][1] - lx]]
        ly_err = [[ly - learned_data["f1_ci"][0]], [learned_data["f1_ci"][1] - ly]]
        ax.errorbar(
            lx, ly, xerr=lx_err, yerr=ly_err,
            fmt="X", color="#2980b9", ecolor="#2980b9", elinewidth=1.5, capsize=4,
            markersize=13, markeredgecolor="black", markeredgewidth=1.4,
            zorder=8, label="FeedbackRAG (Learned Policy)"
        )
        ax.annotate(
            f" FeedbackRAG (Learned Policy)\n ({lx:.1f} calls, {ly:.1f}% F1)",
            (lx, ly),
            textcoords="offset points",
            xytext=(12, 6),
            fontsize=9.5,
            weight="heavy",
            color="#1b4f72"
        )

    # Connect Pareto Frontier line
    if thresh_data:
        frontier_pts = sorted([
            (baselines["Always-L0"]["avg_calls"], baselines["Always-L0"]["f1"]),
            (thresh_data["avg_calls"], thresh_data["f1"]),
            (baselines["Always-L3"]["avg_calls"], baselines["Always-L3"]["f1"])
        ], key=lambda p: p[0])
        px, py = zip(*frontier_pts)
        ax.plot(px, py, linestyle="--", color="#27ae60", alpha=0.65, linewidth=2, label="Empirical Pareto Frontier")

    ax.set_title("Pareto Efficiency: Accuracy vs. Compute Cost (500 Questions with 95% Bootstrap CIs)", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Average LLM Calls per Query (Lower is Cheaper)", fontsize=11, labelpad=10)
    ax.set_ylabel("Exact Match & Token F1 (%) (Higher is Better)", fontsize=11, labelpad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8.5)

    out_file = "results/plots/pareto_frontier.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"[Plot] Pareto Frontier plot with 95% CIs successfully saved to {out_file}")

if __name__ == "__main__":
    generate_pareto_plot()
