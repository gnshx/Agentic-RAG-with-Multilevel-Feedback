import streamlit as st
import json
import time
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm import get_llm_instance
from src.retriever import HybridRetriever
from src.planner import QueryPlanner
from src.generator import Generator
from src.levels import run_level_0, run_level_1, run_level_2, run_level_3
from src.controller import run_controller
from src.learned_policy import LearnedEscalationPolicy
from src.cost_tracker import CostTracker

st.set_page_config(
    page_title="FeedbackRAG: Multilevel Feedback Queue Agentic RAG",
    page_icon="⚡",
    layout="wide"
)

# Custom High-Contrast Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2e3648;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge-pass {
        background-color: #1b4332;
        color: #74c69d;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .badge-fail {
        background-color: #49111c;
        color: #ff85a1;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    corpus = []
    if os.path.exists("data/corpus.json"):
        with open("data/corpus.json", "r", encoding="utf-8") as f:
            corpus = json.load(f)
    llm = get_llm_instance(cfg)
    retriever = HybridRetriever(corpus=corpus)
    planner = QueryPlanner(llm=llm)
    generator = Generator(llm=llm)
    policy = LearnedEscalationPolicy.load("results/learned_policy.pkl")
    return cfg, corpus, llm, retriever, planner, generator, policy

cfg, corpus, llm, retriever, planner, generator, policy = load_system()

levels_map = {
    0: lambda s: run_level_0(s, llm, cfg),
    1: lambda s: run_level_1(s, llm, retriever, generator, cfg),
    2: lambda s: run_level_2(s, llm, retriever, planner, generator, cfg),
    3: lambda s: run_level_3(s, llm, retriever, generator, cfg)
}

# Sidebar
st.sidebar.title("⚡ FeedbackRAG Engine")
st.sidebar.markdown("**Multilevel Feedback Queue RAG (IIT Bhilai)**")

exec_mode = st.sidebar.radio(
    "Escalation Mechanism:",
    options=["learned", "thresholds"],
    format_func=lambda x: "🤖 Learned Escalation Policy (Gradient Boosting)" if x == "learned" else "⚙️ Hand-Tuned Heuristic Thresholds"
)

st.sidebar.subheader("Threshold Controls (Heuristic Mode)")
t0 = st.sidebar.slider("L0 Tau (Consistency S1)", 0.0, 1.0, float(cfg["levels"]["L0"]["tau"]), 0.05)
t1_ret = st.sidebar.slider("L1 Retrieval Quality (S2)", 0.0, 1.0, float(cfg["levels"]["L1"]["tau_retrieval"]), 0.05)
t1_gnd = st.sidebar.slider("L1 Groundedness (S3)", 0.0, 1.0, float(cfg["levels"]["L1"]["tau_ground"]), 0.05)
t2_gnd = st.sidebar.slider("L2 Groundedness (S3)", 0.0, 1.0, float(cfg["levels"]["L2"]["tau_ground"]), 0.05)
t2_comp = st.sidebar.slider("L2 Completeness (S4)", 0.0, 1.0, float(cfg["levels"]["L2"]["tau_complete"]), 0.05)

cfg["levels"]["L0"]["tau"] = t0
cfg["levels"]["L1"]["tau_retrieval"] = t1_ret
cfg["levels"]["L1"]["tau_ground"] = t1_gnd
cfg["levels"]["L2"]["tau_ground"] = t2_gnd
cfg["levels"]["L2"]["tau_complete"] = t2_comp

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Corpus Size:** `{len(corpus)} passages`")
st.sidebar.markdown(f"**Trained Policy Status:** `{'Loaded (Gradient Boosting)' if policy and policy.is_fitted else 'Heuristic Only'}`")

# Main Title
st.title("FeedbackRAG: Cost-Accuracy Optimal Agentic RAG")
st.caption("IIT Bhilai • Natural Language Processing • Course Group 14")

tabs = st.tabs([
    "🚀 Interactive Query Demo",
    "📊 Benchmark Results (500 Questions)",
    "📈 Pareto Frontier",
    "🔬 Credibility & Diagnostics"
])

with tabs[0]:
    col_input, col_presets = st.columns([3, 1])

    preset_q = None
    with col_presets:
        st.markdown("**Benchmark Presets:**")
        if st.button("Direct: Capital of France", use_container_width=True):
            preset_q = "What is the capital city of France?"
        if st.button("Multi-hop: Eiffel River", use_container_width=True):
            preset_q = "Which river flows through the European capital city where the Eiffel Tower is situated?"
        if st.button("Bridge: Inception Award", use_container_width=True):
            preset_q = "What award did the director of the film Inception win for his 2023 biographical movie?"
        if st.button("Composition: Linux Creator Capital", use_container_width=True):
            preset_q = "What is the capital of the country where the creator of the Linux kernel was born?"

    with col_input:
        user_query = st.text_input(
            "Enter your question:",
            value=preset_q if preset_q else "Which river flows through the European capital city where the Eiffel Tower is situated?",
            placeholder="Type any multi-hop, compositional, or factual question..."
        )
        run_btn = st.button("Execute FeedbackRAG Pipeline", type="primary", use_container_width=True)

    if run_btn and user_query:
        with st.spinner("Processing through MLFQ state machine..."):
            t_start = time.time()
            state = run_controller(user_query, levels_map, cfg, llm, policy=policy, mode=exec_mode)
            total_time = round(time.time() - t_start, 4)

        # Overview Metrics in Real Units
        m1, m2, m3, m4, m5 = st.columns(5)
        lvl_names = {0: "Level 0 (Direct)", 1: "Level 1 (Single-Pass RAG)", 2: "Level 2 (Iterative RAG)", 3: "Level 3 (Full Agentic)"}
        m1.metric("Exit Level", lvl_names.get(state["level"], "Unknown"))
        m2.metric("LLM Calls", state["calls"])
        m3.metric("Total Tokens", state.get("total_tokens", 0))
        m4.metric("Est. Cost ($ USD)", f"${state.get('cost_usd', 0.0):.6f}")
        m5.metric("Latency", f"{state['latency']}s")

        # Answer Presentation
        st.subheader("Verified Answer")
        st.success(f"**{state['answer']}**")
        if state.get("citations"):
            st.markdown(f"**Citations:** `{', '.join(state['citations'])}`")

        # Level Trace Details
        st.subheader("Escalation Decision Trace")
        for step in state["trace"]:
            lvl = step["level"]
            signals = step["signals"]
            calls = step["calls_used_level"]
            dec = step.get("decision", {})

            with st.expander(f"Level {lvl}: {lvl_names[lvl]} — {calls} Calls (Action: {dec.get('action', 'return')})", expanded=(lvl == state["level"])):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown("**Feedback Signals Evaluated:**")
                    if "S1" in signals:
                        st.markdown(f"• **S1 (Consistency):** `{signals['S1']:.2f}`")
                    if "S2" in signals:
                        st.markdown(f"• **S2 (Retrieval Grade):** `{signals['S2']:.2f}`")
                    if "S3" in signals:
                        st.markdown(f"• **S3 (Groundedness):** `{signals['S3']:.2f}`")
                    if "S4" in signals:
                        st.markdown(f"• **S4 (Completeness):** `{signals['S4']:.2f}`")
                with c2:
                    st.markdown(f"**Decision Meta:** `{dec}`")
                    st.markdown(f"**Draft Preview:** *{step['answer_preview']}*")

        # Evidence Display
        if state.get("docs"):
            with st.expander("📚 Retrieved Evidence Context", expanded=False):
                for i, d in enumerate(state["docs"]):
                    st.markdown(f"**[{i+1}] {d.get('title', 'Unknown')}** (Fused Score: `{d.get('score', 0):.4f}`)")
                    st.caption(d.get("text", ""))

with tabs[1]:
    st.subheader("Comprehensive Benchmark Results (500 Multi-Hop Questions with 95% Bootstrap CIs)")
    b_path = "results/tables/baselines_results.json"
    t_path = "results/tables/feedbackrag_test_thresholds_results.json"
    l_path = "results/tables/feedbackrag_test_learned_results.json"

    if os.path.exists(b_path) and os.path.exists(t_path) and os.path.exists(l_path):
        with open(b_path) as f: b_data = json.load(f)
        with open(t_path) as f: t_data = json.load(f)
        with open(l_path) as f: l_data = json.load(f)

        rows = [
            {"System": "Always-L0 (Direct LLM)", "Exact Match (95% CI)": b_data["Always-L0"]["em_formatted"], "Token F1 (95% CI)": b_data["Always-L0"]["f1_formatted"], "Avg Calls": b_data["Always-L0"]["avg_calls"], "Cost ($/Q)": f"${b_data['Always-L0']['avg_cost_usd']:.6f}"},
            {"System": "Always-L1 (Vanilla Hybrid RAG)", "Exact Match (95% CI)": b_data["Always-L1"]["em_formatted"], "Token F1 (95% CI)": b_data["Always-L1"]["f1_formatted"], "Avg Calls": b_data["Always-L1"]["avg_calls"], "Cost ($/Q)": f"${b_data['Always-L1']['avg_cost_usd']:.6f}"},
            {"System": "Always-L2 (Iterative Multi-hop)", "Exact Match (95% CI)": b_data["Always-L2"]["em_formatted"], "Token F1 (95% CI)": b_data["Always-L2"]["f1_formatted"], "Avg Calls": b_data["Always-L2"]["avg_calls"], "Cost ($/Q)": f"${b_data['Always-L2']['avg_cost_usd']:.6f}"},
            {"System": "CRAG (Corrective RAG Reimpl)", "Exact Match (95% CI)": b_data["CRAG-Reimpl"]["em_formatted"], "Token F1 (95% CI)": b_data["CRAG-Reimpl"]["f1_formatted"], "Avg Calls": b_data["CRAG-Reimpl"]["avg_calls"], "Cost ($/Q)": f"${b_data['CRAG-Reimpl']['avg_cost_usd']:.6f}"},
            {"System": "Self-RAG (Reflection Reimpl)", "Exact Match (95% CI)": b_data["Self-RAG-Reimpl"]["em_formatted"], "Token F1 (95% CI)": b_data["Self-RAG-Reimpl"]["f1_formatted"], "Avg Calls": b_data["Self-RAG-Reimpl"]["avg_calls"], "Cost ($/Q)": f"${b_data['Self-RAG-Reimpl']['avg_cost_usd']:.6f}"},
            {"System": "Always-L3 (Full Agentic RAG)", "Exact Match (95% CI)": b_data["Always-L3"]["em_formatted"], "Token F1 (95% CI)": b_data["Always-L3"]["f1_formatted"], "Avg Calls": b_data["Always-L3"]["avg_calls"], "Cost ($/Q)": f"${b_data['Always-L3']['avg_cost_usd']:.6f}"},
            {"System": "FeedbackRAG (Thresholds)", "Exact Match (95% CI)": t_data["em_formatted"], "Token F1 (95% CI)": t_data["f1_formatted"], "Avg Calls": t_data["avg_calls"], "Cost ($/Q)": f"${t_data['avg_cost_usd']:.6f}"},
            {"System": "FeedbackRAG (Learned Policy)", "Exact Match (95% CI)": l_data["em_formatted"], "Token F1 (95% CI)": l_data["f1_formatted"], "Avg Calls": l_data["avg_calls"], "Cost ($/Q)": f"${l_data['avg_cost_usd']:.6f}"}
        ]
        st.table(rows)

        st.subheader("Signal Ablation Study (500 Questions)")
        if os.path.exists("results/tables/ablation_results.json"):
            with open("results/tables/ablation_results.json") as f:
                st.table(json.load(f))
    else:
        st.info("Evaluation data loading...")

with tabs[2]:
    st.subheader("Accuracy vs. Cost Pareto Frontier")
    if os.path.exists("results/plots/pareto_frontier.png"):
        st.image("results/plots/pareto_frontier.png", caption="Pareto Trade-off: Accuracy (F1 %) vs Average LLM Calls per Query", use_container_width=True)

with tabs[3]:
    st.subheader("Empirical Credibility & Diagnostic Suite")
    c_diag1, c_diag2 = st.columns(2)

    with c_diag1:
        st.markdown("**1. Judge Validation (50 Hand-Labeled Pairs)**")
        if os.path.exists("results/tables/judge_validation_results.json"):
            with open("results/tables/judge_validation_results.json") as f:
                st.json(json.load(f))

        st.markdown("**2. Abstention Test (50 Unanswerable Queries)**")
        if os.path.exists("results/tables/abstention_results.json"):
            with open("results/tables/abstention_results.json") as f:
                st.table(json.load(f))

    with c_diag2:
        st.markdown("**3. Adversarial Robustness (50 Contaminated Distractors)**")
        if os.path.exists("results/tables/adversarial_results.json"):
            with open("results/tables/adversarial_results.json") as f:
                st.json(json.load(f))

        st.markdown("**4. Error Analysis Taxonomy (30 Failures Breakdown)**")
        if os.path.exists("results/tables/error_analysis.json"):
            with open("results/tables/error_analysis.json") as f:
                st.table(json.load(f)["category_breakdown"])
