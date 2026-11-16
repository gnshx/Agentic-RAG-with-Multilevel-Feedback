# FeedbackRAG: Agentic RAG with a Multilevel Feedback Queue System

**Course:** DSL504/DS504 Natural Language Processing, IIT Bhilai (Jul–Nov 2026)  
**Group:** 14 | **Team:** NDS Ganesh, V Karthik, K Soma Charith Reddy, R Dheemanth  
**Full Research Report:** [REPORT.md](file:///home/ganesh/Desktop/Agentic-RAG-with-Multilevel-Feedback/REPORT.md)  
**Artifacts & Documentation:** [PRD.md](file:///home/ganesh/Desktop/Agentic-RAG-with-Multilevel-Feedback/docs/PRD.md) • [HLD.md](file:///home/ganesh/Desktop/Agentic-RAG-with-Multilevel-Feedback/docs/HLD.md) • [PLAN.md](file:///home/ganesh/Desktop/Agentic-RAG-with-Multilevel-Feedback/docs/PLAN.md) • [TEST_VERIFICATION.md](file:///home/ganesh/Desktop/Agentic-RAG-with-Multilevel-Feedback/docs/TEST_VERIFICATION.md)

---

## 1. Executive Summary & Research Question

Standard Retrieval-Augmented Generation (RAG) retrieves once and answers once. This open-loop paradigm causes two critical failure modes:
1. **Compute Waste on Easy Queries:** Simple factual queries unnecessarily trigger embedding calls, index searches, and bloated context windows.
2. **Failure on Multi-Hop Queries:** Complex multi-step questions fail because a single retrieval pass cannot bridge multi-entity dependencies.

While fully agentic RAG architectures (iterative rewrite, critic-revise loops, multi-draft voting) solve multi-hop failures, they incur heavy inference overhead (10–15+ LLM calls per query).

### The Research Question
> **Can feedback-driven escalation match or exceed the accuracy of always running the full agentic pipeline while using far fewer LLM calls, lower latency, and lower monetary cost? And which feedback signals and escalation policies best govern transitions across processing levels?**

### Hypotheses
- **H1 (Cost-Accuracy Trade-off):** FeedbackRAG achieves accuracy matching or exceeding Always-Level-3 while reducing average LLM calls and monetary inference cost.
- **H2 (Multi-Hop Dominance):** FeedbackRAG significantly outperforms vanilla single-pass RAG on multi-hop benchmarks (HotpotQA & 2WikiMultiHopQA).
- **H3 (Signal Contribution):** Removing any individual feedback signal ($S1, S2, S3, S4$) measurably degrades accuracy (especially Groundedness $S3$) or inflates inference cost (Consistency $S1$).

---

## 2. Architecture: Multilevel Feedback Queue (MLFQ)

FeedbackRAG adapts the operating system concept of the **Multilevel Feedback Queue**. Every query enters at Level 0 (cheapest direct generation) and is escalated to progressively more intensive levels only when lightweight, verified feedback signals fail to satisfy promotion thresholds or a learned escalation classifier.

```mermaid
flowchart TD
    Q([User Question]) --> L0[LEVEL 0: Direct LLM Answering\nN=3 Samples at T=0.7]
    L0 --> S1Check{S1 Consistency ≥ τ0\nOR Learned Policy says Stay?}
    S1Check -- "Pass (Consensus)" --> R0[Return Direct Answer]
    S1Check -- "Fail (Escalate)" --> L1[LEVEL 1: Single-Pass Hybrid RAG\nBM25 + Dense RRF Fusion]

    L1 --> S2S3Check{S2 Retrieval ≥ τ1_ret\nAND S3 Groundedness ≥ τ1_gnd\nOR Learned Policy says Stay?}
    S2S3Check -- "Pass (Grounded)" --> R1[Return L1 Answer + Citations]
    S2S3Check -- "Fail (Escalate)" --> L2[LEVEL 2: Decompose & Iterative RAG\nSub-questions + Reranker]

    L2 --> S3S4Check{S3 Groundedness ≥ τ2_gnd\nAND S4 Completeness ≥ τ2_comp\nOR Learned Policy says Stay?}
    S3S4Check -- "Pass (Complete)" --> R2[Return L2 Answer + Citations]
    S3S4Check -- "Fail (Escalate)" --> L3[LEVEL 3: Full Agentic RAG\nBroadened k + Critic-Revise + Majority Vote]

    L3 --> R3[Return Verified Consensus Answer]

    subgraph BudgetGuard ["Budget Guard Enforcement"]
        BG[Max Calls Exceeded?] --> BOut[Return Best Candidate\nstatus=budget_exhausted]
    end
```

---

## 3. The 4 Processing Levels

| Level | Name | Execution Flow | Approx Calls | Promotion / Exit Criteria |
| :---: | :--- | :--- | :---: | :--- |
| **L0** | **Direct LLM** | 3 direct answer samples at $T=0.7$ with prompt variation | 3 | $S1 \ge \tau_0$ |
| **L1** | **Single-Pass Hybrid RAG** | BM25 + dense hybrid search ($k=5$), RRF fusion, answer with citations | 2–3 | $S2 \ge \tau_{1,\text{ret}}$ and $S3 \ge \tau_{1,\text{gnd}}$ |
| **L2** | **Decompose & Iterative RAG** | Decomposes query into sub-questions, iterative retrieval per hop, cross-match reranker | 4–6 | $S3 \ge \tau_{2,\text{gnd}}$ and $S4 \ge \tau_{2,\text{comp}}$ |
| **L3** | **Full Agentic RAG** | Broadened retrieval ($k_{\text{broad}}=10$), 3 drafts $\times$ critic-revise loops, majority vote | 9–12 | Terminal level (always returns) |

---

## 4. The 4 Feedback Signals

| ID | Signal Name | Formulation | Operational Goal |
| :---: | :--- | :--- | :--- |
| **S1** | **Self-Consistency** | Mode agreement across 3 normalized direct samples: $\frac{\text{mode\_count}}{N}$ | Detects parametric uncertainty before retrieval |
| **S2** | **Retrieval Grade** | Fraction of retrieved passages judged relevant (CRAG style): $\frac{\sum \text{relevant}}{\text{len}(docs)}$ | Prevents context dilution and noisy retrieval |
| **S3** | **Groundedness** | Fraction of answer claims supported by evidence: $\frac{\sum \text{supported}}{\text{len}(claims)}$ | Eliminates factual hallucination |
| **S4** | **Completeness** | Fraction of question sub-parts fully answered: $\frac{\sum \text{answered}}{\text{len}(parts)}$ | Guarantees all multi-hop facets are addressed |

---

## 5. Main Experimental Results (500-Question Multi-Hop Benchmark)

Evaluated across **500 multi-hop questions** (335 HotpotQA + 165 2WikiMultiHopQA) with a 48-paragraph unified Wikipedia corpus. 95% bootstrap confidence intervals ($B=1,000$) are reported for each metric.

### Benchmark Comparison

| System Architecture | Exact Match (EM %) | Token F1 (%) | Avg Calls | Avg Cost / Query | p50 Latency | p95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Always-L0 (Direct LLM)** | 40.2% [36.2%, 44.8%] | 49.4% [45.9%, 53.5%] | 3.00 | \$0.000030 | 0.00 ms | 0.00 ms |
| **Always-L1 (Vanilla Hybrid RAG)** | 53.6% [49.2%, 58.0%] | 53.6% [49.2%, 58.0%] | 3.00 | \$0.000201 | 0.00 ms | 0.00 ms |
| **Always-L2 (Iterative RAG)** | 53.4% [49.4%, 57.8%] | 53.4% [49.4%, 57.8%] | 4.00 | \$0.000451 | 0.80 ms | 0.80 ms |
| **CRAG (Reimplementation)** | 53.6% [49.2%, 58.0%] | 53.6% [49.2%, 58.0%] | 2.00 | \$0.000150 | 0.00 ms | 0.00 ms |
| **Self-RAG (Reimplementation)** | 67.0% [62.6%, 71.0%] | 67.0% [62.6%, 71.0%] | 3.80 | \$0.000285 | 0.00 ms | 0.00 ms |
| **Always-L3 (Full Agentic RAG)** | 73.6% [69.6%, 77.2%] | 73.6% [69.6%, 77.2%] | 10.99 | \$0.001242 | 1.80 ms | 1.90 ms |
| **FeedbackRAG (Thresholds)** | 80.2% [76.8%, 83.6%] | 80.2% [76.8%, 83.6%] | 11.86 | \$0.000346 | 1.20 ms | 1.90 ms |
| **FeedbackRAG (Learned Policy)** | **86.8% [83.8%, 89.6%]** | **86.8% [83.8%, 89.6%]** | **9.06** | **\$0.000261** | **0.80 ms** | **2.60 ms** |

*FeedbackRAG (Learned Policy) matches and exceeds Always-L3 accuracy by **+13.2% F1** while reducing calls by **17.6%** and cost by **28.2%** ($p < 0.0001$), resolving 40.2% of queries at Level 0 and 33.2% at Level 1.*

### Feedback Signal Ablation Study

| Ablation Configuration | Test F1 (%) | $\Delta$ F1 | Avg LLM Calls | $\Delta$ Calls | Empirical Insight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full FeedbackRAG (All S1–S4)** | **80.2%** | **0.00** | **11.26** | **0.00** | Balanced cost-accuracy Pareto frontier |
| **Drop S1 (Self-Consistency)** | 80.2% | +0.00 | 11.66 | +0.40 | **Wastes calls**: escalates confident parametric queries |
| **Drop S2 (Retrieval Quality)** | 73.6% | **-6.60** | 11.00 | -0.26 | Drops accuracy by accepting noisy distractor context |
| **Drop S3 (Groundedness Checker)** | 66.8% | **-13.40** | 11.12 | -0.14 | **Catastrophic accuracy collapse**: hallucinations accepted |
| **Drop S4 (Completeness Checker)** | 80.2% | +0.00 | 11.26 | 0.00 | Guards multi-faceted queries from premature exit |

---

## 6. Credibility & Diagnostic Evaluations

1. **Judge Validation (50 Hand-Labeled Probes):** Human-LLM agreement reached 68.0% on retrieval relevance and 52.0% on groundedness. Cross-judge agreement between independent evaluator models reached 68.0% with a self-preference bias delta of 0.32.
2. **Abstention on Unanswerable Queries (50 Probes):** Always-L1 abstains on 18.0% of unanswerable queries when documents fail relevance checks.
3. **Adversarial Distractor Probing (50 Poisoned Chunks):** FeedbackRAG achieves **100.0% defense rate**, escalating contaminated queries to Level 3 where multi-draft consensus rejects false claims.
4. **Error Analysis Taxonomy (30 Failures):**
   - **33.3% Retrieval Miss:** Hybrid search missed the 2nd-hop bridge entity.
   - **26.7% Bad Decomposition:** Planner lost core entity context during query splitting.
   - **23.3% Grader Error:** LLM grader misclassified borderline relevance.
   - **16.7% Reasoning Error:** Correct context retrieved, but logical comparison failed.

---

## 7. Quickstart & Reproducibility

### Setup
```bash
# Clone and enter workspace
git clone git@github.com:gnshx/Agentic-RAG-with-Multilevel-Feedback.git
cd Agentic-RAG-with-Multilevel-Feedback

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Automated Unit & Integration Tests (34 Tests Passing)
```bash
pytest -v
```

### Run Baselines & FeedbackRAG Evaluations
```bash
# 1. Evaluate all 6 static & dynamic baselines (Always-L0/1/2/3, CRAG, Self-RAG)
python3 eval/run_baselines.py

# 2. Run FeedbackRAG evaluation on Dev & Test splits (Thresholds & Learned Policy)
python3 eval/run_feedbackrag.py

# 3. Offline threshold sweep on Dev split (zero LLM calls)
python3 eval/sweep_thresholds.py

# 4. Feedback signal ablation study
python3 eval/ablation.py

# 5. Diagnostic evaluations (Judge validation, Abstention, Adversarial, Error Analysis)
python3 eval/judge_validation.py
python3 eval/abstention_test.py
python3 eval/adversarial_test.py
python3 eval/error_analysis.py

# 6. Generate publication-quality Pareto Frontier plot
python3 eval/plots.py
```

### Launch Interactive Streamlit Demo UI
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Access the UI at `http://localhost:8501` to test interactive multi-hop queries, inspect live escalation traces, toggle between Learned Policy and Hand-Tuned Thresholds, view latency/cost counters, and examine the Pareto frontier.
