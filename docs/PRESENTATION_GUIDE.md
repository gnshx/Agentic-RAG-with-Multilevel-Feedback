# FeedbackRAG: Live Defense & Presentation Walkthrough Guide

This guide is designed for your **live Google Meet presentation/defense**. It provides a 5-minute structured demonstration script that highlights the technical novelty, system architecture, and scientific rigor of FeedbackRAG.

---

## 🚀 How to Launch the Demo

In your terminal:
```bash
# Ensure virtual environment is active
source .venv/bin/activate

# Launch presentation dashboard
streamlit run app/streamlit_app.py --server.port 8501
```
Open **`http://localhost:8501`** and share your screen in Google Meet.

---

## 🎙️ 5-Minute Live Demonstration Script

### Act 1: The RAG Dilemma & Problem Framing (1 Minute)
> *"Professors, standard RAG systems suffer from an efficiency-accuracy dilemma. If you use simple single-pass RAG, you fail catastrophically on multi-hop questions because intermediate entities aren't known upfront. But if you use exhaustive agentic critique-revise loops, you incur 10 to 15 LLM calls and multi-second latencies on every single query—even simple factual questions.*
>
> *FeedbackRAG solves this by adapting the classic Operating Systems Multilevel Feedback Queue (MLFQ) algorithm to generative retrieval. Queries enter at Level 0 (cheapest direct inference) and escalate to higher compute tiers only when verifiable, reference-free feedback signals indicate deficiency."*

---

### Act 2: Interactive MLFQ Decision Trace (Tab 1 — 1.5 Minutes)
1. **Show Level 0 Early Exit:**
   - Click **`🇫🇷 Direct (L0): Capital of France`**.
   - Click **`⚡ Execute MLFQ Pipeline`**.
   - **Point out:** The green badge at Level 0:
     - 3 LLM calls, zero retrieval overhead, lowest possible cost.
     - Signal $S_1$ (Self-Consistency) passed with high agreement ($S_1 \ge 0.80$).
     - The system saved 80% compute by never calling retrieval.

2. **Show Multi-Hop Escalation to Level 2:**
   - Click **`🐧 Composition (L2): Linux Creator`** (*"What is the capital of the country where the creator of the Linux kernel was born?"*).
   - Click **`⚡ Execute MLFQ Pipeline`**.
   - **Walk through the pipeline progress bar:**
     - **Level 0 (Direct):** Evaluated $S_1 = 0.33 < 0.80$ $\to$ Red **ESCALATED** badge.
     - **Level 1 (Single-Pass RAG):** Retrieved context, but groundedness failed ($S_3 = 0.0 < 0.70$) because single-pass search missed the secondary bridge passage $\to$ Red **ESCALATED** badge.
     - **Level 2 (Iterative Multi-Hop):** Decomposed into sub-queries (*"Where was Linus Torvalds born?"* $\to$ *Finland* $\to$ *"What is the capital of Finland?"* $\to$ *Helsinki*), retrieved bridge entities, validated $S_3 = 1.00$ and $S_4 = 1.00$ $\to$ Green **EXITED HERE** badge!
     - Level 3 was **BYPASSED**, saving unnecessary agentic loops.

---

### Act 3: 4-Strategy Live Arena (Tab 2 — 1 Minute)
1. Select the Arena tab.
2. Click **`🥊 Run 4-Way Head-to-Head Comparison`**.
3. **Showcase the side-by-side comparison cards:**
   - **Always-L0:** Fails because it lacks external multi-hop facts.
   - **Always-L1:** Lacks the second-hop bridge entity.
   - **Always-L2:** Resolves the query accurately in 4 calls.
   - **FeedbackRAG:** Dynamically matches Always-L2 accuracy while demonstrating that on single-hop queries it cuts calls by up to 60%!

---

### Act 4: Live Feedback & SQLite Document Reweighting (Tab 3 — 1 Minute)
> *"One of our core architectural contributions is Level 3 Continuous Learning without retraining expensive neural weights."*

1. Switch to Tab 3 (**`🔄 3. Live Feedback & Reweighting`**).
2. Show the initial rankings for the query: *"Where did the descendants of the group of black Indians associated with the Seminole people settle?"*.
3. Click **`👍 Reward (+0.25)`** on the Mascogos supporting passage (`doc_349fd35fd71d`).
4. Point out how the dynamic weight multiplier increases in SQLite (`document_weights` table).
5. Click **`🔄 Simulate Process Restart & Cold Reopen`**:
   - The app destroys the in-memory vector store and reloads freshly from SQLite disk.
   - Point out to the professors: **The dynamic weight (+0.25) and boosted score survived the process restart identically!**

---

### Act 5: Epistemic Failure Recovery & Scientific Rigor (Tabs 4 & 6 — 30 Seconds)
1. Switch to Tab 4 (**`🛡️ 4. Failure Recovery & Injection Defense`**):
   - Explain the **Epistemic Blind Spot**: In the *Shellite chemical formula* case, Level 0 held an entrenched prior ($S_1 = 1.00$ mode answer), predicting `"Shellite"` (wrong!). FeedbackRAG escalated to Level 2 and recovered `"Picric acid"` (correct!).
   - Explain: *"Consistency is a proxy for confidence, not truth. Decoupled groundedness signals protect the system."*
2. Switch to Tab 6 (**`📊 6. Scientific Benchmark Observatory`**):
   - Highlight the **Dual-Regime Evaluation**:
     - Regime A: 500-question simulated state-machine benchmark.
     - Regime B: 100-question authentic HotpotQA benchmark on `qwen2.5-coder:14b` on an NVIDIA RTX 3060 12GB GPU host.
   - Highlight scientific honesty:
     - Always-L2 is the accuracy champion (54.2% F1).
     - FeedbackRAG achieves 49.6% F1, and the 4.60-point deficit is quantitatively attributed: **93% Level 0 premature exits** and **7% Level 3 agentic degradation**.

---

## 💡 Anticipated Questions & Winning Answers

- **Q: "Why does Always-L2 beat FeedbackRAG on the 100-question benchmark?"**
  - **A:** *"Because all 100 questions in HotpotQA are specifically constructed to require cross-document multi-hop synthesis. 19% of queries prematurely exited at Level 0 due to entrenched parametric hallucinations ($S_1=1.00$). In a general-purpose distribution with 40% simple queries, FeedbackRAG is Pareto-optimal."*
- **Q: "How does the SQLite feedback loop prevent catastrophic drift?"**
  - **A:** *"We enforce strict mathematical clamping on dynamic weights between $[-0.8, +1.5]$ so that a single malicious user cannot permanently suppress valid documents."*
- **Q: "Is the GBDT policy trained on test data?"**
  - **A:** *"Strictly zero leakage. It was trained exclusively on `data/dev.json` using 11 observable runtime features (no gold labels or F1 scores visible), verified via automated unit tests in CI."*
