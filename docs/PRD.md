# PRD: FeedbackRAG — Agentic RAG with a Multilevel Feedback System

**Course:** DSL504/DS504 Natural Language Processing, IIT Bhilai (Jul–Nov 2026)
**Group:** 14 | **Team:** NDS Ganesh, V Karthik, K Soma Charith Reddy, R Dheemanth
**Version:** 1.0 | **Status:** Build today, submit today

---

## 1. Problem statement

Standard RAG retrieves once and answers once. Easy questions waste retrieval compute, and hard (multi-hop) questions fail because one retrieval pass is not enough. Fully agentic pipelines (rewrite, retrieve, critique, revise, vote) fix hard questions but cost many LLM calls on every query, including easy ones.

## 2. Idea

Borrow the **Multilevel Feedback Queue (MLFQ)** idea from operating systems. Every query starts at the cheapest processing level. After each level, **feedback signals** (consistency, retrieval quality, groundedness, completeness) decide whether to return the answer or **escalate** to a more expensive level. A budget guard caps cost.

## 3. Research question and hypotheses

**RQ:** Can feedback-driven escalation match the accuracy of always running the full agentic pipeline while using far fewer LLM calls and less latency? Which feedback signals best predict that escalation is needed?

- **H1:** FeedbackRAG reaches accuracy within 2 F1 points of always-Level-3 with at least 30% fewer LLM calls on average.
- **H2:** FeedbackRAG beats vanilla RAG on multi-hop questions (HotpotQA) in EM/F1.
- **H3:** Removing any single feedback signal lowers accuracy or raises cost (each signal contributes).

These are hypotheses, not promised results. Report what is actually measured.

## 4. Goals and non-goals

**Goals**
- G1: Working 4-level pipeline with a feedback controller.
- G2: Evaluation of 4 baselines vs FeedbackRAG on HotpotQA (100 test questions).
- G3: Accuracy-vs-cost Pareto plot and ablation table.
- G4: Demo UI showing the answer, citations and the escalation trace.
- G5: Report and slides.

**Non-goals**
- Training or fine-tuning models (same LLM, different prompts).
- Production deployment, auth, scaling.
- Full Wikipedia indexing.

## 5. Users and use cases

| User | Use case |
|---|---|
| Course evaluator | Reads results, checks the hypotheses and the experimental rigor |
| Demo viewer | Asks a question, sees the answer, the evidence, and which levels ran |
| Team developers | Run the eval harness and reproduce the tables |

## 6. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR1 | Level 0: answer without retrieval, 3 sampled answers for self-consistency | Must |
| FR2 | Level 1: hybrid retrieval (BM25 + dense), top-k, single-pass answer with citations | Must |
| FR3 | Level 2: query rewrite and decomposition, iterative multi-hop retrieval, reranking | Must |
| FR4 | Level 3: broadened retrieval, draft-critic-revise loop, majority vote over N drafts | Should |
| FR5 | Feedback signals: consistency, retrieval grade, groundedness, completeness | Must |
| FR6 | Controller escalates when signals fall below per-level thresholds | Must |
| FR7 | Budget guard: max LLM calls per query | Must |
| FR8 | Per-query trace log (levels visited, signal values, calls, latency) | Must |
| FR9 | Eval harness: EM, F1, faithfulness, calls, latency | Must |
| FR10 | Baselines: no-retrieval, vanilla RAG, always-L2, always-L3 | Must |
| FR11 | Threshold sweep using cached signals (no repeated LLM calls) | Should |
| FR12 | Streamlit demo with trace view | Should |
| FR13 | Robustness test with injected distractor chunks | Could |

## 7. Non-functional requirements

- **Reproducibility:** fixed seeds, `temperature=0` for graders and checkers, config in one YAML file, results cached to disk.
- **Cost:** free-tier API or local model only. The full test run must finish in under 45 minutes.
- **Reliability:** every loop is bounded (max 2 retries per level), and every LLM call has a timeout and a retry.
- **Transparency:** every answer ships with citations and a trace.

## 8. Success metrics

| Metric | Target |
|---|---|
| EM / F1 on 100 HotpotQA test questions | Report for every system |
| Avg LLM calls per query | At least 30% below always-L3 (H1) |
| Faithfulness (supported claims / total claims) | Higher than vanilla RAG |
| Escalation rate per level | Reported with analysis |
| Ablation deltas | Reported for each removed signal |

## 9. Assumptions and risks

| Risk | Mitigation |
|---|---|
| API rate limits | Cache all LLM outputs by prompt hash, add retries, use a smaller test split if needed |
| LLM grader is noisy | `temperature=0`, strict JSON output, spot-check 20 samples by hand |
| Thresholds overfit the dev split | Tune on 50 dev questions, report on separate 100 test questions |
| Result does not support H1 | Report honestly and analyze why |
| Time runs out | Cut order in PLAN.md (robustness, then L3 voting, then reranker) |

## 10. Deliverables

1. Code repository (`feedbackrag/`)
2. `results/` with tables, Pareto plot and ablation
3. Report (derive from `HLD.md` and results)
4. Slides
5. Demo video or live demo

## 11. References

- Singh et al., *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG*, arXiv:2501.09136
- Yan et al., *Corrective Retrieval Augmented Generation (CRAG)*, arXiv:2401.15884
- Asai et al., *Self-RAG*, arXiv:2310.11511
- Jeong et al., *Adaptive-RAG*, arXiv:2403.14403 (verify before citing)
- Yao et al., *ReAct*, arXiv:2210.03629 (verify before citing)
- Shinn et al., *Reflexion*, arXiv:2303.11366 (verify before citing)
- Es et al., *RAGAS*, arXiv:2309.15217 (verify before citing)
- Yang et al., *HotpotQA*, EMNLP 2018
