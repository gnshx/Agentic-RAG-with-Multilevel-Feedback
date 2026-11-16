# TEST and VERIFICATION PLAN: FeedbackRAG

## 1. Unit tests (pytest, no LLM needed: use a mock LLM)

| ID | Test | Expected |
|---|---|---|
| U1 | `metrics.em("The Eiffel Tower", "eiffel tower")` | 1 (normalization drops case, articles, punctuation) |
| U2 | `metrics.f1` on partial overlap | Matches the hand-computed value |
| U3 | `llm.call` same prompt twice | Second call hits the cache, call counter increases once |
| U4 | `llm.call` with a failing backend | Retries up to `max_retries`, then raises a clear error |
| U5 | JSON parse of a grader output with extra text | Extracts the JSON, or returns score 0 on failure |
| U6 | Controller with mock signals above thresholds at L0 | Returns at L0, `calls` is 3 |
| U7 | Controller with L0 and L1 signals below thresholds | Reaches L2 |
| U8 | Controller at L3 | Always returns, never reaches level 4 |
| U9 | Budget guard: `max_calls_per_query=5`, escalation requested | Returns with `status="budget_exhausted"` |
| U10 | Retriever `search(q, 5)` | Returns exactly 5 results sorted by score |
| U11 | RRF fusion with known rankings | Matches the hand-computed fused order |
| U12 | Signal range | All signals are floats in [0, 1] |

## 2. Component and integration tests

| ID | Test | Expected |
|---|---|---|
| I1 | Retrieval sanity: 20 dev questions, check whether a gold supporting paragraph is in the top-5 | Recall@5 reported (should be clearly above random) |
| I2 | Level 1 end to end on 5 questions | Answer, citations that reference retrieved chunk ids, trace entry |
| I3 | Level 2 decomposition on 5 multi-hop questions | Sub-questions are non-empty and each retrieves chunks |
| I4 | Full controller on 10 questions | Every trace lists the levels visited, signals and call count |
| I5 | Citations point to real doc ids in `docs` | No invented ids |
| I6 | Cached rerun of the same 10 questions | Identical outputs, zero new API calls |

## 3. System and experiment verification

| ID | Check | Pass criterion |
|---|---|---|
| E1 | Baselines run on dev and test | EM/F1 tables exist for all 4 baselines and FeedbackRAG |
| E2 | No data leakage | Thresholds tuned on dev only, test never used for tuning (assert in code) |
| E3 | Determinism | Fixed seed, `temperature=0` for graders, two reruns give identical metrics |
| E4 | Cost accounting | Sum of per-query `calls` equals the LLM client's total counter |
| E5 | Pareto plot | Includes every baseline and the FeedbackRAG sweep curve, with axes labeled |
| E6 | Ablation | One row per removed signal, with ΔF1 and Δcalls |
| E7 | Hypothesis check | H1, H2, H3 are each marked supported, partly supported, or not supported with the measured numbers |

## 4. Grader validation (to trust the LLM judges)

- [ ] Hand-label 20 random (answer, evidence) pairs as supported or unsupported. Compare to the S3 groundedness output and report agreement.
- [ ] Hand-label 20 random (question, chunk) pairs as relevant or not. Compare to the S2 output and report agreement.
- [ ] Report the agreement honestly, including the disagreements. If agreement is low, say so in Limitations.

## 5. Negative and edge cases

| ID | Case | Expected |
|---|---|---|
| N1 | Empty question | Clear error or message, no crash |
| N2 | Very long question | Truncated or handled, no crash |
| N3 | Question unanswerable from the corpus | The system says it cannot find supporting evidence rather than inventing an answer; groundedness stays low and the question escalates, and the final answer is flagged |
| N4 | LLM returns invalid JSON | Falls back to score 0 and escalates |
| N5 | API timeout | Retried, then the level fails over safely without hanging |
| N6 | (Optional) Inject 5 irrelevant chunks | S2 drops, escalation fires, the answer does not rely on the junk |

## 5b. Demo checks (Streamlit)

- [ ] The page loads and a question returns an answer.
- [ ] The trace panel lists each level, its signals, calls and latency.
- [ ] Citations show the source text.
- [ ] A simple question stops at an early level; a multi-hop question escalates.

## 6. Pre-submission checklist

- [ ] `pytest` passes.
- [ ] A fresh clone runs the README steps end to end.
- [ ] `results/` contains the metrics table, the Pareto plot, the ablation table, and `trace.jsonl`.
- [ ] Every number in the report and slides matches `results/`.
- [ ] No API keys committed (check `.gitignore`).
- [ ] References checked: IDs verified before citing.
- [ ] Limitations section mentions grader noise, the small corpus, and dev-set tuning.
- [ ] Backup demo video recorded.

## 7. Metric definitions

- **EM:** exact match after normalization (lowercase, remove punctuation and articles).
- **F1:** token-level F1 between prediction and gold answer.
- **Faithfulness:** supported claims / total claims (S3, validated against hand labels).
- **Calls:** LLM calls per query, from the LLM client counter.
- **Escalation rate:** share of queries reaching each level.
