# HLD: FeedbackRAG — High-Level Design and Architecture

## 1. System overview

FeedbackRAG answers questions by escalating through 4 processing levels. A **Feedback Controller** inspects cheap signals after each level and decides: **return** or **escalate**.

```
                        +-------------------------+
 Question ------------> |   Feedback Controller   | <---- config.yaml (thresholds, budget)
                        |  (LangGraph state graph)|
                        +-----------+-------------+
                                    |
        +---------------+-----------+-----------+---------------+
        v               v                       v               v
   LEVEL 0          LEVEL 1                 LEVEL 2         LEVEL 3
   Direct LLM       Hybrid RAG              Decompose +     Full agentic:
   (3 samples)      single pass             iterative RAG   broaden, critic,
                                            + rerank        revise, vote
        |               |                       |               |
        +------+--------+-----------+-----------+---------------+
               v                    v
        Signal Computer         Trace Logger  --> results/*.jsonl
   (consistency, retrieval,
    groundedness, completeness)
               |
               v
        Return answer + citations + trace
```

## 2. Components

| # | Component | Responsibility | Tech |
|---|---|---|---|
| C1 | Controller | State machine, escalation rules, budget guard | LangGraph |
| C2 | LLM client | One wrapper for all calls, caching by prompt hash, retry, timeout, call counter | Gemini/Groq API or Ollama (Qwen2.5-3B) |
| C3 | Retriever | BM25 + dense hybrid, score fusion (RRF) | rank_bm25, sentence-transformers (bge-small), FAISS |
| C4 | Reranker (optional) | Rerank top-k for Level 2 | cross-encoder (ms-marco-MiniLM) |
| C5 | Query planner | Rewrite and decompose multi-hop questions | LLM prompt |
| C6 | Generator | Answer with citations from evidence | LLM prompt |
| C7 | Signal computer | The 4 feedback signals | LLM prompts + embedding or string match |
| C8 | Trace logger | Per-query JSONL: levels, signals, calls, latency | Python |
| C9 | Eval harness | EM, F1, faithfulness, cost, Pareto plot, ablations | Python, pandas, matplotlib |
| C10 | Demo UI | Ask, answer, evidence and trace view | Streamlit |

## 3. Levels in detail

| Level | What runs | Approx. LLM calls | Signals checked to exit |
|---|---|---|---|
| L0 | 3 sampled direct answers | 3 | S1 consistency ≥ τ0 |
| L1 | Hybrid retrieval top-5, one generation | 2–3 | S2 retrieval grade ≥ τ1 and S3 groundedness ≥ τ1 |
| L2 | Rewrite, decompose into sub-questions, retrieve per sub-question, rerank, generate | 5–8 | S3 groundedness ≥ τ2 and S4 completeness ≥ τ2 |
| L3 | Broaden k, draft-critic-revise (max 2 rounds), majority vote over N=3 drafts | 10–15 | Always returns (final level) |

If a **budget guard** limit is hit at any level, the controller returns the best answer so far, flagged `budget_exhausted`.

## 4. Feedback signals

| ID | Signal | How computed | Range |
|---|---|---|---|
| S1 | Self-consistency | 3 samples at T=0.7, normalized answers, fraction agreeing with the mode | 0–1 |
| S2 | Retrieval grade | LLM grades each retrieved chunk relevant/irrelevant (JSON), signal = fraction relevant | 0–1 |
| S3 | Groundedness | LLM splits the answer into claims, checks each against evidence (JSON), signal = supported/total | 0–1 |
| S4 | Completeness | LLM lists the question's required parts and marks each answered, signal = answered/required | 0–1 |

All grader and checker calls use `temperature=0` and strict JSON output, with a parse-failure fallback of score 0 (forces escalation, the safe direction).

## 5. Shared state schema

```python
class State(TypedDict):
    question: str
    level: int                 # 0..3
    docs: list[dict]           # {id, text, score}
    subquestions: list[str]
    drafts: list[str]
    answer: str
    citations: list[str]
    signals: dict              # {"S1":..,"S2":..,"S3":..,"S4":..}
    calls: int                 # LLM calls used
    latency: float
    trace: list[dict]          # one entry per level run
    status: str                # "ok" | "budget_exhausted"
```

## 6. Data flow

1. Question enters the controller at level 0.
2. The level runs and produces `answer` and signals.
3. The controller compares the signals to that level's thresholds in `config.yaml`.
4. If they pass, or this is L3, or the budget is exhausted, it returns. Otherwise `level += 1` and the next level runs, reusing cached retrievals where possible.
5. The trace is appended to `results/trace.jsonl`.

## 7. Corpus and data

- **Dataset:** HotpotQA (distractor setting). Sample 150 questions: 50 dev and 100 test, fixed seed.
- **Corpus:** the union of context paragraphs from the sampled questions (gold plus distractors), chunked by paragraph. This avoids downloading Wikipedia.
- **Index:** FAISS flat index over bge-small embeddings, plus a BM25 index.

## 8. Configuration (`config.yaml`)

```yaml
llm: {provider: gemini, model: <model>, temperature: 0, timeout_s: 30, max_retries: 3}
retrieval: {k: 5, k_broad: 10, fusion: rrf}
levels:
  L0: {samples: 3, tau: 0.67}
  L1: {tau_retrieval: 0.5, tau_ground: 0.8}
  L2: {tau_ground: 0.8, tau_complete: 0.8}
  L3: {drafts: 3, max_revisions: 2}
budget: {max_calls_per_query: 20}
seed: 42
```

Thresholds start at these defaults and are tuned on the 50 dev questions only.

## 9. Design decisions

| Decision | Reason |
|---|---|
| One LLM, prompt-based graders | No training time today, and it is reproducible |
| Cache every LLM call | Allows cheap threshold sweeps and re-runs |
| Compute all signals for all levels once, then replay thresholds offline | Pareto curve without re-calling the LLM |
| Bounded loops | Guarantees termination and a predictable cost |
| Fail-safe parse fallback → escalate | A grader error never silently approves a bad answer |

## 10. Folder structure

```
feedbackrag/
  config.yaml
  src/
    llm.py  retriever.py  planner.py  generator.py
    signals.py  levels.py  controller.py  trace.py
  eval/
    prepare_data.py  run_baselines.py  run_feedbackrag.py
    sweep_thresholds.py  metrics.py  plots.py  ablation.py
  app/streamlit_app.py
  tests/  (see TEST_VERIFICATION.md)
  results/
  docs/ (PRD.md, HLD.md, PLAN.md, TEST_VERIFICATION.md)
```

## 11. Limitations (state these in the report)

- LLM graders can be wrong or biased towards their own output.
- A HotpotQA-derived corpus is small and easier than open-web retrieval.
- Results depend on the chosen LLM, and thresholds tuned on a 50-question dev split may not transfer.
