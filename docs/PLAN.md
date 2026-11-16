# PLAN: Build and Submit FeedbackRAG Today

Times are relative to start (T+0). Adjust to your actual deadline. Work in parallel after the first hour.

## Roles

| Person | Owns |
|---|---|
| **Ganesh** | Controller, levels wiring, config, budget guard, integration, final merge |
| **Karthik** | Data prep, corpus, BM25 + dense retriever, reranker, S2 retrieval grader |
| **Soma Charith** | LLM client + cache, generator, planner, S1/S3/S4 signals, Level 3 |
| **Dheemanth** | Eval harness, baselines, sweeps, plots, ablation, Streamlit UI, report assembly |

## Timeline

### T+0:00 – T+1:00 — Setup and contracts (everyone)
- [ ] Create the repo and folder structure from HLD.md; everyone clones it.
- [ ] Get API keys (or start Ollama) and test one call.
- [ ] Freeze the `State` schema and the function signatures (below).
- [ ] Ganesh: `config.yaml` and the controller skeleton with stub levels.
- [ ] Dheemanth: `prepare_data.py` creates `data/dev.json` (50) and `data/test.json` (100) with seed 42.

**Interface contract (do not change without telling the team):**
```python
llm.call(prompt, temperature=0, json=False) -> str          # cached, counts calls
retriever.search(query, k) -> list[{id, text, score}]
generator.answer(question, docs) -> {answer, citations}
signals.consistency(question, samples) -> float
signals.retrieval_grade(question, docs) -> float
signals.groundedness(answer, docs) -> float
signals.completeness(question, answer) -> float
levels.run_level(n, state) -> state
```

### T+1:00 – T+3:00 — Parallel build
| Who | Task | Done when |
|---|---|---|
| Soma | `llm.py` with cache and retry; `generator.py`; prompts for S1, S3, S4 | A sample question returns an answer and signal values |
| Karthik | Index build; hybrid search with RRF; S2 grader; (reranker last) | `search("...", 5)` returns sensible chunks |
| Ganesh | Level 0 and Level 1 wired into the graph; escalation logic | Run one question through L0→L1 with a trace |
| Dheemanth | `metrics.py` (EM, F1), `run_baselines.py`: no-retrieval and vanilla RAG | Baseline numbers on dev |

**Checkpoint A (T+3:00):** vanilla RAG and no-retrieval baselines produce EM/F1 on dev. If not, stop and fix before continuing.

### T+3:00 – T+4:30 — Levels 2 and 3, full integration
- [ ] Soma: planner (rewrite and decompose), Level 3 draft-critic-revise and vote.
- [ ] Karthik: iterative retrieval per sub-question, reranker.
- [ ] Ganesh: Level 2 and Level 3 in the controller; budget guard.
- [ ] Dheemanth: `run_feedbackrag.py` logs **all four signals at every level for every dev and test question** into `results/signals.jsonl`.

**Checkpoint B (T+4:30):** full pipeline runs end to end on 5 questions with traces.

### T+4:30 – T+5:30 — Experiments
- [ ] Run always-L2 and always-L3 baselines on dev and test (cached).
- [ ] `sweep_thresholds.py`: tune on dev, replay on test with cached signals, produce the Pareto plot.
- [ ] `ablation.py`: drop each of S1–S4, record the delta.
- [ ] Hand-check 20 grader outputs for sanity (record the result in the report).

### T+5:30 – T+6:30 — Packaging
- [ ] Streamlit demo (question box, answer, citations, trace per level).
- [ ] Report: Introduction, Related Work, Method (from HLD), Experiments, Results, Limitations.
- [ ] Slides (8–10): problem, idea (MLFQ analogy), architecture, signals, setup, results table, Pareto plot, ablation, limitations.
- [ ] README with run commands; record a 2-minute backup demo video.

### Last 30 minutes — Buffer
Run the test checklist in TEST_VERIFICATION.md, fix only blocking bugs, and submit.

## Cut order if time runs short
1. Drop the robustness experiment (FR13).
2. Drop the Level 3 majority vote (keep draft-critic-revise).
3. Drop the reranker.
4. Drop the Streamlit UI (use a CLI trace print and screenshots).

**Never cut:** Levels 0–2, the controller, the baselines, the Pareto plot, the trace log.

## Fallbacks
| Problem | Fallback |
|---|---|
| API rate limits | Use the cache, reduce the test set to 50, or switch to a local Qwen2.5-3B |
| Grader JSON parse fails | Strict JSON prompt, one retry, then score 0 |
| Controller bugs late | Run levels independently and compute escalation offline from cached signals (the sweep already does this) |
| Results do not support H1 | Report honestly, analyze which signals failed and why |

## Definition of done
- [ ] All Must requirements in PRD.md pass the checks in TEST_VERIFICATION.md.
- [ ] Results table, Pareto plot and ablation table are in `results/`.
- [ ] Report, slides and README are complete.
- [ ] Repo runs from a clean clone with the README steps.
