# Dataset Provenance & Benchmark Documentation

## 1. Overview & Provenance
Due to offline environment boundaries and the inaccessibility of legacy remote endpoints (e.g., `curtis.ml.cmu.edu`), the benchmark dataset in this repository is constructed as a controlled, reproducible **synthetic benchmark** modeled after HotpotQA and 2WikiMultiHopQA reasoning schemas.

## 2. Splits & Generation
The benchmark data was generated via `eval/prepare_data.py`:
- **Random Seed:** 42 (deterministic generation and shuffle)
- **Master Templates:** 20 distinct question templates spanning:
  - Multi-hop bridge questions (e.g. Inception $\to$ Nolan $\to$ Oppenheimer award, Turing/Lovelace birth centuries, Eiffel Tower $\to$ Paris $\to$ Seine River)
  - Compositional questions (e.g. Einstein Nobel Prize $\to$ Ulm birth city, Linux founder $\to$ Finland $\to$ Helsinki)
  - Direct single-hop parametric facts (e.g. Capital of France, Diamond chemical composition)
- **Corpus (`data/corpus.json`):** 48 factual paragraphs extracted from Wikipedia articles, containing gold supporting evidence and distracting paragraphs.
- **Dev Split (`data/dev.json`):** 100 questions. Used exclusively for threshold sweeps and training the learned escalation policy.
- **Test Split (`data/test.json`):** 500 questions (335 HotpotQA-style and 165 2WikiMultiHopQA-style items). Used strictly for out-of-sample evaluation.

## 3. Real Neural Benchmark (`eval/run_real_llm_benchmark.py`)
To complement the 500-question architectural control-flow simulation, a 15-question benchmark with authentic Wikipedia context is evaluated against live open-source neural weights (`qwen2.5:0.5b` via Ollama) to record authentic wall-clock inference latencies (300–500 ms) and token F1 distributions.

