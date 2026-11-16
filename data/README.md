# FeedbackRAG Empirical Datasets & Provenance

This directory contains both the controlled state machine evaluation benchmark and the genuine multi-hop NLP corpora.

## 1. Genuine Academic Multi-Hop Benchmarks (Official Provenance)
Extracted directly from the official benchmark validation splits:
- **`data/real_dev.json` (100 questions):**
  - **Source:** Official HotpotQA Distractor validation split (`hotpotqa/hotpot_qa`, Stanford / CMU, Yang et al., 2018).
  - **Reasoning Type:** Multi-hop bridge questions and comparison questions with gold supporting paragraph annotations.
  - **Use:** Training and tuning of the MLFQ escalation policies, conformal calibration, and threshold tuning.
- **`data/real_test.json` (100 questions):**
  - **Source:** Held-out HotpotQA Distractor validation slice (`indices 100-200`).
  - **Use:** In-domain evaluation of Exact Match, Token F1, cost, calls, and empirical coverage. Strict anti-leakage guarantee: never seen during dev training.
- **`data/real_2wiki_test.json` (100 questions):**
  - **Source:** Official 2WikiMultiHopQA validation split (`voidful/2wikimultihopqa`, Ho et al., 2020).
  - **Use:** Cross-dataset out-of-domain generalization evaluation. Evaluates whether escalation policies trained on HotpotQA transfer to different entity relation topologies.
- **`data/real_corpus.json` (2,826 unique articles, 233,744 words):**
  - Unified multi-document Wikipedia knowledge base containing all gold supporting articles and high-similarity distractors.
  - Average paragraph length: 82.7 words.

## 2. Controlled State Machine Benchmark (Regime A)
- `data/dev.json` (100 questions) & `data/test.json` (500 questions) over `data/corpus.json` (48 reference chunks):
  - Used for isolated state machine graph verification, synthetic unit tests (U1–U12), and deterministic offline CI validation.
