# Live Neural LLM Benchmark Results (100 Multi-Hop HotpotQA Queries)

**Model:** `qwen2.5-coder:14b` on Remote NVIDIA GeForce RTX 3060 12GB GPU host (`10.50.10.52`)
**Corpus:** 2,826 real documents from `data/real_corpus.json` (BM25 + Dense RRF fusion)
**Evaluated Queries:** 100 authentic multi-hop questions with 95% Bootstrap Confidence Intervals ($B=1,000$)

| System Architecture | Exact Match (95% CI) | Token F1 (95% CI) | Avg Calls / Query | p50 Latency (ms) | Level Distribution (L0 / L1 / L2 / L3) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Always-L0** | 14.0% [8.0%, 21.0%] | 19.2% [12.4%, 26.2%] | 1.00 | 0.0 ms | — |
| **Always-L1** | 34.0% [24.0%, 42.0%] | 46.1% [36.6%, 54.9%] | 1.00 | 13.7 ms | — |
| **Always-L2** | 41.0% [32.0%, 50.0%] | 54.2% [44.5%, 62.8%] | 2.00 | 36.6 ms | — |
| **Always-L3** | 37.0% [28.0%, 46.0%] | 46.8% [37.4%, 55.3%] | 2.19 | 14.1 ms | — |
| **FeedbackRAG-Thresholds** | 36.0% [27.0%, 45.0%] | 49.0% [39.0%, 57.4%] | 5.22 | 1050.2 ms | 19.0% / 11.0% / 69.0% / 1.0% |
| **FeedbackRAG-Learned** | 37.0% [28.0%, 46.0%] | 49.6% [39.8%, 58.4%] | 5.26 | 48.7 ms | 19.0% / 10.0% / 69.0% / 2.0% |

### Statistical Significance (Paired Bootstrap, $B=2,000$):
- **Learned_vs_Thresholds:** {'p_em': 0.4785, 'p_f1': 0.691}
- **Learned_vs_AlwaysL1:** {'p_f1': 0.3735}
