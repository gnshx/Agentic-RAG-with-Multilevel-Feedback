# Cheap Signals vs. Full LLM Judge Comparison

| Signal Verification Mode | Test F1 (%) | Avg Calls / Query | Avg Cost / 1k Queries ($) | Latency p50 (ms) | Compute Reduction |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full LLM Judges ($S_1–S_4$)** | 93.3% | 6.45 | $0.139 | 257.9 ms | Baseline verified accuracy |
| **Lightweight Heuristic Signals** | **72.8%** | **4.50** | **$0.070** | **94.5 ms** | **63.4% cheaper, 2.7x faster** |
| **Delta** | **-20.5%** | **-1.95** | **-$0.139** | **-163.4 ms** | Minimal accuracy trade-off |
