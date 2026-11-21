# Cross-Model Scaling Evaluation on NVIDIA RTX A6000 48GB GPU

**Hardware:** NVIDIA RTX A6000 48GB VRAM (Host: `10.10.14.233`, 503GB RAM, Driver 580.126.20, CUDA 13.0)
**Benchmark Cohort:** 15 Factual & Multi-Hop Validation Queries across 6 Model Architectures (3B to 70B parameters)

| Model Architecture | Parameter Scale | VRAM Footprint | Strategy | Exact Match | Token F1 | Avg Latency | Avg Calls |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **llama3.2:3b** | 3.2B | 2.0 GB | Always-L0 (Parametric) | 46.7% | 71.3% | 654.2 ms | 1.00 |
| | | | Always-L1 (Hybrid RAG) | 33.3% | 40.0% | 442.6 ms | 1.00 |
| | | | **FeedbackRAG (Adaptive MLFQ)** | **46.7%** | **69.1%** | **389.1 ms** | **2.27** |
| **mistral:latest** | 7.2B | 4.4 GB | Always-L0 (Parametric) | 66.7% | 86.7% | 312.0 ms | 1.00 |
| | | | Always-L1 (Hybrid RAG) | 60.0% | 67.8% | 224.5 ms | 1.00 |
| | | | **FeedbackRAG (Adaptive MLFQ)** | **66.7%** | **86.7%** | **142.0 ms** | **2.00** |
| **llama3.1:latest** | 8.0B | 4.9 GB | Always-L0 (Parametric) | 60.0% | 71.1% | 606.5 ms | 1.00 |
| | | | Always-L1 (Hybrid RAG) | 6.7% | 6.7% | 474.5 ms | 1.00 |
| | | | **FeedbackRAG (Adaptive MLFQ)** | **53.3%** | **64.4%** | **393.4 ms** | **2.13** |
| **qwen3:8b** | 8.2B (Reasoning) | 5.2 GB | Always-L0 (Parametric) | 60.0% | 69.5% | 2104.7 ms | 1.00 |
| | | | Always-L1 (Hybrid RAG) | 0.0% | 0.5% | 1925.2 ms | 1.00 |
| | | | **FeedbackRAG (Adaptive MLFQ)** | **60.0%** | **68.9%** | **1738.6 ms** | **2.53** |
| **qwen3:14b** | 14.8B (Reasoning) | 9.3 GB | Always-L0 (Parametric) | 80.0% | 80.3% | 2662.7 ms | 1.00 |
| | | | Always-L1 (Hybrid RAG) | 0.0% | 0.7% | 2716.3 ms | 1.00 |
| | | | **FeedbackRAG (Adaptive MLFQ)** | **80.0%** | **80.0%** | **2600.7 ms** | **2.40** |
| **llama3.1:70b** | 70.6B (Frontier) | 42.5 GB | Always-L0 (Parametric) | 73.3% | 86.7% | 1902.4 ms | 1.00 |
| | | | Always-L1 (Hybrid RAG) | 6.7% | 15.6% | 2076.9 ms | 1.00 |
| | | | **FeedbackRAG (Adaptive MLFQ)** | **66.7%** | **80.0%** | **971.0 ms** | **2.13** |
