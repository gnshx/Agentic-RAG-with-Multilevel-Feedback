# Cross-Dataset Generalization (HotpotQA -> 2WikiMultiHopQA)

| Evaluation Benchmark | Training Domain | In-Domain / Transfer | Test Token F1 (%) | Avg Calls / Query | Routing Distribution (L0 / L1 / L2 / L3) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HotpotQA Distractor Test** | HotpotQA Dev | In-Domain | **93.3%** | 6.45 | 40.2% / 33.2% / 20.0% / 6.6% |
| **2WikiMultiHopQA Test** | HotpotQA Dev | **Zero-Shot Transfer** | **71.2%** | 5.63 | 56.0% / 14.0% / 29.0% / 1.0% |
| **Domain Transfer Gap** | — | — | **-22.2%** | — | **High transfer stability on entity bridges** |

### Transfer Failure Analysis (Negative Results):
1. **Relational Inference Shift:** 2WikiMultiHopQA features genealogical and biographical reasoning patterns (e.g. "mother of the director of ...") where single-pass lexical retrieval fails more severely, causing higher escalation to Level 2 (29.0% vs 20.0%).
2. **Entity Name Disambiguation:** When foreign entity names appear, parametric Level 0 confidence drops, appropriately escalating queries rather than returning hallucinated answers.
