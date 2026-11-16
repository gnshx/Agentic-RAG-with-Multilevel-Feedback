# Selective Answering & Risk-Coverage Performance

Evaluated across **200 queries** (150 answerable HotpotQA items + 50 unanswerable counter-factual probes):

| Coverage Level (%) | Answered Count | Empirical Risk (Error %) | Answered Accuracy (F1 %) | Operational Action |
| :---: | :---: | :---: | :---: | :--- |
| **100.0% (Full)** | 200 | 38.5% | 61.5% | Complete answering without confidence abstention |
| **80.0%** | 160 | 26.9% | 73.1% | Abstains on bottom 20% unconfident questions |
| **60.0%** | 120 | 19.2% | 80.8% | Balanced enterprise SLA threshold |
| **50.0%** | 100 | 17.0% | 83.0% | Strict mission-critical routing |
| **20.0%** | 40 | 12.5% | 87.5% | Maximum precision regime |

**Key Finding:** As coverage decreases from 100% to 50%, empirical error drops by **55.8%**, proving that the MLFQ confidence signals accurately isolate questions prone to hallucination or lack of supporting evidence.
