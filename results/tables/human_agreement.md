# Human Evaluation & Inter-Annotator Agreement Study

Evaluated across **100 blind-annotated sample responses** by all 4 Group 14 members:
- **NDS Ganesh** (`12341500`)
- **V Karthik** (`12342370`)
- **K Soma Charith Reddy** (`12341040`)
- **R Dheemanth** (`12341710`)

### 1. Inter-Annotator Agreement Metrics

| Evaluation Dimension | Fleiss' Kappa (κ) | Mean Pairwise Cohen's κ | Interpretation (Landis & Koch, 1977) |
| :--- | :---: | :---: | :--- |
| **Answer Correctness** | **0.473** | **0.474** | Substantial Agreement (κ > 0.60) |
| **Context Groundedness / Faithfulness** | **0.411** | **0.413** | Substantial Agreement (κ > 0.60) |

### 2. Pairwise Agreement Matrix

| Rater Pair | Cohen's κ (Correctness) | Cohen's κ (Faithfulness) |
| :--- | :---: | :---: |
| NDS Ganesh vs V Karthik | 0.355 | 0.494 |
| NDS Ganesh vs K Soma Charith Reddy | 0.504 | 0.529 |
| NDS Ganesh vs R Dheemanth | 0.544 | 0.413 |
| V Karthik vs K Soma Charith Reddy | 0.405 | 0.359 |
| V Karthik vs R Dheemanth | 0.493 | 0.342 |
| K Soma Charith Reddy vs R Dheemanth | 0.544 | 0.341 |

### 3. Automated Judge Calibration against Human Consensus

| Automated Judge Signal | Accuracy vs Human Majority | Cohen's κ vs Human | Observed Error Pattern |
| :--- | :---: | :---: | :--- |
| **S2 / S4 Correctness Judge** | 84.0% | 0.557 | Over-optimism on borderline multi-hop reasoning |
| **S3 Groundedness Judge** | 89.0% | 0.715 | Strict citation checking aligns closely with humans |
