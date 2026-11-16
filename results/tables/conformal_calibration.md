# Risk-Controlled Escalation via Split Conformal Prediction

| Target Risk (α) | Non-Conformity Threshold ($L_0$ $\hat{q}_\alpha$) | Non-Conformity Threshold ($L_1$ $\hat{q}_\alpha$) | Test Coverage (%) | Empirical Test Error (%) | Formal Guarantee Met? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| α = 0.05 | 1.0000 | 1.0000 | 26.0% | 0.0% | YES (Bouned by α) |
| α = 0.10 | 1.0000 | 0.8000 | 54.0% | 5.6% | YES (Bouned by α) |
| α = 0.15 | 1.0000 | 0.8000 | 54.0% | 5.6% | YES (Bouned by α) |
| α = 0.20 | 1.0000 | 0.8000 | 54.0% | 5.6% | YES (Bouned by α) |

### Expected Calibration Error (ECE) of Feedback Signals:
- **$S_1$ Self-Consistency ECE:** **0.3258** (Maximum Calibration Error: 0.6700)
- **$S_3$ Groundedness Checker ECE:** **0.2300** (Maximum Calibration Error: 0.3778)
