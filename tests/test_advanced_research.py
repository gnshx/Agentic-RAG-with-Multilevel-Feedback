import pytest
import numpy as np
from src.budget_policy import BudgetDialPolicy
from src.conformal import ConformalEscalationController, compute_ece
from src.signals import retrieval_margin, lexical_groundedness_proxy

def test_budget_dial_utility_escalation():
    # High lambda -> cost penalty dominates -> do not escalate
    policy_frugal = BudgetDialPolicy(budget_lambda=50000.0)
    state = {"signals": {"S1": 0.33}, "question": "test question", "answer": "ans"}
    escalate, gain, thresh = policy_frugal.should_escalate(state, 0)
    assert not escalate
    assert gain < thresh

    # Low lambda -> pure accuracy -> escalate
    policy_aggressive = BudgetDialPolicy(budget_lambda=0.0)
    escalate_agg, gain_agg, thresh_agg = policy_aggressive.should_escalate(state, 0)
    assert escalate_agg
    assert gain_agg >= thresh_agg

def test_conformal_controller_nonconformity():
    controller = ConformalEscalationController(target_alpha=0.10)
    # Level 0: 1.0 - S1
    r0 = controller.compute_nonconformity({"S1": 1.0}, 0)
    assert r0 == 0.0
    r0_low = controller.compute_nonconformity({"S1": 0.33}, 0)
    assert abs(r0_low - 0.67) < 1e-4

    # Level 1: 1.0 - min(S2, S3)
    r1 = controller.compute_nonconformity({"S2": 0.8, "S3": 0.6}, 1)
    assert abs(r1 - 0.40) < 1e-4

def test_compute_ece():
    # Perfect calibration
    confidences = [0.1, 0.2, 0.8, 0.9]
    accuracies = [0.0, 0.0, 1.0, 1.0]
    res = compute_ece(confidences, accuracies, n_bins=5)
    assert "ece" in res and res["ece"] >= 0.0

def test_cheap_signals_margin_and_lexical():
    docs = [{"score": 0.8}, {"score": 0.4}]
    margin = retrieval_margin(docs)
    assert abs(margin - 0.5) < 1e-4

    docs_overlap = [{"text": "Paris is the capital of France."}]
    lex = lexical_groundedness_proxy("Paris, France", docs_overlap, method="overlap")
    assert lex > 0.5
