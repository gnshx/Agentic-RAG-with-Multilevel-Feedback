import pytest
import os
import json
import numpy as np
from src.signals import retrieval_score_margin, lexical_groundedness_proxy
from src.conformal import (
    ConformalEscalationController,
    compute_ece,
    compute_signals_ece
)
from src.budget_policy import (
    BudgetDialPolicy,
    CostAwareBudgetPolicy,
    extract_budget_features,
    compute_oracle,
    compute_headroom,
    DEFAULT_LAMBDAS
)


# ==============================================================================
# SECTION 1: CHEAP SIGNALS TESTS
# ==============================================================================

def test_retrieval_score_margin_numeric():
    # Empty
    assert retrieval_score_margin([]) == 0.0
    # Single score
    assert retrieval_score_margin([0.045]) == 0.045
    # Two scores
    assert round(retrieval_score_margin([0.05, 0.03]), 4) == 0.0200
    # Unsorted multiple scores
    assert round(retrieval_score_margin([0.01, 0.08, 0.05]), 4) == 0.0300


def test_retrieval_score_margin_doc_dicts():
    docs = [
        {"id": "doc1", "score": 0.032},
        {"id": "doc2", "score": 0.025},
        {"id": "doc3", "score": 0.010}
    ]
    margin = retrieval_score_margin(docs)
    assert round(margin, 5) == 0.00700

    # Malformed / empty dicts handled safely
    assert retrieval_score_margin([{}, {}]) == 0.0


def test_lexical_groundedness_proxy_exact_and_disjoint():
    # Disjoint vocabulary
    assert lexical_groundedness_proxy("apples and oranges", "cars and bicycles") == 0.0
    # Empty input
    assert lexical_groundedness_proxy("", "any text") == 0.0
    assert lexical_groundedness_proxy("any claim", "") == 0.0

    # High overlap
    claim = "Marie Curie won two Nobel Prizes"
    context = "Marie Curie was a physicist who won two Nobel Prizes in physics and chemistry."
    score_jaccard = lexical_groundedness_proxy(claim, context, method="jaccard")
    score_overlap = lexical_groundedness_proxy(claim, context, method="overlap")

    assert 0.0 < score_jaccard <= 1.0
    assert 0.0 < score_overlap <= 1.0
    # Content word precision / overlap is typically higher than jaccard
    assert score_overlap >= score_jaccard


def test_lexical_groundedness_proxy_doc_dicts():
    claims = ["Newton discovered gravity", "Newton attended Cambridge"]
    docs = [
        {"id": "d1", "text": "Isaac Newton was an English mathematician who discovered universal gravity."},
        {"id": "d2", "text": "He studied at Trinity College, Cambridge."}
    ]
    score = lexical_groundedness_proxy(claims, docs, method="overlap")
    assert score > 0.50


# ==============================================================================
# SECTION 2: CONFORMAL PREDICTION & CALIBRATION TESTS
# ==============================================================================

def test_compute_ece_perfect_calibration():
    # Confidences match actual empirical accuracies perfectly
    confidences = [1.0] * 50 + [0.0] * 50
    accuracies = [1.0] * 50 + [0.0] * 50
    res = compute_ece(confidences, accuracies, n_bins=10)
    assert res["ece"] == 0.0
    assert res["mce"] == 0.0


def test_compute_ece_uncalibrated():
    # Overconfident: confidences = 1.0, but all answers are wrong (0.0)
    confidences = [1.0] * 50
    accuracies = [0.0] * 50
    res = compute_ece(confidences, accuracies, n_bins=10)
    assert round(res["ece"], 2) == 1.00
    assert round(res["mce"], 2) == 1.00


def test_compute_ece_empty():
    res = compute_ece([], [])
    assert res["ece"] == 0.0
    assert res["bins"] == []


def test_conformal_leakage_assertion():
    ctrl = ConformalEscalationController(target_alpha=0.10)
    with pytest.raises(AssertionError) as exc_info:
        ctrl.calibrate_on_dev([], split_name="test")
    assert "DATA LEAKAGE VIOLATION" in str(exc_info.value)


def test_conformal_calibration_and_risk_control():
    # Synthetic calibration set
    # Correct queries have high S1 (1.0), erroneous queries have low S1 (0.33)
    dev_records = [
        {"level": 0, "signals": {"S1": 1.0}, "f1": 1.0, "em": 1.0} for _ in range(30)
    ] + [
        {"level": 0, "signals": {"S1": 0.33}, "f1": 0.0, "em": 0.0} for _ in range(70)
    ]

    ctrl = ConformalEscalationController(target_alpha=0.05)
    ctrl.calibrate_on_dev(dev_records, split_name="dev")
    assert ctrl.is_calibrated

    # Threshold for level 0 should be 1.0 to avoid any errors
    assert ctrl.calibrated_thresholds[0] == 1.0

    # Test set evaluation
    test_records = [
        {"level": 0, "signals": {"S1": 1.0}, "f1": 1.0, "em": 1.0} for _ in range(20)
    ] + [
        {"level": 0, "signals": {"S1": 0.33}, "f1": 0.0, "em": 0.0} for _ in range(80)
    ]

    eval_res = ctrl.evaluate_coverage_and_risk(test_records, level=0)
    assert eval_res["coverage_pct"] == 20.0
    assert eval_res["empirical_error_rate_pct"] == 0.0
    assert eval_res["test_alpha_satisfied"] is True


def test_conformal_sweep_alphas():
    dev_records = [
        {"level": 0, "signals": {"S1": 1.0}, "f1": 1.0, "em": 1.0} for _ in range(40)
    ] + [
        {"level": 0, "signals": {"S1": 0.67}, "f1": 0.8, "em": 1.0} for _ in range(10)
    ] + [
        {"level": 0, "signals": {"S1": 0.33}, "f1": 0.0, "em": 0.0} for _ in range(50)
    ]

    test_records = dev_records.copy()
    sweep = ConformalEscalationController.calibrate_and_sweep_alphas(
        dev_records, test_records, alphas=[0.05, 0.10, 0.15], level=0
    )
    assert len(sweep) == 3
    for entry in sweep:
        assert entry["test_alpha_satisfied"] is True
        assert "coverage_pct" in entry
        assert "empirical_error_rate_pct" in entry


def test_compute_signals_ece_structure():
    records = {
        "q1": {
            0: {"signals": {"S1": 0.67}, "f1": 1.0, "em": 1.0},
            1: {"signals": {"S2": 0.5, "S3": 0.8}, "f1": 1.0, "em": 1.0},
            2: {"signals": {"S4": 0.75}, "f1": 1.0, "em": 1.0},
            3: {"signals": {}, "f1": 1.0, "em": 1.0}
        },
        "q2": {
            0: {"signals": {"S1": 0.33}, "f1": 0.0, "em": 0.0},
            1: {"signals": {"S2": 0.0, "S3": 0.2}, "f1": 0.0, "em": 0.0},
            2: {"signals": {"S4": 0.25}, "f1": 0.0, "em": 0.0},
            3: {"signals": {}, "f1": 0.0, "em": 0.0}
        }
    }
    ece_dict = compute_signals_ece(records)
    for sig in ["S1", "S2", "S3", "S4"]:
        assert sig in ece_dict
        assert "ece" in ece_dict[sig]
        assert "mce" in ece_dict[sig]


# ==============================================================================
# SECTION 3: BUDGET POLICY & ORACLE TESTS
# ==============================================================================

def test_budget_policy_leakage_assertion():
    policy = BudgetDialPolicy(lambda_dial=1000.0)
    with pytest.raises(AssertionError) as exc_info:
        policy.train_on_dev_signals({}, split_name="test")
    assert "DATA LEAKAGE VIOLATION" in str(exc_info.value)


def test_budget_policy_training_and_decision():
    # Build a minimal synthetic 2-query dev set
    dev_records = {
        "dev_q1": {
            0: {"signals": {"S1": 0.33}, "f1": 0.0, "em": 0.0, "calls": 3, "answer": "Wrong", "question": "Q1?", "docs": []},
            1: {"signals": {"S2": 0.5, "S3": 1.0}, "f1": 1.0, "em": 1.0, "calls": 3, "answer": "Right", "question": "Q1?", "docs": [{"score": 0.05, "text": "evidence"}]},
            2: {"signals": {"S3": 1.0, "S4": 1.0}, "f1": 1.0, "em": 1.0, "calls": 4, "answer": "Right", "question": "Q1?", "docs": []},
            3: {"signals": {}, "f1": 1.0, "em": 1.0, "calls": 10, "answer": "Right", "question": "Q1?", "docs": []}
        },
        "dev_q2": {
            0: {"signals": {"S1": 1.0}, "f1": 1.0, "em": 1.0, "calls": 3, "answer": "Correct", "question": "Q2?", "docs": []},
            1: {"signals": {"S2": 0.8, "S3": 1.0}, "f1": 1.0, "em": 1.0, "calls": 3, "answer": "Correct", "question": "Q2?", "docs": [{"score": 0.08, "text": "evidence"}]},
            2: {"signals": {"S3": 1.0, "S4": 1.0}, "f1": 1.0, "em": 1.0, "calls": 4, "answer": "Correct", "question": "Q2?", "docs": []},
            3: {"signals": {}, "f1": 1.0, "em": 1.0, "calls": 10, "answer": "Correct", "question": "Q2?", "docs": []}
        }
    }

    policy = CostAwareBudgetPolicy(lambda_dial=100.0)
    report = policy.train_on_dev_signals(dev_records, split_name="dev")
    assert policy.is_fitted
    assert report["training_samples"] == 6

    # Test prediction
    pred_gain = policy.predict_f1_gain(dev_records["dev_q1"][0], 0)
    assert isinstance(pred_gain, float)

    # Level 3 is terminal
    should_esc, gain, thresh = policy.should_escalate(dev_records["dev_q1"][3], 3)
    assert should_esc is False


def test_budget_policy_sweep_lambda():
    dev_records = {
        "q1": {
            0: {"signals": {"S1": 0.33}, "f1": 0.0, "em": 0.0, "calls": 3, "answer": "Wrong", "question": "Q1", "docs": []},
            1: {"signals": {"S2": 0.6, "S3": 0.9}, "f1": 1.0, "em": 1.0, "calls": 3, "answer": "Right", "question": "Q1", "docs": [{"score": 0.05, "text": "evidence"}]},
            2: {"signals": {"S3": 1.0, "S4": 1.0}, "f1": 1.0, "em": 1.0, "calls": 4, "answer": "Right", "question": "Q1", "docs": []},
            3: {"signals": {}, "f1": 1.0, "em": 1.0, "calls": 10, "answer": "Right", "question": "Q1", "docs": []}
        }
    }

    policy = BudgetDialPolicy(lambda_dial=10.0)
    policy.train_on_dev_signals(dev_records, split_name="dev")

    # Run sweep across default lambdas
    sweep = policy.sweep_lambda(dev_records, lambdas=[0.0, 100.0, 5000.0])
    assert len(sweep) == 3
    for s in sweep:
        assert "lambda" in s
        assert "avg_calls" in s
        assert "avg_cost_usd" in s
        assert "f1" in s


def test_oracle_and_headroom():
    records = {
        "q1": {
            0: {"f1": 0.0, "em": 0.0, "calls": 3, "cost_usd": 0.000030},
            1: {"f1": 1.0, "em": 1.0, "calls": 3, "cost_usd": 0.000030},
            2: {"f1": 1.0, "em": 1.0, "calls": 4, "cost_usd": 0.000040},
            3: {"f1": 1.0, "em": 1.0, "calls": 10, "cost_usd": 0.000100}
        },
        "q2": {
            0: {"f1": 1.0, "em": 1.0, "calls": 3, "cost_usd": 0.000030},
            1: {"f1": 1.0, "em": 1.0, "calls": 3, "cost_usd": 0.000030},
            2: {"f1": 1.0, "em": 1.0, "calls": 4, "cost_usd": 0.000040},
            3: {"f1": 1.0, "em": 1.0, "calls": 10, "cost_usd": 0.000100}
        }
    }

    oracle = compute_oracle(records)
    # Oracle picks level 1 for q1 (earliest level with f1=1.0) and level 0 for q2
    assert oracle["f1"] == 100.0
    assert oracle["per_query_choices"]["q1"] == 1
    assert oracle["per_query_choices"]["q2"] == 0

    policy_res = {"f1": 90.0, "em": 85.0, "avg_calls": 5.0, "avg_cost_usd": 0.000050}
    headroom = compute_headroom(policy_res, oracle)
    assert headroom["headroom_f1"] == 10.0
    assert headroom["headroom_em"] == 15.0
    assert headroom["headroom_f1_pct"] == 10.0


def test_budget_policy_save_and_load(tmp_path):
    dev_records = {
        "q1": {
            0: {"signals": {"S1": 0.33}, "f1": 0.0, "em": 0.0, "calls": 3, "answer": "A", "question": "Q", "docs": []},
            1: {"signals": {"S2": 0.6, "S3": 0.9}, "f1": 1.0, "em": 1.0, "calls": 3, "answer": "B", "question": "Q", "docs": []},
            2: {"signals": {"S3": 1.0, "S4": 1.0}, "f1": 1.0, "em": 1.0, "calls": 4, "answer": "C", "question": "Q", "docs": []},
            3: {"signals": {}, "f1": 1.0, "em": 1.0, "calls": 10, "answer": "D", "question": "Q", "docs": []}
        }
    }

    policy = BudgetDialPolicy(lambda_dial=500.0)
    policy.train_on_dev_signals(dev_records, split_name="dev")

    model_file = str(tmp_path / "budget_policy.pkl")
    policy.save(model_file)
    assert os.path.exists(model_file)

    loaded = BudgetDialPolicy.load(model_file)
    assert loaded is not None
    assert loaded.is_fitted
    assert loaded.lambda_dial == 500.0
