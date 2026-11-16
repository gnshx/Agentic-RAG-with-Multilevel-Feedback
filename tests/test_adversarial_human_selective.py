"""
Unit tests for Adversarial Evaluation, Human Evaluation, and Selective Answering.
"""

import json
import os
import numpy as np
import pytest

from eval.adversarial_evaluation import (
    COUNTERFACTUAL_PROBES,
    PROMPT_INJECTION_PROBES,
    AdversarialEvaluator
)
from eval.human_evaluation import compute_fleiss_kappa, RATERS, ensure_enriched_dataset
from eval.selective_answering import (
    compute_multilevel_confidence,
    compute_risk_coverage_curve,
    compute_aurc
)
from src.llm import LLM


def test_adversarial_probe_counts_and_schema():
    assert len(COUNTERFACTUAL_PROBES) == 50
    assert len(PROMPT_INJECTION_PROBES) == 50

    for p in COUNTERFACTUAL_PROBES:
        assert "id" in p
        assert "question" in p
        assert "gold_answer" in p
        assert "gold_context" in p
        assert "distractor_text" in p
        assert "corrupted_target" in p

    for p in PROMPT_INJECTION_PROBES:
        assert "id" in p
        assert "question" in p
        assert "gold_answer" in p
        assert "gold_context" in p
        assert "injection_text" in p
        assert "injection_payload" in p


def test_adversarial_evaluator_signals():
    llm = LLM()
    evaluator = AdversarialEvaluator(llm=llm)

    sample_cf = COUNTERFACTUAL_PROBES[0]
    res_cf = evaluator.evaluate_signals_and_controller(sample_cf, is_injection=False)
    assert res_cf["s2_detected"] is True
    assert res_cf["s3_rejected"] is True
    assert res_cf["feedbackrag_defended"] is True

    sample_inj = PROMPT_INJECTION_PROBES[0]
    res_inj = evaluator.evaluate_signals_and_controller(sample_inj, is_injection=True)
    assert res_inj["s2_detected"] is True
    assert res_inj["s3_rejected"] is True
    assert res_inj["s4_rejected"] is True
    assert res_inj["feedbackrag_defended"] is True


def test_fleiss_kappa_math():
    # Perfect agreement
    perfect = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    kappa_perf = compute_fleiss_kappa(perfect)
    assert pytest.approx(kappa_perf, 0.01) == 1.0

    # Intermediate agreement
    mixed = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0]])
    kappa_mix = compute_fleiss_kappa(mixed)
    assert 0.0 < kappa_mix < 1.0


def test_human_eval_dataset_integrity():
    items = ensure_enriched_dataset()
    assert len(items) == 100

    expected_raters = {"NDS Ganesh", "V Karthik", "K Soma Charith Reddy", "R Dheemanth"}
    for item in items:
        assert "sample_id" in item
        assert "question" in item
        assert "gold_answer" in item
        r_corr = item.get("rater_correctness", {})
        assert expected_raters.issubset(set(r_corr.keys()))


def test_selective_answering_risk_coverage():
    # 10 queries, 8 correct, 2 errors
    # If confidences rank correct first, risk at 80% coverage should be 0.0
    confidences = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.2, 0.1])
    correctness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])

    covs, risks, accs = compute_risk_coverage_curve(confidences, correctness, [1.0, 0.8, 0.5])
    assert pytest.approx(risks[0], 0.01) == 0.20   # 100% coverage: 2/10 errors
    assert pytest.approx(risks[1], 0.01) == 0.00   # 80% coverage: 0 errors
    assert pytest.approx(risks[2], 0.01) == 0.00   # 50% coverage: 0 errors

    aurc = compute_aurc([0.5, 1.0], [0.0, 0.2])
    assert aurc > 0.0


def test_multilevel_confidence():
    # High confidence on grounded level 1 answer
    c_high = compute_multilevel_confidence(1, {"S2": 1.0, "S3": 1.0}, "Verified answer")
    assert c_high >= 0.90

    # Low confidence when groundedness is low
    c_low = compute_multilevel_confidence(1, {"S2": 0.5, "S3": 0.2}, "Uncertain answer")
    assert c_low < 0.50

    # Low confidence on explicit abstention
    c_abs = compute_multilevel_confidence(2, {"S2": 1.0, "S3": 1.0}, "I cannot find sufficient evidence")
    assert c_abs <= 0.15
