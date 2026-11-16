import re
import string
from collections import Counter
from typing import Tuple

def normalize_answer(s: str) -> str:
    """Lower text and remove citations, punctuation, articles and extra whitespace."""
    def remove_citations(text: str) -> str:
        return re.sub(r"\[\d+\]", " ", text)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(remove_citations(lower(s)))))

def em(prediction: str, ground_truth: str) -> float:
    """
    Exact Match (EM) metric in [0.0, 1.0].
    Returns 1.0 if normalized prediction matches normalized ground truth, else 0.0.
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score in [0.0, 1.0].
    Computes token precision, recall, and harmonic mean between prediction and reference.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return 1.0 if prediction_tokens == ground_truth_tokens else 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1_score = (2 * precision * recall) / (precision + recall)
    return float(f1_score)
