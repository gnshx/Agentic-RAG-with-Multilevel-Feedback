import numpy as np
from typing import List, Tuple, Dict, Any

def bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes empirical bootstrap confidence interval for a metric:
    Returns (mean, ci_lower, ci_upper).
    """
    if not values:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))

    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    return round(mean_val, 4), round(ci_lower, 4), round(ci_upper, 4)

def bootstrap_paired_test(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 2000,
    seed: int = 42
) -> float:
    """
    Two-sided paired bootstrap hypothesis test:
    H0: Mean(scores_a) == Mean(scores_b)
    H1: Mean(scores_a) != Mean(scores_b)
    Returns two-sided p-value.
    """
    assert len(scores_a) == len(scores_b), "Paired test requires equal length score lists!"
    rng = np.random.RandomState(seed)

    diffs = np.array(scores_a, dtype=np.float64) - np.array(scores_b, dtype=np.float64)
    observed_diff = abs(float(np.mean(diffs)))
    centered_diffs = diffs - np.mean(diffs)

    n = len(diffs)
    boot_diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        boot_diffs[i] = np.mean(centered_diffs * signs)

    p_value = float(np.mean(np.abs(boot_diffs) >= observed_diff))
    return round(p_value, 5)

def format_metric_with_ci(mean: float, lower: float, upper: float, is_percent: bool = True) -> str:
    """Formats metric with 95% confidence interval."""
    if is_percent:
        return f"{mean * 100:.1f}% [{lower * 100:.1f}%, {upper * 100:.1f}%]"
    return f"{mean:.2f} [{lower:.2f}, {upper:.2f}]"

