"""Statistical analysis: paired t-tests, Cohen's d, bootstrap CIs, win rates."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t-test between two sets of measurements.

    Returns (t_statistic, p_value).
    """
    t_stat, p_val = stats.ttest_rel(a, b)
    return float(t_stat), float(p_val)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size for paired samples."""
    a, b = np.array(a), np.array(b)
    diff = a - b
    return float(diff.mean() / max(diff.std(ddof=1), 1e-10))


def bootstrap_ci(
    data: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (lower, upper) bounds.
    """
    rng = np.random.RandomState(seed)
    data = np.array(data)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def win_rate(a: list[float], b: list[float]) -> float:
    """Fraction of cases where a < b (lower is better, e.g. WER)."""
    wins = sum(1 for x, y in zip(a, b) if x < y)
    ties = sum(1 for x, y in zip(a, b) if x == y)
    return (wins + 0.5 * ties) / max(1, len(a))
