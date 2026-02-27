"""Tests for statistical analysis utilities."""

import pytest

from pulse_whisper.analysis.stats import bootstrap_ci, cohens_d, paired_t_test, win_rate


class TestPairedTTest:
    def test_identical_samples(self):
        # ttest_rel returns NaN for zero-variance differences
        import math
        a = [1.0, 2.0, 3.0, 4.0]
        t_stat, p_val = paired_t_test(a, a)
        assert math.isnan(t_stat) or abs(t_stat) < 1e-10

    def test_different_samples(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [10.0, 20.0, 30.0, 40.0]
        t_stat, p_val = paired_t_test(a, b)
        assert p_val < 0.05


class TestCohensD:
    def test_identical(self):
        a = [1.0, 2.0, 3.0]
        assert abs(cohens_d(a, a)) < 1e-6

    def test_large_effect(self):
        a = [10.0, 20.0, 30.0]
        b = [1.0, 2.0, 3.0]
        d = cohens_d(a, b)
        assert abs(d) > 0.8  # large effect


class TestBootstrapCI:
    def test_returns_tuple(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = bootstrap_ci(data)
        assert lower < upper

    def test_ci_contains_mean(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = bootstrap_ci(data, ci=0.95)
        mean = sum(data) / len(data)
        assert lower <= mean <= upper


class TestWinRate:
    def test_all_wins(self):
        a = [0.1, 0.2, 0.3]
        b = [0.5, 0.6, 0.7]
        assert win_rate(a, b) == 1.0

    def test_no_wins(self):
        a = [0.5, 0.6, 0.7]
        b = [0.1, 0.2, 0.3]
        assert win_rate(a, b) == 0.0

    def test_ties(self):
        a = [0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5]
        assert win_rate(a, b) == 0.5
