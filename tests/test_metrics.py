"""Tests for evaluation metrics."""

import pytest

from pulse_whisper.eval.metrics import (
    compute_cer,
    compute_hallucination_rate,
    compute_hallucination_severity,
    compute_wer,
    wer_degradation,
)


class TestComputeWER:
    def test_perfect(self):
        assert compute_wer(["hello world"], ["hello world"]) == 0.0

    def test_completely_wrong(self):
        wer = compute_wer(["foo bar"], ["hello world"])
        assert wer > 0

    def test_empty_references(self):
        assert compute_wer([], []) == 0.0

    def test_empty_reference_strings_filtered(self):
        wer = compute_wer(["hello", "world"], ["hello", ""])
        # Only one valid pair
        assert wer == 0.0


class TestComputeCER:
    def test_perfect(self):
        assert compute_cer(["hello"], ["hello"]) == 0.0

    def test_one_char_error(self):
        cer = compute_cer(["hallo"], ["hello"])
        assert 0 < cer < 1


class TestHallucinationRate:
    def test_no_hallucination(self):
        rate = compute_hallucination_rate(["", "", ""])
        assert rate == 0.0

    def test_all_hallucination(self):
        rate = compute_hallucination_rate(["hello world", "foo bar", "test"])
        assert rate == 1.0

    def test_partial_hallucination(self):
        rate = compute_hallucination_rate(["hello world", "", "test", ""])
        assert rate == 0.5

    def test_min_tokens(self):
        rate = compute_hallucination_rate(["a", "hello world there"], min_tokens=2)
        assert rate == 0.5


class TestHallucinationSeverity:
    def test_empty(self):
        assert compute_hallucination_severity([]) == 0.0

    def test_all_empty(self):
        assert compute_hallucination_severity(["", "", ""]) == 0.0

    def test_severity(self):
        sev = compute_hallucination_severity(["hello world", "foo bar baz", ""])
        assert sev == 2.5  # (2 + 3) / 2


class TestWERDegradation:
    def test_no_change(self):
        assert wer_degradation(0.1, 0.1) == 0.0

    def test_worse(self):
        deg = wer_degradation(0.1, 0.2)
        assert abs(deg - 1.0) < 1e-6  # 100% worse

    def test_better(self):
        deg = wer_degradation(0.2, 0.1)
        assert deg < 0  # negative = better

    def test_zero_baseline(self):
        assert wer_degradation(0.0, 0.1) == float("inf")
        assert wer_degradation(0.0, 0.0) == 0.0
