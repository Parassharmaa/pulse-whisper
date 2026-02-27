"""Evaluation metrics: WER, CER, hallucination rate.

Uses jiwer for word/character error rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jiwer

# Standard Whisper text normalization: lowercase, strip whitespace
_NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
])


@dataclass
class EvalResult:
    """Container for evaluation results at a single gap level."""
    gap_level: str
    wer: float
    cer: float
    num_samples: int
    predictions: list[str] = field(default_factory=list, repr=False)
    references: list[str] = field(default_factory=list, repr=False)


@dataclass
class HallucinationResult:
    """Container for hallucination test results."""
    input_type: str  # "silence", "white_noise", "speech_with_pauses"
    hallucination_rate: float  # fraction of inputs producing non-empty output
    avg_output_length: float  # average number of tokens in hallucinated output
    num_samples: int
    outputs: list[str] = field(default_factory=list, repr=False)


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Compute Word Error Rate with text normalization."""
    if not references:
        return 0.0
    filtered = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not filtered:
        return 0.0
    preds, refs = zip(*filtered)
    return jiwer.wer(list(refs), list(preds), reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Compute Character Error Rate with text normalization."""
    if not references:
        return 0.0
    filtered = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not filtered:
        return 0.0
    preds, refs = zip(*filtered)
    return jiwer.cer(list(refs), list(preds), reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)


def compute_hallucination_rate(outputs: list[str], min_tokens: int = 1) -> float:
    """Compute hallucination rate: fraction of outputs with actual text.

    For silence/noise inputs, any text output is a hallucination.

    Args:
        outputs: Model outputs from silence/noise inputs.
        min_tokens: Minimum word count to consider as hallucination.

    Returns:
        Fraction of outputs that are hallucinations.
    """
    if not outputs:
        return 0.0
    hallucinated = sum(1 for o in outputs if len(o.strip().split()) >= min_tokens)
    return hallucinated / len(outputs)


def compute_hallucination_severity(outputs: list[str]) -> float:
    """Average length (in words) of hallucinated outputs.

    Higher severity = model generates more fake text during silence.
    """
    lengths = [len(o.strip().split()) for o in outputs if o.strip()]
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def wer_degradation(wer_baseline: float, wer_gapped: float) -> float:
    """Compute WER degradation: how much worse the model gets with gaps.

    Returns relative increase: (gapped - baseline) / baseline.
    Positive = worse, negative = better.
    """
    if wer_baseline == 0.0:
        return float("inf") if wer_gapped > 0 else 0.0
    return (wer_gapped - wer_baseline) / wer_baseline
