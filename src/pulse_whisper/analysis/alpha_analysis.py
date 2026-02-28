"""Alpha growth tracking and frequency analysis for pulse layers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PulseAnalysis:
    """Container for pulse layer analysis results."""
    layer_idx: int
    alpha: float
    amplitude_mean: float
    amplitude_std: float
    omega_mean: float
    omega_std: float
    omega_min: float
    omega_max: float


def extract_pulse_stats(model: nn.Module) -> list[PulseAnalysis]:
    """Extract pulse parameter statistics from a PulseWhisperEncoder.

    Args:
        model: PulseWhisperEncoder with injected pulse layers.

    Returns:
        List of PulseAnalysis, one per injected layer.
    """
    results = []

    if not hasattr(model, "injected_layers"):
        return results

    # Support both ModuleDict (keyed by layer index str) and ModuleList
    if hasattr(model.injected_layers, 'items'):
        layer_items = [(int(k), v) for k, v in model.injected_layers.items()]
    else:
        layer_items = list(enumerate(model.injected_layers))

    for i, layer in layer_items:
        if hasattr(layer, "alpha"):
            alpha = layer.alpha.item()
        else:
            alpha = 0.0

        if hasattr(layer, "amplitude"):
            amp = layer.amplitude.detach()
            amp_mean = amp.mean().item()
            amp_std = amp.std().item()
        else:
            amp_mean = amp_std = 0.0

        if hasattr(layer, "omega"):
            omega = layer.omega.detach()
            omega_mean = omega.mean().item()
            omega_std = omega.std().item()
            omega_min = omega.min().item()
            omega_max = omega.max().item()
        else:
            omega_mean = omega_std = omega_min = omega_max = 0.0

        results.append(PulseAnalysis(
            layer_idx=i,
            alpha=alpha,
            amplitude_mean=amp_mean,
            amplitude_std=amp_std,
            omega_mean=omega_mean,
            omega_std=omega_std,
            omega_min=omega_min,
            omega_max=omega_max,
        ))

    return results


def log_pulse_stats(model: nn.Module, step: int | None = None) -> None:
    """Log pulse parameter stats to logger."""
    stats = extract_pulse_stats(model)
    prefix = f"[step {step}] " if step is not None else ""

    for s in stats:
        logger.info(
            f"{prefix}Layer {s.layer_idx}: α={s.alpha:.6f}, "
            f"|A|={s.amplitude_mean:.4f}±{s.amplitude_std:.4f}, "
            f"ω={s.omega_mean:.2f}±{s.omega_std:.2f} [{s.omega_min:.2f}, {s.omega_max:.2f}]"
        )
