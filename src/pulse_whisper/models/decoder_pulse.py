"""Decoder Head Pulse: oscillatory injection into decoder self-attention.

Inspired by Calm-Whisper (Interspeech 2025) which showed specific decoder
self-attention heads cause 75%+ of hallucinations. Instead of fine-tuning
head weights, we inject learnable oscillatory signals into the attention
K/V projections at target heads, keeping all decoder weights frozen.

For Whisper-Tiny (4 layers × 6 heads), we target all heads and let
per-head alpha learn which ones benefit from pulse injection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecoderHeadPulse(nn.Module):
    """Oscillatory pulse injection for a single decoder attention head.

    Injects pulse signal into the key and value projections of a decoder
    self-attention head. The pulse modulates how the head attends,
    regularizing it during silence/noise inputs.

    Equation: K' = K + alpha_k * A_k * sin(omega * t + phi_k(K))
              V' = V + alpha_v * A_v * sin(omega * t + phi_v(V))
    """

    def __init__(
        self,
        head_dim: int,
        n_frequencies: int = 16,
        alpha_init: float = 0.01,
        alpha_max: float | None = 0.1,
        use_phase_net: bool = False,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.n_frequencies = n_frequencies
        self.alpha_max = alpha_max

        # Separate alpha for K and V modulation
        self.alpha_k = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_v = nn.Parameter(torch.tensor(alpha_init))

        # Shared frequency basis
        self.omega = nn.Parameter(torch.randn(n_frequencies) * 0.1 + 1.0)

        # Separate amplitude + projection for K and V
        self.amplitude_k = nn.Parameter(torch.randn(n_frequencies) * 0.01)
        self.amplitude_v = nn.Parameter(torch.randn(n_frequencies) * 0.01)
        self.proj_k = nn.Linear(n_frequencies, head_dim, bias=False)
        self.proj_v = nn.Linear(n_frequencies, head_dim, bias=False)

        # Optional phase nets
        self.use_phase_net = use_phase_net
        if use_phase_net:
            self.phase_net_k = nn.Sequential(
                nn.Linear(head_dim, n_frequencies),
                nn.Tanh(),
            )
            self.phase_net_v = nn.Sequential(
                nn.Linear(head_dim, n_frequencies),
                nn.Tanh(),
            )

        # Initialize projections small
        nn.init.normal_(self.proj_k.weight, std=0.01)
        nn.init.normal_(self.proj_v.weight, std=0.01)

    def _clamp_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        if self.alpha_max is not None:
            return alpha.clamp(max=self.alpha_max)
        return alpha

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject pulse into key and value states for one head.

        Args:
            key_states: (batch, seq_len, head_dim)
            value_states: (batch, seq_len, head_dim)
            time_steps: (seq_len,)

        Returns:
            Modulated (key_states, value_states).
        """
        # Base oscillation: (seq_len, n_freq)
        phase = torch.einsum('f,t->tf', self.omega, time_steps)

        # Key modulation
        phase_k = phase.unsqueeze(0).expand(key_states.shape[0], -1, -1)
        if self.use_phase_net:
            phase_k = phase_k + self.phase_net_k(key_states)
        osc_k = torch.sin(phase_k) * self.amplitude_k
        osc_k = self.proj_k(osc_k)
        alpha_k = self._clamp_alpha(self.alpha_k)
        key_out = key_states + alpha_k * osc_k

        # Value modulation
        phase_v = phase.unsqueeze(0).expand(value_states.shape[0], -1, -1)
        if self.use_phase_net:
            phase_v = phase_v + self.phase_net_v(value_states)
        osc_v = torch.sin(phase_v) * self.amplitude_v
        osc_v = self.proj_v(osc_v)
        alpha_v = self._clamp_alpha(self.alpha_v)
        value_out = value_states + alpha_v * osc_v

        return key_out, value_out


class DecoderPulseInjector(nn.Module):
    """Manages pulse injection across all target decoder heads.

    Creates DecoderHeadPulse modules for each (layer, head) pair and
    provides hooks to inject into the decoder forward pass.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        target_layers: list[int] | None = None,
        target_heads: list[int] | None = None,
        n_frequencies: int = 16,
        alpha_init: float = 0.01,
        alpha_max: float | None = 0.1,
        use_phase_net: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Default: all layers and all heads
        if target_layers is None:
            target_layers = list(range(num_layers))
        if target_heads is None:
            target_heads = list(range(num_heads))

        self.target_layers = set(target_layers)
        self.target_heads = set(target_heads)

        # Create pulse modules keyed by "layer_head"
        self.pulse_modules = nn.ModuleDict()
        for layer_idx in target_layers:
            for head_idx in target_heads:
                key = f"{layer_idx}_{head_idx}"
                self.pulse_modules[key] = DecoderHeadPulse(
                    head_dim=head_dim,
                    n_frequencies=n_frequencies,
                    alpha_init=alpha_init,
                    alpha_max=alpha_max,
                    use_phase_net=use_phase_net,
                )

    def inject(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject pulse into key/value states for a specific decoder layer.

        Args:
            layer_idx: Which decoder layer this is.
            key_states: (batch, num_heads, seq_len, head_dim)
            value_states: (batch, num_heads, seq_len, head_dim)

        Returns:
            Modified (key_states, value_states) with pulse injection at target heads.
        """
        if layer_idx not in self.target_layers:
            return key_states, value_states

        batch, num_heads, seq_len, head_dim = key_states.shape
        time_steps = torch.arange(seq_len, dtype=key_states.dtype, device=key_states.device)

        # Clone to avoid in-place modification of frozen model tensors
        key_out = key_states.clone()
        value_out = value_states.clone()

        for head_idx in self.target_heads:
            if head_idx >= num_heads:
                continue
            module_key = f"{layer_idx}_{head_idx}"
            if module_key not in self.pulse_modules:
                continue

            pulse = self.pulse_modules[module_key]
            # Extract per-head states: (batch, seq_len, head_dim)
            k_head = key_states[:, head_idx]
            v_head = value_states[:, head_idx]

            k_mod, v_mod = pulse(k_head, v_head, time_steps)
            key_out[:, head_idx] = k_mod
            value_out[:, head_idx] = v_mod

        return key_out, value_out

    def get_pulse_stats(self) -> list[dict]:
        """Extract alpha and frequency stats for all pulse modules."""
        stats = []
        for key, module in self.pulse_modules.items():
            layer_idx, head_idx = key.split("_")
            stats.append({
                "layer": int(layer_idx),
                "head": int(head_idx),
                "alpha_k": module.alpha_k.item(),
                "alpha_v": module.alpha_v.item(),
                "omega_mean": module.omega.detach().mean().item(),
                "omega_std": module.omega.detach().std().item(),
                "amp_k_mean": module.amplitude_k.detach().mean().item(),
                "amp_v_mean": module.amplitude_v.detach().mean().item(),
            })
        return stats
