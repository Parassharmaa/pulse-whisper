"""PulseWhisperDecoder: Whisper with decoder head pulse injection.

Injects oscillatory signals into decoder self-attention K/V projections
to regularize hallucination-causing heads. Keeps all Whisper weights frozen,
trains only the pulse injection parameters.

Architecture: Frozen Whisper encoder → Frozen Whisper decoder with pulse
hooks on self-attention K/V at target (layer, head) pairs.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput

from pulse_whisper.models.decoder_pulse import DecoderPulseInjector


class PulseWhisperDecoder(nn.Module):
    """Whisper model with decoder self-attention pulse injection.

    Freezes all Whisper parameters and trains only DecoderPulseInjector
    parameters that modulate K/V in decoder self-attention heads.
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-tiny",
        target_layers: list[int] | None = None,
        target_heads: list[int] | None = None,
        n_frequencies: int = 16,
        alpha_init: float = 0.01,
        alpha_max: float | None = 0.1,
        use_phase_net: bool = False,
    ) -> None:
        super().__init__()
        self.whisper_model_name = whisper_model_name

        # Load Whisper
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        config = self.whisper.config

        # Clear max_length from generation config
        if hasattr(self.whisper, "generation_config"):
            self.whisper.generation_config.max_length = None

        # Freeze all Whisper parameters
        for param in self.whisper.parameters():
            param.requires_grad = False

        # Create decoder pulse injector
        self.decoder_pulse = DecoderPulseInjector(
            num_layers=config.decoder_layers,
            num_heads=config.decoder_attention_heads,
            head_dim=config.d_model // config.decoder_attention_heads,
            target_layers=target_layers,
            target_heads=target_heads,
            n_frequencies=n_frequencies,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            use_phase_net=use_phase_net,
        )

        # Install hooks on decoder self-attention modules
        self._install_hooks()

    def _install_hooks(self) -> None:
        """Install forward hooks on decoder self-attention to inject pulse."""
        self._hooks = []
        for layer_idx, layer in enumerate(self.whisper.model.decoder.layers):
            hook = layer.self_attn.register_forward_hook(
                partial(self._self_attn_hook, layer_idx=layer_idx)
            )
            self._hooks.append(hook)

    def _self_attn_hook(self, module, input, output, layer_idx: int):
        """Hook that modifies self-attention output by injecting pulse into K/V.

        This hook intercepts the attention output and re-computes it with
        pulse-modified K/V states. We access the cached K/V from the module's
        most recent forward pass.
        """
        # The hook fires after forward, but we need to inject DURING forward.
        # Instead, we'll use a different approach: wrap the k_proj and v_proj.
        pass

    def _remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _run_decoder_with_pulse(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run decoder manually, injecting pulse at self-attention K/V.

        We manually step through each decoder layer, intercepting the
        self-attention computation to inject pulse into K/V states.
        """
        decoder = self.whisper.model.decoder

        # Embedding
        positions = decoder.embed_positions.weight[:decoder_input_ids.shape[1]]
        hidden_states = decoder.embed_tokens(decoder_input_ids) + positions
        hidden_states = nn.functional.dropout(hidden_states, p=decoder.dropout, training=self.training)

        for layer_idx, layer in enumerate(decoder.layers):
            # === Self-attention with pulse injection ===
            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)

            # Manually compute Q, K, V
            self_attn = layer.self_attn
            bsz, tgt_len = hidden_states.shape[:2]

            query_states = self_attn.q_proj(hidden_states) * self_attn.scaling
            key_states = self_attn.k_proj(hidden_states)
            value_states = self_attn.v_proj(hidden_states)

            # Reshape to (batch, num_heads, seq_len, head_dim)
            query_states = query_states.view(bsz, tgt_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, tgt_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, tgt_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)

            # >>> PULSE INJECTION into K/V <<<
            key_states, value_states = self.decoder_pulse.inject(
                layer_idx, key_states, value_states
            )

            # Attention computation
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))

            # Causal mask
            causal_mask = torch.triu(
                torch.full((tgt_len, tgt_len), float("-inf"), device=hidden_states.device),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = nn.functional.dropout(attn_weights, p=self_attn.dropout if self.training else 0.0, training=self.training)

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, -1)
            attn_output = self_attn.out_proj(attn_output)

            hidden_states = nn.functional.dropout(attn_output, p=layer.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # === Cross-attention (unchanged, frozen) ===
            residual = hidden_states
            hidden_states = layer.encoder_attn_layer_norm(hidden_states)

            cross_attn = layer.encoder_attn
            q = cross_attn.q_proj(hidden_states) * cross_attn.scaling
            k = cross_attn.k_proj(encoder_hidden_states)
            v = cross_attn.v_proj(encoder_hidden_states)

            q = q.view(bsz, tgt_len, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)
            k = k.view(bsz, -1, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)
            v = v.view(bsz, -1, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)

            cross_attn_weights = torch.matmul(q, k.transpose(-2, -1))
            cross_attn_weights = nn.functional.softmax(cross_attn_weights, dim=-1)
            cross_output = torch.matmul(cross_attn_weights, v)
            cross_output = cross_output.transpose(1, 2).reshape(bsz, tgt_len, -1)
            cross_output = cross_attn.out_proj(cross_output)

            hidden_states = nn.functional.dropout(cross_output, p=layer.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # === Feed-forward (unchanged, frozen) ===
            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.activation_fn(layer.fc1(hidden_states))
            hidden_states = nn.functional.dropout(hidden_states, p=layer.activation_dropout, training=self.training)
            hidden_states = layer.fc2(hidden_states)
            hidden_states = nn.functional.dropout(hidden_states, p=layer.dropout, training=self.training)
            hidden_states = residual + hidden_states

        hidden_states = decoder.layer_norm(hidden_states)
        return hidden_states

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass: frozen encoder → decoder with pulse injection.

        Args:
            input_features: Mel spectrogram (batch, n_mels, seq_len).
            labels: Target token IDs for loss computation.
            decoder_input_ids: Decoder input IDs (auto-generated from labels if None).

        Returns:
            Dict with 'loss' and 'logits'.
        """
        # Run frozen encoder
        with torch.no_grad():
            encoder_outputs = self.whisper.model.encoder(input_features)
            encoder_hidden = encoder_outputs.last_hidden_state

        # Prepare decoder inputs
        if decoder_input_ids is None and labels is not None:
            decoder_start_id = self.whisper.config.decoder_start_token_id
            decoder_input_ids = labels.new_zeros(labels.shape)
            decoder_input_ids[:, 1:] = labels[:, :-1].clone()
            decoder_input_ids[:, 0] = decoder_start_id
            decoder_input_ids = decoder_input_ids.masked_fill(
                decoder_input_ids == -100, self.whisper.config.pad_token_id
            )

        # Run decoder with pulse injection
        decoder_hidden = self._run_decoder_with_pulse(decoder_input_ids, encoder_hidden)

        # LM head
        lm_logits = self.whisper.proj_out(decoder_hidden)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": lm_logits}

    @torch.no_grad()
    def generate(self, input_features: torch.Tensor, **generate_kwargs) -> torch.Tensor:
        """Generate text using pulse-modified decoder.

        For generation, we can't easily use the custom decoder loop with
        HF's generate(). Instead, we wrap the self-attention K/V projections
        to inject pulse during the standard generate() call.
        """
        # Install K/V projection wrappers for generation
        self._wrap_kv_projections()
        try:
            output = self.whisper.generate(
                input_features=input_features,
                **generate_kwargs,
            )
        finally:
            # Always restore original projections
            self._unwrap_kv_projections()
        return output

    def _wrap_kv_projections(self) -> None:
        """Temporarily wrap decoder self-attention K/V projections with pulse injection."""
        self._original_kv = {}
        for layer_idx, layer in enumerate(self.whisper.model.decoder.layers):
            if layer_idx not in self.decoder_pulse.target_layers:
                continue

            self_attn = layer.self_attn
            # Save originals
            self._original_kv[layer_idx] = {
                'k_proj_forward': self_attn.k_proj.forward,
                'v_proj_forward': self_attn.v_proj.forward,
            }

            # Create wrapped projections that inject pulse
            pulse_injector = self.decoder_pulse
            num_heads = self_attn.num_heads
            head_dim = self_attn.head_dim

            def make_wrapped_k(orig_k, orig_v, li, sa):
                def wrapped_k(hidden_states):
                    k_out = orig_k(hidden_states)
                    v_raw = orig_v(hidden_states)
                    bsz, seq_len = hidden_states.shape[:2]
                    k_reshaped = k_out.view(bsz, seq_len, sa.num_heads, sa.head_dim).transpose(1, 2)
                    v_reshaped = v_raw.view(bsz, seq_len, sa.num_heads, sa.head_dim).transpose(1, 2)
                    k_mod, _ = pulse_injector.inject(li, k_reshaped, v_reshaped)
                    return k_mod.transpose(1, 2).reshape(bsz, seq_len, -1)
                return wrapped_k

            def make_wrapped_v(orig_k, orig_v, li, sa):
                def wrapped_v(hidden_states):
                    k_raw = orig_k(hidden_states)
                    v_out = orig_v(hidden_states)
                    bsz, seq_len = hidden_states.shape[:2]
                    k_reshaped = k_raw.view(bsz, seq_len, sa.num_heads, sa.head_dim).transpose(1, 2)
                    v_reshaped = v_out.view(bsz, seq_len, sa.num_heads, sa.head_dim).transpose(1, 2)
                    _, v_mod = pulse_injector.inject(li, k_reshaped, v_reshaped)
                    return v_mod.transpose(1, 2).reshape(bsz, seq_len, -1)
                return wrapped_v

            orig_k = self_attn.k_proj.forward
            orig_v = self_attn.v_proj.forward
            self_attn.k_proj.forward = make_wrapped_k(orig_k, orig_v, layer_idx, self_attn)
            self_attn.v_proj.forward = make_wrapped_v(orig_k, orig_v, layer_idx, self_attn)

    def _unwrap_kv_projections(self) -> None:
        """Restore original K/V projections after generation."""
        for layer_idx, originals in self._original_kv.items():
            self_attn = self.whisper.model.decoder.layers[layer_idx].self_attn
            self_attn.k_proj.forward = originals['k_proj_forward']
            self_attn.v_proj.forward = originals['v_proj_forward']
        self._original_kv = {}

    def pulse_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable decoder pulse parameters."""
        return list(self.decoder_pulse.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_decoder_pulse_model(
    whisper_size: str = "tiny",
    target_layers: list[int] | None = None,
    target_heads: list[int] | None = None,
    n_frequencies: int = 16,
    alpha_init: float = 0.01,
    alpha_max: float | None = 0.1,
    use_phase_net: bool = False,
) -> PulseWhisperDecoder:
    """Factory to build a PulseWhisperDecoder."""
    model_name = f"openai/whisper-{whisper_size}"
    return PulseWhisperDecoder(
        whisper_model_name=model_name,
        target_layers=target_layers,
        target_heads=target_heads,
        n_frequencies=n_frequencies,
        alpha_init=alpha_init,
        alpha_max=alpha_max,
        use_phase_net=use_phase_net,
    )
