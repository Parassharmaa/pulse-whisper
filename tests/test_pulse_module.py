"""Tests for PulseLayer and NoiseLayer."""

import torch
import pytest

from pulse_whisper.models.pulse_module import PulseLayer, NoiseLayer


@pytest.fixture
def hidden_size():
    return 64


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 10


@pytest.fixture
def sample_hidden(batch_size, seq_len, hidden_size):
    return torch.randn(batch_size, seq_len, hidden_size)


@pytest.fixture
def time_steps(seq_len):
    return torch.arange(seq_len, dtype=torch.float32)


class TestPulseLayer:
    def test_output_shape(self, sample_hidden, time_steps, hidden_size):
        layer = PulseLayer(hidden_size, alpha_init=0.01, use_phase_net=True)
        out = layer(sample_hidden, time_steps)
        assert out.shape == sample_hidden.shape

    def test_output_differs_from_input(self, sample_hidden, time_steps, hidden_size):
        layer = PulseLayer(hidden_size, alpha_init=0.01, use_phase_net=True)
        out = layer(sample_hidden, time_steps)
        assert not torch.allclose(out, sample_hidden, atol=1e-8)

    def test_without_phase_net(self, sample_hidden, time_steps, hidden_size):
        layer = PulseLayer(hidden_size, alpha_init=0.01, use_phase_net=False)
        out = layer(sample_hidden, time_steps)
        assert out.shape == sample_hidden.shape
        assert not torch.allclose(out, sample_hidden, atol=1e-8)

    def test_alpha_controls_magnitude(self, sample_hidden, time_steps, hidden_size):
        layer_small = PulseLayer(hidden_size, alpha_init=0.001, use_phase_net=False)
        layer_large = PulseLayer(hidden_size, alpha_init=1.0, use_phase_net=False)
        # Copy weights so only alpha differs
        layer_large.amplitude.data = layer_small.amplitude.data.clone()
        layer_large.omega.data = layer_small.omega.data.clone()

        diff_small = (layer_small(sample_hidden, time_steps) - sample_hidden).abs().mean()
        diff_large = (layer_large(sample_hidden, time_steps) - sample_hidden).abs().mean()
        assert diff_large > diff_small * 10

    def test_gradient_flow(self, sample_hidden, time_steps, hidden_size):
        layer = PulseLayer(hidden_size, alpha_init=0.01, use_phase_net=True)
        out = layer(sample_hidden, time_steps)
        loss = out.sum()
        loss.backward()
        assert layer.alpha.grad is not None
        assert layer.amplitude.grad is not None
        assert layer.omega.grad is not None

    def test_get_pulse_signal(self, sample_hidden, time_steps, hidden_size):
        layer = PulseLayer(hidden_size, alpha_init=0.01, use_phase_net=True)
        signal = layer.get_pulse_signal(sample_hidden, time_steps)
        assert signal.shape == sample_hidden.shape

    def test_time_varying(self, hidden_size):
        layer = PulseLayer(hidden_size, alpha_init=0.5, use_phase_net=False)
        h = torch.ones(1, 5, hidden_size)
        t = torch.arange(5, dtype=torch.float32)
        out = layer(h, t)
        # Different time steps should produce different outputs
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6)


class TestNoiseLayer:
    def test_output_shape(self, sample_hidden, hidden_size):
        layer = NoiseLayer(hidden_size, noise_scale_init=0.01)
        layer.train()
        out = layer(sample_hidden)
        assert out.shape == sample_hidden.shape

    def test_noise_in_train_mode(self, sample_hidden, hidden_size):
        layer = NoiseLayer(hidden_size, noise_scale_init=0.1)
        layer.train()
        out = layer(sample_hidden)
        assert not torch.allclose(out, sample_hidden, atol=1e-6)

    def test_no_noise_in_eval_mode(self, sample_hidden, hidden_size):
        layer = NoiseLayer(hidden_size, noise_scale_init=0.1)
        layer.eval()
        out = layer(sample_hidden)
        assert torch.allclose(out, sample_hidden)

    def test_accepts_time_steps(self, sample_hidden, time_steps, hidden_size):
        layer = NoiseLayer(hidden_size)
        layer.eval()
        out = layer(sample_hidden, time_steps)
        assert out.shape == sample_hidden.shape
