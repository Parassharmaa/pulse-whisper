"""Tests for gap injection on mel spectrograms."""

import torch
import pytest

from pulse_whisper.data.gapped_audio import (
    GapLevel,
    create_gap_mask,
    inject_silence_gaps,
    random_gap_augmentation,
)


class TestCreateGapMask:
    def test_no_gap(self):
        mask = create_gap_mask(100, GapLevel.NONE, batch_size=2)
        assert mask.shape == (2, 100)
        assert mask.sum() == 0

    def test_small_gap(self):
        mask = create_gap_mask(100, GapLevel.SMALL)
        assert mask.shape == (1, 100)
        gap_frac = mask.float().mean().item()
        assert abs(gap_frac - 0.05) < 0.02

    def test_medium_gap(self):
        mask = create_gap_mask(100, GapLevel.MEDIUM)
        gap_frac = mask.float().mean().item()
        assert abs(gap_frac - 0.15) < 0.02

    def test_large_gap(self):
        mask = create_gap_mask(100, GapLevel.LARGE)
        gap_frac = mask.float().mean().item()
        assert abs(gap_frac - 0.30) < 0.02

    def test_multi_gap(self):
        mask = create_gap_mask(200, GapLevel.MULTI)
        gap_frac = mask.float().mean().item()
        assert 0.1 < gap_frac < 0.3  # roughly 20%

    def test_contiguous_gap(self):
        mask = create_gap_mask(100, GapLevel.MEDIUM)
        # Should be a single contiguous block
        transitions = (mask[0, 1:] != mask[0, :-1]).sum().item()
        assert transitions == 2  # one start, one end

    def test_reproducible_with_seed(self):
        m1 = create_gap_mask(100, GapLevel.MULTI, seed=42)
        m2 = create_gap_mask(100, GapLevel.MULTI, seed=42)
        assert torch.equal(m1, m2)

    def test_string_gap_level(self):
        mask = create_gap_mask(100, "gap_5")
        assert mask.shape == (1, 100)


class TestInjectSilenceGaps:
    def test_no_gap_unchanged(self):
        mel = torch.randn(2, 80, 100)
        gapped, mask = inject_silence_gaps(mel, GapLevel.NONE)
        assert torch.allclose(gapped, mel)
        assert mask.sum() == 0

    def test_gap_modifies_spectrogram(self):
        mel = torch.randn(2, 80, 100)
        gapped, mask = inject_silence_gaps(mel, GapLevel.MEDIUM)
        assert not torch.allclose(gapped, mel)

    def test_gap_values(self):
        mel = torch.ones(1, 80, 100)
        gapped, mask = inject_silence_gaps(mel, GapLevel.MEDIUM, silence_value=-1.0)
        # Gapped regions should be -1.0
        gap_region = gapped[0, :, mask[0]]
        assert torch.allclose(gap_region, torch.tensor(-1.0))

    def test_2d_input(self):
        mel = torch.randn(80, 100)
        gapped, mask = inject_silence_gaps(mel, GapLevel.SMALL)
        assert gapped.shape == (80, 100)

    def test_3d_input(self):
        mel = torch.randn(4, 80, 100)
        gapped, mask = inject_silence_gaps(mel, GapLevel.LARGE)
        assert gapped.shape == (4, 80, 100)
        assert mask.shape == (4, 100)


class TestRandomGapAugmentation:
    def test_output_shape(self):
        mel = torch.randn(2, 80, 100)
        aug = random_gap_augmentation(mel, [0.0, 0.05, 0.15])
        assert aug.shape == mel.shape

    def test_zero_fraction_possible(self):
        mel = torch.randn(2, 80, 100)
        # With only 0.0 in the list, should return unchanged
        aug = random_gap_augmentation(mel, [0.0])
        assert torch.allclose(aug, mel)
