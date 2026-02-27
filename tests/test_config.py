"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from pulse_whisper.training.config import ExperimentConfig, load_config


class TestLoadConfig:
    def test_defaults(self):
        config = load_config()
        assert config.model.whisper_size == "tiny"
        assert config.training.lr == 1e-3
        assert config.training.fp16 is True
        assert config.eval.test_set == "test-clean"

    def test_load_from_yaml(self, tmp_path):
        yaml_content = {
            "model": {"whisper_size": "small", "alpha_init": 0.05},
            "training": {"lr": 5e-4, "max_epochs": 20},
        }
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_config(config_path)
        assert config.model.whisper_size == "small"
        assert config.model.alpha_init == 0.05
        assert config.training.lr == 5e-4
        assert config.training.max_epochs == 20
        # Unchanged defaults
        assert config.model.freeze_whisper is True
        assert config.training.batch_size == 16

    def test_nonexistent_file(self):
        config = load_config("/nonexistent/path.yaml")
        assert config.model.whisper_size == "tiny"

    def test_empty_yaml(self, tmp_path):
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        config = load_config(config_path)
        assert config.model.whisper_size == "tiny"
