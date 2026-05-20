"""Tests for axera.config — Pydantic v2 validation."""

from __future__ import annotations

import json

import pytest
from axera.config import (
    LIPConfig,
    ModelConfig,
    MOPSOConfig,
    TrainerConfig,
)


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.epochs == 100
        assert cfg.batch_size == 32
        assert cfg.device == "auto"

    def test_from_json(self, tmp_path):
        data = {"epochs": 50, "batch_size": 8, "optimizer": "lion"}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data))
        cfg = TrainerConfig.from_json(path)
        assert cfg.epochs == 50
        assert cfg.optimizer == "lion"

    def test_invalid_device(self):
        with pytest.raises(Exception):
            TrainerConfig(device="tpu")

    def test_valid_cuda_device(self):
        cfg = TrainerConfig(device="cuda:0")
        assert cfg.device == "cuda:0"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("AXERA_TRAINER_EPOCHS", "42")
        monkeypatch.setenv("AXERA_TRAINER_BATCH_SIZE", "64")
        cfg = TrainerConfig.from_env()
        assert cfg.epochs == 42
        assert cfg.batch_size == 64


class TestModelConfig:
    def test_in_features_required(self):
        with pytest.raises(Exception):
            ModelConfig()     # missing in_features

    def test_binary_n_classes_auto_corrected(self):
        cfg = ModelConfig(in_features=5, task="binary", n_classes=5)
        assert cfg.n_classes == 2

    def test_layer_list(self):
        cfg = ModelConfig(
            in_features=8,
            layers=[{"type": "GMDH", "k": 2}, {"type": "Dense", "units": 4}],
        )
        assert len(cfg.layers) == 2


class TestMOPSOConfig:
    def test_n_pop_bounds(self):
        with pytest.raises(Exception):
            MOPSOConfig(n_pop=5)   # below minimum of 10


class TestLIPConfig:
    def test_degree_bounds(self):
        with pytest.raises(Exception):
            LIPConfig(degree=0)
        with pytest.raises(Exception):
            LIPConfig(degree=9)
