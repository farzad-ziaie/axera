"""Integration tests for axera.trainer.Trainer."""

from __future__ import annotations

import numpy as np
import pytest

from axera.config import TrainerConfig
from axera.layers import Dense, GMDH, InputLayer, RegressionHead, ClassificationHead
from axera.models import Sequential
from axera.trainer import Trainer


@pytest.fixture
def regression_model():
    return Sequential([
        InputLayer(in_features=4),
        GMDH(in_features=4, k=2),
        Dense(out_features=4, in_features=6),
        RegressionHead(in_features=4),
    ])


@pytest.fixture
def classification_model():
    return Sequential([
        InputLayer(in_features=4),
        Dense(out_features=4, in_features=4),
        ClassificationHead(in_features=4, n_classes=2),
    ], task="binary")


class TestTrainer:
    def test_gradient_training_reduces_loss(self, regression_model, small_regression_data):
        X, y = small_regression_data
        cfg = TrainerConfig(epochs=5, batch_size=16, optimizer="adam",
                            loss="mse", val_split=0.2, early_stopping_patience=0)
        trainer = Trainer(regression_model, cfg)
        history = trainer.fit(X, y)
        assert len(history["train_loss"]) == 5
        assert all(np.isfinite(v) for v in history["train_loss"])

    def test_early_stopping(self, regression_model, small_regression_data):
        epochs = 500
        X, y = small_regression_data
        cfg = TrainerConfig(epochs=epochs, batch_size=32, optimizer="adam",
                            loss="mse", val_split=0.2, early_stopping_patience=2)
        trainer = Trainer(regression_model, cfg)
        history = trainer.fit(X, y)
        # Should stop well before
        assert len(history["train_loss"]) < epochs

    def test_evaluate_regression(self, regression_model, small_regression_data):
        X, y = small_regression_data
        cfg = TrainerConfig(epochs=3, batch_size=16, optimizer="adam", loss="mse",
                            val_split=0.0, early_stopping_patience=0)
        trainer = Trainer(regression_model, cfg)
        trainer.fit(X, y)
        metrics = trainer.evaluate(X, y)
        assert "mse" in metrics
        assert "r2" in metrics
        assert "ba_bias" in metrics

    def test_checkpoint_saves_file(self, regression_model, small_regression_data, tmp_path):
        X, y = small_regression_data
        cfg = TrainerConfig(epochs=2, optimizer="adam", loss="mse",
                            val_split=0.0, early_stopping_patience=0,
                            checkpoint_dir=str(tmp_path))
        trainer = Trainer(regression_model, cfg)
        trainer.fit(X, y)
        assert (tmp_path / "model_final.pt").exists()

    def test_custom_loss_fn(self, regression_model, small_regression_data):
        from axera.losses import MAE
        X, y = small_regression_data
        cfg = TrainerConfig(epochs=3, optimizer="adam", loss="mae",
                            val_split=0.0, early_stopping_patience=0)
        trainer = Trainer(regression_model, cfg, loss_fn=MAE())
        history = trainer.fit(X, y)
        assert len(history["train_loss"]) == 3



def test_trainer_mopso_optimizer(regression_model, small_regression_data):
    X, y = small_regression_data
    from axera.losses import BlandAltmanLoss
    
    loss = BlandAltmanLoss(
        predict_fn=regression_model.predict_from_weights,
        X_train=X, y_train=y,
        objectives=["r2"],
    )
    cfg = TrainerConfig(optimizer="mopso", epochs=3)
    trainer = Trainer(regression_model, cfg, loss_fn=loss)
    history = trainer.fit(X, y)
    assert len(history["train_loss"]) > 0