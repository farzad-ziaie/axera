"""
Validated, typed configuration objects for every Axera component.

All configs are Pydantic v2 ``BaseModel`` subclasses.  They can be loaded
from dicts, JSON files, YAML files, or environment variables.

Environment variable naming convention:
  ``AXERA_<CLASS>_<FIELD>``  e.g. ``AXERA_TRAINER_EPOCHS=200``

Examples
--------
>>> cfg = TrainerConfig(epochs=200, batch_size=16, device="cuda")
>>> cfg.model_dump()
{'epochs': 200, 'batch_size': 16, ...}
>>> cfg = TrainerConfig.from_env()   # reads AXERA_TRAINER_* env vars
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ── LIP activation ────────────────────────────────────────────────────────────

class LIPConfig(BaseModel):
    """Configuration for the LIP polynomial activation."""
    degree: int = Field(2, ge=1, le=8, description="Polynomial degree (1–8)")
    bias: bool = True

    model_config = {"env_prefix": "AXERA_LIP_"}


# ── GMDH layer ────────────────────────────────────────────────────────────────

class GMDHConfig(BaseModel):
    """Configuration for a GMDH layer."""
    k: int = Field(2, ge=2, description="Number of inputs per neuron (k-wise combinations)")
    activation: str = "lip"
    degree: int = Field(2, ge=1, le=8)
    bias: bool = True

    model_config = {"env_prefix": "AXERA_GMDH_"}


# ── Dense layer ───────────────────────────────────────────────────────────────

class DenseConfig(BaseModel):
    """Configuration for a Dense layer."""
    units: int = Field(32, ge=1)
    activation: str = "lip"
    degree: int = Field(2, ge=1, le=8)
    bias: bool = True

    model_config = {"env_prefix": "AXERA_DENSE_"}


# ── Model ─────────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    """Top-level model configuration."""
    task: Literal["regression", "binary", "multiclass"] = "regression"
    in_features: int = Field(..., ge=1, description="Number of input features")
    n_classes: int = Field(1, ge=1, description="Number of output classes (ignored for regression)")
    normalize_input: Literal["none", "standard", "minmax"] = "standard"
    layers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Layer specs: [{'type': 'GMDH', 'k': 2, ...}, {'type': 'Dense', 'units': 8, ...}]",
    )

    @field_validator("n_classes")
    @classmethod
    def _validate_n_classes(cls, v: int, info: Any) -> int:
        task = info.data.get("task", "regression")
        if task == "binary" and v != 2:
            return 2
        return v

    model_config = {"env_prefix": "AXERA_MODEL_"}


# ── MOPSO optimizer ───────────────────────────────────────────────────────────

class MOPSOConfig(BaseModel):
    """Configuration for the MOPSO derivative-free optimizer."""
    n_pop: int          = Field(100, ge=10, le=5000, description="Swarm size")
    n_repo: int         = Field(50,  ge=5,  le=500,  description="Archive (repository) size")
    w: float            = Field(0.25, ge=0.0, le=1.0, description="Inertia weight")
    w_damp: float       = Field(0.998, ge=0.9, le=1.0, description="Inertia damping per iteration")
    c1: float           = Field(0.2, ge=0.0, le=2.0, description="Personal learning coefficient")
    c2: float           = Field(0.2, ge=0.0, le=2.0, description="Social learning coefficient")
    beta: float         = Field(0.1, ge=0.0, le=5.0, description="Leader selection pressure")
    gamma: float        = Field(0.1, ge=0.0, le=5.0, description="Archive deletion pressure")
    n_grid: int         = Field(100, ge=10, description="Grid resolution per objective")
    alpha: float        = Field(0.1, ge=0.0, le=1.0, description="Grid inflation rate")
    var_min: float      = -2.0
    var_max: float      =  2.0
    n_workers: int      = Field(1, ge=1, description="Parallel workers (1 = sequential)")

    model_config = {"env_prefix": "AXERA_MOPSO_"}


# ── Adam optimizer ────────────────────────────────────────────────────────────

class AdamConfig(BaseModel):
    """Configuration for the Adam gradient optimizer."""
    lr: float           = Field(0.001, gt=0)
    beta1: float        = Field(0.9,   ge=0.0, lt=1.0)
    beta2: float        = Field(0.999, ge=0.0, lt=1.0)
    eps: float          = Field(1e-8,  gt=0)
    weight_decay: float = Field(0.0,   ge=0)
    amsgrad: bool       = False

    model_config = {"env_prefix": "AXERA_ADAM_"}


# ── Trainer ───────────────────────────────────────────────────────────────────

class TrainerConfig(BaseModel):
    """Configuration for the Trainer."""
    epochs: int         = Field(100, ge=1)
    batch_size: int     = Field(32,  ge=1)
    device: str         = "auto"
    optimizer: str      = "adam"     # 'adam' | 'adamw' | 'mopso' | 'de'
    loss: str           = "logcosh"  # 'mse' | 'mae' | 'logcosh' | 'bland_altman'
    seed: int           = 42
    val_split: float    = Field(0.15, ge=0.0, lt=1.0)
    early_stopping_patience: int = Field(20, ge=0)
    log_every_n_steps: int       = Field(10, ge=1)
    amp: bool           = False      # automatic mixed precision (GPU only)
    compile_model: bool = False      # torch.compile (PyTorch 2.x)
    checkpoint_dir: Optional[str]    = None

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        allowed = {"auto", "cpu", "cuda", "mps"}
        if v not in allowed and not v.startswith("cuda:"):
            raise ValueError(f"device must be one of {allowed} or 'cuda:N'")
        return v

    @classmethod
    def from_env(cls) -> "TrainerConfig":
        """Load config from AXERA_TRAINER_* environment variables."""
        prefix = "AXERA_TRAINER_"
        data = {
            k[len(prefix):].lower(): v
            for k, v in os.environ.items()
            if k.startswith(prefix)
        }
        return cls(**data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TrainerConfig":
        """Load config from a JSON file."""
        with open(path) as f:
            return cls(**json.load(f))

    model_config = {"env_prefix": "AXERA_TRAINER_"}


# ── Inference ─────────────────────────────────────────────────────────────────

class InferenceConfig(BaseModel):
    """Configuration for inference / prediction calls."""
    batch_size: int = Field(256, ge=1)
    device: str     = "auto"
    amp: bool       = False

    model_config = {"env_prefix": "AXERA_INFER_"}


__all__ = [
    "LIPConfig", "GMDHConfig", "DenseConfig", "ModelConfig",
    "MOPSOConfig", "AdamConfig", "TrainerConfig", "InferenceConfig",
]
