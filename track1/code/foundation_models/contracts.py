"""Model-independent contracts used by every foundation-model container."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SeriesInput:
    task_id: str
    origin: date
    history: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.history, dtype=np.float32)
        if values.ndim != 1 or len(values) < 28:
            raise ValueError("history must be a one-dimensional series with at least 28 days")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("history must contain finite nonnegative visitor counts")
        object.__setattr__(self, "history", values)


@dataclass(frozen=True)
class AdapterSettings:
    checkpoint: str | None
    checkpoint_revision: str | None
    source_revision: str
    device: str = "cuda"
    context_length: int | None = None
    batch_size: int = 32
    num_samples: int = 100
    seed: int = 42
    cache_dir: Path = Path("/model-cache")
    checkpoint_dir: Path = Path("/outputs/checkpoints")
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.source_revision) != 40 or any(c not in "0123456789abcdef" for c in self.source_revision.lower()):
            raise ValueError("source_revision must be a full 40-character Git commit SHA")
        if self.checkpoint and (
            not self.checkpoint_revision
            or len(self.checkpoint_revision) != 40
            or any(c not in "0123456789abcdef" for c in self.checkpoint_revision.lower())
        ):
            raise ValueError("checkpoint_revision must be a full 40-character revision SHA")
        if self.batch_size < 1 or self.num_samples < 1:
            raise ValueError("batch_size and num_samples must be positive")
        if self.context_length is not None and self.context_length < 28:
            raise ValueError("context_length must be at least 28 days")


@dataclass(frozen=True)
class AdapterOutput:
    raw_prediction: np.ndarray
    effective_context_days: int
    point_source: str
    quantiles: dict[str, np.ndarray] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def validate(self, horizon: int) -> "AdapterOutput":
        values = np.asarray(self.raw_prediction, dtype=float).reshape(-1)
        if len(values) != horizon or not np.isfinite(values).all():
            raise ValueError(f"adapter must return exactly {horizon} finite forecasts")
        if self.effective_context_days < 1:
            raise ValueError("effective_context_days must be positive")
        for name, item in self.quantiles.items():
            q = np.asarray(item, dtype=float).reshape(-1)
            if len(q) != horizon or not np.isfinite(q).all():
                raise ValueError(f"quantile {name} must contain {horizon} finite values")
        return AdapterOutput(values, self.effective_context_days, self.point_source,
                             self.quantiles, self.detail)


class FoundationAdapter(ABC):
    model_name: str
    training_mode: str

    def __init__(self, settings: AdapterSettings):
        self.settings = settings

    @abstractmethod
    def load(self) -> None:
        """Load code and weights. No target artifact is available in this process."""

    @abstractmethod
    def predict_batch(self, items: Sequence[SeriesInput], horizon: int) -> list[AdapterOutput]:
        """Return one result per input series in the same order."""

    def quantile_levels(self) -> list[float]:
        """Registered levels; runtime.extra.quantile_levels may add more (default 0.1, 0.5, 0.9)."""
        levels = self.settings.extra.get("quantile_levels", [0.1, 0.5, 0.9])
        return sorted({float(level) for level in levels} | {0.5})

    def effective_history(self, item: SeriesInput) -> np.ndarray:
        if self.settings.context_length is None:
            return item.history
        return item.history[-self.settings.context_length :]


def require_cuda(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on container
        raise RuntimeError("PyTorch is required for CUDA execution") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
