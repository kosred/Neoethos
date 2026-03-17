"""
Python wrappers for Pure Rust models exposed via forex_bindings.
These ensure compatibility with the existing ExpertModel interface.
"""

from __future__ import annotations

import logging
from typing import Any

import forex_bindings as fb  # type: ignore
import numpy as np

from .base import ExpertModel

logger = logging.getLogger(__name__)


class RustExpertWrapper(ExpertModel):
    def __init__(self, rust_model: Any):
        self._model = rust_model

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        try:
            # We assume X and y are already coerced to the right formats by the trainer/dataset_manager
            # or we do it here if needed. Rust expects f64 for features and i32 for labels.
            from ..execution.frame_utils import frame_to_2d_float32

            x_np, cols = frame_to_2d_float32(X)
            y_np = np.asarray(y, dtype=np.int32).reshape(-1)
            self._model.fit(x_np.astype(np.float64), y_np, cols)
        except Exception as e:
            logger.error(f"Rust model fit failed: {e}")
            raise

    def predict_proba(self, X: Any) -> np.ndarray:
        try:
            from ..execution.frame_utils import frame_to_2d_float32

            x_np, cols = frame_to_2d_float32(X)
            return self._model.predict_proba(x_np.astype(np.float64), cols)
        except Exception as e:
            logger.error(f"Rust model predict_proba failed: {e}")
            return np.zeros((len(X), 3))

    def save(self, path: str) -> None:
        self._model.save(path)

    def load(self, path: str) -> None:
        self._model.load(path)


class XGBoostExpert(RustExpertWrapper):
    def __init__(self, params: dict | None = None):
        super().__init__(fb.XGBoostModel(params or {}))


class ElasticNetExpert(RustExpertWrapper):
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5):
        super().__init__(fb.ElasticNetModel(alpha, l1_ratio))


class BayesianLogitExpert(RustExpertWrapper):
    def __init__(self):
        super().__init__(fb.BayesianLogitModel())


class IsolationForestExpert(RustExpertWrapper):
    def __init__(self, n_trees: int = 100, sample_size: int = 256):
        super().__init__(fb.IsolationForestModel(n_trees, sample_size))


class GeneticStrategyExpert(RustExpertWrapper):
    def __init__(self, population: int = 50, generations: int = 10, max_indicators: int = 8):
        super().__init__(fb.GeneticModel(population, generations, max_indicators))


class SwarmForecasterExpert(RustExpertWrapper):
    def __init__(self, memory_limit_mb: float = 1024.0):
        super().__init__(fb.SwarmForecasterModel(memory_limit_mb))
