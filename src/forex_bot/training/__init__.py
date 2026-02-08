"""Lazy exports for training components."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "MetaBlender",
    "HyperparameterOptimizer",
    "ModelTrainer",
]

if TYPE_CHECKING:
    from .ensemble import MetaBlender
    from .optimization import HyperparameterOptimizer
    from .trainer import ModelTrainer


def __getattr__(name: str) -> Any:
    if name == "MetaBlender":
        return getattr(import_module(".ensemble", __name__), name)
    if name == "HyperparameterOptimizer":
        return getattr(import_module(".optimization", __name__), name)
    if name == "ModelTrainer":
        return getattr(import_module(".trainer", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
