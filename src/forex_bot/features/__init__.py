"""Feature engineering and signal generation."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "FeatureEngineer",
    "SignalEngine",
    "TALIB_AVAILABLE",
    "TALibStrategyGene",
    "TALibStrategyMixer",
]

if TYPE_CHECKING:
    from .engine import SignalEngine
    from .pipeline import FeatureEngineer
    from .talib_mixer import TALIB_AVAILABLE, TALibStrategyGene, TALibStrategyMixer


def __getattr__(name: str) -> Any:
    if name == "FeatureEngineer":
        return getattr(import_module(".pipeline", __name__), name)
    if name == "SignalEngine":
        return getattr(import_module(".engine", __name__), name)
    if name in {"TALIB_AVAILABLE", "TALibStrategyGene", "TALibStrategyMixer"}:
        return getattr(import_module(".talib_mixer", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
