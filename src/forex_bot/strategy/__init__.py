"""Strategy helpers (Rust-accelerated backtesting)."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "fast_evaluate_strategy",
    "batch_evaluate_strategies",
    "infer_pip_metrics",
    "infer_sl_tp_pips_auto",
    "infer_stop_target_pips",
    "AutonomousDiscoveryEngine",
    "TensorDiscoveryEngine",
    "genetic",
]

if TYPE_CHECKING:
    from .discovery import AutonomousDiscoveryEngine
    from .discovery_tensor import TensorDiscoveryEngine
    from .fast_backtest import (
        batch_evaluate_strategies,
        fast_evaluate_strategy,
        infer_pip_metrics,
        infer_sl_tp_pips_auto,
    )
    from .stop_target import infer_stop_target_pips
    from . import genetic


def __getattr__(name: str) -> Any:
    if name in {
        "fast_evaluate_strategy",
        "batch_evaluate_strategies",
        "infer_pip_metrics",
        "infer_sl_tp_pips_auto",
    }:
        return getattr(import_module(".fast_backtest", __name__), name)
    if name == "infer_stop_target_pips":
        return getattr(import_module(".stop_target", __name__), name)
    if name == "AutonomousDiscoveryEngine":
        return getattr(import_module(".discovery", __name__), name)
    if name == "TensorDiscoveryEngine":
        return getattr(import_module(".discovery_tensor", __name__), name)
    if name == "genetic":
        return import_module(".genetic", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
