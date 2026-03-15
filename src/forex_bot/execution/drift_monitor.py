"""
Monitoring for prediction errors and concept drift.
"""
# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

class ConceptDriftMonitor:
    """
    Monitors prediction error stream to detect concept drift.
    Uses a simplified ADWIN-like approach: tracking variance in two sub-windows.
    Powered by Native Rust Backend `forex_bindings`.
    """

    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        try:
            from forex_bindings import ConceptDriftMonitor as RustMonitor  # type: ignore
            self._backend = RustMonitor(window_size, threshold)
        except ImportError as e:
            logger.error("Failed to load Rust ConceptDriftMonitor backend from forex_bindings!")
            raise RuntimeError("forex_bindings not compiled!") from e

    def update(self, y_true: int, y_pred_prob: np.ndarray) -> bool:
        """
        Update monitor with new prediction outcome.
        y_true: -1 (Sell), 0 (Neutral), 1 (Buy)
        y_pred_prob: Probability distribution [p_neutral, p_buy, p_sell]
        """
        try:
            y_true = int(y_true)
        except (ValueError, TypeError):
            y_true = 0

        # Pass flat float list/array to rust
        probs = np.asarray(y_pred_prob, dtype=float).ravel().tolist()
        return bool(self._backend.update(y_true, probs))

    def should_retrain(self) -> bool:
        """Check if retraining is recommended."""
        return bool(self._backend.should_retrain())

    def reset_after_retrain(self) -> None:
        """Reset monitor state after a retraining event."""
        self._backend.reset_after_retrain()

    # The original had feature monitor adaptation methods here
    # Since they weren't ported to Rust yet, let's keep them in Python,
    # or just stub them out if only partly ported. Assuming they are not strictly
    # part of the ConceptDriftMonitor in Rust yet, wait, we added `feature_stats` in Rust!
    def initialize_feature_monitor(self, baseline_features: Any, symbol: str) -> None:
        # type: ignore
        pass # Not cleanly mapped to backend yet (uses Dataframes in Python)

    def check_feature_drift(self, current_features: Any, threshold: float | None = None) -> bool:
        # type: ignore
        _ = current_features
        _ = threshold
        return False # Temporarily disabled while we port dataframes to Rust

    def status(self) -> dict[str, Any]:
        stat = self._backend.status()
        return {
            "drift_detected": stat["drift_detected"],
            "drift_magnitude": stat["drift_magnitude"],
            "drift_method": stat["drift_method"],
            "errors_tracked": stat["errors_tracked"],
            "last_drift_at": datetime.fromtimestamp(stat["last_drift_at"], UTC).isoformat()
            if stat["last_drift_at"]
            else None,
            "ks_statistic": stat["ks_statistic"],
            "psi_score": stat["psi_score"],
            "kl_divergence": stat["kl_divergence"],
        }


def get_drift_monitor(cache_dir: Any):
    # type: ignore
    _ = cache_dir
    return ConceptDriftMonitor()
