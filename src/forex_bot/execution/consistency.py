import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class ConsistencyMetrics:
    score: float
    daily_profit_consistency: float
    daily_trade_consistency: float
    daily_risk_consistency: float
    weekly_profit_consistency: float
    weekly_drawdown_consistency: float
    trade_size_consistency: float
    hold_time_consistency: float
    win_rate_rolling: float
    grade: str


class ConsistencyTracker:
    """
    Tracks trading consistency over a rolling window (prop-firm friendly).
    Powered by Native Rust Backend `forex-bindings`.
    """

    def __init__(self, cache_dir: Path, lookback_days: int = 30):
        try:
            from forex_bindings import ConsistencyTracker as RustTracker
            self._backend = RustTracker(str(cache_dir), lookback_days)
        except ImportError as e:
            logger.error("Failed to load Rust ConsistencyTracker backend from forex_bindings!")
            raise RuntimeError("forex_bindings not compiled!") from e

    def update(self, trade_event: dict) -> None:
        """
        trade_event expected keys: entry_time (iso), pnl, risk_pct, size, hold_minutes, win(bool/int)
        """
        try:
            self._backend.update(trade_event)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("ConsistencyTracker dropped trade: %s", e)

    def get_metrics(self) -> ConsistencyMetrics:
        # The Rust backend creates a standard Python dataclass instance
        return self._backend.get_metrics()
