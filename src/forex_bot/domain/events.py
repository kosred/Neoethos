from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Union

import numpy as np


@dataclass(slots=True)
class SignalResult:
    signal: int  # -1, 0, 1
    confidence: float
    model_votes: dict[str, float]
    regime: str
    meta_features: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    # HPC FIX: Use NumPy for zero-copy prob storage
    probs: np.ndarray | None = None  # [neutral, buy, sell] for the latest bar

    trade_probability: Any | None = None
    stacking_confidence: Any | None = None
    regimes: Any | None = None
    uncertainty: Any | None = None
    recommended_rr: Any | None = None
    recommended_sl: Any | None = None
    win_probability: Any | None = None
    signals: Any | None = None  # Full series for backtest


@dataclass(slots=True)
class PreparedDataset:
    # HPC FIX: Strict typing for sharded data
    X: Union[Any, np.ndarray]
    y: Union[Any, np.ndarray]
    index: Any  # DatetimeIndex
    feature_names: list[str]
    metadata: Any | None = None  # OHLCV for backtesting
    labels: Any | None = None  # For training convenience


@dataclass(slots=True)
class TradeEvent:
    symbol: str
    side: str  # 'buy' or 'sell'
    volume: float
    open_price: float
    open_time: datetime
    close_price: float | None = None
    close_time: datetime | None = None
    pnl: float | None = None
    commission: float = 0.0
    swap: float = 0.0
    comment: str = ""
    magic: int = 0


@dataclass(slots=True)
class RiskEvent:
    """Risk event for ledger tracking."""

    category: str
    message: str
    severity: str = "info"
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_id: str = field(init=False)

    def __post_init__(self):
        # UUIDv4 avoids collisions and is fast; truncate for compactness.
        # 16 hex chars = 64 bits; far lower collision risk than 12 (48 bits).
        self.event_id = uuid.uuid4().hex[:16]

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "message": self.message,
            "severity": self.severity,
            "context": self.context,
        }

