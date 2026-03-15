from __future__ import annotations
import time as time_module
import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

_FEATURE_ROW_KIND = "feature_row_v1"

class BoundedLRUFeatureStore:
    """
    LRU cache with bounded size for entry feature storage (2025 best practice).
    Prevents memory leaks by automatically evicting least recently used entries.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.store: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.access_times: dict[int, float] = {}

    def add(self, ticket: int, features: dict[str, Any]) -> None:
        """Add entry to store, evicting LRU if at capacity."""
        if ticket in self.store:
            # Move to end (most recently used)
            self.store.move_to_end(ticket)
        else:
            if len(self.store) >= self.max_size:
                # Remove least recently used (first item)
                lru_ticket = next(iter(self.store))
                del self.store[lru_ticket]
                self.access_times.pop(lru_ticket, None)
                logger.debug(f"Evicted LRU entry {lru_ticket} from feature store")

            self.store[ticket] = features

        self.access_times[ticket] = time_module.monotonic()

    def get(self, ticket: int) -> dict[str, Any] | None:
        """Get entry and mark as recently used."""
        if ticket in self.store:
            self.store.move_to_end(ticket)
            self.access_times[ticket] = time_module.monotonic()
            return self.store[ticket]
        return None

    def remove(self, ticket: int) -> None:
        """Remove entry from store."""
        self.store.pop(ticket, None)
        self.access_times.pop(ticket, None)

    def items(self):
        """Iterate over all items."""
        return self.store.items()

    def __len__(self):
        return len(self.store)

    def __contains__(self, ticket: int):
        return ticket in self.store

    def cleanup_stale(self, max_age_seconds: int = 86400) -> int:
        """Remove entries older than max_age. Returns count of removed entries."""
        now = time_module.monotonic()
        to_remove = [
            ticket for ticket, access_time in self.access_times.items() if (now - access_time) > max_age_seconds
        ]
        for ticket in to_remove:
            self.remove(ticket)
        return len(to_remove)


class _FeatureRowFrame:
    """Single-row frame rehydrated from persisted entry features."""

    def __init__(self, columns: list[str], values: list[Any]) -> None:
        self.columns = [str(col) for col in list(columns or [])]
        self.index = np.asarray([0], dtype=np.int64)
        self._data: dict[str, np.ndarray] = {}
        for col, value in zip(self.columns, list(values or []), strict=False):
            scalar = 0.0 if value is None else float(value)
            self._data[str(col)] = np.asarray([scalar], dtype=np.float64)

    @property
    def empty(self) -> bool:
        return int(len(self.columns)) <= 0

    def __len__(self) -> int:
        return 1 if self.columns else 0

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]


@dataclass(slots=True)
class MT5Position:
    """Represents an actual MT5 position"""

    ticket: int
    symbol: str
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    swap: float
    commission: float
    time: datetime
    type: int  # 0=buy, 1=sell
    magic: int


@dataclass(slots=True)
class MT5Deal:
    """Represents a closed deal/trade"""

    deal: int
    order: int
    time: datetime
    symbol: str
    type: int
    entry: int
    volume: float
    price: float
    profit: float
    commission: float
    swap: float
    magic: int  # added magic for tracking


@dataclass(slots=True)
class OrderRequest:
    """Deduplication tracking for order requests"""

    request_id: str
    symbol: str
    order_type: str  # 'buy' or 'sell'
    volume: float
    sl: float | None
    tp: float | None
    timestamp: datetime
    bar_timestamp: datetime  # Current bar timestamp
    result: dict[str, Any] | None = None
    verified: bool = False
