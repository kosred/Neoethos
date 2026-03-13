from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from forex_bot.execution.mt5_state_manager import BoundedLRUFeatureStore, MT5StateManager


class _Frame:
    def __init__(self, data: dict[str, list[float]], index: list[int] | None = None) -> None:
        self._data = {str(k): np.asarray(v, dtype=np.float64).reshape(-1) for k, v in data.items()}
        self.columns = list(self._data.keys())
        n_rows = len(next(iter(self._data.values()))) if self._data else 0
        self.index = np.asarray(index if index is not None else np.arange(n_rows), dtype=np.int64)

    @property
    def empty(self) -> bool:
        return int(self.index.size) <= 0

    def __len__(self) -> int:
        return int(self.index.size)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]


def _make_manager(path: Path) -> MT5StateManager:
    settings = SimpleNamespace(system=SimpleNamespace(symbol="EURUSD"))
    manager = MT5StateManager(mt5_connection=SimpleNamespace(), settings=settings)
    manager._entry_store_path = path
    manager.entry_feature_store = BoundedLRUFeatureStore(max_size=1000)
    return manager


def test_entry_features_persist_as_single_row_contract_and_reload_as_frame(tmp_path: Path) -> None:
    store_path = tmp_path / "entry_features_EURUSD.json"
    manager = _make_manager(store_path)
    bar_time = datetime(2026, 3, 13, 12, 0, tzinfo=UTC)
    features = _Frame({"adx": [18.0, 31.0], "rsi14": [55.0, 58.0]}, index=[100, 101])

    manager.record_entry_features(
        ticket=101,
        symbol="EURUSD",
        bar_time=bar_time,
        features=features,
        signal=1,
        order_ticket=101,
        deal_ticket=101,
        magic=101,
    )

    raw = json.loads(store_path.read_text())
    assert raw["101"]["features"] == {
        "kind": "feature_row_v1",
        "columns": ["adx", "rsi14"],
        "values": [31.0, 58.0],
    }

    restored = _make_manager(store_path)
    restored._load_entry_feature_store()

    async def _fake_recent_closed_deals(limit: int = 10) -> list[dict[str, object]]:
        del limit
        return [
            {
                "deal": 101,
                "order": 101,
                "magic": 101,
                "time": bar_time + timedelta(minutes=5),
                "symbol": "EURUSD",
                "profit": 12.5,
                "volume": 0.1,
                "price": 1.1,
            }
        ]

    restored.get_recent_closed_deals = _fake_recent_closed_deals  # type: ignore[method-assign]
    matched = asyncio.run(restored.get_recent_closed_with_features(limit=1))

    assert len(matched) == 1
    restored_features = matched[0]["features"]
    assert hasattr(restored_features, "columns")
    assert list(restored_features.columns) == ["adx", "rsi14"]
    assert np.allclose(np.asarray(restored_features["adx"]).reshape(-1), np.asarray([31.0], dtype=np.float64))
    assert np.allclose(np.asarray(restored_features["rsi14"]).reshape(-1), np.asarray([58.0], dtype=np.float64))
