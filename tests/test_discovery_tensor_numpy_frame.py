from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forex_bot.strategy import discovery_tensor as dt


class _ArrayFrame:
    def __init__(self, data: dict[str, np.ndarray], index: np.ndarray, attrs: dict[str, str] | None = None) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return len(self.index) <= 0

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]

    def copy(self) -> "_ArrayFrame":
        return _ArrayFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            np.asarray(self.index).copy(),
            dict(self.attrs),
        )

    def tail(self, n: int) -> "_ArrayFrame":
        take = max(0, int(n))
        if take <= 0:
            return _ArrayFrame({k: v[:0] for k, v in self._data.items()}, self.index[:0], dict(self.attrs))
        return _ArrayFrame({k: v[-take:] for k, v in self._data.items()}, self.index[-take:], dict(self.attrs))


def test_discovery_rust_path_accepts_numpy_frame(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    rows = 80
    close = np.linspace(1.10, 1.20, rows, dtype=np.float64)
    frame = _ArrayFrame(
        {
            "open": close - 0.0001,
            "high": close + 0.0004,
            "low": close - 0.0004,
            "close": close,
            "volume": np.full(rows, 100.0, dtype=np.float64),
        },
        np.datetime64("2024-01-01T00:00:00")
        + np.arange(rows, dtype=np.int64) * np.timedelta64(1, "m"),
        attrs={"symbol": "EURUSD", "timeframe": "M1"},
    )

    calls: list[dict[str, np.ndarray | None]] = []

    class _DummyBindings:
        @staticmethod
        def search_discovery_ohlcv(open_, high, low, close_, ts, volume, *_args):
            calls.append(
                {
                    "open": np.asarray(open_, dtype=np.float64),
                    "high": np.asarray(high, dtype=np.float64),
                    "low": np.asarray(low, dtype=np.float64),
                    "close": np.asarray(close_, dtype=np.float64),
                    "ts": None if ts is None else np.asarray(ts, dtype=np.int64),
                    "volume": None if volume is None else np.asarray(volume, dtype=np.float64),
                }
            )
            return {
                "feature_names": ["RSI"],
                "rust_ranked": True,
                "portfolio": [
                    {
                        "indices": [0],
                        "weights": [1.0],
                        "fitness": 2.0,
                        "sharpe_ratio": 1.1,
                        "trades": 20.0,
                        "max_dd_pct": 0.02,
                    }
                ],
            }

    monkeypatch.setattr(dt, "_fb", _DummyBindings(), raising=False)
    monkeypatch.setattr(dt, "_RUST_DISCOVERY", True, raising=False)

    engine = dt.TensorDiscoveryEngine(device="cpu", n_experts=8, timeframes=["M1"], settings=None)
    engine.run_unsupervised_search({"M1": frame}, iterations=16)

    assert len(calls) == 1
    call = calls[0]
    np.testing.assert_allclose(call["close"], close)
    np.testing.assert_allclose(call["open"], close - 0.0001)
    np.testing.assert_allclose(call["high"], close + 0.0004)
    np.testing.assert_allclose(call["low"], close - 0.0004)
    assert call["ts"] is not None
    assert call["ts"].shape[0] == rows
    assert call["volume"] is not None
    assert call["volume"].shape[0] == rows

    out_path = Path("cache/talib_knowledge_EURUSD.json")
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload.get("symbol") == "EURUSD"
    assert payload.get("timeframe") == "M1"
    assert len(list(payload.get("best_genes") or [])) == 1
