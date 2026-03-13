from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from forex_bot.data import loader


def _write_symbol_parquet(
    root: Path,
    *,
    symbol: str = "EURUSD",
    timeframe: str = "M1",
    timestamp: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> None:
    target = root / f"symbol={symbol}" / f"timeframe={timeframe}"
    target.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "timestamp": pa.array(np.asarray(timestamp, dtype=np.int64)),
            "open": pa.array(np.asarray(open_, dtype=np.float64)),
            "high": pa.array(np.asarray(high, dtype=np.float64)),
            "low": pa.array(np.asarray(low, dtype=np.float64)),
            "close": pa.array(np.asarray(close, dtype=np.float64)),
            "volume": pa.array(np.asarray(volume, dtype=np.float64)),
        }
    )
    pq.write_table(table, target / "data.parquet")


def test_build_frame_from_arrays_returns_rust_frame() -> None:
    frame = loader._build_frame_from_arrays(
        {
            "open": np.array([1.0, 2.0], dtype=np.float64),
            "high": np.array([1.1, 2.1], dtype=np.float64),
            "low": np.array([0.9, 1.9], dtype=np.float64),
            "close": np.array([1.05, 2.05], dtype=np.float64),
        },
        index=np.array(["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[ns]"),
    )

    assert isinstance(frame, loader._RustFrame)
    assert list(frame.columns) == ["open", "high", "low", "close"]
    assert len(frame) == 2


def test_dead_strict_rust_data_mode_helper_is_removed() -> None:
    assert not hasattr(loader, "_strict_rust_data_mode_enabled")


def test_load_frames_skips_python_path_when_rust_backend_disabled(monkeypatch):
    settings = SimpleNamespace(
        system=SimpleNamespace(
            data_dir="missing_data_dir",
            base_timeframe="M1",
            multi_resolution_enabled=False,
            multi_resolution_timeframes=[],
            required_timeframes=[],
            higher_timeframes=[],
        )
    )
    dl = loader.DataLoader(settings)
    monkeypatch.setattr(loader, "_use_rust_data_backend", lambda: False)
    monkeypatch.setattr(
        dl,
        "_load_frames_rust",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Rust loader should not run when backend is disabled")),
    )
    out = dl._load_frames("EURUSD")
    assert out == {}


def test_load_frames_returns_empty_when_rust_loader_returns_nothing(monkeypatch):
    settings = SimpleNamespace(
        system=SimpleNamespace(
            data_dir="missing_data_dir",
            base_timeframe="M1",
            multi_resolution_enabled=False,
            multi_resolution_timeframes=[],
            required_timeframes=[],
            higher_timeframes=[],
        )
    )
    dl = loader.DataLoader(settings)
    monkeypatch.setattr(loader, "_use_rust_data_backend", lambda: True)
    monkeypatch.setattr(dl, "_load_frames_rust", lambda *_args, **_kwargs: {})
    out = dl._load_frames("EURUSD")
    assert out == {}


def test_load_frames_rust_builds_strict_frame_without_pandas(monkeypatch, tmp_path):

    def _fake_load_symbol_frames(*, root, symbol, timeframes, resample_missing, base_tf):
        assert symbol == "EURUSD"
        return {
            "M1": {
                "timestamp": np.array(
                    ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
                    dtype="datetime64[ns]",
                ),
                "open": np.array([1.10, 1.11, 1.12], dtype=np.float64),
                "high": np.array([1.11, 1.12, 1.13], dtype=np.float64),
                "low": np.array([1.09, 1.10, 1.11], dtype=np.float64),
                "close": np.array([1.105, 1.115, 1.125], dtype=np.float64),
            }
        }

    fake = types.SimpleNamespace(load_symbol_frames=_fake_load_symbol_frames)
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)

    settings = SimpleNamespace(
        system=SimpleNamespace(
            data_dir=str(tmp_path),
            base_timeframe="M1",
            multi_resolution_enabled=False,
            multi_resolution_timeframes=[],
            required_timeframes=[],
            higher_timeframes=[],
        )
    )
    dl = loader.DataLoader(settings)
    frames = dl._load_frames_rust("EURUSD")
    assert "M1" in frames
    frame = frames["M1"]
    assert not frame.empty
    assert set(["open", "high", "low", "close", "volume"]).issubset(set(frame.columns))
    assert len(frame) == 3
    assert frame.attrs.get("source") == "disk"


def test_canonical_dataset_loader_consumer_receives_sorted_deduped_rust_frames(monkeypatch, tmp_path):

    ts = np.array(
        [
            1_704_067_320_000_000_000,
            1_704_067_200_000_000_000,
            1_704_067_260_000_000_000,
            1_704_067_260_000_000_000,
            1_704_067_380_000_000_000,
        ],
        dtype=np.int64,
    )
    close = np.array([1.1040, 1.1000, 1.1020, 1.1025, 1.1060], dtype=np.float64)
    _write_symbol_parquet(
        tmp_path,
        timestamp=ts,
        open_=close - 0.0002,
        high=close + 0.0004,
        low=close - 0.0004,
        close=close,
        volume=np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64),
    )

    settings = SimpleNamespace(
        system=SimpleNamespace(
            data_dir=str(tmp_path),
            base_timeframe="M1",
            multi_resolution_enabled=False,
            multi_resolution_timeframes=[],
            required_timeframes=[],
            higher_timeframes=[],
        )
    )
    dl = loader.DataLoader(settings)
    frames = dl._load_frames_rust("EURUSD")

    assert "M1" in frames
    frame = frames["M1"]
    idx_ns = np.asarray(frame.index).astype("datetime64[ns]").astype(np.int64)
    assert idx_ns.ndim == 1
    assert np.all(idx_ns[1:] >= idx_ns[:-1])
    assert np.all(idx_ns[1:] != idx_ns[:-1])


def test_load_frames_uses_rust_payload_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "_use_rust_data_backend", lambda: True)

    def _fake_load_symbol_frames(*, root, symbol, timeframes, resample_missing, base_tf):
        return {
            "M1": {
                "timestamp": np.array([1704067200000, 1704067260000], dtype=np.int64),
                "open": np.array([1.0, 1.1], dtype=np.float64),
                "high": np.array([1.1, 1.2], dtype=np.float64),
                "low": np.array([0.9, 1.0], dtype=np.float64),
                "close": np.array([1.05, 1.15], dtype=np.float64),
                "volume": np.array([10.0, 11.0], dtype=np.float64),
            }
        }

    fake = types.SimpleNamespace(load_symbol_frames=_fake_load_symbol_frames)
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)

    settings = SimpleNamespace(
        system=SimpleNamespace(
            data_dir=str(tmp_path),
            base_timeframe="M1",
            multi_resolution_enabled=False,
            multi_resolution_timeframes=[],
            required_timeframes=[],
            higher_timeframes=[],
        )
    )
    dl = loader.DataLoader(settings)
    out = dl._load_frames("EURUSD")
    assert "M1" in out
    assert not out["M1"].empty


def test_resample_missing_from_available_supports_rust_frame(monkeypatch):
    idx = np.array(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T00:01:00",
            "2024-01-01T00:02:00",
            "2024-01-01T00:03:00",
            "2024-01-01T00:04:00",
            "2024-01-01T00:05:00",
        ],
        dtype="datetime64[ns]",
    )
    frame = loader._RustFrame(
        {
            "open": np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64),
            "high": np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6], dtype=np.float64),
            "low": np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float64),
            "close": np.array([1.05, 1.15, 1.25, 1.35, 1.45, 1.55], dtype=np.float64),
            "volume": np.array([10, 11, 12, 13, 14, 15], dtype=np.float64),
        },
        idx,
    )
    frame.attrs["source"] = "disk"
    frames = {"M1": frame}
    out = loader._resample_missing_from_available(frames, ["M1", "M5"])
    assert "M5" in out
    m5 = out["M5"]
    assert isinstance(m5, loader._RustFrame)
    assert len(m5) == 2
    np.testing.assert_allclose(np.asarray(m5["open"]), np.array([1.0, 1.5], dtype=np.float64))
    np.testing.assert_allclose(np.asarray(m5["close"]), np.array([1.45, 1.55], dtype=np.float64))
    np.testing.assert_allclose(np.asarray(m5["volume"]), np.array([60.0, 15.0], dtype=np.float64))


def test_resample_missing_from_available_supports_generic_frame_like(monkeypatch):

    class _GenericFrame:
        def __init__(self, data, index):
            self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
            self.columns = list(self._data.keys())
            self.index = np.asarray(index)
            self.attrs = {"source": "disk"}

        @property
        def empty(self):
            return len(self.index) == 0

        def __len__(self):
            return int(len(self.index))

        def __getitem__(self, key):
            return self._data[str(key)]

    idx = np.array(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T00:01:00",
            "2024-01-01T00:02:00",
            "2024-01-01T00:03:00",
            "2024-01-01T00:04:00",
            "2024-01-01T00:05:00",
        ],
        dtype="datetime64[ns]",
    )
    frame = _GenericFrame(
        {
            "open": np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64),
            "high": np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6], dtype=np.float64),
            "low": np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float64),
            "close": np.array([1.05, 1.15, 1.25, 1.35, 1.45, 1.55], dtype=np.float64),
            "volume": np.array([10, 11, 12, 13, 14, 15], dtype=np.float64),
        },
        idx,
    )

    out = loader._resample_missing_from_available({"M1": frame}, ["M1", "M5"])
    assert "M5" in out
    m5 = out["M5"]
    assert isinstance(m5, loader._RustFrame)
    assert len(m5) == 2
    np.testing.assert_allclose(np.asarray(m5["open"]), np.array([1.0, 1.5], dtype=np.float64))
    np.testing.assert_allclose(np.asarray(m5["close"]), np.array([1.45, 1.55], dtype=np.float64))
    np.testing.assert_allclose(np.asarray(m5["volume"]), np.array([60.0, 15.0], dtype=np.float64))


def test_mt5_adapter_get_live_frames_builds_rust_frame_without_pandas(monkeypatch):

    class _FakeMT5:
        TIMEFRAME_M1 = 1

        @staticmethod
        def copy_rates_from_pos(symbol, tf, pos, tail):
            assert symbol == "EURUSD"
            assert tf == 1
            assert pos == 0
            assert tail == 4
            return np.array(
                [
                    (1704067200, 1.10, 1.11, 1.09, 1.105, 12),
                    (1704067260, 1.11, 1.12, 1.10, 1.115, 15),
                ],
                dtype=[
                    ("time", "i8"),
                    ("open", "f8"),
                    ("high", "f8"),
                    ("low", "f8"),
                    ("close", "f8"),
                    ("tick_volume", "i8"),
                ],
            )

    monkeypatch.setitem(sys.modules, "MetaTrader5", _FakeMT5())
    settings = SimpleNamespace(
        system=SimpleNamespace(
            broker_backend="mt5_local",
            data_dir="unused",
        )
    )
    adapter = loader.MT5Adapter(settings)
    adapter._connected = True
    out = asyncio.run(adapter.get_live_frames("EURUSD", timeframes=["M1"], tail=4))
    assert "M1" in out
    frame = out["M1"]
    assert isinstance(frame, loader._RustFrame)
    np.testing.assert_allclose(np.asarray(frame["volume"]), np.array([12.0, 15.0], dtype=np.float64))
