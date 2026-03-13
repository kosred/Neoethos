from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tests._compat_pd import pd

from forex_bot.core.config import Settings
from forex_bot.features import pipeline as plmod
from forex_bot.features.pipeline import FeatureEngineer


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


def test_prepare_rust_features_returns_numpy_in_pandas_free(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    monkeypatch.setenv("FOREX_BOT_RUST_FEATURES", "1")
    monkeypatch.setenv("FOREX_BOT_LABEL_TRIPLE_BARRIER", "0")
    monkeypatch.delenv("FOREX_BOT_BASE_SIGNAL", raising=False)

    n = 240
    ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    close = np.linspace(1.05, 1.15, n, dtype=np.float32)
    high = close + 0.0005
    low = close - 0.0005
    open_ = close - 0.0002
    features = np.column_stack(
        [
            np.linspace(25.0, 75.0, n, dtype=np.float32),
            np.linspace(-0.1, 0.1, n, dtype=np.float32),
            np.linspace(0.0, 1.0, n, dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    payload = {
        "feature_names": ["ta_rsi", "ta_macd_outmacdhist", "f0"],
        "features": features,
        "timestamps": ts,
        "base_timestamps": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }

    fake = types.SimpleNamespace(load_symbol_features=lambda **_kwargs: payload)
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)
    monkeypatch.setattr(plmod, "_RUST_FEATURES_BACKEND_OK", True, raising=False)
    monkeypatch.setattr(plmod, "_RUST_LABELS_BACKEND_OK", False, raising=False)

    fe = FeatureEngineer(Settings())
    ds = fe.prepare({}, symbol="EURUSD")

    assert isinstance(ds.X, np.ndarray)
    assert isinstance(ds.y, np.ndarray)
    assert ds.X.dtype == np.float32
    assert ds.y.dtype == np.int8
    assert ds.X.shape[0] == len(ds.y)
    assert ds.X.shape[1] == len(ds.feature_names)
    assert ds.X.shape[1] >= features.shape[1]
    assert isinstance(ds.index, np.ndarray)
    assert ds.index.dtype == np.int64


def test_canonical_dataset_binding_exposes_explicit_contract(tmp_path) -> None:
    import forex_bindings  # type: ignore

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

    payload = forex_bindings.load_symbol_features(
        root=str(tmp_path),
        symbol="EURUSD",
        base_tf="M1",
        higher_tfs=None,
        include_raw=True,
        cache_dir=None,
        cache_ttl_minutes=0,
        cache_enabled=False,
        resample_missing=False,
        arrow_tensor=False,
        feature_profile="core",
        htf_feature_profile=None,
        max_features=8,
        max_htf_features=0,
    )

    assert isinstance(payload, dict)
    assert "features" in payload
    assert "feature_names" in payload
    assert "labels" in payload
    assert "index_ns" in payload

    features = np.asarray(payload["features"])
    labels = np.asarray(payload["labels"])
    index_ns = np.asarray(payload["index_ns"])
    feature_names = list(payload["feature_names"])

    assert features.ndim == 2
    assert features.dtype == np.float32
    assert labels.ndim == 1
    assert labels.dtype == np.int8
    assert index_ns.ndim == 1
    assert index_ns.dtype == np.int64
    assert features.shape[0] == labels.shape[0] == index_ns.shape[0]
    assert features.shape[1] == len(feature_names)
    assert np.all(index_ns[1:] >= index_ns[:-1])
    assert np.all(index_ns[1:] != index_ns[:-1])

    market_metadata = payload.get("market_metadata")
    if market_metadata is not None:
        assert len(market_metadata) == index_ns.shape[0]


def test_align_series_by_ts_uses_rust_ffill_binding(monkeypatch) -> None:
    calls = {"ffill": 0}

    def _align(src_idx_ns, src_vals, tgt_idx_ns, fill):
        calls["ffill"] += 1
        assert float(fill) == 0.0
        return np.asarray([0.0, 1.0, 1.0, -1.0], dtype=np.float64)

    monkeypatch.setitem(sys.modules, "forex_bindings", types.SimpleNamespace(align_ffill_values_by_ns=_align))
    out = FeatureEngineer._align_series_by_ts(
        np.asarray([5, 10, 20, 30], dtype=np.int64),
        np.asarray([10, 30], dtype=np.int64),
        np.asarray([1.0, -1.0], dtype=np.float64),
        default=0.0,
        dtype=np.float32,
    )

    assert calls["ffill"] == 1
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 1.0, -1.0], dtype=np.float32))


def test_align_series_exact_by_ts_uses_rust_exact_binding(monkeypatch) -> None:
    calls = {"exact": 0}

    def _align(src_idx_ns, src_vals, tgt_idx_ns, fill):
        calls["exact"] += 1
        assert float(fill) == -2.0
        return np.asarray([-2.0, 1.0, -2.0, -1.0], dtype=np.float64)

    monkeypatch.setitem(sys.modules, "forex_bindings", types.SimpleNamespace(align_exact_values_by_ns=_align))
    out = FeatureEngineer._align_series_exact_by_ts(
        np.asarray([5, 10, 20, 30], dtype=np.int64),
        np.asarray([10, 30], dtype=np.int64),
        np.asarray([1.0, -1.0], dtype=np.float64),
        default=-2.0,
        dtype=np.float32,
    )

    assert calls["exact"] == 1
    np.testing.assert_allclose(out, np.array([-2.0, 1.0, -2.0, -1.0], dtype=np.float32))


def test_align_series_by_ts_fallback_prefers_rust_sorted_index_order(monkeypatch) -> None:
    calls = {"sort": 0}

    def _sorted_index_order(idx_ns):
        calls["sort"] += 1
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setitem(sys.modules, "forex_bindings", types.SimpleNamespace(sorted_index_order=_sorted_index_order))

    out = FeatureEngineer._align_series_by_ts(
        np.asarray([10, 25, 30], dtype=np.int64),
        np.asarray([30, 10, 20], dtype=np.int64),
        np.asarray([3.0, 1.0, 2.0], dtype=np.float64),
        default=0.0,
        dtype=np.float32,
    )

    assert calls["sort"] == 1
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_align_series_exact_by_ts_fallback_prefers_rust_sorted_index_order(monkeypatch) -> None:
    calls = {"sort": 0}

    def _sorted_index_order(idx_ns):
        calls["sort"] += 1
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setitem(sys.modules, "forex_bindings", types.SimpleNamespace(sorted_index_order=_sorted_index_order))

    out = FeatureEngineer._align_series_exact_by_ts(
        np.asarray([10, 25, 30], dtype=np.int64),
        np.asarray([30, 10, 20], dtype=np.int64),
        np.asarray([3.0, 1.0, 2.0], dtype=np.float64),
        default=0.0,
        dtype=np.float32,
    )

    assert calls["sort"] == 1
    np.testing.assert_allclose(out, np.array([1.0, 0.0, 3.0], dtype=np.float32))


def test_score_to_discovered_signal_prefers_rust_rank_scores_desc(monkeypatch) -> None:
    calls = {"rank": 0}

    def _rank_scores_desc(scores, absolute=False):
        calls["rank"] += 1
        assert bool(absolute) is False
        np.testing.assert_allclose(np.asarray(scores, dtype=np.float64), np.array([0.2, 0.9, 0.5, 0.8], dtype=np.float64))
        return np.array([1, 3, 2, 0], dtype=np.int64)

    monkeypatch.setitem(sys.modules, "forex_bindings", types.SimpleNamespace(rank_scores_desc=_rank_scores_desc))

    out = FeatureEngineer._score_to_discovered_signal(
        np.array([0.2, -0.9, 0.5, 0.8], dtype=np.float64),
        threshold=0.1,
        min_coverage=0.0,
        max_coverage=0.5,
    )

    assert calls["rank"] == 1
    np.testing.assert_array_equal(out, np.array([0, -1, 0, 1], dtype=np.int8))


def test_rust_only_disables_python_frame_fallback(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setenv("FOREX_BOT_RUST_FEATURES", "1")

    fake = types.SimpleNamespace(load_symbol_features=lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)
    monkeypatch.setattr(plmod, "_RUST_FEATURES_BACKEND_OK", True, raising=False)

    fe = FeatureEngineer(Settings())
    dummy_frame = types.SimpleNamespace(attrs={})
    ds = fe.prepare({"M1": dummy_frame}, symbol="EURUSD")

    assert isinstance(ds.X, np.ndarray)
    assert isinstance(ds.y, np.ndarray)
    assert ds.X.shape == (0, 0)
    assert ds.y.shape == (0,)


def test_rust_features_default_disables_python_frame_fallback(monkeypatch) -> None:
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.delenv("FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK", raising=False)
    monkeypatch.setenv("FOREX_BOT_RUST_FEATURES", "1")

    fake = types.SimpleNamespace(load_symbol_features=lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)
    monkeypatch.setattr(plmod, "_RUST_FEATURES_BACKEND_OK", True, raising=False)

    fe = FeatureEngineer(Settings())
    dummy_frame = types.SimpleNamespace(attrs={})
    ds = fe.prepare({"M1": dummy_frame}, symbol="EURUSD")

    assert isinstance(ds.X, np.ndarray)
    assert isinstance(ds.y, np.ndarray)
    assert ds.X.shape == (0, 0)
    assert ds.y.shape == (0,)


def test_rust_features_blocks_in_memory_frames_even_with_fallback_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("FOREX_BOT_TREE_BACKEND", raising=False)
    monkeypatch.delenv("FOREX_BOT_FEATURES_BACKEND", raising=False)
    monkeypatch.setenv("FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK", "1")
    monkeypatch.setenv("FOREX_BOT_RUST_FEATURES", "1")

    fake = types.SimpleNamespace(load_symbol_features=lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)
    monkeypatch.setattr(plmod, "_RUST_FEATURES_BACKEND_OK", True, raising=False)

    fe = FeatureEngineer(Settings())
    idx = pd.date_range("2025-01-01", periods=64, freq="min", tz="UTC")
    close = np.linspace(1.0, 1.2, len(idx), dtype=np.float64)
    frame = pd.DataFrame(
        {
            "open": close - 0.0001,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
            "volume": np.full(len(idx), 100.0, dtype=np.float64),
            "rsi": np.linspace(25.0, 75.0, len(idx), dtype=np.float64),
            "macd_hist": np.linspace(-0.2, 0.2, len(idx), dtype=np.float64),
        },
        index=idx,
    )

    ds = fe.prepare({"M1": frame}, symbol="EURUSD")

    assert isinstance(ds.X, np.ndarray)
    assert isinstance(ds.y, np.ndarray)
    assert ds.X.shape == (0, 0)
    assert ds.y.shape == (0,)


def test_prepare_rust_features_uses_discovered_signal_from_rust_bulk(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    monkeypatch.setenv("FOREX_BOT_RUST_FEATURES", "1")
    monkeypatch.setenv("FOREX_BOT_LABEL_TRIPLE_BARRIER", "0")
    monkeypatch.setenv("FOREX_BOT_BASE_SIGNAL_SOURCE", "discovery")

    n = 32
    ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    close = np.linspace(1.10, 1.20, n, dtype=np.float32)
    high = close + 0.0005
    low = close - 0.0005
    open_ = close - 0.0002
    features = np.column_stack(
        [
            np.full(n, 50.0, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.linspace(0.0, 1.0, n, dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    payload = {
        "feature_names": ["ta_rsi", "ta_macd_outmacdhist", "f0"],
        "features": features,
        "timestamps": ts,
        "base_timestamps": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }

    discovered = np.zeros(n, dtype=np.int8)
    discovered[::3] = 1
    discovered[2::3] = -1
    calls = {"bulk": 0}

    def _bulk(*_args, **_kwargs):
        calls["bulk"] += 1
        return discovered.reshape(-1, 1)

    fake = types.SimpleNamespace(load_symbol_features=lambda **_kwargs: payload, talib_bulk_signals_ohlcv=_bulk)
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)
    monkeypatch.setattr(plmod, "_RUST_FEATURES_BACKEND_OK", True, raising=False)
    monkeypatch.setattr(plmod, "_RUST_LABELS_BACKEND_OK", False, raising=False)

    fe = FeatureEngineer(Settings())
    gene = types.SimpleNamespace(
        indicators=["RSI"],
        params={},
        weights={"RSI": 1.0},
        long_threshold=0.3,
        short_threshold=-0.3,
        fitness=2.0,
    )
    monkeypatch.setattr(fe, "_load_discovered_base_signal_genes", lambda _symbol, max_genes=100: [gene])

    ds = fe.prepare({}, symbol="EURUSD")

    assert calls["bulk"] > 0
    assert "base_signal" in ds.feature_names
    base_idx = ds.feature_names.index("base_signal")
    got = np.asarray(ds.X[:, base_idx], dtype=np.int8)
    np.testing.assert_array_equal(got, discovered[: got.shape[0]])


def test_triple_barrier_labels_rust_only_blocks_python_fallback(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.setenv("FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK", "0")
    monkeypatch.setattr(plmod, "_RUST_LABELS_BACKEND_OK", False, raising=False)

    n = 64
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    close = pd.Series(np.linspace(1.1, 1.2, n, dtype=np.float64), index=idx)
    high = close + 0.0005
    low = close - 0.0005
    cfg = plmod._LabelConfig(
        horizon=8,
        min_dist=0.0,
        use_triple_barrier=True,
        max_hold=8,
        sl_pips=8.0,
        tp_pips=12.0,
    )

    fe = FeatureEngineer(Settings())
    out = fe._compute_labels(close, cfg, high=high, low=low, symbol="EURUSD", base_signal=None)

    vals = out.to_numpy(dtype=np.int8, copy=False)
    assert vals.shape[0] == n
    assert np.count_nonzero(vals) == 0


def test_triple_barrier_labels_python_fallback_opt_in_still_returns_neutral(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.setenv("FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK", "1")
    monkeypatch.setattr(plmod, "_RUST_LABELS_BACKEND_OK", False, raising=False)

    n = 64
    close = np.linspace(1.1, 1.2, n, dtype=np.float64)
    high = close + 0.0005
    low = close - 0.0005
    cfg = plmod._LabelConfig(
        horizon=8,
        min_dist=0.0,
        use_triple_barrier=True,
        max_hold=8,
        sl_pips=8.0,
        tp_pips=12.0,
    )

    fe = FeatureEngineer(Settings())
    out = fe._compute_labels_numpy(close, cfg, high=high, low=low, symbol="EURUSD", base_signal=None)

    assert out.shape[0] == n
    assert np.count_nonzero(out) == 0


def test_compute_labels_numpy_inputs_work_in_strict_pandas_free(monkeypatch) -> None:
    monkeypatch.setattr(plmod, "_RUST_LABELS_BACKEND_OK", False, raising=False)

    close = np.array([1.0, 2.0, 1.0, 3.0], dtype=np.float64)
    high = close + 0.01
    low = close - 0.01
    cfg = plmod._LabelConfig(
        horizon=1,
        min_dist=0.0,
        use_triple_barrier=False,
        max_hold=0,
        sl_pips=None,
        tp_pips=None,
    )

    fe = FeatureEngineer(Settings())
    out = fe._compute_labels(close, cfg, high=high, low=low, symbol="EURUSD", base_signal=np.array([1, 0, -1, 0]))

    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out.astype(np.int8), np.array([1, -1, 1, 0], dtype=np.int8))

