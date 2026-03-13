from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from forex_bot.domain.events import PreparedDataset
from forex_bot.execution.training_service import TrainingService
from tests._compat_pd import pd


class _ArrayFrame:
    def __init__(self, data, index, attrs=None):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


def test_global_align_and_split_with_numpy_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOREX_BOT_GLOBAL_EVAL_FROM", raising=False)
    monkeypatch.setenv("FOREX_BOT_GLOBAL_EVAL_YEARS", "0")

    n = 1200
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    ds1 = PreparedDataset(
        X=np.random.default_rng(1).normal(size=(n, 2)).astype(np.float32),
        y=np.random.default_rng(2).integers(0, 3, size=n, dtype=np.int8),
        index=idx,
        feature_names=["a", "b"],
        metadata=None,
        labels=None,
    )
    ds2 = PreparedDataset(
        X=np.random.default_rng(3).normal(size=(n, 2)).astype(np.float32),
        y=np.random.default_rng(4).integers(0, 3, size=n, dtype=np.int8),
        index=idx,
        feature_names=["b", "c"],
        metadata=None,
        labels=None,
    )

    svc = object.__new__(TrainingService)
    cols, aligned = svc._align_global_feature_space([("EURUSD", ds1), ("GBPUSD", ds2)])

    assert cols == ["a", "b", "c"]
    assert len(aligned) == 2
    for _sym, ds in aligned:
        assert isinstance(ds.X, np.ndarray)
        assert ds.X.shape == (n, 3)
        assert isinstance(ds.y, np.ndarray)
        assert ds.y.shape == (n,)

    train_parts, eval_map, meta = svc._split_global_train_eval(
        aligned,
        train_ratio=0.8,
        embargo_bars=8,
        min_train_rows=100,
        min_eval_rows=50,
    )
    assert train_parts
    assert eval_map
    assert meta.get("cutoff_mode") == "ratio"
    for _sym, ds in train_parts:
        assert isinstance(ds.X, np.ndarray)
        assert isinstance(ds.y, np.ndarray)
        assert len(ds.X) == len(ds.y)


def test_global_align_dataframe_inputs_forced_numpy_output() -> None:
    n = 600
    idx = pd.to_datetime((np.arange(n, dtype=np.int64) + 1) * 60_000_000_000, utc=True)
    x1 = pd.DataFrame({"a": np.linspace(0.0, 1.0, num=n), "b": np.linspace(1.0, 2.0, num=n)}, index=idx)
    x2 = pd.DataFrame({"b": np.linspace(2.0, 3.0, num=n), "c": np.linspace(3.0, 4.0, num=n)}, index=idx)
    y1 = pd.Series(np.random.default_rng(1).integers(0, 3, size=n, dtype=np.int8), index=idx, dtype=np.int8)
    y2 = pd.Series(np.random.default_rng(2).integers(0, 3, size=n, dtype=np.int8), index=idx, dtype=np.int8)
    m1 = pd.DataFrame({"close": np.linspace(1.0, 1.1, num=n)}, index=idx)
    m2 = pd.DataFrame({"close": np.linspace(1.2, 1.3, num=n)}, index=idx)

    svc = object.__new__(TrainingService)
    cols, aligned = svc._align_global_feature_space(
        [
            ("EURUSD", PreparedDataset(X=x1, y=y1, index=idx, feature_names=list(x1.columns), metadata=m1, labels=y1)),
            ("GBPUSD", PreparedDataset(X=x2, y=y2, index=idx, feature_names=list(x2.columns), metadata=m2, labels=y2)),
        ],
        prefer_numpy=True,
    )

    assert cols == ["a", "b", "c"]
    assert len(aligned) == 2
    for _sym, ds in aligned:
        assert isinstance(ds.X, np.ndarray)
        assert isinstance(ds.y, np.ndarray)
        assert isinstance(ds.index, np.ndarray)
        assert ds.index.dtype == np.int64
        assert ds.X.shape == (n, 3)
        assert ds.y.shape == (n,)
        assert ds.metadata is not None
        assert len(ds.metadata) == n


def test_global_train_from_numpy_datasets_runs_with_pair_context_and_pandas_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_ENABLED", "1")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_WINDOW", "32")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_MIN_OVERLAP", "50")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_MAX_PEERS", "1")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_LAG", "1")
    monkeypatch.setenv("FOREX_BOT_GLOBAL_POOL_MEMMAP", "0")

    n = 1400
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    rng = np.random.default_rng(9)
    step = rng.normal(0.0, 0.0003, size=n).astype(np.float32)
    close_a = 1.05 + np.cumsum(step)
    close_b = 1.22 + np.cumsum((0.7 * step) + rng.normal(0.0, 0.0002, size=n).astype(np.float32))

    def _make_ds(close: np.ndarray, seed: int) -> PreparedDataset:
        y = np.random.default_rng(seed).integers(0, 3, size=n, dtype=np.int8)
        x = np.column_stack([close.astype(np.float32), np.linspace(0.0, 1.0, num=n, dtype=np.float32)]).astype(
            np.float32
        )
        return PreparedDataset(
            X=x,
            y=y,
            index=idx,
            feature_names=["close", "f1"],
            metadata=None,
            labels=y,
        )

    ds1 = _make_ds(close_a, seed=1)
    ds2 = _make_ds(close_b, seed=2)

    calls: dict[str, object] = {}

    class _Persistence:
        def save_run_summary(self, summary):
            calls["saved_summary"] = dict(summary)

    class _Trainer:
        def __init__(self):
            self.run_summary = {}
            self.persistence = _Persistence()

        def train_all(self, dataset, optimize, stop_event, models_override, exclude_models, memmap_dataset_dir=None):
            calls["called"] = True
            calls["rows"] = len(dataset.X)
            calls["features"] = list(dataset.feature_names)
            calls["optimize"] = bool(optimize)
            calls["memmap"] = memmap_dataset_dir
            assert isinstance(dataset.X, np.ndarray)
            assert isinstance(dataset.y, np.ndarray)
            assert dataset.X.shape[0] == dataset.y.shape[0]

    svc = object.__new__(TrainingService)
    svc.settings = SimpleNamespace(
        models=SimpleNamespace(global_train_ratio=0.8, global_max_rows_per_symbol=0, global_max_rows=0),
        risk=SimpleNamespace(meta_label_max_hold_bars=0, triple_barrier_max_bars=0),
        system=SimpleNamespace(cache_dir=str(tmp_path), symbol="EURUSD"),
    )
    svc.trainer = _Trainer()

    asyncio.run(
        svc._train_global_from_datasets(
            [("EURUSD", ds1), ("GBPUSD", ds2)],
            symbols=["EURUSD", "GBPUSD"],
            optimize=False,
            stop_event=None,
            exclude_models=None,
        )
    )

    assert calls.get("called") is True
    assert calls.get("rows", 0) > 0
    features = calls.get("features", [])
    assert isinstance(features, list)
    assert any(str(c).startswith("pair_") for c in features)
    saved = calls.get("saved_summary")
    assert isinstance(saved, dict)
    assert saved.get("global_training", {}).get("frame_native") is True


def test_global_train_from_dataframe_datasets_uses_numpy_when_frame_native(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.setenv("FOREX_BOT_GLOBAL_POOL_MEMMAP", "0")

    n = 1400
    idx = pd.to_datetime((np.arange(n, dtype=np.int64) + 1) * 60_000_000_000, utc=True)
    x1 = pd.DataFrame({"f0": np.linspace(0.0, 1.0, num=n), "f1": np.linspace(1.0, 2.0, num=n)}, index=idx)
    x2 = pd.DataFrame({"f0": np.linspace(2.0, 3.0, num=n), "f1": np.linspace(3.0, 4.0, num=n)}, index=idx)
    y1 = pd.Series(np.random.default_rng(31).integers(0, 3, size=n, dtype=np.int8), index=idx, dtype=np.int8)
    y2 = pd.Series(np.random.default_rng(32).integers(0, 3, size=n, dtype=np.int8), index=idx, dtype=np.int8)
    m1 = pd.DataFrame({"high": np.linspace(1.0, 1.2, num=n), "low": np.linspace(0.8, 1.0, num=n), "close": np.linspace(0.9, 1.1, num=n)}, index=idx)
    m2 = pd.DataFrame({"high": np.linspace(1.2, 1.4, num=n), "low": np.linspace(1.0, 1.2, num=n), "close": np.linspace(1.1, 1.3, num=n)}, index=idx)

    calls: dict[str, object] = {}

    class _Persistence:
        def save_run_summary(self, summary):
            calls["saved_summary"] = dict(summary)

    class _Trainer:
        def __init__(self):
            self.run_summary = {}
            self.persistence = _Persistence()

        def train_all(self, dataset, optimize, stop_event, models_override, exclude_models, memmap_dataset_dir=None):
            calls["called"] = True
            calls["meta"] = dataset.metadata
            assert isinstance(dataset.X, np.ndarray)
            assert isinstance(dataset.y, np.ndarray)
            assert dataset.X.shape[0] == dataset.y.shape[0]

    svc = object.__new__(TrainingService)
    svc.settings = SimpleNamespace(
        models=SimpleNamespace(global_train_ratio=0.8, global_max_rows_per_symbol=0, global_max_rows=0),
        risk=SimpleNamespace(meta_label_max_hold_bars=0, triple_barrier_max_bars=0),
        system=SimpleNamespace(cache_dir=str(tmp_path), symbol="EURUSD"),
    )
    svc.trainer = _Trainer()

    asyncio.run(
        svc._train_global_from_datasets(
            [
                ("EURUSD", PreparedDataset(X=x1, y=y1, index=idx, feature_names=list(x1.columns), metadata=m1, labels=y1)),
                ("GBPUSD", PreparedDataset(X=x2, y=y2, index=idx, feature_names=list(x2.columns), metadata=m2, labels=y2)),
            ],
            symbols=["EURUSD", "GBPUSD"],
            optimize=False,
            stop_event=None,
            exclude_models=None,
        )
    )

    assert calls.get("called") is True
    meta = calls.get("meta")
    assert meta is not None
    assert hasattr(meta, "columns")
    assert len(meta) > 0
    saved = calls.get("saved_summary")
    assert isinstance(saved, dict)
    assert saved.get("global_training", {}).get("frame_native") is True


def test_global_train_from_numpy_datasets_treats_rust_only_as_frame_native(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("FOREX_BOT_PANDAS_FREE", raising=False)
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setenv("FOREX_BOT_GLOBAL_POOL_MEMMAP", "0")

    n = 1200
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    x1 = np.random.default_rng(21).normal(size=(n, 2)).astype(np.float32)
    y1 = np.random.default_rng(22).integers(0, 3, size=n, dtype=np.int8)
    x2 = np.random.default_rng(23).normal(size=(n, 2)).astype(np.float32)
    y2 = np.random.default_rng(24).integers(0, 3, size=n, dtype=np.int8)

    ds1 = PreparedDataset(X=x1, y=y1, index=idx, feature_names=["f0", "f1"], metadata=None, labels=y1)
    ds2 = PreparedDataset(X=x2, y=y2, index=idx, feature_names=["f0", "f1"], metadata=None, labels=y2)

    calls: dict[str, object] = {}

    class _Persistence:
        def save_run_summary(self, summary):
            calls["saved_summary"] = dict(summary)

    class _Trainer:
        def __init__(self):
            self.run_summary = {}
            self.persistence = _Persistence()

        def train_all(self, dataset, optimize, stop_event, models_override, exclude_models, memmap_dataset_dir=None):
            calls["called"] = True
            assert isinstance(dataset.X, np.ndarray)
            assert isinstance(dataset.y, np.ndarray)
            assert dataset.X.shape[0] == dataset.y.shape[0]

    svc = object.__new__(TrainingService)
    svc.settings = SimpleNamespace(
        models=SimpleNamespace(global_train_ratio=0.8, global_max_rows_per_symbol=0, global_max_rows=0),
        risk=SimpleNamespace(meta_label_max_hold_bars=0, triple_barrier_max_bars=0),
        system=SimpleNamespace(cache_dir=str(tmp_path), symbol="EURUSD"),
    )
    svc.trainer = _Trainer()

    asyncio.run(
        svc._train_global_from_datasets(
            [("EURUSD", ds1), ("GBPUSD", ds2)],
            symbols=["EURUSD", "GBPUSD"],
            optimize=False,
            stop_event=None,
            exclude_models=None,
        )
    )

    assert calls.get("called") is True
    saved = calls.get("saved_summary")
    assert isinstance(saved, dict)
    assert saved.get("global_training", {}).get("frame_native") is True


def test_global_train_memmap_writes_frame_native_metadata_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.setenv("FOREX_BOT_GLOBAL_POOL_MEMMAP", "1")

    n = 1400
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    x = np.column_stack(
        [
            np.linspace(1.0, 2.0, num=n, dtype=np.float32),
            np.linspace(2.0, 3.0, num=n, dtype=np.float32),
        ]
    ).astype(np.float32)
    y = np.random.default_rng(41).integers(0, 3, size=n, dtype=np.int8)
    meta = _ArrayFrame(
        {
            "open": np.linspace(1.0, 1.2, num=n, dtype=np.float64),
            "high": np.linspace(1.1, 1.3, num=n, dtype=np.float64),
            "low": np.linspace(0.9, 1.1, num=n, dtype=np.float64),
            "close": np.linspace(1.05, 1.25, num=n, dtype=np.float64),
            "volume": np.linspace(10.0, 20.0, num=n, dtype=np.float64),
        },
        index=idx,
        attrs={"symbol": "EURUSD"},
    )

    calls: dict[str, object] = {}

    class _Persistence:
        def save_run_summary(self, summary):
            calls["saved_summary"] = dict(summary)

    class _Trainer:
        def __init__(self):
            self.run_summary = {}
            self.persistence = _Persistence()

        def train_all(self, dataset, optimize, stop_event, models_override, exclude_models, memmap_dataset_dir=None):
            calls["called"] = True
            calls["memmap"] = Path(memmap_dataset_dir) if memmap_dataset_dir is not None else None
            assert memmap_dataset_dir is not None
            assert (Path(memmap_dataset_dir) / "metadata.pkl").exists()
            loaded = joblib.load(Path(memmap_dataset_dir) / "metadata.pkl")
            assert hasattr(loaded, "columns")
            assert {"open", "high", "low", "close"}.issubset(set(getattr(loaded, "columns", [])))

    svc = object.__new__(TrainingService)
    svc.settings = SimpleNamespace(
        models=SimpleNamespace(global_train_ratio=0.8, global_max_rows_per_symbol=0, global_max_rows=0),
        risk=SimpleNamespace(meta_label_max_hold_bars=0, triple_barrier_max_bars=0),
        system=SimpleNamespace(cache_dir=str(tmp_path), symbol="EURUSD"),
    )
    svc.trainer = _Trainer()

    asyncio.run(
        svc._train_global_from_datasets(
            [
                (
                    "EURUSD",
                    PreparedDataset(
                        X=x,
                        y=y,
                        index=idx,
                        feature_names=["f0", "f1"],
                        metadata=meta,
                        labels=y,
                    ),
                )
            ],
            symbols=["EURUSD"],
            optimize=False,
            stop_event=None,
            exclude_models=None,
        )
    )

    assert calls.get("called") is True
    memmap_dir = calls.get("memmap")
    assert isinstance(memmap_dir, Path)
    assert (memmap_dir / "metadata.pkl").exists()


def test_merge_symbol_shards_dataframe_outputs_dataframe_sorted_deduped() -> None:
    idx1 = pd.to_datetime(np.array([3, 1, 2], dtype=np.int64) * 60_000_000_000, utc=True)
    idx2 = pd.to_datetime(np.array([2, 4], dtype=np.int64) * 60_000_000_000, utc=True)
    x1 = pd.DataFrame({"a": [3.0, 1.0, 2.0]}, index=idx1)
    x2 = pd.DataFrame({"a": [20.0, 40.0], "b": [200.0, 400.0]}, index=idx2)
    y1 = pd.Series([1, 0, 1], index=idx1, dtype=np.int8)
    y2 = pd.Series([2, 2], index=idx2, dtype=np.int8)
    m1 = pd.DataFrame({"close": [1.3, 1.1, 1.2]}, index=idx1)
    m2 = pd.DataFrame({"close": [2.2, 2.4]}, index=idx2)

    svc = object.__new__(TrainingService)
    ds = svc._merge_symbol_shards(
        "EURUSD",
        [
            PreparedDataset(X=x1, y=y1, index=idx1, feature_names=list(x1.columns), metadata=m1, labels=y1),
            PreparedDataset(X=x2, y=y2, index=idx2, feature_names=list(x2.columns), metadata=m2, labels=y2),
        ],
    )

    assert ds is not None
    assert hasattr(ds.X, "columns")
    assert list(ds.feature_names) == ["a", "b"]
    assert list(ds.X.columns) == ["a", "b"]
    assert len(ds.X) == 4  # de-dup timestamp=2 keeps first occurrence
    assert bool(ds.X.index.is_monotonic_increasing)
    np.testing.assert_allclose(ds.X["a"].to_numpy(dtype=np.float64), np.array([1.0, 2.0, 3.0, 40.0]))
    np.testing.assert_allclose(ds.X["b"].to_numpy(dtype=np.float64), np.array([0.0, 0.0, 0.0, 400.0]))
    assert len(ds.y) == len(ds.X)
    assert ds.metadata is not None
    assert len(ds.metadata) == len(ds.X)


def test_global_split_sorts_unsorted_numpy_dataset_before_cutoff() -> None:
    n = 400
    base = (np.arange(1, n + 1, dtype=np.int64)) * 60_000_000_000
    order = np.concatenate([np.arange(200, 400, dtype=np.int64), np.arange(0, 200, dtype=np.int64)])
    idx = base[order]
    x = base.astype(np.float32).reshape(-1, 1)[order]
    ds = PreparedDataset(
        X=x,
        y=(np.arange(n, dtype=np.int8) % 3),
        index=idx,
        feature_names=["a"],
        metadata=None,
        labels=None,
    )

    svc = object.__new__(TrainingService)
    train_parts, eval_map, meta = svc._split_global_train_eval(
        [("EURUSD", ds)],
        train_ratio=0.5,
        embargo_bars=0,
        min_train_rows=1,
        min_eval_rows=1,
    )

    assert meta.get("cutoff_mode") == "ratio"
    assert len(train_parts) == 1
    assert "EURUSD" in eval_map

    train_ds = train_parts[0][1]
    eval_ds = eval_map["EURUSD"]
    np.testing.assert_array_equal(np.asarray(train_ds.index, dtype=np.int64), base[:200])
    np.testing.assert_array_equal(np.asarray(eval_ds.index, dtype=np.int64), base[200:])
    np.testing.assert_allclose(np.asarray(train_ds.X, dtype=np.float32)[:, 0], base[:200].astype(np.float32))
    np.testing.assert_allclose(np.asarray(eval_ds.X, dtype=np.float32)[:, 0], base[200:].astype(np.float32))


def test_merge_symbol_shards_numpy_outputs_numpy_sorted_deduped() -> None:
    idx1 = np.array([3, 1, 2], dtype=np.int64) * 60_000_000_000
    idx2 = np.array([2, 4], dtype=np.int64) * 60_000_000_000
    ds1 = PreparedDataset(
        X=np.array([[3.0], [1.0], [2.0]], dtype=np.float32),
        y=np.array([1, 0, 1], dtype=np.int8),
        index=idx1,
        feature_names=["a"],
        metadata=None,
        labels=None,
    )
    ds2 = PreparedDataset(
        X=np.array([[20.0, 200.0], [40.0, 400.0]], dtype=np.float32),
        y=np.array([2, 2], dtype=np.int8),
        index=idx2,
        feature_names=["a", "b"],
        metadata=None,
        labels=None,
    )

    svc = object.__new__(TrainingService)
    merged = svc._merge_symbol_shards("GBPUSD", [ds1, ds2])

    assert merged is not None
    assert isinstance(merged.X, np.ndarray)
    assert isinstance(merged.y, np.ndarray)
    assert merged.X.shape == (4, 2)
    assert merged.y.shape == (4,)
    np.testing.assert_allclose(merged.X[:, 0], np.array([1.0, 2.0, 3.0, 40.0], dtype=np.float32))
    np.testing.assert_allclose(merged.X[:, 1], np.array([0.0, 0.0, 0.0, 400.0], dtype=np.float32))


def test_merge_symbol_shards_dataframe_forced_numpy_output() -> None:
    idx1 = pd.to_datetime(np.array([3, 1, 2], dtype=np.int64) * 60_000_000_000, utc=True)
    idx2 = pd.to_datetime(np.array([2, 4], dtype=np.int64) * 60_000_000_000, utc=True)
    x1 = pd.DataFrame({"a": [3.0, 1.0, 2.0]}, index=idx1)
    x2 = pd.DataFrame({"a": [20.0, 40.0], "b": [200.0, 400.0]}, index=idx2)
    y1 = pd.Series([1, 0, 1], index=idx1, dtype=np.int8)
    y2 = pd.Series([2, 2], index=idx2, dtype=np.int8)
    m1 = pd.DataFrame({"close": [1.3, 1.1, 1.2]}, index=idx1)
    m2 = pd.DataFrame({"close": [2.2, 2.4]}, index=idx2)

    svc = object.__new__(TrainingService)
    ds = svc._merge_symbol_shards(
        "EURUSD",
        [
            PreparedDataset(X=x1, y=y1, index=idx1, feature_names=list(x1.columns), metadata=m1, labels=y1),
            PreparedDataset(X=x2, y=y2, index=idx2, feature_names=list(x2.columns), metadata=m2, labels=y2),
        ],
        prefer_numpy=True,
    )

    assert ds is not None
    assert isinstance(ds.X, np.ndarray)
    assert isinstance(ds.y, np.ndarray)
    assert isinstance(ds.index, np.ndarray)
    assert ds.index.dtype == np.int64
    assert ds.metadata is not None
    assert len(ds.metadata) == len(ds.X)
    assert ds.X.shape == (4, 2)
    np.testing.assert_allclose(ds.X[:, 0], np.array([1.0, 2.0, 3.0, 40.0], dtype=np.float32))
    np.testing.assert_allclose(ds.X[:, 1], np.array([0.0, 0.0, 0.0, 400.0], dtype=np.float32))


def test_global_train_frame_native_skips_legacy_post_eval_but_records_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "0")
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.setenv("FOREX_BOT_GLOBAL_POOL_MEMMAP", "0")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_ENABLED", "0")

    n = 1400
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    close = np.linspace(1.05, 1.15, num=n, dtype=np.float32)
    x = np.column_stack(
        [
            close,
            np.linspace(0.0, 1.0, num=n, dtype=np.float32),
        ]
    ).astype(np.float32)
    y = np.random.default_rng(123).integers(0, 3, size=n, dtype=np.int8)
    meta = _ArrayFrame(
        {
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "atr": np.full(n, 0.0005, dtype=np.float32),
        },
        index=idx,
        attrs={"symbol": "EURUSD"},
    )

    calls: dict[str, object] = {}

    class _Model:
        def predict_proba(self, X, metadata=None):  # noqa: ANN001,N803
            if metadata is not None:
                calls["metadata_passed"] = True
            n_rows = int(len(X))
            out = np.zeros((n_rows, 3), dtype=np.float64)
            out[:, 1] = 1.0
            return out

    class _Persistence:
        def save_run_summary(self, summary):
            calls["saved_summary"] = dict(summary)

    class _Trainer:
        def __init__(self):
            self.run_summary = {}
            self.persistence = _Persistence()
            self.models = {"m1": _Model()}

        def train_all(self, dataset, optimize, stop_event, models_override, exclude_models, memmap_dataset_dir=None):
            calls["called"] = True
            calls["rows"] = int(len(dataset.X))

    svc = object.__new__(TrainingService)
    svc.settings = SimpleNamespace(
        models=SimpleNamespace(global_train_ratio=0.8, global_max_rows_per_symbol=0, global_max_rows=0),
        risk=SimpleNamespace(
            meta_label_max_hold_bars=0,
            triple_barrier_max_bars=0,
            daily_drawdown_limit=0.05,
            max_trades_per_day=10,
            meta_label_sl_pips=20.0,
            meta_label_tp_pips=40.0,
            min_risk_reward=2.0,
            backtest_spread_pips=1.5,
            commission_per_lot=0.0,
            trailing_enabled=False,
            trailing_atr_multiplier=1.0,
            trailing_be_trigger_r=1.0,
            atr_stop_multiplier=1.5,
            meta_label_min_dist=0.0,
        ),
        system=SimpleNamespace(cache_dir=str(tmp_path), symbol="EURUSD"),
    )
    svc.trainer = _Trainer()

    asyncio.run(
        svc._train_global_from_datasets(
            [("EURUSD", PreparedDataset(X=x, y=y, index=idx, feature_names=["close", "f1"], metadata=meta, labels=y))],
            symbols=["EURUSD"],
            optimize=False,
            stop_event=None,
            exclude_models=None,
        )
    )

    assert calls.get("called") is True
    assert int(calls.get("rows", 0)) > 0
    saved = calls.get("saved_summary")
    assert isinstance(saved, dict)
    model_metrics = saved.get("model_metrics", {})
    assert isinstance(model_metrics, dict)
    assert model_metrics == {}
    assert saved.get("global_training", {}).get("frame_native") is True
    assert calls.get("metadata_passed") is not True

