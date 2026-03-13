from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from forex_bot.domain.events import PreparedDataset
from forex_bot.training.trainer import ModelTrainer


class _ArrayFrame:
    def __init__(self, data, index, attrs=None):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    def __len__(self) -> int:
        return int(len(self.index))

    @property
    def shape(self) -> tuple[int, int]:
        return int(len(self.index)), int(len(self.columns))

    def __getitem__(self, key):
        return self._data[str(key)]


def _make_memmap_dataset(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.array([0, 1, 2, 1, 0, 2, 1, 0], dtype=np.int8)
    np.save(path / "X.npy", x)
    np.save(path / "y.npy", y)
    (path / "columns.json").write_text(json.dumps(["f0", "f1", "f2"]), encoding="utf-8")


def test_train_all_uses_frame_native_memmap_short_circuit_even_with_legacy_toggle_off(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    _make_memmap_dataset(tmp_path / "pool")

    trainer = object.__new__(ModelTrainer)
    called: dict[str, object] = {}

    def _stub_frame_native(**kwargs):
        called.update(kwargs)
        return True

    trainer._train_all_frame_native_memmap = _stub_frame_native  # type: ignore[attr-defined]

    ds = PreparedDataset(
        X=np.zeros((4, 2), dtype=np.float32),
        y=np.array([0, 1, 0, 2], dtype=np.int8),
        index=np.arange(4, dtype=np.int64),
        feature_names=["a", "b"],
    )

    trainer.train_all(ds, optimize=True, memmap_dataset_dir=tmp_path / "pool")

    assert bool(called)
    # optimize is disabled before entering pandas-free memmap path.
    assert called["optimize"] is False
    assert called["memmap_dataset_dir"] == (tmp_path / "pool")


def test_materialize_numpy_memmap_dataset_persists_metadata_artifact(tmp_path: Path) -> None:
    trainer = object.__new__(ModelTrainer)
    trainer.settings = SimpleNamespace(system=SimpleNamespace(cache_dir=str(tmp_path)))

    n = 12
    ds = PreparedDataset(
        X=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        y=np.arange(n, dtype=np.int8) % 3,
        index=np.arange(n, dtype=np.int64),
        feature_names=["f0", "f1"],
        metadata=_ArrayFrame(
            {
                "open": np.linspace(1.0, 1.1, num=n, dtype=np.float64),
                "high": np.linspace(1.1, 1.2, num=n, dtype=np.float64),
                "low": np.linspace(0.9, 1.0, num=n, dtype=np.float64),
                "close": np.linspace(1.05, 1.15, num=n, dtype=np.float64),
            },
            index=np.arange(n, dtype=np.int64),
            attrs={"symbol": "EURUSD"},
        ),
    )

    out_dir = trainer._materialize_numpy_memmap_dataset(ds)

    assert out_dir is not None
    meta_path = out_dir / "metadata.pkl"
    assert meta_path.exists()
    loaded = joblib.load(meta_path)
    assert hasattr(loaded, "columns")
    assert {"open", "high", "low", "close"}.issubset(set(getattr(loaded, "columns", [])))


def test_frame_native_memmap_helper_trains_without_dataframe(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    memmap_dir = tmp_path / "pool"
    _make_memmap_dataset(memmap_dir)

    trainer = object.__new__(ModelTrainer)
    trainer.settings = SimpleNamespace(
        system=SimpleNamespace(enable_gpu=False, num_gpus=0, device="cpu")
    )
    trainer.run_summary = {}
    trainer.models = {"xgboost": object()}
    trainer.distributed_enabled = False
    trainer.rank = 0

    calls: dict[str, int] = {
        "save_run_summary": 0,
        "save_active_models_list": 0,
        "save_models_bundle": 0,
        "cleanup_logs": 0,
    }

    class _PersistenceStub:
        def save_run_summary(self, _summary):
            calls["save_run_summary"] += 1

        def save_active_models_list(self, _models):
            calls["save_active_models_list"] += 1

        def save_models_bundle(self, _models):
            calls["save_models_bundle"] += 1

        def cleanup_logs(self):
            calls["cleanup_logs"] += 1

    trainer.persistence = _PersistenceStub()
    trainer._get_enabled_models = lambda: ["xgboost"]  # type: ignore[assignment]
    trainer._maybe_shard_models = lambda models: list(models)  # type: ignore[assignment]

    def _parallel_stub(enabled_models, X_fit, y_fit, **kwargs):
        assert enabled_models == ["xgboost"]
        assert isinstance(X_fit, np.ndarray)
        assert isinstance(y_fit, np.ndarray)
        assert X_fit.shape == (0, 0)
        assert y_fit.shape == (0,)
        assert Path(kwargs["memmap_dataset_dir"]) == memmap_dir
        return {"xgboost": 1.25}

    trainer._train_models_parallel = _parallel_stub  # type: ignore[assignment]

    ok = trainer._train_all_frame_native_memmap(  # type: ignore[attr-defined]
        optimize=False,
        stop_event=None,
        models_override=None,
        exclude_models=None,
        memmap_dataset_dir=memmap_dir,
        meta_fit=None,
    )

    assert ok is True
    assert trainer.run_summary["train_samples"] == 8
    assert trainer.run_summary["feature_columns"] == ["f0", "f1", "f2"]
    assert trainer.run_summary["train_durations_sec"] == {"xgboost": 1.25}
    assert calls["save_run_summary"] == 1
    assert calls["save_active_models_list"] == 1
    assert calls["save_models_bundle"] == 1
    assert calls["cleanup_logs"] == 1


def test_frame_native_memmap_helper_requires_metadata_for_metadata_models(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    memmap_dir = tmp_path / "pool"
    _make_memmap_dataset(memmap_dir)

    trainer = object.__new__(ModelTrainer)
    trainer.settings = SimpleNamespace(
        system=SimpleNamespace(enable_gpu=False, num_gpus=0, device="cpu", symbol="")
    )
    trainer.run_summary = {}
    trainer.models = {"genetic": object()}
    trainer.distributed_enabled = False
    trainer.rank = 0
    trainer.persistence = SimpleNamespace(
        save_run_summary=lambda _summary: None,
        save_active_models_list=lambda _models: None,
        save_models_bundle=lambda _models: None,
        cleanup_logs=lambda: None,
    )
    trainer._get_enabled_models = lambda: ["genetic"]  # type: ignore[assignment]
    trainer._maybe_shard_models = lambda models: list(models)  # type: ignore[assignment]

    def _parallel_stub(*_args, **_kwargs):
        raise AssertionError("_train_models_parallel should not run without required metadata")

    trainer._train_models_parallel = _parallel_stub  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="metadata"):
        trainer._train_all_frame_native_memmap(  # type: ignore[attr-defined]
            optimize=False,
            stop_event=None,
            models_override=None,
            exclude_models=None,
            memmap_dataset_dir=memmap_dir,
            meta_fit=None,
        )


def test_frame_native_memmap_helper_skips_when_no_compatible_rust_models(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    memmap_dir = tmp_path / "pool"
    _make_memmap_dataset(memmap_dir)

    trainer = object.__new__(ModelTrainer)
    trainer.settings = SimpleNamespace(
        system=SimpleNamespace(enable_gpu=False, num_gpus=0, device="cpu")
    )
    trainer.run_summary = {}
    trainer.models = {}
    trainer.distributed_enabled = False
    trainer.rank = 0

    calls: dict[str, int] = {
        "save_run_summary": 0,
        "save_active_models_list": 0,
        "save_models_bundle": 0,
        "cleanup_logs": 0,
    }

    class _PersistenceStub:
        def save_run_summary(self, _summary):
            calls["save_run_summary"] += 1

        def save_active_models_list(self, _models):
            calls["save_active_models_list"] += 1

        def save_models_bundle(self, _models):
            calls["save_models_bundle"] += 1

        def cleanup_logs(self):
            calls["cleanup_logs"] += 1

    trainer.persistence = _PersistenceStub()
    trainer._get_enabled_models = lambda: []  # type: ignore[assignment]
    trainer._maybe_shard_models = lambda models: list(models)  # type: ignore[assignment]

    def _parallel_stub(*_args, **_kwargs):
        raise AssertionError("_train_models_parallel should not run when no rust-compatible models are enabled")

    trainer._train_models_parallel = _parallel_stub  # type: ignore[assignment]

    ok = trainer._train_all_frame_native_memmap(  # type: ignore[attr-defined]
        optimize=False,
        stop_event=None,
        models_override=None,
        exclude_models=None,
        memmap_dataset_dir=memmap_dir,
        meta_fit=None,
    )

    assert ok is True
    info = trainer.run_summary.get("frame_native_memmap", {})
    assert info.get("enabled") is True
    assert info.get("reason") == "no_compatible_rust_models"
    assert info.get("models") == []
    assert calls["save_run_summary"] == 1
    assert calls["save_active_models_list"] == 1
    assert calls["save_models_bundle"] == 1
    assert calls["cleanup_logs"] == 1


def test_train_all_frame_native_aborts_when_memmap_unavailable(monkeypatch) -> None:
    trainer = object.__new__(ModelTrainer)
    calls = {"materialize": 0, "frame_native": 0}

    def _materialize(_dataset):
        calls["materialize"] += 1
        return None

    def _frame_native(**_kwargs):
        calls["frame_native"] += 1
        return True

    trainer._materialize_numpy_memmap_dataset = _materialize  # type: ignore[attr-defined]
    trainer._train_all_frame_native_memmap = _frame_native  # type: ignore[attr-defined]

    ds = PreparedDataset(
        X=np.zeros((16, 3), dtype=np.float32),
        y=np.zeros(16, dtype=np.int8),
        index=np.arange(16, dtype=np.int64),
        feature_names=["a", "b", "c"],
    )
    trainer.train_all(ds, optimize=False, memmap_dataset_dir=None)

    assert calls["materialize"] == 1
    assert calls["frame_native"] == 0


def test_distribute_worker_threads_uses_full_cpu_budget() -> None:
    threads = ModelTrainer._distribute_worker_threads(cpu_budget=11, total_concurrent=8)

    assert threads == [2, 2, 2, 1, 1, 1, 1, 1]
    assert sum(threads) == 11
