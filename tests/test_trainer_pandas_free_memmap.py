from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from forex_bot.domain.events import PreparedDataset
from forex_bot.training.trainer import ModelTrainer


def _make_memmap_dataset(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.array([0, 1, 2, 1, 0, 2, 1, 0], dtype=np.int8)
    np.save(path / "X.npy", x)
    np.save(path / "y.npy", y)
    (path / "columns.json").write_text(json.dumps(["f0", "f1", "f2"]), encoding="utf-8")


def test_train_all_uses_pandas_free_memmap_short_circuit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    _make_memmap_dataset(tmp_path / "pool")

    trainer = object.__new__(ModelTrainer)
    called: dict[str, object] = {}

    def _stub_pandas_free(**kwargs):
        called.update(kwargs)
        return True

    trainer._train_all_pandas_free_memmap = _stub_pandas_free  # type: ignore[attr-defined]

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


def test_pandas_free_memmap_helper_trains_without_dataframe(monkeypatch, tmp_path: Path) -> None:
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

    ok = trainer._train_all_pandas_free_memmap(  # type: ignore[attr-defined]
        optimize=False,
        stop_event=None,
        models_override=None,
        exclude_models=None,
        memmap_dataset_dir=memmap_dir,
    )

    assert ok is True
    assert trainer.run_summary["train_samples"] == 8
    assert trainer.run_summary["feature_columns"] == ["f0", "f1", "f2"]
    assert trainer.run_summary["train_durations_sec"] == {"xgboost": 1.25}
    assert calls["save_run_summary"] == 1
    assert calls["save_active_models_list"] == 1
    assert calls["save_models_bundle"] == 1
    assert calls["cleanup_logs"] == 1


def test_pandas_free_memmap_helper_skips_when_no_compatible_rust_models(monkeypatch, tmp_path: Path) -> None:
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

    ok = trainer._train_all_pandas_free_memmap(  # type: ignore[attr-defined]
        optimize=False,
        stop_event=None,
        models_override=None,
        exclude_models=None,
        memmap_dataset_dir=memmap_dir,
    )

    assert ok is True
    info = trainer.run_summary.get("pandas_free_memmap", {})
    assert info.get("enabled") is True
    assert info.get("reason") == "no_compatible_rust_models"
    assert info.get("models") == []
    assert calls["save_run_summary"] == 1
    assert calls["save_active_models_list"] == 1
    assert calls["save_models_bundle"] == 1
    assert calls["cleanup_logs"] == 1


def test_train_all_pandas_free_strict_skips_fallback_when_memmap_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.delenv("FOREX_BOT_PANDAS_FREE_STRICT", raising=False)

    trainer = object.__new__(ModelTrainer)
    calls = {"materialize": 0, "pandas_free": 0}

    def _materialize(_dataset):
        calls["materialize"] += 1
        return None

    def _pandas_free(**_kwargs):
        calls["pandas_free"] += 1
        return True

    trainer._materialize_numpy_memmap_dataset = _materialize  # type: ignore[attr-defined]
    trainer._train_all_pandas_free_memmap = _pandas_free  # type: ignore[attr-defined]

    ds = PreparedDataset(
        X=np.zeros((16, 3), dtype=np.float32),
        y=np.zeros(16, dtype=np.int8),
        index=np.arange(16, dtype=np.int64),
        feature_names=["a", "b", "c"],
    )
    trainer.train_all(ds, optimize=False, memmap_dataset_dir=None)

    assert calls["materialize"] == 1
    assert calls["pandas_free"] == 0


def test_train_all_strict_off_still_does_not_allow_pandas_fallback(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE_STRICT", "0")

    trainer = object.__new__(ModelTrainer)
    calls = {"materialize": 0, "pandas_free": 0}

    def _materialize(_dataset):
        calls["materialize"] += 1
        return None

    def _pandas_free(**_kwargs):
        calls["pandas_free"] += 1
        return True

    trainer._materialize_numpy_memmap_dataset = _materialize  # type: ignore[attr-defined]
    trainer._train_all_pandas_free_memmap = _pandas_free  # type: ignore[attr-defined]

    ds = PreparedDataset(
        X=np.zeros((16, 3), dtype=np.float32),
        y=np.zeros(16, dtype=np.int8),
        index=np.arange(16, dtype=np.int64),
        feature_names=["a", "b", "c"],
    )
    trainer.train_all(ds, optimize=False, memmap_dataset_dir=None)

    assert calls["materialize"] == 1
    assert calls["pandas_free"] == 0
