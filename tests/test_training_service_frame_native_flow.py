from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np

from forex_bot.domain.events import PreparedDataset
from forex_bot.execution.training_service import TrainingService


def _dataset(rows: int = 128) -> PreparedDataset:
    x = np.random.default_rng(11).normal(size=(rows, 4)).astype(np.float32)
    y = np.random.default_rng(12).integers(0, 3, size=rows, dtype=np.int8)
    idx = np.arange(rows, dtype=np.int64)
    return PreparedDataset(
        X=x,
        y=y,
        index=idx,
        feature_names=[f"f{i}" for i in range(x.shape[1])],
        metadata=None,
        labels=y,
    )


def test_train_frame_native_bypasses_loader(monkeypatch) -> None:
    svc = object.__new__(TrainingService)
    svc.settings = SimpleNamespace(
        system=SimpleNamespace(symbol="EURUSD"),
        news=SimpleNamespace(enable_news=False),
    )

    class _FailLoader:
        async def ensure_history(self, _symbol):
            raise AssertionError("ensure_history should not run in pandas-free train path")

        async def get_training_data(self, _symbol):
            raise AssertionError("get_training_data should not run in pandas-free train path")

    calls: dict[str, int] = {"train_all": 0, "stop": 0}
    svc.data_loader = _FailLoader()
    svc.feature_engineer = SimpleNamespace(prepare=lambda _frames, news_features=None, symbol=None: _dataset(96))

    def _train_all(ds, optimize, stop_event, *_args):
        calls["train_all"] += 1
        assert isinstance(ds.X, np.ndarray)
        assert len(ds.X) == 96
        assert optimize is False

    svc.trainer = SimpleNamespace(train_all=_train_all)
    svc._maybe_stop_ray = lambda: calls.__setitem__("stop", calls["stop"] + 1)

    asyncio.run(svc.train(optimize=False, stop_event=None))
    assert calls["train_all"] == 1
    assert calls["stop"] == 1


def test_train_global_routes_to_frame_native_path(monkeypatch) -> None:
    svc = object.__new__(TrainingService)
    svc.settings = SimpleNamespace(system=SimpleNamespace(symbol="EURUSD"))
    calls: dict[str, object] = {"stop": 0}

    async def _pf(symbols, optimize, stop_event):
        calls["symbols"] = list(symbols)
        calls["optimize"] = bool(optimize)
        calls["called"] = True

    svc._train_global_frame_native = _pf
    svc._maybe_stop_ray = lambda: calls.__setitem__("stop", int(calls["stop"]) + 1)

    asyncio.run(svc.train_global(["EURUSD", "GBPUSD"], optimize=True, stop_event=None))
    assert calls.get("called") is True
    assert calls.get("symbols") == ["EURUSD", "GBPUSD"]
    assert calls.get("optimize") is True
    assert calls["stop"] == 1


def test_train_global_frame_native_collects_non_empty_datasets() -> None:
    svc = object.__new__(TrainingService)
    svc.feature_engineer = SimpleNamespace(
        prepare=lambda _frames, news_features=None, symbol=None: _dataset(64) if symbol == "EURUSD" else _dataset(0)
    )

    called: dict[str, object] = {}

    async def _from_datasets(datasets, symbols, optimize, stop_event, exclude_models=None):
        called["datasets"] = datasets
        called["symbols"] = symbols
        called["optimize"] = optimize
        called["exclude_models"] = exclude_models
        return None

    svc._train_global_from_datasets = _from_datasets

    asyncio.run(svc._train_global_frame_native(["EURUSD", "GBPUSD"], optimize=False, stop_event=None))

    assert "datasets" in called
    datasets = called["datasets"]
    assert isinstance(datasets, list)
    assert len(datasets) == 1
    assert datasets[0][0] == "EURUSD"
    assert called["symbols"] == ["EURUSD"]
