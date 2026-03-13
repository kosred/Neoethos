from __future__ import annotations

import numpy as np
import pytest

from forex_bot.training.benchmark_service import BenchmarkService


class _ArrayFrame:
    def __init__(self, data, index, attrs=None):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


def test_run_micro_benchmark_accepts_numpy_arrays() -> None:
    bench = BenchmarkService()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(256, 8)).astype(np.float32)
    y = rng.integers(-1, 2, size=256, dtype=np.int8)

    out = bench.run_micro_benchmark(x, y, "cpu")
    assert out["samples_trained"] > 0
    assert out["actual_duration_sec"] >= 0.0
    assert out["samples_per_second"] >= 0.0


def test_run_micro_benchmark_accepts_frame_like_arrays() -> None:
    bench = BenchmarkService()
    rng = np.random.default_rng(17)
    n = 300
    x = _ArrayFrame(
        {
            "f0": rng.normal(size=n).astype(np.float32),
            "f1": rng.normal(size=n).astype(np.float32),
            "f2": rng.normal(size=n).astype(np.float32),
            "f3": rng.normal(size=n).astype(np.float32),
        },
        index=np.arange(n, dtype=np.int64),
    )
    y = rng.integers(-1, 2, size=n, dtype=np.int8)

    out = bench.run_micro_benchmark(x, y, "cpu")
    assert out["samples_trained"] > 0
    assert out["actual_duration_sec"] >= 0.0
    assert out["samples_per_second"] >= 0.0


def test_estimate_time_uses_numpy_probe_without_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = BenchmarkService()
    seen: dict[str, int] = {}

    def _probe(model_name: str, x: np.ndarray, y: np.ndarray, device: str):
        assert model_name == "xgboost"
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert device == "cpu"
        seen["rows"] = int(x.shape[0])
        return 0.01, 1.0

    monkeypatch.setattr(bench, "_probe_scaling_law", _probe)
    x_probe = np.random.default_rng(11).normal(size=(2048, 6)).astype(np.float32)
    est = bench.estimate_time(
        models=["xgboost"],
        n_samples=1000,
        benchmark_result=None,
        gpu=False,
        probe_kwargs={"X": x_probe},
    )

    # (slope*n + intercept) * epochs, epochs(xgboost)=5
    assert est == pytest.approx(55.0)
    assert seen["rows"] == 2048
