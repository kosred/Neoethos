from __future__ import annotations

import numpy as np

from forex_bot.training import optimization as opt_mod
from forex_bot.training.optimization import HyperparameterOptimizer


class _ArrayFrame:
    def __init__(self, data, index, attrs=None):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.columns = list(self._data.keys())
        self.index = np.asarray(index).reshape(-1)
        self.attrs = dict(attrs or {})

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


def test_optimizer_slice_rows_accepts_frame_like_without_iloc() -> None:
    frame = _ArrayFrame(
        {
            "f0": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "f1": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        },
        index=np.array([10, 11, 12], dtype=np.int64),
        attrs={"symbol": "EURUSD"},
    )
    rows = np.array([0, 2], dtype=np.int64)
    out = HyperparameterOptimizer._slice_rows(frame, rows)
    np.testing.assert_allclose(np.asarray(out["f0"]), np.array([1.0, 3.0], dtype=np.float32), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.index), np.array([10, 12], dtype=np.int64), rtol=0, atol=0)


def test_optimizer_meta_column_resolves_case_insensitive_on_frame_like() -> None:
    meta = _ArrayFrame({"Close": np.array([1.0, 1.1, 1.2], dtype=np.float64)}, index=np.array([0, 1, 2]))
    out = HyperparameterOptimizer._meta_column(meta, "close", dtype=np.float64)
    assert out is not None
    np.testing.assert_allclose(out, np.array([1.0, 1.1, 1.2], dtype=np.float64), rtol=0, atol=1e-12)


def test_optimizer_meta_month_day_indices_uses_rust_binding_when_available(monkeypatch) -> None:
    fake = type(
        "_Fake",
        (),
        {
            "derive_time_index_arrays": staticmethod(
                lambda index_ns: (
                    np.asarray(index_ns, dtype=np.int64) // 1_000_000,
                    np.array([301, 302, 303], dtype=np.int64),
                    np.array([401, 402, 403], dtype=np.int64),
                )
            )
        },
    )()
    monkeypatch.setattr(opt_mod, "_fb", fake, raising=False)

    idx_ns = np.array([1_704_067_200_000_000_000, 1_704_153_600_000_000_000, 1_704_240_000_000_000_000], dtype=np.int64)
    opt = object.__new__(HyperparameterOptimizer)
    month_idx, day_idx = HyperparameterOptimizer._meta_month_day_indices(opt, {"index": idx_ns}, idx_ns.size)

    np.testing.assert_array_equal(month_idx, np.array([301, 302, 303], dtype=np.int64))
    np.testing.assert_array_equal(day_idx, np.array([401, 402, 403], dtype=np.int64))
