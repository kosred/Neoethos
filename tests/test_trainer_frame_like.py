from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tests._compat_pd import pd

from forex_bot.domain.events import PreparedDataset
from forex_bot.training import parallel_worker as pw
from forex_bot.training import trainer as trainer_mod
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


def test_slice_rows_accepts_frame_like_bool_mask() -> None:
    frame = _ArrayFrame(
        {"f0": np.array([1.0, 2.0, 3.0]), "f1": np.array([10.0, 20.0, 30.0])},
        index=np.array([100, 101, 102], dtype=np.int64),
    )
    out = ModelTrainer._slice_rows(frame, np.array([True, False, True], dtype=bool))
    assert list(getattr(out, "columns", [])) == ["f0", "f1"]
    assert np.asarray(out["f0"]).tolist() == [1.0, 3.0]
    assert np.asarray(getattr(out, "index")).tolist() == [100, 102]


def test_align_feature_frame_accepts_frame_like_and_fills_missing() -> None:
    frame = _ArrayFrame(
        {"f0": np.array([1.0, 2.0]), "f1": np.array([3.0, 4.0])},
        index=np.array([5, 6], dtype=np.int64),
    )
    out = ModelTrainer._align_feature_frame(frame, ["f1", "missing", "f0"])
    assert list(getattr(out, "columns", [])) == ["f1", "missing", "f0"]
    assert np.asarray(out["f1"], dtype=np.float32).tolist() == [3.0, 4.0]
    assert np.asarray(out["missing"], dtype=np.float32).tolist() == [0.0, 0.0]
    assert np.asarray(out["f0"], dtype=np.float32).tolist() == [1.0, 2.0]


def test_align_feature_frame_uses_rust_binding_when_available(monkeypatch) -> None:
    calls: dict[str, int] = {"count": 0}

    def _fake_align_feature_matrix(src_matrix, src_col_idx, dst_col_idx, dst_width):
        calls["count"] += 1
        return np.array([[9.0, 0.0, 1.0], [8.0, 0.0, 2.0]], dtype=np.float32)

    fake = SimpleNamespace(align_feature_matrix=_fake_align_feature_matrix)
    monkeypatch.setattr(trainer_mod, "_fb", fake, raising=False)

    frame = _ArrayFrame(
        {"f0": np.array([1.0, 2.0]), "f1": np.array([3.0, 4.0])},
        index=np.array([5, 6], dtype=np.int64),
    )
    out = ModelTrainer._align_feature_frame(frame, ["f1", "missing", "f0"])
    assert calls["count"] == 1
    assert np.asarray(out["f1"], dtype=np.float32).tolist() == [9.0, 8.0]
    assert np.asarray(out["missing"], dtype=np.float32).tolist() == [0.0, 0.0]
    assert np.asarray(out["f0"], dtype=np.float32).tolist() == [1.0, 2.0]


def test_month_day_indices_uses_rust_binding_when_available(monkeypatch) -> None:
    fake = SimpleNamespace(
        derive_time_index_arrays=lambda index_ns: (
            np.asarray(index_ns, dtype=np.int64) // 1_000_000,
            np.array([2024 * 12 + 1, 2024 * 12 + 1], dtype=np.int64),
            np.array([20240101, 20240102], dtype=np.int64),
        )
    )
    monkeypatch.setattr(trainer_mod, "_fb", fake, raising=False)
    idx_ns = np.array([1_704_067_200_000_000_000, 1_704_153_600_000_000_000], dtype=np.int64)
    month_idx, day_idx = ModelTrainer._month_day_indices_from_index(idx_ns, idx_ns.size)
    np.testing.assert_array_equal(month_idx, np.array([2024 * 12 + 1, 2024 * 12 + 1], dtype=np.int64))
    np.testing.assert_array_equal(day_idx, np.array([20240101, 20240102], dtype=np.int64))


def test_index_to_int64_normalizes_object_datetime_index() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    obj_idx = np.asarray(list(idx), dtype=object)
    expected = np.asarray([int(ts.value) for ts in list(idx)], dtype=np.int64)

    out = trainer_mod._index_to_int64(obj_idx)

    assert out is not None
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, expected)


def test_is_datetime_index_accepts_object_datetime_array() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    assert trainer_mod._is_datetime_index(np.asarray(list(idx), dtype=object)) is True


def test_month_day_indices_accept_object_datetime_array_without_rust() -> None:
    idx = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    obj_idx = np.asarray(list(idx), dtype=object)

    month_idx, day_idx = ModelTrainer._month_day_indices_from_index(obj_idx, 2)

    assert month_idx.shape == (2,)
    assert day_idx.shape == (2,)
    assert int(day_idx[1]) >= int(day_idx[0])


def test_rust_sorted_index_order_uses_binding_when_available(monkeypatch) -> None:
    calls = {"sort": 0}

    def _sorted_index_order(idx_ns):
        calls["sort"] += 1
        assert np.asarray(idx_ns, dtype=np.int64).shape[0] == 3
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(
        trainer_mod,
        "_fb",
        SimpleNamespace(sorted_index_order=_sorted_index_order),
        raising=False,
    )
    idx = pd.to_datetime(
        [
            "2025-01-03T00:00:00Z",
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z",
        ]
    )

    out = trainer_mod._rust_sorted_index_order(idx)

    assert calls["sort"] == 1
    assert out is not None
    np.testing.assert_array_equal(out, np.array([1, 2, 0], dtype=np.int64))


def test_sorted_time_order_prefers_binding_for_unsorted_datetime_index(monkeypatch) -> None:
    calls = {"sort": 0}

    def _sorted_index_order(idx_ns):
        calls["sort"] += 1
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(
        trainer_mod,
        "_fb",
        SimpleNamespace(sorted_index_order=_sorted_index_order),
        raising=False,
    )
    idx = pd.to_datetime(
        [
            "2025-01-03T00:00:00Z",
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z",
        ]
    )

    out = trainer_mod._sorted_time_order(idx, 3)

    assert calls["sort"] == 1
    assert out is not None
    np.testing.assert_array_equal(out, np.array([1, 2, 0], dtype=np.int64))


def test_rust_rank_scores_desc_uses_binding_when_available(monkeypatch) -> None:
    calls = {"rank": 0}

    def _rank_scores_desc(scores, absolute=False):
        calls["rank"] += 1
        assert bool(absolute) is False
        np.testing.assert_allclose(np.asarray(scores, dtype=np.float64), np.array([0.4, 2.0, 1.0], dtype=np.float64))
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(
        trainer_mod,
        "_fb",
        SimpleNamespace(rank_scores_desc=_rank_scores_desc),
        raising=False,
    )

    out = trainer_mod._rust_rank_scores_desc(np.array([0.4, 2.0, 1.0], dtype=np.float64))

    assert calls["rank"] == 1
    assert out is not None
    np.testing.assert_array_equal(out, np.array([1, 2, 0], dtype=np.int64))


def test_sorted_time_order_returns_none_for_monotonic_datetime_index() -> None:
    idx = pd.to_datetime(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z",
            "2025-01-03T00:00:00Z",
        ]
    )

    assert trainer_mod._sorted_time_order(idx, 3) is None


def test_l1_regime_mask_accepts_frame_like_suffix_column(tmp_path: Path) -> None:
    trainer = object.__new__(ModelTrainer)
    trainer.settings = SimpleNamespace(risk=SimpleNamespace(regime_adx_trend=25.0, regime_adx_range=20.0))
    trainer.models_dir = tmp_path
    trainer.run_summary = {}
    frame = _ArrayFrame(
        {"signal_adx": np.array([10.0, 22.0, 30.0], dtype=np.float32)},
        index=np.arange(3, dtype=np.int64),
    )
    mask = trainer._l1_regime_mask(frame)
    assert mask.tolist() == [0, 1, 2]


def test_apply_l1_feature_selection_disabled_works_with_frame_like(tmp_path: Path) -> None:
    trainer = object.__new__(ModelTrainer)
    trainer.settings = SimpleNamespace(
        models=SimpleNamespace(l1_feature_selection_enabled=False),
        risk=SimpleNamespace(regime_adx_trend=25.0, regime_adx_range=20.0),
    )
    trainer.models_dir = tmp_path
    trainer.run_summary = {}

    x_fit = _ArrayFrame(
        {
            "f0": np.random.default_rng(1).normal(size=64).astype(np.float32),
            "f1": np.random.default_rng(2).normal(size=64).astype(np.float32),
        },
        index=np.arange(64, dtype=np.int64),
    )
    x_eval = _ArrayFrame(
        {
            "f0": np.random.default_rng(3).normal(size=16).astype(np.float32),
            "f1": np.random.default_rng(4).normal(size=16).astype(np.float32),
        },
        index=np.arange(16, dtype=np.int64),
    )
    y_fit = np.random.default_rng(5).integers(0, 3, size=64, dtype=np.int8)

    fit_sel, eval_sel, cols = trainer._apply_l1_feature_selection(x_fit, y_fit, x_eval)
    assert fit_sel is x_fit
    assert eval_sel is x_eval
    assert cols == ["f0", "f1"]


def test_coerce_numpy_dataset_accepts_frame_like() -> None:
    n = 24
    x = _ArrayFrame(
        {
            "f0": np.random.default_rng(1).normal(size=n).astype(np.float32),
            "f1": np.random.default_rng(2).normal(size=n).astype(np.float32),
        },
        index=np.arange(n, dtype=np.int64),
    )
    y = np.random.default_rng(3).integers(0, 3, size=n, dtype=np.int8)
    ds = PreparedDataset(
        X=x,
        y=y,
        index=np.arange(n, dtype=np.int64),
        feature_names=[],
    )
    out = ModelTrainer._coerce_numpy_dataset(ds)
    assert out is not None
    x_np, y_np, names, idx = out
    assert x_np.shape == (n, 2)
    assert y_np.shape == (n,)
    assert names == ["f0", "f1"]
    assert idx is not None
    assert idx.shape == (n,)


def test_persist_metadata_artifact_falls_back_to_numpy_frame(tmp_path: Path) -> None:
    class _UnpicklableFrame(_ArrayFrame):
        def __getstate__(self):
            raise RuntimeError("unpicklable frame")

    meta = _UnpicklableFrame(
        {
            "close": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "high": np.array([1.1, 2.1, 3.1], dtype=np.float64),
            "low": np.array([0.9, 1.9, 2.9], dtype=np.float64),
        },
        index=np.array([100, 101, 102], dtype=np.int64),
        attrs={"symbol": "EURUSD"},
    )
    path = tmp_path / "metadata.pkl"

    out = ModelTrainer._persist_metadata_artifact(meta, path)
    assert out == path
    assert path.exists()

    loaded = pw._load_metadata_artifact(path)
    assert loaded is not None
    assert hasattr(loaded, "columns")
    assert "close" in list(getattr(loaded, "columns", []))
    np.testing.assert_allclose(np.asarray(loaded["close"]), np.array([1.0, 2.0, 3.0], dtype=np.float64))
