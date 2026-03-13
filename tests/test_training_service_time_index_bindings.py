from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from tests._compat_pd import pd

from forex_bot.execution import training_service as ts


def test_month_day_indices_uses_rust_binding_when_available(monkeypatch) -> None:
    fake = SimpleNamespace(
        derive_time_index_arrays=lambda index_ns: (
            np.asarray(index_ns, dtype=np.int64) // 1_000_000,
            np.array([2024 * 12 + 1, 2024 * 12 + 1], dtype=np.int64),
            np.array([20240101, 20240102], dtype=np.int64),
        )
    )
    monkeypatch.setattr(ts, "_fb", fake, raising=False)
    idx_ns = np.array([1_704_067_200_000_000_000, 1_704_153_600_000_000_000], dtype=np.int64)
    month_idx, day_idx = ts.TrainingService._month_day_indices_from_index(idx_ns)
    np.testing.assert_array_equal(month_idx, np.array([2024 * 12 + 1, 2024 * 12 + 1], dtype=np.int64))
    np.testing.assert_array_equal(day_idx, np.array([20240101, 20240102], dtype=np.int64))


def test_align_ffill_by_ns_uses_rust_binding_when_available(monkeypatch) -> None:
    fake = SimpleNamespace(
        align_ffill_values_by_ns=lambda src_idx_ns, src_vals, tgt_idx_ns, fill=0.0: np.array([11.0, 22.0, 44.0], dtype=np.float64)
    )
    monkeypatch.setattr(ts, "_fb", fake, raising=False)

    src_idx = np.array([1, 2, 3], dtype=np.int64)
    src_vals = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    tgt_idx = np.array([1, 2, 4], dtype=np.int64)
    out = ts._align_ffill_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float32, fill=0.0)

    assert out is not None
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([11.0, 22.0, 44.0], dtype=np.float32))


def test_align_ffill_by_ns_fallback_uses_rust_sorted_index_order(monkeypatch) -> None:
    calls: dict[str, int] = {"sort": 0}

    def _fake_sorted_index_order(idx_ns):
        calls["sort"] += 1
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(ts, "_fb", SimpleNamespace(sorted_index_order=_fake_sorted_index_order), raising=False)

    src_idx = np.array([30, 10, 20], dtype=np.int64)
    src_vals = np.array([3.0, 1.0, 2.0], dtype=np.float64)
    tgt_idx = np.array([10, 25, 30], dtype=np.int64)
    out = ts._align_ffill_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float32, fill=0.0)

    assert out is not None
    assert calls["sort"] == 1
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_align_exact_by_ns_uses_rust_binding_when_available(monkeypatch) -> None:
    fake = SimpleNamespace(
        align_exact_values_by_ns=lambda src_idx_ns, src_vals, tgt_idx_ns, fill=0.0: np.array([1.0, 0.0, 3.0], dtype=np.float64)
    )
    monkeypatch.setattr(ts, "_fb", fake, raising=False)
    src_idx = np.array([10, 20, 30], dtype=np.int64)
    src_vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    tgt_idx = np.array([10, 25, 30], dtype=np.int64)
    out = ts._align_exact_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float32, fill=0.0)
    assert out is not None
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([1.0, 0.0, 3.0], dtype=np.float32))


def test_align_exact_by_ns_fallback_uses_rust_sorted_index_order(monkeypatch) -> None:
    calls: dict[str, int] = {"sort": 0}

    def _fake_sorted_index_order(idx_ns):
        calls["sort"] += 1
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(ts, "_fb", SimpleNamespace(sorted_index_order=_fake_sorted_index_order), raising=False)

    src_idx = np.array([30, 10, 20], dtype=np.int64)
    src_vals = np.array([3.0, 1.0, 2.0], dtype=np.float64)
    tgt_idx = np.array([10, 25, 30], dtype=np.int64)
    out = ts._align_exact_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float32, fill=0.0)

    assert out is not None
    assert calls["sort"] == 1
    np.testing.assert_allclose(out, np.array([1.0, 0.0, 3.0], dtype=np.float32))


def test_training_service_static_aligners_use_rust_bindings(monkeypatch) -> None:
    fake = SimpleNamespace(
        align_exact_values_by_ns=lambda src_idx_ns, src_vals, tgt_idx_ns, fill=0.0: np.array([5.0, 0.0, 7.0], dtype=np.float64),
        align_ffill_values_by_ns=lambda src_idx_ns, src_vals, tgt_idx_ns, fill=0.0: np.array([5.0, 6.0, 7.0], dtype=np.float64),
    )
    monkeypatch.setattr(ts, "_fb", fake, raising=False)

    src_idx = np.array([10, 20, 30], dtype=np.int64)
    src_vals = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    tgt_idx = np.array([10, 25, 30], dtype=np.int64)

    out_exact = ts.TrainingService._align_values_by_timestamp(src_idx, src_vals, tgt_idx)
    out_ffill = ts.TrainingService._align_values_ffill_by_timestamp(src_idx, src_vals, tgt_idx, default=0.0, dtype=np.float32)

    np.testing.assert_allclose(out_exact, np.array([5.0, 0.0, 7.0], dtype=np.float32))
    np.testing.assert_allclose(out_ffill, np.array([5.0, 6.0, 7.0], dtype=np.float32))


def test_align_feature_matrix_uses_rust_binding_when_available(monkeypatch) -> None:
    calls: dict[str, int] = {"count": 0}

    def _fake_align_feature_matrix(src_matrix, src_col_idx, dst_col_idx, dst_width):
        calls["count"] += 1
        return np.array([[0.0, 10.0, 0.0], [0.0, 20.0, 0.0]], dtype=np.float32)

    fake = SimpleNamespace(align_feature_matrix=_fake_align_feature_matrix)
    monkeypatch.setattr(ts, "_fb", fake, raising=False)

    src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src_cols = np.array([0], dtype=np.int64)
    dst_cols = np.array([1], dtype=np.int64)

    out = ts._align_feature_matrix(src, src_cols, dst_cols, dst_width=3)
    assert calls["count"] == 1
    np.testing.assert_allclose(out, np.array([[0.0, 10.0, 0.0], [0.0, 20.0, 0.0]], dtype=np.float32))


def test_index_to_ns_int64_normalizes_object_datetime_index() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    obj_idx = np.asarray(list(idx), dtype=object)
    expected = np.asarray([int(ts.value) for ts in list(idx)], dtype=np.int64)

    out = ts._index_to_ns_int64(obj_idx)

    assert out is not None
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, expected)


def test_is_datetime_index_accepts_object_datetime_array() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    assert ts._is_datetime_index(np.asarray(list(idx), dtype=object)) is True


def test_month_day_indices_accept_object_datetime_array_without_rust() -> None:
    idx = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    month_idx, day_idx = ts.TrainingService._month_day_indices_from_index(np.asarray(list(idx), dtype=object))
    assert month_idx.shape == (2,)
    assert day_idx.shape == (2,)
    assert int(day_idx[1]) >= int(day_idx[0])


def test_rust_sorted_index_order_uses_binding_when_available(monkeypatch) -> None:
    calls: dict[str, int] = {"count": 0}

    def _fake_sorted_index_order(idx_ns):
        calls["count"] += 1
        assert np.asarray(idx_ns, dtype=np.int64).shape[0] == 3
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(ts, "_fb", SimpleNamespace(sorted_index_order=_fake_sorted_index_order), raising=False)
    idx = pd.to_datetime(
        [
            "2025-01-03T00:00:00Z",
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z",
        ]
    )

    out = ts._rust_sorted_index_order(idx)

    assert calls["count"] == 1
    assert out is not None
    np.testing.assert_array_equal(out, np.array([1, 2, 0], dtype=np.int64))


def test_sorted_time_order_prefers_binding_for_unsorted_datetime_index(monkeypatch) -> None:
    calls: dict[str, int] = {"count": 0}

    def _fake_sorted_index_order(idx_ns):
        calls["count"] += 1
        assert np.asarray(idx_ns, dtype=np.int64).shape[0] == 3
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(ts, "_fb", SimpleNamespace(sorted_index_order=_fake_sorted_index_order), raising=False)
    idx = pd.to_datetime(
        [
            "2025-01-03T00:00:00Z",
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z",
        ]
    )

    out = ts._sorted_time_order(idx, 3)

    assert calls["count"] == 1
    assert out is not None
    np.testing.assert_array_equal(out, np.array([1, 2, 0], dtype=np.int64))


def test_rust_rank_scores_desc_uses_binding_when_available(monkeypatch) -> None:
    calls: dict[str, int] = {"count": 0}

    def _fake_rank_scores_desc(scores, absolute=False):
        calls["count"] += 1
        assert bool(absolute) is False
        np.testing.assert_allclose(np.asarray(scores, dtype=np.float64), np.array([0.2, 1.5, 0.7], dtype=np.float64))
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(ts, "_fb", SimpleNamespace(rank_scores_desc=_fake_rank_scores_desc), raising=False)

    out = ts._rust_rank_scores_desc(np.array([0.2, 1.5, 0.7], dtype=np.float64))

    assert calls["count"] == 1
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

    assert ts._sorted_time_order(idx, 3) is None


def test_sort_dedup_rows_by_index_uses_rust_binding_when_available(monkeypatch) -> None:
    calls: dict[str, int] = {"count": 0}

    def _fake_sort_dedup_rows_by_index(x, y, idx_ns):
        calls["count"] += 1
        out_x = np.array([[9.0, 1.0], [8.0, 2.0]], dtype=np.float32)
        out_y = np.array([2, 1], dtype=np.int8)
        out_idx = np.array([100, 200], dtype=np.int64)
        return out_x, out_y, out_idx

    fake = SimpleNamespace(sort_dedup_rows_by_index=_fake_sort_dedup_rows_by_index)
    monkeypatch.setattr(ts, "_fb", fake, raising=False)

    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    y = np.array([0, 1, 2], dtype=np.int8)
    idx = np.array([200, 100, 100], dtype=np.int64)

    out_x, out_y, out_idx = ts._sort_dedup_rows_by_index(x, y, idx)
    assert calls["count"] == 1
    np.testing.assert_allclose(out_x, np.array([[9.0, 1.0], [8.0, 2.0]], dtype=np.float32))
    np.testing.assert_array_equal(out_y, np.array([2, 1], dtype=np.int8))
    np.testing.assert_array_equal(out_idx, np.array([100, 200], dtype=np.int64))


def test_sort_dedup_rows_by_index_fallback_uses_rust_sorted_index_order(monkeypatch) -> None:
    calls: dict[str, int] = {"sort": 0}

    def _fake_sorted_index_order(idx_ns):
        calls["sort"] += 1
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(ts, "_fb", SimpleNamespace(sorted_index_order=_fake_sorted_index_order), raising=False)

    x = np.array([[3.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([3, 1, 2], dtype=np.int8)
    idx = np.array([300, 100, 100], dtype=np.int64)

    out_x, out_y, out_idx = ts._sort_dedup_rows_by_index(x, y, idx)

    assert calls["sort"] == 1
    np.testing.assert_allclose(out_x, np.array([[1.0], [3.0]], dtype=np.float32))
    np.testing.assert_array_equal(out_y, np.array([1, 3], dtype=np.int8))
    np.testing.assert_array_equal(out_idx, np.array([100, 300], dtype=np.int64))
