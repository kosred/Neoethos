from __future__ import annotations

from types import SimpleNamespace

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


def test_optimizer_infer_meta_trading_days_uses_rust_binding_when_available(monkeypatch) -> None:
    fake = type(
        "_Fake",
        (),
        {
            "count_weekday_trading_days": staticmethod(lambda index_ns: 7),
        },
    )()
    monkeypatch.setattr(opt_mod, "_fb", fake, raising=False)

    idx_ns = np.array(
        [
            1_704_067_200_000_000_000,
            1_704_153_600_000_000_000,
            1_704_240_000_000_000_000,
        ],
        dtype=np.int64,
    )
    opt = object.__new__(HyperparameterOptimizer)

    days = HyperparameterOptimizer._infer_meta_trading_days(opt, {"index": idx_ns})

    assert days == 7.0


def test_optimizer_objective_metrics_uses_shared_threshold_helper(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_thresholded(probs, *, conf_threshold, y_true=None, classes=None):
        calls["count"] += 1
        return np.array([1, 0, -1], dtype=np.int8), 0.5

    monkeypatch.setattr(opt_mod, "threshold_signals_and_accuracy", _fake_thresholded, raising=False)

    opt = object.__new__(HyperparameterOptimizer)
    opt.prop_conf_threshold = 0.66
    opt.settings = SimpleNamespace(
        risk=SimpleNamespace(),
        system=SimpleNamespace(symbol="EURUSD"),
    )

    metrics = HyperparameterOptimizer._objective_metrics(
        opt,
        np.array([1, 0, -1], dtype=np.int64),
        np.array(
            [
                [0.1, 0.8, 0.1],
                [0.6, 0.3, 0.1],
                [0.1, 0.2, 0.7],
            ],
            dtype=np.float64,
        ),
        None,
    )

    assert metrics["prop_score"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert calls["count"] == 1


def test_optimizer_prop_score_aggregation_prefers_rust_binding(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_aggregate(
        net_profit,
        sortino,
        drawdown,
        profit_factor,
        trades,
        daily_dd,
        months,
        dd_limit,
        daily_dd_limit,
        min_monthly,
        initial_balance,
        acc,
        prop_weight,
        acc_weight,
        win_rate=None,
        include_win_rate_bonus=False,
        ignore_zero_trade_entries=True,
    ):
        calls["count"] += 1
        return (1.25, 0.2, 0.1, 5.0, 0.04, 0.6, 0.2, 1.7)

    fake = type("_Fake", (), {"aggregate_prop_score_metrics": staticmethod(_fake_aggregate)})()
    monkeypatch.setattr(opt_mod, "_fb", fake, raising=False)

    summary = opt_mod._aggregate_prop_score_metrics(
        net_profit=np.array([10.0, 20.0], dtype=np.float64),
        sortino=np.array([0.5, 0.7], dtype=np.float64),
        drawdown=np.array([0.1, 0.2], dtype=np.float64),
        profit_factor=np.array([1.2, 1.6], dtype=np.float64),
        trades=np.array([2.0, 3.0], dtype=np.float64),
        daily_dd=np.array([0.02, 0.03], dtype=np.float64),
        months=1.0,
        dd_limit=0.08,
        daily_dd_limit=0.04,
        min_monthly=0.04,
        initial_balance=100000.0,
        acc=0.55,
        prop_weight=1.0,
        acc_weight=0.1,
    )

    assert summary["prop_score"] == 1.25
    assert summary["drawdown"] == 0.2
    assert summary["daily_dd"] == 0.1
    assert summary["trades"] == 5.0
    assert summary["monthly_return"] == 0.04
    assert summary["sortino"] == 0.6
    assert summary["calmar"] == 0.2
    assert summary["profit_factor"] == 1.7
    assert calls["count"] == 1


def test_optimizer_single_symbol_zero_trade_preserves_previous_score(monkeypatch) -> None:
    monkeypatch.setattr(opt_mod, "_fb", None, raising=False)

    summary = opt_mod._aggregate_prop_score_metrics(
        net_profit=np.array([2_000.0], dtype=np.float64),
        sortino=np.array([1.2], dtype=np.float64),
        drawdown=np.array([0.05], dtype=np.float64),
        profit_factor=np.array([1.4], dtype=np.float64),
        trades=np.array([0.0], dtype=np.float64),
        daily_dd=np.array([0.02], dtype=np.float64),
        months=1.0,
        dd_limit=0.08,
        daily_dd_limit=0.04,
        min_monthly=0.04,
        initial_balance=100_000.0,
        acc=0.55,
        prop_weight=1.0,
        acc_weight=0.1,
        win_rate=np.array([0.60], dtype=np.float64),
        include_win_rate_bonus=True,
        ignore_zero_trade_entries=False,
    )

    np.testing.assert_allclose(summary["prop_score"], 4.635, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["drawdown"], 0.05, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["daily_dd"], 0.02, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["trades"], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["monthly_return"], 0.02, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["sortino"], 1.2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["calmar"], 0.4, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["profit_factor"], 1.4, rtol=0, atol=1e-12)


def test_optimizer_prop_score_aggregation_falls_back_to_python(monkeypatch) -> None:
    monkeypatch.setattr(opt_mod, "_fb", None, raising=False)

    summary = opt_mod._aggregate_prop_score_metrics(
        net_profit=np.array([10_000.0, 5_000.0], dtype=np.float64),
        sortino=np.array([1.0, 0.5], dtype=np.float64),
        drawdown=np.array([0.10, 0.20], dtype=np.float64),
        profit_factor=np.array([1.4, 1.2], dtype=np.float64),
        trades=np.array([4.0, 1.0], dtype=np.float64),
        daily_dd=np.array([0.03, 0.02], dtype=np.float64),
        months=2.0,
        dd_limit=0.08,
        daily_dd_limit=0.04,
        min_monthly=0.04,
        initial_balance=100_000.0,
        acc=0.55,
        prop_weight=1.0,
        acc_weight=0.1,
    )

    np.testing.assert_allclose(summary["prop_score"], 3.066625, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["drawdown"], 0.2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["daily_dd"], 0.03, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["trades"], 5.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["monthly_return"], 0.045, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["sortino"], 0.9, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["calmar"], 0.425, rtol=0, atol=1e-12)
    np.testing.assert_allclose(summary["profit_factor"], 1.36, rtol=0, atol=1e-12)


def test_optimizer_prop_score_aggregation_matches_real_rust_binding() -> None:
    if opt_mod._fb is None or not hasattr(opt_mod._fb, "aggregate_prop_score_metrics"):
        return

    multi_expected = opt_mod._aggregate_prop_score_metrics_python(
        net_profit=np.array([10_000.0, 5_000.0], dtype=np.float64),
        sortino=np.array([1.0, 0.5], dtype=np.float64),
        drawdown=np.array([0.10, 0.20], dtype=np.float64),
        profit_factor=np.array([1.4, 1.2], dtype=np.float64),
        trades=np.array([4.0, 1.0], dtype=np.float64),
        daily_dd=np.array([0.03, 0.02], dtype=np.float64),
        months=2.0,
        dd_limit=0.08,
        daily_dd_limit=0.04,
        min_monthly=0.04,
        initial_balance=100_000.0,
        acc=0.55,
        prop_weight=1.0,
        acc_weight=0.1,
    )
    multi_actual_tuple = opt_mod._fb.aggregate_prop_score_metrics(
        np.array([10_000.0, 5_000.0], dtype=np.float64),
        np.array([1.0, 0.5], dtype=np.float64),
        np.array([0.10, 0.20], dtype=np.float64),
        np.array([1.4, 1.2], dtype=np.float64),
        np.array([4.0, 1.0], dtype=np.float64),
        np.array([0.03, 0.02], dtype=np.float64),
        2.0,
        0.08,
        0.04,
        0.04,
        100_000.0,
        0.55,
        1.0,
        0.1,
    )
    multi_actual = {
        "prop_score": float(multi_actual_tuple[0]),
        "drawdown": float(multi_actual_tuple[1]),
        "daily_dd": float(multi_actual_tuple[2]),
        "trades": float(multi_actual_tuple[3]),
        "monthly_return": float(multi_actual_tuple[4]),
        "sortino": float(multi_actual_tuple[5]),
        "calmar": float(multi_actual_tuple[6]),
        "profit_factor": float(multi_actual_tuple[7]),
    }
    for key in multi_expected:
        np.testing.assert_allclose(multi_actual[key], multi_expected[key], rtol=0, atol=1e-12)

    zero_trade_expected = opt_mod._aggregate_prop_score_metrics_python(
        net_profit=np.array([2_000.0], dtype=np.float64),
        sortino=np.array([1.2], dtype=np.float64),
        drawdown=np.array([0.05], dtype=np.float64),
        profit_factor=np.array([1.4], dtype=np.float64),
        trades=np.array([0.0], dtype=np.float64),
        daily_dd=np.array([0.02], dtype=np.float64),
        months=1.0,
        dd_limit=0.08,
        daily_dd_limit=0.04,
        min_monthly=0.04,
        initial_balance=100_000.0,
        acc=0.55,
        prop_weight=1.0,
        acc_weight=0.1,
        win_rate=np.array([0.60], dtype=np.float64),
        include_win_rate_bonus=True,
        ignore_zero_trade_entries=False,
    )
    zero_trade_actual_tuple = opt_mod._fb.aggregate_prop_score_metrics(
        np.array([2_000.0], dtype=np.float64),
        np.array([1.2], dtype=np.float64),
        np.array([0.05], dtype=np.float64),
        np.array([1.4], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.02], dtype=np.float64),
        1.0,
        0.08,
        0.04,
        0.04,
        100_000.0,
        0.55,
        1.0,
        0.1,
        win_rate=np.array([0.60], dtype=np.float64),
        include_win_rate_bonus=True,
        ignore_zero_trade_entries=False,
    )
    zero_trade_actual = {
        "prop_score": float(zero_trade_actual_tuple[0]),
        "drawdown": float(zero_trade_actual_tuple[1]),
        "daily_dd": float(zero_trade_actual_tuple[2]),
        "trades": float(zero_trade_actual_tuple[3]),
        "monthly_return": float(zero_trade_actual_tuple[4]),
        "sortino": float(zero_trade_actual_tuple[5]),
        "calmar": float(zero_trade_actual_tuple[6]),
        "profit_factor": float(zero_trade_actual_tuple[7]),
    }
    for key in zero_trade_expected:
        np.testing.assert_allclose(zero_trade_actual[key], zero_trade_expected[key], rtol=0, atol=1e-12)
