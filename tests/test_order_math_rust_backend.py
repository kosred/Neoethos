from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pandas as pd

from forex_bot.core.config import Settings
from forex_bot.execution.order_execution import OrderExecutor


class _DummyRiskManager:
    def __init__(self) -> None:
        self._spread_state: dict[str, float] = {}

    @staticmethod
    def _session_bucket_utc(_dt) -> str:
        return "london"

    @staticmethod
    def _compute_pip_metrics(_info):
        return 0.0001, 10.0

    @staticmethod
    def update_spread_state(**_kwargs) -> None:
        return None


def _make_executor() -> OrderExecutor:
    settings = Settings()
    settings.system.symbol = "EURUSD"
    settings.system.base_timeframe = "M1"
    risk = _DummyRiskManager()
    mt5 = SimpleNamespace()
    ex = OrderExecutor(settings=settings, risk_manager=risk, mt5_manager=mt5)
    ex._last_rr = 2.0
    return ex


def test_get_pip_size_uses_rust_binding(monkeypatch):
    ex = _make_executor()
    fake = types.SimpleNamespace(
        pip_size_from_symbol=lambda symbol, point=None, digits=None: 0.00042,
        compute_order_prices=lambda *args, **kwargs: (0.0, 0.0, 0.0),
        evaluate_trade_edge=lambda *args, **kwargs: (True, 0.0, 0.0),
    )
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)
    monkeypatch.setattr("forex_bot.execution.order_execution._rust_order_backend_available", lambda **_: True)
    got = ex._get_pip_size("EURUSD", {"point": 0.00001, "digits": 5})
    assert abs(got - 0.00042) < 1e-12


def test_calculate_prices_uses_rust_binding(monkeypatch):
    ex = _make_executor()
    fake = types.SimpleNamespace(
        pip_size_from_symbol=lambda symbol, point=None, digits=None: 0.0001,
        compute_order_prices=lambda entry_price, signal, sl_pips, rr, pip_size: (1.001, 1.123, 0.0025),
        evaluate_trade_edge=lambda *args, **kwargs: (True, 10.0, 2.0),
    )
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)
    monkeypatch.setattr("forex_bot.execution.order_execution._rust_order_backend_available", lambda **_: True)
    result = SimpleNamespace(signal=1, recommended_rr=None)
    frames = {"M1": pd.DataFrame({"close": [1.1000]})}
    out = ex._calculate_prices(result, frames, sl_pips=25.0, info={"point": 0.00001, "digits": 5}, tick_price={"ask": 1.1002, "bid": 1.1000})
    assert out is not None
    sl, tp, entry, sl_dist, rr = out
    assert abs(sl - 1.001) < 1e-12
    assert abs(tp - 1.123) < 1e-12
    assert abs(sl_dist - 0.0025) < 1e-12
    assert abs(entry - 1.1002) < 1e-12
    assert abs(rr - 2.0) < 1e-12


def test_edge_over_cost_uses_rust_binding(monkeypatch):
    ex = _make_executor()
    fake = types.SimpleNamespace(
        pip_size_from_symbol=lambda symbol, point=None, digits=None: 0.0001,
        compute_order_prices=lambda *args, **kwargs: (0.0, 0.0, 0.0),
        evaluate_trade_edge=lambda *args, **kwargs: (False, 4.0, 3.0),
    )
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)
    monkeypatch.setattr("forex_bot.execution.order_execution._rust_order_backend_available", lambda **_: True)
    passed = ex._edge_over_cost_ok(
        sl_pips=20.0,
        rr=2.0,
        tick={"ask": 1.1002, "bid": 1.1000},
        symbol_info={"point": 0.00001, "digits": 5},
    )
    assert passed is False
