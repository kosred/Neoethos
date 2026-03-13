from __future__ import annotations

import sys
import types

from forex_bot.core.config import Settings
from forex_bot.execution.risk import RiskManager


def _build_risk_manager(tmp_path, monkeypatch) -> RiskManager:
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    settings.system.symbol = "EURUSD_TEST"
    settings.system.trading_session_start = "00:00"
    settings.system.trading_session_end = "23:59"
    settings.risk.block_night_session = False
    rm = RiskManager(settings)
    rm.meta_controller.get_risk_parameters = lambda _state: (1.0, 0.0, True)  # type: ignore[method-assign]
    return rm


def test_rust_only_blocks_python_position_sizing_when_backend_missing(tmp_path, monkeypatch) -> None:
    rm = _build_risk_manager(tmp_path, monkeypatch)
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setenv("FOREX_BOT_RUST_RISK", "1")
    monkeypatch.setattr("forex_bot.execution.risk._rust_risk_backend_available", lambda **_: False)

    size = rm.calculate_position_size(
        equity=10_000.0,
        stop_loss_pips=20.0,
        confidence=0.8,
        uncertainty=0.0,
        symbol_info={"point": 0.00001, "digits": 5},
        market_regime="trend",
        market_volatility=0.0015,
    )
    assert size == 0.0


def test_position_sizing_blocks_when_rust_backend_missing_even_without_rust_only(tmp_path, monkeypatch) -> None:
    rm = _build_risk_manager(tmp_path, monkeypatch)
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.setattr("forex_bot.execution.risk._rust_risk_backend_available", lambda **_: False)

    size = rm.calculate_position_size(
        equity=10_000.0,
        stop_loss_pips=20.0,
        confidence=0.8,
        uncertainty=0.0,
        symbol_info={"point": 0.00001, "digits": 5},
        market_regime="trend",
        market_volatility=0.0015,
    )
    assert size == 0.0


def test_rust_only_blocks_python_position_sizing_when_rust_call_fails(tmp_path, monkeypatch) -> None:
    rm = _build_risk_manager(tmp_path, monkeypatch)
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setenv("FOREX_BOT_RUST_RISK", "1")
    monkeypatch.setattr("forex_bot.execution.risk._rust_risk_backend_available", lambda **_: True)

    fake = types.SimpleNamespace(
        compute_position_size_lots=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        infer_pip_metrics=lambda *_args, **_kwargs: (0.0001, 10.0),
    )
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)

    size = rm.calculate_position_size(
        equity=10_000.0,
        stop_loss_pips=20.0,
        confidence=0.8,
        uncertainty=0.0,
        symbol_info={"point": 0.00001, "digits": 5},
        market_regime="trend",
        market_volatility=0.0015,
    )
    assert size == 0.0
