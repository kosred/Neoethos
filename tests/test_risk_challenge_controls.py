from __future__ import annotations

import sys
import types
from datetime import datetime

from forex_bot.core.config import Settings
from forex_bot.execution.risk import RiskManager


def _build_risk_manager(tmp_path, monkeypatch) -> RiskManager:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("FOREX_BOT_RUST_RISK", raising=False)
    settings = Settings()
    settings.system.symbol = "EURUSD_TEST"
    settings.system.trading_session_start = "00:00"
    settings.system.trading_session_end = "23:59"
    settings.risk.block_night_session = False
    settings.risk.challenge_mode = True
    settings.risk.challenge_phase = "phase_1"
    rm = RiskManager(settings)
    monkeypatch.setattr("forex_bot.execution.risk._rust_risk_backend_available", lambda **_: True)

    def _compute_position_size_lots(
        *,
        equity: float,
        risk_pct: float,
        stop_loss_pips: float,
        pip_value: float,
        max_lot_size: float,
        lot_step: float = 0.01,
        min_lot: float = 0.0,
    ) -> float:
        raw = float(equity) * float(risk_pct) / max(float(stop_loss_pips) * float(pip_value), 1e-9)
        raw = max(float(min_lot), min(float(max_lot_size), raw))
        step = max(float(lot_step), 1e-6)
        return float(int(raw / step) * step)

    monkeypatch.setitem(
        sys.modules,
        "forex_bindings",
        types.SimpleNamespace(
            compute_position_size_lots=_compute_position_size_lots,
            infer_pip_metrics=lambda *_args, **_kwargs: (0.0001, 10.0),
        ),
    )
    rm.is_trading_session = lambda: True  # type: ignore[method-assign]
    return rm


def test_pre_stop_drawdown_gate_blocks_before_hard_stop(tmp_path, monkeypatch) -> None:
    rm = _build_risk_manager(tmp_path, monkeypatch)
    now = datetime.now(rm._session_tz)
    rm._last_session_date = now.date()
    rm.day_start_equity = 10_000.0
    rm.day_peak_equity = 10_000.0
    rm.total_peak_equity = 10_000.0

    # 3.65% DD with a 4.0% hard stop should trigger pre-stop brake at 90% of limit.
    allowed, reason = rm.check_trade_allowed(
        equity=9_635.0,
        confidence=0.95,
        timestamp=now,
        market_volatility=0.0020,
        ensemble_disagreement=0.0,
    )
    assert not allowed
    assert "pre-stop" in reason.lower()


def test_drawdown_soft_brakes_reduce_position_size_progressively(tmp_path, monkeypatch) -> None:
    rm = _build_risk_manager(tmp_path, monkeypatch)
    rm.meta_controller.get_risk_parameters = lambda _state: (1.0, 0.0, True)  # type: ignore[method-assign]
    rm._last_session_date = datetime.now(rm._session_tz).date()
    rm.day_start_equity = 10_000.0
    rm.day_peak_equity = 10_000.0
    rm.total_peak_equity = 10_000.0

    kwargs = {
        "stop_loss_pips": 10.0,
        "confidence": 0.85,
        "uncertainty": 0.0,
        "symbol_info": {"digits": 5, "point": 0.00001},
        "market_regime": "trend",
        "market_volatility": 0.0015,
    }
    size_low = rm.calculate_position_size(equity=10_000.0, **kwargs)
    size_mid = rm.calculate_position_size(equity=9_720.0, **kwargs)  # ~2.8% DD
    size_high = rm.calculate_position_size(equity=9_660.0, **kwargs)  # ~3.4% DD

    assert size_low > 0.0
    assert size_high < size_mid < size_low


def test_challenge_progress_multiplier_boosts_when_behind_and_cuts_when_ahead(tmp_path, monkeypatch) -> None:
    rm = _build_risk_manager(tmp_path, monkeypatch)
    now = datetime(2026, 2, 23, 12, 0, tzinfo=rm._session_tz)
    rm.challenge_start_date = datetime(2026, 1, 1, tzinfo=rm._session_tz).date()
    rm.challenge_start_equity = 10_000.0
    rm.settings.risk.challenge_target_return_pct = 0.10
    rm.settings.risk.challenge_target_trading_days = 44

    # Behind pace: tiny return far into the schedule -> boost.
    boost = rm._challenge_progress_multiplier(10_020.0, now, daily_dd_pct=0.0)
    # Ahead pace: large early return -> reduce.
    reduce = rm._challenge_progress_multiplier(11_000.0, now, daily_dd_pct=0.0)

    assert boost > 1.0
    assert reduce < 1.0
