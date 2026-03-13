from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np

from forex_bot.execution.mt5_state_manager import MT5Position
from forex_bot.execution.trade_doctor import TradeDoctor


class _ArrayFrame:
    def __init__(self, data, index):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


class _DummyExitAgent:
    def __init__(self, settings):
        self.settings = settings

    def load(self, _path: str) -> None:
        return None

    def get_action(self, _state):
        return 0.0

    def observe_exit(self, *_args, **_kwargs) -> None:
        return None


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        dynamic={"doctor_params": {}},
        risk=SimpleNamespace(time_stop_bars=8),
        system=SimpleNamespace(symbol="EURUSD", base_timeframe="M5"),
    )


def _buy_position(ticket: int = 1001, *, minutes_ago: float = 120.0) -> MT5Position:
    return MT5Position(
        ticket=ticket,
        symbol="EURUSD",
        volume=0.10,
        price_open=1.0000,
        price_current=1.1000,
        sl=0.9980,
        tp=1.1200,
        profit=0.0,
        swap=0.0,
        commission=0.0,
        time=datetime.now(UTC) - timedelta(minutes=float(minutes_ago)),
        type=0,
        magic=12345,
    )


def test_trade_doctor_diagnose_accepts_numpy_frame(monkeypatch):
    from forex_bot.execution import trade_doctor as td

    monkeypatch.setattr(td, "ExitAgent", _DummyExitAgent, raising=True)
    doctor = TradeDoctor(_settings())

    n = 16
    close = np.full(n, 1.1000, dtype=np.float64)
    frame = _ArrayFrame(
        {
            "close": close,
            "rsi": np.full(n, 55.0, dtype=np.float64),
            "atr": np.full(n, 0.0010, dtype=np.float64),
        },
        index=np.arange(n, dtype=np.int64),
    )

    out = doctor.diagnose([_buy_position()], {"M1": frame})
    assert isinstance(out, list)
    assert len(out) == 1
    assert int(out[0].ticket) == 1001
    assert "Numba" in str(out[0].reason)


def test_trade_doctor_diagnose_uses_default_rsi_atr_when_missing(monkeypatch):
    from forex_bot.execution import trade_doctor as td

    monkeypatch.setattr(td, "ExitAgent", _DummyExitAgent, raising=True)
    doctor = TradeDoctor(_settings())

    n = 20
    close = np.linspace(1.0000, 1.1000, n, dtype=np.float64)
    frame = _ArrayFrame({"close": close}, index=np.arange(n, dtype=np.int64))

    out = doctor.diagnose([_buy_position(ticket=1002, minutes_ago=5.0)], {"M5": frame})
    assert isinstance(out, list)
    assert all(int(x.ticket) == 1002 for x in out) or len(out) == 0
