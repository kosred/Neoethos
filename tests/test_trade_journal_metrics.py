from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tests._compat_pd import pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forex_bot.strategy import evo_prop


def test_trade_journal_reports_hours_activity_and_monthly_breakdown(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_PROP_SWAP_LONG_PER_DAY", "24.0")
    monkeypatch.setenv("FOREX_BOT_PROP_SWAP_SHORT_PER_DAY", "24.0")

    idx = pd.date_range("2025-01-01", periods=12, freq="h", tz="UTC")
    close = np.array(
        [
            1.0000,
            1.0000,
            1.0011,
            1.0010,
            0.9998,
            0.9999,
            1.0000,
            1.0001,
            1.0002,
            1.0003,
            1.0004,
            1.0005,
        ],
        dtype=np.float64,
    )
    high = close + 0.0002
    low = close - 0.0002
    high[2] = 1.0012  # Long TP hit.
    low[4] = 0.9997  # Short TP hit.

    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    sig = np.zeros(len(df), dtype=np.int8)
    sig[0] = 1
    sig[2] = -1

    journal = evo_prop._trade_journal_from_signals(
        df=df,
        signals=sig,
        sl_pips=10.0,
        tp_pips=10.0,
        pip_value=0.0001,
        pip_value_per_lot=10.0,
        spread_pips=0.0,
        commission_per_trade=0.0,
    )

    assert bool(journal.get("computed", False))
    assert float(journal.get("trade_count", 0.0)) == 2.0
    assert float(journal.get("wins", 0.0)) == 2.0
    assert float(journal.get("losses", 0.0)) == 0.0
    assert float(journal.get("avg_holding_hours", 0.0)) > 0.9
    assert float(journal.get("avg_trades_per_day", 0.0)) > 0.0
    assert float(journal.get("avg_trades_per_month", 0.0)) > 0.0
    assert float(journal.get("profit_per_trade", 0.0)) > 0.0
    assert float(journal.get("avg_trade_dd_pct", 0.0)) >= 0.0
    assert float(journal.get("net_profit_no_swap", 0.0)) > float(journal.get("net_profit", 0.0))

    monthly = journal.get("monthly", {})
    assert isinstance(monthly, dict)
    assert "2025-01" in monthly
    m = monthly["2025-01"]
    assert float(m.get("trades", 0.0)) == 2.0
    assert float(m.get("wins", 0.0)) == 2.0
    assert float(m.get("losses", 0.0)) == 0.0
    assert float(m.get("win_rate", 0.0)) == 1.0


def test_trade_journal_prefers_rust_binding_when_available(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    close = np.array([1.0, 1.0002, 1.0004, 1.0001, 1.0003, 1.0005], dtype=np.float64)
    high = close + 0.0002
    low = close - 0.0002
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    sig = np.array([1, 0, -1, 0, 0, 0], dtype=np.int8)

    calls: dict[str, object] = {}

    class _FakeFB:
        @staticmethod
        def trade_journal_metrics(*args, **kwargs):
            calls["called"] = True
            return {
                "computed": True,
                "trade_count": 1.0,
                "wins": 1.0,
                "losses": 0.0,
                "avg_holding_hours": 1.0,
                "avg_trades_per_day": 1.0,
                "avg_trades_per_month": 30.0,
                "profit_per_trade": 10.0,
                "avg_trade_dd_pct": 0.0,
                "net_profit": 10.0,
                "net_profit_no_swap": 12.0,
                "monthly": {"2025-01": {"trades": 1.0, "wins": 1.0, "losses": 0.0, "win_rate": 1.0}},
            }

    monkeypatch.setattr(evo_prop, "_fb", _FakeFB(), raising=True)
    monkeypatch.setattr(evo_prop, "_RUST_TRADE_JOURNAL", True, raising=True)

    journal = evo_prop._trade_journal_from_signals(
        df=df,
        signals=sig,
        sl_pips=10.0,
        tp_pips=10.0,
        pip_value=0.0001,
        pip_value_per_lot=10.0,
        spread_pips=0.0,
        commission_per_trade=0.0,
    )

    assert calls.get("called") is True
    assert bool(journal.get("computed", False))
    assert float(journal.get("trade_count", 0.0)) == 1.0

