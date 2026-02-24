from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.run_prop_discovery import DiscoveryStateStore


def test_discovery_state_truth_and_forward_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("FOREX_BOT_PROP_MIN_TRUTH_PROBABILITY", "0.70")
    monkeypatch.setenv("FOREX_BOT_PROP_FORWARD_TEST_REQUIRED", "1")
    monkeypatch.setenv("FOREX_BOT_PROP_ANOMALY_GUARD", "0")
    monkeypatch.delenv("FOREX_BOT_PROP_KEEP_MIN_SHARPE", raising=False)
    monkeypatch.delenv("FOREX_BOT_PROP_KEEP_MIN_WIN_RATE", raising=False)
    monkeypatch.delenv("FOREX_BOT_PROP_KEEP_MIN_PROFIT_FACTOR", raising=False)

    store = DiscoveryStateStore(
        tmp_path / "state.sqlite",
        profit_key="net_profit",
        threshold=0.0,
        min_trades=10.0,
        max_dd=None,
    )
    try:
        ok, *_ = store._is_profitable(
            {
                "net_profit": 1000.0,
                "trades": 25.0,
                "truth_probability": 0.80,
                "forward_test_passed": True,
            },
            history_months=12.0,
        )
        assert ok

        low_truth, *_ = store._is_profitable(
            {
                "net_profit": 1000.0,
                "trades": 25.0,
                "truth_probability": 0.60,
                "forward_test_passed": True,
            },
            history_months=12.0,
        )
        assert not low_truth

        no_forward, *_ = store._is_profitable(
            {
                "net_profit": 1000.0,
                "trades": 25.0,
                "truth_probability": 0.95,
                "forward_test_passed": False,
            },
            history_months=12.0,
        )
        assert not no_forward
    finally:
        store.close()
