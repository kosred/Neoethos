from __future__ import annotations

from types import SimpleNamespace

from forex_bot.core.config import Settings
from forex_bot.strategy.discovery_tensor import _passes_quality, _strategy_quality_limits


def test_strategy_quality_limits_use_settings_defaults(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_DISCOVERY_MIN_SHARPE", raising=False)
    monkeypatch.delenv("FOREX_BOT_DISCOVERY_MIN_PROFIT_FACTOR", raising=False)
    monkeypatch.delenv("FOREX_BOT_DISCOVERY_MIN_WIN_RATE", raising=False)

    settings = Settings()
    settings.models.prop_search_holdout_min_sharpe = 1.7
    settings.models.prop_search_holdout_min_profit_factor = 1.35

    min_sharpe, min_pf, min_wr = _strategy_quality_limits(settings)
    assert min_sharpe == 1.7
    assert min_pf == 1.35
    assert min_wr == 0.0


def test_strategy_quality_limits_env_overrides(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_MIN_SHARPE", "2.1")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_MIN_PROFIT_FACTOR", "1.8")
    monkeypatch.setenv("FOREX_BOT_DISCOVERY_MIN_WIN_RATE", "0.55")

    min_sharpe, min_pf, min_wr = _strategy_quality_limits(None)
    assert min_sharpe == 2.1
    assert min_pf == 1.8
    assert min_wr == 0.55


def test_passes_quality_handles_percent_win_rate():
    gene = SimpleNamespace(sharpe_ratio=1.6, profit_factor=1.4, win_rate=55.0)
    assert _passes_quality(
        gene,
        min_sharpe=1.5,
        min_profit_factor=1.3,
        min_win_rate=0.50,
    )

    weak = SimpleNamespace(sharpe_ratio=1.2, profit_factor=1.1, win_rate=0.48)
    assert not _passes_quality(
        weak,
        min_sharpe=1.5,
        min_profit_factor=1.3,
        min_win_rate=0.50,
    )

