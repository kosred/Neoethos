from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "mod_name",
    [
        "forex_bot.execution.news_service",
        "forex_bot.execution.drift_monitor",
        "forex_bot.execution.trading_loop",
        "forex_bot.execution.bot",
    ],
)
def test_execution_modules_import_with_pandas_block(monkeypatch: pytest.MonkeyPatch, mod_name: str) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    mod = importlib.import_module(mod_name)
    importlib.reload(mod)
