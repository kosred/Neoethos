from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "mod_name",
    [
        "forex_bot.training.optimization",
        "forex_bot.training.benchmark_service",
        "forex_bot.training.ensemble",
        "forex_bot.training.evaluation",
        "forex_bot.training.evaluation_service",
        "forex_bot.training.walkforward",
    ],
)
def test_training_modules_import_with_pandas_block(monkeypatch: pytest.MonkeyPatch, mod_name: str) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    mod = importlib.import_module(mod_name)
    importlib.reload(mod)
