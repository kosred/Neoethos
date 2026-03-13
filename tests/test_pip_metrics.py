from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from forex_bot.strategy import fast_backtest as fb
from forex_bot.strategy.fast_backtest import infer_pip_metrics


def test_infer_pip_metrics_eurusd_standard_lot_usd_account() -> None:
    pip_size, pip_value = infer_pip_metrics("EURUSD", price=1.10)
    assert pip_size == 0.0001
    assert math.isclose(pip_value, 10.0, rel_tol=1e-12, abs_tol=1e-12)


def test_infer_pip_metrics_usdjpy_uses_price_conversion() -> None:
    pip_size, pip_value = infer_pip_metrics("USDJPY", price=150.0)
    assert pip_size == 0.01
    assert math.isclose(pip_value, 1000.0 / 150.0, rel_tol=1e-12, abs_tol=1e-12)


def test_infer_pip_metrics_cross_pair_uses_reference_quote_to_usd() -> None:
    pip_size, pip_value = infer_pip_metrics(
        "EURGBP",
        price=0.85,
        reference_prices={"GBPUSD": 1.27},
    )
    assert pip_size == 0.0001
    assert math.isclose(pip_value, 12.7, rel_tol=1e-12, abs_tol=1e-12)


def test_infer_pip_metrics_cross_jpy_uses_base_to_usd_reference() -> None:
    pip_size, pip_value = infer_pip_metrics(
        "EURJPY",
        price=160.0,
        reference_prices={"EURUSD": 1.08},
    )
    assert pip_size == 0.01
    assert math.isclose(pip_value, 1000.0 * (1.08 / 160.0), rel_tol=1e-12, abs_tol=1e-12)


def test_infer_pip_metrics_xauusd_standard_contract() -> None:
    pip_size, pip_value = infer_pip_metrics("XAUUSD", price=2000.0)
    assert pip_size == 0.01
    assert math.isclose(pip_value, 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_infer_pip_metrics_prefers_rust_binding_when_available(monkeypatch) -> None:
    fake = SimpleNamespace(
        infer_pip_metrics=lambda symbol, price=None, account_currency="USD", reference_prices=None: (0.00042, 42.0)
    )
    monkeypatch.setattr(fb, "_forex_bindings", fake, raising=False)
    pip_size, pip_value = fb.infer_pip_metrics("EURUSD", price=1.10)
    assert pip_size == 0.00042
    assert pip_value == 42.0


def test_infer_pip_metrics_raises_when_rust_backend_missing(monkeypatch) -> None:
    monkeypatch.setattr(fb, "_forex_bindings", None, raising=False)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        fb.infer_pip_metrics("EURUSD", price=1.10)
