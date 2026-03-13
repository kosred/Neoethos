from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tests._compat_pd import pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forex_bot.features.talib_mixer import TALibStrategyGene
from forex_bot.strategy import evo_prop


def _settings(*, required: bool = True, holdout_years: float = 3.0, min_truth: float = 0.70):
    return SimpleNamespace(
        models=SimpleNamespace(
            prop_search_holdout_fraction=0.20,
            prop_search_holdout_min_rows=500,
            prop_search_holdout_min_sharpe=1.20,
            prop_search_holdout_min_win_rate=0.52,
            prop_search_holdout_min_profit_factor=1.30,
            prop_search_holdout_min_trades=20,
            prop_search_holdout_required=required,
            prop_search_holdout_years=holdout_years,
            prop_search_holdout_min_truth_probability=min_truth,
        ),
        risk=SimpleNamespace(total_drawdown_limit=0.07),
    )


def _price_df(days: int = 2400) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=days, freq="D", tz="UTC")
    x = np.linspace(1.0, 1.4, len(idx), dtype=np.float64)
    return pd.DataFrame(
        {
            "open": x,
            "high": x + 0.0010,
            "low": x - 0.0010,
            "close": x + 0.0002,
        },
        index=idx,
    )


def test_split_discovery_holdout_prefers_last_n_years():
    df = _price_df(days=2400)
    search_df, holdout_df = evo_prop._split_discovery_holdout(df, _settings(required=True, holdout_years=3.0))
    assert holdout_df is not None
    assert not holdout_df.empty
    assert len(search_df) > 0
    hold_days, _hold_months = evo_prop._history_span_days_months(holdout_df)
    assert hold_days >= 365.0 * 2.8
    assert holdout_df.index.min() > search_df.index.max()


def test_holdout_validation_drops_low_truth_probability(monkeypatch):
    df = _price_df(days=1200)

    class _DummyMixer:
        def __init__(self):
            self.available_indicators = ["RSI"]

        def bulk_calculate_indicators(self, df, genes):
            return {}

    def _stub_eval(df, gene, mixer, cache, settings):
        gene.fitness = 500.0
        gene.net_profit = 500.0
        gene.sharpe_ratio = 1.30
        gene.win_rate = 0.54
        gene.profit_factor = 1.35
        gene.trades = 25.0
        gene.max_dd_pct = 0.03
        gene.expectancy = 20.0
        return gene.fitness

    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _DummyMixer, raising=False)
    monkeypatch.setattr(evo_prop, "_evaluate_gene", _stub_eval, raising=False)
    monkeypatch.setattr(evo_prop, "_strategy_passes_filter", lambda *args, **kwargs: True, raising=False)

    gene = TALibStrategyGene(indicators=["RSI"], strategy_id="low_truth")
    gene.net_profit = 80_000.0
    gene.sharpe_ratio = 2.20
    gene.win_rate = 0.62
    gene.profit_factor = 1.80
    gene.trades = 300.0
    gene.max_dd_pct = 0.04

    out = evo_prop._apply_holdout_validation(
        selected=[gene],
        holdout_df=df,
        settings=_settings(required=True, holdout_years=3.0, min_truth=0.70),
        max_dd=0.10,
        min_profit=0.0,
        min_trades=1.0,
        initial_balance=100_000.0,
        search_history_months=24.0,
    )
    assert out == []


def test_holdout_validation_keeps_high_truth_probability(monkeypatch):
    df = _price_df(days=1200)

    class _DummyMixer:
        def __init__(self):
            self.available_indicators = ["RSI"]

        def bulk_calculate_indicators(self, df, genes):
            return {}

    def _stub_eval(df, gene, mixer, cache, settings):
        gene.fitness = 5_000.0
        gene.net_profit = 5_000.0
        gene.sharpe_ratio = 1.80
        gene.win_rate = 0.58
        gene.profit_factor = 1.40
        gene.trades = 60.0
        gene.max_dd_pct = 0.02
        gene.expectancy = 80.0
        return gene.fitness

    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _DummyMixer, raising=False)
    monkeypatch.setattr(evo_prop, "_evaluate_gene", _stub_eval, raising=False)
    monkeypatch.setattr(evo_prop, "_strategy_passes_filter", lambda *args, **kwargs: True, raising=False)

    gene = TALibStrategyGene(indicators=["RSI"], strategy_id="high_truth")
    gene.net_profit = 10_000.0
    gene.sharpe_ratio = 2.00
    gene.win_rate = 0.60
    gene.profit_factor = 1.50
    gene.trades = 120.0
    gene.max_dd_pct = 0.03

    out = evo_prop._apply_holdout_validation(
        selected=[gene],
        holdout_df=df,
        settings=_settings(required=True, holdout_years=3.0, min_truth=0.70),
        max_dd=0.10,
        min_profit=0.0,
        min_trades=1.0,
        initial_balance=100_000.0,
        search_history_months=24.0,
    )
    assert len(out) == 1
    assert bool(out[0].forward_test_passed)
    assert float(out[0].truth_probability) >= 0.70

