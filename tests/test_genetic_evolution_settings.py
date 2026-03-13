import numpy as np
from tests._compat_pd import pd

from forex_bot.strategy import genetic as genmod
from forex_bot.strategy import fast_backtest as fb


class _DummyMixer:
    available_indicators = ["RSI", "SMA"]

    def compute_signals(self, df, gene):
        return pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)


def test_genetic_evolution_settings_default(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_GENETIC_STRICT_RUST", raising=False)
    monkeypatch.delenv("FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK", raising=False)

    mixer = _DummyMixer()
    evo = genmod.GeneticStrategyEvolution(population_size=1, mixer=mixer)
    gene = genmod.GeneticGene(
        indicators=["RSI", "SMA"],
        params={"RSI": {}, "SMA": {}},
        weights={"RSI": 1.0, "SMA": 1.0},
    )
    evo.population = [gene]

    idx = pd.date_range("2020-01-01", periods=100, freq="min")
    df = pd.DataFrame(
        {
            "close": np.linspace(1.0, 2.0, len(idx)),
            "high": np.linspace(1.05, 2.05, len(idx)),
            "low": np.linspace(0.95, 1.95, len(idx)),
        },
        index=idx,
    )

    def _fake_eval(*_args, **_kwargs):
        return np.array([100.0, 1.0, 1.0, 0.01, 0.6, 1.2, 0.2, 1.0, 10, 0.5, 0.01])

    monkeypatch.setattr(fb, "fast_evaluate_strategy", _fake_eval)

    evo._evaluate_population(df, evo.population)
    assert evo.population[0].evaluated is True
    assert evo.population[0].fitness == float("-inf")


def test_genetic_evolution_settings_allow_python_fallback_env_is_ignored(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_GENETIC_STRICT_RUST", raising=False)
    monkeypatch.setenv("FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK", "1")

    mixer = _DummyMixer()
    evo = genmod.GeneticStrategyEvolution(population_size=1, mixer=mixer)
    gene = genmod.GeneticGene(
        indicators=["RSI", "SMA"],
        params={"RSI": {}, "SMA": {}},
        weights={"RSI": 1.0, "SMA": 1.0},
    )
    evo.population = [gene]

    idx = pd.date_range("2020-01-01", periods=100, freq="min")
    df = pd.DataFrame(
        {
            "close": np.linspace(1.0, 2.0, len(idx)),
            "high": np.linspace(1.05, 2.05, len(idx)),
            "low": np.linspace(0.95, 1.95, len(idx)),
        },
        index=idx,
    )

    def _fake_eval(*_args, **_kwargs):
        return np.array([100.0, 1.0, 1.0, 0.01, 0.6, 1.2, 0.2, 1.0, 10, 0.5, 0.01])

    monkeypatch.setattr(fb, "fast_evaluate_strategy", _fake_eval)

    evo._evaluate_population(df, evo.population)
    assert evo.population[0].evaluated is True
    assert evo.population[0].fitness == float("-inf")

