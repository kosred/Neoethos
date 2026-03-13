from __future__ import annotations

import numpy as np
from tests._compat_pd import pd

from forex_bot.strategy import genetic as genmod


def _df(rows: int = 64) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC")
    close = np.linspace(1.0, 1.1, rows, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.0004,
            "low": close - 0.0004,
            "close": close,
            "volume": np.full(rows, 100.0, dtype=np.float64),
        },
        index=idx,
    )


def test_genetic_evolution_default_mode_is_strict_rust(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_GENETIC_STRICT_RUST", raising=False)
    monkeypatch.delenv("FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK", raising=False)
    assert genmod.GeneticStrategyEvolution._strict_rust_requested() is True


def test_genetic_evolution_allow_python_fallback_opt_in(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_GENETIC_STRICT_RUST", raising=False)
    monkeypatch.setenv("FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK", "1")
    assert genmod.GeneticStrategyEvolution._strict_rust_requested() is True


def test_genetic_evolution_prefers_rust_population_eval(monkeypatch):
    evo = genmod.GeneticStrategyEvolution(population_size=1, mixer=None)
    gene = genmod.GeneticGene(
        indicators=["rsi", "sma"],
        params={"RSI": {}, "SMA": {}},
        weights={"RSI": 1.0, "SMA": 0.8},
    )
    evo.population = [gene]
    captured: dict[str, object] = {}

    class _DummyBindings:
        def evaluate_population_talib_ohlcv(self, *_args, **kwargs):
            captured.update(kwargs)
            return np.asarray([[222.0, 1.2, 0.0, 0.02, 0.55, 1.4, 5.0, 0.0, 12.0, 0.0, 0.0]], dtype=np.float64)

    monkeypatch.setattr(genmod, "_fb", _DummyBindings(), raising=False)
    monkeypatch.setattr(genmod, "_RUST_TALIB_POP", True, raising=False)

    evo._evaluate_population(_df(), evo.population)
    assert bool(evo.population[0].evaluated)
    assert abs(float(evo.population[0].fitness) - 222.0) < 1e-9
    assert list(captured.get("indicator_sets") or []) == [["RSI", "SMA"]]


def test_genetic_evolution_python_fallback_opt_in_is_ignored(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_GENETIC_STRICT_RUST", raising=False)
    monkeypatch.setenv("FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK", "1")

    class _DummyMixer:
        def __init__(self) -> None:
            self.bulk_calls = 0
            self.signal_calls = 0
            self.cache_seen = 0

        def bulk_calculate_indicators(self, df, population):
            self.bulk_calls += 1
            return {"sentinel": 1}

        def compute_signals(self, df, gene, cache=None):
            self.signal_calls += 1
            if isinstance(cache, dict) and cache.get("sentinel") == 1:
                self.cache_seen += 1
            return np.ones(len(df), dtype=np.int8)

    mixer = _DummyMixer()
    evo = genmod.GeneticStrategyEvolution(population_size=2, mixer=mixer)
    evo.population = [
        genmod.GeneticGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}),
        genmod.GeneticGene(indicators=["SMA"], params={}, weights={"SMA": 1.0}),
    ]

    monkeypatch.setattr(genmod, "_RUST_TALIB_POP", False, raising=False)
    monkeypatch.setattr(genmod, "_fb", None, raising=False)
    monkeypatch.setattr(genmod.fb, "infer_pip_metrics", lambda _symbol: (0.0001, 10.0), raising=False)
    monkeypatch.setattr(genmod.fb, "fast_evaluate_strategy", lambda **_kwargs: np.asarray([5.0], dtype=np.float64), raising=False)

    evo._evaluate_population(_df(), evo.population)

    assert mixer.bulk_calls == 0
    assert mixer.signal_calls == 0
    assert mixer.cache_seen == 0
    assert all(bool(g.evaluated) for g in evo.population)
    assert all(g.fitness == float("-inf") for g in evo.population)


def test_genetic_evolution_uses_rust_bulk_plus_batch_bridge(monkeypatch):
    class _NoPythonMixer:
        def bulk_calculate_indicators(self, df, population):  # pragma: no cover - should not be used
            raise AssertionError("Python cache fallback should not run when Rust bulk bridge succeeds")

        def compute_signals(self, df, gene, cache=None):  # pragma: no cover - should not be used
            raise AssertionError("Python signal fallback should not run when Rust bulk bridge succeeds")

    calls = {"bulk": 0, "batch": 0}

    class _DummyBindings:
        def talib_bulk_signals_ohlcv(self, open_, high, low, close, **kwargs):
            calls["bulk"] += 1
            n = int(len(close))
            m = int(len(kwargs.get("indicator_sets") or []))
            out = np.zeros((n, m), dtype=np.int8)
            if m > 0:
                out[:, 0] = 1
            if m > 1:
                out[:, 1] = -1
            return out

    def _batch_eval(close_prices, high_prices, low_prices, signals, month_indices, day_indices, sl_pips, tp_pips, **kwargs):
        calls["batch"] += 1
        rows = int(np.asarray(signals).shape[0])
        out = np.zeros((rows, 11), dtype=np.float64)
        out[:, 0] = np.arange(10, 10 + rows, dtype=np.float64)
        return out

    evo = genmod.GeneticStrategyEvolution(population_size=2, mixer=_NoPythonMixer())
    evo.population = [
        genmod.GeneticGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}),
        genmod.GeneticGene(indicators=["SMA"], params={}, weights={"SMA": 1.0}),
    ]

    monkeypatch.setattr(genmod, "_RUST_TALIB_POP", False, raising=False)
    monkeypatch.setattr(genmod, "_fb", _DummyBindings(), raising=False)
    monkeypatch.setattr(genmod.fb, "infer_pip_metrics", lambda _symbol: (0.0001, 10.0), raising=False)
    monkeypatch.setattr(genmod.fb, "batch_evaluate_strategies", _batch_eval, raising=False)

    evo._evaluate_population(_df(), evo.population)

    assert calls["bulk"] == 1
    assert calls["batch"] == 1
    assert all(bool(g.evaluated) for g in evo.population)
    assert abs(float(evo.population[0].fitness) - 10.0) < 1e-9
    assert abs(float(evo.population[1].fitness) - 11.0) < 1e-9


def test_genetic_evolution_strict_rust_skips_python_fallback(monkeypatch):
    class _DummyMixer:
        def bulk_calculate_indicators(self, df, population):
            return {}

        def compute_signals(self, df, gene, cache=None):  # pragma: no cover - should not be used
            raise AssertionError("Python fallback should not run in strict Rust mode")

    evo = genmod.GeneticStrategyEvolution(population_size=1, mixer=_DummyMixer())
    evo.population = [genmod.GeneticGene(indicators=["RSI"], params={}, weights={"RSI": 1.0})]

    monkeypatch.setenv("FOREX_BOT_GENETIC_STRICT_RUST", "1")
    monkeypatch.setattr(genmod, "_RUST_TALIB_POP", False, raising=False)
    monkeypatch.setattr(genmod, "_fb", None, raising=False)
    monkeypatch.setattr(genmod.fb, "fast_evaluate_strategy", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not run")), raising=False)

    evo._evaluate_population(_df(), evo.population)

    assert bool(evo.population[0].evaluated)
    assert evo.population[0].fitness == float("-inf")


def test_genetic_evolution_accepts_numpy_frame(monkeypatch):
    class _ArrayFrame:
        def __init__(self, data: dict[str, np.ndarray], index: np.ndarray, attrs: dict[str, str] | None = None) -> None:
            self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
            self.columns = list(self._data.keys())
            self.index = np.asarray(index).reshape(-1)
            self.attrs = dict(attrs or {})

        @property
        def empty(self) -> bool:
            return len(self.index) <= 0

        def __len__(self) -> int:
            return int(self.index.shape[0])

        def __getitem__(self, key: str) -> np.ndarray:
            return self._data[str(key)]

    n = 64
    close = np.linspace(1.0, 1.1, n, dtype=np.float64)
    frame = _ArrayFrame(
        {
            "open": close,
            "high": close + 0.0004,
            "low": close - 0.0004,
            "close": close,
            "volume": np.full(n, 100.0, dtype=np.float64),
        },
        np.datetime64("2024-01-01T00:00:00")
        + np.arange(n, dtype=np.int64) * np.timedelta64(1, "m"),
        attrs={"symbol": "EURUSD"},
    )

    captured: dict[str, object] = {}

    class _DummyBindings:
        def evaluate_population_talib_ohlcv(self, *_args, **kwargs):
            captured.update(kwargs)
            return np.asarray([[123.0, 1.2, 0.0, 0.02, 0.55, 1.4, 5.0, 0.0, 12.0, 0.0, 0.0]], dtype=np.float64)

    evo = genmod.GeneticStrategyEvolution(population_size=1, mixer=None)
    evo.population = [
        genmod.GeneticGene(
            indicators=["RSI"],
            params={"RSI": {}},
            weights={"RSI": 1.0},
        )
    ]
    monkeypatch.setattr(genmod, "_fb", _DummyBindings(), raising=False)
    monkeypatch.setattr(genmod, "_RUST_TALIB_POP", True, raising=False)

    evo._evaluate_population(frame, evo.population)

    assert bool(evo.population[0].evaluated)
    assert abs(float(evo.population[0].fitness) - 123.0) < 1e-9
    ts = captured.get("timestamps")
    assert ts is not None
    assert np.asarray(ts, dtype=np.int64).shape[0] == n


def test_genetic_evolution_month_day_indices_uses_rust_binding(monkeypatch):
    calls = {"derive": 0}

    class _DummyBindings:
        @staticmethod
        def derive_time_index_arrays(idx_ns):
            calls["derive"] += 1
            n = int(np.asarray(idx_ns, dtype=np.int64).shape[0])
            return (
                np.arange(n, dtype=np.int64),
                np.full(n, 77, dtype=np.int64),
                np.full(n, 9901, dtype=np.int64),
            )

    monkeypatch.setattr(genmod, "_fb", _DummyBindings(), raising=False)
    month_idx, day_idx = genmod.GeneticStrategyEvolution._month_day_indices(
        np.array(
            [
                1_704_067_200_000_000_000,
                1_704_067_260_000_000_000,
                1_704_067_320_000_000_000,
            ],
            dtype=np.int64,
        ),
        3,
    )

    assert calls["derive"] == 1
    np.testing.assert_array_equal(month_idx, np.array([77, 77, 77], dtype=np.int64))
    np.testing.assert_array_equal(day_idx, np.array([9901, 9901, 9901], dtype=np.int64))


def test_genetic_evolution_datetime_helpers_accept_object_datetime_array() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    obj_idx = np.asarray(list(idx), dtype=object)

    assert genmod.GeneticStrategyEvolution._is_datetime_index(obj_idx) is True

    unix_ms = genmod.GeneticStrategyEvolution._datetime_index_to_unix_ms(obj_idx)

    assert unix_ms is not None
    np.testing.assert_array_equal(
        np.asarray(unix_ms, dtype=np.int64),
        np.asarray([int(ts.value) // 1_000_000 for ts in list(idx)], dtype=np.int64),
    )

