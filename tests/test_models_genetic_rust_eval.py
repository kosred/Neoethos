from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
from tests._compat_pd import pd

from forex_bot.features.talib_mixer import TALibStrategyGene
from forex_bot.models import genetic as mgmod
from forex_bot.models.genetic import GeneticStrategyExpert


class _DummyMixer:
    def __init__(self) -> None:
        self.available_indicators = ["RSI"]
        self._next = 0

    def generate_random_strategy(self, *, max_indicators: int = 1) -> TALibStrategyGene:
        sid = f"g{self._next}"
        self._next += 1
        return TALibStrategyGene(
            indicators=["RSI"],
            params={},
            weights={"RSI": 1.0},
            long_threshold=0.5,
            short_threshold=-0.5,
            strategy_id=sid,
        )

    @staticmethod
    def bulk_calculate_indicators(df, population):  # pragma: no cover - rust-first test path
        return {}

    @staticmethod
    def compute_signals(df, gene, cache=None):  # pragma: no cover - rust-first test path
        return np.zeros(len(df), dtype=np.float64)


def _ohlc_df(n: int = 64):
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    close = np.linspace(1.10, 1.20, n, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": close - 0.0001,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
        },
        index=idx,
    )


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

    def copy(self) -> _ArrayFrame:
        return _ArrayFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            np.asarray(self.index).copy(),
            dict(self.attrs),
        )


def _ohlc_numpy_frame(n: int = 64) -> _ArrayFrame:
    close = np.linspace(1.10, 1.20, n, dtype=np.float64)
    idx = np.datetime64("2025-01-01T00:00:00") + np.arange(n, dtype=np.int64) * np.timedelta64(1, "m")
    return _ArrayFrame(
        {
            "open": close - 0.0001,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
            "volume": np.full(n, 100.0, dtype=np.float64),
        },
        idx,
        attrs={"symbol": "EURUSD"},
    )


def test_genetic_strategy_expert_rust_eval_accepts_numpy_frame(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _eval_pop(open_, high, low, close, **kwargs):
        captured.update(kwargs)
        n = int(len(kwargs.get("indicator_sets") or []))
        out = np.zeros((n, 11), dtype=np.float64)
        if n > 0:
            out[:, 1] = 1.5
            out[:, 8] = 20.0
        return out

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(evaluate_population_talib_ohlcv=_eval_pop))
    model = GeneticStrategyExpert(population_size=1, generations=1, max_indicators=1)
    gene = TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="nf")

    scores = model._evaluate_population_rust(_ohlc_numpy_frame(48), [gene])
    assert scores is not None
    assert len(scores) == 1
    assert scores[0] == 1.5
    ts = captured.get("timestamps")
    assert ts is not None
    assert np.asarray(ts, dtype=np.int64).shape[0] == 48


def test_genetic_strategy_expert_predict_proba_accepts_numpy_frame(monkeypatch) -> None:
    def _bulk(open_arr, high_arr, low_arr, close_arr, **kwargs):
        n = int(len(close_arr))
        m = int(len(kwargs.get("indicator_sets") or []))
        out = np.zeros((m, n), dtype=np.int8)
        if m > 0:
            out[:, :] = 1
        return out

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(talib_bulk_signals_ohlcv=_bulk))

    model = GeneticStrategyExpert(population_size=1, generations=1, max_indicators=1)
    model.portfolio = [TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="p0")]

    meta = _ohlc_numpy_frame(24)
    x_np = np.zeros((len(meta), 2), dtype=np.float32)
    probs = model.predict_proba(x_np, metadata=meta)

    assert probs.shape == (len(meta), 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(len(meta), dtype=np.float64), rtol=0.0, atol=1e-6)


def test_genetic_strategy_expert_align_ffill_prefers_rust_binding(monkeypatch) -> None:
    calls = {"align": 0}

    def _align(src_idx_ns, src_vals, tgt_idx_ns, fill):
        calls["align"] += 1
        assert float(fill) == 0.0
        return np.asarray([0.0, 1.0, 1.0, -1.0], dtype=np.float64)

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(align_ffill_values_by_ns=_align))
    out = GeneticStrategyExpert._align_ffill_by_ns(
        np.asarray([10, 30], dtype=np.int64),
        np.asarray([1.0, -1.0], dtype=np.float64),
        np.asarray([5, 10, 20, 30], dtype=np.int64),
        dtype=np.float64,
    )

    assert out is not None
    assert calls["align"] == 1
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 1.0, -1.0], dtype=np.float64))


def test_genetic_strategy_expert_align_ffill_fallback_prefers_rust_sorted_index_order(monkeypatch) -> None:
    calls = {"sort": 0}

    def _sorted_index_order(idx_ns):
        calls["sort"] += 1
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(sorted_index_order=_sorted_index_order))
    out = GeneticStrategyExpert._align_ffill_by_ns(
        np.asarray([30, 10, 20], dtype=np.int64),
        np.asarray([3.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([10, 25, 30], dtype=np.int64),
        dtype=np.float64,
    )

    assert out is not None
    assert calls["sort"] == 1
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_genetic_strategy_expert_sorted_time_order_returns_none_when_monotonic() -> None:
    out = GeneticStrategyExpert._sorted_time_order(np.asarray([10, 20, 30], dtype=np.int64), 3)
    assert out is None


def test_genetic_strategy_expert_datetime_helpers_accept_object_datetime_array() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    obj_idx = np.asarray(list(idx), dtype=object)

    assert GeneticStrategyExpert._is_datetime_index(obj_idx) is True

    out = GeneticStrategyExpert._index_to_ns(obj_idx)

    assert out is not None
    np.testing.assert_array_equal(out, np.asarray([int(ts.value) for ts in list(idx)], dtype=np.int64))


def test_genetic_strategy_expert_fit_prefers_rust_score_ranking(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    calls = {"eval": 0, "rank": 0}

    def _eval_pop(open_, high, low, close, **kwargs):
        calls["eval"] += 1
        n = int(len(kwargs.get("indicator_sets") or []))
        out = np.zeros((n, 11), dtype=np.float64)
        if n == 4:
            out[:, 1] = np.array([1.0, 4.0, 2.0, 3.0], dtype=np.float64)
            out[:, 8] = 20.0
        return out

    def _rank_scores_desc(scores, absolute=False):
        calls["rank"] += 1
        assert bool(absolute) is False
        np.testing.assert_allclose(np.asarray(scores, dtype=np.float64), np.array([1.0, 4.0, 2.0, 3.0], dtype=np.float64))
        return np.array([1, 3, 2, 0], dtype=np.int64)

    monkeypatch.setitem(
        sys.modules,
        "forex_bindings",
        SimpleNamespace(evaluate_population_talib_ohlcv=_eval_pop, rank_scores_desc=_rank_scores_desc),
    )

    model = GeneticStrategyExpert(population_size=4, generations=1, max_indicators=1)
    model.mixer = _DummyMixer()
    model.max_indicators = 1

    df = _ohlc_df(80)
    y = np.zeros(len(df), dtype=np.int8)
    x = np.zeros((len(df), 1), dtype=np.float32)
    model.fit(x, y, metadata=df)

    assert calls["eval"] >= 1
    assert calls["rank"] >= 1
    assert model.best_gene is not None
    assert model.best_gene.strategy_id == "g1"


def test_genetic_strategy_expert_fit_prefers_rust_population_eval(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    calls = {"eval": 0}

    def _eval_pop(open_, high, low, close, **kwargs):
        calls["eval"] += 1
        n = int(len(kwargs.get("indicator_sets") or []))
        out = np.zeros((n, 11), dtype=np.float64)
        if n > 0:
            out[:, 1] = np.arange(1, n + 1, dtype=np.float64)  # sharpe
            out[:, 8] = 20.0  # trades
        return out

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(evaluate_population_talib_ohlcv=_eval_pop))

    model = GeneticStrategyExpert(population_size=4, generations=1, max_indicators=1)
    model.mixer = _DummyMixer()
    model.max_indicators = 1

    df = _ohlc_df(80)
    y = np.zeros(len(df), dtype=np.int8)
    x = np.zeros((len(df), 1), dtype=np.float32)
    model.fit(x, y, metadata=df)

    assert calls["eval"] >= 1
    assert model.best_gene is not None
    assert float(model.best_gene.fitness) == 4.0
    assert model.best_gene.strategy_id == "g3"


def test_genetic_strategy_expert_rust_eval_applies_trade_gate(monkeypatch) -> None:
    def _eval_pop(open_, high, low, close, **kwargs):
        n = int(len(kwargs.get("indicator_sets") or []))
        out = np.zeros((n, 11), dtype=np.float64)
        if n >= 2:
            out[0, 1] = 2.5
            out[0, 8] = 9.0  # below trade gate -> -1
            out[1, 1] = 1.0
            out[1, 8] = 20.0
        return out

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(evaluate_population_talib_ohlcv=_eval_pop))

    model = GeneticStrategyExpert(population_size=2, generations=1, max_indicators=1)
    genes = [
        TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="a"),
        TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="b"),
    ]
    scores = model._evaluate_population_rust(_ohlc_df(48), genes)

    assert scores is not None
    assert len(scores) == 2
    assert scores[0] == -1.0
    assert scores[1] == 1.0


def test_genetic_strategy_expert_rust_eval_passes_symbol_pip_and_smc_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}
    calls = {"pip": 0}

    def _eval_pop(open_, high, low, close, **kwargs):
        captured.update(kwargs)
        n = int(len(kwargs.get("indicator_sets") or []))
        out = np.zeros((n, 11), dtype=np.float64)
        if n > 0:
            out[:, 1] = 1.0
            out[:, 8] = 20.0
        return out

    def _infer(symbol: str, **_kwargs):
        calls["pip"] += 1
        captured["symbol"] = symbol
        return 0.01, 25.0

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(evaluate_population_talib_ohlcv=_eval_pop))
    monkeypatch.setattr(mgmod, "infer_pip_metrics", _infer, raising=False)

    model = GeneticStrategyExpert(population_size=1, generations=1, max_indicators=1)
    gene = TALibStrategyGene(
        indicators=["RSI"],
        params={},
        weights={"RSI": 1.0},
        strategy_id="smc",
        use_ob=True,
        use_bos=True,
        use_choch=True,
    )
    df = _ohlc_df(32)
    df.attrs["symbol"] = "USDJPY"

    scores = model._evaluate_population_rust(df, [gene])
    assert scores is not None
    assert len(scores) == 1
    assert scores[0] == 1.0
    assert calls["pip"] == 1
    assert captured.get("symbol") == "USDJPY"
    assert float(captured.get("pip_value", 0.0)) == 0.01
    assert float(captured.get("pip_value_per_lot", 0.0)) == 25.0
    assert list(captured.get("use_ob_flags") or []) == [1]
    assert list(captured.get("use_bos_flags") or []) == [1]
    assert list(captured.get("use_choch_flags") or []) == [1]


def test_genetic_strategy_expert_rust_eval_uses_bulk_batch_bridge_when_population_api_missing(monkeypatch) -> None:
    calls = {"bulk": 0, "batch": 0}

    def _bulk(open_, high, low, close, **kwargs):
        calls["bulk"] += 1
        n = int(len(close))
        m = int(len(kwargs.get("indicator_sets") or []))
        out = np.zeros((n, m), dtype=np.int8)
        if m > 0:
            out[:, 0] = 1
        if m > 1:
            out[:, 1] = -1
        return out

    def _batch(**kwargs):
        calls["batch"] += 1
        rows = int(np.asarray(kwargs["signals"]).shape[0])
        out = np.zeros((rows, 11), dtype=np.float64)
        out[:, 1] = np.arange(1, rows + 1, dtype=np.float64)
        out[:, 8] = 20.0
        return out

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(talib_bulk_signals_ohlcv=_bulk))
    monkeypatch.setattr(mgmod, "batch_evaluate_strategies", _batch, raising=False)
    monkeypatch.setattr(mgmod, "infer_pip_metrics", lambda _symbol, **_kwargs: (0.0001, 10.0), raising=False)

    model = GeneticStrategyExpert(population_size=2, generations=1, max_indicators=1)
    genes = [
        TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="a"),
        TALibStrategyGene(indicators=["SMA"], params={}, weights={"SMA": 1.0}, strategy_id="b"),
    ]
    scores = model._evaluate_population_rust(_ohlc_df(48), genes)

    assert calls["bulk"] == 1
    assert calls["batch"] == 1
    assert scores is not None
    assert len(scores) == 2
    assert scores[0] == 1.0
    assert scores[1] == 2.0


def test_genetic_strategy_expert_predict_proba_ignores_python_fallback_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("FOREX_BOT_GENETIC_STRICT_RUST", raising=False)
    monkeypatch.setenv("FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK", "1")

    class _PredictMixer:
        def __init__(self) -> None:
            self.available_indicators = ["RSI"]
            self.bulk_calls = 0
            self.signal_calls = 0
            self.cache_seen = 0

        def bulk_calculate_indicators(self, df, population):
            self.bulk_calls += 1
            return {"sentinel": 7}

        def compute_signals(self, df, gene, cache=None):
            self.signal_calls += 1
            if isinstance(cache, dict) and cache.get("sentinel") == 7:
                self.cache_seen += 1
            return np.ones(len(df), dtype=np.float64)

    model = GeneticStrategyExpert(population_size=2, generations=1, max_indicators=1)
    model.mixer = _PredictMixer()
    model.portfolio = [
        TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="p0"),
        TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="p1"),
    ]
    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace())

    x = _ohlc_df(24)
    probs = model.predict_proba(x, metadata=x)

    assert probs.shape == (len(x), 3)
    assert model.mixer.bulk_calls == 0
    assert model.mixer.signal_calls == 0
    assert model.mixer.cache_seen == 0
    np.testing.assert_allclose(
        probs,
        np.full((len(x), 3), 1.0 / 3.0, dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(len(x), dtype=np.float64), rtol=0.0, atol=1e-6)


def test_genetic_strategy_expert_predict_proba_prefers_rust_bulk(monkeypatch) -> None:
    calls = {"bulk": 0}

    def _bulk(open_arr, high_arr, low_arr, close_arr, **kwargs):
        calls["bulk"] += 1
        n = int(len(close_arr))
        m = int(len(kwargs.get("indicator_sets") or []))
        out = np.zeros((m, n), dtype=np.int8)  # transposed orientation
        if m > 0:
            out[:, :] = 1
        return out

    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace(talib_bulk_signals_ohlcv=_bulk))

    class _NoPythonMixer:
        def __init__(self) -> None:
            self.available_indicators = ["RSI"]

        def bulk_calculate_indicators(self, df, population):  # pragma: no cover - should not be used
            raise AssertionError("Python fallback cache path should not be used when Rust bulk succeeds")

        def compute_signals(self, df, gene, cache=None):  # pragma: no cover - should not be used
            raise AssertionError("Python fallback signal path should not be used when Rust bulk succeeds")

    model = GeneticStrategyExpert(population_size=2, generations=1, max_indicators=1)
    model.mixer = _NoPythonMixer()
    model.portfolio = [
        TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="p0"),
        TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="p1"),
    ]

    x = _ohlc_df(24)
    probs = model.predict_proba(x, metadata=x)

    assert calls["bulk"] == 1
    assert probs.shape == (len(x), 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(len(x), dtype=np.float64), rtol=0.0, atol=1e-6)


def test_genetic_strategy_expert_predict_proba_numpy_x_uses_metadata_index() -> None:
    class _PredictMixer:
        def __init__(self) -> None:
            self.available_indicators = ["RSI"]

        def bulk_calculate_indicators(self, df, population):
            return {"sentinel": 1}

        def compute_signals(self, df, gene, cache=None):
            return np.ones(len(df), dtype=np.float64)

    model = GeneticStrategyExpert(population_size=1, generations=1, max_indicators=1)
    model.mixer = _PredictMixer()
    model.portfolio = [TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="p0")]

    meta = _ohlc_df(16)
    x_np = np.zeros((len(meta), 2), dtype=np.float32)
    probs = model.predict_proba(x_np, metadata=meta)

    assert probs.shape == (len(meta), 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(len(meta), dtype=np.float64), rtol=0.0, atol=1e-6)


def test_genetic_strategy_expert_fit_strict_rust_skips_python_fallback(monkeypatch, tmp_path) -> None:
    class _StrictMixer:
        def __init__(self) -> None:
            self.available_indicators = ["RSI"]
            self._next = 0

        def generate_random_strategy(self, *, max_indicators: int = 1) -> TALibStrategyGene:
            sid = f"s{self._next}"
            self._next += 1
            return TALibStrategyGene(
                indicators=["RSI"],
                params={},
                weights={"RSI": 1.0},
                long_threshold=0.4,
                short_threshold=-0.4,
                strategy_id=sid,
            )

        def bulk_calculate_indicators(self, df, population):  # pragma: no cover - should not be used
            raise AssertionError("Python fallback should not run in strict Rust mode")

        def compute_signals(self, df, gene, cache=None):  # pragma: no cover - should not be used
            raise AssertionError("Python fallback should not run in strict Rust mode")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOREX_BOT_GENETIC_STRICT_RUST", "1")
    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace())

    model = GeneticStrategyExpert(population_size=3, generations=1, max_indicators=1)
    model.mixer = _StrictMixer()
    model.max_indicators = 1

    df = _ohlc_df(64)
    y = np.zeros(len(df), dtype=np.int8)
    x = np.zeros((len(df), 1), dtype=np.float32)
    model.fit(x, y, metadata=df)

    assert model.best_gene is not None
    assert float(model.best_gene.fitness) == -1.0


def test_genetic_strategy_expert_predict_proba_strict_rust_returns_neutral_without_rust_bulk(monkeypatch) -> None:
    class _StrictMixer:
        def __init__(self) -> None:
            self.available_indicators = ["RSI"]

        def bulk_calculate_indicators(self, df, population):  # pragma: no cover - should not be used
            raise AssertionError("Python fallback should not run in strict Rust mode")

        def compute_signals(self, df, gene, cache=None):  # pragma: no cover - should not be used
            raise AssertionError("Python fallback should not run in strict Rust mode")

    monkeypatch.setenv("FOREX_BOT_GENETIC_STRICT_RUST", "1")
    monkeypatch.setitem(sys.modules, "forex_bindings", SimpleNamespace())

    model = GeneticStrategyExpert(population_size=1, generations=1, max_indicators=1)
    model.mixer = _StrictMixer()
    model.portfolio = [TALibStrategyGene(indicators=["RSI"], params={}, weights={"RSI": 1.0}, strategy_id="p0")]

    meta = _ohlc_df(12)
    probs = model.predict_proba(meta, metadata=meta)

    assert probs.shape == (len(meta), 3)
    np.testing.assert_allclose(probs, np.full((len(meta), 3), 1.0 / 3.0, dtype=np.float64), rtol=0.0, atol=1e-9)

