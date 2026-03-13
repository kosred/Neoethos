import numpy as np
from tests._compat_pd import pd

from forex_bot.features import talib_mixer as tm
from forex_bot.features.talib_mixer import TALibStrategyGene, TALibStrategyMixer


class _ArrayFrame:
    def __init__(self, data, index, attrs=None):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


class _DummyFunc:
    def __init__(self, name: str):
        self.name = name
        self.info = {"parameters": {}}

    def __call__(self, df, **_params):
        return pd.Series(np.linspace(0.0, 1.0, len(df)), index=df.index)


class _DummyAbstract:
    Function = _DummyFunc


def _array_ohlc_frame(rows: int = 12) -> _ArrayFrame:
    x = np.linspace(1.0, 1.1, rows, dtype=np.float64)
    return _ArrayFrame(
        {
            "open": x,
            "high": x + 0.0005,
            "low": x - 0.0005,
            "close": x,
        },
        index=np.arange(rows, dtype=np.int64),
        attrs={"symbol": "EURUSD", "timeframe": "M1"},
    )


def test_compute_signals_without_rust_cache_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "0")
    # Opt-in is intentionally ignored: mixer is Rust-only now.
    monkeypatch.setenv("FOREX_BOT_TALIB_ALLOW_PY_FALLBACK", "1")
    monkeypatch.delenv("FOREX_BOT_DISCOVERY_RUST_ONLY", raising=False)
    monkeypatch.delenv("FOREX_BOT_TREE_BACKEND", raising=False)
    monkeypatch.setattr(tm, "_RUST_TALIB_MIXER", False, raising=False)
    monkeypatch.setattr(tm, "TALIB_AVAILABLE", True)
    monkeypatch.setattr(tm, "abstract", _DummyAbstract)
    monkeypatch.setattr(tm, "ALL_INDICATORS", ["RSI"])
    monkeypatch.setattr(tm, "TALIB_INDICATORS", {"momentum": ["RSI"]})

    mixer = TALibStrategyMixer()
    idx = pd.date_range("2020-01-01", periods=50, freq="min")
    df = pd.DataFrame({"close": np.linspace(1.0, 2.0, len(idx))}, index=idx)
    gene = TALibStrategyGene(
        indicators=["RSI"],
        params={"RSI": {"timeperiod": 14}},
        weights={"RSI": 1.0},
        long_threshold=0.0,
        short_threshold=0.0,
        strategy_id="unit",
    )

    sig = mixer.compute_signals(df, gene, cache=None)
    assert len(sig) == len(df)
    np.testing.assert_array_equal(np.asarray(sig, dtype=np.int8), np.zeros(len(df), dtype=np.int8))


def test_compute_signals_defaults_to_zero_without_python_fallback(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "0")
    monkeypatch.delenv("FOREX_BOT_TALIB_ALLOW_PY_FALLBACK", raising=False)
    monkeypatch.delenv("FOREX_BOT_DISCOVERY_RUST_ONLY", raising=False)
    monkeypatch.delenv("FOREX_BOT_TREE_BACKEND", raising=False)
    monkeypatch.setattr(tm, "_RUST_TALIB_MIXER", False, raising=False)
    monkeypatch.setattr(tm, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(tm, "ALL_INDICATORS", ["RSI"], raising=False)

    mixer = TALibStrategyMixer()

    def _boom(*_args, **_kwargs):
        raise AssertionError("Python indicator compute path should be disabled by default")

    monkeypatch.setattr(mixer, "_compute_indicator", _boom, raising=True)
    idx = pd.date_range("2020-01-01", periods=24, freq="min")
    df = pd.DataFrame({"close": np.linspace(1.0, 2.0, len(idx))}, index=idx)
    gene = TALibStrategyGene(
        indicators=["RSI"],
        params={"RSI": {"timeperiod": 14}},
        weights={"RSI": 1.0},
        long_threshold=0.0,
        short_threshold=0.0,
        strategy_id="unit-zero",
    )

    sig = mixer.compute_signals(df, gene, cache=None)
    np.testing.assert_array_equal(np.asarray(sig, dtype=np.int8), np.zeros(len(df), dtype=np.int8))


def test_strict_rust_bulk_signals_enabled_by_default(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.delenv("FOREX_BOT_TALIB_RUST_BULK_SIGNALS", raising=False)
    monkeypatch.setattr(tm, "_RUST_TALIB_MIXER", True)

    class _FakeBindings:
        @staticmethod
        def talib_bulk_signals_ohlcv(
            open,
            high,
            low,
            close,
            indicator_sets,
            weight_sets=None,
            long_thresholds=None,
            short_thresholds=None,
            volume=None,
            include_raw=False,
            causal_min_bars=30,
        ):
            n_rows = len(close)
            n_genes = len(indicator_sets)
            out = np.zeros((n_rows, n_genes), dtype=np.int8)
            if n_rows > 1 and n_genes > 0:
                out[1:, :] = 1
            return out

    monkeypatch.setattr(tm, "_fb", _FakeBindings())
    monkeypatch.setattr(tm, "TALIB_AVAILABLE", True)
    monkeypatch.setattr(tm, "ALL_INDICATORS", ["RSI"])

    mixer = TALibStrategyMixer()
    idx = pd.date_range("2020-01-01", periods=8, freq="min")
    df = pd.DataFrame(
        {
            "open": np.linspace(1.0, 1.1, len(idx)),
            "high": np.linspace(1.0, 1.1, len(idx)) + 0.0005,
            "low": np.linspace(1.0, 1.1, len(idx)) - 0.0005,
            "close": np.linspace(1.0, 1.1, len(idx)),
        },
        index=idx,
    )
    gene = TALibStrategyGene(
        indicators=["RSI"],
        params={},
        weights={"RSI": 1.0},
        long_threshold=0.0,
        short_threshold=0.0,
        strategy_id="strict-rust",
    )

    cache = mixer.bulk_calculate_indicators(df, [gene])
    assert isinstance(cache, dict)
    sig = mixer.compute_signals(df, gene, cache=cache)
    assert len(sig) == len(df)
    assert int(np.asarray(sig).sum()) > 0


def test_compute_signals_uses_rust_binding_on_demand_without_precomputed_cache(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.delenv("FOREX_BOT_TALIB_RUST_BULK_SIGNALS", raising=False)
    monkeypatch.setattr(tm, "_RUST_TALIB_MIXER", True, raising=False)
    monkeypatch.setattr(tm, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(tm, "ALL_INDICATORS", ["RSI"], raising=False)

    calls = {"count": 0}

    class _FakeBindings:
        @staticmethod
        def talib_bulk_signals_ohlcv(
            open,
            high,
            low,
            close,
            indicator_sets,
            weight_sets=None,
            long_thresholds=None,
            short_thresholds=None,
            volume=None,
            include_raw=False,
            causal_min_bars=30,
        ):
            calls["count"] += 1
            n_rows = len(close)
            n_genes = len(indicator_sets)
            out = np.zeros((n_rows, n_genes), dtype=np.int8)
            if n_rows > 2 and n_genes > 0:
                out[2:, :] = 1
            return out

    monkeypatch.setattr(tm, "_fb", _FakeBindings(), raising=False)

    mixer = TALibStrategyMixer()
    idx = pd.date_range("2020-01-01", periods=9, freq="min")
    df = pd.DataFrame(
        {
            "open": np.linspace(1.0, 1.1, len(idx)),
            "high": np.linspace(1.0, 1.1, len(idx)) + 0.0005,
            "low": np.linspace(1.0, 1.1, len(idx)) - 0.0005,
            "close": np.linspace(1.0, 1.1, len(idx)),
        },
        index=idx,
    )
    gene = TALibStrategyGene(
        indicators=["RSI"],
        params={},
        weights={"RSI": 1.0},
        long_threshold=0.0,
        short_threshold=0.0,
        strategy_id="on-demand-rust",
    )

    sig = mixer.compute_signals(df, gene, cache=None)

    assert calls["count"] == 1
    np.testing.assert_array_equal(
        np.asarray(sig, dtype=np.int8),
        np.array([0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8),
    )


def test_custom_smc_indicator_computes_without_talib(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "0")
    monkeypatch.delenv("FOREX_BOT_DISCOVERY_RUST_ONLY", raising=False)
    monkeypatch.delenv("FOREX_BOT_TREE_BACKEND", raising=False)
    monkeypatch.setattr(tm, "TALIB_AVAILABLE", False, raising=False)
    monkeypatch.setattr(tm, "abstract", None, raising=False)
    monkeypatch.setattr(tm, "_RUST_TALIB_MIXER", False, raising=False)
    monkeypatch.setattr(tm, "ALL_INDICATORS", ["SMC_BOS"], raising=False)

    mixer = TALibStrategyMixer()
    idx = pd.date_range("2020-01-01", periods=64, freq="min")
    close = np.linspace(1.0, 1.2, len(idx))
    df = pd.DataFrame(
        {
            "open": close - 0.0002,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
        },
        index=idx,
    )
    raw = mixer._compute_indicator(df, "SMC_BOS", None)
    raw_arr = np.asarray(raw, dtype=np.float64)
    assert raw_arr.shape[0] == len(df)
    assert (raw_arr > 0).any()

    gene = TALibStrategyGene(
        indicators=["SMC_BOS"],
        params={},
        weights={"SMC_BOS": 1.0},
        long_threshold=0.0,
        short_threshold=0.0,
        strategy_id="smc-bos",
    )
    sig = mixer.compute_signals(df, gene, cache=None)
    assert len(sig) == len(df)


def test_strict_rust_bulk_signals_accepts_transposed_matrix(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.delenv("FOREX_BOT_TALIB_RUST_BULK_SIGNALS", raising=False)
    monkeypatch.setattr(tm, "_RUST_TALIB_MIXER", True, raising=False)
    monkeypatch.setattr(tm, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(tm, "ALL_INDICATORS", ["RSI"], raising=False)

    class _FakeBindings:
        @staticmethod
        def talib_bulk_signals_ohlcv(
            open,
            high,
            low,
            close,
            indicator_sets,
            weight_sets=None,
            long_thresholds=None,
            short_thresholds=None,
            volume=None,
            include_raw=False,
            causal_min_bars=30,
        ):
            n_rows = len(close)
            n_genes = len(indicator_sets)
            out = np.zeros((n_genes, n_rows), dtype=np.int8)
            if n_rows > 0 and n_genes > 0:
                out[:, :] = 1
            return out

    monkeypatch.setattr(tm, "_fb", _FakeBindings(), raising=False)

    mixer = TALibStrategyMixer()
    idx = pd.date_range("2020-01-01", periods=10, freq="min")
    df = pd.DataFrame(
        {
            "open": np.linspace(1.0, 1.1, len(idx)),
            "high": np.linspace(1.0, 1.1, len(idx)) + 0.0005,
            "low": np.linspace(1.0, 1.1, len(idx)) - 0.0005,
            "close": np.linspace(1.0, 1.1, len(idx)),
        },
        index=idx,
    )
    gene = TALibStrategyGene(
        indicators=["RSI"],
        params={},
        weights={"RSI": 1.0},
        long_threshold=0.0,
        short_threshold=0.0,
        strategy_id="strict-rust-transposed",
    )

    cache = mixer.bulk_calculate_indicators(df, [gene])
    assert isinstance(cache, dict)
    sig = mixer.compute_signals(df, gene, cache=cache)
    assert len(sig) == len(df)
    np.testing.assert_array_equal(np.asarray(sig, dtype=np.int8), np.ones(len(df), dtype=np.int8))


def test_strict_rust_bulk_signals_accepts_numpy_frame(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.delenv("FOREX_BOT_TALIB_RUST_BULK_SIGNALS", raising=False)
    monkeypatch.setattr(tm, "_RUST_TALIB_MIXER", True, raising=False)
    monkeypatch.setattr(tm, "TALIB_AVAILABLE", True, raising=False)
    monkeypatch.setattr(tm, "ALL_INDICATORS", ["RSI"], raising=False)

    class _FakeBindings:
        @staticmethod
        def talib_bulk_signals_ohlcv(
            open,
            high,
            low,
            close,
            indicator_sets,
            weight_sets=None,
            long_thresholds=None,
            short_thresholds=None,
            volume=None,
            include_raw=False,
            causal_min_bars=30,
        ):
            n_rows = len(close)
            n_genes = len(indicator_sets)
            out = np.zeros((n_rows, n_genes), dtype=np.int8)
            if n_rows > 0 and n_genes > 0:
                out[:, :] = 1
            return out

    monkeypatch.setattr(tm, "_fb", _FakeBindings(), raising=False)

    mixer = TALibStrategyMixer()
    df = _array_ohlc_frame(rows=14)
    gene = TALibStrategyGene(
        indicators=["RSI"],
        params={},
        weights={"RSI": 1.0},
        long_threshold=0.0,
        short_threshold=0.0,
        strategy_id="strict-rust-numpy-frame",
    )

    cache = mixer.bulk_calculate_indicators(df, [gene])
    assert isinstance(cache, dict)
    sig = mixer.compute_signals(df, gene, cache=cache)
    assert len(sig) == len(df)
    np.testing.assert_array_equal(np.asarray(sig, dtype=np.int8), np.ones(len(df), dtype=np.int8))

