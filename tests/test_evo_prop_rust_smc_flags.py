from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from tests._compat_pd import pd

from forex_bot.features.talib_mixer import TALibStrategyGene
from forex_bot.strategy import evo_prop


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

    def copy(self, deep=False):
        _ = deep
        return _ArrayFrame(
            {k: np.asarray(v).copy() for k, v in self._data.items()},
            np.asarray(self.index).copy(),
            attrs=dict(self.attrs),
        )


def _price_df(rows: int = 64) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(1.0, 1.1, rows, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": x,
            "high": x + 0.0005,
            "low": x - 0.0005,
            "close": x + 0.0001,
        },
        index=idx,
    )


def _price_array_frame(rows: int = 64) -> _ArrayFrame:
    idx = np.arange(rows, dtype=np.int64)
    x = np.linspace(1.0, 1.1, rows, dtype=np.float64)
    return _ArrayFrame(
        {
            "open": x,
            "high": x + 0.0005,
            "low": x - 0.0005,
            "close": x + 0.0001,
            "volume": np.ones(rows, dtype=np.float64),
        },
        index=idx,
        attrs={"symbol": "EURUSD", "timeframe": "H1"},
    )


def test_convert_rust_gene_preserves_extended_smc_flags():
    gene = {
        "indices": [0],
        "weights": [1.0],
        "long_threshold": 0.7,
        "short_threshold": -0.7,
        "strategy_id": "g1",
        "use_ob": True,
        "use_fvg": False,
        "use_liq_sweep": True,
        "mtf_confirmation": True,
        "use_premium_discount": False,
        "use_inducement": True,
        "use_bos": True,
        "use_choch": True,
        "use_eqh": False,
        "use_eql": True,
        "use_displacement": True,
    }
    out = evo_prop._convert_rust_gene(gene, ["smc_bos"], {"RSI"})
    assert out is not None
    assert bool(out.use_bos)
    assert bool(out.use_choch)
    assert not bool(out.use_eqh)
    assert bool(out.use_eql)
    assert bool(out.use_displacement)


def test_batch_population_eval_returns_none_without_rust_population_path(monkeypatch):
    df = _price_df(rows=32)
    genes = [TALibStrategyGene(indicators=["RSI"], strategy_id="g_none")]
    mixer = SimpleNamespace(_rust_signal_cache={}, _rust_signal_index=None)

    monkeypatch.setattr(evo_prop, "_fb", object(), raising=False)
    monkeypatch.setattr(evo_prop, "_RUST_TALIB_POP", False, raising=False)

    out = evo_prop._batch_evaluate_population_rust(df, genes, mixer, cache=None, settings=SimpleNamespace())
    assert out is None


def test_batch_population_eval_passes_extended_smc_flags_to_rust(monkeypatch):
    df = _price_df(rows=48)
    gene = TALibStrategyGene(
        indicators=["RSI"],
        strategy_id="g_rust",
        use_ob=True,
        use_bos=True,
        use_choch=False,
        use_eqh=True,
        use_eql=False,
        use_displacement=True,
        sl_pips=20.0,
        tp_pips=40.0,
    )
    genes = [gene]
    mixer = SimpleNamespace(_rust_signal_cache={}, _rust_signal_index=None)
    captured: dict[str, object] = {}

    class _DummyBindings:
        def evaluate_population_talib_ohlcv(self, *_args, **kwargs):
            captured.update(kwargs)
            n = len(kwargs.get("indicator_sets") or [])
            return np.zeros((n, 11), dtype=np.float64)

    monkeypatch.setattr(evo_prop, "_fb", _DummyBindings(), raising=False)
    monkeypatch.setattr(evo_prop, "_RUST_TALIB_POP", True, raising=False)

    settings = SimpleNamespace(
        risk=SimpleNamespace(meta_label_sl_pips=20.0, meta_label_tp_pips=40.0),
    )
    out = evo_prop._batch_evaluate_population_rust(df, genes, mixer, cache=None, settings=settings)
    assert out is not None
    assert out.shape == (1, 11)
    assert list(captured.get("use_bos_flags") or []) == [1]
    assert list(captured.get("use_choch_flags") or []) == [0]
    assert list(captured.get("use_eqh_flags") or []) == [1]
    assert list(captured.get("use_eql_flags") or []) == [0]
    assert list(captured.get("use_displacement_flags") or []) == [1]


def test_batch_population_eval_uses_bulk_batch_bridge_when_population_api_missing(monkeypatch):
    df = _price_df(rows=36)
    genes = [
        TALibStrategyGene(indicators=["RSI"], strategy_id="g_a", sl_pips=20.0, tp_pips=40.0),
        TALibStrategyGene(indicators=["SMA"], strategy_id="g_b", sl_pips=25.0, tp_pips=50.0),
    ]
    mixer = SimpleNamespace(_rust_signal_cache={}, _rust_signal_index=None)
    captured: dict[str, object] = {}

    class _DummyBindings:
        def talib_bulk_signals_ohlcv(self, open_, high, low, close, **kwargs):
            captured["bulk_kwargs"] = kwargs
            n = int(len(close))
            m = int(len(kwargs.get("indicator_sets") or []))
            out = np.zeros((n, m), dtype=np.int8)
            if m > 0:
                out[:, 0] = 1
            if m > 1:
                out[:, 1] = -1
            return out

        def batch_evaluate_strategies(self, close, high, low, signals, month_idx, day_idx, sl_pips, tp_pips, *args):
            captured["signals_shape"] = tuple(np.asarray(signals).shape)
            captured["sl_pips"] = np.asarray(sl_pips, dtype=np.float64).tolist()
            captured["tp_pips"] = np.asarray(tp_pips, dtype=np.float64).tolist()
            rows = int(np.asarray(signals).shape[0])
            out = np.zeros((rows, 11), dtype=np.float64)
            out[:, 0] = np.arange(100.0, 100.0 + rows, dtype=np.float64)
            return out

    monkeypatch.setattr(evo_prop, "_fb", _DummyBindings(), raising=False)
    monkeypatch.setattr(evo_prop, "_RUST_TALIB_POP", False, raising=False)

    settings = SimpleNamespace(
        risk=SimpleNamespace(meta_label_sl_pips=20.0, meta_label_tp_pips=40.0),
    )
    out = evo_prop._batch_evaluate_population_rust(df, genes, mixer, cache=None, settings=settings)

    assert out is not None
    assert out.shape == (2, 11)
    np.testing.assert_allclose(out[:, 0], np.array([100.0, 101.0], dtype=np.float64))
    assert tuple(captured.get("signals_shape") or ()) == (2, len(df))
    assert captured.get("sl_pips") == [20.0, 20.0]
    assert captured.get("tp_pips") == [40.0, 40.0]


def test_batch_population_eval_uses_bulk_batch_bridge_with_numpy_frame(monkeypatch):
    df = _price_array_frame(rows=36)
    genes = [
        TALibStrategyGene(indicators=["RSI"], strategy_id="g_np_a", sl_pips=20.0, tp_pips=40.0),
        TALibStrategyGene(indicators=["SMA"], strategy_id="g_np_b", sl_pips=25.0, tp_pips=50.0),
    ]
    mixer = SimpleNamespace(_rust_signal_cache={}, _rust_signal_index=None)
    captured: dict[str, object] = {}

    class _DummyBindings:
        def talib_bulk_signals_ohlcv(self, open_, high, low, close, **kwargs):
            captured["bulk_shapes"] = (
                int(np.asarray(open_).shape[0]),
                int(np.asarray(high).shape[0]),
                int(np.asarray(low).shape[0]),
                int(np.asarray(close).shape[0]),
            )
            n = int(len(close))
            m = int(len(kwargs.get("indicator_sets") or []))
            out = np.zeros((n, m), dtype=np.int8)
            if m > 0:
                out[:, 0] = 1
            if m > 1:
                out[:, 1] = -1
            return out

        def batch_evaluate_strategies(self, close, high, low, signals, month_idx, day_idx, sl_pips, tp_pips, *args):
            captured["signals_shape"] = tuple(np.asarray(signals).shape)
            rows = int(np.asarray(signals).shape[0])
            out = np.zeros((rows, 11), dtype=np.float64)
            out[:, 0] = np.arange(200.0, 200.0 + rows, dtype=np.float64)
            return out

    monkeypatch.setattr(evo_prop, "_fb", _DummyBindings(), raising=False)
    monkeypatch.setattr(evo_prop, "_RUST_TALIB_POP", False, raising=False)

    settings = SimpleNamespace(
        risk=SimpleNamespace(meta_label_sl_pips=20.0, meta_label_tp_pips=40.0),
    )
    out = evo_prop._batch_evaluate_population_rust(df, genes, mixer, cache=None, settings=settings)

    assert out is not None
    assert out.shape == (2, 11)
    np.testing.assert_allclose(out[:, 0], np.array([200.0, 201.0], dtype=np.float64))
    assert tuple(captured.get("bulk_shapes") or ()) == (36, 36, 36, 36)
    assert tuple(captured.get("signals_shape") or ()) == (2, len(df))


def test_evaluate_gene_prefers_rust_population_eval(monkeypatch):
    df = _price_df(rows=40)
    gene = TALibStrategyGene(indicators=["RSI"], strategy_id="g_single")
    settings = SimpleNamespace(
        risk=SimpleNamespace(meta_label_sl_pips=20.0, meta_label_tp_pips=40.0),
    )

    class _DummyMixer:
        def compute_signals(self, *_args, **_kwargs):
            raise AssertionError("Python signal path should not be used when Rust eval succeeds")

    def _stub_batch(*_args, **_kwargs):
        return np.asarray([[123.0, 1.25, 0.0, 0.03, 0.57, 1.40, 6.0, 0.0, 42.0, 0.0, 0.0]], dtype=np.float64)

    monkeypatch.setattr(evo_prop, "_RUST_POP_EVAL", True, raising=False)
    monkeypatch.setattr(evo_prop, "_batch_evaluate_population_rust", _stub_batch, raising=False)

    score = evo_prop._evaluate_gene(df, gene, _DummyMixer(), cache=None, settings=settings)
    assert abs(score - 123.0) < 1e-9
    assert abs(float(gene.sharpe_ratio) - 1.25) < 1e-9
    assert abs(float(gene.max_dd_pct) - 0.03) < 1e-9
    assert abs(float(gene.win_rate) - 0.57) < 1e-9
    assert abs(float(gene.profit_factor) - 1.40) < 1e-9
    assert abs(float(gene.expectancy) - 6.0) < 1e-9


def test_evaluate_gene_strict_rust_skips_python_fallback(monkeypatch):
    df = _price_df(rows=32)
    gene = TALibStrategyGene(indicators=["RSI"], strategy_id="g_strict")
    settings = SimpleNamespace(
        risk=SimpleNamespace(meta_label_sl_pips=20.0, meta_label_tp_pips=40.0),
    )

    class _DummyMixer:
        def compute_signals(self, *_args, **_kwargs):
            raise AssertionError("Python signal path should not run in rust-only mode")

    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setattr(evo_prop, "_RUST_POP_EVAL", False, raising=False)

    score = evo_prop._evaluate_gene(df, gene, _DummyMixer(), cache=None, settings=settings)
    assert score == 0.0
    assert float(gene.fitness) == 0.0


def test_evaluate_gene_py_eval_override_disabled_skips_python_fallback(monkeypatch):
    df = _price_df(rows=32)
    gene = TALibStrategyGene(indicators=["RSI"], strategy_id="g_no_py_eval")
    settings = SimpleNamespace(
        risk=SimpleNamespace(meta_label_sl_pips=20.0, meta_label_tp_pips=40.0),
    )

    class _DummyMixer:
        def compute_signals(self, *_args, **_kwargs):
            raise AssertionError("Python signal path should not run when FOREX_BOT_PROP_PY_FALLBACK=0")

    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("FOREX_BOT_TREE_BACKEND", raising=False)
    monkeypatch.setenv("FOREX_BOT_PROP_PY_FALLBACK", "0")
    monkeypatch.setattr(evo_prop, "_RUST_POP_EVAL", False, raising=False)

    score = evo_prop._evaluate_gene(df, gene, _DummyMixer(), cache=None, settings=settings)
    assert score == 0.0
    assert float(gene.fitness) == 0.0


def test_attach_trade_journals_prefers_rust_bulk_signals(monkeypatch):
    df = _price_df(rows=24)
    gene = TALibStrategyGene(indicators=["RSI"], strategy_id="g_journal", sl_pips=20.0, tp_pips=40.0)
    selected = [gene]
    calls = {"bulk": 0, "journal": 0}

    class _DummyMixer:
        def __init__(self) -> None:
            self.available_indicators = ["RSI"]

        @staticmethod
        def _gene_key(g: TALibStrategyGene):
            return (tuple(str(i).upper() for i in (g.indicators or [])), float(g.long_threshold), float(g.short_threshold))

        @staticmethod
        def bulk_calculate_indicators(df_arg, population):
            return {}

        def compute_signals(self, *_args, **_kwargs):  # pragma: no cover - should not be used
            raise AssertionError("Python signal path should not run when rust bulk signals are available")

    class _DummyBindings:
        def talib_bulk_signals_ohlcv(self, open_, high, low, close, **kwargs):
            calls["bulk"] += 1
            n = int(len(close))
            m = int(len(kwargs.get("indicator_sets") or []))
            out = np.zeros((n, m), dtype=np.int8)
            if m > 0:
                out[:, 0] = 1
            return out

    def _journal_stub(**kwargs):
        calls["journal"] += 1
        sig = np.asarray(kwargs.get("signals"), dtype=np.int8)
        return {"computed": True, "signal_sum": int(np.sum(sig))}

    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _DummyMixer, raising=False)
    monkeypatch.setattr(evo_prop, "_fb", _DummyBindings(), raising=False)
    monkeypatch.setattr(evo_prop, "_trade_journal_from_signals", _journal_stub, raising=False)

    settings = SimpleNamespace(
        risk=SimpleNamespace(meta_label_sl_pips=20.0, meta_label_tp_pips=40.0, min_risk_reward=2.0),
    )
    evo_prop._attach_trade_journals(
        selected=selected,
        search_df=df,
        holdout_df=None,
        settings=settings,
    )

    assert calls["bulk"] >= 1
    assert calls["journal"] == 1
    assert isinstance(gene.in_sample_journal, dict)
    assert bool(gene.in_sample_journal.get("computed"))
    assert int(gene.in_sample_journal.get("signal_sum", 0)) == len(df)
    assert gene.holdout_journal.get("reason") == "no_holdout"


def test_attach_trade_journals_py_journal_override_disabled_skips_python_signal_fallback(monkeypatch):
    df = _price_df(rows=24)
    gene = TALibStrategyGene(indicators=["RSI"], strategy_id="g_journal_disabled", sl_pips=20.0, tp_pips=40.0)
    selected = [gene]

    class _DummyMixer:
        def __init__(self) -> None:
            self.available_indicators = ["RSI"]

        @staticmethod
        def _gene_key(g: TALibStrategyGene):
            return (tuple(str(i).upper() for i in (g.indicators or [])), float(g.long_threshold), float(g.short_threshold))

        @staticmethod
        def bulk_calculate_indicators(df_arg, population):
            return {}

        def compute_signals(self, *_args, **_kwargs):  # pragma: no cover - should not be used
            raise AssertionError("Python signal path should not run when journal fallback is disabled")

    monkeypatch.setenv("FOREX_BOT_PROP_PY_FALLBACK", "0")
    monkeypatch.delenv("FOREX_BOT_RUST_ONLY", raising=False)
    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _DummyMixer, raising=False)
    monkeypatch.setattr(evo_prop, "_fb", object(), raising=False)

    settings = SimpleNamespace(
        risk=SimpleNamespace(meta_label_sl_pips=20.0, meta_label_tp_pips=40.0, min_risk_reward=2.0),
    )
    evo_prop._attach_trade_journals(
        selected=selected,
        search_df=df,
        holdout_df=None,
        settings=settings,
    )

    assert isinstance(gene.in_sample_journal, dict)
    assert gene.in_sample_journal.get("reason") == "rust_signal_unavailable"


def test_attach_trade_journals_prefers_rust_bulk_signals_with_numpy_frame(monkeypatch):
    df = _price_array_frame(rows=24)
    gene = TALibStrategyGene(indicators=["RSI"], strategy_id="g_np_journal", sl_pips=20.0, tp_pips=40.0)
    selected = [gene]
    calls = {"bulk": 0, "journal": 0}

    class _DummyMixer:
        def __init__(self) -> None:
            self.available_indicators = ["RSI"]

        @staticmethod
        def _gene_key(g: TALibStrategyGene):
            return (tuple(str(i).upper() for i in (g.indicators or [])), float(g.long_threshold), float(g.short_threshold))

        @staticmethod
        def bulk_calculate_indicators(df_arg, population):
            return {}

    class _DummyBindings:
        def talib_bulk_signals_ohlcv(self, open_, high, low, close, **kwargs):
            calls["bulk"] += 1
            n = int(len(close))
            m = int(len(kwargs.get("indicator_sets") or []))
            out = np.zeros((n, m), dtype=np.int8)
            if m > 0:
                out[:, 0] = 1
            return out

    def _journal_stub(**kwargs):
        calls["journal"] += 1
        sig = np.asarray(kwargs.get("signals"), dtype=np.int8)
        return {"computed": True, "signal_sum": int(np.sum(sig))}

    monkeypatch.setattr(evo_prop, "TALibStrategyMixer", _DummyMixer, raising=False)
    monkeypatch.setattr(evo_prop, "_fb", _DummyBindings(), raising=False)
    monkeypatch.setattr(evo_prop, "_trade_journal_from_signals", _journal_stub, raising=False)

    settings = SimpleNamespace(
        risk=SimpleNamespace(meta_label_sl_pips=20.0, meta_label_tp_pips=40.0, min_risk_reward=2.0),
    )
    evo_prop._attach_trade_journals(
        selected=selected,
        search_df=df,
        holdout_df=None,
        settings=settings,
    )

    assert calls["bulk"] >= 1
    assert calls["journal"] == 1
    assert isinstance(gene.in_sample_journal, dict)
    assert bool(gene.in_sample_journal.get("computed"))

