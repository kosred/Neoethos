import numpy as np
from types import SimpleNamespace
from tests._compat_pd import pd

from forex_bot.features import talib_mixer as talib_mixer_mod
from forex_bot.features.talib_mixer import TALibStrategyGene, TALibStrategyMixer, signal_shift_prev, signal_to_numpy


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


def test_signal_to_numpy_aligns_and_fills():
    idx = pd.date_range("2024-01-01", periods=4, freq="min")
    sig = pd.Series([1.0, np.nan, -1.0], index=idx[1:])
    out = signal_to_numpy(sig, index=idx, dtype=np.float64, fill_value=0.0, forward_fill=True)
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 1.0, -1.0], dtype=np.float64))


def test_signal_to_numpy_uses_rust_alignment_when_available(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=4, freq="min")
    sig = pd.Series([1.0, -1.0], index=idx[[1, 3]])
    calls = {"ffill": 0}

    def _align(src_idx_ns, src_vals, tgt_idx_ns, fill):
        calls["ffill"] += 1
        assert np.asarray(src_idx_ns, dtype=np.int64).shape[0] == 2
        assert np.asarray(tgt_idx_ns, dtype=np.int64).shape[0] == 4
        assert float(fill) == 0.0
        return np.asarray([0.0, 1.0, 1.0, -1.0], dtype=np.float64)

    monkeypatch.setattr(
        talib_mixer_mod,
        "_fb",
        SimpleNamespace(align_ffill_values_by_ns=_align),
        raising=False,
    )
    out = signal_to_numpy(sig, index=idx, dtype=np.float64, fill_value=0.0, forward_fill=True)
    assert calls["ffill"] == 1
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 1.0, -1.0], dtype=np.float64))


def test_signal_to_numpy_fallback_prefers_rust_sorted_index_order(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=4, freq="min")
    sig = pd.Series([3.0, 1.0, 2.0], index=idx[[3, 1, 2]])
    calls = {"sort": 0}

    def _sorted_index_order(idx_ns):
        calls["sort"] += 1
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(
        talib_mixer_mod,
        "_fb",
        SimpleNamespace(sorted_index_order=_sorted_index_order),
        raising=False,
    )
    out = signal_to_numpy(sig, index=idx, dtype=np.float64, fill_value=0.0, forward_fill=True)

    assert calls["sort"] == 1
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64))


def test_signal_to_numpy_exact_fallback_prefers_rust_sorted_index_order(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=4, freq="min")
    sig = pd.Series([3.0, 1.0, 2.0], index=idx[[3, 1, 2]])
    calls = {"sort": 0}

    def _sorted_index_order(idx_ns):
        calls["sort"] += 1
        return np.array([1, 2, 0], dtype=np.int64)

    monkeypatch.setattr(
        talib_mixer_mod,
        "_fb",
        SimpleNamespace(sorted_index_order=_sorted_index_order),
        raising=False,
    )
    out = signal_to_numpy(sig, index=idx, dtype=np.float64, fill_value=0.0, forward_fill=False)

    assert calls["sort"] == 1
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64))


def test_signal_shift_prev_on_numpy():
    sig = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    out = signal_shift_prev(sig, dtype=np.float64, fill_value=0.0)
    np.testing.assert_allclose(out, np.array([0.0, 1.0, -1.0], dtype=np.float64))


def test_compute_signals_uses_cached_array_in_strict_mode():
    idx = pd.date_range("2024-01-01", periods=3, freq="min")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.0, 1.1, 1.2],
        },
        index=idx,
    )
    gene = TALibStrategyGene(
        indicators=["RSI"],
        weights={"RSI": 1.0},
        long_threshold=0.3,
        short_threshold=-0.3,
        strategy_id="cached",
    )
    mixer = TALibStrategyMixer()
    mixer._strict_rust = True
    mixer._rust_signal_cache[mixer._gene_key(gene)] = np.array([1.0, 0.0, -1.0], dtype=np.float64)

    out = mixer.compute_signals(df, gene, cache={})
    arr = signal_to_numpy(out, index=idx, dtype=np.float64, fill_value=0.0, forward_fill=False)
    np.testing.assert_allclose(arr, np.array([1.0, 0.0, -1.0], dtype=np.float64))


def test_compute_signals_uses_cached_array_in_strict_mode_numpy_frame():
    idx = np.arange(4, dtype=np.int64)
    df = _ArrayFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2, 1.3],
        },
        index=idx,
        attrs={"symbol": "EURUSD", "timeframe": "M1"},
    )
    gene = TALibStrategyGene(
        indicators=["RSI"],
        weights={"RSI": 1.0},
        long_threshold=0.3,
        short_threshold=-0.3,
        strategy_id="cached-numpy",
    )
    mixer = TALibStrategyMixer()
    mixer._strict_rust = True
    mixer._rust_signal_cache[mixer._gene_key(gene)] = np.array([1.0, 0.0, -1.0, 1.0], dtype=np.float64)

    out = mixer.compute_signals(df, gene, cache={})
    arr = signal_to_numpy(out, index=idx, dtype=np.float64, fill_value=0.0, forward_fill=False)
    np.testing.assert_allclose(arr, np.array([1.0, 0.0, -1.0, 1.0], dtype=np.float64))

