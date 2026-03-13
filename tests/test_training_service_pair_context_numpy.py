from __future__ import annotations

import numpy as np
import pytest

from forex_bot.domain.events import PreparedDataset
from forex_bot.execution import training_service as ts_mod
from forex_bot.execution.training_service import TrainingService
from tests._compat_pd import pd


def _ds(close: np.ndarray, idx: np.ndarray, seed: int) -> PreparedDataset:
    close_f = np.asarray(close, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.01, size=close_f.shape[0]).astype(np.float32)
    x = np.column_stack([close_f, noise]).astype(np.float32)
    y = rng.integers(0, 3, size=close_f.shape[0], dtype=np.int8)
    return PreparedDataset(
        X=x,
        y=y,
        index=np.asarray(idx, dtype=np.int64),
        feature_names=["close", "noise"],
        metadata=None,
        labels=y,
    )


def test_inject_cross_pair_context_numpy_adds_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_ENABLED", "1")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_WINDOW", "48")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_MIN_OVERLAP", "80")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_MAX_PEERS", "1")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_LAG", "1")

    n = 600
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    rng = np.random.default_rng(123)
    step = rng.normal(0.0, 0.00035, size=n).astype(np.float32)
    close_a = 1.10 + np.cumsum(step)
    close_b = 1.25 + np.cumsum((0.8 * step) + rng.normal(0.0, 0.00015, size=n).astype(np.float32))

    d1 = _ds(close_a, idx, seed=1)
    d2 = _ds(close_b, idx, seed=2)

    svc = object.__new__(TrainingService)
    out = svc._inject_cross_pair_context([("EURUSD", d1), ("GBPUSD", d2)])
    by_sym = {sym: ds for sym, ds in out}

    assert set(by_sym.keys()) == {"EURUSD", "GBPUSD"}
    eur = by_sym["EURUSD"]
    gbp = by_sym["GBPUSD"]
    assert isinstance(eur.X, np.ndarray)
    assert isinstance(gbp.X, np.ndarray)
    assert eur.X.shape[0] == n and gbp.X.shape[0] == n
    assert eur.X.shape[1] > d1.X.shape[1]
    assert gbp.X.shape[1] > d2.X.shape[1]
    assert eur.X.shape[1] == len(eur.feature_names)
    assert gbp.X.shape[1] == len(gbp.feature_names)
    assert "pair_peer_ret_mean" in eur.feature_names
    assert "pair_lead_ret" in eur.feature_names

    lead_col = eur.feature_names.index("pair_lead_ret")
    # Lag=1 means first row cannot use same-bar peer info.
    assert float(eur.X[0, lead_col]) == 0.0


def test_inject_cross_pair_context_numpy_respects_disable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_ENABLED", "0")
    n = 200
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    close = np.linspace(1.0, 1.2, num=n, dtype=np.float32)
    d1 = _ds(close, idx, seed=3)
    d2 = _ds(close * 1.01, idx, seed=4)

    svc = object.__new__(TrainingService)
    out = svc._inject_cross_pair_context([("EURUSD", d1), ("GBPUSD", d2)])
    assert out[0][1].X.shape[1] == d1.X.shape[1]
    assert out[1][1].X.shape[1] == d2.X.shape[1]


def test_inject_cross_pair_context_mixed_input_uses_numpy_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_ENABLED", "1")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_WINDOW", "32")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_MIN_OVERLAP", "60")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_MAX_PEERS", "1")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_LAG", "1")

    n = 500
    idx_ns = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    idx_dt = pd.to_datetime(idx_ns, utc=True)
    rng = np.random.default_rng(77)
    step = rng.normal(0.0, 0.00025, size=n).astype(np.float32)
    close_a = 1.10 + np.cumsum(step)
    close_b = 1.18 + np.cumsum((0.8 * step) + rng.normal(0.0, 0.0002, size=n).astype(np.float32))

    x_df = pd.DataFrame({"noise": np.linspace(0.0, 1.0, num=n, dtype=np.float32)}, index=idx_dt)
    meta_df = pd.DataFrame({"close": close_a.astype(np.float32)}, index=idx_dt)
    y_df = np.random.default_rng(8).integers(0, 3, size=n, dtype=np.int8)
    ds_df = PreparedDataset(
        X=x_df,
        y=y_df,
        index=idx_dt,
        feature_names=["noise"],
        metadata=meta_df,
        labels=y_df,
    )

    y_np = np.random.default_rng(9).integers(0, 3, size=n, dtype=np.int8)
    x_np = np.column_stack([close_b.astype(np.float32), np.linspace(1.0, 2.0, num=n, dtype=np.float32)]).astype(
        np.float32
    )
    ds_np = PreparedDataset(
        X=x_np,
        y=y_np,
        index=idx_ns,
        feature_names=["close", "f1"],
        metadata=None,
        labels=y_np,
    )

    svc = object.__new__(TrainingService)
    out = svc._inject_cross_pair_context([("EURUSD", ds_df), ("GBPUSD", ds_np)])
    by_sym = {sym: ds for sym, ds in out}

    eur = by_sym["EURUSD"]
    gbp = by_sym["GBPUSD"]
    assert isinstance(eur.X, np.ndarray)
    assert isinstance(gbp.X, np.ndarray)
    assert eur.X.shape[0] == n and gbp.X.shape[0] == n
    assert eur.X.shape[1] == len(eur.feature_names)
    assert gbp.X.shape[1] == len(gbp.feature_names)
    assert "pair_lead_ret" in eur.feature_names
    assert "pair_corr_mean" in eur.feature_names


def test_inject_cross_pair_context_prefers_rust_peer_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_ENABLED", "1")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_WINDOW", "32")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_MIN_OVERLAP", "60")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_MAX_PEERS", "1")
    monkeypatch.setenv("FOREX_BOT_PAIR_CORR_LAG", "1")

    calls = {"rank": 0}

    def _rust_rank_scores_desc(scores, absolute=False):
        calls["rank"] += 1
        assert bool(absolute) is False
        return np.array([1, 0], dtype=np.int64)

    monkeypatch.setattr(ts_mod, "_rust_rank_scores_desc", _rust_rank_scores_desc, raising=False)

    n = 480
    idx = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000
    rng = np.random.default_rng(1234)
    base_step = rng.normal(0.0, 0.0003, size=n).astype(np.float32)
    close_a = 1.10 + np.cumsum(base_step)
    close_b = 1.25 + np.cumsum((0.85 * base_step) + rng.normal(0.0, 0.00012, size=n).astype(np.float32))
    close_c = 145.0 + np.cumsum((0.55 * base_step) + rng.normal(0.0, 0.00008, size=n).astype(np.float32))

    d1 = _ds(close_a, idx, seed=11)
    d2 = _ds(close_b, idx, seed=12)
    d3 = _ds(close_c, idx, seed=13)

    svc = object.__new__(TrainingService)
    out = svc._inject_cross_pair_context([("EURUSD", d1), ("GBPUSD", d2), ("USDJPY", d3)])
    by_sym = {sym: ds for sym, ds in out}
    eur = by_sym["EURUSD"]

    assert calls["rank"] >= 1

    sym_idx, _sym_ret = svc._numpy_dataset_returns(d1)  # type: ignore[misc]
    peer_idx, peer_ret = svc._numpy_dataset_returns(d3)  # type: ignore[misc]
    expected_lead = svc._shift_with_lag(svc._align_values_by_timestamp(peer_idx, peer_ret, sym_idx), 1)
    lead_col = eur.feature_names.index("pair_lead_ret")

    np.testing.assert_allclose(
        np.asarray(eur.X[:, lead_col], dtype=np.float32),
        np.asarray(expected_lead, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )

