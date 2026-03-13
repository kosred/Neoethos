from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tests._compat_pd import pd

from forex_bot.strategy import discovery_tensor as dt


def _price_df(rows: int = 64) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(1.0, 1.2, rows, dtype=np.float64)
    df = pd.DataFrame(
        {
            "open": x,
            "high": x + 0.0006,
            "low": x - 0.0006,
            "close": x + 0.0001,
            "volume": np.full(rows, 100.0, dtype=np.float64),
        },
        index=idx,
    )
    df.attrs["symbol"] = "EURUSD"
    df.attrs["timeframe"] = "M1"
    return df


def test_discovery_rust_search_preserves_smc_flags(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    class _DummyBindings:
        @staticmethod
        def search_discovery_ohlcv(*_args, **_kwargs):
            return {
                "feature_names": ["RSI"],
                "rust_ranked": True,
                "portfolio": [
                    {
                        "indices": [0],
                        "weights": [1.0],
                        "fitness": 321.0,
                        "sharpe_ratio": 1.4,
                        "trades": 55.0,
                        "max_dd_pct": 0.02,
                        "long_threshold": 0.6,
                        "short_threshold": -0.6,
                        "combination_method": "weighted_vote",
                        "use_bos": True,
                        "use_choch": False,
                        "use_eqh": True,
                        "use_eql": False,
                        "use_displacement": True,
                        "sl_pips": 20.0,
                        "tp_pips": 40.0,
                    }
                ],
            }

    monkeypatch.setattr(dt, "_fb", _DummyBindings(), raising=False)
    monkeypatch.setattr(dt, "_RUST_DISCOVERY", True, raising=False)

    settings = SimpleNamespace(
        risk=SimpleNamespace(total_drawdown_limit=0.07),
        models=SimpleNamespace(
            prop_search_portfolio_size=8,
            prop_search_holdout_min_sharpe=0.0,
            prop_search_holdout_min_profit_factor=0.0,
        ),
    )

    engine = dt.TensorDiscoveryEngine(device="cpu", n_experts=4, timeframes=["M1"], settings=settings)
    engine.run_unsupervised_search({"M1": _price_df(64)}, iterations=16)

    out_path = Path("cache/talib_knowledge_EURUSD.json")
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    genes = list(payload.get("best_genes") or [])
    assert len(genes) == 1
    gene = genes[0]
    assert gene.get("use_bos") is True
    assert gene.get("use_choch") is False
    assert gene.get("use_eqh") is True
    assert gene.get("use_eql") is False
    assert gene.get("use_displacement") is True


def test_discovery_safe_indices_uses_rust_binding(monkeypatch):
    calls = {"derive": 0}

    class _DummyBindings:
        @staticmethod
        def derive_time_index_arrays(idx_ns):
            calls["derive"] += 1
            n = int(np.asarray(idx_ns, dtype=np.int64).shape[0])
            return (
                np.arange(n, dtype=np.int64),
                np.full(n, 55, dtype=np.int64),
                np.full(n, 8801, dtype=np.int64),
            )

    monkeypatch.setattr(dt, "_fb", _DummyBindings(), raising=False)
    month_idx, day_idx = dt._safe_indices(
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
    np.testing.assert_array_equal(month_idx, np.array([55, 55, 55], dtype=np.int64))
    np.testing.assert_array_equal(day_idx, np.array([8801, 8801, 8801], dtype=np.int64))

