import os
import sys

import numpy as np
from tests._compat_pd import pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from forex_bot.models import unsupervised as umod
from forex_bot.models.unsupervised import ClusterExpert


def test_cluster_expert_flow():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.001, size=1200)
    close = 100.0 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": close})

    model = ClusterExpert(n_regimes=5, lookback=1000)

    model.fit(df)
    assert model.is_fitted is True
    assert len(model.regime_map) > 0

    regime = model.predict(df)
    assert regime not in {"Unknown", "Error"}

    preds = model.predict_proba(df)
    assert preds.shape == (len(df), 3)
    assert np.allclose(preds.sum(axis=1), 1.0)


def test_cluster_expert_extract_features_prefers_rust_binding(monkeypatch):
    calls = {"extract": 0}

    def _extract_regime_features(close_prices, adx_values=None, volatility_window=20):
        calls["extract"] += 1
        assert int(volatility_window) == 20
        close_arr = np.asarray(close_prices, dtype=np.float64).reshape(-1)
        assert close_arr.shape[0] == 6
        return np.array(
            [
                [0.1, 1.0, 10.0],
                [0.2, 2.0, 20.0],
                [0.3, 3.0, 30.0],
                [0.4, 4.0, 40.0],
                [0.5, 5.0, 50.0],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(umod, "_fb", type("FB", (), {"extract_regime_features": staticmethod(_extract_regime_features)}), raising=False)

    df = pd.DataFrame(
        {
            "close": np.array([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], dtype=np.float32),
            "adx": np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=np.float32),
        }
    )
    model = ClusterExpert(n_regimes=3)

    out = model._extract_features(df)

    assert calls["extract"] == 1
    np.testing.assert_allclose(
        out,
        np.array(
            [
                [0.1, 1.0, 10.0],
                [0.2, 2.0, 20.0],
                [0.3, 3.0, 30.0],
                [0.4, 4.0, 40.0],
                [0.5, 5.0, 50.0],
            ],
            dtype=np.float32,
        ),
    )


if __name__ == "__main__":
    test_cluster_expert_flow()

