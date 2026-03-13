from __future__ import annotations

import numpy as np
from tests._compat_pd import pd

from forex_bot.strategy import discovery_tensor as dt


def test_discovery_search_skips_python_path_when_rust_backend_unavailable(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=64, freq="min", tz="UTC")
    x = np.linspace(1.0, 1.2, len(idx), dtype=np.float64)
    df = pd.DataFrame(
        {
            "open": x,
            "high": x + 0.0005,
            "low": x - 0.0005,
            "close": x,
        },
        index=idx,
    )
    frames = {"M1": df}
    engine = dt.TensorDiscoveryEngine(device="cpu", n_experts=4, timeframes=["M1"], settings=None)
    monkeypatch.setattr(dt, "_RUST_DISCOVERY", False, raising=False)
    monkeypatch.setattr(dt, "_fb", None, raising=False)
    engine.run_unsupervised_search(frames, iterations=16)

