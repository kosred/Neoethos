import numpy as np
from tests._compat_pd import pd

from forex_bot.core.config import Settings
from forex_bot.features.pipeline import FeatureEngineer


def _make_ohlcv(index: pd.DatetimeIndex, *, start: float) -> pd.DataFrame:
    close = np.linspace(start, start + 1.0, len(index), dtype=np.float64)
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    volume = np.full(len(index), 100.0, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_mtf_merge_blocks_in_memory_python_frames():
    settings = Settings()
    settings.system.base_timeframe = "M1"
    settings.system.cache_enabled = False

    fe = FeatureEngineer(settings)

    # Cover >1 H1 candle so HTF features vary (otherwise constant columns get dropped as near-zero variance).
    idx_m1 = pd.date_range("2025-01-01 10:00", periods=120, freq="1min", tz="UTC")
    base = _make_ohlcv(idx_m1, start=100.0)

    idx_h1 = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01 09:00", tz="UTC"),
            pd.Timestamp("2025-01-01 10:00", tz="UTC"),
            pd.Timestamp("2025-01-01 11:00", tz="UTC"),
        ]
    )
    h1 = _make_ohlcv(idx_h1, start=200.0)

    ds = fe.prepare({"M1": base, "H1": h1}, symbol="EURUSD")
    assert isinstance(ds.X, np.ndarray)
    assert isinstance(ds.y, np.ndarray)
    assert ds.X.shape == (0, 0)
    assert ds.y.shape == (0,)

