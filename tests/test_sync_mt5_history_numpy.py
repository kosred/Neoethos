from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.sync_mt5_history import _ensure_ohlcv, _rates_to_df  # noqa: E402


def test_ensure_ohlcv_sorts_and_dedups_timestamp_rows() -> None:
    frame = _ensure_ohlcv(
        {
            "timestamp": np.array(
                [
                    "2025-01-02T00:00:00+00:00",
                    "2025-01-01T00:00:00+00:00",
                    "2025-01-01T00:00:00+00:00",
                ],
                dtype=object,
            ),
            "open": np.array([3.0, 1.0, 9.0], dtype=np.float64),
            "high": np.array([3.5, 1.5, 9.5], dtype=np.float64),
            "low": np.array([2.5, 0.5, 8.5], dtype=np.float64),
            "close": np.array([3.2, 1.2, 9.2], dtype=np.float64),
            "volume": np.array([30.0, 10.0, 90.0], dtype=np.float64),
        }
    )

    assert len(frame) == 2
    np.testing.assert_array_equal(
        frame.index,
        np.array(["2025-01-01T00:00:00.000000000", "2025-01-02T00:00:00.000000000"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(np.asarray(frame["open"], dtype=np.float64), np.array([9.0, 3.0], dtype=np.float64))


def test_rates_to_df_accepts_mt5_structured_array() -> None:
    rates = np.array(
        [
            (1735689600, 1.10, 1.11, 1.09, 1.105, 10, 100),
            (1735689660, 1.20, 1.21, 1.19, 1.205, 15, 150),
        ],
        dtype=[
            ("time", "<i8"),
            ("open", "<f8"),
            ("high", "<f8"),
            ("low", "<f8"),
            ("close", "<f8"),
            ("tick_volume", "<i8"),
            ("real_volume", "<i8"),
        ],
    )

    frame = _rates_to_df(rates)

    assert len(frame) == 2
    np.testing.assert_allclose(
        np.asarray(frame["close"], dtype=np.float64),
        np.array([1.105, 1.205], dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(frame["volume"], dtype=np.float64),
        np.array([100.0, 150.0], dtype=np.float64),
    )
