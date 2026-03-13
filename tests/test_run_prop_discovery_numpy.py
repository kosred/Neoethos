from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tests._compat_pd import pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.run_prop_discovery import _history_span_days_months, _with_lookback  # noqa: E402


class _ArrayFrame:
    def __init__(self, data: dict[str, np.ndarray], index: np.ndarray, attrs: dict[str, object] | None = None) -> None:
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


def test_with_lookback_accepts_object_datetime_index_frame() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    frame = _ArrayFrame(
        {"close": np.arange(5, dtype=np.float64)},
        np.asarray(list(idx), dtype=object),
        attrs={"symbol": "EURUSD"},
    )

    out = _with_lookback(frame, 2)

    assert getattr(out, "empty", False) is False
    np.testing.assert_array_equal(
        np.asarray(out["close"], dtype=np.float64),
        np.array([2.0, 3.0, 4.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(out.index),
        np.asarray(list(idx[-3:]), dtype=object),
    )
    assert getattr(out, "attrs", {}).get("symbol") == "EURUSD"


def test_history_span_days_months_accepts_object_datetime_index_frame() -> None:
    idx = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    frame = _ArrayFrame({"close": np.arange(4, dtype=np.float64)}, np.asarray(list(idx), dtype=object))

    days, months = _history_span_days_months(frame)

    assert days >= 3.0
    assert 0.09 <= months <= 0.11
