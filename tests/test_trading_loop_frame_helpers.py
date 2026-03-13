from __future__ import annotations

import numpy as np

from forex_bot.execution import trading_loop as tl


class _ArrayFrame:
    def __init__(self, data, index):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


def test_trading_loop_frame_empty_on_none_and_values():
    assert bool(tl._frame_empty(None))
    frame = _ArrayFrame({"close": [1.0, 1.1]}, index=[0, 1])
    assert not bool(tl._frame_empty(frame))


def test_trading_loop_frame_column_numpy_case_insensitive():
    frame = _ArrayFrame({"Close": [1.0, 1.1, 1.2]}, index=[0, 1, 2])
    close = tl._frame_column_numpy(frame, "close", dtype=np.float64)
    np.testing.assert_allclose(close, np.array([1.0, 1.1, 1.2], dtype=np.float64))
