from __future__ import annotations

import numpy as np

from forex_bot.training.walkforward import embargoed_walkforward_backtest


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


def test_walkforward_backtest_runs_with_frame_like_and_pandas_block(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_BLOCK", "1")
    monkeypatch.setenv("FOREX_BOT_WALKFORWARD_EMBARGO", "30")

    n = 1200
    close = np.linspace(1.0, 1.2, n, dtype=np.float64)
    frame = _ArrayFrame(
        {
            "close": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "feat_1": np.sin(np.linspace(0.0, 3.14, num=n)),
        },
        index=np.arange(n, dtype=np.int64),
        attrs={"symbol": "EURUSD"},
    )
    signals = np.where(np.arange(n) % 7 == 0, 1, 0).astype(np.int8, copy=False)

    metrics = embargoed_walkforward_backtest(
        df=frame,
        signals=signals,
        train_ratio=0.7,
        n_splits=5,
        embargo_minutes=60,
        timeframe_minutes=5,
        use_gpu=False,
    )

    assert int(metrics.get("walk_forward_splits", 0)) >= 1
    assert "avg_pnl" in metrics
    assert "splits" in metrics
