from __future__ import annotations

import logging
import os
from typing import Any
import numpy as np
from ..data.loader import DataLoader
from ..domain.events import PreparedDataset
from .utils import (
    index_to_ns_int64, 
    align_ffill_by_ns, 
    index_to_ns_int64,
    align_ffill_by_ns,
    frame_empty,
    frame_len,
    frame_columns,
    frame_has_column,
    frame_index,
    frame_copy,
    frame_column_numpy,
    frame_set_column,
    fit_len_array,
    base_index_and_bounds,
    rust_only_enabled,
)

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_loader: DataLoader, settings: Any) -> None:
        self.data_loader = data_loader
        self.settings = settings

    async def prepare_raw_data(self, symbol: str) -> dict[str, Any]:
        """Loads and prepares raw OHLCV data for a symbol."""
        await self.data_loader.ensure_history(symbol)
        raw_frames = await self.data_loader.get_training_data(symbol)
        if not isinstance(raw_frames, dict) or not raw_frames:
            raise ValueError(f"No data available for {symbol}")
        return raw_frames

    def index_to_ns(self, index: Any) -> np.ndarray | None:
        """Converts various index types to nanosecond int64 arrays."""
        return index_to_ns_int64(index)

    def align_values_ffill(
        self,
        src_idx: np.ndarray,
        src_vals: np.ndarray,
        tgt_idx: np.ndarray,
        fill: float = 0.0,
    ) -> np.ndarray:
        """Strictly causal forward-fill alignment by timestamp."""
        out = align_ffill_by_ns(src_idx, src_vals, tgt_idx, fill=fill)
        if out is None:
            return np.full(tgt_idx.size, float(fill), dtype=np.float32)
        return out
