from __future__ import annotations
import logging
from datetime import UTC, datetime
from typing import Any
import numpy as np
from .state_models import _FEATURE_ROW_KIND, OrderRequest, MT5Position

logger = logging.getLogger(__name__)

class OrderValidator:
    """
    Handles pre-flight checks and validation for orders.
    """

    def __init__(self, settings):
        self.settings = settings
        self.max_positions_per_symbol = 1
        self.min_order_interval_seconds = 5

    def _as_json_feature_scalar(self, value: Any) -> float | int | bool | None:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            value_f = float(value)
            return value_f if np.isfinite(value_f) else None
        try:
            value_f = float(value)
        except Exception:
            return None
        return value_f if np.isfinite(value_f) else None

    def validate_feature_row_payload(self, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        if str(payload.get("kind", "") or "") != _FEATURE_ROW_KIND:
            return None
        raw_columns = payload.get("columns")
        raw_values = payload.get("values")
        if not isinstance(raw_columns, list) or not isinstance(raw_values, list):
            return None
        if len(raw_columns) != len(raw_values):
            return None
        if len(raw_columns) <= 0:
            return None
        columns: list[str] = []
        values: list[float | int | bool | None] = []
        for col, value in zip(raw_columns, raw_values, strict=False):
            name = str(col or "").strip()
            if not name:
                return None
            columns.append(name)
            values.append(self._as_json_feature_scalar(value))
        return {
            "kind": _FEATURE_ROW_KIND,
            "columns": columns,
            "values": values,
        }

    def build_feature_row_payload(
        self,
        features: Any,
        *,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any] | None:
        validated = self.validate_feature_row_payload(features)
        if validated is not None:
            return validated
        if features is None:
            return None

        # Try to extract columns if it's a pandas-like object
        cols = []
        if hasattr(features, "columns"):
            try:
                cols = [str(col) for col in list(features.columns)]
            except Exception:
                pass
        
        if cols:
            values: list[float | int | bool | None] = []
            for col in cols:
                try:
                    raw = features[col]
                    if hasattr(raw, "to_numpy"):
                        try:
                            arr = raw.to_numpy(copy=False)
                        except TypeError:
                            arr = raw.to_numpy()
                        vec = arr.reshape(-1)
                    else:
                        vec = np.asarray(raw).reshape(-1)
                    values.append(self._as_json_feature_scalar(vec[-1] if vec.size > 0 else None))
                except Exception:
                    values.append(None)
            return self.validate_feature_row_payload({"kind": _FEATURE_ROW_KIND, "columns": cols, "values": values})

        arr = np.asarray(features)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        
        if arr.shape[0] <= 0 or arr.shape[1] <= 0:
            return None
            
        names = list(feature_names or [])
        if len(names) != int(arr.shape[1]):
            names = [f"feature_{idx}" for idx in range(int(arr.shape[1]))]
            
        row = np.asarray(arr[-1]).reshape(-1)
        values = [self._as_json_feature_scalar(value) for value in row.tolist()]
        return self.validate_feature_row_payload({"kind": _FEATURE_ROW_KIND, "columns": names, "values": values})

    async def can_place_order(
        self, 
        symbol: str, 
        volume: float, 
        current_bar_time: datetime,
        cached_positions: list[MT5Position],
        pending_requests: dict[str, OrderRequest],
        completed_requests: list[OrderRequest],
        last_order_time: datetime | None,
        get_margin_free: callable,
        get_symbol_info: callable
    ) -> tuple[bool, str]:
        now = datetime.now(UTC)

        # 1. Max positions check
        existing_positions = [p for p in cached_positions if p.symbol == symbol]
        if len(existing_positions) >= self.max_positions_per_symbol:
            return False, f"Already have {len(existing_positions)} position(s) on {symbol}"

        # 2. Duplicate bar check (pending)
        for req_id, req in pending_requests.items():
            if req.symbol == symbol and req.bar_timestamp == current_bar_time:
                return False, f"Order already pending for this bar (req_id={req_id[:8]})"

        # 3. Duplicate bar check (completed)
        for req in completed_requests:
            if req.symbol == symbol and req.bar_timestamp == current_bar_time and req.verified:
                return False, "Order already placed and verified for this bar"

        # 4. Interval check
        if last_order_time:
            seconds_since_last = (now - last_order_time).total_seconds()
            if seconds_since_last < self.min_order_interval_seconds:
                return (
                    False,
                    f"Too soon since last order ({seconds_since_last:.1f}s < {self.min_order_interval_seconds}s)",
                )

        # 5. Margin check
        margin_free = get_margin_free()
        estimated_margin_needed = volume * 1000  # Conservative estimate
        if margin_free < estimated_margin_needed:
            logger.warning(f"Margin check: need ~${estimated_margin_needed:.0f}, have ${margin_free:.0f}")
            return False, "Insufficient margin available"

        # 6. Lot size validation
        valid_lot = await self.validate_lot_size(symbol, volume, get_symbol_info)
        if not valid_lot:
            return False, f"Invalid lot size {volume} for broker limits"

        return True, "OK"

    async def validate_lot_size(self, symbol: str, volume: float, get_symbol_info: callable) -> bool:
        try:
            symbol_info = await get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Could not get symbol info for {symbol} - VALIDATION FAILED")
                return False

            volume_min = symbol_info.get("volume_min", 0.01)
            volume_max = symbol_info.get("volume_max", 100.0)
            volume_step = symbol_info.get("volume_step", 0.01)

            if volume < volume_min:
                logger.error(f"Lot size {volume} below broker min {volume_min}")
                return False

            if volume > volume_max:
                logger.error(f"Lot size {volume} above broker max {volume_max}")
                return False

            if volume_step > 0:
                steps = round(volume / volume_step)
                aligned_volume = steps * volume_step
                if abs(volume - aligned_volume) > 0.0001:
                    logger.error(
                        "Lot size %s not aligned with step %s (nearest: %.5f)",
                        volume,
                        volume_step,
                        aligned_volume,
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Lot size validation exception: {e} - VALIDATION FAILED", exc_info=True)
            return False
