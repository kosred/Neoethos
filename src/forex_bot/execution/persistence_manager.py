from __future__ import annotations
import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from .state_models import BoundedLRUFeatureStore, _FEATURE_ROW_KIND

logger = logging.getLogger(__name__)

class PersistenceManager:
    """
    Handles persistence for entry-time features used in online learning.
    """

    def __init__(self, symbol: str, entry_feature_store: BoundedLRUFeatureStore):
        self.symbol = symbol
        self.entry_feature_store = entry_feature_store
        self._entry_store_path = Path("cache") / f"entry_features_{self.symbol}.json"

    def _as_json_feature_scalar(self, value: Any) -> float | int | bool | None:
        import numpy as np
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            value_f = float(value)
            return value_f if np.isfinite(value_f) else None
        if isinstance(value, (np.datetime64, datetime)):
            return None
        try:
            value_f = float(value)
        except Exception:
            return None
        return value_f if np.isfinite(value_f) else None

    def _validate_feature_row_payload(self, payload: Any) -> dict[str, Any] | None:
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

    def load_entry_store(self) -> None:
        try:
            if not self._entry_store_path.exists():
                return
            raw = json.loads(self._entry_store_path.read_text())
            restored = {}
            for k, v in raw.items():
                try:
                    bar_time = None
                    if v.get("bar_time"):
                        bar_time = datetime.fromisoformat(v["bar_time"])
                    features = self._validate_feature_row_payload(v.get("features"))
                    if features is None:
                        continue
                    restored[int(k)] = {
                        "symbol": v.get("symbol"),
                        "bar_time": bar_time,
                        "features": features,
                        "signal": v.get("signal"),
                        "magic": v.get("magic"),
                    }
                except Exception:
                    continue
            for k, v in restored.items():
                self.entry_feature_store.add(int(k), v)
            logger.info(f"Loaded {len(restored)} entries into feature store")
        except Exception as exc:
            logger.debug(f"Failed to load entry feature store: {exc}")

    def persist_entry_store(self) -> None:
        try:
            async def _async_write():
                def _write():
                    try:
                        self._entry_store_path.parent.mkdir(parents=True, exist_ok=True)
                        serializable = {}
                        for k, v in self.entry_feature_store.items():
                            try:
                                bar_time = v.get("bar_time")
                                bar_time_str = bar_time.isoformat() if isinstance(bar_time, datetime) else bar_time
                                features = self._validate_feature_row_payload(v.get("features"))
                                if features is None:
                                    continue
                                serializable[int(k)] = {
                                    "symbol": v.get("symbol"),
                                    "bar_time": bar_time_str,
                                    "features": features,
                                    "signal": v.get("signal"),
                                    "magic": v.get("magic"),
                                }
                            except Exception:
                                continue
                        self._entry_store_path.write_text(json.dumps(serializable))
                    except Exception as e:
                        logger.warning(f"Background persistence failed: {e}")

                try:
                    await asyncio.to_thread(_write)
                except Exception as exc:
                    logger.debug(f"Async persistence failed: {exc}")

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_async_write())
            except RuntimeError:
                asyncio.run(_async_write())

        except Exception as exc:
            logger.debug(f"Failed to schedule persistence: {exc}")

    def cleanup_entry_store(self):
        now = datetime.now(UTC)
        keys_to_remove = []
        for k, v in self.entry_feature_store.items():
            try:
                bt = v.get("bar_time")
                if isinstance(bt, str):
                    bt = datetime.fromisoformat(bt)
                if bt and (now - bt).total_seconds() > 86400:
                    keys_to_remove.append(k)
            except Exception as cleanup_exc:
                logger.debug(f"Error checking entry {k} for cleanup: {cleanup_exc}")
                keys_to_remove.append(k)

        if not keys_to_remove:
            return

        for k in keys_to_remove:
            self.entry_feature_store.remove(k)

        logger.debug(f"Cleaned up {len(keys_to_remove)} stale entries from feature store")
        self.persist_entry_store()
