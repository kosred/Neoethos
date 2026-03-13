from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

from ..domain.events import PreparedDataset
from .cpcv import CombinatorialPurgedCV
from .evaluation import probs_to_signals, prop_backtest, quick_backtest

logger = logging.getLogger(__name__)


def _is_dataframe_like(values: Any) -> bool:
    return bool(hasattr(values, "columns") and hasattr(values, "index"))


class _NumpyFrame:
    """Minimal frame container used for slicing non-dataframe-module frame-like objects."""

    def __init__(self, data: dict[str, Any], index: Any, attrs: dict[str, Any] | None = None) -> None:
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.index = np.asarray(index).reshape(-1)
        self.columns = list(self._data.keys())
        self.attrs = dict(attrs or {})

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[str(key)]


def _frame_columns(obj: Any) -> list[str]:
    cols = getattr(obj, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _frame_resolve_column(obj: Any, name: str) -> str | None:
    target = str(name).strip().lower()
    for col in _frame_columns(obj):
        if str(col).strip().lower() == target:
            return col
    return None


def _slice_rows(values: Any, start: int, end: int) -> Any:
    s = max(0, int(start))
    e = max(s, int(end))
    if _is_dataframe_like(values):
        idx = np.arange(s, e, dtype=np.int64)
        with contextlib.suppress(Exception):
            return values.take(idx)
        with contextlib.suppress(Exception):
            base_idx = np.asarray(getattr(values, "index")).reshape(-1)
            return values.loc[base_idx[idx]]
    cols = _frame_columns(values)
    if cols and hasattr(values, "__getitem__"):
        out_data: dict[str, np.ndarray] = {}
        for col in cols:
            try:
                vec = np.asarray(values[col]).reshape(-1)  # type: ignore[index]
                out_data[str(col)] = vec[s:e]
            except Exception:
                continue
        idx = getattr(values, "index", None)
        if idx is None:
            out_idx = np.arange(max(0, e - s), dtype=np.int64)
        else:
            idx_arr = np.asarray(idx).reshape(-1)
            out_idx = idx_arr[s:e] if idx_arr.size >= e else np.arange(max(0, e - s), dtype=np.int64)
        attrs = getattr(values, "attrs", None)
        return _NumpyFrame(out_data, out_idx, attrs=(dict(attrs) if isinstance(attrs, dict) else None))
    arr = np.asarray(values)
    return arr[s:e]


def _frame_to_2d_float32(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        try:
            arr = values.to_numpy(dtype=np.float32, copy=False)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr
        except Exception:
            pass
    cols = _frame_columns(values)
    if cols and hasattr(values, "__getitem__"):
        mats: list[np.ndarray] = []
        n_rows = 0
        for col in cols:
            try:
                vec = np.asarray(values[col], dtype=np.float32).reshape(-1)  # type: ignore[index]
                mats.append(vec)
                n_rows = max(n_rows, int(vec.size))
            except Exception:
                continue
        if mats:
            out = np.zeros((n_rows, len(mats)), dtype=np.float32)
            for j, vec in enumerate(mats):
                take = min(n_rows, int(vec.size))
                if take > 0:
                    out[:take, j] = vec[:take]
            return out
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _index_to_ns_int64(index: Any) -> np.ndarray | None:
    if index is None:
        return None
    try:
        if hasattr(index, "asi8"):
            arr = np.asarray(index.asi8, dtype=np.int64).reshape(-1)
            return arr if arr.size > 0 else np.zeros(0, dtype=np.int64)
    except Exception:
        pass
    try:
        arr = np.asarray(index).reshape(-1)
    except Exception:
        return None
    if arr.size <= 0:
        return np.zeros(0, dtype=np.int64)
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]").astype(np.int64, copy=False)
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype(np.int64, copy=False)
        if arr.dtype.kind == "f":
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
    except Exception:
        pass
    out = np.zeros(arr.size, dtype=np.int64)
    for i, value in enumerate(arr.tolist()):
        try:
            ns = getattr(value, "value", None)
            if ns is not None:
                out[i] = int(ns)
            else:
                out[i] = int(np.datetime64(value, "ns").astype(np.int64))
        except Exception:
            out[i] = 0
    return out


class EvaluationService:
    """
    Handles model evaluation strategies: Walk-Forward Analysis and CPCV.
    """

    def __init__(self, settings: Any, models_dir: Path):
        self.settings = settings
        self.models_dir = models_dir

    @staticmethod
    def _cpu_budget() -> int:
        for key in ("FOREX_BOT_CPU_BUDGET", "FOREX_BOT_CPU_THREADS"):
            val = os.environ.get(key)
            if val:
                try:
                    return max(1, int(val))
                except Exception:
                    pass
        cpu_count = os.cpu_count() or 1
        try:
            reserve = int(os.environ.get("FOREX_BOT_CPU_RESERVE", "1") or 1)
        except Exception:
            reserve = 1
        return max(1, cpu_count - max(0, reserve))

    def _cpcv_max_rows(self) -> int:
        max_rows_cfg = int(getattr(self.settings.models, "cpcv_max_rows", 250_000) or 0)
        env_override = str(os.environ.get("FOREX_BOT_CPCV_MAX_ROWS", "")).strip()
        if env_override:
            with contextlib.suppress(Exception):
                max_rows_cfg = int(env_override)
        return max_rows_cfg

    @staticmethod
    def _to_numpy_xy(dataset: PreparedDataset) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        try:
            X = dataset.X
            y = dataset.y
            x_np = _frame_to_2d_float32(X)
            if x_np.ndim == 1:
                x_np = x_np.reshape(-1, 1)
            y_np = np.asarray(y).reshape(-1)
            n = int(min(x_np.shape[0], y_np.shape[0]))
            if n <= 1:
                return None, None
            return x_np[:n], y_np[:n]
        except Exception:
            return None, None

    @staticmethod
    def _extract_meta_column(meta: Any, name: str, n_rows: int, *, dtype: Any = np.float64) -> np.ndarray | None:
        if meta is None:
            return None
        vals = None
        try:
            if hasattr(meta, "columns"):
                col_name = _frame_resolve_column(meta, name)
                if col_name is not None:
                    col = meta[col_name]
                    vals = col.to_numpy(dtype=dtype, copy=False) if hasattr(col, "to_numpy") else np.asarray(col)
            elif isinstance(meta, dict) and name in meta:
                vals = np.asarray(meta.get(name))
            elif isinstance(meta, dict):
                for k, v in meta.items():
                    if str(k).strip().lower() == str(name).strip().lower():
                        vals = np.asarray(v)
                        break
        except Exception:
            vals = None
        if vals is None:
            return None
        arr = np.asarray(vals, dtype=dtype).reshape(-1)
        if arr.size < n_rows:
            return None
        return arr[-n_rows:]

    @staticmethod
    def _extract_close(meta: Any, n_rows: int) -> np.ndarray | None:
        return EvaluationService._extract_meta_column(meta, "close", n_rows, dtype=np.float64)

    @staticmethod
    def _extract_symbol(meta: Any) -> str:
        symbol = None
        try:
            attrs = getattr(meta, "attrs", None)
            if isinstance(attrs, dict):
                symbol = attrs.get("symbol")
        except Exception:
            symbol = None
        if symbol is None and isinstance(meta, dict):
            symbol = meta.get("symbol")
        sym = str(symbol or "EURUSD").strip().upper()
        return sym if sym else "EURUSD"

    @staticmethod
    def _extract_eval_frame(meta: Any, n_rows: int) -> dict[str, Any] | None:
        close = EvaluationService._extract_meta_column(meta, "close", n_rows, dtype=np.float64)
        if close is None:
            return None
        frame: dict[str, Any] = {"close": close}
        high = EvaluationService._extract_meta_column(meta, "high", n_rows, dtype=np.float64)
        low = EvaluationService._extract_meta_column(meta, "low", n_rows, dtype=np.float64)
        if high is not None and low is not None:
            frame["high"] = high
            frame["low"] = low
            idx = None
            try:
                idx_raw = getattr(meta, "index", None)
                if idx_raw is not None:
                    idx_arr = _index_to_ns_int64(idx_raw)
                    if idx_arr.size >= n_rows:
                        idx = idx_arr[-n_rows:]
            except Exception:
                idx = None
            if idx is None and isinstance(meta, dict):
                idx_raw = meta.get("index")
                if idx_raw is not None:
                    try:
                        idx_arr = _index_to_ns_int64(idx_raw)
                        if idx_arr.size >= n_rows:
                            idx = idx_arr[-n_rows:]
                    except Exception:
                        idx = None
            if idx is not None:
                frame["index"] = idx
            frame["symbol"] = EvaluationService._extract_symbol(meta)
        return frame

    def _run_cpcv_numpy(self, dataset: PreparedDataset) -> dict[str, Any]:
        x_np, y_np = self._to_numpy_xy(dataset)
        if x_np is None or y_np is None:
            return {}

        eval_frame = self._extract_eval_frame(dataset.metadata, int(x_np.shape[0]))

        max_rows_cfg = self._cpcv_max_rows()
        if max_rows_cfg > 0 and x_np.shape[0] > max_rows_cfg:
            before = int(x_np.shape[0])
            x_np = x_np[-max_rows_cfg:]
            y_np = y_np[-max_rows_cfg:]
            if isinstance(eval_frame, dict):
                for key, val in list(eval_frame.items()):
                    if isinstance(val, np.ndarray):
                        eval_frame[key] = np.asarray(val).reshape(-1)[-max_rows_cfg:]
            logger.info(f"CPCV (NumPy): Capped rows {before:,} -> {len(x_np):,} (max_rows={max_rows_cfg:,})")

        cv = CombinatorialPurgedCV(
            n_splits=self.settings.models.cpcv_n_splits,
            n_test_groups=self.settings.models.cpcv_n_test_groups,
            embargo_pct=self.settings.models.cpcv_embargo_pct,
            purge_pct=self.settings.models.cpcv_purge_pct,
        )
        try:
            splits = cv.split(x_np, y_np)
        except Exception as exc:
            logger.error(f"CPCV (NumPy) split failed: {exc}")
            return {}

        all_backtests: list[dict[str, Any]] = []

        def _majority_predict(train_y: np.ndarray, n_rows: int) -> np.ndarray:
            if train_y.size <= 0:
                return np.zeros(n_rows, dtype=np.int8)
            uniq, cnt = np.unique(train_y, return_counts=True)
            cls = uniq[int(np.argmax(cnt))]
            return np.full(n_rows, cls, dtype=np.asarray(train_y).dtype)

        for i, (train_idx, test_idx) in enumerate(splits):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            x_train = x_np[train_idx]
            y_train = y_np[train_idx]
            x_test = x_np[test_idx]
            y_test = y_np[test_idx]
            try:
                if LogisticRegression is not None:
                    model = LogisticRegression(max_iter=200)
                    model.fit(x_train, y_train)
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(x_test)
                        signals = probs_to_signals(np.asarray(probs))
                    else:
                        signals = np.asarray(model.predict(x_test))
                else:
                    signals = _majority_predict(np.asarray(y_train), len(test_idx))
            except Exception as exc:
                logger.debug("CPCV (NumPy) split %s model failed: %s", i + 1, exc)
                continue

            bt_metrics: dict[str, Any] = {}
            if isinstance(eval_frame, dict) and "close" in eval_frame and len(np.asarray(eval_frame["close"]).reshape(-1)) == len(x_np):
                test_frame: dict[str, Any] = {}
                for key, val in eval_frame.items():
                    if isinstance(val, np.ndarray):
                        arr = np.asarray(val).reshape(-1)
                        if arr.size == len(x_np):
                            test_frame[key] = arr[test_idx]
                    else:
                        test_frame[key] = val
                if {"close", "high", "low"}.issubset(set(test_frame.keys())):
                    try:
                        bt_metrics = prop_backtest(test_frame, signals)
                    except Exception as exc:
                        logger.debug("CPCV (NumPy) split %s prop_backtest failed: %s", i + 1, exc)
                        bt_metrics = {}
                if not bt_metrics and "close" in test_frame:
                    bt_metrics = quick_backtest({"close": np.asarray(test_frame["close"]).reshape(-1)}, signals)
            if not bt_metrics:
                y_true = np.asarray(y_test).reshape(-1)
                y_pred = np.asarray(signals).reshape(-1)[: y_true.shape[0]]
                acc = float((y_pred == y_true).mean()) if y_true.size > 0 else 0.0
                pnl = float((2.0 * acc) - 1.0)
                bt_metrics = {
                    "accuracy": acc,
                    "pnl_score": pnl,
                    "win_rate": acc,
                    "sharpe": 0.0,
                    "trades": float(y_true.size),
                }
            if "pnl_score" not in bt_metrics and "net_profit" in bt_metrics:
                bt_metrics["pnl_score"] = float(bt_metrics.get("net_profit", 0.0))
            if "max_dd" not in bt_metrics and "max_dd_pct" in bt_metrics:
                bt_metrics["max_dd"] = float(bt_metrics.get("max_dd_pct", 0.0))
            bt_metrics["split"] = i + 1
            all_backtests.append(bt_metrics)

        if not all_backtests:
            return {"n_splits": 0, "error": "All splits failed"}

        result = {
            "n_splits": len(all_backtests),
            "n_combinations": len(splits),
            "phi": cv.calculate_phi(x_np),
            "avg_pnl": float(np.mean([b.get("pnl_score", 0.0) for b in all_backtests])),
            "std_pnl": float(np.std([b.get("pnl_score", 0.0) for b in all_backtests])),
            "avg_win_rate": float(np.mean([b.get("win_rate", 0.0) for b in all_backtests])),
            "avg_sharpe": float(np.mean([b.get("sharpe", 0.0) for b in all_backtests])),
            "avg_trades": float(np.mean([b.get("trades", 0.0) for b in all_backtests])),
            "splits": all_backtests,
        }
        return result

    def run_walkforward(
        self, dataset: PreparedDataset, models: dict, ensemble_func, start_index: int = 0
    ) -> dict[str, Any]:
        """
        Sequential walk-forward evaluation using the current ensemble (no retraining).

        Args:
            dataset: The full dataset.
            models: Dictionary of trained models.
            ensemble_func: Function to generate predictions.
            start_index: Index to start evaluation from (e.g. end of training set).
                         Prevents leakage by skipping training data.
        """
        try:
            splits = max(1, int(getattr(self.settings.models, "walkforward_splits", 0)))
            if splits < 2:
                return {"walkforward_splits": 0}

            X = dataset.X
            y_arr = np.asarray(dataset.y).reshape(-1)

            # Slice the dataset to only include data AFTER the start_index (OOS)
            if start_index >= len(X):
                logger.warning("Walk-forward start_index >= dataset length. Skipping.")
                return {"walkforward_splits": 0}

            X_oos = _slice_rows(X, start_index, len(X))
            y_oos = np.asarray(y_arr).reshape(-1)[start_index:]
            n = len(y_oos)

            if n < 200:  # Minimum samples needed for meaningful splits
                return {"walkforward_splits": 0, "reason": "not_enough_oos_data"}

            window = max(1, n // splits)

            results = []
            for i in range(splits):
                start = i * window
                end = min(n, (i + 1) * window)
                if end - start < 80:
                    break

                # In this "no-retrain" mode, we treat the OOS data as a sequence of test blocks
                # We don't really have a "train" phase here because the models are static.
                # However, to simulate a rolling window eval, we just predict on the chunk.

                chunk_X = _slice_rows(X_oos, start, end)
                chunk_y = y_oos[start:end]

                probas = ensemble_func(chunk_X)
                if probas is None or len(probas) == 0:
                    continue
                pred_labels = probas.argmax(axis=1)
                acc = float((pred_labels == np.asarray(chunk_y)).mean())
                results.append({"split": i + 1, "accuracy": acc})

            metrics = {
                "walkforward_splits": len(results),
                "avg_accuracy": float(np.mean([r["accuracy"] for r in results])) if results else 0.0,
                "splits": results,
            }

            # Save
            (self.models_dir / "walkforward_metrics.json").write_text(json.dumps(metrics, indent=2))
            return metrics

        except Exception as e:
            logger.warning(f"Walk-forward eval failed: {e}")
            return {}

    def run_cpcv(self, dataset: PreparedDataset, models: dict) -> dict[str, Any]:
        """Run Combinatorial Purged Cross-Validation."""
        if not self.settings.models.enable_cpcv:
            return {}

        try:
            logger.info("Running CPCV...")
            res = self._run_cpcv_numpy(dataset)
            if res:
                (self.models_dir / "cpcv_metrics.json").write_text(json.dumps(res, indent=2))
            return res
        except Exception as e:
            logger.error(f"CPCV eval failed: {e}")
            return {}

