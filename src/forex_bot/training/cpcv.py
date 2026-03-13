from __future__ import annotations

import logging
import os
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)
try:
    import forex_bindings as _fb  # type: ignore
except Exception:
    _fb = None  # type: ignore


def _is_dataframe_like(values: Any) -> bool:
    return bool(
        hasattr(values, "columns")
        and hasattr(values, "index")
        and callable(getattr(values, "to_numpy", None))
    )


class _NumpyFrame:
    """Minimal frame-like container used when CPCV slices non-dataframe-module frames."""

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


def _frame_columns(frame: Any) -> list[str]:
    cols = getattr(frame, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _frame_resolve_column(frame: Any, name: str) -> str | None:
    target = str(name).strip().lower()
    for col in _frame_columns(frame):
        if str(col).strip().lower() == target:
            return col
    return None


def _slice_frame_rows(values: Any, rows: np.ndarray) -> Any:
    rows_arr = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows_arr.size <= 0:
        return _NumpyFrame({}, np.zeros(0, dtype=np.int64))
    cols = _frame_columns(values)
    if not cols:
        return np.asarray(values)[rows_arr]
    out_data: dict[str, np.ndarray] = {}
    max_i = int(np.max(rows_arr)) if rows_arr.size > 0 else -1
    for col in cols:
        try:
            arr = np.asarray(values[col]).reshape(-1)  # type: ignore[index]
            if arr.shape[0] > max_i >= 0:
                out_data[str(col)] = arr[rows_arr]
            else:
                out_data[str(col)] = arr
        except Exception:
            continue
    idx = getattr(values, "index", None)
    if idx is None:
        out_idx = rows_arr.copy()
    else:
        idx_arr = np.asarray(idx).reshape(-1)
        if idx_arr.shape[0] > max_i >= 0:
            out_idx = idx_arr[rows_arr]
        else:
            out_idx = rows_arr.copy()
    attrs = getattr(values, "attrs", None)
    return _NumpyFrame(out_data, out_idx, attrs=(dict(attrs) if isinstance(attrs, dict) else None))


def _default_scoring_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true).reshape(-1)
    y_p = np.asarray(y_pred).reshape(-1)
    n = int(min(y_t.size, y_p.size))
    if n <= 0:
        return 0.0
    return float((y_t[:n] == y_p[:n]).mean())


def _slice_rows(values: Any, rows: np.ndarray) -> Any:
    if isinstance(values, dict):
        out: dict[str, Any] = {}
        max_i = int(np.max(rows)) if len(rows) > 0 else -1
        for k, v in values.items():
            try:
                arr = np.asarray(v)
                if arr.ndim <= 0:
                    out[str(k)] = v
                elif arr.shape[0] > max_i >= 0:
                    out[str(k)] = arr[rows]
                else:
                    out[str(k)] = arr
            except Exception:
                out[str(k)] = v
        return out
    if _is_dataframe_like(values):
        idx = np.asarray(rows, dtype=np.int64).reshape(-1)
        try:
            return values.take(idx)
        except Exception:
            try:
                base_idx = np.asarray(getattr(values, "index")).reshape(-1)
                return values.loc[base_idx[idx]]
            except Exception:
                pass
    if hasattr(values, "columns") and hasattr(values, "__getitem__"):
        try:
            return _slice_frame_rows(values, rows)
        except Exception:
            pass
    arr = np.asarray(values)
    return arr[rows]


def _extract_column(frame: Any, name: str) -> np.ndarray:
    col = None
    if isinstance(frame, dict):
        col = frame.get(name)
        if col is None:
            target = str(name).strip().lower()
            for k, v in frame.items():
                if str(k).strip().lower() == target:
                    col = v
                    break
    else:
        col_name = _frame_resolve_column(frame, name)
        if col_name is not None:
            try:
                col = frame[col_name]
            except Exception:
                col = None
    if col is None:
        return np.zeros(0, dtype=np.float64)
    try:
        if hasattr(col, "to_numpy"):
            return np.asarray(col.to_numpy(dtype=np.float64, copy=False), dtype=np.float64).reshape(-1)
    except Exception:
        pass
    return np.asarray(col, dtype=np.float64).reshape(-1)


def _extract_index(frame: Any, n_rows: int) -> np.ndarray:
    idx = None
    try:
        idx = getattr(frame, "index", None)
    except Exception:
        idx = None
    if idx is None and isinstance(frame, dict):
        idx = frame.get("index")
    if idx is None:
        return np.arange(n_rows, dtype=np.int64)
    arr = np.asarray(idx).reshape(-1)
    if arr.size != n_rows:
        return np.arange(n_rows, dtype=np.int64)
    return arr


def _month_day_indices(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(index).reshape(-1)
    n = int(arr.size)
    if n <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    def _rust_from_ns(ns_values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if _fb is None or not hasattr(_fb, "derive_time_index_arrays"):
            return None
        try:
            _unix_ms, month_idx, day_idx = _fb.derive_time_index_arrays(
                np.asarray(ns_values, dtype=np.int64).reshape(-1)
            )
        except Exception:
            return None
        month_arr = np.asarray(month_idx, dtype=np.int64).reshape(-1)
        day_arr = np.asarray(day_idx, dtype=np.int64).reshape(-1)
        if month_arr.size != n or day_arr.size != n:
            return None
        return month_arr, day_arr

    if np.issubdtype(arr.dtype, np.datetime64):
        dt = arr.astype("datetime64[ns]")
        rust = _rust_from_ns(dt.astype(np.int64, copy=False))
        if rust is not None:
            return rust
        return dt.astype("datetime64[M]").astype(np.int64), dt.astype("datetime64[D]").astype(np.int64)

    if arr.dtype.kind in {"i", "u"}:
        ints = arr.astype(np.int64, copy=False)
        if ints.size > 0 and int(np.max(np.abs(ints))) > 10**12:
            rust = _rust_from_ns(ints)
            if rust is not None:
                return rust
            dt = ints.astype("datetime64[ns]")
            return dt.astype("datetime64[M]").astype(np.int64), dt.astype("datetime64[D]").astype(np.int64)
        return ints // 31, ints

    if arr.dtype.kind == "f":
        ints = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
        return ints // 31, ints

    day = np.arange(n, dtype=np.int64)
    return day // 31, day


def _index_is_monotonic(index: Any) -> bool:
    if index is None:
        return True
    raw = getattr(index, "is_monotonic_increasing", None)
    if raw is not None:
        try:
            return bool(raw)
        except Exception:
            pass
    arr = np.asarray(index).reshape(-1)
    if arr.size <= 1:
        return True
    if np.issubdtype(arr.dtype, np.datetime64):
        vals = arr.astype("datetime64[ns]").astype(np.int64, copy=False)
    elif arr.dtype.kind in {"i", "u"}:
        vals = arr.astype(np.int64, copy=False)
    elif arr.dtype.kind == "f":
        vals = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
    else:
        # If index is non-numeric/non-datetime, keep original behavior permissive.
        return True
    return bool(np.all(vals[1:] >= vals[:-1]))


def _evaluate_split_task(
    x: Any,
    y: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_factory: Callable,
    scoring_func: Callable,
    sample_weights: Any | None,
    max_daily_loss_pct: float,
    max_trades_per_day: int,
    min_trading_days: int,
) -> tuple[float, dict[str, Any]]:
    """Picklable worker for ProcessPoolExecutor."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return 0.0, {}

    x_train = _slice_rows(x, train_idx)
    y_train = _slice_rows(y, train_idx)
    x_test = _slice_rows(x, test_idx)
    y_test = _slice_rows(y, test_idx)

    if sample_weights is not None:
        w_train = _slice_rows(sample_weights, train_idx)
    else:
        w_train = None

    model = model_factory()

    try:
        if w_train is not None and hasattr(model, "fit") and "sample_weight" in model.fit.__code__.co_varnames:
            model.fit(x_train, y_train, sample_weight=w_train)
        else:
            model.fit(x_train, y_train)
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return 0.0, {}

    try:
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(x_test)
            classes = getattr(model, "classes_", None)
            if classes is not None and len(classes) == y_pred_proba.shape[1]:
                y_pred = np.asarray(classes)[y_pred_proba.argmax(axis=1)]
            else:
                y_pred = y_pred_proba.argmax(axis=1)
        else:
            y_pred = model.predict(x_test)
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        return 0.0, {}

    try:
        score = scoring_func(np.asarray(y_test), y_pred)
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return 0.0, {}

    from ..strategy.fast_backtest import fast_evaluate_strategy, infer_pip_metrics

    max_dd = 0.0
    win_rate = 0.0
    trades = 0

    close = _extract_column(x_test, "close")
    high = _extract_column(x_test, "high")
    low = _extract_column(x_test, "low")
    if close.size > 0 and high.size > 0 and low.size > 0:
        try:
            n_bt = int(min(close.size, high.size, low.size, len(y_pred)))
            close = close[:n_bt]
            high = high[:n_bt]
            low = low[:n_bt]
            signals = np.asarray(y_pred, dtype=np.int8).reshape(-1)[:n_bt]
            idx = _extract_index(x_test, n_bt)
            month_idx, day_idx = _month_day_indices(idx)

            symbol = "EURUSD"
            with np.errstate(all="ignore"):
                try:
                    attrs = getattr(x_test, "attrs", None)
                    if isinstance(attrs, dict):
                        symbol = str(attrs.get("symbol", "EURUSD") or "EURUSD")
                except Exception:
                    symbol = "EURUSD"
            pip_size, pip_val_lot = infer_pip_metrics(symbol)

            # HPC Unified Backtest
            arr = fast_evaluate_strategy(
                close_prices=close,
                high_prices=high,
                low_prices=low,
                signals=signals,
                month_indices=month_idx,
                day_indices=day_idx,
                sl_pips=30.0,
                tp_pips=60.0,
                pip_value=pip_size,
                pip_value_per_lot=pip_val_lot,
                spread_pips=1.5,
                commission_per_trade=7.0,
            )

            max_dd = float(arr[3])
            win_rate = float(arr[4])
            trades = int(arr[8])

        except Exception as e:
            logger.error(f"CPCV internal backtest failed: {e}")
            max_dd = 1.0

    metrics = {
        "max_dd": max_dd,
        "trades": trades,
        "win_rate": win_rate,
        "daily_loss_breach": max_dd > max_daily_loss_pct,
        "trade_limit_violation": trades > max_trades_per_day,
        "min_trading_days_ok": trades >= min_trading_days,
    }

    return float(score), metrics


@dataclass(slots=True)
class CPCVResult:
    """Results from CPCV evaluation"""

    mean_score: float
    std_score: float
    scores: list[float]
    n_combinations: int
    phi: float  # Uniqueness score (lower = more overfitting risk)
    avg_max_dd: float = 0.0
    avg_trades: float = 0.0
    avg_win_rate: float = 0.0
    any_daily_loss_breach: bool = False
    any_trade_limit_violation: bool = False
    all_min_trading_days_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_score": float(self.mean_score),
            "std_score": float(self.std_score),
            "scores": [float(s) for s in self.scores],
            "n_combinations": int(self.n_combinations),
            "phi": float(self.phi),
            "avg_max_dd": float(self.avg_max_dd),
            "avg_trades": float(self.avg_trades),
            "avg_win_rate": float(self.avg_win_rate),
            "any_daily_loss_breach": bool(self.any_daily_loss_breach),
            "any_trade_limit_violation": bool(self.any_trade_limit_violation),
            "all_min_trading_days_ok": bool(self.all_min_trading_days_ok),
        }


class CombinatorialPurgedCV:
    """
    CPCV: The truth serum for backtesting.

    Unlike standard K-Fold, CPCV:
    - Purges samples with overlapping outcome periods from test set
    - Adds embargo periods between train/test to prevent information leakage
    - Tests all combinations of K groups, not just sequential splits

    Parameters
    ----------
    n_splits : int
        Number of groups to split data into (default 5)
    n_test_groups : int
        Number of groups to use for testing in each combination (default 2)
    embargo_pct : float
        Percentage of samples to embargo after train set (default 0.01 = 1%)
    purge_pct : float
        Percentage of samples to purge before test set (default 0.02 = 2%)
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.02,
    ) -> None:
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        x: Any,
        y: Any | None = None,
        sample_weights: Any | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorial purged train/test splits.

        A small warm-up window is always kept for training so that even the earliest
        test combinations have non-empty train sets. An embargo buffer is inserted
        between the warm-up window and the first test fold to guarantee a visible
        gap between train and test indices.

        Returns list of (train_indices, test_indices) tuples.
        """
        # CPCV relies on time-ordered data for purging/embargo to be meaningful.
        idx = getattr(x, "index", None)
        if idx is not None and not _index_is_monotonic(idx):
            raise ValueError(
                "CPCV requires time-ordered data (monotonic increasing index). "
                "Sort by timestamp before running CPCV to avoid look-ahead bias."
            )

        n_samples = len(x)
        if n_samples == 0:
            raise ValueError("CPCV requires non-empty data")

        purge_size = int(np.ceil(n_samples * self.purge_pct)) if self.purge_pct > 0 else 0
        embargo_size = int(np.ceil(n_samples * self.embargo_pct)) if self.embargo_pct > 0 else 0

        warmup_size = max(n_samples // self.n_splits, purge_size + embargo_size, self.n_splits)
        if warmup_size + self.n_splits >= n_samples:
            warmup_size = max(1, n_samples // (self.n_splits + 1))

        cv_start = min(n_samples, warmup_size + embargo_size)
        if n_samples - cv_start < self.n_splits:
            raise ValueError("Not enough samples after warm-up/embargo for requested splits")

        indices = np.arange(n_samples)
        base_train = indices[:warmup_size]
        cv_indices = indices[cv_start:]

        kfold = KFold(n_splits=self.n_splits, shuffle=False)
        groups: list[np.ndarray] = []
        for _, test_idx in kfold.split(cv_indices):
            groups.append(cv_indices[test_idx])

        test_combinations = list(combinations(range(self.n_splits), self.n_test_groups))
        logger.info(
            f"CPCV: {len(test_combinations)} combinations of {self.n_test_groups} test"
            f" groups from {self.n_splits} splits"
        )

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for test_group_indices in test_combinations:
            test_idx = np.concatenate([groups[i] for i in test_group_indices])
            test_idx.sort()

            if len(test_idx) == 0:
                splits.append((np.array([], dtype=int), np.array([], dtype=int)))
                continue

            earliest_group = min(test_group_indices)
            train_parts = [base_train]
            if earliest_group > 0:
                train_parts.extend(groups[i] for i in range(earliest_group) if i not in test_group_indices)

            train_idx = np.concatenate(train_parts) if len(train_parts) > 0 else np.array([], dtype=int)
            train_idx.sort()

            train_idx, test_idx = self._purge_and_embargo(train_idx, test_idx, n_samples, purge_size, embargo_size)

            if len(train_idx) == 0 and len(base_train) > 0:
                train_idx = base_train.copy()
                train_idx, test_idx = self._purge_and_embargo(train_idx, test_idx, n_samples, purge_size, embargo_size)

            splits.append((train_idx, test_idx))

        return splits

    def _purge_and_embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_samples: int,
        purge_size: int,
        embargo_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply purging and embargo to prevent data leakage.

        Purging: Remove train samples that overlap with test outcomes
        Embargo: Add time gap between train and test
        """
        if len(test_idx) == 0 or len(train_idx) == 0:
            return train_idx, test_idx

        test_start = int(test_idx[0])

        purge_threshold = max(0, test_start - purge_size)
        train_idx = train_idx[train_idx < purge_threshold]

        if len(train_idx) > 0:
            train_end = int(train_idx[-1])
            embargo_threshold = min(n_samples, train_end + embargo_size)
            test_idx = test_idx[test_idx >= embargo_threshold]

        return train_idx, test_idx

    def calculate_phi(
        self,
        x: Any,
    ) -> float:
        """
        Calculate uniqueness score φ (phi).

        φ = average number of times each sample appears in test set / n_combinations

        φ close to 1.0 = good (each sample tested ~once)
        φ much lower = potential overfitting (some samples rarely tested)
        """
        n_samples = len(x)
        test_counts = np.zeros(n_samples)

        splits = self.split(x)

        for _, test_idx in splits:
            test_counts[test_idx] += 1

        if n_samples == 0:
            return 0.0

        tested_mask = test_counts > 0
        coverage = tested_mask.mean()

        if tested_mask.any():
            avg_tests = float(test_counts[tested_mask].mean())
            frequency_score = avg_tests / max(1, len(splits))
        else:
            frequency_score = 0.0

        phi = float(np.clip(0.5 * coverage + 0.5 * frequency_score, 0.0, 1.0))
        return phi

    def score(
        self,
        x: Any,
        y: Any,
        model_factory: Callable,
        scoring_func: Callable | None = None,
        sample_weights: Any | None = None,
        n_jobs: int = 1,
        max_daily_loss_pct: float = 0.05,
        max_trades_per_day: int = 15,
        min_trading_days: int = 3,
    ) -> CPCVResult:
        """
        Evaluate model using CPCV.

        Parameters
        ----------
        x : Any
            Feature matrix (NumPy array or dataframe-like).
        y : Any
            Labels (NumPy array or series-like).
        model_factory : Callable
            Function that returns a new untrained model instance
        scoring_func : Callable, optional
            Function(y_true, y_pred) -> score. If None, uses accuracy.
        sample_weights : Any, optional
            Sample weights for training.
        n_jobs : int
            Number of parallel jobs (default 1)

        Returns
        -------
        CPCVResult
            Evaluation results with mean score, std, and phi
        """
        splits = self.split(x, y, sample_weights)

        if scoring_func is None:
            scoring_func = _default_scoring_func

        scores = []
        dd_list: list[float] = []
        trades_list: list[float] = []
        winrate_list: list[float] = []
        daily_loss_breaches = []
        trade_limit_breaches = []
        min_days_ok_list = []

        if n_jobs <= 0:
            n_jobs = int(os.cpu_count() or 1)
        n_jobs = max(1, n_jobs)
        # Optional safety cap via env (unset means no cap)
        try:
            max_jobs_env = int(os.environ.get("FOREX_BOT_CPCV_MAX_JOBS", "0") or 0)
        except Exception:
            max_jobs_env = 0
        if max_jobs_env > 0:
            n_jobs = min(n_jobs, max_jobs_env)
        # Avoid spawning more workers than folds
        n_jobs = min(n_jobs, len(splits))

        if n_jobs > 1:
            # Use spawn context to avoid CUDA fork issues
            spawn_ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=spawn_ctx) as executor:
                futures = []
                for train_idx, test_idx in splits:
                    future = executor.submit(
                        _evaluate_split_task,
                        x,
                        y,
                        train_idx,
                        test_idx,
                        model_factory,
                        scoring_func,
                        sample_weights,
                        max_daily_loss_pct,
                        max_trades_per_day,
                        min_trading_days,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        score, metrics = future.result()
                        scores.append(score)
                        dd_list.append(metrics.get("max_dd", 0.0))
                        trades_list.append(metrics.get("trades", 0))
                        winrate_list.append(metrics.get("win_rate", 0.0))
                        daily_loss_breaches.append(metrics.get("daily_loss_breach", False))
                        trade_limit_breaches.append(metrics.get("trade_limit_violation", False))
                        min_days_ok_list.append(metrics.get("min_trading_days_ok", True))
                    except Exception as e:
                        logger.error(f"CPCV split failed: {e}")
                        scores.append(0.0)
                        dd_list.append(0.0)
                        trades_list.append(0)
                        winrate_list.append(0.0)
                        daily_loss_breaches.append(False)
                        trade_limit_breaches.append(False)
                        min_days_ok_list.append(False)
        else:
            for train_idx, test_idx in splits:
                try:
                    score, metrics = _evaluate_split_task(
                        x,
                        y,
                        train_idx,
                        test_idx,
                        model_factory,
                        scoring_func,
                        sample_weights,
                        max_daily_loss_pct,
                        max_trades_per_day,
                        min_trading_days,
                    )
                    scores.append(score)
                    dd_list.append(metrics.get("max_dd", 0.0))
                    trades_list.append(metrics.get("trades", 0))
                    winrate_list.append(metrics.get("win_rate", 0.0))
                    daily_loss_breaches.append(metrics.get("daily_loss_breach", False))
                    trade_limit_breaches.append(metrics.get("trade_limit_violation", False))
                    min_days_ok_list.append(metrics.get("min_trading_days_ok", True))
                except Exception as e:
                    logger.error(f"CPCV split failed: {e}")
                    scores.append(0.0)
                    dd_list.append(0.0)
                    trades_list.append(0)
                    winrate_list.append(0.0)
                    daily_loss_breaches.append(False)
                    trade_limit_breaches.append(False)
                    min_days_ok_list.append(False)

        phi = self.calculate_phi(x)

        return CPCVResult(
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            scores=scores,
            n_combinations=len(splits),
            phi=phi,
            avg_max_dd=float(np.mean(dd_list)) if dd_list else 0.0,
            avg_trades=float(np.mean(trades_list)) if trades_list else 0.0,
            avg_win_rate=float(np.mean(winrate_list)) if winrate_list else 0.0,
            any_daily_loss_breach=any(daily_loss_breaches),
            any_trade_limit_violation=any(trade_limit_breaches),
            all_min_trading_days_ok=all(min_days_ok_list) if min_days_ok_list else False,
        )

    @staticmethod
    def _evaluate_split(
        x: Any,
        y: Any,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        model_factory: Callable,
        scoring_func: Callable,
        sample_weights: Any | None,
        max_daily_loss_pct: float,
        max_trades_per_day: int,
        min_trading_days: int,
    ) -> tuple[float, dict[str, Any]]:
        return _evaluate_split_task(
            x,
            y,
            train_idx,
            test_idx,
            model_factory,
            scoring_func,
            sample_weights,
            max_daily_loss_pct,
            max_trades_per_day,
            min_trading_days,
        )


def cpcv_backtest(
    x: Any,
    y: Any,
    metadata: Any,
    model_factory: Callable,
    n_splits: int = 5,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    purge_pct: float = 0.02,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """
    Run CPCV backtest with trading metrics.

    Returns detailed metrics including PnL, win rate, Sharpe, etc.
    """
    from ..training.evaluation import probs_to_signals, prop_backtest, quick_backtest

    cv = CombinatorialPurgedCV(
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        embargo_pct=embargo_pct,
        purge_pct=purge_pct,
    )

    splits = cv.split(x, y)

    all_backtests = []

    for i, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        x_train = _slice_rows(x, train_idx)
        y_train = _slice_rows(y, train_idx)
        x_test = _slice_rows(x, test_idx)

        model = model_factory()

        try:
            model.fit(x_train, y_train)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_test)
                signals = probs_to_signals(probs)
            else:
                signals = model.predict(x_test)

            test_metadata = _slice_rows(metadata, test_idx)
            bt_metrics: dict[str, Any] = {}
            try:
                bt_metrics = prop_backtest(test_metadata, signals)
            except Exception:
                bt_metrics = {}
            if not bt_metrics:
                bt_metrics = quick_backtest(test_metadata, signals)

            if bt_metrics:
                if "pnl_score" not in bt_metrics and "net_profit" in bt_metrics:
                    bt_metrics["pnl_score"] = float(bt_metrics.get("net_profit", 0.0))
                if "max_dd" not in bt_metrics and "max_dd_pct" in bt_metrics:
                    bt_metrics["max_dd"] = float(bt_metrics.get("max_dd_pct", 0.0))
                bt_metrics["split"] = i + 1
                all_backtests.append(bt_metrics)

        except Exception as e:
            logger.error(f"CPCV backtest split {i + 1} failed: {e}")

    if not all_backtests:
        return {"n_splits": 0, "error": "All splits failed"}

    result = {
        "n_splits": len(all_backtests),
        "n_combinations": len(splits),
        "phi": cv.calculate_phi(x),
        "avg_pnl": np.mean([b.get("pnl_score", 0) for b in all_backtests]),
        "std_pnl": np.std([b.get("pnl_score", 0) for b in all_backtests]),
        "avg_win_rate": np.mean([b.get("win_rate", 0) for b in all_backtests]),
        "avg_sharpe": np.mean([b.get("sharpe", 0) for b in all_backtests]),
        "avg_trades": np.mean([b.get("trades", 0) for b in all_backtests]),
        "splits": all_backtests,
    }

    return result

