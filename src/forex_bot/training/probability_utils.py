from __future__ import annotations

from typing import Any

import numpy as np

try:
    import forex_bindings as _fb  # type: ignore
except Exception:
    _fb = None  # type: ignore


def pad_probs_neutral_buy_sell(probs: Any, classes: list[int] | None = None) -> np.ndarray:
    if probs is None:
        return np.zeros((0, 3), dtype=float)

    arr = np.asarray(probs, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    class_map = None if classes is None else [int(v) for v in list(classes)]
    if _fb is not None and hasattr(_fb, "pad_probs_neutral_buy_sell"):
        try:
            out = _fb.pad_probs_neutral_buy_sell(np.asarray(arr, dtype=np.float64), class_map)
            padded = np.asarray(out, dtype=float)
            if padded.ndim == 2 and padded.shape == (arr.shape[0], 3):
                return padded
        except Exception:
            pass

    n = int(arr.shape[0])
    out = np.zeros((n, 3), dtype=float)
    if class_map is not None and len(class_map) == int(arr.shape[1]):
        for col, cls_val in enumerate(class_map):
            if cls_val == 0:
                out[:, 0] = arr[:, col]
            elif cls_val == 1:
                out[:, 1] = arr[:, col]
            elif cls_val in (-1, 2):
                out[:, 2] = arr[:, col]
        return out
    if int(arr.shape[1]) == 3:
        return arr
    if int(arr.shape[1]) == 2:
        out[:, 0] = arr[:, 0]
        out[:, 1] = arr[:, 1]
        return out
    out[:, 0] = 1.0 - arr[:, 0]
    out[:, 1] = arr[:, 0]
    return out


def threshold_signals_and_accuracy(
    probs: Any,
    *,
    conf_threshold: float,
    y_true: Any | None = None,
    classes: list[int] | None = None,
) -> tuple[np.ndarray, float]:
    padded = pad_probs_neutral_buy_sell(probs, classes=classes)
    if padded.size == 0:
        return np.zeros(0, dtype=np.int8), 0.0

    labels = None
    if y_true is not None:
        labels = np.asarray(y_true, dtype=np.int64).reshape(-1)

    if _fb is not None and hasattr(_fb, "threshold_signals_and_accuracy"):
        try:
            out_signals, out_accuracy = _fb.threshold_signals_and_accuracy(
                np.asarray(padded, dtype=np.float64),
                float(conf_threshold),
                None if labels is None else np.asarray(labels, dtype=np.int64),
            )
            signals = np.asarray(out_signals, dtype=np.int8).reshape(-1)
            if signals.shape[0] == padded.shape[0]:
                return signals, float(out_accuracy)
        except Exception:
            pass

    p_buy = padded[:, 1]
    p_sell = padded[:, 2]
    trade_prob = np.maximum(p_buy, p_sell)
    direction = np.where(p_buy >= p_sell, 1, -1).astype(np.int8, copy=False)
    signals = np.where(trade_prob >= float(conf_threshold), direction, 0).astype(np.int8, copy=False)

    if labels is None or labels.size <= 0:
        return signals, 0.0

    n = int(min(signals.size, labels.size))
    if n <= 0:
        return signals, 0.0
    cmp_labels = np.asarray(labels[:n], dtype=np.int64)
    cmp_labels = np.where(cmp_labels == 2, -1, cmp_labels).astype(np.int64, copy=False)
    accuracy = float(np.mean(signals[:n].astype(np.int64, copy=False) == cmp_labels))
    return signals, accuracy
