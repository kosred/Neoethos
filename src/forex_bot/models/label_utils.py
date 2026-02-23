from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def remap_labels_neutral_buy_sell(y: pd.Series | np.ndarray) -> np.ndarray:
    """
    Canonical 3-class mapping used by linear/xgboost paths:
    -1 (sell) -> 2, 0 (neutral) -> 0, 1 (buy) -> 1
    """
    arr = np.asarray(y, dtype=int)
    out = np.where(arr == -1, 2, arr).astype(int, copy=False)
    return np.clip(out, 0, 2)


def remap_labels_sell_neutral_buy(y: pd.Series | np.ndarray) -> np.ndarray:
    """
    Canonical 3-class mapping used by tree/rust paths:
    -1 (sell) -> 0, 0 (neutral) -> 1, 1 (buy) -> 2
    """
    arr = np.asarray(y, dtype=int)
    out = np.zeros_like(arr, dtype=int)
    out[arr == -1] = 0
    out[arr == 0] = 1
    out[arr == 1] = 2
    return out


def margins_to_probs(decision: np.ndarray) -> np.ndarray:
    """
    Convert decision margins to probability matrix.
    Binary margins map to 3-class output with sell near-zero.
    """
    dec = np.asarray(decision, dtype=float)
    if dec.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-np.clip(dec, -30, 30)))
        p0 = 1.0 - p1
        raw = np.stack([p0, p1, np.zeros_like(p0)], axis=1)
        rs = raw.sum(axis=1, keepdims=True)
        rs = np.where(rs <= 0, 1.0, rs)
        return raw / rs
    dec = dec - np.max(dec, axis=1, keepdims=True)
    ex = np.exp(np.clip(dec, -30, 30))
    rs = ex.sum(axis=1, keepdims=True)
    rs = np.where(rs <= 0, 1.0, rs)
    return ex / rs


def probs_to_three_class(
    probs: np.ndarray | Iterable,
    classes: np.ndarray | list[int] | None = None,
    *,
    class_to_output: dict[int, int] | None = None,
) -> np.ndarray:
    """
    Map arbitrary class-order probability outputs to fixed 3-class matrix.
    """
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    out = np.zeros((n, 3), dtype=float)

    if classes is None:
        if arr.shape[1] >= 3:
            out[:, :3] = arr[:, :3]
            return out
        if arr.shape[1] == 2:
            out[:, 0] = arr[:, 0]
            out[:, 1] = arr[:, 1]
            return out
        out[:, 0] = 1.0 - arr[:, 0]
        out[:, 1] = arr[:, 0]
        return out

    mapping = class_to_output or {0: 0, 1: 1, 2: 2}
    cls = [int(c) for c in list(classes)]
    for i, c in enumerate(cls):
        if i >= arr.shape[1]:
            break
        out_idx = mapping.get(c)
        if out_idx is not None and 0 <= int(out_idx) <= 2:
            out[:, int(out_idx)] = arr[:, i]

    rs = out.sum(axis=1, keepdims=True)
    rs = np.where(rs <= 0, 1.0, rs)
    return out / rs
