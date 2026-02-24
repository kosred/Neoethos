from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _canon_y(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=int)
    arr = np.where(arr == -1, 2, arr).astype(int, copy=False)
    return np.clip(arr, 0, 2)


@dataclass
class ConformalClassifierGate:
    alpha: float = 0.10
    qhat: float = 1.0
    fitted: bool = False
    n_calib: int = 0

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> bool:
        p = np.asarray(probs, dtype=float)
        if p.ndim != 2 or p.shape[1] < 3:
            return False
        y = _canon_y(np.asarray(y_true, dtype=int))
        n = min(len(y), p.shape[0])
        if n < 64:
            return False
        p = np.clip(p[:n, :3], 1e-8, 1.0)
        y = y[:n]
        scores = 1.0 - p[np.arange(n), y]
        alpha = float(max(1e-6, min(0.99, self.alpha)))
        q_level = np.ceil((n + 1) * (1.0 - alpha)) / float(n)
        q_level = float(max(0.0, min(1.0, q_level)))
        try:
            q = float(np.quantile(scores, q_level, method="higher"))
        except TypeError:
            q = float(np.quantile(scores, q_level, interpolation="higher"))
        self.qhat = float(max(0.0, min(1.0, q)))
        self.fitted = True
        self.n_calib = int(n)
        return True

    def prediction_set(self, probs_row: np.ndarray) -> list[int]:
        row = np.asarray(probs_row, dtype=float).reshape(-1)
        if row.size < 3:
            return [0, 1, 2]
        p = np.clip(row[:3], 1e-8, 1.0)
        keep = [i for i, v in enumerate(1.0 - p) if float(v) <= float(self.qhat)]
        if not keep:
            return [int(np.argmax(p))]
        return keep

    def should_abstain(self, probs_row: np.ndarray, min_set_size: int = 3) -> tuple[bool, int]:
        if not self.fitted:
            return False, 1
        s = self.prediction_set(probs_row)
        size = int(len(s))
        return size >= int(max(1, min_set_size)), size

