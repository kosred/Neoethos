from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except Exception:
    IsotonicRegression = None  # type: ignore
    LogisticRegression = None  # type: ignore
    SKLEARN_AVAILABLE = False


def _canon_y(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=int)
    arr = np.where(arr == -1, 2, arr).astype(int, copy=False)
    arr = np.clip(arr, 0, 2)
    return arr


@dataclass
class _ClassCal:
    kind: str
    model: object | None
    constant: float | None


class ProbabilityCalibrator:
    """
    Lightweight multi-class probability calibration.
    Supports one-vs-rest Platt (logistic) and isotonic calibration.
    """

    def __init__(self, method: str = "platt") -> None:
        self.method = str(method or "platt").strip().lower()
        if self.method not in {"platt", "isotonic"}:
            self.method = "platt"
        self.fitted = False
        self.class_models: dict[int, _ClassCal] = {}

    def fit(self, probs: np.ndarray, y: np.ndarray) -> bool:
        if not SKLEARN_AVAILABLE:
            return False
        p = np.asarray(probs, dtype=float)
        if p.ndim != 2 or p.shape[1] < 3:
            return False
        yy = _canon_y(np.asarray(y, dtype=int))
        if len(yy) != p.shape[0] or len(yy) < 64:
            return False
        p = p[:, :3]
        p = np.clip(p, 1e-6, 1.0 - 1e-6)

        self.class_models.clear()
        for cls in (0, 1, 2):
            tgt = (yy == cls).astype(int)
            pos = int(tgt.sum())
            neg = int(len(tgt) - pos)
            if pos == 0 or neg == 0:
                self.class_models[cls] = _ClassCal(kind="const", model=None, constant=float(pos > 0))
                continue
            if self.method == "isotonic" and IsotonicRegression is not None:
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(p[:, cls], tgt)
                self.class_models[cls] = _ClassCal(kind="isotonic", model=model, constant=None)
            else:
                # Platt-style one-vs-rest calibration on log-odds.
                x_cls = np.log(p[:, cls] / (1.0 - p[:, cls])).reshape(-1, 1)
                lr = LogisticRegression(max_iter=300, n_jobs=1)
                lr.fit(x_cls, tgt)
                self.class_models[cls] = _ClassCal(kind="platt", model=lr, constant=None)
        self.fitted = len(self.class_models) == 3
        return self.fitted

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(probs, dtype=float)
        if arr.ndim != 2:
            return arr
        if not self.fitted or not self.class_models:
            return arr
        out = np.zeros((arr.shape[0], 3), dtype=float)
        p = np.clip(arr[:, :3], 1e-6, 1.0 - 1e-6)
        for cls in (0, 1, 2):
            cal = self.class_models.get(cls)
            if cal is None:
                out[:, cls] = p[:, cls]
                continue
            if cal.kind == "const":
                out[:, cls] = float(cal.constant or 0.0)
                continue
            if cal.kind == "isotonic" and cal.model is not None:
                out[:, cls] = np.asarray(cal.model.predict(p[:, cls]), dtype=float)
                continue
            if cal.kind == "platt" and cal.model is not None:
                logits = np.log(p[:, cls] / (1.0 - p[:, cls])).reshape(-1, 1)
                out[:, cls] = np.asarray(cal.model.predict_proba(logits)[:, 1], dtype=float)
                continue
            out[:, cls] = p[:, cls]
        out = np.clip(out, 1e-6, 1.0)
        rs = out.sum(axis=1, keepdims=True)
        rs = np.where(rs <= 0.0, 1.0, rs)
        return out / rs

