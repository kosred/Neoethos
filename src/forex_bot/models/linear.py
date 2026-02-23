from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .base import ExpertModel
from .label_utils import margins_to_probs, probs_to_three_class, remap_labels_neutral_buy_sell

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
    from sklearn.tree import DecisionTreeClassifier

    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None  # type: ignore
    PassiveAggressiveClassifier = None  # type: ignore
    SGDClassifier = None  # type: ignore
    DecisionTreeClassifier = None  # type: ignore
    SKLEARN_AVAILABLE = False

try:
    from vowpalwabbit.sklearn_vw import VWClassifier

    VW_AVAILABLE = True
except Exception:
    VWClassifier = None  # type: ignore
    VW_AVAILABLE = False

try:
    from river.tree import HoeffdingTreeClassifier as RiverHoeffdingTreeClassifier

    RIVER_AVAILABLE = True
except Exception:
    RiverHoeffdingTreeClassifier = None  # type: ignore
    RIVER_AVAILABLE = False


def _canon_y(y: pd.Series | np.ndarray) -> np.ndarray:
    # Canonical class order: 0=neutral, 1=buy, 2=sell.
    return remap_labels_neutral_buy_sell(y)


def _align_features(x: pd.DataFrame, cols: list[str] | None) -> pd.DataFrame:
    if cols is None or not cols:
        return x
    out = x.reindex(columns=cols, fill_value=0.0)
    return out


def _pad_probs_with_classes(probs: np.ndarray, classes: np.ndarray | list[int] | None) -> np.ndarray:
    return probs_to_three_class(probs, classes)


def _decision_to_probs(decision: np.ndarray) -> np.ndarray:
    return margins_to_probs(decision)


def _iter_feature_dict_rows(x_df: pd.DataFrame):
    """
    Convert numeric frame rows to feature dicts with lower overhead than per-row pandas to_dict.
    """
    names = [str(c) for c in x_df.columns]
    arr = x_df.to_numpy(dtype=np.float32, copy=False)
    n_feat = int(arr.shape[1]) if arr.ndim == 2 else 0
    for row in arr:
        yield {names[j]: float(row[j]) for j in range(n_feat)}


class _LinearBase(ExpertModel):
    model_name: str = "linear_base"

    def __init__(self, **kwargs: Any) -> None:
        self.model: Any = None
        self.feature_columns: list[str] | None = None
        self.classes_: np.ndarray | None = None
        self.constant_proba: np.ndarray | None = None
        self.params = dict(kwargs)

    def _build_model(self) -> Any:
        raise NotImplementedError

    def _prepare_x(self, x: pd.DataFrame) -> np.ndarray:
        x_df = x.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return x_df.to_numpy(dtype=np.float32, copy=False)

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:  # noqa: ARG002
        if not SKLEARN_AVAILABLE:
            logger.warning("%s skipped: scikit-learn not available.", self.model_name)
            self.model = None
            return
        if x is None or len(x) == 0:
            self.model = None
            return
        x_df = x.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.feature_columns = list(x_df.columns)
        x_arr = x_df.to_numpy(dtype=np.float32, copy=False)
        y_arr = _canon_y(y)
        uniq, cnt = np.unique(y_arr, return_counts=True)
        if len(uniq) < 2:
            pri = np.zeros(3, dtype=float)
            c = int(uniq[0]) if len(uniq) else 0
            pri[c] = 1.0
            self.constant_proba = pri
            self.classes_ = np.array([0, 1, 2], dtype=int)
            self.model = None
            return
        self.constant_proba = None
        self.model = self._build_model()
        self.model.fit(x_arr, y_arr)
        self.classes_ = np.asarray(getattr(self.model, "classes_", np.array([0, 1, 2], dtype=int)))

    def predict_proba(self, x: pd.DataFrame, **kwargs: Any) -> np.ndarray:  # noqa: ARG002
        if x is None:
            return np.zeros((0, 3), dtype=float)
        if self.constant_proba is not None:
            n = len(x)
            return np.tile(self.constant_proba.reshape(1, -1), (n, 1))
        if self.model is None:
            return np.zeros((len(x), 3), dtype=float)
        x_df = _align_features(
            x.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0),
            self.feature_columns,
        )
        x_arr = x_df.to_numpy(dtype=np.float32, copy=False)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_arr)
            classes = getattr(self.model, "classes_", self.classes_)
            return _pad_probs_with_classes(probs, classes)
        decision = self.model.decision_function(x_arr)
        probs = _decision_to_probs(decision)
        return _pad_probs_with_classes(probs, self.classes_)

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "classes_": self.classes_,
            "constant_proba": self.constant_proba,
            "params": self.params,
        }
        joblib.dump(payload, p / f"{self.model_name}.joblib")

    def load(self, path: str) -> None:
        fp = Path(path) / f"{self.model_name}.joblib"
        if not fp.exists():
            return
        try:
            payload = joblib.load(fp)
            if isinstance(payload, dict):
                self.model = payload.get("model")
                self.feature_columns = payload.get("feature_columns")
                self.classes_ = payload.get("classes_")
                self.constant_proba = payload.get("constant_proba")
                self.params = dict(payload.get("params") or self.params)
        except Exception as exc:
            logger.warning("Failed loading %s: %s", self.model_name, exc)


class ElasticNetExpert(_LinearBase):
    model_name = "elasticnet"

    def _build_model(self) -> Any:
        if SGDClassifier is None:
            raise RuntimeError("scikit-learn missing")
        alpha = float(self.params.get("alpha", 1e-4) or 1e-4)
        l1_ratio = float(self.params.get("l1_ratio", 0.5) or 0.5)
        max_iter = int(self.params.get("max_iter", 2000) or 2000)
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        )


class BayesianLogitExpert(_LinearBase):
    """
    Robust Bayesian-style logistic baseline.
    Uses stronger regularization as a practical approximation to informative priors.
    """

    model_name = "bayes_logit"

    def _build_model(self) -> Any:
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn missing")
        c_val = float(self.params.get("C", 0.5) or 0.5)
        return LogisticRegression(
            max_iter=int(self.params.get("max_iter", 800) or 800),
            solver="lbfgs",
            class_weight="balanced",
            C=max(1e-4, c_val),
            multi_class="auto",
            n_jobs=1,
            random_state=42,
        )


class OnlinePassiveAggressiveExpert(_LinearBase):
    model_name = "online_pa"

    def _build_model(self) -> Any:
        if PassiveAggressiveClassifier is None:
            raise RuntimeError("scikit-learn missing")
        return PassiveAggressiveClassifier(
            C=float(self.params.get("C", 0.5) or 0.5),
            max_iter=int(self.params.get("max_iter", 2000) or 2000),
            class_weight="balanced",
            random_state=42,
        )


class OnlineHoeffdingExpert(_LinearBase):
    model_name = "online_hoeffding"

    def _build_model(self) -> Any:
        # River Hoeffding is optional; use a stable shallow tree fallback when unavailable.
        if RIVER_AVAILABLE:
            return RiverHoeffdingTreeClassifier()
        if DecisionTreeClassifier is None:
            raise RuntimeError("scikit-learn missing")
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:  # noqa: ARG002
        if RIVER_AVAILABLE:
            x_df = x.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            self.feature_columns = list(x_df.columns)
            y_arr = _canon_y(y)
            model = self._build_model()
            for feats, yi in zip(_iter_feature_dict_rows(x_df), y_arr, strict=False):
                model.learn_one(feats, int(yi))
            self.model = model
            self.classes_ = np.array([0, 1, 2], dtype=int)
            self.constant_proba = None
            return
        super().fit(x, y, **kwargs)

    def predict_proba(self, x: pd.DataFrame, **kwargs: Any) -> np.ndarray:  # noqa: ARG002
        if RIVER_AVAILABLE and self.model is not None:
            if x is None or len(x) == 0:
                return np.zeros((0, 3), dtype=float)
            x_df = _align_features(
                x.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0),
                self.feature_columns,
            )
            out = np.zeros((len(x_df), 3), dtype=float)
            for i, row in enumerate(_iter_feature_dict_rows(x_df)):
                p = self.model.predict_proba_one(row) or {}
                out[i, 0] = float(p.get(0, 0.0))
                out[i, 1] = float(p.get(1, 0.0))
                out[i, 2] = float(p.get(2, 0.0))
                s = float(out[i].sum())
                if s <= 0:
                    out[i, 0] = 1.0
                else:
                    out[i] /= s
            return out
        return super().predict_proba(x, **kwargs)


class VowpalWabbitExpert(_LinearBase):
    model_name = "vw"

    def _build_model(self) -> Any:
        if not VW_AVAILABLE:
            raise RuntimeError("vowpalwabbit not available")
        return VWClassifier(loss_function="logistic", oaa=3, passes=3, random_seed=42)

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:  # noqa: ARG002
        if not VW_AVAILABLE:
            logger.warning("vw skipped: vowpalwabbit not available.")
            self.model = None
            return
        super().fit(x, y, **kwargs)
