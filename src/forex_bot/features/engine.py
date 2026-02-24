from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..core.config import Settings
from ..domain.events import PreparedDataset, SignalResult
from ..training.calibration import ProbabilityCalibrator
from ..training.conformal import ConformalClassifierGate
from ..training.evaluation import pad_probs, probs_to_signals
from ..training.persistence_service import PersistenceService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ModelProba:
    name: str
    proba: np.ndarray


class SignalEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models_dir = Path(os.environ.get("FOREX_BOT_MODELS_DIR", "models"))
        self.logs_dir = Path(os.environ.get("FOREX_BOT_LOGS_DIR", "logs"))
        self.persistence = PersistenceService(self.models_dir, self.logs_dir, settings=self.settings)
        self.models: dict[str, Any] = {}
        self.meta_blender: Any | None = None
        self._onnx = None
        self.calibrators: dict[str, ProbabilityCalibrator] = {}
        self.conformal_gate: ConformalClassifierGate | None = None
        self.selected_features: list[str] = []
        self.selected_features_by_regime: dict[str, list[str]] = {}

    def _load_selected_features(self) -> list[str]:
        path = self.models_dir / "selected_features.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [str(c) for c in payload if str(c).strip()]
        except Exception as exc:
            logger.debug("SignalEngine: failed to load selected features: %s", exc)
        return []

    def _load_selected_features_by_regime(self) -> dict[str, list[str]]:
        path = self.models_dir / "selected_features_by_regime.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {}
            out: dict[str, list[str]] = {}
            for key, vals in payload.items():
                bucket = str(key).strip().lower()
                if not bucket or not isinstance(vals, list):
                    continue
                cols = [str(c) for c in vals if str(c).strip()]
                if cols:
                    out[bucket] = cols
            return out
        except Exception as exc:
            logger.debug("SignalEngine: failed to load per-regime selected features: %s", exc)
            return {}

    def _load_calibrators(self) -> dict[str, ProbabilityCalibrator]:
        path = self.models_dir / "calibrators.joblib"
        if not path.exists():
            return {}
        try:
            payload = joblib.load(path)
            if isinstance(payload, dict):
                out = {str(k): v for k, v in payload.items() if isinstance(v, ProbabilityCalibrator)}
                return out
        except Exception as exc:
            logger.debug("SignalEngine: failed to load calibrators: %s", exc)
        return {}

    def _load_conformal_gate(self) -> ConformalClassifierGate | None:
        path = self.models_dir / "conformal_gate.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            if not bool(payload.get("enabled", True)):
                return None
            gate = ConformalClassifierGate(alpha=float(payload.get("alpha", 0.10) or 0.10))
            gate.qhat = float(payload.get("qhat", 1.0) or 1.0)
            gate.fitted = bool(payload.get("fitted", False))
            gate.n_calib = int(payload.get("n_calib", 0) or 0)
            return gate if gate.fitted else None
        except Exception as exc:
            logger.debug("SignalEngine: failed to load conformal gate: %s", exc)
            return None

    def load_models(self, models_dir: str | Path | None = None) -> None:
        if models_dir is not None:
            self.models_dir = Path(models_dir)
            self.persistence = PersistenceService(self.models_dir, self.logs_dir, settings=self.settings)
        try:
            self.models, self.meta_blender = self.persistence.load_models()
        except Exception as exc:
            logger.warning("Failed to load models: %s", exc)
            self.models = {}
            self.meta_blender = None
        self.selected_features = self._load_selected_features()
        self.selected_features_by_regime = self._load_selected_features_by_regime()
        self.calibrators = self._load_calibrators()
        self.conformal_gate = self._load_conformal_gate()
        # Optional ONNX inference path
        use_onnx = str(os.environ.get("FOREX_BOT_USE_ONNX", "")).strip().lower() in {"1", "true", "yes", "on"}
        if use_onnx:
            try:
                from ..models.onnx_exporter import ONNXExporter

                exporter = ONNXExporter(str(self.models_dir))
                exporter.load_models()
                self._onnx = exporter
            except Exception as exc:
                logger.warning("ONNX load failed: %s", exc)
                self._onnx = None

    def _align_selected_features(self, X: pd.DataFrame, regime_bucket: str | None = None) -> pd.DataFrame:
        cols: list[str] = []
        use_regime = bool(getattr(self.settings.models, "l1_feature_selection_per_regime", True))
        if use_regime and regime_bucket:
            cols = list(self.selected_features_by_regime.get(str(regime_bucket).strip().lower(), []) or [])
        if not cols:
            cols = list(self.selected_features or [])
        if not cols:
            return X
        return X.reindex(columns=cols, fill_value=0.0)

    def _predict_model_proba(self, name: str, model: Any, X: pd.DataFrame) -> np.ndarray | None:
        try:
            proba = pad_probs(model.predict_proba(X))
            if proba.shape[0] != len(X):
                return None
            cal = self.calibrators.get(name)
            if cal is not None:
                proba = cal.predict_proba(proba)
            return proba
        except Exception as exc:
            logger.debug("Model prediction failed (%s): %s", name, exc)
            return None

    def _collect_model_probas(self, X: pd.DataFrame, models_subset: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        models = models_subset if models_subset is not None else self.models
        out: dict[str, np.ndarray] = {}
        for name, model in models.items():
            proba = self._predict_model_proba(name, model, X)
            if proba is not None:
                out[name] = proba
        return out

    def _ensemble_proba(
        self,
        X: pd.DataFrame,
        models_subset: dict[str, Any] | None = None,
        *,
        proba_cache: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        cache = proba_cache if proba_cache is not None else self._collect_model_probas(X, models_subset=models_subset)
        probas = list(cache.values())
        if not probas:
            return np.zeros((len(X), 3), dtype=float)
        return np.mean(probas, axis=0)

    def _meta_features(
        self,
        X: pd.DataFrame,
        models_subset: dict[str, Any] | None = None,
        *,
        proba_cache: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        cache = proba_cache if proba_cache is not None else self._collect_model_probas(X, models_subset=models_subset)
        feats = pd.DataFrame(index=X.index)
        for name, proba in cache.items():
            feats[f"{name}_buy"] = proba[:, 1]
        if feats.empty:
            feats = pd.DataFrame(np.zeros((len(X), 1)), columns=["dummy"], index=X.index)
        return feats

    def _infer_regime_bucket(self, X: pd.DataFrame, last_idx: int) -> str:
        # Heuristic regime routing from ADX + realized volatility proxy.
        adx = 0.0
        vol = self._latest_market_volatility(X, last_idx)
        row = X.iloc[last_idx]
        if "adx" in row.index:
            with np.errstate(all="ignore"):
                adx = float(row.get("adx", 0.0) or 0.0)
        else:
            for c in row.index:
                if str(c).endswith("_adx"):
                    with np.errstate(all="ignore"):
                        adx = float(row.get(c, 0.0) or 0.0)
                    break
        trend_thr = float(getattr(self.settings.risk, "regime_adx_trend", 25.0) or 25.0)
        range_thr = float(getattr(self.settings.risk, "regime_adx_range", 20.0) or 20.0)
        if adx >= trend_thr:
            return "trend"
        if adx <= range_thr:
            return "range"
        v_ref = float(getattr(self.settings.risk, "volatility_target", 0.0015) or 0.0015)
        if v_ref > 0 and vol > (v_ref * 1.5):
            return "high_vol"
        return "neutral"

    def _select_models_for_regime(self, X: pd.DataFrame, last_idx: int) -> tuple[dict[str, Any], str]:
        if not bool(getattr(self.settings.models, "regime_router_enabled", True)):
            return self.models, "neutral"
        if not self.models:
            return {}, "neutral"
        bucket = self._infer_regime_bucket(X, last_idx)
        if bucket == "trend":
            candidates = set(getattr(self.settings.models, "regime_trend_models", []) or [])
        elif bucket == "range":
            candidates = set(getattr(self.settings.models, "regime_range_models", []) or [])
        else:
            candidates = set(getattr(self.settings.models, "regime_neutral_models", []) or [])
        chosen = {k: v for k, v in self.models.items() if (not candidates or k in candidates)}
        min_models = int(getattr(self.settings.models, "regime_router_min_models", 2) or 2)
        if len(chosen) < max(1, min_models):
            return self.models, bucket
        return chosen, bucket

    def _predict_proba(
        self,
        X: pd.DataFrame,
        models_subset: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        model_probas = self._collect_model_probas(X, models_subset=models_subset)
        use_all_models = models_subset is None or len(models_subset) == len(self.models)
        if self._onnx is not None and use_all_models:
            try:
                return self._onnx.predict_with_meta_blender(X), model_probas
            except Exception:
                pass
        if self.meta_blender is not None:
            try:
                feats = self._meta_features(X, models_subset=models_subset, proba_cache=model_probas)
                return pad_probs(self.meta_blender.predict_proba(feats)), model_probas
            except Exception:
                pass
        return self._ensemble_proba(X, models_subset=models_subset, proba_cache=model_probas), model_probas

    @staticmethod
    def _latest_market_volatility(X: pd.DataFrame, last_idx: int) -> float:
        if X is None or len(X) == 0:
            return 0.0
        row = X.iloc[last_idx]
        close = 0.0
        for c in ("close", "Close", "price"):
            if c in row.index:
                try:
                    close = float(row[c])
                except Exception:
                    close = 0.0
                if close > 0:
                    break
        atr_val = 0.0
        for c in ("atr14", "atr", "ATR", "M1_atr14", "M5_atr14"):
            if c in row.index:
                try:
                    atr_val = float(row[c])
                except Exception:
                    atr_val = 0.0
                if np.isfinite(atr_val) and atr_val > 0:
                    break
        if close > 0 and atr_val > 0:
            return float(max(0.0, atr_val / close))
        return 0.0

    @staticmethod
    def _ensemble_disagreement(last_model_probs: list[np.ndarray]) -> float:
        if not last_model_probs:
            return 0.0
        mat = np.asarray(last_model_probs, dtype=float)
        if mat.ndim != 2 or mat.shape[0] <= 1:
            return 0.0
        preds = np.argmax(mat, axis=1)
        vals, cnts = np.unique(preds, return_counts=True)
        if len(vals) == 0:
            return 0.0
        majority = int(np.max(cnts))
        disagree = 1.0 - (float(majority) / float(max(1, len(preds))))
        return float(max(0.0, min(1.0, disagree)))

    def generate_ensemble_signals(self, dataset: PreparedDataset) -> SignalResult:
        X = dataset.X
        if X is None or len(X) == 0:
            return SignalResult(
                signal=0,
                confidence=0.0,
                model_votes={},
                regime="Normal",
                meta_features={},
                probs=np.zeros(3, dtype=float),
                signals=pd.Series(dtype=int),
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_raw = X

        last_idx = -1 if len(X) > 0 else None
        active_models, regime_bucket = self._select_models_for_regime(X_raw, last_idx or -1)
        X = self._align_selected_features(X_raw, regime_bucket=regime_bucket)

        probas, model_proba_cache = self._predict_proba(X, models_subset=active_models)
        signals = probs_to_signals(probas)
        last_idx = -1 if len(signals) > 0 else None
        if last_idx is None:
            last_signal = 0
            confidence = 0.0
            last_probs = np.zeros(3, dtype=float)
            market_volatility = 0.0
            disagreement = 0.0
            uncertainty_value = 0.0
            conformal_set_size = 1
            conformal_abstained = False
        else:
            last_signal = int(signals[last_idx])
            last_probs = probas[last_idx]
            confidence = float(np.max(last_probs)) if last_probs is not None else 0.0
            market_volatility = self._latest_market_volatility(X_raw, last_idx)
            disagreement = 0.0
            uncertainty_value = max(0.0, min(1.0, 1.0 - confidence))
            conformal_set_size = 1
            conformal_abstained = False

            if self.conformal_gate is not None and bool(getattr(self.settings.risk, "conformal_enabled", True)):
                min_size = int(getattr(self.settings.risk, "conformal_abstain_min_set_size", 3) or 3)
                abstain, set_size = self.conformal_gate.should_abstain(last_probs, min_set_size=min_size)
                conformal_set_size = int(set_size)
                conformal_abstained = bool(abstain)
                if abstain:
                    last_signal = 0

        votes: dict[str, float] = {}
        model_last_probs: list[np.ndarray] = []
        for name, proba in model_proba_cache.items():
            if last_idx is None:
                continue
            votes[name] = float(proba[last_idx, 1] - proba[last_idx, 2])
            model_last_probs.append(np.asarray(proba[last_idx], dtype=float))

        if last_idx is not None and model_last_probs:
            disagreement = self._ensemble_disagreement(model_last_probs)
            uncertainty_value = float(max(0.0, min(1.0, max(1.0 - confidence, disagreement))))

        uncertainty_series = None
        if last_idx is not None:
            try:
                uncertainty_series = pd.Series([float(uncertainty_value)], index=[X.index[last_idx]])
            except Exception:
                uncertainty_series = pd.Series([float(uncertainty_value)])

        return SignalResult(
            signal=last_signal,
            confidence=confidence,
            model_votes=votes,
            regime=str(regime_bucket or "neutral"),
            meta_features={
                "ensemble_disagreement": float(disagreement),
                "market_volatility": float(market_volatility),
                "uncertainty": float(uncertainty_value),
                "regime_bucket": str(regime_bucket or "neutral"),
                "conformal_set_size": int(conformal_set_size),
                "conformal_abstained": bool(conformal_abstained),
            },
            probs=np.asarray(last_probs, dtype=float),
            signals=pd.Series(signals, index=X.index),
            uncertainty=uncertainty_series,
        )


__all__ = ["SignalEngine"]
