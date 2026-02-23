import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    prange = range

from .base import ExpertModel

logger = logging.getLogger(__name__)

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, parallel=True)
    def _rolling_std_numba(data, window):
        n = len(data)
        out = np.zeros(n, dtype=np.float32)
        for i in prange(window, n):
            chunk = data[i-window+1 : i+1]
            out[i] = np.std(chunk)
        return out
else:
    def _rolling_std_numba(data, window):
        arr = np.asarray(data, dtype=np.float32)
        n = arr.size
        out = np.zeros(n, dtype=np.float32)
        if n == 0:
            return out
        w = max(1, int(window))
        if n < w:
            return out
        arr64 = arr.astype(np.float64, copy=False)
        c1 = np.cumsum(arr64)
        c2 = np.cumsum(arr64 * arr64)
        sum_w = c1[w - 1 :] - np.concatenate(([0.0], c1[:-w]))
        sq_w = c2[w - 1 :] - np.concatenate(([0.0], c2[:-w]))
        mean_w = sum_w / float(w)
        var_w = (sq_w / float(w)) - (mean_w * mean_w)
        out[w - 1 :] = np.sqrt(np.maximum(var_w, 0.0)).astype(np.float32, copy=False)
        return out

from sklearn.mixture import GaussianMixture

class MarketRegimeClassifier(ExpertModel):
    """
    HPC FIX: Fully Unsupervised Gaussian Regime Discovery.
    Discovers the latent 'Hidden States' of the market without human labels.
    """

    def __init__(self, n_regimes: int = 8, **kwargs):
        # Increased to 8 regimes to capture more subtle market anomalies
        self.n_regimes = n_regimes
        self.model = GaussianMixture(n_components=n_regimes, covariance_type='full', random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        # Backward compatibility for older callers/tests.
        self.regime_map: dict[int, str] = {}

    def fit(self, df: pd.DataFrame, y=None, **kwargs) -> None:
        features = self._extract_features(df)
        if features.size == 0:
            return

        X = self.scaler.fit_transform(features)
        self.model.fit(X)
        try:
            labels = self.model.predict(X)
            uniq = sorted({int(v) for v in labels.tolist()})
            self.regime_map = {idx: f"Regime_{idx}" for idx in uniq}
        except Exception:
            self.regime_map = {idx: f"Regime_{idx}" for idx in range(self.n_regimes)}
        self.is_fitted = True
        logger.info(f"Unsupervised GMM fitted: {self.n_regimes} latent regimes discovered.")

    def predict_regime_distribution(self, df: pd.DataFrame) -> np.ndarray:
        """
        HPC FIX: Multi-Regime Posterior Distribution.
        Returns a vector of probabilities [p0, p1, ..., p7].
        """
        if not self.is_fitted:
            return np.zeros(self.n_regimes)
        try:
            # Process entire window to get a stable distribution
            feat = self._extract_features(df.tail(50))
            if feat.size == 0:
                return np.zeros(self.n_regimes)
            
            X = self.scaler.transform(feat)
            # Use posteriors (soft assignment)
            probs = self.model.predict_proba(X)
            # Return the latest posterior
            return probs[-1]
        except Exception:
            return np.zeros(self.n_regimes)

    def predict(self, df: pd.DataFrame) -> str:
        """Fallback for legacy components: returns the 'primary' regime ID."""
        dist = self.predict_regime_distribution(df)
        regime_idx = int(np.argmax(dist))
        return self.regime_map.get(regime_idx, str(regime_idx))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Mock implementation for ExpertModel compatibility.
        Does not actually predict buy/sell signals, just used for regime detection.
        Returns neutral probabilities.
        """
        n = len(X)
        probs = np.zeros((n, 3), dtype=np.float32)
        probs[:, 0] = 1.0  # All Neutral
        return probs

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        HPC Optimized: feature extraction without intermediate DataFrames.
        """
        try:
            closes = df["close"].to_numpy(dtype=np.float32, copy=False)
            n = int(closes.shape[0])
            if n < 3:
                return np.empty((0, 3), dtype=np.float32)

            # Log Returns (vectorized, guarded against zero/invalid prices)
            returns = np.zeros(n, dtype=np.float32)
            prev = closes[:-1]
            curr = closes[1:]
            ratio = np.ones_like(curr, dtype=np.float32)
            np.divide(curr, prev, out=ratio, where=(prev != 0.0))
            returns[1:] = np.log(np.clip(ratio, 1e-12, None)).astype(np.float32, copy=False)

            # Rolling volatility
            volatility = _rolling_std_numba(returns, 20)

            # Use precomputed ADX if present, else volatility proxy.
            if "adx" in df.columns:
                adx = df["adx"].to_numpy(dtype=np.float32, copy=False)
                if adx.shape[0] != n:
                    adx = np.resize(adx, n).astype(np.float32, copy=False)
            else:
                adx = volatility * 100.0

            # Shift-by-1 and drop first row (to avoid look-ahead leakage).
            features = np.empty((n - 1, 3), dtype=np.float32)
            features[:, 0] = returns[:-1]
            features[:, 1] = volatility[:-1]
            features[:, 2] = adx[:-1]

            # Guard against non-finite values from broken source data.
            np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return features
        except Exception:
            return np.empty((0, 3), dtype=np.float32)

    def save(self, path: str) -> None:
        if self.is_fitted:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, p / "regime_classifier.joblib")

    @staticmethod
    def load(path: str) -> "MarketRegimeClassifier | None":
        p = Path(path) / "regime_classifier.joblib"
        if p.exists():
            return joblib.load(p)
        return None

    # Alias for registry compatibility
    load_model = load


# Alias for registry
ClusterExpert = MarketRegimeClassifier
