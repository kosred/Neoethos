from __future__ import annotations

import numpy as np
import pandas as pd

from forex_bot.models import trees_rust as tr


class _FailingRustModel:
    def __init__(self, idx: int = 1, params: dict | None = None) -> None:
        self.idx = idx
        self.params = params or {}

    def fit(self, *_args, **_kwargs) -> None:
        raise RuntimeError("rust fit failed")

    def predict_proba(self, *_args, **_kwargs):
        raise RuntimeError("rust predict failed")

    def save(self, *_args, **_kwargs) -> None:
        raise RuntimeError("rust save failed")

    def load(self, *_args, **_kwargs) -> None:
        raise RuntimeError("rust load failed")


class _PythonFallbackModel:
    def __init__(self, params: dict | None = None, idx: int = 1) -> None:
        self.params = params or {}
        self.idx = idx
        self.fit_called = False

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.fit_called = True

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([0.2, 0.6, 0.2], dtype=np.float32), (len(x), 1))

    def save(self, _path: str) -> None:
        return None

    def load(self, _path: str) -> None:
        return None


def test_rust_tree_switches_to_python_fallback_on_fit_failure(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_TREE_RUST_FALLBACK", "1")
    monkeypatch.setattr(tr, "_fb", object(), raising=False)

    class _TestExpert(tr._RustTreeBase):
        _model_cls = _FailingRustModel
        _python_fallback_class_name = "LightGBMExpert"

        def _ensure_python_fallback_model(self):  # type: ignore[override]
            if getattr(self, "_fallback_model", None) is None:
                self._fallback_model = _PythonFallbackModel(params=self._params, idx=self._idx)
            return self._fallback_model

    x = pd.DataFrame({"f1": [0.1, 0.2, 0.3], "f2": [1.0, 1.1, 1.2]})
    y = pd.Series([1, 0, -1])

    model = _TestExpert(params={"n_estimators": 10}, idx=2)
    model.fit(x, y)
    out = model.predict_proba(x)

    assert out.shape == (3, 3)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(3, dtype=np.float32), atol=1e-6)
