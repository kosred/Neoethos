from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forex_bot.training import parallel_worker as pw


def test_pandas_free_strict_enabled_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOREX_BOT_PANDAS_FREE_STRICT", raising=False)
    assert pw._pandas_free_strict_enabled() is True


def test_load_training_data_pandas_free_uses_numpy_when_rust_available(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_x = np.zeros((8, 3), dtype=np.float32)
    sentinel_y = np.zeros(8, dtype=np.int8)
    monkeypatch.setattr(pw, "_load_memmap_arrays", lambda _p: (sentinel_x, sentinel_y))
    monkeypatch.setattr(pw, "_load_memmap_dataset", lambda _p: (_ for _ in ()).throw(AssertionError("pandas path used")))
    monkeypatch.setattr(pw, "_rust_tree_model_available", lambda _name: True)

    x, y, uses_pandas = pw._load_training_data(
        Path("."),
        model_name="xgboost",
        pandas_free=True,
    )

    assert uses_pandas is False
    assert x is sentinel_x
    assert y is sentinel_y


def test_load_training_data_pandas_free_strict_raises_without_rust_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOREX_BOT_PANDAS_FREE_STRICT", raising=False)
    monkeypatch.setattr(pw, "_rust_tree_model_available", lambda _name: False)
    monkeypatch.setattr(pw, "_load_memmap_arrays", lambda _p: (_ for _ in ()).throw(AssertionError("numpy path should not run")))
    monkeypatch.setattr(pw, "_load_memmap_dataset", lambda _p: (_ for _ in ()).throw(AssertionError("pandas fallback should not run")))

    with pytest.raises(RuntimeError):
        pw._load_training_data(
            Path("."),
            model_name="xgboost",
            pandas_free=True,
        )


def test_load_training_data_non_strict_still_requires_rust_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE_STRICT", "0")
    monkeypatch.setattr(pw, "_rust_tree_model_available", lambda _name: False)
    monkeypatch.setattr(pw, "_load_memmap_arrays", lambda _p: (_ for _ in ()).throw(AssertionError("numpy path should not run")))
    monkeypatch.setattr(
        pw,
        "_load_memmap_dataset",
        lambda _p: (_ for _ in ()).throw(AssertionError("pandas fallback should not run")),
    )

    with pytest.raises(RuntimeError):
        pw._load_training_data(
            Path("."),
            model_name="xgboost",
            pandas_free=True,
        )


def test_load_training_data_pandas_free_linear_model_uses_numpy_without_rust_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_x = np.zeros((5, 2), dtype=np.float32)
    sentinel_y = np.zeros(5, dtype=np.int8)
    monkeypatch.setattr(pw, "_rust_tree_model_available", lambda _name: False)
    monkeypatch.setattr(pw, "_load_memmap_arrays", lambda _p: (sentinel_x, sentinel_y))
    x, y, uses_pandas = pw._load_training_data(
        Path("."),
        model_name="elasticnet",
        pandas_free=True,
    )
    assert uses_pandas is False
    assert x is sentinel_x
    assert y is sentinel_y


def test_slice_rows_preserves_frame_like_metadata() -> None:
    class _Frame:
        def __init__(self):
            self.columns = ["close", "high", "low"]
            self.index = np.array([10, 11, 12], dtype=np.int64)
            self.attrs = {"symbol": "EURUSD"}
            self._data = {
                "close": np.array([1.0, 2.0, 3.0], dtype=np.float64),
                "high": np.array([1.1, 2.1, 3.1], dtype=np.float64),
                "low": np.array([0.9, 1.9, 2.9], dtype=np.float64),
            }

        def __getitem__(self, key):
            return self._data[str(key)]

    out = pw._slice_rows(_Frame(), 2)
    assert hasattr(out, "columns")
    assert list(getattr(out, "columns", [])) == ["close", "high", "low"]
    np.testing.assert_allclose(np.asarray(out["close"]), np.array([1.0, 2.0], dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(getattr(out, "index")), np.array([10, 11], dtype=np.int64))


def test_train_single_model_process_loads_metadata_even_without_pandas_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"metadata": None}
    sentinel_meta = {"close": np.array([1.0, 2.0, 3.0], dtype=np.float64)}

    class _DummyModel:
        def fit(self, _X, _y, **kwargs):
            captured["metadata"] = kwargs.get("metadata")

        def predict_proba(self, X, **_kwargs):
            n = int(len(X))
            return np.column_stack(
                [
                    np.full(n, 0.5, dtype=np.float32),
                    np.full(n, 0.5, dtype=np.float32),
                ]
            )

        def save(self, _path):
            return None

    class _DummyFactory:
        def __init__(self, *_args, **_kwargs):
            pass

        def create_model(self, *_args, **_kwargs):
            return _DummyModel()

    class _DummyOptimizer:
        def __init__(self, *_args, **_kwargs):
            pass

        def load_params(self):
            return {}

    class _DummyProbe:
        def detect(self):
            return {}

    class _DummyTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def apply(self):
            return None

    monkeypatch.setattr(
        pw,
        "_load_training_data",
        lambda *_args, **_kwargs: (
            np.zeros((8, 3), dtype=np.float32),
            np.zeros(8, dtype=np.int8),
            False,
        ),
    )
    monkeypatch.setattr(pw, "_load_metadata_artifact", lambda _path: sentinel_meta)
    monkeypatch.setattr(pw, "HyperparameterOptimizer", _DummyOptimizer)
    monkeypatch.setattr(pw, "ModelFactory", _DummyFactory)
    monkeypatch.setattr(pw, "Settings", lambda: object())
    monkeypatch.setattr(pw, "HardwareProbe", lambda: _DummyProbe())
    monkeypatch.setattr(pw, "AutoTuner", lambda *_args, **_kwargs: _DummyTuner())
    monkeypatch.setattr(pw, "_strict_model_check_enabled", lambda: False)

    result = pw._train_single_model_process(
        (
            str(tmp_path),
            "xgboost",
            str(tmp_path / "out"),
            1,
            1,
            1,
            str(tmp_path / "metadata.pkl"),
        )
    )

    assert result[2] is True
    assert captured["metadata"] is sentinel_meta
