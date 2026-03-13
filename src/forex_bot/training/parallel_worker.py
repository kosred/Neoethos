from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
from functools import lru_cache
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any
import multiprocessing

import numpy as np

from ..core.config import Settings
from ..core.system import AutoTuner, HardwareProbe, resolve_cpu_budget, thread_limits
from .model_factory import ModelFactory
from .optimization import HyperparameterOptimizer

logger = logging.getLogger(__name__)

_PANDAS_FREE_MODEL_ALLOWLIST = {
    "lightgbm",
    "xgboost",
    "xgboost_rf",
    "xgboost_dart",
    "catboost",
    "catboost_alt",
    # Numpy-native linear experts.
    "elasticnet",
    "bayes_logit",
    "online_pa",
    "online_hoeffding",
    "vw",
}

_RUST_TREE_BINDING_CLASSES = {
    "lightgbm": "LightGBMModel",
    "xgboost": "XGBoostModel",
    "xgboost_rf": "XGBoostRFModel",
    "xgboost_dart": "XGBoostDARTModel",
    "catboost": "CatBoostModel",
    "catboost_alt": "CatBoostAltModel",
}

_PANDAS_FREE_RUST_TREE_REQUIRED = set(_RUST_TREE_BINDING_CLASSES.keys())


def _pandas_free_enabled() -> bool:
    raw = str(os.environ.get("FOREX_BOT_PANDAS_FREE", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _pandas_free_strict_enabled() -> bool:
    raw = str(os.environ.get("FOREX_BOT_PANDAS_FREE_STRICT", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _joblib_module():
    import joblib  # type: ignore

    return joblib


def _load_metadata_artifact(path: Path) -> Any | None:
    if not path.exists():
        return None
    # Prefer joblib because this project writes most artifacts with it.
    with contextlib.suppress(Exception):
        return _joblib_module().load(path)
    # Fallback to stdlib pickle for plain pickle payloads.
    with contextlib.suppress(Exception):
        with path.open("rb") as fh:
            return pickle.load(fh)
    return None


def _is_pandas_dataframe(value: Any) -> bool:
    return bool(
        hasattr(value, "columns")
        and hasattr(value, "index")
        and callable(getattr(value, "to_numpy", None))
    )


class _NumpyFrame:
    """Minimal frame container used when slicing generic frame-like metadata."""

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


def _slice_frame_head(values: Any, n_rows: int) -> Any:
    n = max(0, int(n_rows))
    cols = list(getattr(values, "columns", []) or [])
    if not cols:
        return np.asarray(values)[:n]
    out_data: dict[str, np.ndarray] = {}
    for col in cols:
        with contextlib.suppress(Exception):
            raw = values[col]  # type: ignore[index]
            vec = raw.to_numpy(copy=False) if hasattr(raw, "to_numpy") else np.asarray(raw)
            out_data[str(col)] = np.asarray(vec).reshape(-1)[:n]
    idx_obj = getattr(values, "index", None)
    if idx_obj is None:
        out_idx = np.arange(n, dtype=np.int64)
    else:
        idx_arr = np.asarray(idx_obj).reshape(-1)
        out_idx = idx_arr[:n] if idx_arr.size >= n else np.arange(n, dtype=np.int64)
    attrs = getattr(values, "attrs", None)
    return _NumpyFrame(out_data, out_idx, attrs=(dict(attrs) if isinstance(attrs, dict) else None))


def _slice_rows(values: Any, n_rows: int) -> Any:
    n = max(0, int(n_rows))
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values[:n]
    if _is_pandas_dataframe(values):
        with contextlib.suppress(Exception):
            return values.take(np.arange(n, dtype=np.int64))
        with contextlib.suppress(Exception):
            base_idx = np.asarray(getattr(values, "index")).reshape(-1)
            return values.loc[base_idx[:n]]
    if hasattr(values, "columns") and hasattr(values, "__getitem__"):
        with contextlib.suppress(Exception):
            return _slice_frame_head(values, n)
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr[:n]


def _rust_tree_model_available(model_name: str) -> bool:
    cls_name = _RUST_TREE_BINDING_CLASSES.get(str(model_name))
    if not cls_name:
        return False
    try:
        import forex_bindings  # type: ignore

        return hasattr(forex_bindings, cls_name)
    except Exception:
        return False


def _load_memmap_arrays(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cols_path = dataset_dir / "columns.json"
    x_path = dataset_dir / "X.npy"
    y_path = dataset_dir / "y.npy"
    if not cols_path.exists() or not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Dataset cache is incomplete: {dataset_dir}")
    X_mm = np.load(x_path, mmap_mode="c")
    y_mm = np.load(y_path, mmap_mode="c")
    return np.asarray(X_mm), np.asarray(y_mm)


def _load_memmap_dataset(dataset_dir: Path) -> tuple[Any, Any]:
    cols_path = dataset_dir / "columns.json"
    x_path = dataset_dir / "X.npy"
    y_path = dataset_dir / "y.npy"
    if not cols_path.exists() or not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Dataset cache is incomplete: {dataset_dir}")

    X_mm, y_mm = _load_memmap_arrays(dataset_dir)
    # Return numpy arrays only (frame-native mode).
    return np.asarray(X_mm), np.asarray(y_mm)


def _load_training_data(
    dataset_dir: Path,
    *,
    model_name: str | None,
    pandas_free: bool,
) -> tuple[Any, Any, bool]:
    """
    Returns (X, y, uses_pandas_dataset).
    """
    # Always prefer frame-native path. pandas_free flag retained for API compat.
    os.environ.setdefault("FOREX_BOT_TREE_BACKEND", "rust_strict")
    if model_name is None:
        X, y = _load_memmap_arrays(dataset_dir)
        return X, y, False

    name = str(model_name)
    if name in _PANDAS_FREE_MODEL_ALLOWLIST:
        if name in _PANDAS_FREE_RUST_TREE_REQUIRED:
            if _rust_tree_model_available(name):
                X, y = _load_memmap_arrays(dataset_dir)
                return X, y, False
            raise RuntimeError(
                f"Rust tree binding is required for model '{name}', but binding is missing."
            )
        # Non-tree allowlisted models consume numpy arrays directly.
        X, y = _load_memmap_arrays(dataset_dir)
        return X, y, False
    raise RuntimeError(f"Rust tree workers do not support non-rust-tree model '{name}'.")


def _pad_probs(p: Any) -> np.ndarray:
    if p is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(p)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.shape[1] == 2:
        out = np.zeros((len(arr), 3), dtype=np.float32)
        out[:, 0] = arr[:, 0]
        out[:, 1] = arr[:, 1]
        return out
    return arr[:, :3].astype(np.float32, copy=False)


def _strict_model_check_enabled() -> bool:
    raw = str(os.environ.get("FOREX_BOT_STRICT_MODEL_CHECK", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _roundtrip_check_enabled() -> bool:
    raw = str(os.environ.get("FOREX_BOT_MODEL_ROUNDTRIP", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fit_accepts_metadata(model: Any) -> bool:
    try:
        sig = inspect.signature(model.fit)
        if "metadata" in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        return False


def _predict_kwargs(model: Any, metadata: Any | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if metadata is None:
        return kwargs
    try:
        sig = inspect.signature(model.predict_proba)
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if "metadata" in sig.parameters or has_kwargs:
            kwargs["metadata"] = metadata
    except Exception:
        pass
    return kwargs


def _smoke_predict(
    model: Any,
    sample_x: Any,
    sample_meta: Any | None,
) -> tuple[bool, str]:
    if sample_x is None or len(sample_x) == 0:
        return False, "empty sample"
    try:
        probs = _pad_probs(model.predict_proba(sample_x, **_predict_kwargs(model, sample_meta)))
    except Exception as exc:
        return False, f"inference exception: {exc}"
    if probs.ndim != 2:
        return False, f"invalid output rank: {probs.ndim}"
    if probs.shape[0] != len(sample_x):
        return False, f"row mismatch: got {probs.shape[0]} expected {len(sample_x)}"
    if probs.shape[1] < 2:
        return False, f"invalid output width: {probs.shape[1]}"
    if not np.all(np.isfinite(probs)):
        return False, "non-finite probabilities"
    return True, "ok"


def _roundtrip_smoke_check(
    *,
    settings: Settings,
    model_name: str,
    best_params: dict[str, Any],
    idx: int,
    model: Any,
    sample_x: Any,
    sample_meta: Any | None,
    out_dir: Path,
) -> tuple[bool, str]:
    if not _roundtrip_check_enabled():
        return True, "roundtrip disabled"
    stamp = f"{int(time.time() * 1_000_000)}_{os.getpid()}"
    tmp_dir = out_dir / "_healthcheck" / f"{model_name}_{idx}_{stamp}"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(tmp_dir))
        probe_factory = ModelFactory(settings, tmp_dir)
        probe = probe_factory.create_model(model_name, best_params, idx)
        if not hasattr(probe, "load"):
            return False, "reloaded model has no load()"
        probe.load(str(tmp_dir))
        ok, reason = _smoke_predict(probe, sample_x, sample_meta)
        if not ok:
            return False, f"roundtrip inference failed: {reason}"
        return True, "ok"
    except Exception as exc:
        return False, f"roundtrip exception: {exc}"
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _apply_thread_env(threads: int) -> None:
    if threads <= 0:
        return
    threads = max(1, int(threads))
    os.environ["FOREX_BOT_CPU_BUDGET"] = str(threads)
    os.environ["FOREX_BOT_CPU_THREADS"] = str(threads)
    os.environ["FOREX_BOT_RUST_THREADS"] = str(threads)
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(threads)


def _train_single_model_process(args: tuple[str, str, str, int, int, int, str | None]) -> tuple[str, float, bool]:
    """Train a single model in a separate process to avoid GIL contention."""
    dataset_dir, model_name, out_dir, idx, threads_per_model, cpu_threads, metadata_path = args
    t0 = time.perf_counter()
    try:
        thread_budget = max(1, int(threads_per_model or 1))
        if int(cpu_threads or 0) > 0:
            thread_budget = min(thread_budget, int(cpu_threads))
        _apply_thread_env(thread_budget)
        settings = Settings()
        try:
            profile = HardwareProbe().detect()
            AutoTuner(settings, profile).apply()
        except Exception:
            pass

        pandas_free = _pandas_free_enabled()
        X, y, _uses_pandas_dataset = _load_training_data(
            Path(dataset_dir),
            model_name=model_name,
            pandas_free=pandas_free,
        )
        metadata = None
        if metadata_path:
            try:
                meta_path = Path(metadata_path)
                metadata = _load_metadata_artifact(meta_path)
            except Exception:
                metadata = None

        optimizer = HyperparameterOptimizer(settings)
        best_params = optimizer.load_params()
        factory = ModelFactory(settings, Path(out_dir))

        model = factory.create_model(model_name, best_params, idx)
        with thread_limits(blas_threads=thread_budget):
            fit_kwargs = {}
            if metadata is not None and _fit_accepts_metadata(model):
                fit_kwargs["metadata"] = metadata
            model.fit(X, y, **fit_kwargs)

        sample_n = min(256, len(X))
        sample = _slice_rows(X, sample_n)
        sample_meta = _slice_rows(metadata, sample_n) if metadata is not None else None

        ok, reason = _smoke_predict(model, sample, sample_meta)
        if not ok:
            raise RuntimeError(f"inference smoke check failed: {reason}")
        if _strict_model_check_enabled():
            rt_ok, rt_reason = _roundtrip_smoke_check(
                settings=settings,
                model_name=model_name,
                best_params=best_params,
                idx=idx,
                model=model,
                sample_x=sample,
                sample_meta=sample_meta,
                out_dir=Path(out_dir),
            )
            if not rt_ok:
                raise RuntimeError(f"roundtrip check failed: {rt_reason}")
        model.save(str(out_dir))
        duration = time.perf_counter() - t0
        return (model_name, duration, True)
    except Exception:
        duration = time.perf_counter() - t0
        return (model_name, duration, False)


def run_worker(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forex AI parallel training worker (internal)")
    parser.add_argument("--dataset-dir", required=True, help="Path to cached memmap dataset directory")
    parser.add_argument("--models", required=True, help="Comma-separated model names to train")
    parser.add_argument("--out-dir", required=True, help="Directory to save trained model artifacts")
    parser.add_argument("--cpu-threads", type=int, default=0, help="BLAS/OMP threads to use (0=auto)")
    parser.add_argument("--max-concurrent-models", type=int, default=0, help="Max models to train concurrently (0=auto)")
    parser.add_argument("--metadata-path", default="", help="Optional path to metadata pickle for model.fit")
    args = parser.parse_args(argv)

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    if not models:
        raise ValueError("No models provided for worker")

    pandas_free = _pandas_free_enabled()
    if pandas_free:
        os.environ.setdefault("FOREX_BOT_TREE_BACKEND", "rust_strict")
        kept: list[str] = []
        for m in models:
            if m not in _PANDAS_FREE_MODEL_ALLOWLIST:
                continue
            if m in _PANDAS_FREE_RUST_TREE_REQUIRED and not _rust_tree_model_available(m):
                continue
            kept.append(m)
        dropped = [m for m in models if m not in _PANDAS_FREE_MODEL_ALLOWLIST]
        dropped_missing = [
            m
            for m in models
            if m in _PANDAS_FREE_RUST_TREE_REQUIRED and not _rust_tree_model_available(m)
        ]
        if dropped:
            logger.info("[WORKER] Pandas-free mode: skipping unsupported model families: %s", dropped)
        if dropped_missing:
            logger.info("[WORKER] Pandas-free mode: skipping models without Rust bindings: %s", dropped_missing)
        models = kept
        if not models:
            raise ValueError("Pandas-free mode has no compatible models to train.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    try:
        profile = HardwareProbe().detect()
        AutoTuner(settings, profile).apply()
    except Exception:
        # Worker can still proceed with safe defaults if probing fails.
        pass
    X, y, _uses_pandas_dataset = _load_training_data(
        Path(args.dataset_dir),
        model_name=None,
        pandas_free=pandas_free,
    )

    optimizer = HyperparameterOptimizer(settings)
    best_params = optimizer.load_params()
    factory = ModelFactory(settings, out_dir)

    cpu_total = max(1, os.cpu_count() or 1)
    cpu_threads = int(args.cpu_threads or 0)
    if cpu_threads <= 0:
        env_threads = os.environ.get("FOREX_BOT_CPU_THREADS") or os.environ.get("FOREX_BOT_CPU_BUDGET")
        try:
            cpu_threads = int(env_threads) if env_threads else 0
        except Exception:
            cpu_threads = 0
        if cpu_threads <= 0:
            cpu_threads = resolve_cpu_budget()
        cpu_threads = max(1, min(cpu_total, cpu_threads))

    _apply_thread_env(cpu_threads)

    durations: dict[str, float] = {}
    trained: list[str] = []

    # Optional metadata for OHLC-dependent models
    metadata = None
    if args.metadata_path:
        try:
            meta_path = Path(str(args.metadata_path))
            metadata = _load_metadata_artifact(meta_path)
        except Exception:
            metadata = None

    # Parallel model training within worker.
    # Default to process pools when running >1 model to avoid GIL contention.
    max_concurrent_models = int(args.max_concurrent_models or 0)
    if max_concurrent_models <= 0:
        # Auto: scale to available threads with a minimum threads-per-model target
        try:
            min_threads = int(os.environ.get("FOREX_BOT_CPU_MIN_THREADS_PER_MODEL", "1") or 1)
        except Exception:
            min_threads = 1
        min_threads = max(1, min_threads)
        max_concurrent_models = max(1, min(len(models), cpu_threads // min_threads))
    max_concurrent_models = max(1, min(len(models), max_concurrent_models))
    threads_per_model = max(1, cpu_threads // max(1, max_concurrent_models))

    logger.info(f"[WORKER] Training {len(models)} models with {max_concurrent_models} concurrent, "
                f"{threads_per_model} threads each (CPU budget: {cpu_threads})")

    def train_single_model(idx: int, name: str) -> tuple[str, float, bool]:
        """Train a single model and return (name, duration, success)"""
        t0 = time.perf_counter()
        try:
            model = factory.create_model(name, best_params, idx)
            with thread_limits(blas_threads=threads_per_model):
                fit_kwargs = {}
                if metadata is not None and _fit_accepts_metadata(model):
                    fit_kwargs["metadata"] = metadata
                model.fit(X, y, **fit_kwargs)

            sample_n = min(256, len(X))
            sample = _slice_rows(X, sample_n)
            sample_meta = _slice_rows(metadata, sample_n) if metadata is not None else None
            ok, reason = _smoke_predict(model, sample, sample_meta)
            if not ok:
                raise RuntimeError(f"inference smoke check failed: {reason}")
            if _strict_model_check_enabled():
                rt_ok, rt_reason = _roundtrip_smoke_check(
                    settings=settings,
                    model_name=name,
                    best_params=best_params,
                    idx=idx,
                    model=model,
                    sample_x=sample,
                    sample_meta=sample_meta,
                    out_dir=out_dir,
                )
                if not rt_ok:
                    raise RuntimeError(f"roundtrip check failed: {rt_reason}")
            model.save(str(out_dir))
            duration = time.perf_counter() - t0
            logger.info(f"[WORKER] Successfully trained {name} in {duration:.1f}s")
            return (name, duration, True)
        except Exception as e:
            logger.error(f"[WORKER] Failed {name}: {e}", exc_info=True)
            return (name, 0.0, False)

    worker_mode_raw = os.environ.get("FOREX_BOT_PARALLEL_WORKER_MODE")
    if worker_mode_raw:
        worker_mode = str(worker_mode_raw).strip().lower()
        use_process_pool = worker_mode in {"process", "mp", "multiprocess"}
    else:
        use_process_pool = max_concurrent_models > 1

    if use_process_pool:
        from concurrent.futures import ProcessPoolExecutor, as_completed
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[tuple[str, float, bool]] = []

    if max_concurrent_models > 1:
        # Parallel training
        if use_process_pool:
            spawn_ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_concurrent_models, mp_context=spawn_ctx) as executor:
                future_to_model = {
                    executor.submit(
                        _train_single_model_process,
                        (
                            str(args.dataset_dir),
                            name,
                            str(out_dir),
                            idx,
                            threads_per_model,
                            cpu_threads,
                            str(args.metadata_path) if args.metadata_path else "",
                        ),
                    ): name
                    for idx, name in enumerate(models, start=1)
                }
                for future in as_completed(future_to_model):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        model_name = future_to_model[future]
                        logger.error(f"[WORKER] Exception training {model_name}: {e}")
                        results.append((model_name, 0.0, False))
        else:
            with ThreadPoolExecutor(max_workers=max_concurrent_models) as executor:
                future_to_model = {
                    executor.submit(train_single_model, idx, name): name
                    for idx, name in enumerate(models, start=1)
                }
                for future in as_completed(future_to_model):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        model_name = future_to_model[future]
                        logger.error(f"[WORKER] Exception training {model_name}: {e}")
                        results.append((model_name, 0.0, False))
    else:
        # Sequential training (fallback for low-resource systems)
        logger.info("[WORKER] Sequential training mode (limited CPU resources)")
        for idx, name in enumerate(models, start=1):
            results.append(train_single_model(idx, name))

    # Collect results
    for name, duration, success in results:
        if success:
            trained.append(name)
            durations[name] = duration

    # Write a small manifest for the coordinator.
    try:
        (out_dir / "worker_manifest.json").write_text(
            json.dumps(
                {
                    "trained": trained,
                    "durations_sec": durations,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    # Ensure artifacts exist for at least one model; otherwise signal failure.
    return 0 if trained else 2

