from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any
import multiprocessing

import numpy as np
import pandas as pd

from ..core.config import Settings
from ..core.system import AutoTuner, HardwareProbe, resolve_cpu_budget, thread_limits
from .model_factory import ModelFactory
from .optimization import HyperparameterOptimizer

logger = logging.getLogger(__name__)


def _load_memmap_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    meta_path = dataset_dir / "meta.json"
    cols_path = dataset_dir / "columns.json"
    x_path = dataset_dir / "X.npy"
    y_path = dataset_dir / "y.npy"
    idx_path = dataset_dir / "index.npy"

    if not cols_path.exists() or not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Dataset cache is incomplete: {dataset_dir}")

    meta: dict[str, Any] = {}
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    cols = json.loads(cols_path.read_text(encoding="utf-8"))
    # Copy-on-write keeps the backing file read-only but marks the array writeable, which
    # avoids unnecessary copies in some downstream pipelines that require writeable inputs.
    X_mm = np.load(x_path, mmap_mode="c")
    y_mm = np.load(y_path, mmap_mode="c")

    index = None
    if idx_path.exists():
        try:
            idx_ns = np.load(idx_path, mmap_mode="r")
            if str(meta.get("index_kind", "")).startswith("datetime"):
                index = pd.to_datetime(idx_ns.astype(np.int64), utc=True)
            else:
                index = pd.Index(idx_ns)
        except Exception:
            index = None

    X = pd.DataFrame(X_mm, columns=list(cols), index=index)
    y = pd.Series(y_mm, index=X.index, dtype=np.int8)
    return X, y


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


def _predict_kwargs(model: Any, metadata: pd.DataFrame | None) -> dict[str, Any]:
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
    sample_x: pd.DataFrame,
    sample_meta: pd.DataFrame | None,
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
    sample_x: pd.DataFrame,
    sample_meta: pd.DataFrame | None,
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

        X, y = _load_memmap_dataset(Path(dataset_dir))
        metadata = None
        if metadata_path:
            try:
                meta_path = Path(metadata_path)
                if meta_path.exists():
                    metadata = pd.read_pickle(meta_path)
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

        sample = X.iloc[: min(256, len(X))]
        sample_meta = None
        if metadata is not None and isinstance(metadata, pd.DataFrame):
            with contextlib.suppress(Exception):
                sample_meta = metadata.reindex(sample.index)

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    try:
        profile = HardwareProbe().detect()
        AutoTuner(settings, profile).apply()
    except Exception:
        # Worker can still proceed with safe defaults if probing fails.
        pass
    X, y = _load_memmap_dataset(Path(args.dataset_dir))

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
            if meta_path.exists():
                metadata = pd.read_pickle(meta_path)
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

            sample = X.iloc[: min(256, len(X))]
            sample_meta = None
            if metadata is not None and isinstance(metadata, pd.DataFrame):
                with contextlib.suppress(Exception):
                    sample_meta = metadata.reindex(sample.index)
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
