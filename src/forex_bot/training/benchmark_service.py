from __future__ import annotations

import logging
import multiprocessing
import time
from typing import Any

import numpy as np
import torch

from ..models.device import select_device
from ..models.registry import get_model_class

logger = logging.getLogger(__name__)


def _is_dataframe_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "index"))


def _is_frame_like(value: Any) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "index") and hasattr(value, "__getitem__"))


def _frame_columns(value: Any) -> list[str]:
    cols = getattr(value, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _slice_frame_rows(value: Any, rows: int) -> Any:
    n = max(0, int(rows))
    cols = _frame_columns(value)
    if not cols or not hasattr(value, "__getitem__"):
        return np.asarray(value)[:n].copy()
    data: dict[str, np.ndarray] = {}
    for col in cols:
        try:
            vec = np.asarray(value[col]).reshape(-1)  # type: ignore[index]
            data[str(col)] = vec[:n]
        except Exception:
            continue
    idx = getattr(value, "index", None)
    idx_arr = np.asarray(idx).reshape(-1) if idx is not None else np.arange(n, dtype=np.int64)
    attrs = getattr(value, "attrs", None)
    return _FrameSlice(data, idx_arr[:n], attrs=(dict(attrs) if isinstance(attrs, dict) else None))


class _FrameSlice:
    def __init__(self, data: dict[str, np.ndarray], index: np.ndarray, attrs: dict[str, Any] | None = None) -> None:
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


def _row_count(data: Any) -> int:
    if data is None:
        return 0
    with np.errstate(all="ignore"):
        try:
            if hasattr(data, "shape") and len(getattr(data, "shape", ())) >= 1:
                return int(data.shape[0])
        except Exception:
            pass
    try:
        return int(len(data))
    except Exception:
        return 0


def _feature_dim(data: Any, default: int = 0) -> int:
    if data is None:
        return int(default)
    with np.errstate(all="ignore"):
        try:
            shape = getattr(data, "shape", None)
            if shape is not None and len(shape) >= 2:
                return int(shape[1])
        except Exception:
            pass
    if _is_dataframe_like(data):
        try:
            return int(len(data.columns))
        except Exception:
            return int(default)
    if _is_frame_like(data):
        try:
            return int(len(_frame_columns(data)))
        except Exception:
            return int(default)
    arr = np.asarray(data)
    if arr.ndim >= 2:
        return int(arr.shape[1])
    if arr.ndim == 1:
        return 1
    return int(default)


def _slice_rows(data: Any, rows: int) -> Any:
    if data is None:
        return None
    n = max(0, int(rows))
    if _is_dataframe_like(data):
        try:
            idx = np.arange(n, dtype=np.int64)
            out = data.take(idx)
            return out.copy() if hasattr(out, "copy") else out
        except Exception:
            pass
        try:
            base_idx = np.asarray(getattr(data, "index")).reshape(-1)
            out = data.loc[base_idx[:n]]
            return out.copy() if hasattr(out, "copy") else out
        except Exception:
            pass
    if _is_frame_like(data):
        return _slice_frame_rows(data, n)
    arr = np.asarray(data)
    if arr.ndim == 0:
        return arr.reshape(1)[:n]
    return arr[:n].copy()


def _as_2d_float32(data: Any) -> np.ndarray:
    if data is None:
        return np.zeros((0, 0), dtype=np.float32)
    if _is_dataframe_like(data):
        try:
            arr = data.to_numpy(dtype=np.float32, copy=False)
        except Exception:
            arr = None
        if arr is None:
            cols = _frame_columns(data)
            mats: list[np.ndarray] = []
            n_rows = 0
            for col in cols:
                try:
                    vec = np.asarray(data[col], dtype=np.float32).reshape(-1)  # type: ignore[index]
                    mats.append(vec)
                    n_rows = max(n_rows, int(vec.size))
                except Exception:
                    continue
            if mats:
                arr = np.zeros((n_rows, len(mats)), dtype=np.float32)
                for j, vec in enumerate(mats):
                    take = min(n_rows, int(vec.size))
                    if take > 0:
                        arr[:take, j] = vec[:take]
            else:
                arr = np.asarray(data)
    elif _is_frame_like(data):
        cols = _frame_columns(data)
        mats: list[np.ndarray] = []
        n_rows = 0
        for col in cols:
            try:
                vec = np.asarray(data[col], dtype=np.float32).reshape(-1)  # type: ignore[index]
                mats.append(vec)
                n_rows = max(n_rows, int(vec.size))
            except Exception:
                continue
        if mats:
            arr = np.zeros((n_rows, len(mats)), dtype=np.float32)
            for j, vec in enumerate(mats):
                take = min(n_rows, int(vec.size))
                if take > 0:
                    arr[:take, j] = vec[:take]
        else:
            arr = np.asarray(data)
    else:
        arr = np.asarray(data)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    out = np.asarray(arr, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _as_1d_int8(data: Any) -> np.ndarray:
    if data is None:
        return np.zeros((0,), dtype=np.int8)
    arr = np.asarray(data)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    out = np.asarray(arr, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.int8, copy=False)


def _gpu_baseline(device_name: str | None) -> float:
    """
    Rough relative throughput multiplier vs CPU for common GPUs.
    Higher is faster. 1.0 = CPU baseline.
    """
    if not device_name:
        return 1.0
    name = device_name.lower()
    # Approximate relative factors (very coarse)
    table = {
        "a4000": 10.0,
        "a5000": 12.0,
        "a6000": 15.0,
        "a100": 25.0,
        "h100": 40.0,
        "l40": 12.0,
        "l40s": 18.0,
        "v100": 15.0,
        "t4": 6.0,
        "rtx 3090": 14.0,
        "rtx 4090": 25.0,
    }
    for key, val in table.items():
        if key in name:
            return val
    return 6.0  # default GPU uplift if unknown


class BenchmarkService:
    """Estimates training complexity and time."""

    COMPLEXITY_MAP = {
        "lightgbm": 5e-5,
        "xgboost": 5e-4,
        "xgboost_rf": 6e-4,
        "xgboost_dart": 7e-4,
        "catboost": 1.2e-3,
        "catboost_alt": 1.3e-3,
        "mlp": 8e-4,
        "transformer": 1e-3,
        "nbeats": 1.1e-3,
        "tide": 1.0e-3,
        "tabnet": 1.5e-3,
        "kan": 1.2e-3,
        "evolution": 0.5e-3,
        "genetic": 5e-3,
        "unsupervised": 2e-4,
        "rl_ppo": 2e-3,
        "rl_sac": 2e-3,
        "rllib_ppo": 5e-3,
        "rllib_sac": 5e-3,
    }

    SCALING_EXPONENTS = {
        "lightgbm": 0.75,
        "xgboost": 0.75,
        "catboost": 0.80,
        "transformer": 1.0,
        "nbeats": 1.0,
        "tide": 1.0,
        "tabnet": 1.0,
        "kan": 1.0,
        "rl_ppo": 1.0,
        "rl_sac": 1.0,
        "evolution": 1.0,
        "genetic": 1.0,
    }
    # Rough epoch counts per model; used to scale estimates for full vs incremental training
    EPOCH_MAP = {
        "lightgbm": 5,
        "xgboost": 5,
        "catboost": 6,
        "nbeats": 10,
        "tide": 10,
        "tabnet": 10,
        "kan": 10,
        "transformer": 12,
        "evolution": 6,
        "rl_ppo": 8,
        "rl_sac": 8,
    }
    # Cache probe throughputs keyed by model + shapes + device
    _probe_cache: dict[tuple, float] = {}

    def run_micro_benchmark(self, X: Any, y: Any, device: str) -> dict:
        """
        REAL benchmark: Train small model on actual data, measure ACTUAL time.
        Returns dict with raw measurements, no guessing.
        """
        try:
            # Use a safe sample size that accommodates sequence lookbacks (e.g. 60) and splits
            n_samples = min(10000, _row_count(X))
            if n_samples < 200:  # Too small to benchmark
                return {"samples_trained": 0, "actual_duration_sec": 0, "samples_per_second": 0, "device": device}

            # Tensor-native probe for all dataset containers (numpy/frame-like/dataframe).
            sub_X = _as_2d_float32(_slice_rows(X, n_samples))
            sub_y = _as_1d_int8(_slice_rows(y, n_samples))
            if sub_X.shape[0] > 0 and sub_X.shape[1] > 0:
                rows = min(int(sub_X.shape[0]), int(sub_y.size) if sub_y.size > 0 else int(sub_X.shape[0]))
                sub_X = sub_X[:rows]
                if sub_y.size > 0:
                    sub_y = sub_y[:rows]
                else:
                    sub_y = np.zeros(rows, dtype=np.int8)
                bench_device = select_device(device)
                probe_model = torch.nn.Sequential(
                    torch.nn.Linear(sub_X.shape[1], 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 3),
                ).to(bench_device)
                optimizer = torch.optim.AdamW(probe_model.parameters(), lr=1e-3)
                criterion = torch.nn.CrossEntropyLoss()
                x_tensor = torch.as_tensor(sub_X, dtype=torch.float32, device=bench_device)
                y_tensor = torch.as_tensor(np.clip(sub_y + 1, 0, 2), dtype=torch.long, device=bench_device)
                bs = 128
                steps = max(2, min(24, int(np.ceil(rows / bs))))
                if torch.cuda.is_available() and str(bench_device).startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                start = 0
                for _ in range(steps):
                    end = min(start + bs, rows)
                    xb = x_tensor[start:end]
                    yb = y_tensor[start:end]
                    logits = probe_model(xb)
                    loss = criterion(logits, yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    start = 0 if end >= rows else end
                if torch.cuda.is_available() and str(bench_device).startswith("cuda"):
                    torch.cuda.synchronize()
                duration = time.perf_counter() - t0
                return {
                    "samples_trained": rows,
                    "actual_duration_sec": duration,
                    "samples_per_second": rows / duration if duration > 0 else 0,
                    "device": str(bench_device),
                }
            if not _is_dataframe_like(X):
                return {"samples_trained": 0, "actual_duration_sec": 0, "samples_per_second": 0, "device": device}

            sub_X = _slice_rows(X, n_samples)
            sub_y = _slice_rows(y, n_samples)

            # Use a simplified Transformer for the probe (good representative of heavy compute)
            # Ensure seq_len is small to avoid index errors on small slices
            prefer_gpu = str(device).startswith("cuda")
            model = get_model_class("transformer", prefer_gpu=prefer_gpu)(
                d_model=32,
                nhead=2,
                num_layers=1,
                seq_len=30,  # Small sequence length for safety
                max_time_sec=10,  # Hard cap for probe
                device=device,
                batch_size=32,
            )

            t0 = time.perf_counter()
            try:
                model.fit(sub_X, sub_y)
            except Exception as e:
                logger.debug(f"Probe model fit failed ({e}). Trying simpler model...")
                # Fallback to a simpler model if Transformer fails (e.g. XGBoost if installed, or just skip)
                return {"samples_trained": 0, "actual_duration_sec": 0, "samples_per_second": 0, "device": device}

            duration = time.perf_counter() - t0

            # REAL measurements only
            result = {
                "samples_trained": n_samples,
                "actual_duration_sec": duration,
                "samples_per_second": n_samples / duration if duration > 0 else 0,
                "device": device,
            }

            logger.info(
                f"REAL Benchmark: {n_samples} samples in {duration:.2f}s "
                f"= {result['samples_per_second']:.0f} samples/sec on {device}"
            )
            return result

        except Exception as e:
            logger.warning(f"Benchmark failed: {e}")
            return {"samples_trained": 0, "actual_duration_sec": 0, "samples_per_second": 0, "device": device}

    def _hardware_signature(self, simulated_gpu: str | None = None) -> tuple:
        try:
            if simulated_gpu:
                return ("cuda-sim", simulated_gpu, 1, None, None)
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return ("cuda", props.name, torch.cuda.device_count(), props.multi_processor_count, props.total_memory)
        except Exception:
            pass
        try:
            return ("cpu", multiprocessing.cpu_count())
        except Exception:
            return ("unknown",)

    def probe_throughput(
        self,
        model_name: str,
        X: Any,
        batch_size: int,
        device: str,
        steps: int = 10,
        simulated_gpu: str | None = None,
    ) -> float | None:
        """
        Time a few mini-batches on the real model/config to estimate steps/sec.
        Uses cache keyed by model, feature shape, batch size, and hardware signature.
        """
        try:
            feature_dim = _feature_dim(X, default=0)
            sig = (model_name, feature_dim, batch_size, device, self._hardware_signature(simulated_gpu))
            if sig in self._probe_cache:
                return self._probe_cache[sig]

            prefer_gpu = (simulated_gpu is not None) or str(device).startswith("cuda")
            cls = get_model_class(model_name, prefer_gpu=prefer_gpu)
            # Small model config for probe to avoid long warmup
            kwargs = {}
            if model_name == "transformer":
                kwargs.update({"d_model": 64, "n_heads": 2, "n_layers": 1})
            if model_name in {"tide", "tabnet", "kan", "nbeats"}:
                kwargs.update({"batch_size": min(batch_size, 128)})

            model = cls(**kwargs)
            if hasattr(model, "device"):
                model.device = select_device(device if simulated_gpu is None else "cuda")

            total_rows = _row_count(X)
            if total_rows <= 0:
                return None
            bs = min(batch_size, total_rows)
            Xs = _slice_rows(X, max(bs * steps, bs))
            # Build a dataloader-like loop

            data = torch.as_tensor(
                _as_2d_float32(Xs),
                dtype=torch.float32,
                device=model.device if hasattr(model, "device") else (device if simulated_gpu is None else "cuda"),
            )
            labels = torch.zeros(len(Xs), dtype=torch.int64, device=data.device)
            criterion = torch.nn.CrossEntropyLoss() if hasattr(torch.nn, "CrossEntropyLoss") else None

            model.train()
            if hasattr(model, "optimizer"):
                optimizer = model.optimizer
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) if hasattr(model, "parameters") else None

            # Warmup
            start_idx = 0
            for _ in range(min(3, steps)):
                end_idx = start_idx + bs
                xb = data[start_idx:end_idx]
                yb = labels[start_idx:end_idx]
                out = model(xb) if callable(model) else model.fit  # type: ignore
                if callable(out):
                    out = out(xb)
                if optimizer and criterion and hasattr(out, "shape"):
                    loss = criterion(out, yb % out.shape[1])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                start_idx = (start_idx + bs) % len(data)

            if torch.cuda.is_available() or simulated_gpu:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            start_idx = 0
            for _ in range(steps):
                end_idx = start_idx + bs
                xb = data[start_idx:end_idx]
                yb = labels[start_idx:end_idx]
                out = model(xb) if callable(model) else model.fit  # type: ignore
                if callable(out):
                    out = out(xb)
                if optimizer and criterion and hasattr(out, "shape"):
                    loss = criterion(out, yb % out.shape[1])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                start_idx = (start_idx + bs) % len(data)
            if torch.cuda.is_available() or simulated_gpu:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            steps_per_sec = steps / max(dt, 1e-6)
            self._probe_cache[sig] = steps_per_sec
            return steps_per_sec
        except Exception as e:
            logger.debug(f"Throughput probe failed for {model_name}: {e}")
            return None

    GPU_SPECS = {
        # FP32 TFLOPS, Memory BW (GB/s)
        "a100": (19.5, 1555),
        "h100": (60.0, 3350),
        "a6000": (38.7, 768),
        "a5000": (27.8, 768),
        "a4000": (19.2, 448),
        "rtx 4090": (82.6, 1008),
        "rtx 3090": (35.6, 936),
        "v100": (15.7, 900),
        "t4": (8.1, 320),
        "cpu": (0.5, 50),  # Baseline reference
    }

    # Approx GFLOPs per sample per epoch (forward+backward)
    MODEL_GFLOPS = {
        "transformer": 0.5,
        "nbeats": 0.3,
        "tide": 0.25,
        "tabnet": 0.4,
        "kan": 0.45,
        "xgboost": 0.001,  # Tree models are memory bound, not compute bound
        "lightgbm": 0.001,
        "catboost": 0.002,
        "evolution": 0.1,
    }

    def _theoretical_estimate(self, model_name: str, n_samples: int, epochs: int, device_name: str) -> float:
        """
        Calculate theoretical training time based on compute/memory bounds.
        Time = max(Compute Time, Memory Time)
        """
        # 1. Identify Hardware Specs
        specs = (0.5, 50)  # Default CPU
        norm_name = device_name.lower()
        for key, val in self.GPU_SPECS.items():
            if key in norm_name:
                specs = val
                break

        tflops, bw_gbps = specs

        # 2. Identify Model Complexity
        gflops_per_sample = self.MODEL_GFLOPS.get(model_name, 0.1)

        # 3. Compute Bound Time
        # Total GFLOPs = n_samples * epochs * gflops_per_sample
        # Time = Total GFLOPs / (TFLOPS * 1000 * Efficiency)
        utilization = 0.4  # Conservative real-world utilization
        total_gflops = n_samples * epochs * gflops_per_sample
        compute_time = total_gflops / (tflops * 1000 * utilization)

        # 4. Memory Bound Time (Rough approximation)
        # Assuming 4 bytes per feature * 100 features * 3 (read/write/grad)
        data_gb = (n_samples * 100 * 4 * 3 * epochs) / 1e9
        memory_time = data_gb / (bw_gbps * utilization)

        # Real time is max of bounds (usually compute for DL, memory for Trees)
        est_time = max(compute_time, memory_time)

        # Sanity check: Tree models are CPU bound or super fast on GPU, simple logic doesn't fit well
        if "boost" in model_name and "cuda" in device_name:
            est_time = est_time * 0.1  # GPU trees are very fast

        return est_time

    def _probe_scaling_law(self, model_name: str, X: Any, y: Any, device: str) -> tuple[float, float]:
        """
        Run benchmarks at multiple scales to determine startup cost (c) and per-sample cost (m).
        Returns (m, c) for Time = m * N + c
        """
        sizes = [1000, 2000, 4000]
        times = []
        valid_sizes = []

        # Use real data if available and sufficient, otherwise fallback to random
        use_real_data = X is not None and _row_count(X) >= max(sizes)

        if not use_real_data:
            # Fallback: Create dummy data
            max_size = max(sizes)
            feature_dim = max(1, _feature_dim(X, default=100))
            X_source = np.random.randn(max_size, feature_dim).astype(np.float32)
            y_source = np.random.randint(0, 3, size=max_size, dtype=np.int8)
            meta_source = None
        else:
            # Use Real Data
            X_source = X
            y_source = y
            meta_source = None

        for size in sizes:
            # If using real data, strict check; for dummy we made enough
            if size > _row_count(X_source):
                break

            sub_X = _slice_rows(X_source, size)
            sub_y = _slice_rows(y_source, size)
            sub_meta = _slice_rows(meta_source, size) if meta_source is not None else None

            # Instantiate model
            try:
                # Use simplified config to probe speed quickly
                prefer_gpu = str(device).startswith("cuda")
                cls = get_model_class(model_name, prefer_gpu=prefer_gpu)
                kwargs = {"max_time_sec": 30}  # Cap probe time
                if "transformer" in model_name:
                    kwargs.update({"d_model": 32, "num_layers": 1})
                if "evolution" in model_name:
                    kwargs.update({"population_size": 10, "generations": 1})
                if "genetic" in model_name:
                    kwargs.update({"population_size": 10, "generations": 1})

                model = cls(**kwargs)
                if hasattr(model, "device"):
                    model.device = select_device(device)

                t0 = time.perf_counter()
                if hasattr(model, "fit"):
                    # Try passing metadata first (for RL/Genetic)
                    try:
                        if sub_meta is not None:
                            model.fit(sub_X, sub_y, metadata=sub_meta)
                        else:
                            model.fit(sub_X, sub_y)
                    except TypeError:
                        # Fallback for models that don't accept metadata
                        model.fit(sub_X, sub_y)

                duration = time.perf_counter() - t0

                times.append(duration)
                valid_sizes.append(size)
            except Exception:
                pass

        if len(valid_sizes) < 2:
            return 0.0, 0.0  # Failed to establish trend

        # Fit Line: T = m*N + c
        # Simple Linear Regression
        N = np.array(valid_sizes)
        T = np.array(times)
        A = np.vstack([N, np.ones(len(N))]).T
        m, c = np.linalg.lstsq(A, T, rcond=None)[0]

        return max(0.0, m), max(0.0, c)

    def estimate_time(
        self,
        models: list[str],
        n_samples: int,
        benchmark_result: dict | None,  # Kept for backward compatibility but largely ignored in new logic
        gpu: bool,
        gpu_count: int = 1,
        context: str = "full",
        historical_durations: dict[str, float] | None = None,
        historical_n: int | None = None,
        historical_gpu: tuple[bool, int] | None = None,
        incremental_stats: dict[str, dict[str, Any]] | None = None,
        probe_kwargs: dict[str, Any] | None = None,
        simulate_gpu: str | None = None,
    ) -> float:
        total_time = 0.0
        breakdown = []

        # Probe Data (if available in kwargs)
        X_probe = probe_kwargs.get("X") if probe_kwargs else None

        # Hardware Factor (if simulating)
        hw_factor = 1.0
        if simulate_gpu:
            hw_factor = _gpu_baseline(simulate_gpu)
            logger.info(f"Estimating for {simulate_gpu} (Speedup Factor: {hw_factor:.1f}x vs Baseline)")

        for m in models:
            est_seconds = 0.0
            source = "unknown"

            # 1. Historical Exact Match (Fastest/Most Accurate)
            if context == "incremental" and incremental_stats and m in incremental_stats:
                stat = incremental_stats[m]
                if stat.get("n_samples", 0) > 0:
                    ratio = n_samples / stat["n_samples"]
                    est_seconds = stat["duration_sec"] * ratio
                    source = "incremental_history"

            # 2. Historical Scaling (Full Training)
            elif historical_durations and historical_n and m in historical_durations:
                ratio = n_samples / historical_n
                # Adjust for epoch scaling if context differs
                epoch_scale = 0.35 if context == "incremental" else 1.0
                est_seconds = historical_durations[m] * ratio * epoch_scale
                source = "full_history"

            # 3. Multi-Point Probe (Scientific Extrapolation)
            elif X_probe is not None and not simulate_gpu:
                # We can run a real probe on current hardware
                # Note: We assume y is available or we mock it
                y_probe = np.zeros(_row_count(X_probe), dtype=np.int8)  # Mock labels
                device = "cuda" if gpu else "cpu"

                slope, intercept = self._probe_scaling_law(m, X_probe, y_probe, device)
                if slope > 0:
                    est_seconds = (slope * n_samples) + intercept
                    source = "scaling_law_probe"

            # 4. Theoretical / Fallback (Simulated or No Probe Available)
            if est_seconds == 0.0:
                # Heuristic
                # Time ~ C * N
                complexity = self.COMPLEXITY_MAP.get(m, 5e-3)  # Default to high (5e-3) if unknown
                est_seconds = complexity * n_samples

                # Apply Sub-Linear Scaling Correction for large datasets
                exponent = self.SCALING_EXPONENTS.get(m, 1.0)
                if n_samples > 10000 and exponent < 1.0:
                    # Correction factor: (N / 10k)^(exp - 1)
                    # This reduces the estimate as N grows if exp < 1 (e.g. tree models)
                    correction = (n_samples / 10000.0) ** (exponent - 1.0)
                    est_seconds *= correction

                # Adjust for GPU (rough heuristic scaling)
                if simulate_gpu:
                    est_seconds /= max(hw_factor, 1e-6) * max(gpu_count, 1)
                    source = f"theoretical_{simulate_gpu}"
                elif gpu:
                    est_seconds /= max(_gpu_baseline("gpu"), 1e-6) * max(gpu_count, 1)
                    source = "complexity_heuristic"
                else:
                    source = "complexity_heuristic"

            # Apply hard epoch scaling for final estimate
            epochs = self.EPOCH_MAP.get(m, 5)
            # If the estimate source was per-epoch (like probe), multiply by epochs.
            # If source was 'history' (total time), don't multiply.
            if source in ["scaling_law_probe", "complexity_heuristic", f"theoretical_{simulate_gpu}"]:
                est_seconds *= epochs

            total_time += est_seconds
            breakdown.append(f"{m}: {est_seconds / 60:.1f}min ({source})")

        if breakdown:
            logger.info(f"Scientific Time Estimate (N={n_samples}):")
            for line in breakdown:
                logger.info(f"  → {line}")

        return total_time

