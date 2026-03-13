#!/usr/bin/env python3
"""
Forex AI Trading Bot - Master Autonomous Launcher
2025 HPC Edition (Self-Bootstrapping)
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# --- 0.5 VENV BOOTSTRAP (PREFER PROJECT VENV IF PRESENT) ---
SCRIPT_DIR = Path(__file__).resolve().parent

def _in_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix or hasattr(sys, "real_prefix")

def _venv_python() -> Path:
    if os.name == "nt":
        return SCRIPT_DIR / ".venv" / "Scripts" / "python.exe"
    return SCRIPT_DIR / ".venv" / "bin" / "python"

_venv_py = _venv_python()
if _venv_py.exists() and not _in_venv():
    os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

def _pip_cmd(args: list[str] | None = None, *, upgrade: bool = False) -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    if not _in_venv():
        cmd.append("--user")
        if platform.system().lower() == "linux":
            cmd.append("--break-system-packages")
    if args:
        cmd.extend(args)
    return cmd

def _pip_install(cmd: list[str]) -> None:
    try:
        subprocess.check_call(cmd)
    except Exception:
        if "--break-system-packages" in cmd:
            cmd = [c for c in cmd if c != "--break-system-packages"]
            subprocess.check_call(cmd)
        else:
            raise

# --- 0. HPC GLOBAL ENVIRONMENT (AUTO-DETECT HARDWARE) ---
# Fully automatic hardware detection - NO hardcoding!
#
# To override auto-detection, set these environment variables BEFORE running:
#   FOREX_BOT_CPU_THREADS=X    - Number of worker processes (default: logical_cores - 1)
#   FOREX_BOT_CPU_RESERVE=X    - Cores to reserve for OS (default: 1)
#   OMP_NUM_THREADS=X          - BLAS threads per operation (default: physical_cores - 1)
#   FOREX_BOT_RL_ENVS=X        - RL parallel environments (default: auto, RAM-limited)
#
# Example: set OMP_NUM_THREADS=4 && python forex-ai.py train

# Auto-detect actual physical cores (not hyperthreaded logical cores)
try:
    import psutil
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1
except Exception:
    # Fallback: assume no hyperthreading
    logical_cores = os.cpu_count() or 1
    physical_cores = logical_cores

# CPU budget for worker processes (uses logical cores)
try:
    cpu_reserve = int(os.environ.get("FOREX_BOT_CPU_RESERVE", "1") or 1)
except Exception:
    cpu_reserve = 1
cpu_budget = max(1, logical_cores - max(0, cpu_reserve))

# CRITICAL: When using ProcessPoolExecutor, each process spawns its own BLAS threads!
# If we have 11 workers × 5 BLAS threads = 55 threads (over-subscription!)
# Solution: Set BLAS to 1 thread when using multiprocessing
# This way: 11 workers × 1 BLAS thread = 11 threads (optimal)
def _read_int_env(*keys: str) -> int | None:
    for key in keys:
        val = os.environ.get(key)
        if val:
            try:
                return max(1, int(val))
            except Exception:
                continue
    return None


def _derive_parallel_hints(cpu_budget: int, gpu_count: int) -> tuple[int, int, str]:
    usable_cpu = max(1, int(cpu_budget or 1))
    gpus = max(0, int(gpu_count or 0))
    if gpus <= 0:
        return usable_cpu, 0, "auto"
    return max(1, usable_cpu // gpus), gpus, "auto"


def _is_train_worker_process() -> bool:
    if "--_worker" in sys.argv[1:]:
        return True
    raw = str(os.environ.get("FOREX_BOT_TRAIN_WORKER", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _set_numexpr_thread_env(threads: int, *, overwrite_num_threads: bool) -> None:
    target = max(1, int(threads or 1))
    current_num = _read_int_env("NUMEXPR_NUM_THREADS")
    if overwrite_num_threads or current_num is None:
        num_threads = target
        os.environ["NUMEXPR_NUM_THREADS"] = str(target)
    else:
        num_threads = max(1, int(current_num))
    current_max = _read_int_env("NUMEXPR_MAX_THREADS") or 0
    os.environ["NUMEXPR_MAX_THREADS"] = str(max(target, num_threads, current_max))

auto_mode = str(os.environ.get("FOREX_BOT_AUTO_TUNE", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Auto-split CPU budgets when GPUs are present (training vs discovery/search).
gpu_count = 0
try:
    import torch

    if torch.cuda.is_available():
        gpu_count = int(torch.cuda.device_count())
except Exception:
    gpu_count = 0

cpu_threads_per_gpu, gpu_workers, parallel_models_mode = _derive_parallel_hints(cpu_budget, gpu_count)
remaining = cpu_budget

if gpu_count > 0:
    if auto_mode:
        os.environ["FOREX_BOT_CPU_THREADS_PER_GPU"] = str(cpu_threads_per_gpu)
        os.environ["FOREX_BOT_CPU_BUDGET"] = str(cpu_budget)
        os.environ["FOREX_BOT_CPU_THREADS"] = str(cpu_budget)
    else:
        os.environ.setdefault("FOREX_BOT_CPU_THREADS_PER_GPU", str(cpu_threads_per_gpu))
        os.environ.setdefault("FOREX_BOT_CPU_BUDGET", str(cpu_budget))
        os.environ.setdefault("FOREX_BOT_CPU_THREADS", str(cpu_budget))

    # Use mixed CPU+GPU scheduling by default so GPU hosts still train CPU-native models.
    if auto_mode:
        os.environ["FOREX_BOT_PARALLEL_MODELS"] = parallel_models_mode
        os.environ["FOREX_BOT_GPU_WORKERS"] = str(gpu_workers)
    else:
        os.environ.setdefault("FOREX_BOT_PARALLEL_MODELS", parallel_models_mode)
        os.environ.setdefault("FOREX_BOT_GPU_WORKERS", str(gpu_workers))

    # Keep the event loop responsive so prop search can actually run.
    if auto_mode:
        os.environ["FOREX_BOT_DISCOVERY_ASYNC"] = "1"
    else:
        os.environ.setdefault("FOREX_BOT_DISCOVERY_ASYNC", "1")

    # If we have lots of RAM, allow prop search to stay parallel.
    try:
        mem = psutil.virtual_memory()
        if float(mem.available) / (1024**3) >= 64.0:
            if auto_mode:
                os.environ["FOREX_BOT_PROP_PARALLEL_MEM_FRAC"] = "0.90"
                os.environ["FOREX_BOT_PROP_PARALLEL_OVERHEAD"] = "1.20"
            else:
                os.environ.setdefault("FOREX_BOT_PROP_PARALLEL_MEM_FRAC", "0.90")
                os.environ.setdefault("FOREX_BOT_PROP_PARALLEL_OVERHEAD", "1.20")
    except Exception:
        pass

# Rust acceleration (CPU-heavy paths) - safe no-op if module is not built.      
rust_threads = remaining if gpu_count > 0 else cpu_budget
if auto_mode:
    os.environ["FOREX_BOT_RUST_ACCEL"] = "1"
    os.environ.setdefault("FOREX_BOT_TREE_BACKEND", "auto")
    os.environ["FOREX_BOT_RUST_THREADS"] = str(rust_threads)
    os.environ["RAYON_NUM_THREADS"] = str(rust_threads)
    os.environ["FOREX_BOT_RUST_EVO"] = "1"
    os.environ["FOREX_BOT_RUST_FEATURES"] = "auto"
    os.environ["FOREX_BOT_RUST_FEATURES_ONLY"] = "1"
else:
    os.environ.setdefault("FOREX_BOT_RUST_ACCEL", "1")
    os.environ.setdefault("FOREX_BOT_TREE_BACKEND", "auto")
    os.environ.setdefault("FOREX_BOT_RUST_THREADS", str(rust_threads))
    os.environ.setdefault("RAYON_NUM_THREADS", str(rust_threads))
    os.environ.setdefault("FOREX_BOT_RUST_EVO", "1")
    os.environ.setdefault("FOREX_BOT_RUST_FEATURES", "auto")
    os.environ.setdefault("FOREX_BOT_RUST_FEATURES_ONLY", "1")

# Respect explicit user overrides for BLAS/OMP threads; default to 1 otherwise.
is_train_worker = _is_train_worker_process()
if auto_mode:
    if is_train_worker:
        blas_threads = _read_int_env(
            "FOREX_BOT_BLAS_THREADS",
            "FOREX_BOT_OMP_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "FOREX_BOT_CPU_THREADS",
            "FOREX_BOT_CPU_BUDGET",
        )
        if blas_threads is None:
            blas_threads = 1
    else:
        blas_threads = 1
else:
    blas_threads = _read_int_env(
        "FOREX_BOT_BLAS_THREADS",
        "FOREX_BOT_OMP_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    if blas_threads is None:
        blas_threads = 1  # Single-threaded BLAS when using multiprocessing

os.environ.setdefault("FOREX_BOT_CPU_BUDGET", str(cpu_budget))
os.environ.setdefault("FOREX_BOT_CPU_THREADS", str(cpu_budget))
# BLAS libraries: Single-threaded to prevent N_workers × N_threads explosion
if auto_mode:
    _set_numexpr_thread_env(blas_threads, overwrite_num_threads=True)
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMBA_DEFAULT_NUM_THREADS"] = str(blas_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
    # Disable dynamic threading for predictable performance
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    # PyTorch/TensorFlow thread limits (match BLAS: single-threaded)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(blas_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TORCH_NUM_THREADS"] = str(blas_threads)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
else:
    _set_numexpr_thread_env(blas_threads, overwrite_num_threads=False)
    os.environ.setdefault("OMP_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("NUMBA_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("NUMBA_DEFAULT_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(blas_threads))
    # Disable dynamic threading for predictable performance
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    # PyTorch/TensorFlow thread limits (match BLAS: single-threaded)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(blas_threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# NCCL optimizations for 8x A6000 P2P topology
os.environ["NCCL_P2P_LEVEL"] = "5"
os.environ["NCCL_IB_DISABLE"] = "1"

# Print hardware auto-detection results (ONLY ONCE - not in worker processes)
# Set flag to prevent worker processes from re-printing when they import this module
if not os.environ.get("_FOREX_BOT_HW_DETECTED"):
    os.environ["_FOREX_BOT_HW_DETECTED"] = "1"
    worker_budget = _read_int_env("FOREX_BOT_CPU_THREADS", "FOREX_BOT_CPU_BUDGET") or cpu_budget
    if is_train_worker:
        workload_line = f"[HW AUTO-DETECT] Worker CPU Budget: {worker_budget} | BLAS Threads: {blas_threads}"
        strategy_line = "[HW AUTO-DETECT] Thread Strategy: Explicit worker thread budget"
    elif int(blas_threads) == 1:
        workload_line = (
            f"[HW AUTO-DETECT] Worker Processes: {cpu_budget} x BLAS Threads: {blas_threads} "
            f"= ~{cpu_budget * blas_threads} compute threads"
        )
        strategy_line = "[HW AUTO-DETECT] Thread Strategy: Single-threaded BLAS (prevents N_workers x N_threads explosion)"
    else:
        workload_line = f"[HW AUTO-DETECT] CPU Budget: {cpu_budget} | BLAS Threads: {blas_threads}"
        strategy_line = "[HW AUTO-DETECT] Thread Strategy: Explicit BLAS thread budget"
    print("=" * 70)
    print(f"[HW AUTO-DETECT] Physical Cores: {physical_cores} | Logical Cores: {logical_cores}")
    print(workload_line)
    print(strategy_line)
    try:
        mem = psutil.virtual_memory()
        print(f"[HW AUTO-DETECT] Total RAM: {mem.total/1024/1024/1024:.1f}GB | Available: {mem.available/1024/1024/1024:.1f}GB ({mem.percent:.1f}% used)")
    except Exception:
        pass
    # Auto-detect GPU
    gpu_info = "None detected"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_info = f"{gpu_count}x {gpu_name}"
    except Exception:
        pass
    print(f"[HW AUTO-DETECT] GPU: {gpu_info}")
    print("=" * 70)

# --- 1. SELF-HEALING SYSTEM SETUP ---
def bootstrap():
    """Ensure the system is optimized and ready."""
    is_linux = platform.system().lower() == "linux"
    
    # 1.1 TA-Lib (Use pre-built 2025 wheels to save time)
    try:
        import talib
    except ImportError:
        print("[INIT] TA-Lib missing. Installing pre-built binaries...")
        try:
            # In late 2025, TA-Lib has stable wheels for Python 3.13
            pip_cmd = _pip_cmd(["TA-Lib"])
            _pip_install(pip_cmd)
            print("[INIT] TA-Lib installed via wheel.")
        except Exception:
            print("[WARN] Wheel install failed. Falling back to source (this may take 5 mins)...")
            # ... (Existing source build fallback)

    # 1.2 Python Dependencies
    try:
        import importlib

        importlib.import_module("numpy")
        import pydantic
        # GPU-specific deps are optional on Windows for local CPU testing
        if platform.system().lower() != "windows":
            import torch
            import cupy
    except ImportError:
        print("[INIT] Missing Python libraries. Syncing with pyproject runtime manifest...")
        is_windows = platform.system().lower() == "windows"

        cmd = _pip_cmd(upgrade=True)
        install_target = ".[gpu]" if not is_windows else "."
        cmd += ["-e", install_target]

        try:
            _pip_install(cmd)
            print("[INIT] Stack synchronized. Restarting engine...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"[FATAL] Dependency sync failed: {e}")
            sys.exit(1)

# --- 2. ENGINE PATHS ---
SRC_DIR = SCRIPT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
os.environ["PYTHONPATH"] = str(SRC_DIR)

if __name__ == "__main__":
    # Internal workers for parallel evaluation
    if "--_worker" in sys.argv[1:]:
        from forex_bot.training.parallel_worker import run_worker
        sys.exit(run_worker(sys.argv[sys.argv.index("--_worker") + 1 :]))

    # Autonomous Setup
    bootstrap()

    # Launch the bot
    from forex_bot.main import main
    main()
