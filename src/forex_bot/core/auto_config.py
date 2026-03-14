"""
Hardware Auto-Detection and Profile Configuration.

Replaces 40+ manual FOREX_BOT_* environment variable flags with
automatic hardware detection and intelligent profile selection.

Usage:
    from forex_bot.core.auto_config import auto_configure
    auto_configure()  # Call once at startup before Settings() init

The user only needs to set FOREX_BOT_PROFILE to override (optional):
    "rust"     -> force Rust backend, no Python fallbacks
    "hybrid"   -> allow Python fallback where Rust binding missing
    "python"   -> force Python path (debugging only)
    "auto"     -> detect hardware and choose best profile (default)
"""

import logging
import os
import sys


logger = logging.getLogger(__name__)

# ============================================================================
# HARDWARE DETECTION
# ============================================================================

def detect_gpu_count() -> int:
    """Detect number of available GPUs (CUDA or ROCm)."""
    # 1. Try nvidia-smi count
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return len(result.stdout.strip().splitlines())
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass

    # 2. Try torch.cuda
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass

    # 3. Try CUDA_VISIBLE_DEVICES env
    cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_devs.strip():
        return len([d for d in cuda_devs.split(",") if d.strip()])

    return 0


def detect_gpu_vram_gb() -> float:
    """Detect total VRAM of first GPU in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    except (ImportError, RuntimeError):
        pass
    return 0.0


def detect_ram_gb() -> float:
    """Detect total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    # Fallback: read from /proc/meminfo on Linux
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / (1024 ** 2)
    except (FileNotFoundError, ValueError, OSError):
        pass
    # Windows fallback
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        class MEMORYSTATUSEX(ctypes.Structure):  # noqa: N801
            """Windows MEMORYSTATUSEX structure for GlobalMemoryStatusEx."""
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        mem = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
        kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        return mem.ullTotalPhys / (1024 ** 3)
    except (AttributeError, OSError, ValueError):
        pass
    return 16.0  # conservative fallback


def detect_cpu_count() -> int:
    """Detect number of CPU cores."""
    return os.cpu_count() or 4


def detect_rust_bindings_available() -> bool:
    """Check if Rust forex_bindings module is importable."""
    try:
        import importlib
        importlib.import_module("forex_bindings")
        return True
    except ImportError:
        return False


# ============================================================================
# PROFILE SELECTION
# ============================================================================

def _auto_select_profile(
    ram_gb: float,
    gpu_count: int,
    gpu_vram_gb: float,  # noqa: ARG001 — reserved for future VRAM-aware selection
    cpu_count: int,  # noqa: ARG001 — reserved for future CPU-aware selection
    rust_available: bool,
) -> str:
    """
    Automatically select the best runtime profile based on hardware.

    Returns one of:
        "hpc"        -> 8+ GPUs, 128+ GB RAM (cloud HPC)
        "rust_max"   -> 1+ GPU, 32+ GB RAM, Rust available
        "rust_32gb"  -> 1+ GPU, 16-32 GB RAM, Rust available
        "hybrid"     -> Rust available but limited hardware
        "python"     -> No Rust bindings
        "light"      -> <8 GB RAM (embedded/CI)
    """
    if not rust_available:
        return "python"

    if ram_gb < 8:
        return "light"

    if gpu_count >= 8 and ram_gb >= 128:
        return "hpc"
    elif gpu_count >= 1 and ram_gb >= 32:
        return "rust_max"
    elif gpu_count >= 1 and ram_gb >= 16:
        return "rust_32gb"
    else:
        return "hybrid"


# ============================================================================
# PROFILE ENVIRONMENT VARIABLES
# ============================================================================

# Consolidated profiles: each profile sets only the variables that differ
# from defaults. The default is "rust" — all Rust, no Python fallback.
_PROFILE_OVERRIDES = {
    "hpc": {
        "FOREX_BOT_HPC_MODE": "auto",
        "FOREX_BOT_GPU_PREFERENCE": "multi",
        "FOREX_BOT_PARALLEL_POPULATION": "1",
        "FOREX_BOT_MAX_THREADS": "0",  # auto-detect
        "FOREX_BOT_CHUNK_SIZE": "32768",
    },
    "rust_max": {
        "FOREX_BOT_GPU_PREFERENCE": "cuda",
        "FOREX_BOT_PARALLEL_POPULATION": "1",
        "FOREX_BOT_MAX_THREADS": "0",
        "FOREX_BOT_CHUNK_SIZE": "16384",
    },
    "rust_32gb": {
        "FOREX_BOT_GPU_PREFERENCE": "cuda",
        "FOREX_BOT_MAX_THREADS": "0",
        "FOREX_BOT_CHUNK_SIZE": "8192",
    },
    "hybrid": {
        "FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK": "1",
        "FOREX_BOT_GPU_PREFERENCE": "cpu",
        "FOREX_BOT_MAX_THREADS": "0",
    },
    "python": {
        "FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK": "1",
        "FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK": "1",
        "FOREX_BOT_FEATURES_BACKEND": "python",
        "FOREX_BOT_DATA_BACKEND": "python",
        "FOREX_BOT_TREE_BACKEND": "python",
    },
    "light": {
        "FOREX_BOT_FEATURE_PROFILE": "minimal",
        "FOREX_BOT_MAX_FEATURES": "24",
        "FOREX_BOT_GPU_PREFERENCE": "cpu",
        "FOREX_BOT_MAX_THREADS": "2",
        "FOREX_BOT_CHUNK_SIZE": "2048",
    },
}

# Defaults applied to ALL profiles (Rust-first, zero Python fallback)
_BASE_DEFAULTS = {
    "FOREX_BOT_FEATURES_ALLOW_PY_FALLBACK": "0",
    "FOREX_BOT_GENETIC_ALLOW_PY_FALLBACK": "0",
    "FOREX_BOT_TALIB_ALLOW_PY_FALLBACK": "0",
    "FOREX_BOT_PROP_PY_FALLBACK": "0",
    "FOREX_BOT_PROP_ALLOW_PY_RESCORING": "0",
    "FOREX_BOT_PROP_ALLOW_PY_EXPANSION": "0",
    "FOREX_BOT_STOP_TARGET_ALLOW_PY_FALLBACK": "0",
    "FOREX_BOT_BASE_SIGNAL_ALLOW_PY_MIXER": "0",
    "FOREX_BOT_BASE_SIGNAL_ALLOW_CLASSIC_FALLBACK": "0",
    "FOREX_BOT_TREE_RUST_FALLBACK": "0",
    "FOREX_BOT_FEATURES_BACKEND": "rust_strict",
    "FOREX_BOT_DATA_BACKEND": "rust_strict",
}


def _derive_feature_profile(ram_gb: float) -> tuple[str, str]:
    """Derive feature profile and max_features from RAM."""
    if ram_gb >= 64:
        return "full", "0"  # unlimited
    elif ram_gb >= 32:
        return "core", "96"
    elif ram_gb >= 16:
        return "compact", "48"
    else:
        return "minimal", "24"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def auto_configure() -> str:
    """
    Auto-detect hardware and configure environment variables.

    Call ONCE at startup, before constructing Settings().
    Returns the selected profile name.

    The user can override with FOREX_BOT_PROFILE:
        "auto"    -> full auto-detection (default)
        "rust"    -> force Rust, auto-detect hardware
        "hybrid"  -> allow Python fallback
        "python"  -> force Python
        "hpc"     -> force HPC profile
    """
    user_profile = os.environ.get("FOREX_BOT_PROFILE", "auto").strip().lower()

    # Also accept legacy FOREX_BOT_RUNTIME_PROFILE
    if user_profile == "auto":
        legacy = os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "").strip().lower()
        if legacy:
            user_profile = legacy

    # Detect hardware
    ram_gb = detect_ram_gb()
    gpu_count = detect_gpu_count()
    gpu_vram_gb = detect_gpu_vram_gb()
    cpu_count = detect_cpu_count()
    rust_available = detect_rust_bindings_available()

    # Select profile
    if user_profile == "auto":
        profile = _auto_select_profile(ram_gb, gpu_count, gpu_vram_gb, cpu_count, rust_available)
    elif user_profile in _PROFILE_OVERRIDES:
        profile = user_profile
    elif user_profile.startswith("rust"):
        # Handle "rust_32gb", "rust_max", "rust" etc.
        profile = user_profile if user_profile in _PROFILE_OVERRIDES else "rust_max"
    else:
        profile = user_profile if user_profile in _PROFILE_OVERRIDES else "hybrid"

    # Apply base defaults (only if not already set by user)
    for key, value in _BASE_DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = value

    # Apply profile overrides
    overrides = _PROFILE_OVERRIDES.get(profile, {})
    for key, value in overrides.items():
        if key not in os.environ:
            os.environ[key] = value

    # Auto-derive feature profile from RAM
    feat_profile, max_features = _derive_feature_profile(ram_gb)
    if "FOREX_BOT_FEATURE_PROFILE" not in os.environ:
        os.environ["FOREX_BOT_FEATURE_PROFILE"] = feat_profile
    if "FOREX_BOT_MAX_FEATURES" not in os.environ:
        os.environ["FOREX_BOT_MAX_FEATURES"] = max_features

    # Auto-set thread count
    if os.environ.get("FOREX_BOT_MAX_THREADS", "0") == "0":
        # Use 75% of CPU cores, minimum 2
        threads = max(2, int(cpu_count * 0.75))
        os.environ["FOREX_BOT_MAX_THREADS"] = str(threads)

    logger.info(
        "Auto-configured: profile=%s RAM=%.1fGB GPUs=%d rust=%s feat=%s",
        profile, ram_gb, gpu_count, rust_available, feat_profile,
    )

    return profile


def get_hardware_summary() -> dict:
    """Return hardware detection results for UI/diagnostics."""
    return {
        "ram_gb": round(detect_ram_gb(), 1),
        "gpu_count": detect_gpu_count(),
        "gpu_vram_gb": round(detect_gpu_vram_gb(), 1),
        "cpu_count": detect_cpu_count(),
        "rust_available": detect_rust_bindings_available(),
        "platform": sys.platform,
    }
