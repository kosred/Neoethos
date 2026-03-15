"""
Static Dependency Verifier.
Checks if critical packages are present and warns the user if setup is required.
Does NOT attempt runtime installation or process restarts.
"""

from __future__ import annotations
import importlib
import importlib.metadata
import logging

logger = logging.getLogger(__name__)

CRITICAL_PACKAGES = [
    "numpy",
    "torch",
    "cupy",
    "numba",
    "talib",
    "pydantic",
    "sklearn",
    "xgboost",
    "catboost",
]

def ensure_dependencies() -> None:
    """
    Verify that the environment is correctly set up.
    If packages are missing, it logs a warning but allows the bot to try and run.
    """
    missing = []
    for pkg in CRITICAL_PACKAGES:
        try:
            importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            # Check for alternative names used in some environments
            if pkg == "talib":
                try:
                    importlib.import_module("talib")
                except ImportError:
                    missing.append(pkg)
            elif pkg == "cupy":
                try:
                    importlib.import_module("cupy")
                except ImportError:
                    missing.append(pkg)
            else:
                missing.append(pkg)

    if missing:
        msg = (
            f"\n{'!' * 60}\n"
            f"[WARNING] Missing critical dependencies: {missing}\n"
            "Please run the following command to set up your environment:\n"
            "  python3 -m pip install -e . --user --break-system-packages\n"
            f"{'!' * 60}\n"
        )
        logger.warning(msg)
    else:
        logger.info("✓ Environment dependencies verified.")
