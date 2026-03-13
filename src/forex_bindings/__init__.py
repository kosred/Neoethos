from __future__ import annotations

import contextlib
import site
from pathlib import Path


def _installed_package_dirs() -> list[str]:
    current_dir = Path(__file__).resolve().parent
    candidates: list[str] = []

    raw_roots: list[str] = []
    with contextlib.suppress(Exception):
        raw_roots.extend(site.getsitepackages())
    with contextlib.suppress(Exception):
        raw_roots.append(site.getusersitepackages())

    seen: set[str] = set()
    for raw_root in raw_roots:
        root = Path(str(raw_root)).resolve()
        pkg_dir = root / "forex_bindings"
        if not pkg_dir.is_dir():
            continue
        if pkg_dir == current_dir:
            continue
        text = str(pkg_dir)
        if text in seen:
            continue
        seen.add(text)
        candidates.append(text)
    return candidates


# Importing xgboost primes its native DLL search path on Windows so the
# tree-enabled PyO3 extension can load consistently.
with contextlib.suppress(Exception):
    import xgboost  # type: ignore  # noqa: F401

for _pkg_dir in _installed_package_dirs():
    if _pkg_dir not in __path__:
        __path__.append(_pkg_dir)

from .forex_bindings import *  # type: ignore  # noqa: F401,F403,E402
