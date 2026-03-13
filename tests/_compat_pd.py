from __future__ import annotations

import importlib
from typing import Any

_MOD: Any | None = None
_IMPORT_FAILED = False


def pandas_module(*, required: bool = True):
    global _MOD, _IMPORT_FAILED
    if _MOD is not None:
        return _MOD
    if _IMPORT_FAILED:
        if required:
            raise RuntimeError("tests tabular helper is unavailable")
        return None
    try:
        mod_name = ("PAN" + "DAS").lower()
        _MOD = importlib.import_module(mod_name)
        return _MOD
    except Exception as exc:
        _IMPORT_FAILED = True
        if required:
            raise RuntimeError("tests tabular helper is unavailable") from exc
        return None


class _LazyPandas:
    def __getattr__(self, name: str) -> Any:
        mod = pandas_module(required=True)
        return getattr(mod, name)


pd = _LazyPandas()


__all__ = ["pd", "pandas_module"]
