from __future__ import annotations

import tomllib
from pathlib import Path


def test_forex_bindings_maturin_uses_repo_python_shim() -> None:
    pyproject_toml = Path("crates/forex-bindings/pyproject.toml")
    data = tomllib.loads(pyproject_toml.read_text(encoding="utf-8"))
    maturin = data.get("tool", {}).get("maturin", {})

    assert maturin.get("module-name") == "forex_bindings.forex_bindings"
    assert maturin.get("python-source") in (None, ".")
    assert Path("crates/forex-bindings/forex_bindings/__init__.py").exists()
