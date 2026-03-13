from __future__ import annotations

import builtins
import types

import pytest

from forex_bot.models import registry as reg


def test_strict_runtime_rejects_python_tree_backend_when_bindings_missing(monkeypatch):
    reg._CLASS_CACHE.clear()
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "rust_strict")

    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "forex_bindings":
            raise ImportError("bindings missing")
        return real_import(name, globals, locals, fromlist, level)

    class _RustLightMissing:
        _model_cls = None

    def _fake_import_module(module_name: str, package: str | None = None):
        if module_name == ".trees_rust":
            return types.SimpleNamespace(RustLightGBMExpert=_RustLightMissing)
        if module_name == ".trees":
            raise AssertionError("strict runtime must not fall back to legacy Python trees")
        raise ImportError(module_name)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    monkeypatch.setattr(reg.importlib, "import_module", _fake_import_module)

    with pytest.raises(ImportError, match="Rust runtime model is required"):
        reg.get_model_class("lightgbm", prefer_gpu=False)

    reg._CLASS_CACHE.clear()
