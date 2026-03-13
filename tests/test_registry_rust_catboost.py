from __future__ import annotations

import sys
import types

from forex_bot.models import registry as reg


def test_rust_tree_mapping_includes_catboost_keys():
    assert reg._RUST_TREE_MAPPING.get("catboost") == "RustCatBoostExpert"
    assert reg._RUST_TREE_MAPPING.get("catboost_alt") == "RustCatBoostAltExpert"


def test_use_rust_tree_models_detects_catboost_bindings(monkeypatch):
    fake = types.SimpleNamespace(
        RustCatBoostExpert=object(),
        RustCatBoostAltExpert=object(),
    )
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "auto")
    monkeypatch.setitem(sys.modules, "forex_bindings", fake)
    assert reg._use_rust_tree_models("catboost")
    assert reg._use_rust_tree_models("catboost_alt")


def test_get_model_class_prefers_rust_catboost_when_available(monkeypatch):
    reg._CLASS_CACHE.clear()
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "auto")
    monkeypatch.setattr(reg, "_use_rust_tree_models", lambda name=None: True)

    class _RustCat:
        _model_cls = object()

    class _PyCat:
        pass

    def _fake_import(module_name: str, package: str | None = None):
        if module_name == ".trees_rust":
            return types.SimpleNamespace(RustCatBoostExpert=_RustCat)
        if module_name == ".trees":
            return types.SimpleNamespace(CatBoostExpert=_PyCat)
        raise ImportError(module_name)

    monkeypatch.setattr(reg.importlib, "import_module", _fake_import)
    cls = reg.get_model_class("catboost", prefer_gpu=False)
    assert cls is _RustCat
    reg._CLASS_CACHE.clear()


def test_get_model_class_raises_when_rust_catboost_missing(monkeypatch):
    reg._CLASS_CACHE.clear()
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "auto")
    monkeypatch.setattr(reg, "_use_rust_tree_models", lambda name=None: True)

    class _RustCatMissing:
        _model_cls = None

    class _PyCat:
        pass

    def _fake_import(module_name: str, package: str | None = None):
        if module_name == ".trees_rust":
            return types.SimpleNamespace(RustCatBoostExpert=_RustCatMissing)
        if module_name == ".trees":
            return types.SimpleNamespace(CatBoostExpert=_PyCat)
        raise ImportError(module_name)

    monkeypatch.setattr(reg.importlib, "import_module", _fake_import)
    try:
        reg.get_model_class("catboost", prefer_gpu=False)
        assert False, "Expected Rust-only tree loading to raise when Rust class is missing."
    except ImportError as exc:
        assert "Rust runtime model is required" in str(exc)
    reg._CLASS_CACHE.clear()


def test_get_model_class_raises_in_rust_strict_mode(monkeypatch):
    reg._CLASS_CACHE.clear()
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "rust_strict")
    monkeypatch.setattr(reg, "_use_rust_tree_models", lambda name=None: True)

    class _RustCatMissing:
        _model_cls = None

    def _fake_import(module_name: str, package: str | None = None):
        if module_name == ".trees_rust":
            return types.SimpleNamespace(RustCatBoostExpert=_RustCatMissing)
        if module_name == ".trees":
            return types.SimpleNamespace(CatBoostExpert=object())
        raise ImportError(module_name)

    monkeypatch.setattr(reg.importlib, "import_module", _fake_import)
    try:
        reg.get_model_class("catboost", prefer_gpu=False)
        assert False, "Expected strict mode to raise when Rust binding class is missing."
    except ImportError as exc:
        assert "Rust runtime model is required" in str(exc)
    finally:
        reg._CLASS_CACHE.clear()
