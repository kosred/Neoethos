from __future__ import annotations

import types

from forex_bot.models import registry as reg


def test_get_model_class_prefers_rust_genetic_in_strict_mode(monkeypatch):
    reg._CLASS_CACHE.clear()
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE", "1")
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "rust_strict")

    class _RustGenetic:
        _model_cls = object()

    def _fake_import(module_name: str, package: str | None = None):
        if module_name == ".genetic_rust":
            return types.SimpleNamespace(RustGeneticExpert=_RustGenetic)
        if module_name == ".genetic":
            raise AssertionError("strict runtime must not fall back to legacy Python genetic")
        raise ImportError(module_name)

    monkeypatch.setattr(reg.importlib, "import_module", _fake_import)
    cls = reg.get_model_class("genetic", prefer_gpu=False)
    assert cls is _RustGenetic
    reg._CLASS_CACHE.clear()
