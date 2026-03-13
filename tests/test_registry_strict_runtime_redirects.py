from __future__ import annotations

import sys
import types

from forex_bot.models import registry as reg


def test_resolve_runtime_model_name_redirects_neuralforecast_in_strict_mode(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.delenv("FOREX_BOT_TREE_BACKEND", raising=False)
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    assert reg._resolve_runtime_model_name("patchtst") == "transformer"
    assert reg._resolve_runtime_model_name("timesnet") == "transformer"
    assert reg._resolve_runtime_model_name("tide_nf") == "tide"
    assert reg._resolve_runtime_model_name("nbeatsx_nf") == "nbeats"


def test_resolve_runtime_model_name_keeps_neuralforecast_when_not_strict(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "0")
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "auto")
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    assert reg._resolve_runtime_model_name("patchtst") == "patchtst"
    assert reg._resolve_runtime_model_name("tide_nf") == "tide_nf"


def test_get_model_class_redirects_patchtst_to_transformer_in_strict_mode(monkeypatch):
    reg._CLASS_CACHE.clear()
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)

    class _Transformer:
        pass

    def _fake_import(module_name: str, package: str | None = None):
        if module_name == ".transformers":
            return types.SimpleNamespace(TransformerExpertTorch=_Transformer)
        if module_name == ".transformer_nf":
            raise AssertionError("neuralforecast module should not be loaded in strict runtime mode")
        raise ImportError(module_name)

    monkeypatch.setattr(reg.importlib, "import_module", _fake_import)
    cls = reg.get_model_class("patchtst", prefer_gpu=False)
    assert cls is _Transformer
    reg._CLASS_CACHE.clear()


def test_get_model_class_uses_patchtst_module_when_not_strict(monkeypatch):
    reg._CLASS_CACHE.clear()
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "0")
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "auto")
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)

    class _Patch:
        pass

    def _fake_import(module_name: str, package: str | None = None):
        if module_name == ".transformer_nf":
            return types.SimpleNamespace(PatchTSTExpert=_Patch)
        if module_name == ".transformers":
            raise AssertionError("transformer fallback should not override patchtst when strict mode is off")
        raise ImportError(module_name)

    monkeypatch.setattr(reg.importlib, "import_module", _fake_import)
    cls = reg.get_model_class("patchtst", prefer_gpu=False)
    assert cls is _Patch
    reg._CLASS_CACHE.clear()


def test_use_rust_tree_models_strict_mode_ignores_python_override(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "1")
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "python")
    monkeypatch.delenv("FOREX_BOT_RUNTIME_PROFILE", raising=False)
    monkeypatch.setitem(sys.modules, "forex_bindings", types.SimpleNamespace(RustCatBoostExpert=object()))
    assert reg._use_rust_tree_models("catboost")
