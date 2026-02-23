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
