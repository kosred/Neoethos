from __future__ import annotations

import builtins
from types import SimpleNamespace

from forex_bot.models import registry as reg
from forex_bot.training.trainer import ModelTrainer


def _trainer_with_models(models: list[str]) -> ModelTrainer:
    trainer = object.__new__(ModelTrainer)
    trainer.settings = SimpleNamespace(
        models=SimpleNamespace(
            ml_models=list(models),
            train_all_registered_models=False,
            ensure_linear_anchors=False,
            online_learners_enabled=False,
            use_neuroevolution=False,
            use_rl_agent=False,
            use_sac_agent=False,
            use_rllib_agent=False,
        )
    )
    return trainer


def test_get_enabled_models_frame_native_does_not_import_neuralforecast(monkeypatch):
    trainer = _trainer_with_models(["xgboost"])
    monkeypatch.setattr(reg, "_use_rust_tree_models", lambda _name=None: True)

    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("models.forecast_nf") or name.endswith("models.transformer_nf"):
            raise AssertionError("neuralforecast wrappers must not be imported in pandas-free mode")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    enabled = trainer._get_enabled_models()
    assert enabled == ["xgboost"]


def test_get_enabled_models_without_nf_candidates_skips_nf_imports(monkeypatch):
    trainer = _trainer_with_models(["xgboost"])
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "0")
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "auto")

    calls: list[str] = []
    real_import = builtins.__import__

    def _tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        calls.append(str(name))
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _tracking_import)
    enabled = trainer._get_enabled_models()
    assert "xgboost" in enabled
    assert not any(name.endswith("models.forecast_nf") for name in calls)
    assert not any(name.endswith("models.transformer_nf") for name in calls)


def test_get_enabled_models_applies_runtime_redirect_before_filtering(monkeypatch):
    trainer = _trainer_with_models(["patchtst", "transformer"])
    monkeypatch.setenv("FOREX_BOT_RUST_ONLY", "0")
    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "auto")

    monkeypatch.setattr(
        reg,
        "_resolve_runtime_model_name",
        lambda name: "transformer" if name == "patchtst" else name,
    )

    enabled = trainer._get_enabled_models()
    assert enabled.count("transformer") == 0
    assert "patchtst" not in enabled
