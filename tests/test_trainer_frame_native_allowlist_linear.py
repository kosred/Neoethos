from __future__ import annotations

from types import SimpleNamespace

from forex_bot.training.trainer import ModelTrainer


def test_get_enabled_models_keeps_allowlisted_linear_in_frame_native_runtime(monkeypatch):
    trainer = object.__new__(ModelTrainer)
    trainer.settings = SimpleNamespace(
        models=SimpleNamespace(
            ml_models=["xgboost", "elasticnet"],
            train_all_registered_models=False,
            ensure_linear_anchors=False,
            online_learners_enabled=False,
            use_neuroevolution=False,
            use_rl_agent=False,
            use_sac_agent=False,
            use_rllib_agent=False,
        )
    )

    monkeypatch.setenv("FOREX_BOT_TREE_BACKEND", "rust_strict")

    # Rust trees available for xgboost; linear models do not require rust bindings.
    monkeypatch.setattr(
        "forex_bot.models.registry._use_rust_tree_models",
        lambda name=None: bool(name == "xgboost"),
    )

    enabled = trainer._get_enabled_models()
    assert "xgboost" in enabled
    assert "elasticnet" in enabled
