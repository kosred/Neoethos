from __future__ import annotations

from forex_bot.core.config import Settings
from forex_bot.data.loader import DataLoader
from forex_bot.features.pipeline import FeatureEngineer


def test_loader_and_feature_engine_resolve_full_multiresolution_set(monkeypatch):
    monkeypatch.delenv("FOREX_BOT_USE_ALL_TIMEFRAMES", raising=False)
    settings = Settings()
    settings.system.base_timeframe = "M1"
    settings.system.multi_resolution_enabled = True
    settings.system.multi_resolution_timeframes = ["M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1", "MN1"]
    settings.system.higher_timeframes = ["M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1", "MN1"]
    settings.system.required_timeframes = ["M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1", "MN1"]

    loader = DataLoader(settings)
    fe = FeatureEngineer(settings)

    expected = ["M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1", "MN1"]
    assert loader._timeframes() == expected
    assert fe._resolved_timeframes("M1") == expected


def test_use_all_timeframes_env_expands_loader_and_feature_engine(monkeypatch):
    monkeypatch.setenv("FOREX_BOT_USE_ALL_TIMEFRAMES", "1")
    settings = Settings()
    settings.system.base_timeframe = "M1"
    settings.system.multi_resolution_enabled = False
    settings.system.multi_resolution_timeframes = []
    settings.system.higher_timeframes = []
    settings.system.required_timeframes = []

    loader = DataLoader(settings)
    fe = FeatureEngineer(settings)

    # Should include full canonical list when the env flag is enabled.
    assert "MN1" in loader._timeframes()
    assert "MN1" in fe._resolved_timeframes("M1")
    assert len(loader._timeframes()) >= 21
    assert len(fe._resolved_timeframes("M1")) >= 21
