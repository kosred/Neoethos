from __future__ import annotations

from forex_bot.models import forecast_nf as fnf
from forex_bot.models import transformer_nf as tnf
from forex_bot.models.nbeats_gpu import NBeatsExpert
from forex_bot.models.tide_gpu import TiDEExpert
from forex_bot.models.transformers import TransformerExpertTorch


def test_forecast_aliases_route_to_native_experts() -> None:
    assert issubclass(fnf.TiDENFExpert, TiDEExpert)
    assert issubclass(fnf.NBEATSxNFExpert, NBeatsExpert)


def test_transformer_aliases_route_to_native_expert() -> None:
    assert issubclass(tnf.PatchTSTExpert, TransformerExpertTorch)
    assert issubclass(tnf.TimesNetExpert, TransformerExpertTorch)


def test_alias_construction_works_in_strict_mode(monkeypatch) -> None:
    monkeypatch.setenv("FOREX_BOT_PANDAS_FREE_STRICT", "1")
    assert isinstance(fnf.TiDENFExpert(), TiDEExpert)
    assert isinstance(fnf.NBEATSxNFExpert(), NBeatsExpert)
    assert isinstance(tnf.PatchTSTExpert(), TransformerExpertTorch)
    assert isinstance(tnf.TimesNetExpert(), TransformerExpertTorch)
