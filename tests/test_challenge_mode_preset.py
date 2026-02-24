from __future__ import annotations

import sys
from pathlib import Path

from forex_bot.execution.risk import resolve_challenge_risk_preset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_prop_discovery import _hyperband_stages


def test_resolve_challenge_risk_preset_phase1_is_conservative() -> None:
    preset = resolve_challenge_risk_preset("phase_1")
    assert preset.phase == "phase_1"
    assert preset.risk_per_trade <= 0.004
    assert preset.max_risk_per_trade <= 0.006
    assert preset.min_confidence_threshold >= 0.64
    assert preset.max_trades_per_day <= 4
    assert preset.monthly_profit_target_pct >= 0.06
    assert preset.challenge_target_return_pct >= 0.10
    assert preset.challenge_target_trading_days <= 22


def test_resolve_challenge_risk_preset_funded_targets_5pct() -> None:
    preset = resolve_challenge_risk_preset("funded")
    assert preset.phase == "funded"
    assert preset.monthly_profit_target_pct >= 0.05
    assert preset.max_trades_per_day <= 5
    assert preset.challenge_target_return_pct >= 0.05


def test_hyperband_stages_monotonic_and_last_stage_promotes_disabled() -> None:
    stages = _hyperband_stages(
        base_population=300,
        base_generations=8,
        base_hours=1.0,
        pop_mults=[0.35, 0.7, 1.0],
        gen_mults=[0.5, 0.75, 1.0],
        hour_mults=[0.15, 0.4, 1.0],
        promote_min=[1, 1, 0],
    )
    assert len(stages) == 3
    assert int(stages[0]["population"]) < int(stages[1]["population"]) <= int(stages[2]["population"])
    assert int(stages[0]["generations"]) <= int(stages[1]["generations"]) <= int(stages[2]["generations"])
    assert float(stages[0]["max_hours"]) < float(stages[1]["max_hours"]) <= float(stages[2]["max_hours"])
    assert int(stages[-1]["promote_min"]) == 0
