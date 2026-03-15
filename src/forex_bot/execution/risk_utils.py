from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np

class ChallengePhase(Enum):
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    FUNDED = "funded"

@dataclass(slots=True)
class PropFirmRules:
    max_daily_loss_pct: float = 0.045
    max_total_loss_pct: float = 0.10
    profit_target_pct: float = 0.10
    min_trading_days: int = 5
    max_trading_days: int = 60
    max_lot_size: float = 10.0
    news_trading_allowed: bool = False
    weekend_holding: bool = False
    scaling_enabled: bool = True
    daily_dd_warning_pct: float = 0.035
    daily_dd_stop_trading_pct: float = 0.040
    daily_profit_lock_pct: float = 0.03
    max_trades_per_day: int = 15

@dataclass(slots=True)
class ChallengeRiskPreset:
    phase: str
    risk_per_trade: float
    max_risk_per_trade: float
    min_confidence_threshold: float
    max_trades_per_day: int
    daily_drawdown_limit: float
    total_drawdown_limit: float
    daily_profit_lock_pct: float
    monthly_profit_target_pct: float
    challenge_target_return_pct: float
    challenge_target_trading_days: int

def resolve_challenge_risk_preset(phase: str) -> ChallengeRiskPreset:
    raw = str(phase or "phase_1").strip().lower()
    if raw in {"phase2", "phase_2", "verification", "verify"}:
        return ChallengeRiskPreset(phase="phase_2", risk_per_trade=0.0025, max_risk_per_trade=0.0040,
            min_confidence_threshold=0.68, max_trades_per_day=3, daily_drawdown_limit=0.045,
            total_drawdown_limit=0.10, daily_profit_lock_pct=0.012, monthly_profit_target_pct=0.05,
            challenge_target_return_pct=0.05, challenge_target_trading_days=22)
    if raw in {"funded", "live"}:
        return ChallengeRiskPreset(phase="funded", risk_per_trade=0.0030, max_risk_per_trade=0.0050,
            min_confidence_threshold=0.65, max_trades_per_day=4, daily_drawdown_limit=0.045,
            total_drawdown_limit=0.10, daily_profit_lock_pct=0.0, monthly_profit_target_pct=0.06,
            challenge_target_return_pct=0.06, challenge_target_trading_days=22)
    return ChallengeRiskPreset(phase="phase_1", risk_per_trade=0.0030, max_risk_per_trade=0.0050,
        min_confidence_threshold=0.66, max_trades_per_day=3, daily_drawdown_limit=0.045,
        total_drawdown_limit=0.10, daily_profit_lock_pct=0.015, monthly_profit_target_pct=0.10,
        challenge_target_return_pct=0.10, challenge_target_trading_days=22)

class RevengeTradeDetector:
    """Lightweight sequence-based detector."""

    def __init__(self) -> None:
        self.recent_trades: list[dict] = []
        self.max_trades_tracked = 10

    def record_trade(self, entry_time: datetime, exit_time: datetime, pnl: float,
                     was_stopped: bool, size: float = 0.0, direction: int | None = None) -> None:
        self.recent_trades.append({"entry_time": entry_time, "exit_time": exit_time, "pnl": pnl,
                                   "was_stopped": was_stopped, "duration_minutes": (exit_time - entry_time).total_seconds() / 60,
                                   "size": size, "direction": direction})
        if len(self.recent_trades) > self.max_trades_tracked:
            self.recent_trades.pop(0)

    def is_revenge_trading(self, current_time: datetime) -> bool:
        if len(self.recent_trades) < 2:
            return False
        last = self.recent_trades[-1]
        if (current_time - last["exit_time"]).total_seconds() / 60 < 15 and last["pnl"] < 0:
            return True
        consec = sum(1 for t in reversed(self.recent_trades[-5:]) if t["pnl"] < 0)
        if consec >= 3 and not ((7 <= current_time.hour < 9) or (13 <= current_time.hour < 15)):
            return True
        if len(self.recent_trades) >= 3:
            r = self.recent_trades[-3:]
            mean_prev = float(np.mean([t.get("size", 0.0) for t in r[:-1]]))
            if mean_prev > 0 and r[-1].get("size", 0.0) > 1.5 * mean_prev and r[-2].get("pnl", 0) < 0:
                return True
            dirs = [t.get("direction") for t in r]
            pnls = [t.get("pnl", 0.0) for t in r]
            if all(d is not None for d in dirs) and pnls[-1] < 0 and pnls[-2] < 0 and dirs[-1] == dirs[-2] == dirs[-3]:
                return True
        return False
