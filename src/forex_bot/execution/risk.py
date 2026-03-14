import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows doesn't have fcntl

from ..core.config import Settings
from ..core.storage import RiskLedger
from .meta_controller import MetaController, PropMetaState

logger = logging.getLogger(__name__)

MIN_BREAKEVEN_PROBABILITY = 0.45
RISK_STATE_FILE = Path("cache") / "risk_state.json"
_RUST_RISK_BACKEND_OK: bool | None = None
_RUST_RISK_WARNED_UNAVAILABLE = False


def _rust_risk_backend_available(*, force_log: bool = False) -> bool:
    global _RUST_RISK_BACKEND_OK, _RUST_RISK_WARNED_UNAVAILABLE
    if _RUST_RISK_BACKEND_OK is None:
        try:
            import forex_bindings  # type: ignore

            _RUST_RISK_BACKEND_OK = hasattr(forex_bindings, "compute_position_size_lots")
        except Exception:
            _RUST_RISK_BACKEND_OK = False
    if force_log and not _RUST_RISK_BACKEND_OK and not _RUST_RISK_WARNED_UNAVAILABLE:
        logger.warning(
            "Rust risk backend requested but forex_bindings.compute_position_size_lots is unavailable."
        )
        _RUST_RISK_WARNED_UNAVAILABLE = True
    return bool(_RUST_RISK_BACKEND_OK)


def _disable_rust_risk_backend() -> None:
    global _RUST_RISK_BACKEND_OK
    _RUST_RISK_BACKEND_OK = False


class ChallengePhase(Enum):
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    FUNDED = "funded"


@dataclass(slots=True)
class PropFirmRules:
    max_daily_loss_pct: float = 0.045  # 4.5% daily loss limit (STRICT)
    max_total_loss_pct: float = 0.10
    profit_target_pct: float = 0.10
    min_trading_days: int = 5
    max_trading_days: int = 60
    max_lot_size: float = 10.0
    news_trading_allowed: bool = False
    weekend_holding: bool = False
    scaling_enabled: bool = True
    daily_dd_warning_pct: float = 0.035  # 3.5% warning threshold
    daily_dd_stop_trading_pct: float = 0.040  # 4.0% stop trading buffer
    daily_profit_lock_pct: float = 0.03  # lock profits if hit
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
        return ChallengeRiskPreset(
            phase="phase_2",
            risk_per_trade=0.0025,
            max_risk_per_trade=0.0040,
            min_confidence_threshold=0.68,
            max_trades_per_day=3,
            daily_drawdown_limit=0.045,
            total_drawdown_limit=0.10,
            daily_profit_lock_pct=0.012,
            monthly_profit_target_pct=0.05,
            challenge_target_return_pct=0.05,
            challenge_target_trading_days=22,
        )
    if raw in {"funded", "live"}:
        return ChallengeRiskPreset(
            phase="funded",
            risk_per_trade=0.0030,
            max_risk_per_trade=0.0050,
            min_confidence_threshold=0.65,
            max_trades_per_day=4,
            daily_drawdown_limit=0.045,
            total_drawdown_limit=0.10,
            daily_profit_lock_pct=0.0,
            monthly_profit_target_pct=0.06,
            challenge_target_return_pct=0.06,
            challenge_target_trading_days=22,
        )
    return ChallengeRiskPreset(
        phase="phase_1",
        risk_per_trade=0.0030,
        max_risk_per_trade=0.0050,
        min_confidence_threshold=0.66,
        max_trades_per_day=3,
        daily_drawdown_limit=0.045,
        total_drawdown_limit=0.10,
        daily_profit_lock_pct=0.015,
        monthly_profit_target_pct=0.10,
        challenge_target_return_pct=0.10,
        challenge_target_trading_days=22,
    )


class RevengeTradeDetector:
    def __init__(self):
        self.recent_trades = []
        self.max_trades_tracked = 10

    def record_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        was_stopped: bool,
        size: float = 0.0,
        direction: int | None = None,
    ) -> None:
        trade_data = {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl": pnl,
            "was_stopped": was_stopped,
            "duration_minutes": (exit_time - entry_time).total_seconds() / 60,
            "size": size,
            "direction": direction,
        }
        self.recent_trades.append(trade_data)
        if len(self.recent_trades) > self.max_trades_tracked:
            self.recent_trades.pop(0)

    def is_revenge_trading(self, current_time: datetime) -> bool:
        if len(self.recent_trades) < 2:
            return False
        last_trade = self.recent_trades[-1]
        time_since_last = (current_time - last_trade["exit_time"]).total_seconds() / 60

        if time_since_last < 15 and last_trade["pnl"] < 0:
            return True

        consecutive_losses = 0
        for trade in reversed(self.recent_trades[-5:]):
            if trade["pnl"] < 0:
                consecutive_losses += 1
            else:
                break
        if consecutive_losses >= 3:
            hour = current_time.hour
            optimal_times = (7 <= hour < 9) or (13 <= hour < 15)
            if not optimal_times:
                return True

        if len(self.recent_trades) >= 3:
            recent = self.recent_trades[-3:]
            sizes = [t.get("size", 0.0) for t in recent[:-1]]
            last_size = recent[-1].get("size", 0.0)
            # Need at least 2 previous trades to check revenge pattern
            if sizes and len(recent) >= 2:
                mean_prev = np.mean(sizes)
                if mean_prev > 0 and last_size > 1.5 * mean_prev and recent[-2].get("pnl", 0) < 0:
                    return True

        if len(self.recent_trades) >= 3:
            dirs = [t.get("direction") for t in self.recent_trades[-3:]]
            pnls = [t.get("pnl", 0.0) for t in self.recent_trades[-3:]]
            if all(d is not None for d in dirs) and pnls[-1] < 0 and pnls[-2] < 0:
                if dirs[-1] == dirs[-2] == dirs[-3]:
                    return True
        if len(self.recent_trades) >= 2:
            last = self.recent_trades[-1]
            prev = self.recent_trades[-2]
            gap_min = (last["entry_time"] - prev["exit_time"]).total_seconds() / 60.0
            if gap_min < 30 and last.get("pnl", 0) < 0 and prev.get("pnl", 0) < 0:
                if last.get("direction") is not None and last.get("direction") == prev.get("direction"):
                    return True
        return False


class RiskManager:
    MIN_LOT = 0.01

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.symbol = settings.system.symbol or "GLOBAL"
        self.state_file = Path("cache") / f"risk_state_{self.symbol}.json"
        self.challenge_mode = bool(
            getattr(settings.risk, "challenge_mode", False)
            or str(os.environ.get("FOREX_BOT_CHALLENGE_MODE", "")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.challenge_phase = str(getattr(settings.risk, "challenge_phase", "phase_1") or "phase_1")
        if self.challenge_mode:
            self._apply_challenge_mode_preset()

        session_tz = str(getattr(settings.system, "session_timezone", "UTC") or "UTC")
        try:
            self._session_tz = ZoneInfo(session_tz)
        except Exception:
            logger.warning("Invalid session timezone '%s'; falling back to UTC.", session_tz)
            self._session_tz = ZoneInfo("UTC")

        try:
            self._session_start = self._parse_time(str(getattr(settings.system, "trading_session_start", "00:00")))
        except Exception:
            self._session_start = time(0, 0)
        try:
            self._session_end = self._parse_time(str(getattr(settings.system, "trading_session_end", "23:59")))
        except Exception:
            self._session_end = time(23, 59)

        defaults = PropFirmRules()
        daily_stop = float(
            getattr(settings.risk, "daily_drawdown_limit", defaults.daily_dd_stop_trading_pct)
            or defaults.daily_dd_stop_trading_pct
        )
        daily_stop = max(0.001, daily_stop)
        daily_warn = min(defaults.daily_dd_warning_pct, daily_stop * 0.9)
        daily_warn = max(0.001, min(daily_warn, daily_stop - 1e-4))
        max_trades = int(getattr(settings.risk, "max_trades_per_day", defaults.max_trades_per_day) or 0)
        max_trades = max(1, max_trades)

        self.prop_rules = PropFirmRules(
            max_daily_loss_pct=daily_stop,
            max_total_loss_pct=float(
                getattr(settings.risk, "total_drawdown_limit", defaults.max_total_loss_pct)
                or defaults.max_total_loss_pct
            ),
            daily_dd_warning_pct=daily_warn,
            daily_dd_stop_trading_pct=daily_stop,
            max_trades_per_day=max_trades,
        )
        self._base_prop_max_trades = self.prop_rules.max_trades_per_day
        self._today_max_trades = self.prop_rules.max_trades_per_day

        self.risk_ledger = RiskLedger(max_events=int(getattr(settings.system, "risk_ledger_max_events", 1000) or 1000))
        self.revenge_trading_detector = RevengeTradeDetector()
        self.meta_controller = MetaController(
            max_daily_dd=self.prop_rules.max_daily_loss_pct,
            safety_buffer=max(0.0, self.prop_rules.daily_dd_warning_pct * 0.7),
            base_risk_per_trade=float(getattr(settings.risk, "base_risk_per_trade", settings.risk.risk_per_trade)),
            base_confidence=float(getattr(settings.risk, "min_confidence_threshold", 0.55) or 0.55),
            settings=settings,
            silent=True,
        )

        initial_balance = float(getattr(settings.risk, "initial_balance", 10000.0) or 10000.0)
        now_date = datetime.now(self._session_tz).date()

        self._last_session_date: date | None = None
        self._kill_window_until: datetime | None = None
        self._news_state: dict[str, Any] = {}
        self._spread_state: dict[str, float] = {
            "current_spread": 1.0,
            "current_slippage": 1.0,
            "spread_baseline": 1.0,
            "slippage_baseline": 1.0,
        }

        self.day_start_equity = initial_balance
        self.day_peak_equity = initial_balance
        self.month_start_date = now_date.replace(day=1)
        self.month_start_equity = initial_balance
        self.total_peak_equity = initial_balance

        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.session_trades = 0
        self.session_trade_counts: dict[str, int] = {}
        self.consecutive_losses = 0
        self.circuit_breaker_triggered = False

        self.monthly_return_pct = 0.0
        self.monthly_profit_target_pct = float(getattr(settings.risk, "monthly_profit_target_pct", 0.04) or 0.04)
        self.monthly_target_hit = False
        self.phase_trade_days: set[date] = set()
        self.challenge_start_date: date | None = now_date if self.challenge_mode else None
        self.challenge_start_equity = initial_balance if self.challenge_mode else 0.0
        self.challenge_return_pct = 0.0
        self.challenge_target_hit = False
        self.challenge_target_return_pct = float(getattr(settings.risk, "challenge_target_return_pct", 0.10) or 0.10)
        self.challenge_target_trading_days = int(getattr(settings.risk, "challenge_target_trading_days", 44) or 44)
        self.challenge_target_return_pct = max(0.0, self.challenge_target_return_pct)
        self.challenge_target_trading_days = max(1, self.challenge_target_trading_days)

        self.recovery_mode = False
        self.recovery_conf_boost = 0.10
        self.recovery_min_win_prob = max(
            MIN_BREAKEVEN_PROBABILITY,
            float(getattr(settings.risk, "high_quality_confidence", 0.65) or 0.65),
        )
        max_risk = float(getattr(settings.risk, "max_risk_per_trade", 0.03) or 0.03)
        self.recovery_risk_cap = max(0.0, min(max_risk, max_risk * 0.5))
        self.recovery_max_trades = max(1, self._base_prop_max_trades // 2)

        self.reflection_mode = False
        self.reflection_cooldown_until: datetime | None = None
        self.rolling_outcomes: deque[int] = deque(maxlen=20)
        self.last_risk_mult = 1.0

        self.load_state()
        self.initialize_session()

    @staticmethod
    def _session_bucket_utc(ts: datetime) -> str:
        hour = int(ts.astimezone(ZoneInfo("UTC")).hour)
        if 0 <= hour < 7:
            return "asia"
        if 7 <= hour < 13:
            return "london"
        if 13 <= hour < 21:
            return "newyork"
        return "offhours"

    @staticmethod
    def _in_hour_window(hour: int, start: int, end: int) -> bool:
        s = int(start) % 24
        e = int(end) % 24
        if s == e:
            return True
        if s < e:
            return s <= hour < e
        return (hour >= s) or (hour < e)

    @staticmethod
    def _business_days_between(start: date, end: date) -> int:
        if end < start:
            start, end = end, start
        span = (end - start).days + 1
        weeks, rem = divmod(max(0, span), 7)
        weekdays = weeks * 5
        for i in range(rem):
            if (start.weekday() + i) % 7 < 5:
                weekdays += 1
        return max(0, weekdays)

    def _drawdown_state(self, equity: float) -> tuple[float, float, float, float]:
        daily_dd_pct = (self.day_start_equity - equity) / self.day_start_equity if self.day_start_equity > 0 else 0.0
        intraday_dd_pct = (self.day_peak_equity - equity) / self.day_peak_equity if self.day_peak_equity > 0 else 0.0
        dd_used = max(0.0, max(daily_dd_pct, intraday_dd_pct))
        dd_limit = max(float(self.prop_rules.daily_dd_stop_trading_pct), 1e-9)
        return float(daily_dd_pct), float(intraday_dd_pct), float(dd_used), float(dd_limit)

    def _challenge_progress_multiplier(self, equity: float, now: datetime, daily_dd_pct: float) -> float:
        if not self.challenge_mode or self.challenge_start_equity <= 0:
            return 1.0
        start = self.challenge_start_date or now.date()
        elapsed_days = max(1, self._business_days_between(start, now.date()))
        target_days = max(1, int(getattr(self.settings.risk, "challenge_target_trading_days", self.challenge_target_trading_days) or self.challenge_target_trading_days))
        progress = min(1.0, max(1.0 / float(target_days), float(elapsed_days) / float(target_days)))
        target_return = float(getattr(self.settings.risk, "challenge_target_return_pct", self.challenge_target_return_pct) or self.challenge_target_return_pct) * progress
        current_return = (equity - self.challenge_start_equity) / self.challenge_start_equity
        tol = float(getattr(self.settings.risk, "challenge_progress_tolerance_pct", 0.01) or 0.01)
        boost = float(getattr(self.settings.risk, "challenge_progress_boost_mult", 1.08) or 1.08)
        reduce = float(getattr(self.settings.risk, "challenge_progress_reduce_mult", 0.85) or 0.85)
        if current_return < (target_return - tol) and daily_dd_pct < max(0.005, self.prop_rules.daily_dd_warning_pct * 0.5):
            return float(max(0.5, min(1.5, boost)))
        if current_return > (target_return + tol):
            return float(max(0.3, min(1.0, reduce)))
        return 1.0

    def _apply_challenge_mode_preset(self) -> None:
        preset = resolve_challenge_risk_preset(self.challenge_phase)
        r = self.settings.risk

        try:
            current_risk = float(getattr(r, "risk_per_trade", preset.risk_per_trade) or preset.risk_per_trade)
        except Exception:
            current_risk = preset.risk_per_trade
        r.risk_per_trade = max(0.0001, min(current_risk, preset.risk_per_trade))

        try:
            current_base_risk = float(getattr(r, "base_risk_per_trade", r.risk_per_trade) or r.risk_per_trade)
        except Exception:
            current_base_risk = r.risk_per_trade
        r.base_risk_per_trade = max(0.0001, min(current_base_risk, preset.risk_per_trade))

        try:
            current_max_risk = float(
                getattr(r, "max_risk_per_trade", preset.max_risk_per_trade) or preset.max_risk_per_trade
            )
        except Exception:
            current_max_risk = preset.max_risk_per_trade
        r.max_risk_per_trade = max(r.risk_per_trade, min(current_max_risk, preset.max_risk_per_trade))

        try:
            current_conf = float(
                getattr(r, "min_confidence_threshold", preset.min_confidence_threshold) or preset.min_confidence_threshold
            )
        except Exception:
            current_conf = preset.min_confidence_threshold
        r.min_confidence_threshold = min(0.90, max(current_conf, preset.min_confidence_threshold))
        r.high_quality_confidence = min(0.95, max(float(getattr(r, "high_quality_confidence", 0.65) or 0.65), r.min_confidence_threshold + 0.05))

        try:
            current_trades = int(getattr(r, "max_trades_per_day", preset.max_trades_per_day) or preset.max_trades_per_day)
        except Exception:
            current_trades = preset.max_trades_per_day
        if current_trades <= 0:
            current_trades = preset.max_trades_per_day
        r.max_trades_per_day = max(1, min(current_trades, preset.max_trades_per_day))

        try:
            current_daily_dd = float(getattr(r, "daily_drawdown_limit", preset.daily_drawdown_limit) or preset.daily_drawdown_limit)
        except Exception:
            current_daily_dd = preset.daily_drawdown_limit
        r.daily_drawdown_limit = max(0.001, min(current_daily_dd, preset.daily_drawdown_limit))

        try:
            current_total_dd = float(getattr(r, "total_drawdown_limit", preset.total_drawdown_limit) or preset.total_drawdown_limit)
        except Exception:
            current_total_dd = preset.total_drawdown_limit
        r.total_drawdown_limit = max(0.01, min(current_total_dd, preset.total_drawdown_limit))

        try:
            current_monthly_target = float(
                getattr(r, "monthly_profit_target_pct", preset.monthly_profit_target_pct) or preset.monthly_profit_target_pct
            )
        except Exception:
            current_monthly_target = preset.monthly_profit_target_pct
        r.monthly_profit_target_pct = max(current_monthly_target, preset.monthly_profit_target_pct)

        try:
            current_phase_target = float(
                getattr(r, "challenge_target_return_pct", preset.challenge_target_return_pct) or preset.challenge_target_return_pct
            )
        except Exception:
            current_phase_target = preset.challenge_target_return_pct
        r.challenge_target_return_pct = max(0.0, max(current_phase_target, preset.challenge_target_return_pct))

        try:
            current_target_days = int(
                getattr(r, "challenge_target_trading_days", preset.challenge_target_trading_days) or preset.challenge_target_trading_days
            )
        except Exception:
            current_target_days = preset.challenge_target_trading_days
        current_target_days = max(1, current_target_days)
        if preset.phase in {"phase_1", "phase_2"}:
            r.challenge_target_trading_days = min(current_target_days, preset.challenge_target_trading_days)
        else:
            r.challenge_target_trading_days = current_target_days

        try:
            current_daily_stop = float(getattr(r, "daily_profit_stop_pct", 0.0) or 0.0)
        except Exception:
            current_daily_stop = 0.0
        if preset.daily_profit_lock_pct > 0.0:
            if current_daily_stop <= 0.0:
                r.daily_profit_stop_pct = preset.daily_profit_lock_pct
            else:
                r.daily_profit_stop_pct = min(current_daily_stop, preset.daily_profit_lock_pct)

        try:
            current_hq_risk = float(getattr(r, "high_quality_risk_pct", r.max_risk_per_trade) or r.max_risk_per_trade)
        except Exception:
            current_hq_risk = r.max_risk_per_trade
        r.high_quality_risk_pct = min(current_hq_risk, r.max_risk_per_trade)

        logger.info(
            "Challenge mode preset applied: phase=%s risk=%.3f%% max_risk=%.3f%% min_conf=%.2f max_trades=%d month_target=%.2f%% phase_target=%.2f%%/%sd",
            preset.phase,
            100.0 * float(r.risk_per_trade),
            100.0 * float(r.max_risk_per_trade),
            float(r.min_confidence_threshold),
            int(r.max_trades_per_day),
            100.0 * float(r.monthly_profit_target_pct),
            100.0 * float(getattr(r, "challenge_target_return_pct", 0.0) or 0.0),
            int(getattr(r, "challenge_target_trading_days", 0) or 0),
        )

    def save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "date": self._last_session_date.isoformat() if self._last_session_date else None,
                "month_start_date": self.month_start_date.isoformat() if self.month_start_date else None,
                "month_start_equity": self.month_start_equity,
                "day_start_equity": self.day_start_equity,
                "day_peak_equity": self.day_peak_equity,
                "daily_loss": self.daily_loss,
                "daily_profit": self.daily_profit,
                "session_trades": self.session_trades,
                "session_trade_counts": self.session_trade_counts,
                "consecutive_losses": self.consecutive_losses,
                "total_peak_equity": self.total_peak_equity,
                "circuit_breaker_triggered": self.circuit_breaker_triggered,
                "recovery_mode": self.recovery_mode,
                "monthly_target_hit": self.monthly_target_hit,
                "challenge_start_date": self.challenge_start_date.isoformat() if self.challenge_start_date else None,
                "challenge_start_equity": self.challenge_start_equity,
                "challenge_return_pct": self.challenge_return_pct,
                "challenge_target_hit": self.challenge_target_hit,
                "reflection_mode": self.reflection_mode,
                "reflection_cooldown_until": (
                    self.reflection_cooldown_until.isoformat() if self.reflection_cooldown_until else None
                ),
            }

            with open(self.state_file, 'w') as f:
                # HPC FIX: Exclusive File Lock (Unix only)
                if fcntl is not None and sys.platform != "win32":
                    fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(state, f)
                if fcntl is not None and sys.platform != "win32":
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")

    def load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            state = json.loads(self.state_file.read_text())
            saved_date_str = state.get("date")
            if not saved_date_str:
                return

            saved_date = date.fromisoformat(saved_date_str)
            now_date = datetime.now(self._session_tz).date()
            fallback_balance = float(getattr(self.settings.risk, "initial_balance", 0.0) or 0.0)

            # Restore persistent fields even across day restarts.
            month_saved = state.get("month_start_date")
            if month_saved:
                try:
                    self.month_start_date = date.fromisoformat(month_saved)
                except Exception:
                    self.month_start_date = now_date.replace(day=1)
            self.month_start_equity = float(state.get("month_start_equity", fallback_balance))
            self.total_peak_equity = float(state.get("total_peak_equity", max(fallback_balance, self.month_start_equity)))
            self.monthly_target_hit = bool(state.get("monthly_target_hit", False))
            self.challenge_start_equity = float(state.get("challenge_start_equity", self.challenge_start_equity or fallback_balance))
            self.challenge_return_pct = float(state.get("challenge_return_pct", self.challenge_return_pct))
            self.challenge_target_hit = bool(state.get("challenge_target_hit", self.challenge_target_hit))
            ch_start_raw = state.get("challenge_start_date")
            if ch_start_raw:
                try:
                    self.challenge_start_date = date.fromisoformat(str(ch_start_raw))
                except Exception:
                    self.challenge_start_date = self.challenge_start_date or now_date

            cooldown_raw = state.get("reflection_cooldown_until")
            if cooldown_raw:
                try:
                    self.reflection_cooldown_until = datetime.fromisoformat(str(cooldown_raw))
                except Exception:
                    self.reflection_cooldown_until = None
            else:
                self.reflection_cooldown_until = None
            self.reflection_mode = bool(state.get("reflection_mode", False))

            if saved_date == now_date:
                self._last_session_date = saved_date
                self.day_start_equity = float(state.get("day_start_equity", fallback_balance))
                self.day_peak_equity = float(state.get("day_peak_equity", max(self.total_peak_equity, self.day_start_equity)))
                self.daily_loss = float(state.get("daily_loss", 0.0))
                self.daily_profit = float(state.get("daily_profit", 0.0))
                self.session_trades = int(state.get("session_trades", 0))
                raw_counts = state.get("session_trade_counts", {})
                if isinstance(raw_counts, dict):
                    self.session_trade_counts = {str(k): int(v) for k, v in raw_counts.items()}
                else:
                    self.session_trade_counts = {}
                self.consecutive_losses = int(state.get("consecutive_losses", 0))
                self.circuit_breaker_triggered = bool(state.get("circuit_breaker_triggered", False))
                self.recovery_mode = bool(state.get("recovery_mode", False))
                logger.info(f"Restored risk state for {self.symbol} ({saved_date})")
            else:
                # Keep persistent state but force a fresh day reset from live equity on first check.
                self._last_session_date = None
                self.daily_loss = 0.0
                self.daily_profit = 0.0
                self.session_trades = 0
                self.session_trade_counts = {}
                self.consecutive_losses = 0
                self.circuit_breaker_triggered = False
                self.recovery_mode = False
                logger.info(
                    f"Restored persistent risk state for {self.symbol} from {saved_date}; daily counters reset for {now_date}"
                )
        except Exception as e:
            logger.warning(f"Failed to load risk state for {self.symbol}: {e}")

    def _detect_challenge_phase(self, balance: float) -> ChallengePhase:
        prop_firm_balances = [5000, 10000, 25000, 50000, 100000, 200000, 400000]
        if balance in prop_firm_balances:
            return ChallengePhase.PHASE_1
        return ChallengePhase.FUNDED

    def _parse_time(self, raw: str) -> time:
        hour, minute = raw.split(":")
        return time(int(hour), int(minute))

    def initialize_session(self) -> None:
        now_date = datetime.now(self._session_tz).date()
        if self._last_session_date != now_date:
            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.session_trades = 0
            self.session_trade_counts = {}
            self.consecutive_losses = 0
            # Defer day start equity initialization until first live equity check.
            self._last_session_date = None
            self._kill_window_until = None
            logger.info("Session initialized (fresh)", extra={"event_id": "risk_session_reset"})
        else:
            logger.info("Session initialized (continuing)", extra={"event_id": "risk_session_continue"})

    def _ensure_day(self, equity: float, now: datetime) -> None:
        """Reset daily counters if a new session day starts."""
        if self._last_session_date is None or now.date() != self._last_session_date:
            self._last_session_date = now.date()
            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.session_trades = 0
            self.session_trade_counts = {}
            self.consecutive_losses = 0
            self.day_start_equity = equity
            self.day_peak_equity = equity
            self.circuit_breaker_triggered = False
            self.recovery_mode = False
            self.prop_rules.max_trades_per_day = self._base_prop_max_trades
            logger.info(f"New trading day started. Day Start Equity: {self.day_start_equity:.2f}")
            self.save_state()

    def _ensure_month(self, equity: float, now: datetime) -> None:
        """Reset month tracking when calendar month rolls or first run."""
        month_start = now.date().replace(day=1)
        if self.month_start_date != month_start or self.month_start_equity <= 0:
            self.month_start_date = month_start
            self.month_start_equity = equity
            self.monthly_target_hit = False
            self.risk_ledger.record("MONTH_RESET", f"New month started; base equity={equity:.2f}", severity="info")
            self.save_state()

    def _update_monthly_metrics(self, equity: float) -> None:
        """Track monthly return based on live MT5 equity."""
        if self.month_start_equity <= 0:
            return
        self.monthly_return_pct = (equity - self.month_start_equity) / self.month_start_equity
        if not self.monthly_target_hit and self.monthly_return_pct >= self.monthly_profit_target_pct:
            self.monthly_target_hit = True
            self.risk_ledger.record(
                "MONTH_TARGET_HIT",
                f"Monthly return hit {self.monthly_return_pct:.2%} (target {self.monthly_profit_target_pct:.2%})",
                severity="info",
            )
            self.save_state()

    def _update_challenge_metrics(self, equity: float, now: datetime) -> None:
        if not self.challenge_mode:
            return
        if self.challenge_start_equity <= 0:
            self.challenge_start_equity = float(max(1.0, equity))
        if self.challenge_start_date is None:
            self.challenge_start_date = now.date()
        self.challenge_return_pct = (equity - self.challenge_start_equity) / max(self.challenge_start_equity, 1e-9)
        target = float(getattr(self.settings.risk, "challenge_target_return_pct", self.challenge_target_return_pct) or self.challenge_target_return_pct)
        if not self.challenge_target_hit and target > 0.0 and self.challenge_return_pct >= target:
            self.challenge_target_hit = True
            self.risk_ledger.record(
                "CHALLENGE_TARGET_HIT",
                f"Phase return hit {self.challenge_return_pct:.2%} (target {target:.2%})",
                severity="info",
            )
            self.save_state()

    def _update_recovery_state(self, equity: float) -> None:
        """Switch into/out of recovery mode based on intraday drawdown and equity recovery."""
        if self.day_start_equity <= 0:
            return
        daily_dd_pct = (self.day_start_equity - equity) / self.day_start_equity

        # Enter recovery mode if DD hits warning level
        if daily_dd_pct >= self.prop_rules.daily_dd_warning_pct:
            if not self.recovery_mode:
                self.recovery_mode = True
                self.risk_ledger.record(
                    "RECOVERY_ON", f"Entering recovery mode at DD {daily_dd_pct:.2%}", severity="warning"
                )
        # Exit recovery mode if equity recovers to within 0.5% of break-even OR DD drops below half the warning level
        elif self.recovery_mode:
            recovery_threshold_pct = 0.005  # 0.5% from break-even
            half_warning = self.prop_rules.daily_dd_warning_pct / 2.0

            if equity >= (self.day_start_equity * (1.0 - recovery_threshold_pct)) or daily_dd_pct <= half_warning:
                self.recovery_mode = False
                self.risk_ledger.record(
                    "RECOVERY_OFF",
                    f"Exiting recovery mode (equity={equity:.2f}, DD={daily_dd_pct:.2%})",
                    severity="info",
                )

    def is_trading_session(self) -> bool:
        now = datetime.now(self._session_tz)

        if now.weekday() >= 5:
            return False

        current = now.time()
        if self._session_end < self._session_start:
            in_session = current >= self._session_start or current <= self._session_end
        else:
            in_session = self._session_start <= current <= self._session_end
        return in_session

    def update_news_state(self, policy_flags: dict[str, Any], now: datetime | None = None) -> None:
        if not policy_flags:
            return
        self._news_state.update(policy_flags)
        now = now or datetime.now(tz=self._session_tz)
        if policy_flags.get("tier1_nearby"):
            self._kill_window_until = now + timedelta(minutes=self.settings.news.news_kill_window_min)
            self.risk_ledger.record("NEWS_KILL", "Tier-1 news kill window active", severity="warning")

    def update_spread_state(
        self,
        *,
        live_spread: float,
        live_slippage: float,
        baseline_spread: float,
        baseline_slippage: float,
    ) -> None:
        self._spread_state.update(
            {
                "current_spread": live_spread,
                "current_slippage": live_slippage,
                "spread_baseline": baseline_spread,
                "slippage_baseline": baseline_slippage,
            }
        )

    def check_trade_allowed(
        self,
        equity: float,
        confidence: float,
        timestamp: datetime,
        *,
        market_volatility: float | None = None,
        ensemble_disagreement: float | None = None,
    ) -> tuple[bool, str]:
        self._ensure_day(equity, timestamp)
        self._ensure_month(equity, timestamp)
        self._update_monthly_metrics(equity)
        self._update_challenge_metrics(equity, timestamp)
        self._update_recovery_state(equity)

        if self.challenge_mode and self.challenge_target_hit:
            return False, "Challenge target reached"

        if (not self.challenge_mode) and self.monthly_target_hit:
            return False, "Monthly profit target reached"

        if self.reflection_mode:
            if self.reflection_cooldown_until and timestamp < self.reflection_cooldown_until:
                return False, "reflection_mode_cooldown"
            else:
                self.reflection_mode = False
                self.rolling_outcomes.clear()  # Reset stats for fresh start
                logger.info("Reflection mode cooldown ended. Resuming trading.")

        if equity > self.total_peak_equity:
            self.total_peak_equity = equity
            self.risk_ledger.record("EQUITY_PEAK", f"New equity peak: {equity:.2f}")

        # FIX: Track intraday peak for strict prop firm rules (Intraday Trailing DD)
        self.day_peak_equity = max(self.day_peak_equity, equity)

        if self.circuit_breaker_triggered:
            return False, "Circuit breaker active"

        if not self.is_trading_session():
            return False, "Outside trading session"

        # UTC night block: avoid low-liquidity chop unless realized volatility is high enough.
        if bool(getattr(self.settings.risk, "block_night_session", True)):
            utc_hour = int(timestamp.astimezone(ZoneInfo("UTC")).hour)
            start_h = int(getattr(self.settings.risk, "night_block_start_utc", 0) or 0)
            end_h = int(getattr(self.settings.risk, "night_block_end_utc", 6) or 6)
            if self._in_hour_window(utc_hour, start_h, end_h):
                min_vol = float(getattr(self.settings.risk, "night_min_volatility", 0.0008) or 0.0008)
                cur_vol = float(market_volatility or 0.0)
                if cur_vol < min_vol:
                    return False, f"Night session blocked (vol={cur_vol:.5f} < {min_vol:.5f})"

        if self._kill_window_until and timestamp < self._kill_window_until:
            return False, "News kill window active"

        if self.revenge_trading_detector.is_revenge_trading(timestamp):
            return False, "Revenge trading detected"

        daily_dd_pct, intraday_dd_pct, dd_used, dd_limit = self._drawdown_state(equity)

        if daily_dd_pct >= self.prop_rules.daily_dd_stop_trading_pct:
            self.circuit_breaker_triggered = True
            return False, f"Daily drawdown limit reached ({daily_dd_pct:.2%})"

        if intraday_dd_pct >= self.prop_rules.daily_dd_stop_trading_pct:
            self.circuit_breaker_triggered = True
            return False, f"Intraday trailing limit reached ({intraday_dd_pct:.2%})"

        if (
            daily_dd_pct >= self.prop_rules.daily_dd_warning_pct
            or intraday_dd_pct >= self.prop_rules.daily_dd_warning_pct
        ):
            self.risk_ledger.record(
                "DAILY_DD_WARN",
                f"DD warning (Day: {daily_dd_pct:.2%}, Intra: {intraday_dd_pct:.2%})",
                severity="warning",
            )
            self.recovery_mode = True

        pre_stop_frac = float(getattr(self.settings.risk, "drawdown_pre_stop_fraction", 0.90) or 0.90)
        pre_stop_frac = max(0.0, min(1.0, pre_stop_frac))
        if pre_stop_frac > 0.0 and dd_used >= (pre_stop_frac * dd_limit):
            return False, f"Drawdown pre-stop brake active ({dd_used:.2%}/{dd_limit:.2%})"

        total_dd_pct = (self.total_peak_equity - equity) / self.total_peak_equity if self.total_peak_equity > 0 else 0.0
        if total_dd_pct >= self.settings.risk.total_drawdown_limit:
            self.circuit_breaker_triggered = True
            return False, "Total drawdown limit reached"

        daily_profit_stop = getattr(self.settings.risk, "daily_profit_stop_pct", 0.0) or 0.0
        if daily_profit_stop > 0 and self.daily_profit > 0:
            if (self.daily_profit / self.day_start_equity) >= daily_profit_stop:
                return False, "Daily profit stop reached"
        # Profit lock check - stop trading if daily profit target reached
        if (
            self.prop_rules.daily_profit_lock_pct
            and self.prop_rules.daily_profit_lock_pct > 0
            and self.daily_profit > 0
        ):
            profit_pct = (self.daily_profit / self.day_start_equity) if self.day_start_equity > 0 else 0.0
            if profit_pct >= self.prop_rules.daily_profit_lock_pct:
                return (
                    False,
                    f"Prop profit lock reached ({profit_pct:.2%} >= {self.prop_rules.daily_profit_lock_pct:.2%})",
                )

        max_trades = getattr(self.settings.risk, "max_trades_per_day", 0) or 0
        if max_trades <= 0:
            max_trades = self.prop_rules.max_trades_per_day
        if self.recovery_mode:
            max_trades = min(max_trades, self.recovery_max_trades)
        if max_trades > 0 and self.session_trades >= max_trades:
            return False, "Max trades per day reached"

        max_trades_session = int(getattr(self.settings.risk, "max_trades_per_session", 0) or 0)
        if max_trades_session > 0:
            bucket = self._session_bucket_utc(timestamp)
            if int(self.session_trade_counts.get(bucket, 0)) >= max_trades_session:
                return False, f"Max trades reached for session '{bucket}'"

        spread_baseline = self._spread_state.get("spread_baseline", 0.0) or 1e-6
        slippage_baseline = self._spread_state.get("slippage_baseline", 0.0) or 1e-6
        current_spread = self._spread_state.get("current_spread", spread_baseline)
        current_slippage = self._spread_state.get("current_slippage", slippage_baseline)

        spread_ratio = (current_spread + 1e-9) / spread_baseline
        slippage_ratio = (current_slippage + 1e-9) / slippage_baseline

        if spread_ratio > self.settings.risk.spread_guard_multiplier:
            self.risk_ledger.record("SPREAD_GUARD", f"Spread guard triggered ({spread_ratio:.2f}x)", severity="warning")
            return False, f"Spread too high ({spread_ratio:.1f}x baseline)"

        if slippage_ratio > self.settings.risk.slippage_guard_multiplier:
            self.risk_ledger.record(
                "SLIPPAGE_GUARD", f"Slippage guard triggered ({slippage_ratio:.2f}x)", severity="warning"
            )
            return False, f"Slippage risk too high ({slippage_ratio:.1f}x baseline)"

        if ensemble_disagreement is not None:
            max_disagree = float(getattr(self.settings.risk, "max_ensemble_disagreement", 0.20) or 0.20)
            disagree = float(max(0.0, min(1.0, ensemble_disagreement)))
            if max_disagree > 0 and disagree > max_disagree:
                return False, f"Ensemble disagreement {disagree:.2f} exceeds {max_disagree:.2f}"

        # Dynamic confidence threshold check (volatility/session/recovery aware).
        min_conf = float(getattr(self.settings.risk, "min_confidence_threshold", 0.55) or 0.55)

        if bool(getattr(self.settings.risk, "dynamic_confidence_enabled", True)):
            vol = float(max(0.0, market_volatility or 0.0))
            if vol > 0.0:
                vol_ref = float(getattr(self.settings.risk, "volatility_target", 0.0015) or 0.0015)
                norm_vol = min(1.0, vol / max(vol_ref, 1e-9))
                vol_sens = float(getattr(self.settings.risk, "dynamic_confidence_vol_sensitivity", 0.15) or 0.15)
                min_conf += vol_sens * (1.0 - norm_vol)

        bucket = self._session_bucket_utc(timestamp)
        if bucket == "asia":
            min_conf = max(min_conf, float(getattr(self.settings.risk, "session_asia_confidence_threshold", min_conf) or min_conf))
        elif bucket == "london":
            min_conf = max(min_conf, float(getattr(self.settings.risk, "session_london_confidence_threshold", min_conf) or min_conf))
        elif bucket == "newyork":
            min_conf = max(min_conf, float(getattr(self.settings.risk, "session_newyork_confidence_threshold", min_conf) or min_conf))

        if self.recovery_mode:
            # In recovery mode, require HIGHER confidence (use min to take the more restrictive)
            min_conf = max(min_conf + self.recovery_conf_boost, self.recovery_min_win_prob)
            logger.debug(f"Recovery mode active: confidence threshold raised to {min_conf:.2f}")
        min_conf = float(
            min(
                float(getattr(self.settings.risk, "dynamic_confidence_max", 0.90) or 0.90),
                max(float(getattr(self.settings.risk, "dynamic_confidence_min", 0.50) or 0.50), min_conf),
            )
        )
        if confidence < min_conf:
            return False, f"Confidence {confidence:.2f} below threshold {min_conf:.2f}"

        return True, "OK"

    def calculate_position_size(
        self,
        equity: float,
        stop_loss_pips: float,
        confidence: float,
        uncertainty: float = 0.0,
        symbol_info: dict[str, Any] | None = None,
        market_regime: str = "Normal",
        market_volatility: float | None = None,
    ) -> float:
        if stop_loss_pips <= 0:
            return 0.0

        base_risk = self.settings.risk.risk_per_trade

        if confidence >= 0.80:
            signal_multiplier = 1.00
        elif confidence >= 0.60:
            signal_multiplier = 0.50 + (confidence - 0.60) * 2.5
        else:
            signal_multiplier = 0.30

        uncertainty_penalty = 1.0 - (uncertainty * 0.5)  # Max 50% reduction at full uncertainty

        risk_pct = base_risk * signal_multiplier * uncertainty_penalty

        risk_cap = self.settings.risk.max_risk_per_trade
        if self.recovery_mode:
            risk_cap = min(risk_cap, self.recovery_risk_cap)
        risk_pct = min(risk_pct, risk_cap)

        if bool(getattr(self.settings.risk, "volatility_targeting_enabled", True)):
            cur_vol = float(max(0.0, market_volatility or 0.0))
            tgt_vol = float(getattr(self.settings.risk, "volatility_target", 0.0015) or 0.0015)
            if cur_vol > 0.0 and tgt_vol > 0.0:
                vol_scale = tgt_vol / max(cur_vol, 1e-9)
                min_scale = float(getattr(self.settings.risk, "volatility_target_min_scale", 0.35) or 0.35)
                max_scale = float(getattr(self.settings.risk, "volatility_target_max_scale", 1.30) or 1.30)
                vol_scale = float(min(max_scale, max(min_scale, vol_scale)))
                risk_pct *= vol_scale

        regime_txt = str(market_regime or "").strip().lower()
        if any(tag in regime_txt for tag in ("transition", "uncertain", "shock", "volatile_switch")):
            trans_mult = float(getattr(self.settings.risk, "regime_transition_size_multiplier", 0.5) or 0.5)
            risk_pct *= max(0.0, min(1.0, trans_mult))

        news_cap = self._news_state.get("suggested_risk_cap")
        if news_cap is not None:
            risk_pct = min(risk_pct, float(news_cap))

        daily_dd_pct, _intraday_dd_pct, dd_used, dd_limit = self._drawdown_state(equity)

        if len(self.rolling_outcomes) > 0:
            real_win_rate = sum(self.rolling_outcomes) / len(self.rolling_outcomes)
        else:
            real_win_rate = 0.5  # Default until we have data

        spread_baseline = self._spread_state.get("spread_baseline", 1e-6)
        current_spread = self._spread_state.get("current_spread", spread_baseline)
        spread_ratio = current_spread / spread_baseline if spread_baseline > 0 else 1.0

        if spread_ratio > 1.5:
            vol_regime = "high"
        elif spread_ratio < 0.8:
            vol_regime = "low"
        else:
            vol_regime = "normal"

        meta_state = PropMetaState(
            daily_dd_pct=daily_dd_pct,
            volatility_regime=vol_regime,
            recent_win_rate=real_win_rate,
            consecutive_losses=self.consecutive_losses,
            model_confidence=confidence,
            hour_of_day=datetime.now(self._session_tz).hour,
            market_regime=str(market_regime or "Normal"),
        )

        risk_mult, _required_conf, allow_trade = self.meta_controller.get_risk_parameters(meta_state)
        self.last_risk_mult = risk_mult  # Store for logging

        if not allow_trade:
            return 0.0

        risk_pct *= risk_mult

        # Drawdown circuit soft-brakes: progressively cut risk before hard stop.
        dd_frac = dd_used / max(dd_limit, 1e-9)
        soft1_frac = float(getattr(self.settings.risk, "drawdown_soft_brake_1_fraction", 0.50) or 0.50)
        soft2_frac = float(getattr(self.settings.risk, "drawdown_soft_brake_2_fraction", 0.75) or 0.75)
        soft1_mult = float(getattr(self.settings.risk, "drawdown_soft_brake_1_mult", 0.60) or 0.60)
        soft2_mult = float(getattr(self.settings.risk, "drawdown_soft_brake_2_mult", 0.35) or 0.35)
        soft1_frac = max(0.0, min(1.0, soft1_frac))
        soft2_frac = max(soft1_frac, min(1.0, soft2_frac))
        soft1_mult = max(0.0, min(1.0, soft1_mult))
        soft2_mult = max(0.0, min(1.0, soft2_mult))
        if dd_frac >= soft2_frac:
            risk_pct *= soft2_mult
        elif dd_frac >= soft1_frac:
            risk_pct *= soft1_mult

        try:
            if self.total_peak_equity > 0:
                dd_pct = (self.total_peak_equity - equity) / self.total_peak_equity
                if dd_pct > 0:
                    scale = max(0.3, 1.0 - dd_pct / max(self.settings.risk.total_drawdown_limit, 1e-6))
                    risk_pct *= scale
        except Exception as e:
            logger.warning(f"Risk drawdown scaling failed: {e}", exc_info=True)

        if self.challenge_mode:
            now = datetime.now(self._session_tz)
            risk_pct *= self._challenge_progress_multiplier(equity, now, daily_dd_pct)

        risk_floor = float(getattr(self.settings.risk, "min_risk_per_trade", 0.0) or 0.0)
        risk_floor = max(0.0, min(risk_floor, self.settings.risk.max_risk_per_trade))
        risk_pct = min(self.settings.risk.max_risk_per_trade, max(risk_floor, risk_pct))

        pip_size, pip_value = self._compute_pip_metrics(symbol_info)
        if not np.isfinite(pip_size) or pip_size <= 0.0 or not np.isfinite(pip_value) or pip_value <= 0.0:
            logger.error("Rust pip metrics unavailable; position sizing is blocked.")
            return 0.0

        backend_ok = _rust_risk_backend_available(force_log=True)
        if not backend_ok:
            logger.error("Rust risk backend unavailable; position sizing is blocked.")
            return 0.0

        try:
            import forex_bindings  # type: ignore

            lot_size_rs = float(
                forex_bindings.compute_position_size_lots(
                    equity=float(equity),
                    risk_pct=float(risk_pct),
                    stop_loss_pips=float(stop_loss_pips),
                    pip_value=float(pip_value),
                    max_lot_size=float(self.prop_rules.max_lot_size),
                    lot_step=0.01,
                    min_lot=0.0,
                )
            )
            if np.isfinite(lot_size_rs):
                return max(0.0, min(float(lot_size_rs), self.prop_rules.max_lot_size))
        except Exception as exc:
            _disable_rust_risk_backend()
            logger.error("Rust risk sizing failed; position sizing is blocked: %s", exc)
            return 0.0

        logger.error("Rust risk sizing produced non-finite lot size; position sizing is blocked.")
        return 0.0

    def _compute_pip_metrics(self, symbol_info: dict | None) -> tuple[float, float]:
        """
        Return (pip_size, pip_value_per_lot) from Rust bindings only.
        """
        sym = (self.settings.system.symbol or "").upper()
        price_hint = None
        if symbol_info:
            try:
                px = float(symbol_info.get("bid") or symbol_info.get("ask") or symbol_info.get("last") or 0.0)
                if np.isfinite(px) and px > 0.0:
                    price_hint = px
            except Exception:
                price_hint = None

        try:
            import forex_bindings  # type: ignore

            if not hasattr(forex_bindings, "infer_pip_metrics"):
                logger.error("Rust infer_pip_metrics is unavailable for %s.", sym)
                return 0.0, 0.0
            pip_size, pip_value = forex_bindings.infer_pip_metrics(
                sym,
                price=price_hint,
                account_currency="USD",
                reference_prices=None,
            )
            if np.isfinite(pip_size) and pip_size > 0.0 and np.isfinite(pip_value) and pip_value > 0.0:
                return float(pip_size), float(pip_value)
        except Exception as exc:
            logger.error("Rust pip metric inference failed for %s: %s", sym, exc)
            return 0.0, 0.0

        logger.error("Rust pip metric inference returned invalid values for %s.", sym)
        return 0.0, 0.0

    def on_trade_opened(self, timestamp: datetime) -> None:
        """Increment trade counter on open."""
        self.session_trades += 1
        bucket = self._session_bucket_utc(timestamp)
        self.session_trade_counts[bucket] = int(self.session_trade_counts.get(bucket, 0)) + 1

    def on_trade_closed(self, pnl: float, timestamp: datetime) -> None:
        was_stopped = pnl < 0  # Simplified
        self.revenge_trading_detector.record_trade(timestamp - timedelta(minutes=30), timestamp, pnl, was_stopped)
        if pnl < 0:
            self.daily_loss += abs(pnl)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            self.daily_profit += pnl

        self.rolling_outcomes.append(1 if pnl > 0 else 0)
        if len(self.rolling_outcomes) >= 5:
            win_rate = sum(self.rolling_outcomes) / len(self.rolling_outcomes)
            if win_rate < 0.40:
                self.reflection_mode = True
                self.reflection_cooldown_until = (timestamp or datetime.now(self._session_tz)) + timedelta(hours=4)
                logger.warning(f"REFLECTION MODE ACTIVATED: Recent WR {win_rate:.2%} < 40%. Pausing for 4h.")
                self.risk_ledger.record("REFLECTION_MODE", f"Paused due to low WR: {win_rate:.2%}", severity="warning")

        if timestamp:
            self.phase_trade_days.add(timestamp.date())

        try:
            dd_pct = (
                self.day_start_equity - (self.day_start_equity + self.daily_profit - self.daily_loss)
            ) / self.day_start_equity
            if dd_pct >= (0.5 * self.settings.risk.daily_drawdown_limit):
                # Use a temporary per-day limit instead of permanently mutating config.
                # _base_prop_max_trades preserves the original value for next-day recovery.
                self._today_max_trades = max(3, int(self._base_prop_max_trades // 2))
        except Exception as e:
            logger.warning(f"Trade-close drawdown update failed: {e}", exc_info=True)

        try:
            inferred_equity = self.day_start_equity + self.daily_profit - self.daily_loss
            if inferred_equity > self.total_peak_equity:
                self.total_peak_equity = inferred_equity
                self.risk_ledger.record("EQUITY_PEAK", f"New equity peak (from PnL): {inferred_equity:.2f}")
        except Exception as e:
            logger.warning(f"Trade-close equity tracking failed: {e}", exc_info=True)

        self.save_state()
