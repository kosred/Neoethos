"""
Risk management — all arithmetic delegated to `forex_bindings.RiskManager`.
Python handles JSON state persistence, MT5 callbacks, and news/kill-window state only.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

try:
    import fcntl
except ImportError:
    fcntl = None

import forex_bindings as fb  # type: ignore

from ..core.config import Settings
from ..core.storage import RiskLedger
from .meta_controller import MetaController, PropMetaState

logger = logging.getLogger(__name__)

MIN_BREAKEVEN_PROBABILITY = 0.45
RISK_STATE_FILE = Path("cache") / "risk_state.json"


from .risk_utils import (
    ChallengePhase,
    PropFirmRules,
    ChallengeRiskPreset,
    resolve_challenge_risk_preset,
    RevengeTradeDetector
)

logger = logging.getLogger(__name__)

MIN_BREAKEVEN_PROBABILITY = 0.45
RISK_STATE_FILE = Path("cache") / "risk_state.json"


# ---------------------------------------------------------------------------
# Main RiskManager — thin proxy over forex_bindings.RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    MIN_LOT = 0.01

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.symbol = settings.system.symbol or "GLOBAL"
        self.state_file = Path("cache") / f"risk_state_{self.symbol}.json"
        self.challenge_mode = bool(
            getattr(settings.risk, "challenge_mode", False)
            or str(os.environ.get("FOREX_BOT_CHALLENGE_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}
        )
        self.challenge_phase = str(getattr(settings.risk, "challenge_phase", "phase_1") or "phase_1")
        if self.challenge_mode:
            self._apply_challenge_mode_preset()

        session_tz = str(getattr(settings.system, "session_timezone", "UTC") or "UTC")
        try:
            self._session_tz = ZoneInfo(session_tz)
        except Exception:
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
        daily_stop = float(getattr(settings.risk, "daily_drawdown_limit", defaults.daily_dd_stop_trading_pct) or defaults.daily_dd_stop_trading_pct)
        daily_stop = max(0.001, daily_stop)
        daily_warn = max(0.001, min(defaults.daily_dd_warning_pct, daily_stop * 0.9))
        max_trades = max(1, int(getattr(settings.risk, "max_trades_per_day", defaults.max_trades_per_day) or 0))

        self.prop_rules = PropFirmRules(
            max_daily_loss_pct=daily_stop,
            max_total_loss_pct=float(getattr(settings.risk, "total_drawdown_limit", defaults.max_total_loss_pct) or defaults.max_total_loss_pct),
            daily_dd_warning_pct=daily_warn,
            daily_dd_stop_trading_pct=daily_stop,
            max_trades_per_day=max_trades,
        )
        self._base_prop_max_trades = self.prop_rules.max_trades_per_day
        self._today_max_trades = self.prop_rules.max_trades_per_day

        initial_balance = float(getattr(settings.risk, "initial_balance", 10000.0) or 10000.0)
        now_date = datetime.now(self._session_tz).date()

        # Build Rust backend
        self._rust = fb.RiskManager(
            prop_max_daily_loss_pct=daily_stop,
            prop_max_total_loss_pct=float(getattr(settings.risk, "total_drawdown_limit", defaults.max_total_loss_pct) or defaults.max_total_loss_pct),
            prop_max_trades_per_day=max_trades,
            challenge_mode=self.challenge_mode,
            initial_balance=initial_balance,
        )

        # Python-side state (I/O, news, spread, session counters)
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

        self._last_session_date: date | None = None
        self._kill_window_until: datetime | None = None
        self._news_state: dict[str, Any] = {}
        self._spread_state: dict[str, float] = {
            "current_spread": 1.0, "current_slippage": 1.0,
            "spread_baseline": 1.0, "slippage_baseline": 1.0,
        }

        self.day_start_equity: float = initial_balance
        self.day_peak_equity: float = initial_balance
        self.month_start_date: date = now_date.replace(day=1)
        self.month_start_equity: float = initial_balance
        self.total_peak_equity: float = initial_balance

        self.daily_loss: float = 0.0
        self.daily_profit: float = 0.0
        self.session_trades: int = 0
        self.session_trade_counts: dict[str, int] = {}
        self.consecutive_losses: int = 0
        self.circuit_breaker_triggered: bool = False
        self.monthly_return_pct: float = 0.0
        self.monthly_profit_target_pct: float = float(getattr(settings.risk, "monthly_profit_target_pct", 0.04) or 0.04)
        self.monthly_target_hit: bool = False
        self.phase_trade_days: set[date] = set()
        self.challenge_start_date: date | None = now_date if self.challenge_mode else None
        self.challenge_start_equity: float = initial_balance if self.challenge_mode else 0.0
        self.challenge_return_pct: float = 0.0
        self.challenge_target_hit: bool = False
        self.challenge_target_return_pct: float = max(0.0, float(getattr(settings.risk, "challenge_target_return_pct", 0.10) or 0.10))
        self.challenge_target_trading_days: int = max(1, int(getattr(settings.risk, "challenge_target_trading_days", 44) or 44))

        self.recovery_mode: bool = False
        self.recovery_conf_boost: float = 0.10
        self.recovery_min_win_prob: float = max(MIN_BREAKEVEN_PROBABILITY, float(getattr(settings.risk, "high_quality_confidence", 0.65) or 0.65))
        max_risk = float(getattr(settings.risk, "max_risk_per_trade", 0.03) or 0.03)
        self.recovery_risk_cap: float = max(0.0, min(max_risk, max_risk * 0.5))
        self.recovery_max_trades: int = max(1, self._base_prop_max_trades // 2)

        self.reflection_mode: bool = False
        self.reflection_cooldown_until: datetime | None = None
        self.rolling_outcomes: deque[int] = deque(maxlen=20)
        self.last_risk_mult: float = 1.0

        self.load_state()
        self.initialize_session()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

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
        s, e = int(start) % 24, int(end) % 24
        if s == e:
            return True
        return (s <= hour < e) if s < e else (hour >= s or hour < e)

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

    @staticmethod
    def _parse_time(raw: str) -> time:
        h, m = raw.split(":")
        return time(int(h), int(m))

    # ------------------------------------------------------------------
    # Challenge preset
    # ------------------------------------------------------------------

    def _apply_challenge_mode_preset(self) -> None:
        preset = resolve_challenge_risk_preset(self.challenge_phase)
        r = self.settings.risk
        r.risk_per_trade = max(0.0001, min(float(getattr(r, "risk_per_trade", preset.risk_per_trade) or preset.risk_per_trade), preset.risk_per_trade))
        r.base_risk_per_trade = max(0.0001, min(float(getattr(r, "base_risk_per_trade", r.risk_per_trade) or r.risk_per_trade), preset.risk_per_trade))
        r.max_risk_per_trade = max(r.risk_per_trade, min(float(getattr(r, "max_risk_per_trade", preset.max_risk_per_trade) or preset.max_risk_per_trade), preset.max_risk_per_trade))
        r.min_confidence_threshold = min(0.90, max(float(getattr(r, "min_confidence_threshold", preset.min_confidence_threshold) or preset.min_confidence_threshold), preset.min_confidence_threshold))
        r.high_quality_confidence = min(0.95, max(float(getattr(r, "high_quality_confidence", 0.65) or 0.65), r.min_confidence_threshold + 0.05))
        r.max_trades_per_day = max(1, min(int(getattr(r, "max_trades_per_day", preset.max_trades_per_day) or preset.max_trades_per_day), preset.max_trades_per_day))
        r.daily_drawdown_limit = max(0.001, min(float(getattr(r, "daily_drawdown_limit", preset.daily_drawdown_limit) or preset.daily_drawdown_limit), preset.daily_drawdown_limit))
        r.total_drawdown_limit = max(0.01, min(float(getattr(r, "total_drawdown_limit", preset.total_drawdown_limit) or preset.total_drawdown_limit), preset.total_drawdown_limit))
        r.monthly_profit_target_pct = max(float(getattr(r, "monthly_profit_target_pct", preset.monthly_profit_target_pct) or preset.monthly_profit_target_pct), preset.monthly_profit_target_pct)
        r.challenge_target_return_pct = max(0.0, max(float(getattr(r, "challenge_target_return_pct", preset.challenge_target_return_pct) or preset.challenge_target_return_pct), preset.challenge_target_return_pct))
        if preset.daily_profit_lock_pct > 0.0:
            current = float(getattr(r, "daily_profit_stop_pct", 0.0) or 0.0)
            r.daily_profit_stop_pct = min(current, preset.daily_profit_lock_pct) if current > 0.0 else preset.daily_profit_lock_pct
        logger.info("Challenge preset applied: phase=%s risk=%.3f%% max_trades=%d", preset.phase, 100 * float(r.risk_per_trade), int(r.max_trades_per_day))

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

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
                "reflection_cooldown_until": self.reflection_cooldown_until.isoformat() if self.reflection_cooldown_until else None,
            }
            with open(self.state_file, "w") as f:
                if fcntl is not None and sys.platform != "win32":
                    fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(state, f)
                if fcntl is not None and sys.platform != "win32":
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error("Failed to save risk state: %s", e)

    def load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            state = json.loads(self.state_file.read_text())
            saved_raw = state.get("date")
            if not saved_raw:
                return
            saved_date = date.fromisoformat(saved_raw)
            now_date = datetime.now(self._session_tz).date()
            fallback = float(getattr(self.settings.risk, "initial_balance", 0.0) or 0.0)

            if month_raw := state.get("month_start_date"):
                try:
                    self.month_start_date = date.fromisoformat(month_raw)
                except Exception:
                    self.month_start_date = now_date.replace(day=1)
            self.month_start_equity = float(state.get("month_start_equity", fallback))
            self.total_peak_equity = float(state.get("total_peak_equity", max(fallback, self.month_start_equity)))
            self.monthly_target_hit = bool(state.get("monthly_target_hit", False))
            self.challenge_start_equity = float(state.get("challenge_start_equity", self.challenge_start_equity or fallback))
            self.challenge_return_pct = float(state.get("challenge_return_pct", 0.0))
            self.challenge_target_hit = bool(state.get("challenge_target_hit", False))
            if ch_raw := state.get("challenge_start_date"):
                try:
                    self.challenge_start_date = date.fromisoformat(str(ch_raw))
                except Exception:
                    pass
            if cooldown_raw := state.get("reflection_cooldown_until"):
                try:
                    self.reflection_cooldown_until = datetime.fromisoformat(str(cooldown_raw))
                except Exception:
                    self.reflection_cooldown_until = None
            self.reflection_mode = bool(state.get("reflection_mode", False))

            if saved_date == now_date:
                self._last_session_date = saved_date
                self.day_start_equity = float(state.get("day_start_equity", fallback))
                self.day_peak_equity = float(state.get("day_peak_equity", max(self.total_peak_equity, self.day_start_equity)))
                self.daily_loss = float(state.get("daily_loss", 0.0))
                self.daily_profit = float(state.get("daily_profit", 0.0))
                self.session_trades = int(state.get("session_trades", 0))
                raw_counts = state.get("session_trade_counts", {})
                self.session_trade_counts = {str(k): int(v) for k, v in raw_counts.items()} if isinstance(raw_counts, dict) else {}
                self.consecutive_losses = int(state.get("consecutive_losses", 0))
                self.circuit_breaker_triggered = bool(state.get("circuit_breaker_triggered", False))
                self.recovery_mode = bool(state.get("recovery_mode", False))
                logger.info("Restored risk state for %s (%s)", self.symbol, saved_date)
            else:
                self._last_session_date = None
                self.daily_loss = self.daily_profit = 0.0
                self.session_trades = 0
                self.session_trade_counts = {}
                self.consecutive_losses = 0
                self.circuit_breaker_triggered = self.recovery_mode = False
                logger.info("Restored persistent state for %s; daily counters reset", self.symbol)
        except Exception as e:
            logger.warning("Failed to load risk state for %s: %s", self.symbol, e)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def is_trading_session(self) -> bool:
        now = datetime.now(self._session_tz)
        if now.weekday() >= 5:
            return False
        cur = now.time()
        if self._session_end < self._session_start:
            return cur >= self._session_start or cur <= self._session_end
        return self._session_start <= cur <= self._session_end

    def initialize_session(self) -> None:
        now_date = datetime.now(self._session_tz).date()
        if self._last_session_date != now_date:
            self.daily_loss = self.daily_profit = 0.0
            self.session_trades = 0
            self.session_trade_counts = {}
            self.consecutive_losses = 0
            self._last_session_date = None
            self._kill_window_until = None
            logger.info("Session initialized (fresh)")
        else:
            logger.info("Session initialized (continuing)")

    def _ensure_day(self, equity: float, now: datetime) -> None:
        if self._last_session_date is None or now.date() != self._last_session_date:
            self._last_session_date = now.date()
            self.daily_loss = self.daily_profit = 0.0
            self.session_trades = 0
            self.session_trade_counts = {}
            self.consecutive_losses = 0
            self.day_start_equity = equity
            self.day_peak_equity = equity
            self.circuit_breaker_triggered = False
            self.recovery_mode = False
            self.prop_rules.max_trades_per_day = self._base_prop_max_trades
            logger.info("New trading day. Day start equity: %.2f", equity)
            self.save_state()

    def _ensure_month(self, equity: float, now: datetime) -> None:
        month_start = now.date().replace(day=1)
        if self.month_start_date != month_start or self.month_start_equity <= 0:
            self.month_start_date = month_start
            self.month_start_equity = equity
            self.monthly_target_hit = False
            self.risk_ledger.record("MONTH_RESET", f"New month; equity={equity:.2f}", severity="info")
            self.save_state()

    def _update_monthly_metrics(self, equity: float) -> None:
        if self.month_start_equity <= 0:
            return
        self.monthly_return_pct = (equity - self.month_start_equity) / self.month_start_equity
        if not self.monthly_target_hit and self.monthly_return_pct >= self.monthly_profit_target_pct:
            self.monthly_target_hit = True
            self.risk_ledger.record("MONTH_TARGET_HIT", f"Monthly return {self.monthly_return_pct:.2%}", severity="info")
            self.save_state()

    def _update_challenge_metrics(self, equity: float, now: datetime) -> None:
        if not self.challenge_mode:
            return
        if self.challenge_start_equity <= 0:
            self.challenge_start_equity = max(1.0, equity)
        if self.challenge_start_date is None:
            self.challenge_start_date = now.date()
        self.challenge_return_pct = (equity - self.challenge_start_equity) / max(self.challenge_start_equity, 1e-9)
        target = float(getattr(self.settings.risk, "challenge_target_return_pct", self.challenge_target_return_pct) or self.challenge_target_return_pct)
        if not self.challenge_target_hit and target > 0 and self.challenge_return_pct >= target:
            self.challenge_target_hit = True
            self.risk_ledger.record("CHALLENGE_TARGET_HIT", f"Phase return {self.challenge_return_pct:.2%}", severity="info")
            self.save_state()

    # ------------------------------------------------------------------
    # News / spread state updates
    # ------------------------------------------------------------------

    def update_news_state(self, policy_flags: dict[str, Any], now: datetime | None = None) -> None:
        if not policy_flags:
            return
        self._news_state.update(policy_flags)
        now = now or datetime.now(tz=self._session_tz)
        if policy_flags.get("tier1_nearby"):
            self._kill_window_until = now + timedelta(minutes=self.settings.news.news_kill_window_min)
            self.risk_ledger.record("NEWS_KILL", "Tier-1 news kill window active", severity="warning")

    def update_spread_state(self, *, live_spread: float, live_slippage: float, baseline_spread: float, baseline_slippage: float) -> None:
        self._spread_state.update({
            "current_spread": live_spread, "current_slippage": live_slippage,
            "spread_baseline": baseline_spread, "slippage_baseline": baseline_slippage,
        })

    # ------------------------------------------------------------------
    # Core decisions — delegated to Rust
    # ------------------------------------------------------------------

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

        # Rust handles recovery state update
        self._rust.update_recovery_state(equity)
        self.recovery_mode = bool(equity < self.day_start_equity * (1.0 - self.prop_rules.daily_dd_warning_pct / 2.0))

        if self.challenge_mode and self.challenge_target_hit:
            return False, "Challenge target reached"
        if not self.challenge_mode and self.monthly_target_hit:
            return False, "Monthly profit target reached"
        if self.reflection_mode:
            if self.reflection_cooldown_until and timestamp < self.reflection_cooldown_until:
                return False, "reflection_mode_cooldown"
            self.reflection_mode = False
            self.rolling_outcomes.clear()

        if equity > self.total_peak_equity:
            self.total_peak_equity = equity
        self.day_peak_equity = max(self.day_peak_equity, equity)

        if self.circuit_breaker_triggered:
            return False, "Circuit breaker active"
        if not self.is_trading_session():
            return False, "Outside trading session"

        if bool(getattr(self.settings.risk, "block_night_session", True)):
            utc_hour = int(timestamp.astimezone(ZoneInfo("UTC")).hour)
            start_h = int(getattr(self.settings.risk, "night_block_start_utc", 0) or 0)
            end_h = int(getattr(self.settings.risk, "night_block_end_utc", 6) or 6)
            if self._in_hour_window(utc_hour, start_h, end_h):
                min_vol = float(getattr(self.settings.risk, "night_min_volatility", 0.0008) or 0.0008)
                if float(market_volatility or 0.0) < min_vol:
                    return False, f"Night session blocked (vol={market_volatility or 0.0:.5f}<{min_vol:.5f})"

        if self._kill_window_until and timestamp < self._kill_window_until:
            return False, "News kill window active"
        if self.revenge_trading_detector.is_revenge_trading(timestamp):
            return False, "Revenge trading detected"

        # Drawdown checks
        if self.day_start_equity > 0:
            daily_dd = (self.day_start_equity - equity) / self.day_start_equity
            intraday_dd = (self.day_peak_equity - equity) / self.day_peak_equity if self.day_peak_equity > 0 else 0.0
            if daily_dd >= self.prop_rules.daily_dd_stop_trading_pct:
                self.circuit_breaker_triggered = True
                return False, f"Daily drawdown limit ({daily_dd:.2%})"
            if intraday_dd >= self.prop_rules.daily_dd_stop_trading_pct:
                self.circuit_breaker_triggered = True
                return False, f"Intraday trailing limit ({intraday_dd:.2%})"
            if daily_dd >= self.prop_rules.daily_dd_warning_pct or intraday_dd >= self.prop_rules.daily_dd_warning_pct:
                self.risk_ledger.record("DAILY_DD_WARN", f"DD warning Day={daily_dd:.2%} Intra={intraday_dd:.2%}", severity="warning")
                self.recovery_mode = True
            dd_used = max(0.0, max(daily_dd, intraday_dd))
            dd_limit = max(self.prop_rules.daily_dd_stop_trading_pct, 1e-9)
            pre_stop = max(0.0, min(1.0, float(getattr(self.settings.risk, "drawdown_pre_stop_fraction", 0.90) or 0.90)))
            if pre_stop > 0.0 and dd_used >= pre_stop * dd_limit:
                return False, f"Pre-stop brake ({dd_used:.2%}/{dd_limit:.2%})"

        if self.total_peak_equity > 0 and (self.total_peak_equity - equity) / self.total_peak_equity >= self.settings.risk.total_drawdown_limit:
            self.circuit_breaker_triggered = True
            return False, "Total drawdown limit reached"

        daily_profit_stop = float(getattr(self.settings.risk, "daily_profit_stop_pct", 0.0) or 0.0)
        if daily_profit_stop > 0 and self.daily_profit > 0 and (self.daily_profit / self.day_start_equity) >= daily_profit_stop:
            return False, "Daily profit stop reached"
        if self.prop_rules.daily_profit_lock_pct > 0 and self.daily_profit > 0 and self.day_start_equity > 0:
            if (self.daily_profit / self.day_start_equity) >= self.prop_rules.daily_profit_lock_pct:
                return False, f"Prop profit lock ({self.daily_profit / self.day_start_equity:.2%})"

        max_trades = int(getattr(self.settings.risk, "max_trades_per_day", 0) or 0) or self.prop_rules.max_trades_per_day
        if self.recovery_mode:
            max_trades = min(max_trades, self.recovery_max_trades)
        if max_trades > 0 and self.session_trades >= max_trades:
            return False, "Max trades per day reached"

        max_sess = int(getattr(self.settings.risk, "max_trades_per_session", 0) or 0)
        if max_sess > 0:
            bucket = self._session_bucket_utc(timestamp)
            if int(self.session_trade_counts.get(bucket, 0)) >= max_sess:
                return False, f"Max trades for session '{bucket}'"

        spread_baseline = self._spread_state.get("spread_baseline", 0.0) or 1e-6
        slippage_baseline = self._spread_state.get("slippage_baseline", 0.0) or 1e-6
        spread_ratio = (self._spread_state.get("current_spread", spread_baseline) + 1e-9) / spread_baseline
        slippage_ratio = (self._spread_state.get("current_slippage", slippage_baseline) + 1e-9) / slippage_baseline
        if spread_ratio > self.settings.risk.spread_guard_multiplier:
            return False, f"Spread too high ({spread_ratio:.1f}x)"
        if slippage_ratio > self.settings.risk.slippage_guard_multiplier:
            return False, f"Slippage risk too high ({slippage_ratio:.1f}x)"

        if ensemble_disagreement is not None:
            max_d = float(getattr(self.settings.risk, "max_ensemble_disagreement", 0.20) or 0.20)
            if max_d > 0 and float(ensemble_disagreement) > max_d:
                return False, f"Ensemble disagreement {ensemble_disagreement:.2f}"

        # Dynamic confidence threshold
        min_conf = float(getattr(self.settings.risk, "min_confidence_threshold", 0.55) or 0.55)
        if bool(getattr(self.settings.risk, "dynamic_confidence_enabled", True)):
            vol = float(max(0.0, market_volatility or 0.0))
            if vol > 0:
                vol_ref = float(getattr(self.settings.risk, "volatility_target", 0.0015) or 0.0015)
                norm_vol = min(1.0, vol / max(vol_ref, 1e-9))
                min_conf += float(getattr(self.settings.risk, "dynamic_confidence_vol_sensitivity", 0.15) or 0.15) * (1.0 - norm_vol)

        bucket = self._session_bucket_utc(timestamp)
        session_key = f"session_{bucket}_confidence_threshold"
        min_conf = max(min_conf, float(getattr(self.settings.risk, session_key, min_conf) or min_conf))

        if self.recovery_mode:
            min_conf = max(min_conf + self.recovery_conf_boost, self.recovery_min_win_prob)

        min_conf = float(min(
            float(getattr(self.settings.risk, "dynamic_confidence_max", 0.90) or 0.90),
            max(float(getattr(self.settings.risk, "dynamic_confidence_min", 0.50) or 0.50), min_conf),
        ))
        if confidence < min_conf:
            return False, f"Confidence {confidence:.2f} < threshold {min_conf:.2f}"

        return True, "OK"

    def calculate_position_size(
        self,
        equity: float,
        stop_loss_pips: float,
        confidence: float,
        uncertainty: float = 0.0,
        symbol_info: dict | None = None,
        market_regime: str = "Normal",
        market_volatility: float | None = None,
    ) -> float:
        """Delegates full calculation to Rust, then applies news cap and min-lot clamp."""
        if stop_loss_pips <= 0:
            return 0.0

        base_risk = float(self.settings.risk.risk_per_trade)
        max_risk = float(self.settings.risk.max_risk_per_trade)
        tgt_vol = float(getattr(self.settings.risk, "volatility_target", 0.0015) or 0.0015)
        is_volatile = str(market_regime or "").lower() in ("transition", "uncertain", "shock", "volatile_switch")

        size = self._rust.calculate_position_size(
            equity,
            base_risk,
            max_risk,
            confidence,
            uncertainty,
            float(max(0.0, market_volatility or 0.0)),
            tgt_vol,
            is_volatile,
        )

        # Apply news risk cap (Python-side state)
        news_cap = self._news_state.get("suggested_risk_cap")
        if news_cap is not None:
            cap_lots = equity * float(news_cap) / max(stop_loss_pips * self._compute_pip_metrics(symbol_info or {})[1], 1e-9)
            size = min(size, cap_lots)

        # Compute actual lot size from risk fraction
        pip_size, pip_value = self._compute_pip_metrics(symbol_info or {})
        if pip_size > 0 and pip_value > 0 and stop_loss_pips > 0:
            risk_amount = equity * (size if size < 1.0 else base_risk)
            lots = risk_amount / (stop_loss_pips * pip_value)
            lots = max(self.MIN_LOT, round(lots, 2))
            max_lot = float(self.prop_rules.max_lot_size)
            lots = min(lots, max_lot)
            self.last_risk_mult = size / max(base_risk, 1e-9)
            return lots

        return 0.0

    def _compute_pip_metrics(self, symbol_info: dict) -> tuple[float, float]:
        sym = self.symbol.upper()
        try:
            point = float(symbol_info.get("point", 0.0001) or 0.0001)
            digits = int(symbol_info.get("digits", 5) or 5)
            pip_size = float(fb.pip_size_from_symbol(sym, point=point, digits=digits))
        except Exception:
            pip_size = 0.0001
        try:
            _, pip_value = fb.infer_pip_metrics(sym)
        except Exception:
            pip_value = 10.0
        return pip_size, pip_value

    # ------------------------------------------------------------------
    # Trade event hooks
    # ------------------------------------------------------------------

    def on_trade_opened(self, now: datetime) -> None:
        self.session_trades += 1
        bucket = self._session_bucket_utc(now)
        self.session_trade_counts[bucket] = self.session_trade_counts.get(bucket, 0) + 1
        self.phase_trade_days.add(now.date())
        self.save_state()

    def on_trade_closed(self, pnl: float, equity: float) -> None:
        if pnl > 0:
            self.daily_profit += pnl
            self.consecutive_losses = 0
            self.rolling_outcomes.append(1)
        else:
            self.daily_loss += abs(pnl)
            self.consecutive_losses += 1
            self.rolling_outcomes.append(0)
        self.day_peak_equity = max(self.day_peak_equity, equity)
        if equity > self.total_peak_equity:
            self.total_peak_equity = equity
        self._rust.update_recovery_state(equity)
        self.save_state()
