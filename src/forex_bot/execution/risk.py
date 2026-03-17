"""
Risk management — logic fully delegated to `forex_bindings.RiskManager` (Pure Rust).
Python handles only high-level orchestration, state file I/O, and UI/logging.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import forex_bindings as fb  # type: ignore

from ..core.config import Settings
from ..core.storage import RiskLedger

logger = logging.getLogger(__name__)

class RiskManager:
    MIN_LOT = 0.01

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.symbol = settings.system.symbol or "GLOBAL"
        self.state_file = Path("cache") / f"risk_state_{self.symbol}.json"
        
        challenge_mode = bool(
            getattr(settings.risk, "challenge_mode", False)
            or str(os.environ.get("FOREX_BOT_CHALLENGE_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}
        )
        initial_balance = float(getattr(settings.risk, "initial_balance", 10000.0) or 10000.0)
        daily_stop = float(getattr(settings.risk, "daily_drawdown_limit", 0.045) or 0.045)
        total_stop = float(getattr(settings.risk, "total_drawdown_limit", 0.10) or 0.10)
        max_trades = int(getattr(settings.risk, "max_trades_per_day", 15) or 15)

        # Initialize Rust Backend
        self._rust = fb.RiskManager(
            prop_max_daily_loss_pct=daily_stop,
            prop_max_total_loss_pct=total_stop,
            prop_max_trades_per_day=max_trades,
            challenge_mode=challenge_mode,
            initial_balance=initial_balance
        )

        # Configure Session Times in Rust
        try:
            start_h, start_m = map(int, str(getattr(settings.system, "trading_session_start", "00:05")).split(":"))
            end_h, end_m = map(int, str(getattr(settings.system, "trading_session_end", "23:55")).split(":"))
            self._rust.set_session_times(start_h, start_m, end_h, end_m)
        except Exception:
            self._rust.set_session_times(0, 5, 23, 55)

        # Configure Night Block in Rust
        night_enabled = bool(getattr(settings.risk, "block_night_session", True))
        night_start = int(getattr(settings.risk, "night_block_start_utc", 0) or 0)
        night_end = int(getattr(settings.risk, "night_block_end_utc", 6) or 6)
        night_min_vol = float(getattr(settings.risk, "night_min_volatility", 0.0008) or 0.0008)
        self._rust.set_night_block(night_enabled, night_start, night_end, night_min_vol)

        self.risk_ledger = RiskLedger(max_events=1000)
        self._session_tz = ZoneInfo(str(getattr(settings.system, "session_timezone", "UTC")))
        
        self.load_state()

    def save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            rust_json = self._rust.save_state_json()
            with open(self.state_file, "w") as f:
                f.write(rust_json)
        except Exception as e:
            logger.error("Failed to save risk state: %s", e)

    def load_state(self) -> None:
        if self.state_file.exists():
            try:
                content = self.state_file.read_text()
                self._rust.load_state_json(content)
                logger.info("Restored Risk state from %s", self.state_file)
            except Exception as e:
                logger.warning("Failed to load Risk state: %s", e)

    def check_trade_allowed(
        self,
        equity: float,
        confidence: float,
        timestamp: datetime,
        *,
        market_volatility: float | None = None,
    ) -> tuple[bool, str]:
        # Delegate to Rust
        now_ts = int(timestamp.timestamp())
        hour = timestamp.hour
        minute = timestamp.minute
        weekday = timestamp.weekday()
        vol = float(market_volatility or 0.0)

        allowed, reason = self._rust.check_trade_allowed(
            equity,
            confidence,
            now_ts,
            hour,
            minute,
            weekday,
            vol
        )
        
        if not allowed:
            logger.debug("Trade blocked: %s", reason)
        return allowed, reason

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
        if stop_loss_pips <= 0:
            return 0.0

        base_risk = float(self.settings.risk.risk_per_trade)
        max_risk = float(self.settings.risk.max_risk_per_trade)
        tgt_vol = float(getattr(self.settings.risk, "volatility_target", 0.0015) or 0.0015)
        is_volatile = str(market_regime or "").lower() in ("transition", "uncertain", "shock", "volatile_switch")

        # Get risk fraction from Rust
        risk_fraction = self._rust.calculate_position_size(
            equity,
            base_risk,
            max_risk,
            confidence,
            uncertainty,
            float(market_volatility or 0.0),
            tgt_vol,
            is_volatile,
        )

        # Convert to lots
        _, pip_value = self._compute_pip_metrics(symbol_info or {})
        risk_amount = equity * risk_fraction
        lots = risk_amount / (stop_loss_pips * pip_value)
        lots = max(self.MIN_LOT, round(lots, 2))
        return min(lots, 10.0)

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

    def on_trade_opened(self, _now: datetime) -> None:
        self._rust.on_trade_opened()
        self.save_state()

    def on_trade_closed(self, pnl: float, equity: float) -> None:
        self._rust.on_trade_closed(pnl, equity)
        self.save_state()

    def update_news_state(self, policy_flags: dict[str, Any], now: datetime | None = None) -> None:
        if policy_flags.get("tier1_nearby"):
            now = now or datetime.now(tz=self._session_tz)
            until = now + timedelta(minutes=self.settings.news.news_kill_window_min)
            self._rust.update_kill_window(int(until.timestamp()))
