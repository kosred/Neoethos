"""
Order execution — all logic delegated to the Rust `forex_bindings.OrderExecutor`.
Python is responsible only for async MT5 I/O (place_order, close_position).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import forex_bindings as fb

from ..core.config import Settings
from ..core.storage import RiskLedger, StrategyLedger
from ..execution.mt5_state_manager import MT5StateManager
from ..execution.risk import RiskManager
from ..execution import frame_utils

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Thin Python proxy. All arithmetic is done in Rust via `forex_bindings.OrderExecutor`.
    Only MT5 I/O, logging, and ledger updates live here.
    """

    def __init__(
        self,
        settings: Settings,
        risk_manager: RiskManager,
        mt5_manager: MT5StateManager,
        strategy_ledger: StrategyLedger | None = None,
        risk_ledger: RiskLedger | None = None,
    ) -> None:
        self.settings = settings
        self.risk_manager = risk_manager
        self.mt5 = mt5_manager
        self.strategy_ledger = strategy_ledger
        self.risk_ledger = risk_ledger

        r = settings.risk
        self._rust = fb.OrderExecutor(
            symbol=settings.system.symbol,
            partial_take_profit_enabled=bool(getattr(r, "partial_take_profit_enabled", True)),
            partial_tp_min_total_lot=float(getattr(r, "partial_tp_min_total_lot", 0.03) or 0.03),
            partial_tp_r_levels=self._parse_float_list(getattr(r, "partial_tp_r_levels", "1.0,2.0,3.0"), [1.0, 2.0, 3.0]),
            partial_tp_size_fracs=self._parse_float_list(getattr(r, "partial_tp_size_fracs", "0.5,0.25,0.25"), [0.5, 0.25, 0.25]),
            min_risk_reward=float(getattr(r, "min_risk_reward", 1.5) or 1.5),
            entry_patience_enabled=bool(getattr(r, "entry_patience_enabled", True)),
            entry_patience_bars=int(getattr(r, "entry_patience_bars", 3) or 3),
            entry_patience_pullback_atr=float(getattr(r, "entry_patience_pullback_atr", 0.20) or 0.20),
            min_edge_cost_multiple=float(getattr(r, "min_edge_cost_multiple", 3.0) or 3.0),
            commission_per_lot=float(getattr(r, "commission_per_lot", 7.0) or 7.0),
        )
        self._cost_alpha = 0.12
        self._spread_ema = float(getattr(r, "backtest_spread_pips", 1.5) or 1.5)
        self._slippage_ema = float(getattr(r, "slippage_pips", 0.5) or 0.5)

    def _parse_float_list(self, raw: Any, default: list[float]) -> list[float]:
        try:
            return [float(x.strip()) for x in str(raw).split(",")] if raw else default
        except Exception:
            return default

    def _get_pip_size(self, symbol_info: dict | None = None) -> float:
        sym = self.settings.system.symbol.upper()
        try:
            info = symbol_info or {}
            point = float(info.get("point", 0.0001) or 0.0001)
            digits = int(info.get("digits", 5) or 5)
            return float(fb.pip_size_from_symbol(sym, point=point, digits=digits))
        except Exception:
            return 0.0

    def update_live_cost_state(self, tick: dict, *, symbol_info: dict | None = None) -> None:
        pip_size = self._get_pip_size(symbol_info)
        if pip_size <= 0: return
        
        bid, ask = float(tick.get("bid", 0.0)), float(tick.get("ask", 0.0))
        spread_pips = (ask - bid) / pip_size if ask > bid > 0 else self._spread_ema
        
        a = self._cost_alpha
        self._spread_ema = (1 - a) * self._spread_ema + a * spread_pips
        self.risk_manager.update_spread_state(
            live_spread=spread_pips, live_slippage=self._slippage_ema,
            baseline_spread=self._spread_ema, baseline_slippage=self._slippage_ema
        )

    async def execute_signal(
        self, signal_result: Any, equity: float, frames: dict[str, Any], **kwargs
    ) -> None:
        if signal_result.signal == 0: return

        # 1. Parameter Prep (SL, TP, Size)
        params = await self._prepare_trade_params(signal_result, equity, frames, **kwargs)
        if not params: return
        
        # 2. Pre-flight checks
        if not await self._pre_flight_gate(params, frames, **kwargs): return
        
        # 3. Execution
        await self._execute_order_legs(params, signal_result, **kwargs)

    async def _prepare_trade_params(self, signal_result: Any, equity: float, frames: dict, **kwargs) -> dict | None:
        symbol = self.settings.system.symbol
        base_df = frames.get(self.settings.system.base_timeframe) or frames.get("M1")
        if frame_utils.frame_empty(base_df): return None

        # SL Calculation
        sl_pips = self._calculate_sl_pips(signal_result, base_df)
        if not sl_pips: return None

        # Sizing
        info = kwargs.get("symbol_info") or await self.mt5.connection.get_symbol_info(symbol) or {}
        market_vol = float((getattr(signal_result, "meta_features", {}) or {}).get("market_volatility", 0.0))
        
        size = self.risk_manager.calculate_position_size(
            equity, sl_pips, signal_result.confidence, 0.0, info,
            market_regime=getattr(signal_result, "regime", "Normal"),
            market_volatility=market_vol
        )
        
        # Adjust for extra weight/stance
        if kwargs.get("advice_stance") == "conservative": size *= 0.7
        if size <= 0: return None

        # Prices
        tick = kwargs.get("tick_price") or await self.mt5.connection.get_symbol_price(symbol) or {}
        pip_size = self._get_pip_size(info)
        rr = max(1.5, float(getattr(self.settings.risk, "min_risk_reward", 1.5)))
        
        entry_price = float(tick.get("ask") if signal_result.signal == 1 else tick.get("bid"))
        sl, tp, _, sl_dist, rr_final = self._rust.compute_order_prices(entry_price, signal_result.signal, sl_pips, rr, pip_size)

        return {
            "symbol": symbol, "size": size, "sl": sl, "tp": tp, "sl_pips": sl_pips, 
            "entry_price": entry_price, "sl_dist": sl_dist, "rr": rr_final, 
            "info": info, "tick": tick, "pip_size": pip_size
        }

    def _calculate_sl_pips(self, signal_result: Any, base_df: Any) -> float | None:
        if signal_result.recommended_sl:
            return float(signal_result.recommended_sl)
            
        atr = frame_utils.column_array(base_df, "atr")
        if atr is not None and atr.size > 0:
            pip_size = self._get_pip_size()
            dist = float(atr[-1]) * float(getattr(self.settings.risk, "atr_stop_multiplier", 1.5))
            return dist / pip_size if pip_size > 0 else None
        return None

    async def _pre_flight_gate(self, params: dict, frames: dict, **kwargs) -> bool:
        # Edge Check
        state = getattr(self.risk_manager, "_spread_state", {})
        spread = params["tick"].get("ask", 0) - params["tick"].get("bid", 0)
        spread_pips = (spread / params["pip_size"]) if params["pip_size"] > 0 else params["sl_pips"]
        
        _pip_sz, pip_val = self.risk_manager._compute_pip_metrics(params["info"])
        passed, _, _ = self._rust.evaluate_trade_edge(params["sl_pips"], params["rr"], spread_pips, self._slippage_ema, pip_val)
        if not passed:
            logger.info("Signal skipped by edge gate.")
            return False

        # Patience Check
        base_df = frames.get(self.settings.system.base_timeframe) or frames.get("M1")
        if self._entry_patience_block(params["entry_price"], base_df):
            logger.info("Entry patience gate blocked trade.")
            return False

        return True

    def _entry_patience_block(self, entry_price: float, base_df: Any) -> bool:
        if not self.settings.risk.entry_patience_enabled: return False
        bars = int(self.settings.risk.entry_patience_bars or 3)
        close = frame_utils.column_array(base_df, "close")
        if close is None or close.size <= bars: return False
        
        atr = frame_utils.column_array(base_df, "atr")
        pullback = float(self.settings.risk.entry_patience_pullback_atr or 0.2) * (float(atr[-1]) if atr is not None else 0)
        recent = close[-(bars+1):]
        if entry_price > (np.min(recent[:-1]) + pullback): return True # Overly aggressive buy
        return False

    async def _execute_order_legs(self, params: dict, result: Any, **kwargs) -> None:
        can_split = int(getattr(self.mt5, "max_positions_per_symbol", 1)) > 1
        legs = self._rust.build_order_legs(params["size"], result.signal, params["entry_price"], params["sl"], params["sl_dist"], params["tp"]) if can_split else [(round(params["size"], 2), params["tp"])]

        order_type = "buy" if result.signal == 1 else "sell"
        bar_time = frame_utils.get_bar_time(kwargs.get("frames", {}).get("M1")) or datetime.now(UTC)

        any_success = False
        for vol, leg_tp in legs:
            res = await self.mt5.place_order_with_verification(
                symbol=params["symbol"], order_type=order_type, volume=vol, 
                sl=params["sl"], tp=leg_tp, current_bar_time=bar_time
            )
            if res.get("success"):
                any_success = True
                self._handle_success(res, order_type, vol, params["sl"], leg_tp, result, **kwargs)
        
        if any_success: self.risk_manager.on_trade_opened(datetime.now(UTC))

    def _handle_success(self, result, order_type, size, sl, tp, signal_result, **kwargs):
        ticket = result.get("ticket")
        logger.info(f"[ORDER SUCCESS] Ticket={ticket}")
        if self.strategy_ledger and ticket:
            self.strategy_ledger.log_intent(ticket=ticket, symbol=self.settings.system.symbol, direction=order_type, volume=size, sl=sl, tp=tp, meta_risk_mult=1.0)
        
        if kwargs.get("entry_features") and ticket:
            self.mt5.record_entry_features(ticket=ticket, symbol=self.settings.system.symbol, bar_time=datetime.now(UTC), features=kwargs["entry_features"], signal=signal_result.signal, order_ticket=result.get("order_ticket"), deal_ticket=result.get("deal_ticket"), magic=result.get("magic"))

    async def close_position(self, ticket: int, volume: float, reason: str | None = None) -> None:
        if await self.mt5.close_position_by_ticket(ticket, self.settings.system.symbol, volume=volume):
            logger.info(f"Closed ticket={ticket} reason={reason}")
        else:
            raise RuntimeError(f"Failed to close ticket {ticket}")
