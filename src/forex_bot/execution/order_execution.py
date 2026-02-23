import logging
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import Settings
from ..core.storage import RiskLedger, StrategyLedger
from ..execution.mt5_state_manager import MT5StateManager
from ..execution.risk import RiskManager
from ..strategy.stop_target import infer_stop_target_pips

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Handles the mechanics of calculating order parameters (SL/TP/Size)
    and executing trades via MT5.
    """

    def __init__(
        self,
        settings: Settings,
        risk_manager: RiskManager,
        mt5_manager: MT5StateManager,
        strategy_ledger: StrategyLedger | None = None,
        risk_ledger: RiskLedger | None = None,
    ):
        self.settings = settings
        self.risk_manager = risk_manager
        self.mt5 = mt5_manager
        self.strategy_ledger = strategy_ledger
        self.risk_ledger = risk_ledger
        self._last_rr: float | None = None
        try:
            self._cost_alpha = float(os.environ.get("FOREX_BOT_COST_STATE_ALPHA", "0.12") or 0.12)
        except Exception:
            self._cost_alpha = 0.12
        self._cost_alpha = max(0.01, min(0.50, self._cost_alpha))
        self._spread_ema = max(1e-6, float(getattr(self.settings.risk, "backtest_spread_pips", 1.5) or 1.5))
        self._slippage_ema = max(1e-6, float(getattr(self.settings.risk, "slippage_pips", 0.5) or 0.5))

    async def close_position(self, ticket: int, volume: float, reason: str | None = None) -> None:
        """
        Close a position via MT5StateManager, surfacing failures instead of silently swallowing them.
        """
        symbol = self.settings.system.symbol
        success = await self.mt5.close_position_by_ticket(ticket, symbol, volume=volume)
        if not success:
            raise RuntimeError(f"Close failed for ticket {ticket} ({symbol}): {reason or 'no reason provided'}")
        logger.info(f"Closed position ticket={ticket} vol={volume} reason={reason or 'n/a'}")

    async def execute_signal(
        self,
        signal_result: Any,
        equity: float,
        frames: dict[str, Any],
        alloc_weight: Any = None,
        advice_stance: str | None = None,
        tick_price: dict[str, float] | None = None,
        symbol_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Process a buy/sell signal: calc risk, place order, log intent.
        """
        if signal_result.signal == 0:
            return

        symbol = self.settings.system.symbol

        # 1. Calculate SL Pips
        sl_pips = self._calculate_sl_pips(signal_result, frames)
        if sl_pips is None:
            logger.warning("Could not calculate SL pips; skipping trade.")
            return

        # 2. Calculate Size
        symbol_info = symbol_info or await self.mt5.connection.get_symbol_info(symbol) or {}
        uncertainty = self._latest_scalar(getattr(signal_result, "uncertainty", 0.0), default=0.0)
        try:
            market_volatility = float((getattr(signal_result, "meta_features", {}) or {}).get("market_volatility", 0.0) or 0.0)
        except Exception:
            market_volatility = 0.0

        size = self.risk_manager.calculate_position_size(
            equity,
            sl_pips,
            signal_result.confidence,
            uncertainty,
            symbol_info,
            market_regime=getattr(signal_result, "regime", "Normal"),
            market_volatility=market_volatility,
        )

        # 3. Adjust Size based on Advice/Allocation
        if advice_stance:
            if advice_stance == "conservative":
                size *= 0.7
            elif advice_stance == "aggressive":
                size *= 1.1

        if alloc_weight:
            size = size * max(0.1, min(1.0, alloc_weight.weight))

        if size <= 0:
            logger.info(f"Signal {signal_result.signal} ignored (Calculated size 0).")
            return

        # 4. Calculate Prices
        tick = tick_price or await self.mt5.connection.get_symbol_price(symbol) or {}
        self.update_live_cost_state(tick, symbol_info=symbol_info)
        px = self._calculate_prices(signal_result, frames, sl_pips, symbol_info, tick)
        if px is None:
            return
        sl, tp, entry_price, sl_dist, rr = px

        if not self._edge_over_cost_ok(
            sl_pips=sl_pips,
            rr=rr,
            tick=tick,
            symbol_info=symbol_info,
        ):
            logger.info("Signal skipped by cost/edge gate.")
            return

        base_df = frames.get(self.settings.system.base_timeframe)
        if base_df is None:
            base_df = frames.get("M1")
        if base_df is None or getattr(base_df, "empty", True):
            current_bar_time = datetime.now(UTC)
        else:
            try:
                if "timestamp" in getattr(base_df, "columns", []):
                    current_bar_time = base_df["timestamp"].iloc[-1]
                else:
                    current_bar_time = base_df.index[-1]
                # Normalize to python datetime where possible
                if hasattr(current_bar_time, "to_pydatetime"):
                    current_bar_time = current_bar_time.to_pydatetime()
                if isinstance(current_bar_time, str):
                    current_bar_time = datetime.fromisoformat(current_bar_time)
            except Exception:
                current_bar_time = datetime.now(UTC)
        order_type = "buy" if signal_result.signal == 1 else "sell"

        if self._entry_patience_block(int(signal_result.signal), base_df):
            logger.info("Entry patience gate delayed %s entry pending pullback.", order_type.upper())
            return

        can_split_entries = int(getattr(self.mt5, "max_positions_per_symbol", 1) or 1) > 1
        if can_split_entries:
            legs = self._build_order_legs(
                total_size=size,
                signal=int(signal_result.signal),
                entry_price=entry_price,
                sl=sl,
                sl_dist=sl_dist,
                default_tp=tp,
            )
        else:
            legs = [(round(float(size), 2), float(tp))]
        if len(legs) > 1:
            logger.info("Executing %s split legs: %s", order_type.upper(), legs)
        else:
            logger.info(f"Executing {order_type.upper()} {size} lots (SL={sl:.5f}, TP={tp:.5f})...")

        any_success = False
        last_result = {"success": False, "reason": "Execution failed"}
        for vol, leg_tp in legs:
            result = await self._place_order_with_retry(
                symbol=symbol,
                order_type=order_type,
                volume=vol,
                sl=sl,
                tp=leg_tp,
                current_bar_time=current_bar_time,
            )
            last_result = result
            if result.get("success"):
                any_success = True
                self._handle_success(result, order_type, vol, sl, leg_tp, signal_result, count_trade=False)
                self._record_fill_cost_state(
                    order_type=order_type,
                    expected_entry=entry_price,
                    result=result,
                    tick=tick,
                    symbol_info=symbol_info,
                )

        # Count a split entry as one parent trade for daily/session risk gates.
        if any_success and self.risk_manager:
            self.risk_manager.on_trade_opened(datetime.now(UTC))
        elif not any_success:
            self._handle_failure(last_result)

    def _entry_patience_block(self, signal: int, base_df: Any) -> bool:
        if not bool(getattr(self.settings.risk, "entry_patience_enabled", True)):
            return False
        if base_df is None or getattr(base_df, "empty", True):
            return False
        try:
            bars = int(getattr(self.settings.risk, "entry_patience_bars", 3) or 3)
            bars = max(1, bars)
            close = pd.Series(base_df["close"]).astype(float)
            if len(close) <= bars:
                return False
            atr_val = 0.0
            if "atr" in base_df.columns:
                atr_val = float(pd.Series(base_df["atr"]).astype(float).iloc[-1])
            elif "atr14" in base_df.columns:
                atr_val = float(pd.Series(base_df["atr14"]).astype(float).iloc[-1])
            pullback_atr = float(getattr(self.settings.risk, "entry_patience_pullback_atr", 0.20) or 0.20)
            pullback = max(0.0, pullback_atr * max(0.0, atr_val))
            recent = close.iloc[-(bars + 1) :]
            last = float(recent.iloc[-1])
            prior = recent.iloc[:-1]
            if signal > 0:
                return bool(last >= (float(prior.max()) - pullback))
            if signal < 0:
                return bool(last <= (float(prior.min()) + pullback))
            return False
        except Exception:
            return False

    @staticmethod
    def _latest_scalar(value: Any, *, default: float = 0.0) -> float:
        """
        Convert common vector-like outputs (Series/ndarray/list) into a scalar.

        Many model outputs are per-row Series; for live execution we always want the latest value.
        """
        if value is None:
            return float(default)

        # Pandas objects: take last element.
        if isinstance(value, pd.Series):
            if len(value) == 0:
                return float(default)
            try:
                return float(value.iloc[-1])
            except Exception:
                return float(default)
        if isinstance(value, pd.DataFrame):
            if value.empty:
                return float(default)
            try:
                return float(value.iloc[-1, -1])
            except Exception:
                return float(default)

        # Numpy arrays / sequences: take last flattened value.
        if isinstance(value, (np.ndarray, list, tuple)):
            try:
                arr = np.asarray(value)
                if arr.size == 0:
                    return float(default)
                return float(arr.reshape(-1)[-1])
            except Exception:
                return float(default)

        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _parse_float_list(raw: Any, default: list[float]) -> list[float]:
        if raw is None:
            return list(default)
        txt = str(raw).strip()
        if not txt:
            return list(default)
        out: list[float] = []
        for part in txt.split(","):
            token = str(part).strip()
            if not token:
                continue
            try:
                out.append(float(token))
            except Exception:
                continue
        return out or list(default)

    def _estimate_spread_pips(self, tick: dict[str, float], pip_size: float) -> float:
        if pip_size <= 0:
            return 0.0
        bid = float(tick.get("bid", 0.0) or 0.0)
        ask = float(tick.get("ask", 0.0) or 0.0)
        if ask > 0 and bid > 0 and ask >= bid:
            return max(0.0, (ask - bid) / pip_size)
        return 0.0

    def _push_cost_state(self, *, live_spread: float, live_slippage: float) -> None:
        spread = max(0.0, float(live_spread))
        slippage = max(0.0, float(live_slippage))
        alpha = float(self._cost_alpha)
        self._spread_ema = ((1.0 - alpha) * self._spread_ema) + (alpha * spread) if spread > 0 else self._spread_ema
        self._slippage_ema = (
            ((1.0 - alpha) * self._slippage_ema) + (alpha * slippage) if slippage > 0 else self._slippage_ema
        )
        try:
            self.risk_manager.update_spread_state(
                live_spread=spread if spread > 0 else self._spread_ema,
                live_slippage=slippage if slippage > 0 else self._slippage_ema,
                baseline_spread=max(1e-6, self._spread_ema),
                baseline_slippage=max(1e-6, self._slippage_ema),
            )
        except Exception as exc:
            logger.debug("Cost-state update failed: %s", exc)

    def update_live_cost_state(self, tick: dict[str, float], *, symbol_info: dict[str, Any] | None = None) -> None:
        symbol = self.settings.system.symbol
        info = symbol_info or {}
        pip_size = self._get_pip_size(symbol, info)
        spread_pips = self._estimate_spread_pips(tick, pip_size)
        fallback_slippage = float(getattr(self.settings.risk, "slippage_pips", 0.5) or 0.5)
        self._push_cost_state(live_spread=spread_pips, live_slippage=fallback_slippage)

    def _record_fill_cost_state(
        self,
        *,
        order_type: str,
        expected_entry: float,
        result: dict[str, Any],
        tick: dict[str, float],
        symbol_info: dict[str, Any],
    ) -> None:
        position = result.get("position")
        if position is None:
            return
        pip_size = self._get_pip_size(self.settings.system.symbol, symbol_info)
        if pip_size <= 0:
            return
        try:
            fill_price = float(getattr(position, "price_open", 0.0) or 0.0)
        except Exception:
            fill_price = 0.0
        if fill_price <= 0 or expected_entry <= 0:
            return
        if str(order_type).strip().lower() == "buy":
            slippage = max(0.0, fill_price - expected_entry) / pip_size
        else:
            slippage = max(0.0, expected_entry - fill_price) / pip_size
        spread_pips = self._estimate_spread_pips(tick, pip_size)
        self._push_cost_state(live_spread=spread_pips, live_slippage=slippage)

    async def _place_order_with_retry(
        self,
        *,
        symbol: str,
        order_type: str,
        volume: float,
        sl: float,
        tp: float,
        current_bar_time: datetime,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"success": False, "reason": "Execution failed"}
        for attempt in range(1, 4):
            try:
                result = await self.mt5.place_order_with_verification(
                    symbol=symbol,
                    order_type=order_type,
                    volume=volume,
                    sl=sl,
                    tp=tp,
                    current_bar_time=current_bar_time,
                )
                if result.get("success"):
                    break
                logger.warning(
                    "Order attempt %s failed for %.2f lots (reason=%s)",
                    attempt,
                    volume,
                    result.get("reason"),
                )
                import asyncio

                await asyncio.sleep(1.0)
            except Exception as exc:
                logger.error("Order attempt %s raised exception: %s", attempt, exc)
                import asyncio

                await asyncio.sleep(1.0)
        return result

    def _edge_over_cost_ok(
        self,
        *,
        sl_pips: float,
        rr: float,
        tick: dict[str, float],
        symbol_info: dict[str, Any],
    ) -> bool:
        min_mult = float(getattr(self.settings.risk, "min_edge_cost_multiple", 3.0) or 3.0)
        if min_mult <= 0:
            return True
        pip_size = self._get_pip_size(self.settings.system.symbol, symbol_info)
        bid = float(tick.get("bid", 0.0) or 0.0)
        ask = float(tick.get("ask", 0.0) or 0.0)
        spread_pips_live = ((ask - bid) / pip_size) if (ask > 0 and bid > 0 and pip_size > 0) else 0.0
        state = getattr(self.risk_manager, "_spread_state", {}) or {}
        spread_state = float(state.get("current_spread", 0.0) or 0.0)
        spread_pips = (
            float(spread_pips_live)
            if spread_pips_live > 0
            else (spread_state if spread_state > 0 else float(getattr(self.settings.risk, "backtest_spread_pips", 1.5) or 1.5))
        )
        slippage_state = float(state.get("current_slippage", 0.0) or 0.0)
        slippage_pips = slippage_state if slippage_state > 0 else float(getattr(self.settings.risk, "slippage_pips", 0.5) or 0.5)
        try:
            bucket = self.risk_manager._session_bucket_utc(datetime.now(UTC))
            if bucket in {"london", "newyork"}:
                slippage_pips *= 1.20
            elif bucket == "asia":
                slippage_pips *= 0.85
        except Exception:
            pass
        commission_per_lot = float(getattr(self.settings.risk, "commission_per_lot", 7.0) or 7.0)
        _pip_sz, pip_value_per_lot = self.risk_manager._compute_pip_metrics(symbol_info)
        commission_pips = commission_per_lot / max(pip_value_per_lot, 1e-9)
        total_cost_pips = max(0.0, spread_pips + slippage_pips + commission_pips)
        expected_profit_pips = max(0.0, float(sl_pips) * max(0.0, float(rr)))
        passed = expected_profit_pips >= (min_mult * total_cost_pips)
        if not passed:
            logger.info(
                "Cost/edge gate rejected trade: exp_pips=%.3f cost_pips=%.3f need>=%.3f",
                expected_profit_pips,
                total_cost_pips,
                min_mult * total_cost_pips,
            )
        return passed

    def _build_order_legs(
        self,
        *,
        total_size: float,
        signal: int,
        entry_price: float,
        sl: float,
        sl_dist: float,
        default_tp: float,
    ) -> list[tuple[float, float]]:
        if not bool(getattr(self.settings.risk, "partial_take_profit_enabled", True)):
            return [(round(float(total_size), 2), float(default_tp))]
        min_total = float(getattr(self.settings.risk, "partial_tp_min_total_lot", 0.03) or 0.03)
        if float(total_size) < min_total:
            return [(round(float(total_size), 2), float(default_tp))]

        levels = self._parse_float_list(getattr(self.settings.risk, "partial_tp_r_levels", "1.0,2.0,3.0"), [1.0, 2.0, 3.0])
        fracs = self._parse_float_list(getattr(self.settings.risk, "partial_tp_size_fracs", "0.5,0.25,0.25"), [0.5, 0.25, 0.25])
        n = min(len(levels), len(fracs))
        if n <= 0:
            return [(round(float(total_size), 2), float(default_tp))]
        levels = [max(0.1, float(v)) for v in levels[:n]]
        fracs = [max(0.0, float(v)) for v in fracs[:n]]
        frac_sum = float(sum(fracs))
        if frac_sum <= 0:
            return [(round(float(total_size), 2), float(default_tp))]
        fracs = [f / frac_sum for f in fracs]

        raw_vols = [float(total_size) * f for f in fracs]
        vols = [int(v * 100.0) / 100.0 for v in raw_vols]
        rem = round(float(total_size) - sum(vols), 2)
        if rem > 0 and vols:
            max_i = int(np.argmax(np.asarray(vols, dtype=float)))
            vols[max_i] = round(vols[max_i] + rem, 2)

        legs: list[tuple[float, float]] = []
        for vol, r in zip(vols, levels, strict=False):
            if vol < 0.01:
                continue
            tp = float(entry_price + (r * sl_dist)) if signal == 1 else float(entry_price - (r * sl_dist))
            legs.append((round(float(vol), 2), tp))

        if not legs:
            return [(round(float(total_size), 2), float(default_tp))]
        return legs

    def _calculate_sl_pips(self, result, frames) -> float | None:
        symbol = self.settings.system.symbol
        # Check recommended
        if result.recommended_sl is not None:
            try:
                val = float(result.recommended_sl.iloc[-1])
                if val > 0 and np.isfinite(val):
                    self._last_rr = None
                    return val
            except Exception as e:
                logger.debug(f"Could not extract recommended_sl: {e}")

        mode = str(getattr(self.settings.risk, "stop_target_mode", "blend") or "blend").strip().lower()
        prefer_stop_engine = mode in {"blend", "smart", "hybrid", "adaptive", "auto", "structure", "market_structure", "swing"}
        prefer_atr = mode in {"atr", "atr_only"}
        allow_chandelier = bool(getattr(self.settings.risk, "chandelier_enabled", True)) and (
            mode in {"chandelier", "blend", "smart", "hybrid", "adaptive", "auto"} or not prefer_stop_engine
        )

        base_df = frames.get(self.settings.system.base_timeframe)
        if base_df is None:
            base_df = frames.get("M1")
        if base_df is None or base_df.empty:
            logger.debug("Missing base timeframe data for SL calculation")
            return None

        def _stop_target_candidate() -> tuple[float, float] | None:
            try:
                res = infer_stop_target_pips(
                    base_df,
                    settings=self.settings,
                    pip_size=self._get_pip_size(symbol),
                    signal=int(getattr(result, "signal", 0) or 0),
                )
                if res is None:
                    return None
                sl_pips, _tp_pips, rr = res
                if sl_pips > 0 and np.isfinite(sl_pips):
                    return float(sl_pips), float(rr)
            except Exception as e:
                logger.debug(f"Stop-target engine failed: {e}")
            return None

        def _chandelier_candidate() -> float | None:
            try:
                if not {"high", "low", "close"}.issubset(base_df.columns):
                    return None
                period = int(getattr(self.settings.risk, "chandelier_period", 22) or 22)
                period = max(5, period)
                high = base_df["high"].astype(float)
                low = base_df["low"].astype(float)
                close = base_df["close"].astype(float)
                atr = base_df["atr"].astype(float) if "atr" in base_df.columns else None
                if atr is None or atr.empty:
                    tr1 = (high - low).abs()
                    tr2 = (high - close.shift(1)).abs()
                    tr3 = (low - close.shift(1)).abs()
                    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14, min_periods=2).mean()
                atr_last = float(atr.iloc[-1]) if len(atr) > 0 else 0.0
                if not np.isfinite(atr_last) or atr_last <= 0:
                    return None
                mult = float(getattr(self.settings.risk, "chandelier_atr_multiplier", 3.0) or 3.0)
                pip_size = self._get_pip_size(symbol)
                close_px = float(close.iloc[-1])
                signal = int(getattr(result, "signal", 0) or 0)
                if signal == 1:
                    hh = float(high.tail(period).max())
                    stop_px = hh - (mult * atr_last)
                    dist = max(0.0, close_px - stop_px)
                elif signal == -1:
                    ll = float(low.tail(period).min())
                    stop_px = ll + (mult * atr_last)
                    dist = max(0.0, stop_px - close_px)
                else:
                    dist = 0.0
                if dist > 0 and pip_size > 0:
                    return float(dist / pip_size)
            except Exception as e:
                logger.debug(f"Chandelier SL calculation failed: {e}")
            return None

        def _atr_candidate() -> float | None:
            try:
                atr = base_df["atr"].iloc[-1] if "atr" in base_df.columns else None
                pip_size = self._get_pip_size(symbol)
                if atr and atr > 0:
                    atr_mult = float(getattr(self.settings.risk, "atr_stop_multiplier", 1.5))
                    min_dist = float(getattr(self.settings.risk, "meta_label_min_dist", 0.0))
                    dist = float(atr) * max(atr_mult, 0.0)
                    if min_dist > 0:
                        dist = max(dist, min_dist)
                    if dist > 0 and pip_size > 0:
                        return float(dist / pip_size)
            except Exception as e:
                logger.debug(f"ATR-based SL calculation failed: {e}")
            return None

        if prefer_stop_engine:
            stop_candidate = _stop_target_candidate()
            if stop_candidate is not None:
                sl, rr = stop_candidate
                self._last_rr = rr
                return sl

        if allow_chandelier:
            sl = _chandelier_candidate()
            if sl is not None:
                self._last_rr = None
                return sl

        if prefer_atr or not prefer_stop_engine:
            sl = _atr_candidate()
            if sl is not None:
                self._last_rr = None
                return sl

        if not prefer_stop_engine:
            stop_candidate = _stop_target_candidate()
            if stop_candidate is not None:
                sl, rr = stop_candidate
                self._last_rr = rr
                return sl
        else:
            sl = _atr_candidate()
            if sl is not None:
                self._last_rr = None
                return sl

        return None

    def _calculate_prices(
        self, result, frames, sl_pips, info, tick_price: dict[str, float]
    ) -> tuple[float, float, float, float, float] | None:
        symbol = self.settings.system.symbol
        try:
            base_df = frames.get(self.settings.system.base_timeframe)
            if base_df is None:
                base_df = frames.get("M1")
            if base_df is None or base_df.empty:
                raise RuntimeError("Missing base timeframe data for price calc")
            close_price = float(base_df["close"].iloc[-1])
            bid = float(tick_price.get("bid", 0.0) or 0.0)
            ask = float(tick_price.get("ask", 0.0) or 0.0)
            # Use live bid/ask when available; otherwise fall back to last close
            if result.signal == 1:  # buy uses ask
                entry_price = ask if ask > 0 else close_price
            else:  # sell uses bid
                entry_price = bid if bid > 0 else close_price

            pip_size = self._get_pip_size(symbol, info)
            sl_dist = sl_pips * pip_size

            rr = None
            if result.recommended_rr is not None:
                try:
                    val = float(result.recommended_rr.iloc[-1])
                    if val > 0 and np.isfinite(val):
                        rr = val
                except Exception as e:
                    logger.debug(f"Could not extract recommended_rr: {e}")
            if rr is None and self._last_rr is not None:
                rr = float(self._last_rr)
            if rr is None:
                rr_cfg = float(getattr(self.settings.risk, "min_risk_reward", 0.0) or 0.0)
                if rr_cfg > 0:
                    rr = rr_cfg
            if rr is None:
                return None

            if result.signal == 1:
                return entry_price - sl_dist, entry_price + (rr * sl_dist), entry_price, sl_dist, float(rr)
            else:
                return entry_price + sl_dist, entry_price - (rr * sl_dist), entry_price, sl_dist, float(rr)
        except Exception as e:
            logger.error(f"Price calc failed: {e}")
            return None

    def _handle_success(self, result, order_type, size, sl, tp, signal_result, *, count_trade: bool = True):
        logger.info(f"[ORDER SUCCESS] Ticket={result.get('ticket')}")
        if self.risk_manager and count_trade:
            self.risk_manager.on_trade_opened(datetime.now(UTC))

        if self.strategy_ledger and result.get("ticket"):
            try:
                self.strategy_ledger.log_intent(
                    ticket=result["ticket"],
                    symbol=self.settings.system.symbol,
                    direction=order_type,
                    volume=size,
                    sl=sl,
                    tp=tp,
                    meta_risk_mult=getattr(self.risk_manager, "last_risk_mult", 1.0),
                )
            except Exception as e:
                logger.warning(f"Failed to log trade intent: {e}", exc_info=True)

    def _handle_failure(self, result):
        logger.error(f"[ORDER FAIL] {result.get('reason')}")
        if result.get("requires_manual_check") and self.risk_ledger:
            self.risk_ledger.record("ORDER_UNVERIFIED", f"Critical: {result.get('reason')}", severity="critical")

    def _get_pip_size(self, symbol: str, info=None) -> float:
        sym = (symbol or "").upper()
        if info:
            pt = float(info.get("point", 0.0001) or 0.0001)
            dig = int(info.get("digits", 5) or 5)
            if sym.endswith("JPY") or sym.startswith("JPY"):
                pip_size = pt * (10 if dig >= 3 else 1)
            elif sym.startswith("XAU") or sym.startswith("XAG"):
                pip_size = 0.01
            elif "BTC" in sym or "ETH" in sym or "LTC" in sym:
                pip_size = 1.0
            else:
                # Standard FX logic: 4+ digits = 10 points per pip, else 1
                pip_size = pt * (10 if dig >= 4 else 1)
        else:
            if sym.endswith("JPY") or sym.startswith("JPY"):
                pip_size = 0.01
            elif sym.startswith("XAU") or sym.startswith("XAG"):
                pip_size = 0.01
            elif "BTC" in sym or "ETH" in sym or "LTC" in sym:
                pip_size = 1.0
            else:
                pip_size = 0.0001
        return pip_size
