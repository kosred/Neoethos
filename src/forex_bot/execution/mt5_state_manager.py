from __future__ import annotations
import asyncio
import logging
from collections import deque
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from .state_models import (
    MT5Position, 
    MT5Deal, 
    OrderRequest, 
    BoundedLRUFeatureStore, 
    _FeatureRowFrame
)
from .persistence_manager import PersistenceManager
from .order_validator import OrderValidator

logger = logging.getLogger(__name__)

class MT5StateManager:
    """
    Synchronizes bot state with MT5 reality.
    Coordinator for account state, position tracking, and order validation.
    """

    def __init__(self, mt5_connection, settings):
        self.mt5 = mt5_connection
        self.settings = settings

        self.pending_requests: dict[str, OrderRequest] = {}
        self.completed_requests: deque = deque(maxlen=100)
        self.last_order_time: datetime | None = None

        self.cached_positions: list[MT5Position] = []
        self.cached_account_info: dict[str, Any] = {}
        self.cached_deals_today: list[MT5Deal] = []
        self.last_sync_time: datetime | None = None

        self.session_start = datetime.now(UTC)
        self.orders_placed_today = 0
        self.confirmed_trades_today = 0

        # Concurrency Lock
        self._order_lock = asyncio.Lock()

        self.consecutive_sync_failures = 0
        self.max_consecutive_sync_failures = 10

        # Modular Components
        self.entry_feature_store = BoundedLRUFeatureStore(max_size=1000)
        self.symbol = settings.system.symbol or "GLOBAL"
        self.persistence = PersistenceManager(self.symbol, self.entry_feature_store)
        self.validator = OrderValidator(settings)
        
        self.persistence.load_entry_store()

    @property
    def connection(self):
        return self.mt5

    async def sync_with_mt5(self, symbol: str | None = None) -> bool:
        """Atomic Symbol-Specific Sync."""
        try:
            account_info = await self.mt5.get_account_information()
            if not account_info:
                return False
            self.cached_account_info = account_info

            positions_raw = await self.mt5.positions_get(symbol=symbol) if symbol else await self.mt5.positions_get()
            
            new_positions = []
            for p in (positions_raw or []):
                try:
                    ticket = p.get("ticket", 0)
                    if not ticket: continue
                    
                    new_positions.append(
                        MT5Position(
                            ticket=ticket,
                            symbol=p.get("symbol", ""),
                            volume=p.get("volume", 0.0),
                            price_open=p.get("price_open", 0.0),
                            price_current=p.get("price_current", 0.0),
                            sl=p.get("sl", 0.0),
                            tp=p.get("tp", 0.0),
                            profit=p.get("profit", 0.0),
                            swap=p.get("swap", 0.0),
                            commission=p.get("commission", 0.0),
                            time=datetime.fromtimestamp(p.get("time", 0), tz=UTC),
                            type=p.get("type", 0),
                            magic=p.get("magic", 0),
                        )
                    )
                except Exception:
                    continue
            
            if symbol:
                self.cached_positions = [pos for pos in self.cached_positions if pos.symbol != symbol]
                self.cached_positions.extend(new_positions)
            else:
                self.cached_positions = new_positions

            self.last_sync_time = datetime.now(UTC)
            self.consecutive_sync_failures = 0
            return True
        except Exception as e:
            self.consecutive_sync_failures += 1
            logger.error(f"MT5 sync failed (attempt {self.consecutive_sync_failures}): {e}")
            if self.consecutive_sync_failures >= self.max_consecutive_sync_failures:
                raise RuntimeError("MT5 sync failed too many times.") from e
            return False

    async def _get_history_deals(self, from_date: datetime, to_date: datetime) -> list[MT5Deal]:
        deals = []
        try:
            from_ts = int(from_date.timestamp())
            to_ts = int(to_date.timestamp())
            deals_raw = await self.mt5.get_history_deals(from_ts, to_ts)

            for d in deals_raw:
                try:
                    deal_id = d.get("deal", 0)
                    timestamp = d.get("time", 0)
                    if not deal_id or deal_id <= 0 or not timestamp or timestamp <= 0:
                        continue
                    deals.append(
                        MT5Deal(
                            deal=deal_id,
                            order=d.get("order", 0),
                            time=datetime.fromtimestamp(timestamp, tz=UTC),
                            symbol=d.get("symbol", ""),
                            type=d.get("type", 0),
                            entry=d.get("entry", 0),
                            volume=d.get("volume", 0.0),
                            price=d.get("price", 0.0),
                            profit=d.get("profit", 0.0),
                            commission=d.get("commission", 0.0),
                            swap=d.get("swap", 0.0),
                            magic=d.get("magic", 0),
                        )
                    )
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Failed to fetch history deals: {e}")
        return deals

    def get_real_equity(self) -> float:
        if not self.cached_account_info:
            logger.warning("get_real_equity() called before sync.")
            return 0.0
        equity = self.cached_account_info.get("equity")
        if equity is None or equity <= 0:
            raise RuntimeError(f"CRITICAL: MT5 Equity Unavailable.")
        return float(equity)

    def get_real_balance(self) -> float:
        balance = self.cached_account_info.get("balance")
        return float(balance) if balance is not None else self.get_real_equity()

    def get_real_margin_free(self) -> float:
        return float(self.cached_account_info.get("margin_free", 0))

    def get_positions_for_symbol(self, symbol: str) -> list[MT5Position]:
        return [p for p in self.cached_positions if p.symbol == symbol]

    async def get_recent_closed_deals(self, limit: int = 10) -> list[dict[str, Any]]:
        to_date = datetime.now(UTC)
        from_date = to_date - timedelta(hours=1)
        try:
            deals_list = await self._get_history_deals(from_date, to_date)
            return [
                {
                    "deal": d.deal, "order": d.order, "magic": d.magic, "time": d.time,
                    "symbol": d.symbol, "profit": d.profit, "volume": d.volume, "price": d.price
                }
                for d in deals_list[-limit:]
            ]
        except Exception:
            return []

    async def get_recent_closed_with_features(self, limit: int = 10) -> list[dict[str, Any]]:
        deals = await self.get_recent_closed_deals(limit=limit)
        matched = []
        if not deals: return matched

        self.persistence.cleanup_entry_store()

        for d in deals:
            payload = self.entry_feature_store.get(d["order"]) or self.entry_feature_store.get(d["deal"])
            if not payload: continue

            features_data = payload.get("features")
            if not features_data: continue
            
            features = _FeatureRowFrame(features_data["columns"], features_data["values"])
            matched.append({
                "profit": d["profit"], "volume": d["volume"], "symbol": d["symbol"],
                "time": d["time"], "features": features, "signal": payload.get("signal", 0),
            })
        return matched

    def record_entry_features(self, ticket: int, symbol: str, bar_time: datetime, features: Any, signal: int, **kwargs) -> None:
        try:
            norm_features = self.validator.build_feature_row_payload(features, feature_names=kwargs.get("feature_names"))
            if norm_features is None: return
            
            payload = {
                "symbol": symbol, "bar_time": bar_time, "features": norm_features, 
                "signal": signal, "magic": kwargs.get("magic"),
            }
            keys = [ticket, kwargs.get("order_ticket"), kwargs.get("deal_ticket"), kwargs.get("magic")]
            for kid in keys:
                if kid: self.entry_feature_store.add(int(kid), payload)
            self.persistence.persist_entry_store()
        except Exception as exc:
            logger.debug(f"Failed to record entry features: {exc}")

    def calculate_real_daily_pnl(self) -> tuple[float, float]:
        realized = sum(d.profit + d.commission + d.swap for d in self.cached_deals_today)
        unrealized = sum(p.profit + p.commission + p.swap for p in self.cached_positions)
        return realized, unrealized

    async def place_order_with_verification(self, symbol: str, order_type: str, volume: float, **kwargs) -> dict[str, Any]:
        async with self._order_lock:
            now = datetime.now(UTC)
            bar_time = kwargs.get("current_bar_time") or now
            magic = kwargs.get("magic", 234567)

            can_place, reason = await self.validator.can_place_order(
                symbol, volume, bar_time, self.cached_positions, self.pending_requests,
                list(self.completed_requests), self.last_order_time, self.get_real_margin_free, self.mt5.get_symbol_info
            )
            if not can_place:
                return {"success": False, "reason": reason, "ticket": None}

            request_id = f"{symbol}_{order_type}_{bar_time.timestamp()}_{volume}"
            order_req = OrderRequest(request_id=request_id, symbol=symbol, order_type=order_type, volume=volume,
                                   sl=kwargs.get("sl"), tp=kwargs.get("tp"), timestamp=now, bar_timestamp=bar_time)
            self.pending_requests[request_id] = order_req

            try:
                if order_type == "buy":
                    result = await self.mt5.create_market_buy_order(symbol, volume, **kwargs)
                else:
                    result = await self.mt5.create_market_sell_order(symbol, volume, **kwargs)

                order_req.result = result
                if result.get("retcode") != 10009:
                    self.pending_requests.pop(request_id, None)
                    self.completed_requests.append(order_req)
                    return {"success": False, "reason": f"MT5 error: {result.get('comment')}", "ticket": None}

                await asyncio.sleep(0.5)
                await self.sync_with_mt5()
                
                deal_ticket = result.get("deal")
                found_position = next((p for p in self.cached_positions if p.ticket == deal_ticket), None)
                
                if found_position:
                    order_req.verified = True
                    self.pending_requests.pop(request_id, None)
                    self.completed_requests.append(order_req)
                    self.last_order_time = now
                    self.orders_placed_today += 1
                    self.confirmed_trades_today += 1
                    return {"success": True, "ticket": found_position.ticket, "position": found_position, 
                            "deal_ticket": deal_ticket, "order_ticket": result.get("order"), "bar_time": bar_time, "magic": magic}
                
                return {"success": False, "reason": "Position not confirmed in MT5", "ticket": deal_ticket, "requires_manual_check": True}
            except Exception as e:
                self.pending_requests.pop(request_id, None)
                return {"success": False, "reason": str(e), "ticket": None}

    async def close_position_by_ticket(self, ticket: int, symbol: str, volume: float | None = None) -> bool:
        try:
            result = await self.mt5.close_position_by_ticket(ticket, symbol, volume=volume)
            success = (isinstance(result, dict) and result.get("retcode") == 10009) or (isinstance(result, int) and result > 0)
            if success:
                await self.sync_with_mt5()
                return True
            return False
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False

    def get_health_status(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        since_sync = (now - self.last_sync_time).total_seconds() if self.last_sync_time else 999
        return {
            "last_sync": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "seconds_since_sync": since_sync,
            "sync_healthy": since_sync < 120,
            "positions_open": len(self.cached_positions),
            "equity": self.get_real_equity(),
        }
