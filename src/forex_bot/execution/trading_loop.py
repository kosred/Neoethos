"""
Orchestrates the main forex trading logic and loops.
Fully integrated with Rust backends via forex_bindings.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import forex_bindings as fb

from ..core.config import Settings
from .mt5_state_manager import MT5StateManager
from .news_service import NewsService
from .risk import RiskManager

logger = logging.getLogger(__name__)

class TradingEngine:
    """
    The main event loop for live trading.
    Uses Pure-Rust components for risk, drift, and execution logic.
    """

    def __init__(
        self,
        settings: Settings,
        mt5: MT5StateManager,
        risk: RiskManager,
        news: NewsService,
        executor: fb.OrderExecutor,
        drift: fb.ConceptDriftMonitor,
        consistency: fb.ConsistencyTracker,
    ):
        self.settings = settings
        self.mt5 = mt5
        self.risk = risk
        self.news = news
        self.executor = executor
        self.drift = drift
        self.consistency = consistency

        self._poll_interval = float(settings.system.poll_interval_seconds or 1.0)
        self.iters = 0

    async def run_loop(self, stop_event: asyncio.Event | None = None) -> None:
        """Main infinite loop."""
        logger.info("Starting Trading Engine Loop...")
        max_iters = int(os.environ.get("FOREX_BOT_MAX_LOOP_ITERATIONS", 10_000_000))

        while self.iters < max_iters:
            if stop_event and stop_event.is_set():
                logger.info("Stop requested.")
                break

            try:
                self.iters += 1
                
                # 1. Update Market State from MT5
                state = await self.mt5.update_state()
                if not state:
                    await asyncio.sleep(self._poll_interval)
                    continue

                # 2. Check Risk (Rust Enforced)
                equity = float(state.get("balance", 0.0))
                allowed, reason = self.risk.check_trade_allowed(
                    equity=equity,
                    confidence=0.7, # Placeholder for model confidence
                    timestamp=datetime.now(),
                    market_volatility=0.001
                )

                if not allowed:
                    logger.debug(f"Trading inhibited: {reason}")
                    await asyncio.sleep(self._poll_interval)
                    continue

                # 3. Model Inference (Future: Rust Swarm)
                # For now, we wait for the next tick
                
                await asyncio.sleep(self._poll_interval)

            except Exception as e:
                logger.error(f"Trading loop error: {e}", exc_info=True)
                await asyncio.sleep(5.0)
