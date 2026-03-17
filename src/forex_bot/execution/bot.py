import asyncio
import logging
from pathlib import Path
import forex_bindings as fb

from ..core.config import Settings
from ..core.system import AutoTuner, HardwareProbe
from ..data.loader import DataLoader
from .risk import RiskManager
from .trading_loop import TradingEngine
from .mt5_state_manager import MT5StateManager
from .news_service import NewsService

logger = logging.getLogger(__name__)

class ForexBot:
    """
    The Coordinator.
    Initializes Pure-Rust services via forex_bindings and delegates execution.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # 1. Hardware & Core
        self._rust_core = fb.ForexCore()
        self.hardware_profile = self._rust_core.detect_hardware()
        
        # 2. Rust-Backed Components
        self.data_loader = DataLoader(settings)
        self.risk_manager = RiskManager(settings)
        
        # 3. Rust Orchestrators
        self._rust_trainer = fb.TrainingOrchestrator(
            config_path=str(settings.config_path) if hasattr(settings, "config_path") else "config.yaml",
            models_dir="models"
        )

        self.news_service = NewsService(settings, risk_ledger=getattr(self.risk_manager, "risk_ledger", None))
        
        # MT5 Manager (lazy init)
        self.mt5_manager = None

    async def train(self, optimize: bool = True, stop_event: asyncio.Event | None = None) -> None:
        """Delegate to Pure-Rust Training Orchestrator."""
        symbol = self.settings.system.symbol or "EURUSD"
        base_tf = self.settings.system.base_timeframe or "M1"
        await asyncio.to_thread(self._rust_trainer.train_symbol, symbol, base_tf)

    async def run(self, paper_mode: bool = True, stop_event: asyncio.Event | None = None) -> None:
        """Start the Trading Engine loop with Rust-native components."""
        logger.info("Initializing Pure-Rust Runtime...")

        try:
            # Connect Data (MT5)
            await self.data_loader.connect()
            if self.settings.system.mt5_required and not self.data_loader.is_connected():
                raise RuntimeError("MT5 Connection Failed.")

            # Init MT5 State
            self.mt5_manager = MT5StateManager(self.data_loader.mt5_adapter.connection, self.settings)

            # Rust Risk & Order Execution
            order_executor = fb.OrderExecutor(
                symbol=self.settings.system.symbol or "EURUSD",
                partial_take_profit_enabled=True
            )

            # Create Engine
            engine = TradingEngine(
                settings=self.settings,
                mt5=self.mt5_manager,
                risk=self.risk_manager,
                news=self.news_service,
                executor=order_executor,
                drift=fb.ConceptDriftMonitor(window_size=500), # Rust native
                consistency=fb.ConsistencyTracker() # Rust native
            )

            # Run Loop
            await engine.run_loop(stop_event)

        finally:
            if hasattr(self.data_loader, "disconnect"):
                await self.data_loader.disconnect()
