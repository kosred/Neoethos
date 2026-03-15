from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

from ..core.config import Settings
from ..data.loader import DataLoader
from ..domain.events import PreparedDataset
from ..features.pipeline import FeatureEngineer
from ..strategy.discovery import AutonomousDiscoveryEngine
from ..training.trainer import ModelTrainer
from .data_manager import DataManager
from .discovery_manager import DiscoveryManager
from .global_model_manager import GlobalModelManager

logger = logging.getLogger(__name__)

try:
    import forex_bindings as _fb  # type: ignore
except Exception:  # pragma: no cover - optional native extension
    _fb = None  # type: ignore



# Helper functions removed (moved to DataManager)


# Non-member helper functions moved to .utils


class TrainingService:
    """
    Manages model training, feature engineering for training,
    and strategy discovery cycles.
    """

    def __init__(
        self,
        settings: Settings,
        data_loader: DataLoader,
        trainer: ModelTrainer,
        feature_engineer: FeatureEngineer,
        discovery_engine: AutonomousDiscoveryEngine,
        autotune_hints: Any,
    ):
        self.settings = settings
        self.data_loader = data_loader
        self.trainer = trainer
        self.feature_engineer = feature_engineer
        self.discovery_engine = discovery_engine
        self.autotune_hints = autotune_hints
        
        # Modular managers
        self.data_manager = DataManager(data_loader, settings)
        self.discovery_manager = DiscoveryManager(self.data_manager, settings)
        self.global_model_manager = GlobalModelManager(
            settings, trainer, data_loader, feature_engineer, self.data_manager, self.discovery_manager
        )
        
        self._ray_started = False
        self._progress_path = self.trainer.models_dir / "global_incremental_progress.json"
        self._prop_search_task: asyncio.Task | None = None
        self._discovery_task: asyncio.Task | None = None
        self._prop_search_thread: threading.Thread | None = None
        self._discovery_thread: threading.Thread | None = None

    def _start_background_thread(self, name: str, target) -> threading.Thread:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return thread

    def _start_prop_search_thread(self, symbols: list[str]) -> None:
        if self._prop_search_thread and self._prop_search_thread.is_alive():
            logger.info("[STRATEGY DISCOVERY] Async discovery already running; skipping new launch.")
            return

        def _runner() -> None:
            try:
                asyncio.run(self._run_prop_search_for_symbols(symbols, stop_event=None))
            except Exception as exc:
                logger.warning(f"[STRATEGY DISCOVERY] Background prop search failed: {exc}", exc_info=True)

        logger.info("[STRATEGY DISCOVERY] Running prop search in background thread.")
        self._prop_search_thread = self._start_background_thread("forex-prop-search", _runner)

    def _load_progress(self) -> set[str]:
        """Load completed-symbol list to allow resuming long incremental runs."""
        try:
            data = json.loads(Path(self._progress_path).read_text())
            if isinstance(data, list):
                return {str(s) for s in data}
        except FileNotFoundError:
            return set()
        except Exception as exc:
            logger.warning(f"Failed to load incremental progress: {exc}")
        return set()

    def _save_progress(self, completed: set[str]) -> None:
        try:
            Path(self._progress_path).write_text(json.dumps(sorted(completed)))
        except Exception as exc:
            logger.warning(f"Failed to persist incremental progress: {exc}")


    # _run_prop_search_for_symbols removed (moved to DiscoveryManager)

    @staticmethod
    def _safe_symbol_tag(symbol: str) -> str:
        safe = "".join(c for c in str(symbol or "") if c.isalnum() or c in ("-", "_"))
        return safe or "GLOBAL"

    def _prop_gene_artifact_paths(self, symbol: str) -> list[Path]:
        safe = self._safe_symbol_tag(symbol)
        paths: list[Path] = []
        cache_dir = Path(getattr(self.settings.system, "cache_dir", "cache") or "cache")
        paths.append(cache_dir / f"talib_knowledge_{safe}.json")
        paths.append(cache_dir / "talib_knowledge.json")

        checkpoint = str(
            getattr(
                self.settings.models,
                "prop_search_checkpoint",
                "models/strategy_evo_checkpoint.json",
            )
            or "models/strategy_evo_checkpoint.json"
        )
        ckpt = Path(checkpoint)
        paths.append(ckpt)
        with contextlib.suppress(Exception):
            for candidate in ckpt.parent.glob(f"{ckpt.stem}_{safe}_*{ckpt.suffix}"):
                paths.append(candidate)

        uniq: list[Path] = []
        seen: set[str] = set()
        for p in paths:
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)

        existing = [p for p in uniq if p.exists() and p.is_file()]
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return existing

    async def _run_global_strategy_discovery(
        self,
        datasets: list[tuple[str, PreparedDataset]],
        symbols: list[str],
        *,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Delegates strategy discovery to DiscoveryManager."""
        await self.discovery_manager.run_global_discovery(datasets, symbols, stop_event=stop_event)

    # _load_prop_best_genes removed (moved to DiscoveryManager)

    # _apply_prop_discovered_base_signal removed (moved to DiscoveryManager)


    # inject_rust_mixer_signals removed (moved to mixer_logic.py)



    async def train(self, optimize: bool = True, stop_event: asyncio.Event | None = None) -> None:
        """Run the full training pipeline for the single active symbol."""
        symbol = self.settings.system.symbol
        logger.info(f"Starting training for {symbol}...")
        logger.info("Frame-native mode: loading dataset from Rust features backend.")
        dataset = self.feature_engineer.prepare({}, news_features=None, symbol=symbol)
        if getattr(dataset, "X", None) is None or len(dataset.X) <= 0:
            logger.error("Frame-native mode: no dataset rows available for %s.", symbol)
            return

        await asyncio.to_thread(
            self.trainer.train_all,
            dataset,
            optimize,
            stop_event,
            None,
            None,
        )

        logger.info("Training cycle complete.")
        self._maybe_stop_ray()

    async def train_incremental_all(self, optimize: bool = False, stop_event: asyncio.Event | None = None) -> None:
        """
        Best-effort incremental retraining for the active symbol.
        Runs in a thread to avoid blocking the event loop when triggered from the trading loop.
        """
        symbol = self.settings.system.symbol
        logger.info(f"Starting incremental retraining for drift recovery on {symbol}...")

        try:
            frames = await self.data_loader.get_training_data(symbol)
            if not frames:
                raise RuntimeError("No training data available for incremental retrain")

            dataset = self.feature_engineer.prepare(frames, symbol=symbol)

            # Offload synchronous training work to a thread to keep async loop responsive
            await asyncio.to_thread(self.trainer.train_incremental, dataset, symbol, optimize, stop_event)
            logger.info("Incremental retraining finished.")
        except Exception as exc:
            logger.error(f"Incremental retraining failed: {exc}", exc_info=True)
            raise

    async def train_global(
        self, symbols: list[str], optimize: bool = True, stop_event: asyncio.Event | None = None
    ) -> None:
        """Train one global model across all provided symbols."""
        await self.global_model_manager.train_global(symbols, optimize, stop_event)

    # _train_global_frame_native removed (moved to GlobalModelManager)

    # Global training logic moved to GlobalModelManager
    # HPC Path moved to GlobalModelManager

    def _maybe_start_ray(self) -> None:
        from ..models.rllib_agent import RAY_AVAILABLE, _maybe_init_ray

        try:
            # Always attempt to start Ray when available; ignore feature flags.
            if RAY_AVAILABLE and not self._ray_started:
                if _maybe_init_ray():
                    self._ray_started = True
                    logger.info("Ray initialized for RLlib agents.")
        except Exception as exc:
            logger.warning(f"Ray init skipped: {exc}")

    def _maybe_stop_ray(self) -> None:
        from ..models.rllib_agent import RAY_AVAILABLE

        try:
            if self._ray_started and RAY_AVAILABLE:
                import ray

                if ray.is_initialized():
                    ray.shutdown()
                    logger.info("Ray shutdown.")
        except Exception as e:
            logger.warning(f"Ray shutdown failed: {e}", exc_info=True)


# End of TrainingService.py
