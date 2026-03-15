"""
Orchestrates the main forex trading logic and loops.
"""
# pylint: disable=broad-exception-caught,logging-fstring-interpolation,protected-access
# pylint: disable=attribute-defined-outside-init,unused-variable,missing-function-docstring
# pylint: disable=line-too-long,missing-module-docstring,import-outside-toplevel

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import Settings
from ..execution.drift_monitor import ConceptDriftMonitor
from ..execution.meta_controller import PropMetaState
from ..execution.mt5_state_manager import MT5StateManager
from ..execution.news_service import NewsService
from ..execution.order_execution import OrderExecutor
from ..execution.risk import RiskManager
from ..execution.training_service import TrainingService
from ..features.engine import SignalEngine
from ..models.unsupervised import MarketRegimeClassifier
from ..strategy.fast_backtest import infer_pip_metrics
from ..training.online_learner import OnlineLearner
from . import frame_utils

logger = logging.getLogger(__name__)


def _entry_feature_snapshot(dataset: object) -> dict[str, Any] | None:
    feature_names = getattr(dataset, "feature_names", None)
    names = [str(name) for name in list(feature_names)] if feature_names is not None else None
    return MT5StateManager.build_feature_row_payload(getattr(dataset, "X", None), feature_names=names)


class TradingEngine:
    """
    The main event loop for live trading.
    Decoupled from initialization and training logic.
    """

    def __init__(
        self,
        settings: Settings,
        mt5: MT5StateManager,
        risk: RiskManager,
        signal: SignalEngine,
        news: NewsService,
        executor: OrderExecutor,
        trainer: TrainingService,
        drift: ConceptDriftMonitor,
        learner: OnlineLearner | None,
    ):
        self.settings = settings
        self.mt5 = mt5
        self.risk = risk
        self.signal = signal
        self.news = news
        self.executor = executor
        self.trainer = trainer
        self.drift = drift
        self.learner = learner

        self._poll_interval = settings.system.poll_interval_seconds
        self._consecutive_failures = 0

        self.last_prob = None
        self.last_close_price = None
        self.last_signal = None
        self._feature_monitor_ready = False

        try:
            pip_size, _ = infer_pip_metrics(getattr(self.settings.system, "symbol", ""))
            self._drift_epsilon = max(1e-12, float(pip_size) * 0.1)
        except Exception:
            self._drift_epsilon = 1e-5

        self.regime_classifier = MarketRegimeClassifier()
        model_path = Path("models")
        try:
            loaded = MarketRegimeClassifier.load(model_path)
            if loaded:
                self.regime_classifier = loaded
                logger.info(f"Loaded Regime Classifier: {self.regime_classifier.regime_map}")
        except Exception as e:
            logger.warning(f"Failed to load regime classifier: {e}")

        self.iters = 0
        self._market_closed_logged = False

    async def run_loop(self, stop_event: asyncio.Event | None = None) -> None:
        """Main infinite loop."""
        logger.info("Starting Trading Engine Loop...")

        while True:
            if stop_event and stop_event.is_set():
                logger.info("Stop requested.")
                break

            try:
                self.iters += 1

                # 1. Market Hours Check
                if not await self._process_market_hours():
                    continue

                # 2. Data Fetch & Prep
                frames = await self._fetch_and_prepare_data()
                if not frames:
                    continue

                # 3. State Sync & Regime
                await self._sync_and_update_state(frames)

                # 4. News & Policy
                news_policy = await self._apply_news_policy()

                # 5. Features & Drift Monitoring
                dataset, feature_drift_alert = self._process_features_and_drift(frames)

                # 6. Signal Generation
                result = self._generate_signal(dataset, news_policy, feature_drift_alert)

                # 7. Concept Drift Update
                self._update_concept_drift(frames, result)

                # 8. Execution
                await self._execute_signal_with_risk(result, dataset, news_policy)

                # 9. Precision Sleep
                await self._precision_sleep()

            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _process_market_hours(self) -> bool:
        if not self.risk.is_trading_session():
            now = datetime.now(UTC)
            if now.weekday() >= 5:
                if not self._market_closed_logged:
                    logger.info("Market closed (Weekend). Sleeping intelligently until Monday...")
                    self._market_closed_logged = True
                await asyncio.sleep(3600)
            else:
                if not self._market_closed_logged:
                    logger.info("Market closed (Session break). Sleeping...")
                    self._market_closed_logged = True
                await asyncio.sleep(60)
            return False
        
        if self._market_closed_logged:
            logger.info("Market Open! Resuming trading...")
            self._market_closed_logged = False
        return True

    async def _fetch_and_prepare_data(self) -> dict | None:
        frames = await self.trainer.data_loader.get_live_data(self.settings.system.symbol)
        if not frames:
            self._consecutive_failures += 1
            if self._consecutive_failures > 10:
                logger.error("Critical data failure.")
                await asyncio.sleep(60)
            await asyncio.sleep(5)
            return None
        self._consecutive_failures = 0
        return frames

    async def _sync_and_update_state(self, frames: dict) -> None:
        # Predict regime
        df_m1 = frames.get(self.settings.system.base_timeframe) or frames.get("M1")
        if not frame_utils.frame_empty(df_m1):
            if self.iters % 100 == 0:
                try:
                    fit_df = frames.get("D1", df_m1)
                    self.regime_classifier.fit(fit_df)
                    self.regime_classifier.save(Path("models"))
                except Exception as e:
                    logger.warning(f"Regime refit failed: {e}")
            self._current_regime = self.regime_classifier.predict(df_m1)
        else:
            self._current_regime = "Normal"

        await self.mt5.sync_with_mt5()

    async def _apply_news_policy(self) -> dict:
        news_policy = await self.news.get_news_policy(self.settings.system.symbol)
        self.risk.update_news_state(news_policy)
        
        if bool(news_policy.get("tier1_nearby")) and self.mt5.cached_positions:
            for pos in self.mt5.cached_positions:
                if pos.symbol == self.settings.system.symbol and pos.profit <= 0:
                    try:
                        await self.executor.close_position(pos.ticket, pos.volume, "News protection")
                        logger.info(f"News protection closed ticket={pos.ticket} before high-impact event.")
                    except Exception as exc:
                        logger.warning(f"News protection close failed for ticket={pos.ticket}: {exc}")
        return news_policy

    def _process_features_and_drift(self, frames: dict) -> tuple[Any, bool]:
        base_df = frames.get(self.settings.system.base_timeframe) or frames.get("M1")
        base_idx = frame_utils.frame_index(base_df)
        news_feats = self.news.get_news_features(base_idx)
        dataset = self.trainer.feature_engineer.prepare(frames, news_features=news_feats, symbol=self.settings.system.symbol)
        
        drift_alert = False
        dataset_x = getattr(dataset, "X", None)
        if self.drift and not frame_utils.frame_empty(dataset_x):
            if not self._feature_monitor_ready:
                baseline = min(len(dataset_x), 2000)
                if baseline > 0:
                    self.drift.initialize_feature_monitor(frame_utils.NumpyFrame(dataset_x).tail(baseline), self.settings.system.symbol)
                    self._feature_monitor_ready = True
            
            threshold = float(getattr(self.settings.risk, "feature_drift_threshold", 0.30) or 0.30)
            drift_alert = bool(self.drift.check_feature_drift(dataset_x, threshold=threshold))
            
        return dataset, drift_alert

    def _generate_signal(self, dataset: Any, news_policy: dict, drift_alert: bool) -> Any:
        result = self.signal.generate_ensemble_signals(dataset)
        if drift_alert:
            logger.warning("Feature drift alert triggered; abstaining this cycle.")
            result.signal = 0
            result.confidence = 0.0
            return result
            
        # News override
        try:
            news_conf = float(news_policy.get("news_confidence", 0.0) or 0.0)
            news_sent = float(news_policy.get("news_surprise", 0.0) or 0.0)
            contrarian_thr = float(getattr(self.settings.news, "news_trade_confidence_threshold", 0.90) or 0.90)
            if (bool(getattr(self.settings.news, "news_trade_on_event", False)) and 
                result.signal != 0 and news_conf >= contrarian_thr and abs(news_sent) >= 0.90):
                result.signal = -1 * result.signal
                logger.info(f"News contrarian override applied (conf={news_conf:.2f} sent={news_sent:.2f}).")
        except Exception:
            pass
        return result

    def _update_concept_drift(self, frames: dict, result: Any) -> None:
        df_m1 = frames.get(self.settings.system.base_timeframe) or frames.get("M1")
        current_close = None
        if not frame_utils.frame_empty(df_m1):
            close_arr = frame_utils.frame_column_numpy(df_m1, "close")
            if close_arr.size > 0:
                current_close = float(close_arr[-1])

        if (self.drift and self.last_prob is not None and self.last_close_price is not None and 
            current_close is not None and self.last_signal is not None and self.last_signal != 0):
            
            price_change = current_close - self.last_close_price
            y_true = 1 if price_change > self._drift_epsilon else (-1 if price_change < -self._drift_epsilon else 0)
            self.drift.update(y_true, self.last_prob)

            if self.drift.should_retrain():
                logger.warning("Concept drift detected; triggering background retraining...")
                task = asyncio.create_task(self.trainer.train_incremental_all())
                task.add_done_callback(lambda _: self.drift.reset_after_retrain())

        self.last_prob = getattr(result, "probs", None)
        self.last_signal = getattr(result, "signal", 0)
        self.last_close_price = current_close

    async def _execute_signal_with_risk(self, result: Any, dataset: Any, news_policy: dict) -> None:
        if result.signal == 0:
            return

        equity = self.mt5.get_real_equity()
        regime = getattr(self, "_current_regime", "Normal")
        
        meta_state = PropMetaState(
            daily_dd_pct=(self.risk.day_start_equity - equity) / self.risk.day_start_equity if self.risk.day_start_equity > 0 else 0.0,
            volatility_regime=self._infer_vol_regime(regime),
            recent_win_rate=sum(self.risk.rolling_outcomes) / len(self.risk.rolling_outcomes) if self.risk.rolling_outcomes else 0.5,
            consecutive_losses=self.risk.consecutive_losses,
            model_confidence=result.confidence,
            hour_of_day=datetime.now(UTC).hour,
            market_regime=regime,
        )

        risk_mult, req_conf, allowed = self.risk.meta_controller.get_risk_parameters(meta_state)
        meta_feats = getattr(result, "meta_features", {}) or {}
        market_vol = float(meta_feats.get("market_volatility", 0.0) or 0.0)
        disagreement = float(meta_feats.get("ensemble_disagreement", 0.0) or 0.0)

        trade_allowed, trade_reason = self.risk.check_trade_allowed(
            equity, result.confidence, datetime.now(self.risk._session_tz),
            market_volatility=market_vol, ensemble_disagreement=disagreement
        )

        if not trade_allowed:
            logger.info(f"Risk gate blocked trade: {trade_reason}")
        elif not allowed:
            logger.info(f"Meta-Controller blocked trade (Regime: {regime})")
        elif result.confidence >= req_conf:
            if self.learner and self.learner.is_repeat_mistake(dataset.X):
                logger.warning("Similarity guard blocked trade (matches past loss pattern)")
                return

            # Live symbol/tick update
            with contextlib.suppress(Exception):
                info = await self.mt5.connection.get_symbol_info(self.settings.system.symbol) or {}
                tick = await self.mt5.connection.get_symbol_price(self.settings.system.symbol) or {}
                if tick:
                    self.executor.update_live_cost_state(tick, symbol_info=info)

            await self.executor.execute_signal(
                result, equity, dataset.frames if hasattr(dataset, "frames") else {},
                entry_features=_entry_feature_snapshot(dataset),
                advice_stance=self.news.last_advice.get("stance") if self.news.last_advice else None,
            )
        else:
            logger.debug(f"Trade skipped by confidence gate: conf={result.confidence:.3f} required={req_conf:.3f} regime={regime}")

    async def _precision_sleep(self) -> None:
        now = datetime.now(UTC)
        seconds_to_wait = 60 - now.second - (now.microsecond / 1_000_000.0)
        await asyncio.sleep(max(0.1, seconds_to_wait + 0.05))

    def _infer_vol_regime(self, regime_str: str) -> str:
        if "Volatile" in regime_str: return "high"
        if "Quiet" in regime_str: return "low"
        return "normal"
