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
from ..execution.trade_doctor import TradeDoctor

# Note: ForexBot import removed to avoid circular dependency
# TradingEngine receives individual components, not the bot itself
from ..execution.training_service import TrainingService
from ..features.engine import SignalEngine
from ..models.unsupervised import MarketRegimeClassifier
from ..strategy.fast_backtest import infer_pip_metrics
from ..training.online_learner import OnlineLearner

logger = logging.getLogger(__name__)


class _FrameTailView:
    def __init__(self, data: dict[str, np.ndarray], index: np.ndarray):
        self._data = {str(k): np.asarray(v).reshape(-1) for k, v in data.items()}
        self.columns = list(self._data.keys())
        self.index = np.asarray(index).reshape(-1)

    @property
    def empty(self) -> bool:
        return int(len(self.index)) <= 0

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, key):
        return self._data[str(key)]


def _frame_empty(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(getattr(value, "empty"))
    except Exception:
        pass
    try:
        return int(len(value)) <= 0  # type: ignore[arg-type]
    except Exception:
        return True


def _frame_columns(frame: object) -> list[str]:
    cols = getattr(frame, "columns", None)
    if cols is None:
        return []
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _frame_resolve_column(frame: object, name: str) -> str | None:
    target = str(name).strip().lower()
    for col in _frame_columns(frame):
        if str(col).strip().lower() == target:
            return col
    return None


def _frame_column_numpy(frame: object, name: str, *, dtype: Any = np.float64):
    col = _frame_resolve_column(frame, name)
    if col is None:
        raise KeyError(name)
    values = frame[col]  # type: ignore[index]
    if hasattr(values, "to_numpy"):
        try:
            out = values.to_numpy(dtype=dtype, copy=False)
        except TypeError:
            out = values.to_numpy(dtype=dtype)
        return np.asarray(out, dtype=dtype).reshape(-1)
    return np.asarray(values, dtype=dtype).reshape(-1)


def _is_frame_like(value: object) -> bool:
    return bool(hasattr(value, "columns") and hasattr(value, "index") and hasattr(value, "__getitem__"))


def _frame_tail(value: object, n_rows: int) -> object:
    if value is None:
        return value
    take = max(0, int(n_rows))
    if take <= 0:
        return value
    if hasattr(value, "tail"):
        with contextlib.suppress(Exception):
            return value.tail(take)  # type: ignore[no-any-return]
    if _is_frame_like(value):
        try:
            row_count = int(len(value))  # type: ignore[arg-type]
        except Exception:
            row_count = 0
        if row_count <= 0:
            return value
        start = max(0, row_count - take)
        data: dict[str, np.ndarray] = {}
        cols = _frame_columns(value)
        for col in cols:
            raw = value[col]  # type: ignore[index]
            arr = raw.to_numpy(copy=False) if hasattr(raw, "to_numpy") else np.asarray(raw)
            vec = np.asarray(arr).reshape(-1)
            data[str(col)] = vec[start:row_count]
        idx_obj = getattr(value, "index", None)
        idx_arr = np.asarray(idx_obj).reshape(-1) if idx_obj is not None else np.arange(row_count, dtype=np.int64)
        if idx_arr.size < row_count:
            idx_arr = np.arange(row_count, dtype=np.int64)
        return _FrameTailView(data, idx_arr[start:row_count])
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr[-take:]


def _entry_feature_snapshot(dataset: object) -> dict[str, Any] | None:
    x = getattr(dataset, "X", None)
    feature_names = getattr(dataset, "feature_names", None)
    try:
        names = [str(name) for name in list(feature_names)] if feature_names is not None else None
    except Exception:
        names = None
    return MT5StateManager.build_feature_row_payload(x, feature_names=names)


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
        doctor: TradeDoctor,
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
        self.doctor = doctor
        self.news = news
        self.executor = executor
        self.trainer = trainer
        self.drift = drift
        self.learner = learner

        self._poll_interval = settings.system.poll_interval_seconds
        self._consecutive_failures = 0

        # State for drift monitoring (requires previous prediction vs current outcome)
        self.last_prob = None
        self.last_close_price = None
        self.last_signal = None
        self._feature_monitor_ready = False

        # Drift "neutral" band should be symbol-aware (pip size differs across FX/JPY/metals/crypto).
        try:
            pip_size, _pip_value_per_lot = infer_pip_metrics(getattr(self.settings.system, "symbol", ""))
            self._drift_epsilon = max(1e-12, float(pip_size) * 0.1)  # 0.1 pip
        except Exception:
            self._drift_epsilon = 1e-5

        # Unsupervised Regime Classifier
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
                if not self.risk.is_trading_session():
                    now = datetime.now(UTC)
                    weekday = now.weekday()  # 0=Mon, 6=Sun

                    # Weekend Logic (Saturday=5, Sunday=6)
                    if weekday >= 5:
                        if not self._market_closed_logged:
                            logger.info("Market closed (Weekend). Sleeping intelligently until Monday...")
                            self._market_closed_logged = True
                        # Sleep for 1 hour on weekends to save resources
                        await asyncio.sleep(3600)
                        continue
                    else:
                        if not self._market_closed_logged:
                            logger.info("Market closed (Session break). Sleeping...")
                            self._market_closed_logged = True
                        await asyncio.sleep(60)
                        continue
                else:
                    # Reset logging flag when market opens
                    if self._market_closed_logged:
                        logger.info("Market Open! Resuming trading...")
                        self._market_closed_logged = False

                # 2. Data Fetch (REORDERED: Fetch first to have frames for Doctor)
                frames = await self.trainer.data_loader.get_live_data(self.settings.system.symbol)
                if not frames:
                    self._consecutive_failures += 1
                    if self._consecutive_failures > 10:
                        logger.error("Critical data failure.")
                        await asyncio.sleep(60)
                    await asyncio.sleep(5)
                    continue
                self._consecutive_failures = 0

                # Get current close for drift calculation
                current_close = None
                df_m1 = None
                try:
                    df_m1 = frames.get(self.settings.system.base_timeframe)
                    if df_m1 is None:
                        df_m1 = frames.get("M1")
                    if not _frame_empty(df_m1):
                        close_arr = _frame_column_numpy(df_m1, "close", dtype=np.float64)
                        if close_arr.size > 0:
                            current_close = float(close_arr[-1])
                except Exception:
                    pass

                # 3. Regime Update (Thinking Step)
                regime = "Normal"
                if not _frame_empty(df_m1):
                    # Periodically refit (Self-Tuning)
                    if self.iters % 100 == 0:
                        try:
                            # Use daily data if available for broader context, else M1
                            fit_df = frames.get("D1", df_m1)
                            self.regime_classifier.fit(fit_df)
                            self.regime_classifier.save(Path("models"))
                        except Exception as e:
                            logger.warning(f"Regime refit failed: {e}")

                    # Predict current state
                    regime = self.regime_classifier.predict(df_m1)

                # 4. MT5 Sync & Doctor (Now Doctor receives frames)
                await self.mt5.sync_with_mt5()
                positions = list(self.mt5.cached_positions)

                # Run Trade Doctor with frames
                diagnoses = self.doctor.diagnose(positions, frames)

                # Execute close instructions from Doctor
                for instruction in diagnoses:
                    try:
                        await self.executor.close_position(
                            instruction.ticket, instruction.volume, f"Doctor: {instruction.reason}"
                        )
                        logger.info(f"Doctor closed #{instruction.ticket}: {instruction.reason}")
                    except Exception as e:
                        logger.warning(f"Failed to execute doctor instruction for #{instruction.ticket}: {e}")

                # 5. News & Policy
                news_policy = await self.news.get_news_policy(self.settings.system.symbol)
                self.risk.update_news_state(news_policy)
                if bool(news_policy.get("tier1_nearby")) and positions:
                    # MT5 wrapper currently has no robust SL-modify path; safest fallback is flattening
                    # non-profitable exposure into high-impact events.
                    for pos in positions:
                        if pos.symbol != self.settings.system.symbol:
                            continue
                        try:
                            if float(getattr(pos, "profit", 0.0) or 0.0) <= 0.0:
                                await self.executor.close_position(
                                    pos.ticket,
                                    pos.volume,
                                    "News protection: flattening non-profitable position",
                                )
                                logger.info("News protection closed ticket=%s before high-impact event.", pos.ticket)
                        except Exception as exc:
                            logger.warning("News protection close failed for ticket=%s: %s", pos.ticket, exc)

                # 6. Feature Engineering & Signal
                base_df = frames.get(self.settings.system.base_timeframe)
                if base_df is None:
                    base_df = frames.get("M1")
                base_idx = getattr(base_df, "index", []) if base_df is not None else []
                news_feats = self.news.get_news_features(base_idx)
                dataset = self.trainer.feature_engineer.prepare(
                    frames,
                    news_features=news_feats,
                    symbol=self.settings.system.symbol,
                )
                feature_drift_alert = False
                dataset_x = getattr(dataset, "X", None)
                if self.drift and dataset_x is not None and len(dataset_x) > 0:
                    if not self._feature_monitor_ready:
                        with contextlib.suppress(Exception):
                            baseline_rows = min(len(dataset_x), 2000)
                            if baseline_rows > 0:
                                self.drift.initialize_feature_monitor(
                                    _frame_tail(dataset_x, baseline_rows), self.settings.system.symbol
                                )
                                self._feature_monitor_ready = True
                    with contextlib.suppress(Exception):
                        feature_drift_alert = bool(
                            self.drift.check_feature_drift(
                                dataset_x,
                                threshold=float(getattr(self.settings.risk, "feature_drift_threshold", 0.30) or 0.30),
                            )
                        )

                # 7. Signal
                result = self.signal.generate_ensemble_signals(dataset)
                if feature_drift_alert:
                    logger.warning("Feature drift alert triggered; abstaining this cycle.")
                    meta = dict(getattr(result, "meta_features", {}) or {})
                    meta["feature_drift_alert"] = True
                    result.meta_features = meta
                    result.signal = 0
                    result.confidence = 0.0
                try:
                    news_conf = float(news_policy.get("news_confidence", 0.0) or 0.0)
                    news_sent = float(news_policy.get("news_surprise", 0.0) or 0.0)
                    contrarian_thr = float(getattr(self.settings.news, "news_trade_confidence_threshold", 0.90) or 0.90)
                    if (
                        bool(getattr(self.settings.news, "news_trade_on_event", False))
                        and int(getattr(result, "signal", 0) or 0) != 0
                        and news_conf >= contrarian_thr
                        and abs(news_sent) >= 0.90
                    ):
                        result.signal = int(-1 * int(result.signal))
                        logger.info(
                            "News contrarian override applied (conf=%.2f sent=%.2f).",
                            news_conf,
                            news_sent,
                        )
                except Exception:
                    pass

                # 8. Drift Check (Corrected Logic: Compare LAST prediction with CURRENT outcome)
                if (
                    self.drift
                    and self.last_prob is not None
                    and self.last_close_price is not None
                    and current_close is not None
                    and self.last_signal is not None
                    and int(self.last_signal) != 0
                ):
                    # Determine true label for the *previous* period
                    # If price went up, y_true=1 (Buy). Down, y_true=-1 (Sell). Flat, y_true=0 (Neutral).
                    price_change = current_close - self.last_close_price
                    eps = float(getattr(self, "_drift_epsilon", 1e-5))
                    if price_change > eps:
                        y_true = 1  # Buy/Up
                    elif price_change < -eps:
                        y_true = -1  # Sell/Down
                    else:
                        y_true = 0  # Neutral

                    # Update monitor with (Actual, Predicted_Prob_From_Last_Step)
                    self.drift.update(y_true, self.last_prob)

                    if self.drift.should_retrain():
                        logger.warning("Concept drift detected; triggering background retraining...")
                        task = asyncio.create_task(self._trigger_retraining())
                        if not hasattr(self, "_background_tasks"):
                            self._background_tasks = set()
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)

                # Store current state for NEXT iteration's drift check
                if result and hasattr(result, "probs"):
                    self.last_prob = result.probs
                if result and hasattr(result, "signal"):
                    self.last_signal = result.signal
                if current_close is not None:
                    self.last_close_price = current_close

                # 9. Execution (With Thinking Brain)
                # We need to manually invoke risk check to inject the regime
                equity = self.mt5.get_real_equity()
                entry_features = _entry_feature_snapshot(dataset)
                live_tick = {}
                live_symbol_info = {}
                with contextlib.suppress(Exception):
                    live_symbol_info = await self.mt5.connection.get_symbol_info(self.settings.system.symbol) or {}
                    live_tick = await self.mt5.connection.get_symbol_price(self.settings.system.symbol) or {}
                    if live_tick:
                        self.executor.update_live_cost_state(live_tick, symbol_info=live_symbol_info)

                # Override internal risk check to inject regime
                meta_state = PropMetaState(
                    daily_dd_pct=(self.risk.day_start_equity - equity) / self.risk.day_start_equity
                    if self.risk.day_start_equity > 0
                    else 0.0,
                    volatility_regime=self._infer_vol_regime(regime),
                    recent_win_rate=sum(self.risk.rolling_outcomes) / len(self.risk.rolling_outcomes)
                    if self.risk.rolling_outcomes
                    else 0.5,
                    consecutive_losses=self.risk.consecutive_losses,
                    model_confidence=result.confidence,
                    hour_of_day=datetime.now(UTC).hour,
                    market_regime=regime,
                )

                # Get smart parameters
                risk_mult, req_conf, allowed = self.risk.meta_controller.get_risk_parameters(meta_state)
                meta_feats = getattr(result, "meta_features", {}) or {}
                try:
                    market_volatility = float(meta_feats.get("market_volatility", 0.0) or 0.0)
                except Exception:
                    market_volatility = 0.0
                try:
                    ensemble_disagreement = float(meta_feats.get("ensemble_disagreement", 0.0) or 0.0)
                except Exception:
                    ensemble_disagreement = 0.0
                trade_allowed, trade_reason = self.risk.check_trade_allowed(
                    equity,
                    float(getattr(result, "confidence", 0.0) or 0.0),
                    datetime.now(self.risk._session_tz),
                    market_volatility=market_volatility,
                    ensemble_disagreement=ensemble_disagreement,
                )

                # Inject back into risk manager for this tick
                # We can't easily monkey-patch, but we can respect the outcome
                if result.signal == 0:
                    pass
                elif not trade_allowed:
                    logger.info("Risk gate blocked trade: %s", trade_reason)
                elif not allowed:
                    logger.info(f"Meta-Controller blocked trade (Regime: {regime})")
                elif result.confidence >= req_conf:
                    correlation_blocked = False
                    if bool(getattr(self.settings.risk, "correlation_filter_enabled", True)):
                        cur_sym = str(self.settings.system.symbol or "")
                        cur_bias = self._usd_exposure_bias(cur_sym, int(result.signal))
                        same_bias_open = 0
                        if cur_bias != 0:
                            for pos in positions:
                                if str(pos.symbol).upper() == cur_sym.upper():
                                    continue
                                pos_sig = 1 if int(getattr(pos, "type", 0)) == 0 else -1
                                pos_bias = self._usd_exposure_bias(str(pos.symbol), pos_sig)
                                if pos_bias == cur_bias:
                                    same_bias_open += 1
                        max_corr = int(getattr(self.settings.risk, "max_correlated_positions", 1) or 1)
                        if same_bias_open >= max_corr:
                            logger.info(
                                "Correlation filter blocked trade: %s open USD-correlated positions.",
                                same_bias_open,
                            )
                            correlation_blocked = True
                    # Check repeat mistake guard
                    if correlation_blocked:
                        logger.debug("Trade skipped due to cross-symbol correlation exposure.")
                    elif self.learner and self.learner.is_repeat_mistake(dataset.X):
                        logger.warning("Similarity guard blocked trade (matches past loss pattern)")
                    else:
                        # Pass risk_mult implicitly by scaling size or handling inside executor?
                        # Executor calls risk_manager.calculate_position_size which calls meta_controller.
                        # BUT risk_manager.calculate_position_size re-creates PropMetaState without our regime!
                        # FIX: We need to patch risk manager or pass regime to it.
                        # For now, we trust risk_manager to re-calculate, but we need to inject regime into it?
                        # No, risk_manager doesn't have 'market_regime' field yet.
                        # Wait, we updated PropMetaState definition globally in meta_controller.py
                        # But risk.py creates PropMetaState. It needs to know about 'market_regime'.
                        # See next step.

                        await self.executor.execute_signal(
                            result,
                            self.mt5.get_real_equity(),
                            frames,
                            entry_features=entry_features,
                            advice_stance=self.news.last_advice.get("stance") if self.news.last_advice else None,
                            tick_price=live_tick,
                            symbol_info=live_symbol_info,
                        )
                else:
                    logger.debug(
                        "Trade skipped by confidence gate: conf=%.3f required=%.3f regime=%s",
                        float(result.confidence),
                        float(req_conf),
                        regime,
                    )

                # 10. Online Learning Update
                # ... (Online learning logic)

                # HPC FIX: Market-Aligned Precision Sleep
                # Sync with the next minute boundary to ensure we catch the fresh candle immediately
                now = datetime.now(UTC)
                seconds_to_wait = 60 - now.second - (now.microsecond / 1_000_000.0)
                
                # Add a tiny buffer (50ms) to ensure the broker has the data ready
                await asyncio.sleep(max(0.1, seconds_to_wait + 0.05))

            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    def _infer_vol_regime(self, regime_str: str) -> str:
        if "Volatile" in regime_str:
            return "high"
        if "Quiet" in regime_str:
            return "low"
        return "normal"

    @staticmethod
    def _usd_exposure_bias(symbol: str, signal: int) -> int:
        sym = "".join(ch for ch in str(symbol or "").upper() if ch.isalpha())
        if len(sym) < 6 or signal == 0:
            return 0
        base, quote = sym[:3], sym[3:6]
        sgn = 1 if signal > 0 else -1
        if quote == "USD":
            # Buy EURUSD => short USD (-1); Sell EURUSD => long USD (+1)
            return -sgn
        if base == "USD":
            # Buy USDJPY => long USD (+1); Sell USDJPY => short USD (-1)
            return sgn
        return 0

    async def _trigger_retraining(self) -> None:
        """Background retraining triggered by drift detection."""
        try:
            logger.info("Starting drift-triggered retraining...")
            await self.trainer.train_incremental_all()
            logger.info("Drift retraining completed")
            self.drift.reset_after_retrain()
        except Exception as e:
            logger.error(f"Drift retraining failed: {e}", exc_info=True)

