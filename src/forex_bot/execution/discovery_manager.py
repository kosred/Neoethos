from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import torch

from ..domain.events import PreparedDataset
from ..strategy.discovery_tensor import TensorDiscoveryEngine
from .data_manager import DataManager
from .utils import (
    frame_empty, 
    frame_copy, 
    frame_columns, 
    frame_has_column, 
    frame_index, 
    frame_column_numpy, 
    frame_set_column,
    align_ffill_by_ns,
    fit_len_array
)

logger = logging.getLogger(__name__)

class DiscoveryManager:
    def __init__(self, data_manager: DataManager, settings: Any) -> None:
        self.data_manager = data_manager
        self.settings = settings

    async def run_global_discovery(
        self,
        datasets: list[tuple[str, PreparedDataset]],
        symbols: list[str],
        *,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Orchestrates strategy discovery across all symbols and backends."""
        if not datasets:
            return

        # 1. Select master symbol for discovery (prefer EURUSD)
        target_sym = "EURUSD" if "EURUSD" in [s for s, _ in datasets] else datasets[0][0]
        target_ds = next(ds for sym, ds in datasets if sym == target_sym)
        
        # 2. Run Rust-based Tensor Discovery (Broad Selection)
        try:
            raw_frames = await self.data_manager.prepare_raw_data(target_sym)
            discovery_experts = int(self.settings.models.prop_search_portfolio_size or 500)
            
            discovery_tensor = TensorDiscoveryEngine(
                device="cuda" if self._has_gpu() else "cpu",
                n_experts=discovery_experts,
                settings=self.settings,
            )
            
            # Prepare discovery frames
            discovery_frames, _ = await asyncio.to_thread(
                self.build_discovery_frames,
                raw_frames,
                None, 
                target_sym,
                base_dataset=target_ds
            )

            await asyncio.to_thread(
                discovery_tensor.run_unsupervised_search,
                discovery_frames,
                iterations=1000,
            )
            
            cache_dir = Path(getattr(self.settings.system, "cache_dir", "cache") or "cache")
            knowledge_path = cache_dir / "tensor_knowledge.pt"
            discovery_tensor.save_experts(str(knowledge_path))
            
        except Exception as exc:
            logger.warning(f"Strategy discovery failed: {exc}", exc_info=True)

        # 3. Patch discovered signals into ALL datasets
        await self.patch_signals(datasets)

    def build_discovery_frames(
        self,
        frames: dict[str, Any],
        news_feats: Any | None,
        symbol: str | None,
        base_dataset: PreparedDataset | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Build discovery frames with either base-TF propagation or full per-TF features."""
        from ..features.pipeline import FeatureEngineer
        from .utils import (
            discovery_rust_features_enabled,
            dataset_row_count,
            frame_copy,
            frame_len,
            frame_columns,
            frame_has_column,
            frame_index,
            frame_set_column,
            frame_column_numpy,
            align_ffill_by_ns,
            index_to_ns_int64,
            fit_len_array,
            prepared_dataset_to_frame
        )
        import os
        import contextlib

        full_tf = str(os.environ.get("FOREX_BOT_DISCOVERY_FULL_TF_FEATURES", "1") or "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        base_tf = self.settings.system.base_timeframe

        cfg_tfs = list(getattr(self.settings.system, "higher_timeframes", []) or [])
        if not cfg_tfs:
            cfg_tfs = list(getattr(self.settings.system, "required_timeframes", []) or [])
        timeframes = [base_tf] + cfg_tfs
        timeframes = [tf for tf in dict.fromkeys(timeframes) if tf in frames]
        for tf in frames.keys():
            if tf not in timeframes:
                timeframes.append(tf)
        if not timeframes:
            timeframes = list(frames.keys())

        if not full_tf:
            discovery_frames = frames.copy()
            if base_tf in discovery_frames and base_dataset is not None:
                rich_df = prepared_dataset_to_frame(
                    base_dataset,
                    fallback_frame=discovery_frames.get(base_tf),
                )
                if rich_df is None:
                    rich_df = discovery_frames.get(base_tf)
                discovery_frames[base_tf] = rich_df

            reference_df = discovery_frames.get(base_tf)
            aligned_frames = {}
            for tf in timeframes:
                if tf not in frames or reference_df is None:
                    continue
                local = frame_copy(frames[tf])
                if local is None:
                    local = frames[tf]
                src_idx = index_to_ns_int64(frame_index(reference_df))
                tgt_idx = index_to_ns_int64(frame_index(local))
                tgt_n = frame_len(local)
                for col in frame_columns(reference_df):
                    low = str(col).lower()
                    if low in {"open", "high", "low", "close", "volume"}:
                        continue
                    if frame_has_column(local, col):
                        continue
                    with contextlib.suppress(Exception):
                        src_vals = frame_column_numpy(reference_df, col, dtype=np.float32)
                        aligned = align_ffill_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float32)
                        if aligned is None:
                            aligned = fit_len_array(src_vals, tgt_n, fill=0.0, dtype=np.float32)
                        frame_set_column(local, col, aligned, dtype=np.float32)
                aligned_frames[tf] = local
            aligned_frames = self.inject_mixer_signals(
                aligned_frames if aligned_frames else discovery_frames,
                base_tf=base_tf,
                per_tf=False,
            )
            return aligned_frames, timeframes

        # Full per-TF feature engineering (slow but pure). Prefer Rust feature extraction when enabled.
        prefer_rust_discovery = discovery_rust_features_enabled()
        per_tf = {}
        for tf in timeframes:
            if tf not in frames:
                continue
            try:
                if base_dataset is not None and tf == base_tf:
                    rich_df = prepared_dataset_to_frame(base_dataset, fallback_frame=frames.get(tf))
                else:
                    tf_settings = self.settings.model_copy()
                    tf_settings.system.base_timeframe = tf
                    fe = FeatureEngineer(tf_settings)
                    rich_df = None
                    if prefer_rust_discovery and symbol:
                        ds_tf_rust = fe.prepare({}, news_features=None, symbol=symbol)
                        if dataset_row_count(ds_tf_rust) > 0:
                            rich_df = prepared_dataset_to_frame(ds_tf_rust, fallback_frame=frames.get(tf))
                    if rich_df is None:
                        ds_tf = fe.prepare(frames, news_features=news_feats, symbol=symbol)
                        rich_df = prepared_dataset_to_frame(ds_tf, fallback_frame=frames.get(tf))
                if rich_df is None:
                    continue
                per_tf[tf] = rich_df
            except Exception as exc:
                logger.warning(f"Discovery per-TF feature gen failed for {tf}: {exc}")

        if not per_tf:
            return frames.copy(), timeframes

        all_cols = []
        for df in per_tf.values():
            all_cols.extend(frame_columns(df))
        all_cols = list(dict.fromkeys(all_cols))

        aligned_frames = {}
        for tf, df in per_tf.items():
            aligned = frame_copy(df)
            if aligned is None:
                continue
            n_rows = frame_len(aligned)
            for col in all_cols:
                if frame_has_column(aligned, col):
                    continue
                frame_set_column(aligned, col, np.zeros(n_rows, dtype=np.float32))
            orig = frames.get(tf)
            src_idx = index_to_ns_int64(frame_index(orig))
            tgt_idx = index_to_ns_int64(frame_index(aligned))
            for col in ["open", "high", "low", "close"]:
                if not frame_has_column(orig, col):
                    continue
                with contextlib.suppress(Exception):
                    src_vals = frame_column_numpy(orig, col, dtype=np.float64)
                    vals = align_ffill_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float64)
                    if vals is None:
                        vals = fit_len_array(src_vals, n_rows, fill=0.0, dtype=np.float64)
                    frame_set_column(aligned, col, vals, dtype=np.float64)
            aligned_frames[tf] = aligned

        aligned_frames = self.inject_mixer_signals(
            aligned_frames,
            base_tf=base_tf,
            per_tf=True,
        )
        return aligned_frames, timeframes

    def inject_mixer_signals(
        self,
        frames: dict[str, Any],
        *,
        base_tf: str,
        per_tf: bool,
    ) -> dict[str, Any]:
        """Injects discovery mixer signals using native bindings if available."""
        import os
        import contextlib
        use_mixer = str(os.environ.get("FOREX_BOT_DISCOVERY_USE_TALIB_MIXER", "1") or "1").strip().lower()
        if use_mixer not in {"1", "true", "yes", "on"}:
            return frames

        try:
            n_strategies = int(os.environ.get("FOREX_BOT_DISCOVERY_MIXER_STRATEGIES", "24") or 24)
        except Exception:
            n_strategies = 24
        if n_strategies <= 0:
            return frames
        try:
            max_indicators = int(
                os.environ.get("FOREX_BOT_DISCOVERY_MIXER_MAX_INDICATORS", "0") or 0
            )
        except Exception:
            max_indicators = 0
        if max_indicators <= 0:
            try:
                max_indicators = int(
                    getattr(self.settings.models, "prop_search_max_indicators", 0) or 0
                )
            except Exception:
                max_indicators = 0
        if max_indicators <= 0:
            max_indicators = 3

        with contextlib.suppress(Exception):
            import forex_bindings as _fb # type: ignore
            if hasattr(_fb, "talib_bulk_signals_ohlcv"):
                for tf, df in frames.items():
                    try:
                        o = frame_column_numpy(df, "open", dtype=np.float64)
                        h = frame_column_numpy(df, "high", dtype=np.float64)
                        l = frame_column_numpy(df, "low", dtype=np.float64)
                        c = frame_column_numpy(df, "close", dtype=np.float64)
                        v = frame_column_numpy(df, "volume", dtype=np.float64)
                        
                        signals = _fb.talib_bulk_signals_ohlcv(
                            o, h, l, c, v, n_strategies, max_indicators
                        )
                        if signals is not None and signals.shape[1] > 0:
                            for i in range(signals.shape[1]):
                                frame_set_column(df, f"mixer_sig_{i}", signals[:, i].astype(np.float32))
                    except Exception as e:
                        logger.debug(f"Mixer signal inject failed for {tf}: {e}")
        return frames

    async def patch_signals(self, datasets: list[tuple[str, PreparedDataset]]) -> None:
        """Stub for patching discovered signals back into datasets."""
        # This will be refined as the discovery engine integration matures
        pass

    def normalize_discovery_budget(
        self,
        *,
        experts: int,
        iterations: int,
        has_gpu: bool,
    ) -> tuple[int, int]:
        quick = self._quick_e2e_enabled()
        if has_gpu:
            experts = int(experts)
            iterations = int(iterations)
        else:
            # Keep discovery responsive on CPU-only nodes by default.
            experts = min(int(experts), 40)
            iterations = min(int(iterations), 250)

        if quick:
            # Fast E2E: preserve discovery path but shrink search budget heavily.
            experts = min(int(experts), 4)
            iterations = min(int(iterations), 20)
            experts = max(2, int(experts))
            iterations = max(5, int(iterations))
        else:
            experts = max(8, int(experts))
            iterations = max(50, int(iterations))
        return experts, iterations

    def _quick_e2e_enabled(self) -> bool:
        raw = os.environ.get("FOREX_BOT_QUICK_E2E", "")
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _has_gpu(self) -> bool:
        return bool(getattr(self.settings.system, "enable_gpu", False)) and int(
            getattr(self.settings.system, "num_gpus", 0) or 0
        ) > 0
