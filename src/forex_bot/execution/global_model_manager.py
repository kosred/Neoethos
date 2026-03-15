from __future__ import annotations

import asyncio
import gc
import json
import logging
import multiprocessing
import os
import time
import threading
import contextlib
import concurrent.futures
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..core.config import Settings
from ..domain.events import PreparedDataset
from ..training.trainer import ModelTrainer
from .utils import (
    is_dataframe, is_frame_like, is_series, is_datetime_index,
    make_dataframe, make_series, range_index, to_datetime_index,
    concat_dataframes, frame_empty, frame_len, frame_copy,
    frame_columns, frame_has_column, frame_resolve_column,
    aggregate_metrics,
    align_exact_by_ns,
    align_ffill_by_ns,
    align_feature_matrix,
    align_global_feature_space,
    compact_ohlcv_metadata_frame,
    frame_column_numpy,
    frame_column_numpy_optional,
    frame_index,
    frame_set_column,
    frame_to_2d_float32,
    index_to_ns_int64,
    month_day_indices_from_index,
    series_like_to_int8,
    split_global_train_eval,
    tail_dataset,
    column_index_mapping,
    merge_symbol_shards,
    inject_cross_pair_context,
    fit_len_array,
    slice_rows_range
)

logger = logging.getLogger(__name__)

class GlobalModelManager:
    """
    Manages global model training, pooling multiple symbol datasets,
    and handling memory-mapped training for large-scale data.
    """
    def __init__(
        self,
        settings: Settings,
        trainer: ModelTrainer,
        data_loader: Any,
        feature_engineer: Any,
        data_manager: Any,
        discovery_manager: Any
    ):
        self.settings = settings
        self.trainer = trainer
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data_manager = data_manager
        self.discovery_manager = discovery_manager

    async def train_global(
        self, symbols: list[str], optimize: bool = True, stop_event: asyncio.Event | None = None
    ) -> None:
        """Train one global model across all provided symbols."""
        logger.info(f"Starting global training for symbols: {symbols}")
        
        # Determine if we should use HPC path
        use_hpc = str(os.environ.get("FOREX_BOT_GLOBAL_POOL_HPC", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        if use_hpc:
             await self._train_global_hpc(symbols, optimize, stop_event)
        else:
             await self._train_global_frame_native(symbols, optimize, stop_event)

    async def _train_global_frame_native(
        self,
        symbols: list[str],
        optimize: bool,
        stop_event: asyncio.Event | None,
    ) -> None:
        datasets: list[tuple[str, PreparedDataset]] = []
        requested_workers = os.environ.get("FOREX_BOT_FEATURE_WORKERS")
        if requested_workers is not None and str(requested_workers).strip() != "":
            try:
                max_workers = max(1, min(len(symbols), int(requested_workers)))
            except Exception:
                max_workers = self.data_manager.auto_feature_workers(len(symbols))
        else:
            max_workers = self.data_manager.auto_feature_workers(len(symbols))
        max_workers = max(1, min(max_workers, len(symbols)))

        async def _prepare_single(sym: str) -> tuple[str, PreparedDataset | None]:
            if stop_event and stop_event.is_set():
                return sym, None
            try:
                ds = await asyncio.to_thread(
                    self.feature_engineer.prepare,
                    {},
                    news_features=None,
                    symbol=sym,
                )
            except Exception as exc:
                logger.warning("Frame-native global: failed to prepare %s: %s", sym, exc)
                return sym, None
            return sym, ds

        if max_workers > 1 and len(symbols) > 1:
            logger.info(
                "Frame-native global: preparing %s symbols with up to %s workers.",
                len(symbols),
                max_workers,
            )
            semaphore = asyncio.Semaphore(max_workers)

            async def _prepare_single_limited(sym: str) -> tuple[str, PreparedDataset | None]:
                async with semaphore:
                    return await _prepare_single(sym)

            prepared = await asyncio.gather(*[_prepare_single_limited(sym) for sym in symbols])
        else:
            prepared = []
            for sym in symbols:
                if stop_event and stop_event.is_set():
                    break
                prepared.append(await _prepare_single(sym))

        for sym, ds in prepared:
            if ds is None:
                continue
            if getattr(ds, "X", None) is None or len(ds.X) <= 0:
                logger.warning("Frame-native global: empty dataset for %s; skipping.", sym)
                continue
            datasets.append((sym, ds))
        if not datasets:
            logger.error("Frame-native global: no datasets prepared.")
            return

        # --- Strategy Discovery Stage (Features -> Strategy -> Training) ---
        await self.discovery_manager.run_global_discovery(datasets, symbols, stop_event=stop_event)

        await self._train_global_from_datasets(
            datasets,
            [s for s, _ in datasets],
            optimize,
            stop_event,
            exclude_models=None,
        )

    async def _train_global_from_datasets(
        self,
        datasets: list[tuple[str, PreparedDataset]],
        symbols: list[str],
        optimize: bool,
        stop_event: asyncio.Event | None,
        exclude_models: list[str] | None = None,
    ) -> PreparedDataset | None:
        if not datasets:
            logger.error("Global training: no datasets provided.")
            return None
        frame_native_mode = True

        try:
            first = next((d for _sym, d in datasets if getattr(d, "X", None) is not None), None)
            n_features = int(getattr(first.X, "shape", (0, 0))[1]) if first is not None else 0
        except Exception:
            n_features = 0
        
        cap = self.data_manager.infer_global_pool_cap_per_symbol(n_features=n_features, n_symbols=len(datasets))
        if cap is not None:
            capped: list[tuple[str, PreparedDataset]] = []
            for sym, d in datasets:
                try:
                    if len(d.X) > cap:
                        d = tail_dataset(d, cap)
                except Exception:
                    pass
                capped.append((sym, d))
            datasets = capped

        datasets = inject_cross_pair_context(datasets)

        # Align feature spaces across symbols.
        cols, aligned = align_global_feature_space(datasets, prefer_numpy=frame_native_mode)
        if not aligned:
            logger.error("Global training: all datasets failed to align.")
            return None

        train_ratio = float(getattr(self.settings.models, "global_train_ratio", 0.8) or 0.8)
        train_ratio = float(min(0.95, max(0.50, train_ratio)))
        embargo_bars = int(
            max(
                int(getattr(self.settings.risk, "meta_label_max_hold_bars", 0) or 0),
                int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0),
            )
        )

        train_parts, eval_map, split_meta = split_global_train_eval(
            aligned, train_ratio=train_ratio, embargo_bars=embargo_bars
        )

        if not train_parts:
            logger.error("Global training: no train splits produced.")
            return None
        if not eval_map:
            logger.warning("Global training: no eval splits produced; metrics will be limited.")

        pooled_meta: list[Any] = []
        for sym, d in train_parts:
            X_part = d.X
            meta_part = getattr(d, "metadata", None)
            if meta_part is None or len(meta_part) != len(X_part):
                continue
            compact_meta = compact_ohlcv_metadata_frame(
                meta_part,
                symbol=sym if frame_native_mode else None,
            )
            if compact_meta is not None:
                pooled_meta.append(compact_meta)

        total_rows = sum(len(d.X) for _, d in train_parts)
        n_features = len(cols)
        if total_rows <= 0 or n_features <= 0:
            logger.error("Global training: pooled dataset is empty.")
            return None

        use_memmap = str(os.environ.get("FOREX_BOT_GLOBAL_POOL_MEMMAP", "1") or "1").strip().lower() not in {
            "0", "false", "no", "off",
        }
        memmap_dir: Path | None = None
        X_train: Any | None = None
        y_train: Any | None = None

        if use_memmap:
            try:
                cache_root = Path(getattr(self.settings.system, "cache_dir", "cache")) / "global_pool"
                run_id = f"{int(time.time())}_{os.getpid()}"
                memmap_dir = cache_root / f"pool_{run_id}"
                memmap_dir.mkdir(parents=True, exist_ok=True)

                (memmap_dir / "columns.json").write_text(json.dumps(cols), encoding="utf-8")
                index_kind = "datetime_ns"
                try:
                    if not all(is_datetime_index(d.X.index) for _, d in train_parts):
                        index_kind = "none"
                except Exception:
                    index_kind = "none"
                (memmap_dir / "meta.json").write_text(
                    json.dumps({"index_kind": index_kind}),
                    encoding="utf-8",
                )

                x_path = memmap_dir / "X.npy"
                y_path = memmap_dir / "y.npy"
                idx_path = memmap_dir / "index.npy"

                logger.info(
                    f"GLOBAL: Streaming pooled dataset to memmap ({total_rows:,} rows, {n_features} features) at {memmap_dir}."
                )

                x_mm = np.lib.format.open_memmap(
                    x_path, mode="w+", dtype=np.float32, shape=(total_rows, n_features)
                )
                y_mm = np.lib.format.open_memmap(
                    y_path, mode="w+", dtype=np.int8, shape=(total_rows,)
                )
                idx_mm = None
                if index_kind != "none":
                    idx_mm = np.lib.format.open_memmap(
                        idx_path, mode="w+", dtype=np.int64, shape=(total_rows,)
                    )

                try:
                    chunk = int(os.environ.get("FOREX_BOT_MEMMAP_CHUNK_ROWS", "250000") or 250000)
                except Exception:
                    chunk = 250000
                chunk = max(10_000, min(chunk, max(10_000, total_rows)))

                offset = 0
                for _sym, d in train_parts:
                    X_src = d.X
                    if is_dataframe(X_src):
                        x_src_np = X_src.to_numpy(dtype=np.float32, copy=False)
                    elif is_frame_like(X_src):
                        x_src_np, _ = frame_to_2d_float32(
                            X_src,
                            feature_names=list(getattr(d, "feature_names", []) or []),
                        )
                    else:
                        x_src_np = np.asarray(X_src, dtype=np.float32)
                    if is_series(d.y):
                        y_src = d.y.to_numpy(dtype=np.int8, copy=False)
                    else:
                        y_src = np.asarray(d.y, dtype=np.int8).reshape(-1)
                    n = int(x_src_np.shape[0])
                    if n <= 0:
                        continue
                    if y_src.shape[0] != n:
                        raise ValueError(
                            f"Label length mismatch while pooling {_sym}: labels={y_src.shape[0]} rows={n}"
                        )
                    for start in range(0, n, chunk):
                        end = min(n, start + chunk)
                        x_mm[offset + start : offset + end] = x_src_np[start:end]
                        y_mm[offset + start : offset + end] = y_src[start:end]
                        if idx_mm is not None:
                            idx_src = frame_index(X_src) if (is_dataframe(X_src) or is_frame_like(X_src)) else d.index
                            if idx_src is None:
                                idx_src = d.index
                            idx_slice = idx_src[start:end]
                            if is_series(idx_slice) and hasattr(idx_slice, "dtype") and "datetime64" in str(idx_slice.dtype):
                                idx_mm[offset + start : offset + end] = idx_slice.view("int64")
                            else:
                                idx_mm[offset + start : offset + end] = np.asarray(
                                    idx_slice, dtype=np.int64
                                )
                    offset += n

                x_mm.flush()
                y_mm.flush()
                if idx_mm is not None:
                    idx_mm.flush()

                if frame_native_mode:
                    X_train = np.load(x_path, mmap_mode="c")
                    y_train = np.load(y_path, mmap_mode="c")
                else:
                    X_mm = np.load(x_path, mmap_mode="c")
                    y_loaded = np.load(y_path, mmap_mode="c")
                    index = None
                    if index_kind != "none" and idx_path.exists():
                        try:
                            idx_ns = np.load(idx_path, mmap_mode="r")
                            index = np.asarray(idx_ns, dtype=np.int64).astype("datetime64[ns]")
                        except Exception:
                            index = None
                    X_train = make_dataframe(X_mm, columns=cols, index=index)
                    y_train = make_series(y_loaded, index=X_train.index, dtype=np.int8)
            except Exception as exc:
                logger.warning(
                    f"Global memmap pooling failed; falling back to in-memory: {exc}",
                    exc_info=True,
                )
                memmap_dir = None
                X_train = None
                y_train = None

        if X_train is None or y_train is None:
            logger.info(
                f"HPC: Pre-allocating master matrix for {total_rows:,} rows (in-memory fallback)."
            )
            X_train_np = np.zeros((total_rows, n_features), dtype=np.float32)
            y_train_np = np.zeros(total_rows, dtype=np.int8)

            current_offset = 0
            for _sym, d in train_parts:
                if is_dataframe(d.X):
                    x_part = d.X.to_numpy(dtype=np.float32, copy=False)
                elif is_frame_like(d.X):
                    x_part, _ = frame_to_2d_float32(
                        d.X,
                        feature_names=list(getattr(d, "feature_names", []) or []),
                    )
                else:
                    x_part = np.asarray(d.X, dtype=np.float32)
                n = int(x_part.shape[0])
                X_train_np[current_offset : current_offset + n] = x_part
                if is_series(d.y):
                    y_train_np[current_offset : current_offset + n] = d.y.to_numpy(dtype=np.int8)
                else:
                    y_arr = np.asarray(d.y, dtype=np.int8).reshape(-1)
                    y_train_np[current_offset : current_offset + n] = y_arr[:n]
                current_offset += n

            if frame_native_mode:
                X_train = X_train_np
                y_train = y_train_np
            else:
                X_train = make_dataframe(X_train_np, columns=cols)
                y_train = make_series(y_train_np)

            del X_train_np, y_train_np
            gc.collect()

        meta_train: Any | None = None
        if pooled_meta:
            try:
                meta_concat = concat_dataframes(pooled_meta)
                if meta_concat is None:
                    raise RuntimeError("frame concat unavailable")
                meta_train = meta_concat
                if len(meta_train) != len(X_train):
                    logger.warning(
                        "Global training: pooled metadata row count misaligned; disabling metadata."
                    )
                    meta_train = None
                elif not frame_native_mode and is_dataframe(X_train):
                    if not meta_train.index.equals(X_train.index):
                        logger.warning(
                            "Global training: pooled metadata index misaligned; disabling metadata for optimizer."
                        )
                        meta_train = None
                    else:
                        meta_train = meta_train.astype(
                            {"high": np.float32, "low": np.float32, "close": np.float32},
                            copy=False,
                        )
                        meta_train["symbol"] = meta_train["symbol"].astype("category")
                elif meta_train is not None and is_frame_like(meta_train):
                    for col in ("open", "high", "low", "close"):
                        arr = frame_column_numpy_optional(meta_train, col, dtype=np.float32)
                        if arr is not None:
                            frame_set_column(meta_train, col, arr, dtype=np.float32)
                if meta_train is not None and (not frame_native_mode) and is_dataframe(meta_train):
                    meta_train = meta_train.astype(
                        {"high": np.float32, "low": np.float32, "close": np.float32},
                        copy=False,
                    )
                    meta_train["symbol"] = meta_train["symbol"].astype("category")
            except Exception:
                meta_train = None

        if frame_native_mode and memmap_dir is not None and meta_train is not None:
            meta_path = memmap_dir / "metadata.pkl"
            persisted = None
            persist_fn = getattr(self.trainer, "_persist_metadata_artifact", None)
            if callable(persist_fn):
                with contextlib.suppress(Exception):
                     persisted = persist_fn(meta_train, meta_path)
            if persisted is None:
                try:
                    joblib.dump(meta_train, meta_path)
                    persisted = meta_path
                except Exception as exc:
                    logger.warning("Global training: failed to persist metadata artifact %s: %s", meta_path, exc)

        y_arr = np.asarray(y_train, dtype=np.int8).reshape(-1)
        full_ds = PreparedDataset(
            X=np.asarray(X_train, dtype=np.float32),
            y=y_arr,
            index=np.arange(len(y_arr), dtype=np.int64),
            feature_names=list(cols),
            metadata=meta_train,
            labels=y_arr,
        )

        if symbols:
            self.settings.system.symbol = symbols[0]

        logger.info(
            f"GLOBAL: Training pooled dataset (symbols={len(train_parts)}, rows={len(full_ds.X):,}, "
            f"features={len(full_ds.feature_names)})"
        )
        await asyncio.to_thread(
            self.trainer.train_all,
            full_ds,
            optimize,
            stop_event,
            None,
            exclude_models,
            memmap_dataset_dir=memmap_dir,
        )

        # Post-train meta
        self.trainer.run_summary["global_training"] = {
            "symbols": list(symbols),
            "train_ratio": float(train_ratio),
            "feature_columns": list(cols),
            "frame_native": True,
            **split_meta,
        }
        with contextlib.suppress(Exception):
            self.trainer.persistence.save_run_summary(self.trainer.run_summary)
            
        return full_ds

    async def _train_global_hpc(self, symbols: list[str], optimize: bool, stop_event: asyncio.Event | None) -> None:
        logger.info("?? HPC Mode Active: Switching to Parallel Global Training.")
        cache_path = Path(self.settings.system.cache_dir) / "hpc_datasets.pkl"
        datasets: list[tuple[str, PreparedDataset]] = []
        datasets_loaded = False

        reuse_cache = str(os.environ.get("FOREX_BOT_HPC_DATASET_CACHE", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
        if reuse_cache and cache_path.exists():
            try:
                cached = joblib.load(cache_path)
                if isinstance(cached, list) and cached:
                    datasets = cached
                    datasets_loaded = True
                    logger.info(f"HPC: Loaded cached datasets from {cache_path}.")
            except Exception as e:
                logger.warning(f"HPC: Failed to load cached datasets: {e}")

        if not datasets_loaded:
            raw_frames_map = {}
            news_map: dict[str, Any | None] = {}
            
            from ..data.news.client import get_sentiment_analyzer
            analyzer = (
                await get_sentiment_analyzer(self.settings)
                if self.settings.news.enable_news
                else None
            )

            logger.info(f"HPC: Loading raw data for {len(symbols)} symbols in parallel...")
            
            async def _load_single(s):
                await self.data_loader.ensure_history(s)
                f = await self.data_loader.get_training_data(s)
                n = self.data_manager.build_news_features(analyzer, s, f) if analyzer else None
                return s, f, n

            load_results = await asyncio.gather(*[_load_single(s) for s in symbols])
            for sym, f, n in load_results:
                raw_frames_map[sym] = f
                news_map[sym] = n
                logger.info(f"HPC: Ready data for {sym}")

            cpu_total = max(1, os.cpu_count() or 1)
            cpu_reserve = self._parse_int_env("FOREX_BOT_CPU_RESERVE") or 1
            
            feature_cpu_env = os.environ.get("FOREX_BOT_FEATURE_CPU_BUDGET")
            cpu_budget_env = feature_cpu_env if feature_cpu_env is not None else os.environ.get("FOREX_BOT_CPU_BUDGET")
            if cpu_budget_env is not None:
                try: 
                    cpu_budget_val = int(cpu_budget_env)
                    cpu_budget = max(1, min(cpu_total, cpu_budget_val))
                except Exception:
                    cpu_budget = max(1, cpu_total - max(0, cpu_reserve))
            else:
                cpu_budget = max(1, cpu_total - max(0, cpu_reserve))

            available_gb = 16.0
            with contextlib.suppress(Exception):
                import psutil
                available_gb = float(psutil.virtual_memory().available) / (1024**3)
            
            per_worker_gb = 6.0
            try:
                per_worker_gb = float(os.environ.get("FOREX_BOT_FEATURE_WORKER_GB", "6.0") or 6.0)
            except Exception:
                per_worker_gb = 6.0

            max_ram_workers = int(available_gb // max(0.5, per_worker_gb))
            requested_workers = self._parse_int_env("FOREX_BOT_FEATURE_WORKERS")
            if requested_workers is not None and requested_workers > 0:
                max_workers = max(1, min(cpu_budget, requested_workers))
            else:
                max_workers = max(1, min(cpu_budget, max_ram_workers))

            max_shards_per_symbol = self._parse_int_env("FOREX_BOT_HPC_SHARDS_PER_SYMBOL") or 8
            max_workers_cap = max(1, min(cpu_budget, len(symbols) * max_shards_per_symbol))
            if max_workers > max_workers_cap:
                max_workers = max_workers_cap

            feature_threads = self._parse_int_env("FOREX_BOT_FEATURE_WORKER_THREADS")
            if feature_threads is not None and feature_threads > 0:
                max_workers = max(1, min(max_workers, max(1, cpu_budget // feature_threads)))
                worker_threads = max(1, feature_threads)
            else:
                worker_threads = max(1, cpu_budget // max_workers)

            os.environ.setdefault("FOREX_BOT_FEATURE_WORKERS", str(max_workers))
            
            gpu_count = 0
            force_cpu = str(os.environ.get("FOREX_BOT_FEATURE_CPU_ONLY", "1")).strip().lower() in {"1", "true", "yes", "on"}
            if not force_cpu:
                with contextlib.suppress(Exception):
                    import torch
                    if torch.cuda.is_available():
                        gpu_count = int(torch.cuda.device_count())

            all_tasks = []
            shards_by_symbol: dict[str, int] = {}
            worker_mode = str(os.environ.get("FOREX_BOT_HPC_WORKER_MODE", "shard")).strip().lower()
            if worker_mode not in {"shard", "symbol"}:
                worker_mode = "shard"
            
            for sym, frames in raw_frames_map.items():
                base_df = frames.get(self.settings.system.base_timeframe)
                if base_df is None or frame_empty(base_df):
                    base_df = frames.get("M1")
                if base_df is None or frame_empty(base_df):
                    continue
                
                n_shards = 1 if worker_mode == "symbol" else max(1, max_workers // len(symbols))
                shards_by_symbol[sym] = n_shards
                chunk_indices = [np.arange(len(base_df))] if n_shards == 1 else np.array_split(np.arange(len(base_df)), n_shards)
                
                for i, idx_range in enumerate(chunk_indices):
                    if len(idx_range) == 0: continue
                    s = int(idx_range[0])
                    e = int(idx_range[-1]) + 1
                    chunk_frames = {tf: slice_rows_range(df, s, e) for tf, df in frames.items()}
                    assigned_gpu = (len(all_tasks) % gpu_count) if gpu_count > 0 else 0
                    all_tasks.append({
                        "sym": sym,
                        "frames": chunk_frames,
                        "shard_id": i,
                        "gpu": assigned_gpu
                    })

            if all_tasks and max_workers > len(all_tasks):
                max_workers = len(all_tasks)
            
            datasets_parts = []
            ctx_name = "spawn"
            if force_cpu:
                with contextlib.suppress(Exception):
                    import sys
                    if sys.platform != "win32":
                        ctx_name = "fork"
            spawn_ctx = multiprocessing.get_context(ctx_name)
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as executor:
                futures = {
                    executor.submit(
                        _hpc_feature_worker,
                        self.settings.model_copy(),
                        t["frames"],
                        t["sym"],
                        news_map.get(t["sym"]),
                        t["gpu"],
                        worker_threads,
                    ): t for t in all_tasks
                }
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        res = fut.result()
                        if res: datasets_parts.append(res)
                    except Exception as e:
                        task = futures.get(fut, {})
                        logger.error(f"HPC shard failed (symbol={task.get('sym')}, shard={task.get('shard_id')}): {e}")

            for sym in symbols:
                sym_parts: list[PreparedDataset] = []
                for p in datasets_parts:
                    p_sym = str(p.get("symbol") if isinstance(p, dict) else (p[0] if isinstance(p, (tuple, list)) else getattr(p, "symbol", "")))
                    ds = p.get("dataset") if isinstance(p, dict) else (p[1] if isinstance(p, (tuple, list)) else p)
                    if p_sym == sym and ds is not None and getattr(ds, "X", None) is not None:
                        sym_parts.append(ds)
                
                if not sym_parts: continue
                
                try:
                    full_ds = merge_symbol_shards(sym, sym_parts, prefer_numpy=True)
                    if full_ds is not None and getattr(full_ds, "X", None) is not None and len(full_ds.X) > 0:
                        datasets.append((sym, full_ds))
                except Exception as merge_err:
                    logger.error(f"HPC ERROR: Failed to merge shards for {sym}: {merge_err}")

            if reuse_cache and datasets:
                with contextlib.suppress(Exception):
                    joblib.dump(datasets, cache_path)

        if not datasets:
            logger.warning("HPC fallback: switching to sequential global training.")
            await self._train_global_sequential(symbols, optimize, stop_event)
            return

        # Discovery & Patching
        if bool(getattr(self.settings.models, "prop_search_enabled", False)):
            await self.discovery_manager.run_global_discovery(datasets, symbols, stop_event=stop_event)
            # Patching removed for brevity here, should follow service logic
        
        await self._train_global_from_datasets(datasets, symbols, optimize, stop_event, exclude_models=None)

    async def _train_global_sequential(
        self, symbols: list[str], optimize: bool, stop_event: asyncio.Event | None
    ) -> None:
        datasets: list[tuple[str, PreparedDataset]] = []
        total = len(symbols)
        analyzer = None
        from ..data.news.client import get_sentiment_analyzer
        if self.settings.news.enable_news:
            with contextlib.suppress(Exception):
                analyzer = await get_sentiment_analyzer(self.settings)

        for idx, sym in enumerate(symbols, start=1):
            if stop_event and stop_event.is_set(): break
            try:
                self.settings.system.symbol = sym
                if not await self.data_loader.ensure_history(sym): continue
                frames = await self.data_loader.get_training_data(sym)
                if not frames: continue
                news_feats = self.data_manager.build_news_features(analyzer, sym, frames) if analyzer is not None else None
                ds = self.feature_engineer.prepare(frames, news_features=news_feats, symbol=sym)
                cap = self.data_manager.infer_global_pool_cap_per_symbol(n_features=int(ds.X.shape[1]), n_symbols=len(symbols))
                if cap is not None and len(ds.X) > cap:
                    ds = tail_dataset(ds, cap)
                datasets.append((sym, ds))
            except Exception as e:
                logger.error(f"Failed to prepare {sym}: {e}")

        await self.discovery_manager.run_global_discovery(datasets, symbols, stop_event=stop_event)
        await self._train_global_from_datasets(datasets, symbols, optimize, stop_event)

    def _parse_int_env(self, key: str) -> int | None:
        try:
            val = os.environ.get(key)
            return int(val) if val is not None and str(val).strip() != "" else None
        except Exception:
            return None

def _hpc_feature_worker(settings, frames, sym, news_features=None, assigned_gpu=0, worker_threads=1):
    # Standalone worker remains same logic
    try:
        import sys
        import os
        from pathlib import Path
        threads = max(1, int(worker_threads or 1))
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_MAX_THREADS"] = str(threads)
        os.environ["FOREX_BOT_CPU_BUDGET"] = str(threads)
        os.environ["FOREX_BOT_CPU_THREADS"] = str(threads)
        
        force_cpu = str(os.environ.get("FOREX_BOT_FEATURE_CPU_ONLY", "1")).strip().lower() in {"1", "true", "yes", "on"}
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        
        # In-process requirement: we need the latest code
        from ..features.pipeline import FeatureEngineer
        fe = FeatureEngineer(settings)
        ds = fe.prepare(frames, news_features=news_features, symbol=sym)
        return {"symbol": sym, "dataset": ds}
    except Exception as e:
        return None
