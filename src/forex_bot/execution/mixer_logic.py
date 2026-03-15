import logging
import os
import numpy as np
from typing import Any
from .utils import (
    frame_empty, frame_columns, frame_copy, frame_column_numpy,
    frame_has_column, frame_set_column, index_to_ns_int64,
    align_ffill_by_ns, fit_len_array, frame_len, frame_index
)

logger = logging.getLogger(__name__)

def inject_rust_mixer_signals(
    settings: Any,
    frames: dict[str, Any],
    *,
    base_tf: str = "M1",
    n_strategies: int = 40,
    max_indicators: int = 15,
    per_tf: bool = False,
) -> dict[str, Any]:
    """
    Injects synthetic 'mixer' signals generated via Rust-accelerated TALIB bulk compute.
    This creates a set of randomized indicator combinations to expand the feature space.
    """
    try:
        import forex_bindings as fb
    except ImportError:
        fb = None

    if fb is not None and hasattr(fb, "talib_bulk_signals_ohlcv"):
        raw_pool = (
            str(
                os.environ.get(
                    "FOREX_BOT_DISCOVERY_RUST_INDICATORS",
                    "RSI,ADX,MACD,ATR,NATR,EMA,SMA,CCI,ROC,MOM",
                )
                or "RSI,ADX,MACD,ATR,NATR,EMA,SMA,CCI,ROC,MOM"
            )
        )
        pool = [s.strip().upper() for s in raw_pool.split(",") if str(s).strip()]
        if not pool:
            pool = ["RSI", "ADX", "MACD", "ATR", "NATR", "EMA", "SMA"]
        max_k = max(1, min(int(max_indicators), len(pool)))
        try:
            seed = int(os.environ.get("FOREX_BOT_DISCOVERY_MIXER_SEED", "1337") or 1337)
        except Exception:
            seed = 1337
        rng = np.random.default_rng(seed)
        indicator_sets: list[list[str]] = []
        weight_sets: list[list[float]] = []
        long_thresholds: list[float] = []
        short_thresholds: list[float] = []
        for _ in range(n_strategies):
            k = int(rng.integers(1, max_k + 1))
            inds = rng.choice(np.asarray(pool, dtype=object), size=k, replace=False).tolist()
            indicator_sets.append([str(x).upper() for x in inds])
            weight_sets.append((0.5 + rng.random(k) * 1.0).astype(np.float64).tolist())
            long_thresholds.append(float(0.4 + rng.random() * 0.6))
            short_thresholds.append(float(-1.0 + rng.random() * 0.6))

        try:
            causal_min_bars = int(os.environ.get("FOREX_BOT_TALIB_CAUSAL_MIN_BARS", "30") or 30)
        except Exception:
            causal_min_bars = 30
        causal_min_bars = max(2, causal_min_bars)

        def _rust_apply(df: Any) -> Any:
            if frame_empty(df):
                return df
            cols = {str(c).lower() for c in frame_columns(df)}
            if not {"open", "high", "low", "close"}.issubset(cols):
                return df
            local = frame_copy(df)
            if local is None:
                return df
            open_arr = frame_column_numpy(local, "open", dtype=np.float64)
            high_arr = frame_column_numpy(local, "high", dtype=np.float64)
            low_arr = frame_column_numpy(local, "low", dtype=np.float64)
            close_arr = frame_column_numpy(local, "close", dtype=np.float64)
            volume_arr = (
                frame_column_numpy(local, "volume", dtype=np.float64)
                if bool(getattr(settings.system, "use_volume_features", False)) and frame_has_column(local, "volume")
                else None
            )
            try:
                raw = fb.talib_bulk_signals_ohlcv(
                    open_arr, high_arr, low_arr, close_arr,
                    indicator_sets=indicator_sets,
                    weight_sets=weight_sets,
                    long_thresholds=long_thresholds,
                    short_thresholds=short_thresholds,
                    volume=volume_arr,
                    include_raw=False,
                    causal_min_bars=causal_min_bars,
                )
            except Exception:
                # Fallback if causal_min_bars is not supported in the binding version
                raw = fb.talib_bulk_signals_ohlcv(
                    open_arr, high_arr, low_arr, close_arr,
                    indicator_sets=indicator_sets,
                    weight_sets=weight_sets,
                    long_thresholds=long_thresholds,
                    short_thresholds=short_thresholds,
                    volume=volume_arr,
                    include_raw=False,
                )
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim != 2:
                return local
            if arr.shape[0] == len(indicator_sets) and arr.shape[1] == len(local):
                arr = arr.T
            if arr.shape[0] != len(local):
                return local
            width = min(arr.shape[1], len(indicator_sets))
            for idx in range(width):
                col = f"tmx_sig_{idx}"
                if frame_has_column(local, col):
                    continue
                frame_set_column(local, col, np.asarray(arr[:, idx], dtype=np.float32), dtype=np.float32)
            return local

        try:
            if per_tf:
                out = {}
                for tf, df in frames.items():
                    out[tf] = _rust_apply(df)
                logger.info("Discovery: Injected %s Rust mixer signals per TF.", n_strategies)
                return out

            if base_tf not in frames:
                return frames
            base_df = _rust_apply(frames[base_tf])
            sig_cols = [c for c in frame_columns(base_df) if str(c).startswith("tmx_sig_")]
            out = {base_tf: base_df}
            for tf, df in frames.items():
                if tf == base_tf: continue
                if not sig_cols:
                    out[tf] = df
                    continue
                local = frame_copy(df)
                if local is None:
                    out[tf] = df
                    continue
                src_idx = index_to_ns_int64(frame_index(base_df))
                tgt_idx = index_to_ns_int64(frame_index(local))
                tgt_n = frame_len(local)
                for col in sig_cols:
                    if frame_has_column(local, col): continue
                    with contextlib.suppress(Exception):
                        sig_src = frame_column_numpy(base_df, col, dtype=np.float32)
                        aligned = align_ffill_by_ns(src_idx, sig_src, tgt_idx, dtype=np.float32)
                        if aligned is None:
                            aligned = fit_len_array(sig_src, tgt_n, fill=0.0, dtype=np.float32)
                        frame_set_column(local, col, aligned, dtype=np.float32)
                out[tf] = local
            logger.info("Discovery: Injected %s Rust mixer signals (base TF aligned).", len(sig_cols))
            return out
        except Exception as exc:
            logger.warning("Discovery: Rust mixer signal injection failed: %s", exc)
            return frames

    logger.warning("Discovery: Rust mixer backend unavailable; skipping mixer signals.")
    return frames
