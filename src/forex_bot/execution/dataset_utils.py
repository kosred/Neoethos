import os
import logging
from typing import Any
import contextlib
import numpy as np

from .frame_utils import (
    is_dataframe,
    is_frame_like,
    is_series,
    index_to_ns_int64,
    frame_len,
    frame_index,
    frame_columns,
    frame_resolve_column,
    frame_has_column,
    frame_column_numpy,
    frame_column_numpy_optional,
    frame_set_column,
    fit_len_array,
    NumpyFrame,
    make_dataframe,
    make_series,
    range_index,
    to_datetime_index,
    concat_dataframes,
    frame_empty,
    frame_copy,
)
from .alignment_utils import (
    align_exact_by_ns,
    align_ffill_by_ns,
    align_feature_matrix,
    column_index_mapping,
    sort_dedup_rows_by_index,
)
from .rust_wrappers import (
    rust_rank_scores_desc,
    pair_corr_enabled,
)

logger = logging.getLogger(__name__)

def dataset_row_count(dataset: Any | None) -> int:
    if dataset is None:
        return 0
    x = getattr(dataset, "X", None)
    if x is None:
        return 0
    try:
        return int(len(x))
    except Exception:
        return 0

def aggregate_metrics(
    metrics_list: list[dict[str, Any]],
    *,
    numeric_keys: list[str],
    bool_any_keys: list[str] | None = None,
) -> dict[str, Any]:
    if not metrics_list:
        return {}

    def _num(key: str) -> list[float]:
        vals: list[float] = []
        for m in metrics_list:
            v = m.get(key)
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        return vals

    agg: dict[str, Any] = {}
    for key in numeric_keys:
        vals = _num(key)
        if not vals:
            continue
        agg[key] = float(np.mean(vals))

    for key in bool_any_keys or []:
        agg[key] = any(bool(m.get(key)) for m in metrics_list)
    return agg

def tail_dataset(ds: Any, rows: int) -> Any:
    if rows <= 0:
        return ds
    try:
        x_obj = getattr(ds, "X", None)
        n = len(x_obj)
    except Exception:
        return ds
    if n <= rows:
        return ds

    def _tail(obj):
        try:
            if is_dataframe(obj):
                n_obj = len(obj)
                start = max(0, int(n_obj) - int(rows))
                take = np.arange(start, n_obj, dtype=np.int64)
                with contextlib.suppress(Exception):
                    return obj.take(take)
                return obj[-rows:]
            if is_frame_like(obj):
                n_obj = frame_len(obj)
                return obj.tail(rows) if hasattr(obj, "tail") else obj[-rows:]
            return obj[-rows:]
        except Exception:
            return obj

    X = _tail(getattr(ds, "X", None))
    y = _tail(getattr(ds, "y", None))
    labels = _tail(getattr(ds, "labels", None))
    metadata = getattr(ds, "metadata", None)
    if is_dataframe(metadata) or is_frame_like(metadata):
        metadata = _tail(metadata)
        
    ctor = getattr(ds, "__class__", None)
    if ctor:
        return ctor(
            X=X,
            y=y,
            index=getattr(X, "index", None),
            feature_names=list(getattr(X, "columns", getattr(ds, "feature_names", []))),
            metadata=metadata,
            labels=labels,
        )
    return ds

def rolling_corr_numpy(a: np.ndarray, b: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    a64 = np.asarray(a, dtype=np.float64).reshape(-1)
    b64 = np.asarray(b, dtype=np.float64).reshape(-1)
    n = int(min(a64.size, b64.size))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    a64 = a64[:n]
    b64 = b64[:n]
    out = np.zeros(n, dtype=np.float32)
    w = max(2, int(window))
    mp = max(2, int(min_periods))

    cs_a = np.zeros(n + 1, dtype=np.float64)
    cs_b = np.zeros(n + 1, dtype=np.float64)
    cs_aa = np.zeros(n + 1, dtype=np.float64)
    cs_bb = np.zeros(n + 1, dtype=np.float64)
    cs_ab = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(a64, out=cs_a[1:])
    np.cumsum(b64, out=cs_b[1:])
    np.cumsum(a64 * a64, out=cs_aa[1:])
    np.cumsum(b64 * b64, out=cs_bb[1:])
    np.cumsum(a64 * b64, out=cs_ab[1:])

    for i in range(n):
        start = max(0, i - w + 1)
        m = i - start + 1
        if m < mp:
            continue
        sa = cs_a[i + 1] - cs_a[start]
        sb = cs_b[i + 1] - cs_b[start]
        saa = cs_aa[i + 1] - cs_aa[start]
        sbb = cs_bb[i + 1] - cs_bb[start]
        sab = cs_ab[i + 1] - cs_ab[start]
        ma = sa / m
        mb = sb / m
        var_a = (saa / m) - (ma * ma)
        var_b = (sbb / m) - (mb * mb)
        if var_a <= 1e-12 or var_b <= 1e-12:
            continue
        cov = (sab / m) - (ma * mb)
        out[i] = float(cov / np.sqrt(var_a * var_b))
    return out

def shift_with_lag(values: np.ndarray, lag: int) -> np.ndarray:
    src = np.asarray(values, dtype=np.float32).reshape(-1)
    n = src.shape[0]
    out = np.zeros(n, dtype=np.float32)
    if n <= 0:
        return out
    lag_i = max(1, int(lag))
    if lag_i >= n:
        return out
    out[lag_i:] = src[:-lag_i]
    return out

def numpy_dataset_returns(ds: Any) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        x_np = np.asarray(ds.X, dtype=np.float32)
    except Exception:
        return None
    if x_np.ndim != 2 or x_np.shape[0] <= 1:
        return None

    names = [str(c).strip().lower() for c in (ds.feature_names or [])]
    n_rows = int(x_np.shape[0])
    idx_ns = index_to_ns_int64(getattr(ds, "index", None))
    if idx_ns is None or len(idx_ns) != n_rows:
        idx_ns = np.arange(n_rows, dtype=np.int64)
    else:
        idx_ns = np.asarray(idx_ns, dtype=np.int64).reshape(-1)

    ret = None
    if names:
        ret_i = None
        for key in ("returns", "ret", "return"):
            with contextlib.suppress(ValueError):
                ret_i = names.index(key)
                break
        if ret_i is not None:
            ret = x_np[:, ret_i].astype(np.float32, copy=False)

        if ret is None:
            close_i = None
            for key in ("close", "price_close", "mid_close"):
                with contextlib.suppress(ValueError):
                    close_i = names.index(key)
                    break
            if close_i is not None:
                close = x_np[:, close_i].astype(np.float32, copy=False)
                ret = np.zeros(n_rows, dtype=np.float32)
                prev = np.where(np.abs(close[:-1]) > 1e-9, close[:-1], 1e-9)
                ret[1:] = (close[1:] - close[:-1]) / prev

    if ret is None:
        meta = getattr(ds, "metadata", None)
        if (is_dataframe(meta) or is_frame_like(meta)) and frame_has_column(meta, "close"):
            try:
                close_src = frame_column_numpy(meta, "close", dtype=np.float32)
                meta_idx = index_to_ns_int64(frame_index(meta))
                if meta_idx is not None and len(meta_idx) == close_src.shape[0]:
                    close = align_exact_by_ns(meta_idx, close_src, idx_ns, dtype=np.float32, fill=0.0)
                elif close_src.shape[0] == n_rows:
                    close = close_src.astype(np.float32, copy=False)
                else:
                    close = None
                if close is not None and close.shape[0] == n_rows:
                    ret = np.zeros(n_rows, dtype=np.float32)
                    prev = np.where(np.abs(close[:-1]) > 1e-9, close[:-1], 1e-9)
                    ret[1:] = (close[1:] - close[:-1]) / prev
            except Exception:
                ret = None

    if ret is None:
        return None
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return idx_ns, ret

def inject_cross_pair_context_numpy(
    datasets: list[tuple[str, Any]],
    *,
    window: int,
    lag: int,
    max_peers: int,
    min_overlap: int,
    static_rows: int,
) -> list[tuple[str, Any]]:
    returns_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, ds in datasets:
        parsed = numpy_dataset_returns(ds)
        if parsed is not None:
            returns_map[str(sym)] = parsed

    if len(returns_map) < 2:
        logger.info("[GLOBAL CORR] Skipping pair-correlation features (insufficient NumPy return series).")
        return datasets

    feature_suffixes = [
        "pair_peer_ret_mean", "pair_peer_ret_std", "pair_peer_abs_mean", "pair_corr_mean",
        "pair_corr_abs_mean", "pair_corr_max", "pair_corr_min", "pair_lead_ret",
        "pair_lead_corr", "pair_divergence", "pair_relative_strength",
        "pair_static_corr_mean", "pair_static_corr_abs_max",
    ]

    patched: list[tuple[str, Any]] = []
    for sym, ds in datasets:
        sym_key = str(sym)
        sym_parsed = returns_map.get(sym_key)
        if sym_parsed is None:
            patched.append((sym, ds))
            continue
        sym_idx, sym_ret = sym_parsed

        peer_rank: list[tuple[str, float, float]] = []
        for peer_sym, (peer_idx, peer_ret) in returns_map.items():
            if peer_sym == sym_key:
                continue
            peer_aligned = align_exact_by_ns(peer_idx, peer_ret, sym_idx, dtype=np.float32, fill=0.0)
            if peer_aligned.shape[0] != sym_ret.shape[0]:
                continue
            if sym_ret.shape[0] > static_rows:
                a = sym_ret[-static_rows:]
                b = peer_aligned[-static_rows:]
            else:
                a = sym_ret
                b = peer_aligned
            mask = np.isfinite(a) & np.isfinite(b)
            overlap = int(np.sum(mask))
            if overlap < min_overlap:
                continue
            with contextlib.suppress(Exception):
                corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
                if np.isfinite(corr):
                    peer_rank.append((peer_sym, abs(corr), corr))
        if peer_rank:
            order = rust_rank_scores_desc(np.asarray([item[1] for item in peer_rank], dtype=np.float64))
            if order is not None:
                peer_rank = [peer_rank[int(i)] for i in order.tolist()]
            else:
                peer_rank.sort(key=lambda item: item[1], reverse=True)
        peers = [p for p, _abs, _corr in peer_rank[:max_peers]]
        if not peers:
            patched.append((sym, ds))
            continue

        x_np = np.asarray(ds.X, dtype=np.float32)
        n_rows = int(x_np.shape[0])
        if n_rows <= 0:
            patched.append((sym, ds))
            continue

        sym_hist = shift_with_lag(sym_ret, lag)
        peer_hist = np.zeros((n_rows, len(peers)), dtype=np.float32)
        corr_hist = np.zeros((n_rows, len(peers)), dtype=np.float32)
        min_periods = max(20, window // 4)

        for j, p in enumerate(peers):
            p_idx, p_ret = returns_map[p]
            p_aligned = align_exact_by_ns(p_idx, p_ret, sym_idx, dtype=np.float32, fill=0.0)
            p_hist = shift_with_lag(p_aligned, lag)
            peer_hist[:, j] = p_hist
            corr_hist[:, j] = rolling_corr_numpy(sym_hist, p_hist, window=window, min_periods=min_periods)

        lead_peer = peers[0]
        lead_idx = peers.index(lead_peer)
        static_corrs = [c for _p, _abs, c in peer_rank[:max_peers]]

        with contextlib.suppress(Exception):
            features = np.column_stack(
                [
                    np.mean(peer_hist, axis=1), np.std(peer_hist, axis=1), np.mean(np.abs(peer_hist), axis=1),
                    np.mean(corr_hist, axis=1), np.mean(np.abs(corr_hist), axis=1), np.max(corr_hist, axis=1),
                    np.min(corr_hist, axis=1), peer_hist[:, lead_idx], corr_hist[:, lead_idx],
                    sym_hist - np.mean(peer_hist, axis=1), sym_hist - peer_hist[:, lead_idx],
                    np.full(n_rows, float(np.mean(static_corrs)) if static_corrs else 0.0, dtype=np.float32),
                    np.full(n_rows, float(max(abs(v) for v in static_corrs)) if static_corrs else 0.0, dtype=np.float32),
                ]
            ).astype(np.float32, copy=False)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            x_aug = np.concatenate([x_np, features], axis=1)
            
            ctor = getattr(ds, "__class__", None)
            patched.append(
                (
                    sym,
                    ctor(
                        X=x_aug, y=ds.y, index=ds.index,
                        feature_names=list(ds.feature_names) + feature_suffixes,
                        metadata=ds.metadata,
                        labels=getattr(ds, "labels", None) if getattr(ds, "labels", None) is not None else ds.y,
                    ),
                )
            )
            continue

        patched.append((sym, ds))
    return patched

def inject_cross_pair_context(
    datasets: list[tuple[str, Any]],
    *,
    window: int | None = None,
    lag: int | None = None,
    max_peers: int | None = None,
    min_overlap: int | None = None,
    static_rows: int | None = None,
) -> list[tuple[str, Any]]:
    if len(datasets) < 2 or not pair_corr_enabled():
        return datasets

    def _parse_int_env(key: str) -> int | None:
        try:
            return int(os.environ.get(key, ""))
        except Exception:
            return None

    w = window or _parse_int_env("FOREX_BOT_PAIR_CORR_WINDOW") or 240
    w = max(16, int(w))
    lg = lag or _parse_int_env("FOREX_BOT_PAIR_CORR_LAG") or 1
    lg = max(1, int(lg))
    mp = max_peers or _parse_int_env("FOREX_BOT_PAIR_CORR_MAX_PEERS")
    if mp is None:
        mp = min(4, len(datasets) - 1)
    mp = max(1, min(int(mp), len(datasets) - 1))
    mo = min_overlap or _parse_int_env("FOREX_BOT_PAIR_CORR_MIN_OVERLAP")
    if mo is None:
        mo = max(50, w)
    mo = max(20, int(mo))
    sr = static_rows or _parse_int_env("FOREX_BOT_PAIR_CORR_STATIC_ROWS") or 200_000
    sr = max(5_000, int(sr))
    
    return inject_cross_pair_context_numpy(
        datasets,
        window=w,
        lag=lg,
        max_peers=mp,
        min_overlap=mo,
        static_rows=sr,
    )

def merge_symbol_shards(
    sym: str,
    sym_parts: list[Any],
    *,
    prefer_numpy: bool = False,
) -> Any | None:
    if not sym_parts:
        return None

    feature_names: list[str] = []
    seen: set[str] = set()
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], Any, Any, Any]] = []
    use_dataframe = False
    x_template: Any | None = None
    y_template: Any | None = None

    for ds in sym_parts:
        try:
            X_src = getattr(ds, "X", None)
            if X_src is None: continue
            if is_dataframe(X_src):
                if not prefer_numpy:
                    use_dataframe = True
                    if x_template is None: x_template = X_src
                x_np = X_src.to_numpy(dtype=np.float32, copy=False)
                idx_obj = X_src.index
                names = [str(c) for c in list(X_src.columns)]
            else:
                x_np = np.asarray(X_src, dtype=np.float32)
                idx_obj = getattr(ds, "index", None)
                names = [str(c) for c in list(getattr(ds, "feature_names", []) or [])]
                if len(names) != x_np.shape[1]: names = [f"f{i}" for i in range(x_np.shape[1])]

            rows = int(x_np.shape[0])
            if rows <= 0: continue
            idx_ns = index_to_ns_int64(idx_obj)
            if idx_ns is None or idx_ns.size != rows: idx_ns = np.arange(rows, dtype=np.int64)

            y_src = getattr(ds, "y", None)
            if is_series(y_src):
                if y_template is None: y_template = y_src
                y_arr = series_like_to_int8(y_src, n_rows=rows)
            else:
                y_arr = np.asarray(y_src, dtype=np.int8).reshape(-1)
            
            src_map = {str(name): i for i, name in enumerate(names)}
            for name in names:
                if name not in seen:
                    seen.add(name)
                    feature_names.append(name)

            prepared.append((np.nan_to_num(x_np, nan=0.0).astype(np.float32), y_arr, idx_ns, src_map, getattr(ds, "metadata", None), x_template, y_template))
        except Exception: continue

    if not prepared: return None
    n_features = len(feature_names)
    feature_to_idx = {str(name): i for i, name in enumerate(feature_names)}
    x_chunks, y_chunks, idx_chunks = [], [], []
    for x_np, y_arr, idx_ns, src_map, meta, x_t, y_t in prepared:
        src_cols, dst_cols = column_index_mapping(list(src_map.keys()), feature_to_idx)
        x_chunks.append(align_feature_matrix(x_np, src_cols, dst_cols, dst_width=n_features))
        y_chunks.append(y_arr)
        idx_chunks.append(idx_ns)

    X_all = np.concatenate(x_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)
    idx_all = np.concatenate(idx_chunks, axis=0)
    X_all, y_all, idx_all = sort_dedup_rows_by_index(X_all, y_all, idx_all)

    if use_dataframe:
        idx_dt = idx_all.astype("datetime64[ns]")
        X_out = make_dataframe(X_all, columns=feature_names, index=idx_dt, template=x_template)
        y_out = make_series(y_all, index=idx_dt, dtype=np.int8, template=y_template)
        idx_out = X_out.index
    else:
        X_out, y_out, idx_out = X_all, y_all, idx_all

    from dataclasses import make_dataclass
    DS = make_dataclass("DS", ["X", "y", "index", "feature_names", "metadata", "labels"])
    return DS(X=X_out, y=y_out, index=idx_out, feature_names=feature_names, metadata=None, labels=y_out)

def split_global_train_eval(
    datasets: list[tuple[str, Any]],
    *,
    train_ratio: float,
    embargo_bars: int,
    min_train_rows: int = 1000,
    min_eval_rows: int = 500,
) -> tuple[list[tuple[str, Any]], dict[str, Any], dict[str, Any]]:
    times: list[np.ndarray] = []
    for _, d in datasets:
        idx_ns = index_to_ns_int64(getattr(d, "index", None))
        if idx_ns is not None and idx_ns.size > 0:
            times.append(idx_ns)
    if not times:
        return [], {}, {}

    all_times = np.concatenate(times)
    all_times.sort()
    first_ns = int(all_times[0])
    last_ns = int(all_times[-1])

    eval_from_raw = str(os.environ.get("FOREX_BOT_GLOBAL_EVAL_FROM", "") or "").strip()
    try:
        eval_years = float(os.environ.get("FOREX_BOT_GLOBAL_EVAL_YEARS", "0") or 0.0)
    except Exception:
        eval_years = 0.0
    
    cutoff_ns: int
    if eval_from_raw:
        try:
            cutoff_ns = int(np.datetime64(eval_from_raw).astype("datetime64[ns]").astype(np.int64))
        except Exception:
            cut_i = int(max(0, min(len(all_times) - 1, int(len(all_times) * train_ratio) - 1)))
            cutoff_ns = int(all_times[cut_i])
    elif eval_years > 0.0:
        cutoff_ns = int(last_ns - int(eval_years * 31557600 * 1_000_000_000))
    else:
        cut_i = int(max(0, min(len(all_times) - 1, int(len(all_times) * train_ratio) - 1)))
        cutoff_ns = int(all_times[cut_i])

    cutoff_ns = max(first_ns, min(last_ns, cutoff_ns))
    from datetime import datetime, timezone
    cutoff_dt = datetime.fromtimestamp(cutoff_ns / 1_000_000_000.0, tz=timezone.utc)

    train_parts: list[tuple[str, Any]] = []
    eval_map: dict[str, Any] = {}
    split_info = {"cutoff": cutoff_dt.isoformat(), "per_symbol": {}}

    for sym, d in datasets:
        X = d.X
        n = len(X)
        if n == 0: continue
        idx_ns = index_to_ns_int64(getattr(d, "index", None))
        if idx_ns is None: continue
        
        cut_right = int(np.searchsorted(idx_ns, cutoff_ns, side="right"))
        train_end = max(0, cut_right - int(embargo_bars))
        eval_start = cut_right

        if train_end < min_train_rows or (n - eval_start) < min_eval_rows:
            cut_right = int(n * train_ratio)
            train_end = max(0, cut_right - int(embargo_bars))
            eval_start = cut_right

        if train_end < 100 or (n - eval_start) < 200:
            continue

        ctor = getattr(d, "__class__", None)
        train_ds = ctor(X=X[:train_end], y=d.y[:train_end], index=idx_ns[:train_end], feature_names=list(d.feature_names), 
                        metadata=tail_dataset(d.metadata, train_end) if d.metadata is not None else None, labels=d.y[:train_end])
        eval_ds = ctor(X=X[eval_start:], y=d.y[eval_start:], index=idx_ns[eval_start:], feature_names=list(d.feature_names),
                       metadata=tail_dataset(d.metadata, n - eval_start) if d.metadata is not None else None, labels=d.y[eval_start:])
        
        train_parts.append((sym, train_ds))
        eval_map[sym] = eval_ds
        split_info["per_symbol"][sym] = {"train": len(train_ds.X), "eval": len(eval_ds.X)}

    return train_parts, eval_map, split_info

def merge_ohlc_columns(target: Any, source: Any | None) -> Any:
    if frame_empty(target) or frame_empty(source):
        return target
    cols = [c for c in ("open", "high", "low", "close") if frame_has_column(source, c)]
    if not cols:
        return target
    out = frame_copy(target)
    if out is None:
        return target
    tgt_n = frame_len(out)
    src_idx = index_to_ns_int64(frame_index(source))
    tgt_idx = index_to_ns_int64(frame_index(out))
    for col in cols:
        try:
            src_vals = frame_column_numpy(source, col, dtype=np.float64)
        except Exception:
            continue
        aligned = align_ffill_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float64)
        if aligned is None:
            aligned = fit_len_array(src_vals, tgt_n, fill=0.0, dtype=np.float64)
        frame_set_column(out, col, aligned, dtype=np.float64)
    return out

def coerce_dataset_index(values: Any, rows: int, *, fallback_index: Any | None = None) -> Any:
    n = max(0, int(rows))
    if fallback_index is not None:
        with contextlib.suppress(Exception):
            if len(fallback_index) == n:
                return fallback_index
    if values is None:
        return range_index(n)
    if is_datetime_index(values):
        with contextlib.suppress(Exception):
            idx = values
            if len(idx) > n:
                return idx[:n]
            if len(idx) == n:
                return idx
    arr = np.asarray(values).reshape(-1)
    if arr.size <= 0:
        return range_index(n)
    if arr.size < n:
        if fallback_index is not None:
            with contextlib.suppress(Exception):
                if len(fallback_index) == n:
                    return fallback_index
        return range_index(n)
    arr = arr[:n]
    with contextlib.suppress(Exception):
        if np.issubdtype(arr.dtype, np.datetime64):
            return to_datetime_index(arr)
    with contextlib.suppress(Exception):
        if arr.dtype.kind in {"i", "u"}:
            vmax = int(np.max(np.abs(arr.astype(np.int64, copy=False)))) if arr.size > 0 else 0
            if vmax > 10**14:
                return to_datetime_index(arr.astype(np.int64, copy=False))
            if vmax > 10**11:
                return to_datetime_index(arr.astype(np.int64, copy=False))
    with contextlib.suppress(Exception):
        return to_datetime_index(arr)
    return range_index(n)

def prepared_dataset_to_frame(dataset: Any, *, fallback_frame: Any | None = None) -> Any | None:
    if dataset is None:
        return None
    x = getattr(dataset, "X", None)
    if is_dataframe(x) or is_frame_like(x):
        return x
    if fallback_frame is not None:
        return fallback_frame
    if x is not None:
        return make_dataframe(x, columns=getattr(dataset, "feature_names", None), index=getattr(dataset, "index", None))
    return None

def series_like_to_int8(values: Any, *, row_index: Any | None = None, n_rows: int | None = None) -> np.ndarray:
    src = values
    arr: np.ndarray | None = None
    if row_index is not None:
        src_idx = index_to_ns_int64(getattr(src, "index", None))
        tgt_idx = index_to_ns_int64(row_index)
        if src_idx is not None and tgt_idx is not None:
            src_vals = to_numpy_1d(src, dtype=np.float32)
            aligned = align_exact_by_ns(src_idx, src_vals, tgt_idx, dtype=np.float32, fill=0.0)
            if aligned is not None:
                arr = aligned
    if arr is None:
        arr = to_numpy_1d(src, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int8, copy=False)
    if n_rows is not None:
        return fit_len_array(arr, int(max(0, n_rows)), fill=0.0, dtype=np.int8)
    return arr
