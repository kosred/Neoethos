import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import forex_bindings as _fb  # type: ignore
except Exception:
    _fb = None

def rust_only_enabled() -> bool:
    raw = str(os.environ.get("FOREX_BOT_RUST_ONLY", "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    profile = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
    if profile.startswith("rust"):
        return True
    tree_backend = str(os.environ.get("FOREX_BOT_TREE_BACKEND", "") or "").strip().lower()
    if tree_backend in {"rust_strict", "strict_rust", "rust_only", "rust-only"}:
        return True
    features_backend = str(os.environ.get("FOREX_BOT_FEATURES_BACKEND", "") or "").strip().lower()
    return features_backend in {"rust_strict", "strict_rust", "rust_only", "rust-only"}

def discovery_rust_features_enabled() -> bool:
    raw = os.environ.get("FOREX_BOT_DISCOVERY_RUST_FEATURES")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    profile = str(os.environ.get("FOREX_BOT_RUNTIME_PROFILE", "") or "").strip().lower()
    if profile.startswith("rust"):
        return True
    mode = str(os.environ.get("FOREX_BOT_FEATURES_BACKEND", "") or "").strip().lower()
    if mode in {"rust_strict", "strict_rust", "rust_only", "rust-only"}:
        return True
    return str(os.environ.get("FOREX_BOT_RUST_ONLY", "") or "").strip().lower() in {"1", "true", "yes", "on"}

def rust_rank_scores_desc(scores: Any, *, absolute: bool = False) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "rank_scores_desc"):
        return None
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    try:
        out = _fb.rank_scores_desc(arr, bool(absolute))
    except Exception:
        return None
    order = np.asarray(out, dtype=np.int64).reshape(-1)
    if order.size != arr.size:
        return None
    return order

def rust_align_ffill_by_ns(src_idx: Any, src_vals: Any, tgt_idx: Any, fill: float = 0.0) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "align_ffill_by_ns"):
        return None
    try:
        return _fb.align_ffill_by_ns(
            np.asarray(src_idx, dtype=np.int64).reshape(-1),
            np.asarray(src_vals, dtype=np.float64).reshape(-1),
            np.asarray(tgt_idx, dtype=np.int64).reshape(-1),
            float(fill),
        )
    except Exception:
        return None

def rust_align_exact_by_ns(src_idx: Any, src_vals: Any, tgt_idx: Any, fill: float = 0.0) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "align_exact_by_ns"):
        return None
    try:
        return _fb.align_exact_by_ns(
            np.asarray(src_idx, dtype=np.int64).reshape(-1),
            np.asarray(src_vals, dtype=np.float64).reshape(-1),
            np.asarray(tgt_idx, dtype=np.int64).reshape(-1),
            float(fill),
        )
    except Exception:
        return None

def rust_align_feature_matrix(src: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray, dst_width: int) -> np.ndarray | None:
    if _fb is None or not hasattr(_fb, "align_feature_matrix"):
        return None
    try:
        return _fb.align_feature_matrix(src, src_idx, dst_idx, int(dst_width))
    except Exception:
        return None

def rust_sort_dedup_rows_by_index(x: np.ndarray, y: np.ndarray, idx_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if _fb is None or not hasattr(_fb, "sort_dedup_rows_by_index"):
        return None
    try:
        out_x, out_y, out_idx = _fb.sort_dedup_rows_by_index(x, y, idx_ns)
        return (
            np.asarray(out_x, dtype=np.float32),
            np.asarray(out_y, dtype=np.int8).reshape(-1),
            np.asarray(out_idx, dtype=np.int64).reshape(-1),
        )
    except Exception:
        return None

def pair_corr_enabled() -> bool:
    raw = str(os.environ.get("FOREX_BOT_PAIR_CORR_ENABLED", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}
