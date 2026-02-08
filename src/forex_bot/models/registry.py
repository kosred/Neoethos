"""
Lazy-Loading Model Registry.
Thread-safe implementation for HPC environments.
"""

import logging
import os
import threading
import importlib
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from .base import ExpertModel

logger = logging.getLogger(__name__)

# HPC FIX: Thread Lock for Registry
_REGISTRY_LOCK = threading.Lock()
_CLASS_CACHE: Dict[str, Type['ExpertModel']] = {}

_MODEL_ALIASES = {
    "xgboostrf": "xgboost_rf",
    "xgboostdart": "xgboost_dart",
    "catboostalt": "catboost_alt",
    "n_beats": "nbeats",
    "nbeatsx": "nbeatsx_nf",
    "nbeats_x": "nbeatsx_nf",
    "nbeatsxnf": "nbeatsx_nf",
    "nbeatsx_nf": "nbeatsx_nf",
    "tidenf": "tide_nf",
    "tide_nf": "tide_nf",
    "patch_tst": "patchtst",
    "times_net": "timesnet",
}


def _normalize_model_name(name: str) -> str:
    key = str(name or "").strip().lower()
    key = key.replace(" ", "")
    key = key.replace("-", "_")
    return _MODEL_ALIASES.get(key, key)

# Registry mapping: name -> (module_path, class_name)
MODEL_MAPPING = {
    "lightgbm": ("trees", "LightGBMExpert"),
    "xgboost": ("trees", "XGBoostExpert"),
    "xgboost_rf": ("trees", "XGBoostRFExpert"),
    "xgboost_dart": ("trees", "XGBoostDARTExpert"),
    "catboost": ("trees", "CatBoostExpert"),
    "catboost_alt": ("trees", "CatBoostAltExpert"),
    "mlp": ("mlp", "MLPExpert"),
    "transformer": ("transformers", "TransformerExpertTorch"),
    "kan": ("kan_gpu", "KANExpert"),
    "nbeats": ("nbeats_gpu", "NBeatsExpert"),
    "tabnet": ("tabnet_gpu", "TabNetExpert"),
    "tide": ("tide_gpu", "TiDEExpert"),
    "tide_nf": ("forecast_nf", "TiDENFExpert"),
    "nbeatsx_nf": ("forecast_nf", "NBEATSxNFExpert"),
    "patchtst": ("transformer_nf", "PatchTSTExpert"),
    "timesnet": ("transformer_nf", "TimesNetExpert"),
    "rl_ppo": ("rl", "RLExpertPPO"),
    "rl_sac": ("rl", "RLExpertSAC"),
    "rllib_ppo": ("rllib_agent", "RLlibPPOAgent"),
    "rllib_sac": ("rllib_agent", "RLlibSACAgent"),
    "evolution": ("evolution", "EvoExpertCMA"),
    "genetic": ("genetic", "GeneticStrategyExpert"),
    "unsupervised": ("unsupervised", "ClusterExpert"),
}

_RUST_TREE_MAPPING = {
    "lightgbm": "RustLightGBMExpert",
    "xgboost": "RustXGBoostExpert",
    "xgboost_rf": "RustXGBoostRFExpert",
    "xgboost_dart": "RustXGBoostDARTExpert",
}


def _use_rust_tree_models(model_name: str | None = None) -> bool:
    """Return True if Rust tree bindings should be preferred for the model."""
    raw = os.environ.get("FOREX_BOT_TREE_BACKEND", "auto").strip().lower()
    force_rust = raw in {"rust", "1", "true", "yes", "on"}
    if raw in {"rust", "1", "true", "yes", "on"}:
        pass
    if raw in {"python", "py", "0", "false", "no", "off"}:
        return False
    # auto (or rust): try to detect bindings and feature coverage.
    try:
        import forex_bindings  # type: ignore

        if model_name and model_name in _RUST_TREE_MAPPING:
            cls_name = _RUST_TREE_MAPPING[model_name]
            if force_rust:
                return True
            return hasattr(forex_bindings, cls_name)

        if force_rust:
            return True

        return any(hasattr(forex_bindings, cls_name) for cls_name in _RUST_TREE_MAPPING.values())
    except Exception:
        return False

def register_model(name: str, module_path: str, class_name: str) -> None:
    """Dynamically registers a new model type."""
    canonical_name = _normalize_model_name(name)
    with _REGISTRY_LOCK:
        MODEL_MAPPING[canonical_name] = (module_path, class_name)
        # Clear cache if overwriting
        if canonical_name in _CLASS_CACHE:
            del _CLASS_CACHE[canonical_name]
        logger.info(f"Registered new model: {canonical_name} -> {module_path}.{class_name}")

def get_model_class(name: str, prefer_gpu: bool = False) -> Type['ExpertModel']:
    """Thread-safe lazy-imports the requested model class."""
    canonical_name = _normalize_model_name(name)
    with _REGISTRY_LOCK:
        if canonical_name in _CLASS_CACHE:
            return _CLASS_CACHE[canonical_name]
        
        if canonical_name not in MODEL_MAPPING:
            raise ValueError(f"Model '{name}' not found in registry.")
        
        module_name, class_name = MODEL_MAPPING[canonical_name]
        rust_requested = False

        if canonical_name in _RUST_TREE_MAPPING and _use_rust_tree_models(canonical_name):
            module_name = "trees_rust"
            class_name = _RUST_TREE_MAPPING[canonical_name]
            rust_requested = True

        # Handle CPU fallback for GPU models if needed
        if not prefer_gpu and canonical_name in {"kan", "nbeats", "tabnet", "tide"}:
            module_name = module_name.replace("_gpu", "")

        try:
            # Import with package context
            module = importlib.import_module(f".{module_name}", package="forex_bot.models")
            cls = getattr(module, class_name)
            if rust_requested and getattr(cls, "_model_cls", None) is None:
                raise ImportError(f"Rust bindings missing class for {canonical_name}")
            _CLASS_CACHE[canonical_name] = cls
            return cls
        except Exception as e:
            if rust_requested:
                # Rust bindings failed; fall back to Python implementation.
                base_module, base_class = MODEL_MAPPING[canonical_name]
                try:
                    module = importlib.import_module(f".{base_module}", package="forex_bot.models")
                    cls = getattr(module, base_class)
                    _CLASS_CACHE[canonical_name] = cls
                    logger.warning(
                        "Rust bindings unavailable for '%s' (error: %s); using Python model.",
                        canonical_name,
                        e,
                    )
                    return cls
                except Exception as py_exc:
                    logger.error(
                        "Python fallback import failed for '%s' after Rust error: %s",
                        canonical_name,
                        py_exc,
                    )
                    raise ImportError(f"Could not load model {canonical_name}") from py_exc

            # If GPU module import fails, try CPU implementation as fallback.
            if canonical_name in {"kan", "nbeats", "tabnet", "tide"} and module_name.endswith("_gpu"):
                try:
                    cpu_module = module_name.replace("_gpu", "")
                    module = importlib.import_module(f".{cpu_module}", package="forex_bot.models")
                    cls = getattr(module, class_name)
                    _CLASS_CACHE[canonical_name] = cls
                    logger.warning(
                        "Falling back to CPU model for '%s' after GPU import failure: %s",
                        canonical_name,
                        e,
                    )
                    return cls
                except Exception as cpu_exc:
                    logger.error(
                        "CPU fallback import failed for '%s' after GPU import error: %s",
                        canonical_name,
                        cpu_exc,
                    )
                    raise ImportError(f"Could not load model {canonical_name}") from cpu_exc
            logger.error(f"Failed to lazy-import model '{canonical_name}': {e}")
            raise ImportError(f"Could not load model {canonical_name}") from e

# Keep for backward compatibility with existing code
MODEL_REGISTRY = MODEL_MAPPING

_TREE_MODELS = {
    "lightgbm",
    "xgboost",
    "xgboost_rf",
    "xgboost_dart",
    "catboost",
    "catboost_alt",
}
_NEURAL_MODELS = {
    "mlp",
    "transformer",
    "kan",
    "nbeats",
    "tabnet",
    "tide",
    "tide_nf",
    "nbeatsx_nf",
    "patchtst",
    "timesnet",
}
_RL_MODELS = {"rl_ppo", "rl_sac", "rllib_ppo", "rllib_sac"}
_EVOLUTION_MODELS = {"evolution", "genetic"}
_UNSUPERVISED_MODELS = {"unsupervised"}

AVAILABLE_MODELS = tuple(MODEL_MAPPING.keys())


def get_model_info(name: str) -> Dict[str, Any] | None:
    """Return model metadata for UI/tests."""
    canonical_name = _normalize_model_name(name)
    if canonical_name not in MODEL_MAPPING:
        return None

    if canonical_name in _TREE_MODELS:
        category = "TreeModel"
        requires_gpu = False
        description = "Tree-based model (CPU/GPU optional)"
    elif canonical_name in _NEURAL_MODELS:
        category = "NeuralNetwork"
        requires_gpu = True
        description = "Neural network model (GPU recommended)"
    elif canonical_name in _RL_MODELS:
        category = "ReinforcementLearning"
        requires_gpu = True
        description = "RL agent (GPU recommended)"
    elif canonical_name in _EVOLUTION_MODELS:
        category = "Evolutionary"
        requires_gpu = False
        description = "Evolutionary strategy model"
    elif canonical_name in _UNSUPERVISED_MODELS:
        category = "Unsupervised"
        requires_gpu = False
        description = "Unsupervised clustering model"
    else:
        category = "Unknown"
        requires_gpu = False
        description = "Model metadata not defined"

    return {
        "name": canonical_name,
        "category": category,
        "requires_gpu": requires_gpu,
        "description": description,
    }
