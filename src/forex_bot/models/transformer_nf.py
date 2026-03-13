from __future__ import annotations

import logging
from typing import Any

from .transformers import TransformerExpertTorch

logger = logging.getLogger(__name__)


class PatchTSTExpert(TransformerExpertTorch):
    """
    Legacy alias for the old PatchTST NeuralForecast adapter.

    Routed to the native transformer expert to keep the runtime frame-native.
    """

    def __init__(self, **kwargs: Any) -> None:
        defaults = {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 4,
            "lr": 1e-4,
            "batch_size": 64,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
        logger.info("Model alias patchtst uses native transformer expert.")


class TimesNetExpert(TransformerExpertTorch):
    """
    Legacy alias for the old TimesNet NeuralForecast adapter.

    Routed to the native transformer expert to keep the runtime frame-native.
    """

    def __init__(self, **kwargs: Any) -> None:
        defaults = {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 4,
            "lr": 1e-4,
            "batch_size": 64,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
        logger.info("Model alias timesnet uses native transformer expert.")


__all__ = ["PatchTSTExpert", "TimesNetExpert"]
