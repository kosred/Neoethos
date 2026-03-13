from __future__ import annotations

import logging
from typing import Any

from .nbeats_gpu import NBeatsExpert
from .tide_gpu import TiDEExpert

logger = logging.getLogger(__name__)


class TiDENFExpert(TiDEExpert):
    """
    Legacy alias for the old NeuralForecast TiDE adapter.

    The project now routes this model key to the native TiDE expert to keep the
    execution path frame-native and avoid tabular module dependencies.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        logger.info("Model alias tide_nf uses native TiDE expert.")


class NBEATSxNFExpert(NBeatsExpert):
    """
    Legacy alias for the old NeuralForecast NBEATSx adapter.

    The project now routes this model key to the native N-BEATS expert to keep
    the execution path frame-native and avoid tabular module dependencies.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        logger.info("Model alias nbeatsx_nf uses native N-BEATS expert.")


__all__ = ["TiDENFExpert", "NBEATSxNFExpert"]
