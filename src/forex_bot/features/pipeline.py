from __future__ import annotations

import logging
from typing import Any

from .pipeline_base import LabelConfig, NumpyFrame, tf_minutes
from .pipeline_features import FeatureEngineer
from .pipeline_labels import compute_labels

logger = logging.getLogger(__name__)

class UnifiedFeaturePipeline:
    """
    Facade for the feature engineering pipeline.
    Orchestrates base utilities, feature calculation, and labeling.
    """
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.engineer = FeatureEngineer(settings)

    def prepare_dataset(self, df: Any, symbol: str) -> Any:
        if df is None or len(df) == 0:
            return None
            
        # 1. Feature Engineering (Enhanced)
        df_feat = self.engineer.compute_enhanced_features(df)
        df_feat = self.engineer.compute_volatility_features(df_feat)
        df_feat = self.engineer.compute_session_features(df_feat)
        
        # 2. Labeling (Triple Barrier or Horizon)
        # In a real scenario, we'd get config from settings/env
        cfg = LabelConfig(
            horizon=20, 
            min_dist=0.0001, 
            use_triple_barrier=True, 
            max_hold=100,
            tp_pips=40.0,
            sl_pips=20.0
        )
        labels = compute_labels(df_feat, cfg)
        
        # Return something compatible with Expected PreparedDataset event
        return {
            "X": df_feat,
            "y": labels,
            "symbol": symbol
        }
