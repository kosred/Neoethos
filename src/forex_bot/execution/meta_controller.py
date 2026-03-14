import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class PropMetaState:
    """
    Represents the current state of the account and market
    for the Meta-Controller to make decisions.
    """
    daily_dd_pct: float
    volatility_regime: str
    recent_win_rate: float
    consecutive_losses: int
    model_confidence: float
    hour_of_day: int
    market_regime: str = "Normal"


class MetaController:
    """
    The 'Brain' that auto-tunes risk parameters based on survival objectives.
    Powered by Native Rust Backend `forex-bindings`.
    """

    def __init__(
        self,
        max_daily_dd: float = 0.045,
        safety_buffer: float = 0.025,
        base_risk_per_trade: float = 0.015,
        base_confidence: float = 0.55,
        settings: object | None = None,
        silent: bool = False,
    ):
        try:
            from forex_bindings import MetaController as RustMetaController
            self._backend = RustMetaController(
                max_daily_dd,
                safety_buffer,
                base_risk_per_trade,
                base_confidence,
                settings,
                silent
            )
        except ImportError as e:
            logger.error("Failed to load Rust MetaController backend from forex_bindings!")
            raise RuntimeError("forex_bindings not compiled!") from e

    def get_risk_parameters(self, state: PropMetaState) -> tuple[float, float, bool]:
        """
        Calculates dynamic risk parameters.

        Returns:
            risk_multiplier (float): 0.0 to 1.0+
            confidence_threshold (float): 0.0 to 1.0
            allow_trading (bool): Hard stop flag
        """
        return self._backend.get_risk_parameters(state)
