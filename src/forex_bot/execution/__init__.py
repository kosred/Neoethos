__all__ = [
    "ForexBot",
    "RiskManager",
    "CorrelationRiskManager",
    "CorrelationMatrix",
    "MultiSymbolPortfolio",
    "PortfolioConfig",
    "AntiCorrelationStrategy",
    "SessionBasedSymbolSelector",
    "create_correlation_portfolio",
]


def __getattr__(name):
    if name == "ForexBot":
        from .bot import ForexBot as _ForexBot
        return _ForexBot
    if name == "RiskManager":
        from .risk import RiskManager as _RiskManager
        return _RiskManager
    if name == "CorrelationRiskManager":
        from .correlation_risk import CorrelationRiskManager as _CorrRiskMgr
        return _CorrRiskMgr
    if name == "CorrelationMatrix":
        from .correlation_risk import CorrelationMatrix as _CorrMatrix
        return _CorrMatrix
    if name == "MultiSymbolPortfolio":
        from .portfolio_strategy import MultiSymbolPortfolio as _Portfolio
        return _Portfolio
    if name == "PortfolioConfig":
        from .portfolio_strategy import PortfolioConfig as _PortfolioConfig
        return _PortfolioConfig
    if name == "AntiCorrelationStrategy":
        from .correlation_risk import AntiCorrelationStrategy as _AntiCorr
        return _AntiCorr
    if name == "SessionBasedSymbolSelector":
        from .correlation_risk import SessionBasedSymbolSelector as _SessionSel
        return _SessionSel
    if name == "create_correlation_portfolio":
        from .correlation_risk import create_correlation_portfolio as _create_portfolio
        return _create_portfolio
    raise AttributeError(f"module 'forex_bot.execution' has no attribute {name!r}")
