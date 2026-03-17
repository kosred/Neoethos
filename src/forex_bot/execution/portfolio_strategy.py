"""
Portfolio Strategy Implementation

Multi-symbol portfolio management with correlation-aware position sizing,
anti-correlation trading, and session-based optimization.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Callable

import numpy as np
import pandas as pd

from .correlation_risk import (
    CorrelationMatrix,
    CorrelationRiskManager,
    AntiCorrelationStrategy,
    SessionBasedSymbolSelector,
    RiskLimits,
    Position,
    SymbolCluster,
    calculate_optimal_exposure
)

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trade signal with metadata."""
    symbol: str
    direction: str  # 'long' or 'short'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    strategy: str
    metadata: dict = field(default_factory=dict)


@dataclass
class PortfolioConfig:
    """Portfolio strategy configuration."""
    # Risk limits
    max_total_risk_pct: float = 0.08
    max_cluster_risk_pct: float = 0.03
    max_trade_risk_pct: float = 0.015
    
    # Position limits
    max_positions: int = 7
    max_positions_per_cluster: int = 2
    
    # Strategy weights
    correlation_weight: float = 0.4
    momentum_weight: float = 0.3
    session_weight: float = 0.3
    
    # Anti-correlation settings
    divergence_threshold: float = 0.5
    correlation_breakdown_threshold: float = 0.3
    
    # Session settings
    session_filtering: bool = True
    min_session_priority: int = 2  # 1=primary, 2=secondary
    
    # Hedging
    require_hedge_benefit: bool = True
    min_hedge_score: float = -0.3


class MultiSymbolPortfolio:
    """
    Manages a multi-symbol forex portfolio with correlation awareness.
    
    Features:
    - Correlation-based position sizing
    - Cluster risk limits
    - Anti-correlation trading signals
    - Session-based symbol selection
    - Automatic hedging recommendations
    """
    
    def __init__(self, 
                 balance: float = 10000.0,
                 config: Optional[PortfolioConfig] = None):
        self.balance = balance
        self.config = config or PortfolioConfig()
        
        # Core components
        self.correlation = CorrelationMatrix()
        self.risk_manager = CorrelationRiskManager(
            balance=balance,
            limits=RiskLimits(
                max_cluster_risk_pct=self.config.max_cluster_risk_pct,
                max_trade_risk_pct=self.config.max_trade_risk_pct,
                max_total_risk_pct=self.config.max_total_risk_pct,
                max_positions_per_cluster=self.config.max_positions_per_cluster
            ),
            correlation_matrix=self.correlation
        )
        self.anti_corr_strategy = AntiCorrelationStrategy(
            correlation_matrix=self.correlation,
            divergence_threshold=self.config.divergence_threshold
        )
        self.session_selector = SessionBasedSymbolSelector()
        
        # State
        self.positions: dict[str, Position] = {}
        self.pending_signals: list[TradeSignal] = []
        self.trade_history: list[dict] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_executed = 0
        self.hedge_trades = 0
        
    def update_balance(self, new_balance: float) -> None:
        """Update portfolio balance."""
        self.balance = new_balance
        self.risk_manager.balance = new_balance
        self.equity_curve.append((datetime.now(), new_balance))
        
    def update_prices(self, prices: dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """Update price data for all symbols."""
        ts = timestamp or datetime.now()
        for symbol, price in prices.items():
            self.anti_corr_strategy.update_price(symbol, price, ts)
    
    def evaluate_signal(self, signal: TradeSignal) -> tuple[bool, dict]:
        """
        Evaluate a trade signal against portfolio constraints.
        
        Returns:
            (should_execute, metadata) tuple
        """
        self.signals_generated += 1
        
        metadata = {
            'original_confidence': signal.confidence,
            'adjusted_confidence': signal.confidence,
            'rejection_reason': None,
            'hedge_score': 0.0,
            'session_valid': True,
            'cluster_exposure': 0.0
        }
        
        # Check session suitability
        if self.config.session_filtering:
            should_trade, reason = self.session_selector.should_trade_symbol(
                signal.symbol, signal.timestamp
            )
            if not should_trade:
                metadata['rejection_reason'] = f"Session filter: {reason}"
                return False, metadata
            metadata['session_valid'] = True
        
        # Check if at max positions
        if len(self.positions) >= self.config.max_positions:
            metadata['rejection_reason'] = "Max positions reached"
            return False, metadata
        
        # Calculate position size based on risk
        stop_pips = abs(signal.entry_price - signal.stop_loss)
        if stop_pips == 0:
            metadata['rejection_reason'] = "Invalid stop loss"
            return False, metadata
        
        # Validate against correlation risk manager
        allowed, risk_metadata = self.risk_manager.validate_new_trade(
            symbol=signal.symbol,
            direction=signal.direction,
            size=1.0,  # Will be adjusted
            stop_loss=signal.stop_loss,
            entry_price=signal.entry_price
        )
        
        if not allowed:
            metadata['rejection_reason'] = risk_metadata.get('reason', 'Risk limit')
            return False, metadata
        
        metadata['hedge_score'] = risk_metadata.get('hedge_score', 0.0)
        metadata['cluster_exposure'] = risk_metadata.get('cluster_exposure', 0.0)
        
        # Check hedge benefit requirement
        if (self.config.require_hedge_benefit and 
            len(self.positions) > 0 and 
            metadata['hedge_score'] < self.config.min_hedge_score):
            metadata['rejection_reason'] = f"Insufficient hedge benefit: {metadata['hedge_score']:.2f}"
            return False, metadata
        
        # Adjust confidence based on portfolio context
        adjusted_conf = self._adjust_confidence(signal, metadata)
        metadata['adjusted_confidence'] = adjusted_conf
        
        if adjusted_conf < 0.55:  # Minimum threshold
            metadata['rejection_reason'] = f"Confidence too low after adjustment: {adjusted_conf:.2f}"
            return False, metadata
        
        return True, metadata
    
    def _adjust_confidence(self, signal: TradeSignal, metadata: dict) -> float:
        """Adjust signal confidence based on portfolio context."""
        base_conf = signal.confidence
        
        # Boost for good hedges
        hedge_score = metadata.get('hedge_score', 0.0)
        if hedge_score > 0.5:
            base_conf += 0.05 * min(hedge_score, 2.0)
        
        # Penalty for cluster concentration
        cluster_exp = metadata.get('cluster_exposure', 0.0)
        if cluster_exp > 0.02:
            base_conf -= 0.05
        
        # Boost for session alignment
        priority = self.session_selector.get_symbol_priority([signal.symbol])[0][1]
        if priority == 1:
            base_conf += 0.02
        
        return min(0.95, max(0.0, base_conf))
    
    def calculate_position_size(self, 
                                signal: TradeSignal,
                                risk_metadata: dict) -> float:
        """
        Calculate optimal position size for a signal.
        
        Considers:
        - Account risk limits
        - Correlation with existing positions
        - Confidence level
        """
        # Base risk amount
        max_risk_amount = self.balance * self.config.max_trade_risk_pct
        
        # Adjust for confidence
        confidence_mult = 0.5 + (signal.confidence * 0.5)  # 0.5x to 1.0x
        
        # Adjust for hedge benefit
        hedge_score = risk_metadata.get('hedge_score', 0.0)
        if hedge_score > 0:
            confidence_mult += 0.1 * min(hedge_score, 1.0)  # Boost for good hedges
        
        # Adjust for cluster exposure
        cluster_exp = risk_metadata.get('cluster_exposure', 0.0)
        if cluster_exp > 0.015:
            confidence_mult *= 0.8  # Reduce size if cluster already exposed
        
        risk_amount = max_risk_amount * confidence_mult
        
        # Calculate lots from risk amount
        stop_pips = abs(signal.entry_price - signal.stop_loss)
        pip_value = self._get_pip_value(signal.symbol)
        
        if stop_pips > 0 and pip_value > 0:
            lots = risk_amount / (stop_pips * pip_value)
            return round(max(0.01, lots), 2)
        
        return 0.01
    
    def execute_signal(self, signal: TradeSignal) -> Optional[Position]:
        """
        Execute a trade signal and add to portfolio.
        
        Returns created Position or None if rejected.
        """
        allowed, metadata = self.evaluate_signal(signal)
        
        if not allowed:
            logger.info(f"Signal rejected for {signal.symbol}: {metadata.get('rejection_reason')}")
            return None
        
        # Calculate position size
        size = self.calculate_position_size(signal, metadata)
        
        # Create position
        position = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            size=size,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_amount=self.balance * self.config.max_trade_risk_pct * 
                        (0.5 + signal.confidence * 0.5),
            entry_time=signal.timestamp,
            cluster=self.correlation.get_cluster(signal.symbol)
        )
        
        # Add to portfolio
        self.positions[signal.symbol] = position
        self.risk_manager.add_position(position)
        
        self.signals_executed += 1
        if metadata.get('hedge_score', 0) > 0.5:
            self.hedge_trades += 1
        
        # Record trade
        self.trade_history.append({
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'direction': signal.direction,
            'size': size,
            'entry': signal.entry_price,
            'stop': signal.stop_loss,
            'target': signal.take_profit,
            'confidence': signal.confidence,
            'hedge_score': metadata.get('hedge_score', 0.0),
            'strategy': signal.strategy
        })
        
        logger.info(f"Executed {signal.direction} {signal.symbol} @ {signal.entry_price}, "
                   f"size={size}, conf={signal.confidence:.2f}")
        
        return position
    
    def close_position(self, 
                       symbol: str, 
                       exit_price: float,
                       timestamp: Optional[datetime] = None) -> dict:
        """Close a position and record P&L."""
        symbol = symbol.upper()
        
        if symbol not in self.positions:
            return {'error': f'No position found for {symbol}'}
        
        position = self.positions.pop(symbol)
        self.risk_manager.remove_position(symbol)
        
        # Calculate P&L
        if position.direction == 'long':
            pnl_ticks = (exit_price - position.entry_price)
        else:
            pnl_ticks = (position.entry_price - exit_price)
        
        pip_value = self._get_pip_value(symbol)
        pnl = pnl_ticks * position.size * pip_value
        
        result = {
            'symbol': symbol,
            'direction': position.direction,
            'entry': position.entry_price,
            'exit': exit_price,
            'size': position.size,
            'pnl': pnl,
            'pnl_pct': pnl / self.balance if self.balance > 0 else 0,
            'duration': (timestamp or datetime.now()) - position.entry_time,
            'timestamp': timestamp or datetime.now()
        }
        
        logger.info(f"Closed {symbol} @ {exit_price}, P&L: ${pnl:.2f}")
        
        return result
    
    def get_anti_correlation_signals(self) -> list[TradeSignal]:
        """Generate signals from anti-correlation strategy."""
        signals = []
        
        # Check for divergence opportunities
        opportunities = self.anti_corr_strategy.find_divergence_opportunities()
        
        for opp in opportunities:
            if opp['type'] == 'convergence':
                # Create signal for mean reversion
                symbol = opp['pair_underperforming']
                
                # Only trade if we don't already have a position
                if symbol in self.positions:
                    continue
                
                signal = TradeSignal(
                    symbol=symbol,
                    direction=opp['direction'],
                    confidence=min(0.6 + opp.get('strength', 0) * 0.1, 0.75),
                    entry_price=0,  # Will be filled from market
                    stop_loss=0,    # Will be calculated
                    take_profit=0,  # Will be calculated
                    timestamp=datetime.now(),
                    strategy='anti_correlation',
                    metadata={
                        'divergence_pct': opp['divergence_pct'],
                        'rationale': opp['rationale']
                    }
                )
                signals.append(signal)
        
        # Check for correlation breakdown
        breakdown = self.anti_corr_strategy.check_correlation_breakdown()
        if breakdown:
            logger.warning(f"Correlation breakdown detected: {breakdown}")
        
        return signals
    
    def get_session_optimal_symbols(self, 
                                    timestamp: Optional[datetime] = None) -> list[str]:
        """Get symbols optimized for current trading session."""
        min_priority = self.config.min_session_priority
        
        if min_priority == 1:
            return self.session_selector.get_optimal_symbols(timestamp, 'primary')
        else:
            return self.session_selector.get_optimal_symbols(timestamp, 'secondary')
    
    def get_portfolio_state(self) -> dict[str, Any]:
        """Get comprehensive portfolio state."""
        # Get base risk summary
        risk_summary = self.risk_manager.get_portfolio_summary()
        
        # Add session info
        current_session = self.session_selector.get_current_session()
        optimal_symbols = self.get_session_optimal_symbols()
        
        # Get recommendations
        recommendations = self.risk_manager.recommend_position_adjustments()
        
        # Anti-correlation opportunities
        anti_corr_signals = self.get_anti_correlation_signals()
        
        return {
            'balance': self.balance,
            'open_positions': len(self.positions),
            'positions': [
                {
                    'symbol': p.symbol,
                    'direction': p.direction,
                    'size': p.size,
                    'entry': p.entry_price,
                    'stop': p.stop_loss,
                    'target': p.take_profit,
                    'cluster': p.cluster.value,
                    'risk': p.risk_amount
                }
                for p in self.positions.values()
            ],
            'current_session': current_session,
            'optimal_symbols': optimal_symbols,
            'risk_metrics': risk_summary,
            'recommendations': recommendations,
            'anti_correlation_signals': [
                {
                    'symbol': s.symbol,
                    'direction': s.direction,
                    'confidence': s.confidence,
                    'rationale': s.metadata.get('rationale', '')
                }
                for s in anti_corr_signals[:3]  # Top 3
            ],
            'performance': {
                'signals_generated': self.signals_generated,
                'signals_executed': self.signals_executed,
                'execution_rate': (self.signals_executed / max(1, self.signals_generated)),
                'hedge_trades': self.hedge_trades
            }
        }
    
    def rebalance_portfolio(self) -> list[dict]:
        """
        Analyze portfolio and suggest/execute rebalancing actions.
        
        Returns list of actions taken.
        """
        actions = []
        
        # Get current exposures
        exposures = self.risk_manager.calculate_cluster_exposures()
        
        # Check for cluster imbalances
        for cluster, exp in exposures.items():
            cluster_risk_pct = exp.risk_amount / self.balance if self.balance > 0 else 0
            
            if cluster_risk_pct > self.config.max_cluster_risk_pct:
                # Reduce exposure in this cluster
                actions.append({
                    'action': 'reduce_cluster',
                    'cluster': cluster.value,
                    'current_risk_pct': cluster_risk_pct,
                    'target_risk_pct': self.config.max_cluster_risk_pct * 0.8,
                    'suggested': 'Trim largest position in cluster'
                })
        
        # Check for missing clusters
        active_clusters = set(p.cluster for p in self.positions.values())
        all_clusters = set(SymbolCluster)
        
        missing = all_clusters - active_clusters
        if missing and len(self.positions) < self.config.max_positions:
            for cluster in missing:
                actions.append({
                    'action': 'add_diversification',
                    'cluster': cluster.value,
                    'suggested_symbols': self.correlation.get_cluster_symbols(cluster),
                    'rationale': f'No exposure to {cluster.value} cluster'
                })
        
        return actions
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for a symbol."""
        pip_values = {
            'EURUSD': 10.0,
            'GBPUSD': 10.0,
            'AUDUSD': 10.0,
            'EURJPY': 9.5,
            'GBPJPY': 9.5,
            'EURGBP': 13.5,
            'XAUUSD': 1.0,
        }
        return pip_values.get(symbol.upper(), 10.0)


class PortfolioSignalAggregator:
    """
    Aggregates signals from multiple strategies with correlation awareness.
    
    Prevents signal duplication across correlated pairs and prioritizes
    the best opportunities within each cluster.
    """
    
    def __init__(self, correlation: Optional[CorrelationMatrix] = None):
        self.correlation = correlation or CorrelationMatrix()
        self.signal_history: list[TradeSignal] = []
        
    def aggregate_signals(self, 
                         signals: list[TradeSignal],
                         max_per_cluster: int = 1) -> list[TradeSignal]:
        """
        Aggregate and filter signals to avoid correlated positions.
        
        Args:
            signals: List of raw signals
            max_per_cluster: Maximum signals to take per cluster
        
        Returns:
            Filtered list of signals
        """
        if not signals:
            return []
        
        # Group by cluster
        by_cluster: dict[SymbolCluster, list[TradeSignal]] = defaultdict(list)
        for sig in signals:
            cluster = self.correlation.get_cluster(sig.symbol)
            by_cluster[cluster].append(sig)
        
        # Select best signals from each cluster
        selected = []
        for cluster, cluster_signals in by_cluster.items():
            # Sort by confidence
            sorted_sigs = sorted(cluster_signals, 
                               key=lambda x: x.confidence, 
                               reverse=True)
            
            # Take top N from this cluster
            for sig in sorted_sigs[:max_per_cluster]:
                # Check if too correlated with already selected
                if self._is_too_correlated(sig, selected):
                    continue
                selected.append(sig)
        
        return selected
    
    def _is_too_correlated(self, 
                          new_signal: TradeSignal, 
                          selected: list[TradeSignal],
                          threshold: float = 0.7) -> bool:
        """Check if signal is too correlated with already selected signals."""
        for existing in selected:
            corr = self.correlation.get_correlation(new_signal.symbol, existing.symbol)
            
            # High positive correlation with same direction = too correlated
            if corr > threshold and new_signal.direction == existing.direction:
                return True
            
            # High negative correlation with opposite direction = too correlated
            if corr < -threshold and new_signal.direction != existing.direction:
                return True
        
        return False


def create_portfolio_from_config(config: dict) -> MultiSymbolPortfolio:
    """
    Create a portfolio from a configuration dictionary.
    
    Args:
        config: Configuration with keys like 'balance', 'max_risk', etc.
    
    Returns:
        Configured MultiSymbolPortfolio
    """
    balance = config.get('balance', 10000.0)
    
    portfolio_config = PortfolioConfig(
        max_total_risk_pct=config.get('max_total_risk_pct', 0.08),
        max_cluster_risk_pct=config.get('max_cluster_risk_pct', 0.03),
        max_trade_risk_pct=config.get('max_trade_risk_pct', 0.015),
        max_positions=config.get('max_positions', 7),
        session_filtering=config.get('session_filtering', True)
    )
    
    return MultiSymbolPortfolio(balance=balance, config=portfolio_config)


# Example usage and testing functions
def example_portfolio_setup():
    """Example of setting up a correlation-aware portfolio."""
    
    # Create portfolio
    portfolio = MultiSymbolPortfolio(
        balance=10000.0,
        config=PortfolioConfig(
            max_total_risk_pct=0.08,
            max_cluster_risk_pct=0.03,
            max_positions=7
        )
    )
    
    # Example: Add some price history
    prices = {
        'EURUSD': 1.0850,
        'GBPUSD': 1.2650,
        'AUDUSD': 0.6650,
        'EURJPY': 162.50,
        'GBPJPY': 189.50,
        'EURGBP': 0.8570,
        'XAUUSD': 2035.50
    }
    portfolio.update_prices(prices)
    
    # Get optimal symbols for current session
    optimal = portfolio.get_session_optimal_symbols()
    print(f"Optimal symbols: {optimal}")
    
    # Example signals
    signal1 = TradeSignal(
        symbol='EURUSD',
        direction='long',
        confidence=0.72,
        entry_price=1.0850,
        stop_loss=1.0820,
        take_profit=1.0910,
        timestamp=datetime.now(),
        strategy='momentum'
    )
    
    # Evaluate and potentially execute
    allowed, metadata = portfolio.evaluate_signal(signal1)
    if allowed:
        position = portfolio.execute_signal(signal1)
        print(f"Executed: {position}")
    
    # Get portfolio state
    state = portfolio.get_portfolio_state()
    print(f"Portfolio: {state['open_positions']} positions")
    
    return portfolio


if __name__ == '__main__':
    example_portfolio_setup()
