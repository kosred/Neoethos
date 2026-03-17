"""
Correlation-Aware Portfolio Risk Management

Manages multi-symbol portfolios with correlation-based risk controls,
cluster analysis, and diversification strategies.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SymbolCluster(Enum):
    """Correlation-based symbol clusters."""
    USD_MAJORS = "usd_majors"      # EURUSD, GBPUSD, AUDUSD
    JPY_PAIRS = "jpy_pairs"        # EURJPY, GBPJPY
    EUROPEAN = "european"          # EURGBP
    SAFE_HAVEN = "safe_haven"      # XAUUSD
    COMMODITY = "commodity"        # AUDUSD (also commodity)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: str  # 'long' or 'short'
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    entry_time: datetime
    cluster: SymbolCluster = SymbolCluster.USD_MAJORS


@dataclass
class ClusterExposure:
    """Exposure metrics for a correlation cluster."""
    cluster: SymbolCluster
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    risk_amount: float = 0.0
    positions: list[Position] = field(default_factory=list)
    
    @property
    def direction_bias(self) -> float:
        """Returns -1 (all short) to +1 (all long)."""
        total = self.long_exposure + self.short_exposure
        if total == 0:
            return 0.0
        return (self.long_exposure - self.short_exposure) / total


class CorrelationMatrix:
    """
    Manages correlation data between currency pairs.
    
    Typical forex correlations (based on historical data):
    - EURUSD/GBPUSD: +0.85 (strong positive)
    - EURJPY/GBPJPY: +0.80 (strong positive)
    - EURUSD/EURJPY: +0.70 (moderate positive)
    - USD pairs / XAUUSD: -0.55 to -0.60 (inverse)
    """
    
    # Historical average correlation matrix (7 symbols)
    DEFAULT_CORRELATIONS: dict[str, dict[str, float]] = {
        "EURUSD": {"EURUSD": 1.00, "GBPUSD": 0.85, "EURJPY": 0.70, "GBPJPY": 0.65, 
                   "EURGBP": 0.40, "AUDUSD": 0.55, "XAUUSD": -0.60},
        "GBPUSD": {"EURUSD": 0.85, "GBPUSD": 1.00, "EURJPY": 0.60, "GBPJPY": 0.75,
                   "EURGBP": 0.70, "AUDUSD": 0.50, "XAUUSD": -0.55},
        "EURJPY": {"EURUSD": 0.70, "GBPUSD": 0.60, "EURJPY": 1.00, "GBPJPY": 0.80,
                   "EURGBP": 0.30, "AUDUSD": 0.65, "XAUUSD": -0.45},
        "GBPJPY": {"EURUSD": 0.65, "GBPUSD": 0.75, "EURJPY": 0.80, "GBPJPY": 1.00,
                   "EURGBP": 0.45, "AUDUSD": 0.60, "XAUUSD": -0.40},
        "EURGBP": {"EURUSD": 0.40, "GBPUSD": 0.70, "EURJPY": 0.30, "GBPJPY": 0.45,
                   "EURGBP": 1.00, "AUDUSD": 0.20, "XAUUSD": -0.25},
        "AUDUSD": {"EURUSD": 0.55, "GBPUSD": 0.50, "EURJPY": 0.65, "GBPJPY": 0.60,
                   "EURGBP": 0.20, "AUDUSD": 1.00, "XAUUSD": -0.30},
        "XAUUSD": {"EURUSD": -0.60, "GBPUSD": -0.55, "EURJPY": -0.45, "GBPJPY": -0.40,
                   "EURGBP": -0.25, "AUDUSD": -0.30, "XAUUSD": 1.00},
    }
    
    # Symbol to cluster mapping
    SYMBOL_CLUSTERS: dict[str, SymbolCluster] = {
        "EURUSD": SymbolCluster.USD_MAJORS,
        "GBPUSD": SymbolCluster.USD_MAJORS,
        "AUDUSD": SymbolCluster.USD_MAJORS,
        "EURJPY": SymbolCluster.JPY_PAIRS,
        "GBPJPY": SymbolCluster.JPY_PAIRS,
        "EURGBP": SymbolCluster.EUROPEAN,
        "XAUUSD": SymbolCluster.SAFE_HAVEN,
    }
    
    def __init__(self, correlation_data: Optional[dict] = None):
        self.correlations = correlation_data or self.DEFAULT_CORRELATIONS.copy()
        self.symbols = list(self.correlations.keys())
        self._matrix: Optional[np.ndarray] = None
        self._last_update = datetime.now()
        
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        s1, s2 = symbol1.upper(), symbol2.upper()
        if s1 not in self.correlations or s2 not in self.correlations[s1]:
            return 0.0
        return self.correlations[s1][s2]
    
    def get_cluster(self, symbol: str) -> SymbolCluster:
        """Get cluster for a symbol."""
        return self.SYMBOL_CLUSTERS.get(symbol.upper(), SymbolCluster.USD_MAJORS)
    
    def get_cluster_symbols(self, cluster: SymbolCluster) -> list[str]:
        """Get all symbols in a cluster."""
        return [s for s, c in self.SYMBOL_CLUSTERS.items() if c == cluster]
    
    def get_matrix(self) -> pd.DataFrame:
        """Get correlation matrix as DataFrame."""
        return pd.DataFrame(self.correlations)
    
    def update_correlation(self, symbol1: str, symbol2: str, value: float) -> None:
        """Update correlation value (for dynamic updates)."""
        s1, s2 = symbol1.upper(), symbol2.upper()
        if s1 in self.correlations:
            self.correlations[s1][s2] = value
        if s2 in self.correlations:
            self.correlations[s2][s1] = value
        self._matrix = None
        self._last_update = datetime.now()
    
    def compute_portfolio_correlation(self, positions: dict[str, Position]) -> float:
        """
        Compute average pairwise correlation of portfolio positions.
        Higher = more concentrated risk.
        """
        if len(positions) < 2:
            return 0.0
        
        symbols = list(positions.keys())
        total_corr = 0.0
        count = 0
        
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                corr = self.get_correlation(s1, s2)
                # Weight by position sizes
                weight = abs(positions[s1].size * positions[s2].size)
                total_corr += abs(corr) * weight
                count += weight
        
        return total_corr / count if count > 0 else 0.0
    
    def find_high_correlation_pairs(self, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """Find all pairs with correlation above threshold."""
        pairs = []
        symbols = self.symbols
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                corr = self.get_correlation(s1, s2)
                if abs(corr) >= threshold:
                    pairs.append((s1, s2, corr))
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


@dataclass
class RiskLimits:
    """Portfolio risk limits configuration."""
    max_cluster_risk_pct: float = 0.03          # 3% max per cluster
    max_trade_risk_pct: float = 0.015           # 1.5% max per trade
    max_total_risk_pct: float = 0.08            # 8% total portfolio
    max_portfolio_correlation: float = 0.70     # Max avg correlation
    max_positions_per_cluster: int = 2          # Max positions per cluster
    min_hedge_score: float = -0.5               # Minimum hedge benefit
    correlation_threshold: float = 0.70         # High correlation threshold


class CorrelationRiskManager:
    """
    Manages portfolio risk with correlation awareness.
    
    Key principles:
    1. Cluster-based risk limits prevent over-concentration
    2. Hedge scoring rewards diversification
    3. Correlation-adjusted position sizing
    4. Dynamic exposure monitoring
    """
    
    def __init__(self, 
                 balance: float = 10000.0,
                 limits: Optional[RiskLimits] = None,
                 correlation_matrix: Optional[CorrelationMatrix] = None):
        self.balance = balance
        self.limits = limits or RiskLimits()
        self.correlation = correlation_matrix or CorrelationMatrix()
        
        # Portfolio state
        self.positions: dict[str, Position] = {}
        self.cluster_exposures: dict[SymbolCluster, ClusterExposure] = {}
        self._exposure_history: list[dict] = []
        
        # Statistics
        self.trades_blocked = 0
        self.trades_allowed = 0
        self.hedge_opportunities_taken = 0
        
        self._initialize_clusters()
        
    def _initialize_clusters(self) -> None:
        """Initialize cluster exposure trackers."""
        for cluster in SymbolCluster:
            self.cluster_exposures[cluster] = ClusterExposure(cluster=cluster)
    
    def calculate_cluster_exposures(self) -> dict[SymbolCluster, ClusterExposure]:
        """Calculate current exposure for each cluster."""
        self._initialize_clusters()
        
        for pos in self.positions.values():
            cluster = self.correlation.get_cluster(pos.symbol)
            exposure = self.cluster_exposures[cluster]
            
            exposure.positions.append(pos)
            exposure.risk_amount += pos.risk_amount
            
            if pos.direction == 'long':
                exposure.long_exposure += pos.size
            else:
                exposure.short_exposure += pos.size
                
            exposure.net_exposure = exposure.long_exposure - exposure.short_exposure
            exposure.gross_exposure = exposure.long_exposure + exposure.short_exposure
            
        return self.cluster_exposures
    
    def check_portfolio_risk(self) -> tuple[bool, str]:
        """
        Check if current portfolio violates risk limits.
        
        Returns:
            (allowed, reason) tuple
        """
        exposures = self.calculate_cluster_exposures()
        total_risk = sum(e.risk_amount for e in exposures.values())
        
        # Check total portfolio risk
        total_risk_pct = total_risk / self.balance if self.balance > 0 else 0
        if total_risk_pct > self.limits.max_total_risk_pct:
            return False, f"Total risk {total_risk_pct:.1%} exceeds {self.limits.max_total_risk_pct:.1%}"
        
        # Check per-cluster risk
        for cluster, exposure in exposures.items():
            cluster_risk_pct = exposure.risk_amount / self.balance if self.balance > 0 else 0
            if cluster_risk_pct > self.limits.max_cluster_risk_pct:
                return False, f"Cluster {cluster.value} risk {cluster_risk_pct:.1%} exceeds limit"
            
            # Check positions per cluster
            if len(exposure.positions) > self.limits.max_positions_per_cluster:
                return False, f"Too many positions in {cluster.value}"
        
        # Check portfolio correlation
        if len(self.positions) >= 2:
            port_corr = self.correlation.compute_portfolio_correlation(self.positions)
            if port_corr > self.limits.max_portfolio_correlation:
                return False, f"Portfolio correlation {port_corr:.2f} too high"
        
        return True, "OK"
    
    def calculate_hedge_score(self, new_trade: dict, existing_only: bool = False) -> float:
        """
        Score how well a new trade hedges existing exposure.
        
        Positive score = good hedge (reduces risk)
        Negative score = increases concentration
        
        Scoring logic:
        - Same direction + high positive correlation = bad (-correlation)
        - Opposite direction + high positive correlation = good (+2x correlation)
        - Opposite direction + negative correlation = bad (-abs correlation)
        """
        score = 0.0
        new_symbol = new_trade['symbol'].upper()
        new_direction = new_trade.get('direction', 'long')
        
        positions = self.positions if not existing_only else {}
        if not existing_only:
            existing_positions = new_trade.get('existing_positions')
            if existing_positions:
                positions = existing_positions
        
        if not positions:
            return 0.0
        
        for symbol, position in positions.items():
            if symbol == new_symbol:
                continue
                
            corr = self.correlation.get_correlation(new_symbol, symbol)
            
            if new_direction != position.direction:
                # Opposite direction trades
                if corr > self.limits.correlation_threshold:
                    # Positive correlation + opposite direction = good hedge
                    score += corr * 2.0
                elif corr < -0.5:
                    # Negative correlation + opposite direction = both could lose
                    score -= abs(corr) * 0.5
            else:
                # Same direction trades
                if corr > self.limits.correlation_threshold:
                    # Adding to same-side exposure = concentration risk
                    score -= corr
                elif corr < -0.3:
                    # Negative correlation + same direction = natural hedge
                    score += abs(corr) * 0.5
        
        return score
    
    def validate_new_trade(self, 
                          symbol: str,
                          direction: str,
                          size: float,
                          stop_loss: float,
                          entry_price: float) -> tuple[bool, dict]:
        """
        Validate a new trade against correlation risk rules.
        
        Returns:
            (allowed, metadata) tuple where metadata contains:
            - hedge_score: hedge benefit score
            - cluster_exposure: current cluster exposure
            - risk_amount: calculated risk
            - reason: rejection reason if not allowed
        """
        symbol = symbol.upper()
        cluster = self.correlation.get_cluster(symbol)
        
        metadata = {
            'hedge_score': 0.0,
            'cluster_exposure': 0.0,
            'risk_amount': 0.0,
            'reason': 'OK'
        }
        
        # Calculate risk amount
        risk_pips = abs(entry_price - stop_loss)
        pip_value = self._get_pip_value(symbol)
        risk_amount = size * risk_pips * pip_value
        risk_pct = risk_amount / self.balance if self.balance > 0 else 0
        
        metadata['risk_amount'] = risk_amount
        
        # Check max trade risk
        if risk_pct > self.limits.max_trade_risk_pct:
            metadata['reason'] = f"Trade risk {risk_pct:.2%} exceeds {self.limits.max_trade_risk_pct:.2%}"
            self.trades_blocked += 1
            return False, metadata
        
        # Calculate current exposures
        exposures = self.calculate_cluster_exposures()
        cluster_exp = exposures[cluster]
        
        metadata['cluster_exposure'] = cluster_exp.risk_amount / self.balance if self.balance > 0 else 0
        
        # Check cluster risk limit
        new_cluster_risk = (cluster_exp.risk_amount + risk_amount) / self.balance if self.balance > 0 else 0
        if new_cluster_risk > self.limits.max_cluster_risk_pct:
            metadata['reason'] = f"Cluster risk would be {new_cluster_risk:.2%}, exceeds limit"
            self.trades_blocked += 1
            return False, metadata
        
        # Calculate hedge score
        trade_info = {'symbol': symbol, 'direction': direction}
        hedge_score = self.calculate_hedge_score(trade_info)
        metadata['hedge_score'] = hedge_score
        
        # Check if trade reduces portfolio correlation
        # Simulate adding this position
        simulated_position = Position(
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=entry_price + (entry_price - stop_loss) * 2,  # 1:2 R:R
            risk_amount=risk_amount,
            entry_time=datetime.now(),
            cluster=cluster
        )
        
        # Check portfolio correlation with new position
        if len(self.positions) >= 1:
            sim_positions = self.positions.copy()
            sim_positions[symbol] = simulated_position
            new_port_corr = self.correlation.compute_portfolio_correlation(sim_positions)
            
            if new_port_corr > self.limits.max_portfolio_correlation:
                # High correlation - check if hedge score compensates
                if hedge_score < self.limits.min_hedge_score:
                    metadata['reason'] = f"Portfolio correlation {new_port_corr:.2f} too high, poor hedge"
                    self.trades_blocked += 1
                    return False, metadata
        
        # Trade is allowed
        self.trades_allowed += 1
        if hedge_score > 0.5:
            self.hedge_opportunities_taken += 1
        
        return True, metadata
    
    def add_position(self, position: Position) -> None:
        """Add a position to the portfolio."""
        position.cluster = self.correlation.get_cluster(position.symbol)
        self.positions[position.symbol] = position
        self.calculate_cluster_exposures()
        
    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove a position from the portfolio."""
        position = self.positions.pop(symbol.upper(), None)
        if position:
            self.calculate_cluster_exposures()
        return position
    
    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get comprehensive portfolio summary."""
        exposures = self.calculate_cluster_exposures()
        
        total_risk = sum(e.risk_amount for e in exposures.values())
        total_positions = len(self.positions)
        
        # Calculate weighted correlation
        port_corr = self.correlation.compute_portfolio_correlation(self.positions) if total_positions >= 2 else 0.0
        
        cluster_summary = {}
        for cluster, exp in exposures.items():
            cluster_summary[cluster.value] = {
                'net_exposure': exp.net_exposure,
                'gross_exposure': exp.gross_exposure,
                'risk_amount': exp.risk_amount,
                'risk_pct': exp.risk_amount / self.balance if self.balance > 0 else 0,
                'positions_count': len(exp.positions),
                'direction_bias': exp.direction_bias,
                'symbols': [p.symbol for p in exp.positions]
            }
        
        return {
            'balance': self.balance,
            'total_positions': total_positions,
            'total_risk_amount': total_risk,
            'total_risk_pct': total_risk / self.balance if self.balance > 0 else 0,
            'portfolio_correlation': port_corr,
            'clusters': cluster_summary,
            'trades_allowed': self.trades_allowed,
            'trades_blocked': self.trades_blocked,
            'hedge_opportunities_taken': self.hedge_opportunities_taken
        }
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for a symbol."""
        # Standard forex pip values (approximate)
        pip_values = {
            'EURUSD': 10.0,
            'GBPUSD': 10.0,
            'AUDUSD': 10.0,
            'EURJPY': 9.5,
            'GBPJPY': 9.5,
            'EURGBP': 13.5,
            'XAUUSD': 1.0,  # Gold
        }
        return pip_values.get(symbol.upper(), 10.0)
    
    def recommend_position_adjustments(self) -> list[dict]:
        """
        Recommend portfolio adjustments based on correlation.
        
        Returns list of recommendations to reduce concentration risk.
        """
        recommendations = []
        exposures = self.calculate_cluster_exposures()
        
        # Find over-exposed clusters
        for cluster, exp in exposures.items():
            cluster_risk_pct = exp.risk_amount / self.balance if self.balance > 0 else 0
            
            if cluster_risk_pct > self.limits.max_cluster_risk_pct * 0.8:
                # Near limit - suggest reducing
                excess_risk = exp.risk_amount - (self.balance * self.limits.max_cluster_risk_pct * 0.8)
                
                recommendations.append({
                    'action': 'reduce_cluster_exposure',
                    'cluster': cluster.value,
                    'current_risk_pct': cluster_risk_pct,
                    'suggested_reduction': excess_risk,
                    'priority': 'high' if cluster_risk_pct > self.limits.max_cluster_risk_pct else 'medium'
                })
            
            # Check for one-sided exposure
            if abs(exp.direction_bias) > 0.8 and exp.gross_exposure > 0:
                recommendations.append({
                    'action': 'add_hedge',
                    'cluster': cluster.value,
                    'current_bias': exp.direction_bias,
                    'suggested': 'short' if exp.direction_bias > 0 else 'long',
                    'priority': 'medium'
                })
        
        # Check portfolio correlation
        if len(self.positions) >= 2:
            port_corr = self.correlation.compute_portfolio_correlation(self.positions)
            if port_corr > 0.6:
                recommendations.append({
                    'action': 'reduce_correlation',
                    'current_correlation': port_corr,
                    'suggestion': 'Consider adding uncorrelated positions or reducing correlated ones',
                    'priority': 'high' if port_corr > 0.75 else 'medium'
                })
        
        return sorted(recommendations, key=lambda x: 0 if x['priority'] == 'high' else 1)


class AntiCorrelationStrategy:
    """
    Trading strategy based on correlation breakdowns and divergences.
    
    When highly correlated pairs diverge, trade the mean reversion.
    """
    
    # Pairs that typically move together
    CORRELATED_PAIRS = [
        ('EURUSD', 'GBPUSD', 0.85),
        ('EURJPY', 'GBPJPY', 0.80),
        ('EURUSD', 'AUDUSD', 0.55),
        ('GBPUSD', 'EURJPY', 0.60),
    ]
    
    def __init__(self, 
                 correlation_matrix: Optional[CorrelationMatrix] = None,
                 divergence_threshold: float = 0.5):
        self.correlation = correlation_matrix or CorrelationMatrix()
        self.divergence_threshold = divergence_threshold  # % divergence to trigger
        self.price_history: dict[str, list[tuple[datetime, float]]] = {}
        self.signals: list[dict] = []
        
    def update_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """Update price history for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        ts = timestamp or datetime.now()
        self.price_history[symbol].append((ts, price))
        
        # Keep last 1000 price points
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def get_24h_change(self, symbol: str) -> Optional[float]:
        """Get 24-hour price change percentage."""
        if symbol not in self.price_history:
            return None
        
        history = self.price_history[symbol]
        if len(history) < 2:
            return None
        
        now = history[-1]
        cutoff = now[0] - timedelta(hours=24)
        
        # Find price 24h ago
        old_price = None
        for ts, price in reversed(history[:-1]):
            if ts <= cutoff:
                old_price = price
                break
        
        if old_price is None:
            old_price = history[0][1]
        
        return ((now[1] - old_price) / old_price) * 100 if old_price != 0 else 0
    
    def find_divergence_opportunities(self) -> list[dict]:
        """
        Find pairs that should move together but are diverging.
        
        Returns trading opportunities based on mean reversion.
        """
        opportunities = []
        
        for sym1, sym2, expected_corr in self.CORRELATED_PAIRS:
            change1 = self.get_24h_change(sym1)
            change2 = self.get_24h_change(sym2)
            
            if change1 is None or change2 is None:
                continue
            
            divergence = abs(change1 - change2)
            
            if divergence > self.divergence_threshold:
                # Significant divergence detected
                if change1 > change2:
                    # sym1 up more than sym2 - sym2 should catch up
                    opportunities.append({
                        'type': 'convergence',
                        'pair_underperforming': sym2,
                        'pair_overperforming': sym1,
                        'direction': 'long',
                        'divergence_pct': divergence,
                        'expected_corr': expected_corr,
                        'strength': divergence / self.divergence_threshold,
                        'rationale': f'{sym2} lagging {sym1} by {divergence:.2f}%, expected convergence'
                    })
                else:
                    opportunities.append({
                        'type': 'convergence',
                        'pair_underperforming': sym1,
                        'pair_overperforming': sym2,
                        'direction': 'long',
                        'divergence_pct': divergence,
                        'expected_corr': expected_corr,
                        'strength': divergence / self.divergence_threshold,
                        'rationale': f'{sym1} lagging {sym2} by {divergence:.2f}%, expected convergence'
                    })
                
                # Also add the inverse opportunity
                if change1 > change2:
                    opportunities.append({
                        'type': 'hedge',
                        'pair': sym1,
                        'direction': 'short',
                        'divergence_pct': divergence,
                        'rationale': f'Hedge: {sym1} may correct while {sym2} catches up'
                    })
        
        return sorted(opportunities, key=lambda x: x.get('strength', 0), reverse=True)
    
    def check_correlation_breakdown(self, 
                                    window: int = 20,
                                    breakdown_threshold: float = 0.3) -> Optional[dict]:
        """
        Check if historical correlation is breaking down.
        
        Returns alert if rolling correlation differs significantly from expected.
        """
        alerts = []
        
        for sym1, sym2, expected_corr in self.CORRELATED_PAIRS:
            if sym1 not in self.price_history or sym2 not in self.price_history:
                continue
            
            # Calculate rolling correlation
            prices1 = [p for _, p in self.price_history[sym1][-window:]]
            prices2 = [p for _, p in self.price_history[sym2][-window:]]
            
            if len(prices1) < window or len(prices2) < window:
                continue
            
            # Calculate returns
            returns1 = np.diff(prices1) / prices1[:-1]
            returns2 = np.diff(prices2) / prices2[:-1]
            
            if len(returns1) < 2 or len(returns2) < 2:
                continue
            
            # Rolling correlation
            current_corr = np.corrcoef(returns1, returns2)[0, 1]
            
            if abs(current_corr - expected_corr) > breakdown_threshold:
                alerts.append({
                    'type': 'correlation_breakdown',
                    'pair1': sym1,
                    'pair2': sym2,
                    'expected_corr': expected_corr,
                    'current_corr': current_corr,
                    'deviation': abs(current_corr - expected_corr),
                    'signal': 'regime_change' if current_corr < 0.3 else 'weakening'
                })
        
        return alerts[0] if alerts else None


class SessionBasedSymbolSelector:
    """
    Select optimal symbols based on trading session.
    
    Different forex pairs have different liquidity and volatility
    characteristics during different trading sessions.
    """
    
    # Optimal symbols by session (based on volume and volatility)
    SESSION_SYMBOLS = {
        'asia': {
            'primary': ['AUDUSD', 'EURJPY', 'GBPJPY'],
            'secondary': ['EURUSD', 'GBPUSD'],
            'avoid': ['XAUUSD', 'EURGBP']  # Low liquidity
        },
        'london': {
            'primary': ['EURUSD', 'GBPUSD', 'EURGBP'],
            'secondary': ['EURJPY', 'GBPJPY'],
            'avoid': ['AUDUSD']  # Asia session ending
        },
        'ny': {
            'primary': ['EURUSD', 'GBPUSD', 'XAUUSD'],
            'secondary': ['AUDUSD'],
            'avoid': ['EURGBP']  # European session ending
        },
        'overlap': {  # London-NY overlap (most liquid)
            'primary': ['EURUSD', 'GBPUSD', 'EURJPY', 'GBPJPY', 'XAUUSD'],
            'secondary': ['AUDUSD', 'EURGBP'],
            'avoid': []
        },
        'offhours': {
            'primary': [],  # Avoid trading
            'secondary': [],
            'avoid': ['EURUSD', 'GBPUSD', 'EURJPY', 'GBPJPY', 'EURGBP', 'AUDUSD', 'XAUUSD']
        }
    }
    
    # Session times (UTC)
    SESSION_HOURS = {
        'asia': (0, 8),      # 00:00 - 08:00 UTC
        'london': (8, 16),   # 08:00 - 16:00 UTC
        'overlap': (13, 17), # 13:00 - 17:00 UTC (London-NY)
        'ny': (13, 22),      # 13:00 - 22:00 UTC
    }
    
    def __init__(self):
        self.current_session: Optional[str] = None
        self.session_start: Optional[datetime] = None
        
    def get_current_session(self, timestamp: Optional[datetime] = None) -> str:
        """Determine current trading session based on UTC time."""
        ts = timestamp or datetime.now()
        hour = ts.hour
        
        # Check for overlap first
        if 13 <= hour < 17:
            return 'overlap'
        
        # Weekend check
        if ts.weekday() >= 5:
            return 'offhours'
        
        # Check sessions
        if 0 <= hour < 8:
            return 'asia'
        elif 8 <= hour < 16:
            return 'london'
        elif 13 <= hour < 22:
            return 'ny'
        else:
            return 'offhours'
    
    def get_optimal_symbols(self, 
                           timestamp: Optional[datetime] = None,
                           min_liquidity: str = 'primary') -> list[str]:
        """
        Get list of optimal symbols for current session.
        
        Args:
            min_liquidity: 'primary' (highest), 'secondary', or 'all'
        """
        session = self.get_current_session(timestamp)
        
        if min_liquidity == 'primary':
            return self.SESSION_SYMBOLS[session]['primary']
        elif min_liquidity == 'secondary':
            return (self.SESSION_SYMBOLS[session]['primary'] + 
                   self.SESSION_SYMBOLS[session]['secondary'])
        else:
            # All except avoid
            all_symbols = []
            for cat in ['primary', 'secondary']:
                all_symbols.extend(self.SESSION_SYMBOLS[session][cat])
            return all_symbols
    
    def should_trade_symbol(self, 
                           symbol: str,
                           timestamp: Optional[datetime] = None) -> tuple[bool, str]:
        """
        Check if a symbol should be traded in current session.
        
        Returns:
            (should_trade, reason) tuple
        """
        session = self.get_current_session(timestamp)
        config = self.SESSION_SYMBOLS[session]
        
        if symbol in config['avoid']:
            return False, f"{symbol} has poor liquidity during {session} session"
        
        if symbol in config['primary']:
            return True, "optimal_liquidity"
        
        if symbol in config['secondary']:
            return True, "acceptable_liquidity"
        
        return False, f"{symbol} not recommended for {session} session"
    
    def get_symbol_priority(self, 
                           symbols: list[str],
                           timestamp: Optional[datetime] = None) -> list[tuple[str, int]]:
        """
        Rank symbols by priority for current session.
        
        Returns list of (symbol, priority) where lower priority = better.
        """
        session = self.get_current_session(timestamp)
        config = self.SESSION_SYMBOLS[session]
        
        rankings = []
        for sym in symbols:
            if sym in config['primary']:
                rankings.append((sym, 1))
            elif sym in config['secondary']:
                rankings.append((sym, 2))
            elif sym not in config['avoid']:
                rankings.append((sym, 3))
            else:
                rankings.append((sym, 99))  # Avoid
        
        return sorted(rankings, key=lambda x: x[1])


def calculate_optimal_exposure(balance: float,
                               num_clusters: int = 4,
                               total_max_drawdown: float = 0.08,
                               cluster_max_factor: float = 0.25) -> dict:
    """
    Calculate optimal risk distribution across clusters.
    
    Args:
        balance: Account balance
        num_clusters: Number of correlation clusters
        total_max_drawdown: Maximum acceptable portfolio drawdown
        cluster_max_factor: Max % of total risk per cluster (0.25 = 25%)
    
    Returns:
        Dictionary with risk limits per cluster and trade
    """
    # Total risk budget
    total_risk_budget = balance * total_max_drawdown
    
    # Risk per cluster (distribute evenly)
    cluster_risk = total_risk_budget * cluster_max_factor
    
    # Per trade risk (within cluster, allow 2-3 trades)
    trade_risk = cluster_risk / 2.5  # Conservative: 2.5 trades per cluster
    
    # Hard limits
    cluster_max_pct = cluster_max_factor * total_max_drawdown
    trade_max_pct = cluster_max_pct / 2.5
    
    return {
        'total_risk_budget': total_risk_budget,
        'cluster_max_risk_amount': cluster_risk,
        'cluster_max_risk_pct': cluster_max_pct,
        'trade_max_risk_amount': trade_risk,
        'trade_max_risk_pct': trade_max_pct,
        'total_max_drawdown_pct': total_max_drawdown,
        'recommended_positions': num_clusters * 2  # 2 per cluster
    }


# Convenience function for quick setup
def create_correlation_portfolio(balance: float = 10000.0,
                                  custom_limits: Optional[dict] = None) -> CorrelationRiskManager:
    """
    Create a pre-configured correlation-aware risk manager.
    
    Args:
        balance: Starting balance
        custom_limits: Override default risk limits
    
    Returns:
        Configured CorrelationRiskManager instance
    """
    limits = RiskLimits()
    
    if custom_limits:
        for key, value in custom_limits.items():
            if hasattr(limits, key):
                setattr(limits, key, value)
    
    return CorrelationRiskManager(
        balance=balance,
        limits=limits,
        correlation_matrix=CorrelationMatrix()
    )
