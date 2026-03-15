import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from forex_bot.features.advanced_signals import (
    calculate_hurst_exponent,
    kalman_filter,
    vertical_horizontal_filter,
    chande_momentum_oscillator,
    regime_detector
)

def test_indicators():
    print("Testing Advanced Indicators...")
    
    # Generate test data: 1000 points of a random walk
    np.random.seed(42)
    price = np.cumsum(np.random.randn(1000)) + 100
    
    # 1. Hurst Exponent (Should be ~0.5 for random walk)
    h = calculate_hurst_exponent(price)
    print(f"Hurst Exponent (Random Walk): {h:.4f}")
    
    # Generate a trend
    trend = np.linspace(0, 10, 1000) + np.random.randn(1000) * 0.1
    h_trend = calculate_hurst_exponent(trend)
    print(f"Hurst Exponent (Trending):    {h_trend:.4f}")
    
    # 2. Kalman Filter
    k = kalman_filter(price)
    print(f"Kalman Filter head: {k[:5]}")
    
    # 3. VHF
    vhf = vertical_horizontal_filter(price)
    print(f"VHF last value: {vhf[-1]:.4f}")
    
    # 4. CMO
    cmo = chande_momentum_oscillator(price)
    print(f"CMO last value: {cmo[-1]:.4f}")
    
    # 5. Regime Detector
    regime = regime_detector(price)
    print(f"Regime (Random): {regime['regime']}")
    
    regime_trend = regime_detector(trend)
    print(f"Regime (Trend):  {regime_trend['regime']}")
    
    print("\nTests complete.")

if __name__ == "__main__":
    test_indicators()
