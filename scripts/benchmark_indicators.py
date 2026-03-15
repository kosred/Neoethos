import time
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from forex_bot.features import enhanced_indicators as py_indicators
try:
    import forex_bindings as rust_indicators
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

def benchmark_indicator(name, func_py, func_rust, *args, iterations=5):
    print(f"\nBenchmarking {name}...")
    
    # Warmup
    func_py(*args)
    if RUST_AVAILABLE:
        func_rust(*args)
    
    # Python timing
    start = time.perf_counter()
    for _ in range(iterations):
        func_py(*args)
    py_time = (time.perf_counter() - start) / iterations
    print(f"  Python avg: {py_time:.6f}s")
    
    if RUST_AVAILABLE:
        # Rust timing
        start = time.perf_counter()
        for _ in range(iterations):
            func_rust(*args)
        rust_time = (time.perf_counter() - start) / iterations
        print(f"  Rust avg:   {rust_time:.6f}s")
        print(f"  Speedup:    {py_time / rust_time:.2f}x")
        return py_time, rust_time
    else:
        print("  Rust not available.")
        return py_time, None

def run_suite(size=100_000):
    print(f"--- Running Benchmark Suite (N={size:,}) ---")
    
    # Generate dummy data
    np.random.seed(42)
    close = np.random.randn(size).cumsum() + 100
    high = close + np.random.rand(size) * 2
    low = close - np.random.rand(size) * 2
    
    results = {}
    
    # 1. Causal Tanh Z-Score
    # Force pure python for the benchmark function reference
    def py_zscore(v):
        # Implementation from enhanced_indicators.py without the rust check
        arr = np.asarray(v, dtype=np.float64)
        out = np.zeros(arr.shape[0], dtype=np.float64)
        mean, m2, count = 0.0, 0.0, 0
        for i, val in enumerate(arr):
            if count >= 30:
                var = m2 / count
                std = np.sqrt(var) if var > 0 else 0.0
                z = (val - mean) / std if std > 1e-12 else (val - mean)
                out[i] = np.tanh(z)
            if np.isfinite(val):
                count += 1
                delta = val - mean
                mean += delta / count
                m2 += delta * (val - mean)
        return out

    if RUST_AVAILABLE and hasattr(rust_indicators, "causal_tanh_zscore"):
        results["Causal Tanh Z-Score"] = benchmark_indicator(
            "Causal Tanh Z-Score", py_zscore, rust_indicators.causal_tanh_zscore, close
        )

    # 2. Vortex Indicator
    def py_vortex(h, l, c):
        # Python implementation logic
        n = len(c)
        period = 14
        vp = np.ones(n)
        vm = np.ones(n)
        if n < period + 1: return vp, vm
        vmp = np.abs(h[1:] - l[:-1])
        vmm = np.abs(l[1:] - h[:-1])
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        
        def rolling_sum(arr, w):
            res = np.zeros(len(arr))
            cs = np.cumsum(arr)
            res[w-1] = cs[w-1]
            res[w:] = cs[w:] - cs[:-w]
            return res
            
        s_vmp = rolling_sum(np.pad(vmp, (1, 0)), period)
        s_vmm = rolling_sum(np.pad(vmm, (1, 0)), period)
        s_tr = rolling_sum(np.pad(tr, (1, 0)), period)
        mask = s_tr > 0
        vp[mask] = s_vmp[mask] / s_tr[mask]
        vm[mask] = s_vmm[mask] / s_tr[mask]
        return vp, vm

    if RUST_AVAILABLE and hasattr(rust_indicators, "vortex_indicator"):
        results["Vortex Indicator"] = benchmark_indicator(
            "Vortex Indicator", py_vortex, rust_indicators.vortex_indicator, high, low, close
        )

    # 3. Fisher Transform
    def py_fisher(p):
        period = 10
        n = len(p)
        fisher = np.zeros(n)
        value = np.zeros(n)
        for i in range(period, n):
            win = p[i-period+1:i+1]
            p_min, p_max = np.min(win), np.max(win)
            if p_max != p_min:
                val = 0.66 * ((p[i] - p_min) / (p_max - p_min) - 0.5) + 0.67 * value[i-1]
            else: val = 0.0
            val = max(-0.99, min(0.99, val))
            value[i] = val
            fisher[i] = 0.5 * np.log((1 + val) / (1 - val)) + 0.5 * fisher[i-1]
        return fisher

    if RUST_AVAILABLE and hasattr(rust_indicators, "fisher_transform"):
        results["Fisher Transform"] = benchmark_indicator(
            "Fisher Transform", py_fisher, rust_indicators.fisher_transform, close
        )

if __name__ == "__main__":
    run_suite(10_000)
    run_suite(100_000)
