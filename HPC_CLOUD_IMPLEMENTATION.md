# HPC Cloud Implementation for Hyperstack N3

This document describes the HPC-specific optimizations implemented for running on Hyperstack N3-RTX-A6000x8 instances:
- **8× RTX A6000** (48GB VRAM each = 384GB total)
- **252 AMD EPYC physical cores** with SMT = **504 logical threads**
- **464GB DDR4-3200 RAM**
- **NVLink** 112GB/s between GPU pairs

## Hardware Auto-Detection

The system automatically detects Hyperstack N3 hardware and activates HPC mode:

```rust
use forex_search::hpc::detect_hyperstack_n3;

if detect_hyperstack_n3() {
    println!("🚀 HPC mode activated!");
}
```

Detection criteria:
- ≥ 8 GPUs with ≥ 40GB VRAM each
- ≥ 500 logical threads (252 physical cores × 2 with SMT)
- ≥ 450GB total RAM

## New Modules

### 1. `hpc.rs` - Hardware Detection & NUMA Management

**Features:**
- `detect_hyperstack_n3()` - Detect N3 hardware
- `get_gpu_cpu_affinity(gpu_id)` - Get optimal CPU cores for each GPU
- `set_thread_affinity(cores)` - Pin threads to specific cores
- `is_nvlink_pair(a, b)` - Check GPU NVLink connectivity

**Topology (with SMT):**
```
Socket 0 (NUMA 0):
  - GPUs 0-3
  - Primary threads: 0-125
  - SMT threads: 252-377
  - Total: 252 logical threads

Socket 1 (NUMA 1):
  - GPUs 4-7
  - Primary threads: 126-251
  - SMT threads: 378-503
  - Total: 252 logical threads

NVLink pairs: (0,1), (2,3), (4,5), (6,7)
```

### 2. `hpc_gpu_discovery.rs` - Island Model GA

**Features:**
- 8 islands (one per GPU) evolving independently
- NVLink-based elite migration every N generations
- NUMA-aware thread pinning per island
- Larger chunk sizes (8192) for A6000's 48GB VRAM

**Usage:**
```rust
use forex_search::hpc_gpu_discovery::{run_island_model_discovery, IslandConfig};

let config = IslandConfig {
    base_config: GpuDiscoveryConfig::default(),
    migration_interval: 10,
    migration_fraction: 0.05,
    num_islands: 8,
};

let result = run_island_model_discovery(&frames, &ohlcv, &config)?;
```

### 3. `hpc_simd.rs` - SIMD-Optimized CPU Validation

**Features:**
- AVX2/FMA-optimized backtesting
- SIMD signal computation
- Fast Sharpe ratio calculation
- Automatic scalar fallback if AVX2 unavailable

**Usage:**
```rust
use forex_search::hpc_simd::{compute_sharpe_ratio, batch_evaluate_simd};

let sharpe = compute_sharpe_ratio(&returns);
let signals = batch_evaluate_simd(&features, &indices, &weights, &long_th, &short_th);
```

## Launch Scripts

### `scripts/run_hyperstack_n3.sh`

Unified launcher with hardware verification and system optimization:

```bash
# Check hardware
./scripts/run_hyperstack_n3.sh check

# Run discovery on all 8 GPUs
./scripts/run_hyperstack_n3.sh discovery

# Run dual-socket discovery (2 processes, optimal for NUMA)
./scripts/run_hyperstack_n3.sh dual

# Run full 20-hour pipeline
./scripts/run_hyperstack_n3.sh full
```

## System Optimizations Applied

1. **CPU Governor**: Set to `performance`
2. **Hugepages**: 1024 hugepages allocated
3. **NUMA Balancing**: Disabled for predictable allocation
4. **Swappiness**: Reduced to 10
5. **GPU Persistence Mode**: Enabled

## Thread Affinity

Automatic thread pinning based on GPU:

```rust
// GPU 0-3 use Socket 0 cores
let cores = get_gpu_cpu_affinity(0); // Returns 0..=125
set_thread_affinity(&cores)?;

// GPU 4-7 use Socket 1 cores
let cores = get_gpu_cpu_affinity(4); // Returns 126..=251
set_thread_affinity(&cores)?;
```

## Environment Variables

HPC mode recognizes these environment variables:

```bash
# Force HPC mode (for testing)
FOREX_BOT_HPC_MODE=1

# Island model settings
FOREX_BOT_ISLAND_MODEL=1
FOREX_BOT_ISLAND_MIGRATION_INTERVAL=10

# Population size (default 500K in HPC mode)
FOREX_BOT_POPULATION=500000

# Thread settings
RAYON_NUM_THREADS=240
FOREX_BOT_CPU_THREADS=240
```

## Expected Performance

| Metric | Conservative | Aggressive |
|--------|--------------|------------|
| GPU throughput | 200K evals/sec/GPU | 500K evals/sec/GPU |
| Total GPU evals (18h) | 115 billion | 288 billion |
| CPU validation | 25K strat/sec | 50K strat/sec |
| Final strategies | 1,000 | 5,000 |

## Usage from Python

The HPC features are automatically used when `forex_bindings` is compiled with GPU support:

```python
import forex_bindings as fb

# Hardware detection will auto-activate HPC mode
result = fb.run_gpu_discovery_island(
    frames, 
    ohlcv,
    population=500_000,
    generations=200,
    migration_interval=10,
)
```

## Implementation Checklist

- [x] Hardware auto-detection
- [x] NUMA-aware thread pinning
- [x] GPU-CPU affinity mapping
- [x] Island Model GA with NVLink migration
- [x] SIMD-optimized CPU validation
- [x] Cloud launch script with numactl
- [x] System optimizations (hugepages, governor, etc.)

## Future Enhancements

1. **Multi-Fidelity Screening**: GPU fast-screen → CPU thorough validation
2. **Lock-Free Data Structures**: DashMap/SegQueue for contention reduction
3. **f16 Compression**: Reduce memory bandwidth for feature cubes
4. **Async I/O**: Overlap data loading with computation

## References

- Hardware: Hyperstack N3-RTX-A6000x8
- CPU: AMD EPYC Milan (252 cores, 2× 128-core sockets)
- GPU: 8× NVIDIA RTX A6000 (48GB VRAM each)
- RAM: 464GB DDR4-3200
- Interconnect: NVLink 112GB/s between GPU pairs
