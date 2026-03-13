# SMT-Aware Thread Configuration for Hyperstack N3

## Hardware Overview

| Component | Specification |
|-----------|---------------|
| Physical Cores | 252 (AMD EPYC Milan) |
| SMT Multiplier | 2× |
| **Logical Threads** | **504** |
| Sockets | 2 (NUMA nodes) |
| Memory | 464GB DDR4-3200 |
| GPUs | 8× RTX A6000 (48GB each) |

## Thread Mapping

### NUMA Topology with SMT

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Socket 0 (NUMA 0)                           │
│  Physical Cores 0-125  │  SMT Threads 252-377  │  GPUs 0-3          │
│  ├─ GPU 0: threads 0-31, 252-283                                   │
│  ├─ GPU 1: threads 32-63, 284-315                                  │
│  ├─ GPU 2: threads 64-95, 316-347                                  │
│  └─ GPU 3: threads 96-125, 348-377                                 │
│                         Total: 252 logical threads                  │
├─────────────────────────────────────────────────────────────────────┤
│                         Socket 1 (NUMA 1)                           │
│  Physical Cores 126-251 │ SMT Threads 378-503 │  GPUs 4-7          │
│  ├─ GPU 4: threads 126-157, 378-409                                │
│  ├─ GPU 5: threads 158-189, 410-441                                │
│  ├─ GPU 6: threads 190-221, 442-473                                │
│  └─ GPU 7: threads 222-251, 474-503                                │
│                         Total: 252 logical threads                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Thread Usage Strategy

### Primary Threads (0-251): Compute-Intensive
- **Rayon thread pool**: Set `RAYON_NUM_THREADS=240`
- **Rust evaluation**: Primary computation
- **Model training**: Tree models (LightGBM/XGBoost)
- **CPU validation**: Primary backtest loops

### SMT Threads (252-503): I/O & Coordination
- **Async I/O**: Data loading, network
- **GPU coordination**: CUDA streams, copy engines
- **Python asyncio**: Event loop workers
- **Monitoring**: Metrics, logging

## Configuration by Workload

### 1. GPU Discovery (8 GPUs Active)

```bash
# Per GPU (using NUMA-local threads)
RAYON_NUM_THREADS=60          # 60 primary threads per GPU
FOREX_BOT_CPU_THREADS=120     # 60 primary + 60 SMT for I/O

# Total across system
# 8 GPUs × 60 threads = 480 threads used
# Reserve 24 threads for OS/coordination
```

### 2. CPU Validation (All Threads)

```bash
# Use all available threads
RAYON_NUM_THREADS=240         # All primary threads
FOREX_BOT_CPU_THREADS=480     # All threads (primary + SMT)

# 24 threads (12 physical) reserved for:
# - GPU persistence
# - Async I/O coordination
# - Python main thread
```

### 3. Mixed Workload (Training + Discovery)

```bash
# GPU discovery takes priority
RAYON_NUM_THREADS=240         # Primary threads
FOREX_BOT_CPU_THREADS=360     # 240 primary + 120 SMT

# 144 threads reserved for:
# - Model training (async)
# - Data loading
# - Coordination
```

## Environment Variables

```bash
# Core thread settings
export RAYON_NUM_THREADS=240           # Rust compute (primary only)
export OMP_NUM_THREADS=1               # BLAS single-threaded (prevents oversub)

# Forex bot settings
export FOREX_BOT_CPU_THREADS=480       # Total usable threads
export FOREX_BOT_RUST_THREADS=240      # Rust primary threads
export FOREX_BOT_GPU_WORKERS=8         # 8× A6000

# NUMA/affinity
export FOREX_BOT_NUMA_BIND=1           # Enable NUMA binding
export FOREX_BOT_SMT_STRATEGY=compute  # Primary=compute, SMT=io
```

## Launch Script Examples

### Full 504-Thread Discovery

```bash
#!/bin/bash
# Use numactl for explicit NUMA binding

# Socket 0: 252 threads (0-125, 252-377)
numactl --cpunodebind=0 --membind=0 \
    env RAYON_NUM_THREADS=120 \
        FOREX_BOT_CPU_THREADS=252 \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        ./forex-search --mode discovery &

# Socket 1: 252 threads (126-251, 378-503)
numactl --cpunodebind=1 --membind=1 \
    env RAYON_NUM_THREADS=120 \
        FOREX_BOT_CPU_THREADS=252 \
        CUDA_VISIBLE_DEVICES=4,5,6,7 \
        ./forex-search --mode discovery &

wait
```

### CPU Validation Only

```bash
#!/bin/bash
# Use all 504 threads for CPU validation

export RAYON_NUM_THREADS=240
export FOREX_BOT_CPU_THREADS=480
export FOREX_BOT_VALIDATION_MODE=1

# No NUMA binding needed - uses all sockets
./forex-search --mode validate
```

## Performance Expectations

| Workload | Threads Used | Expected Throughput |
|----------|--------------|---------------------|
| GPU Discovery (8×) | 480 | 4.8M strategies/sec |
| CPU Validation | 480 | 100K strategies/sec |
| Feature Compute | 504 | 10× faster vs 252 threads |
| Model Training | 240 | Same (memory-bound) |

## Monitoring Thread Usage

```bash
# Check thread distribution across cores
watch -n 1 'ps -eo pid,comm,psr | grep forex | sort -k3 -n'

# Check for thread migration (bad for NUMA)
perf stat -e migrations -p $(pgrep forex-search) -a sleep 10

# Monitor per-core utilization
mpstat -P ALL 1
```

## Common Pitfalls

### 1. Oversubscription
```bash
# BAD: 504 Rayon threads + Python workers
RAYON_NUM_THREADS=504
FOREX_BOT_CPU_THREADS=504

# GOOD: Reserve threads for coordination
RAYON_NUM_THREADS=240
FOREX_BOT_CPU_THREADS=480
```

### 2. Cross-NUMA Access
```bash
# BAD: Socket 0 threads accessing Socket 1 memory
# Results in 4× slower memory bandwidth

# GOOD: Use numactl --cpunodebind --membind
```

### 3. SMT for Compute-Only
```bash
# BAD: Using SMT threads for heavy compute
# SMT shares execution units, not 2× performance

# GOOD: Primary threads for compute, SMT for I/O
```

## Verification

Run this on your Hyperstack instance to verify topology:

```bash
#!/bin/bash
echo "=== Thread Topology ==="
echo "Logical threads: $(nproc)"
echo "Physical cores: $(($(nproc) / 2))"

echo ""
echo "=== NUMA Layout ==="
numactl --hardware | grep -E "(available|node [0-9] cpus)"

echo ""
echo "=== CPU Info ==="
lscpu | grep -E "(Socket|NUMA|Thread|Core)"

echo ""
echo "=== GPU Topology ==="
nvidia-smi topo -m | head -10
```

Expected output:
```
=== Thread Topology ===
Logical threads: 504
Physical cores: 252

=== NUMA Layout ===
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 ... 125 252 253 ... 377
node 1 cpus: 126 127 ... 251 378 379 ... 503
```
