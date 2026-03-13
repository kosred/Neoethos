#!/bin/bash
# Hyperstack N3-RTX-A6000x8 Launch Script
# 8× RTX A6000 + 252 EPYC Milan Cores + 464GB RAM
#
# Usage:
#   ./run_hyperstack_n3.sh [command]
# 
# Commands:
#   discovery   - Run massive GPU-accelerated strategy discovery
#   validate    - Validate strategies with CPU
#   full        - Run full 20-hour pipeline

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Hyperstack N3-RTX-A6000x8 HPC Launcher${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

# Verify hardware
verify_hardware() {
    echo -e "\n${BLUE}Verifying hardware configuration...${NC}"
    
    # Check GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
    if [ "$GPU_COUNT" -lt 8 ]; then
        echo -e "${RED}ERROR: Expected 8 GPUs, found $GPU_COUNT${NC}"
        exit 1
    fi
    
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d '[:space:]')
    echo -e "${GREEN}✓ GPUs: $GPU_COUNT× $GPU_NAME${NC}"
    
    # Check CPU threads (252 physical cores × 2 SMT = 504)
    CPU_THREADS=$(nproc)
    if [ "$CPU_THREADS" -lt 500 ]; then
        echo -e "${YELLOW}WARNING: Expected ~504 logical threads (252 cores + SMT), found $CPU_THREADS${NC}"
    else
        CPU_PHYSICAL=$((CPU_THREADS / 2))
        echo -e "${GREEN}✓ CPU: $CPU_PHYSICAL physical cores × 2 = $CPU_THREADS logical threads${NC}"
    fi
    
    # Check RAM
    TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM_GB" -lt 450 ]; then
        echo -e "${YELLOW}WARNING: Expected ~464GB RAM, found ${TOTAL_RAM_GB}GB${NC}"
    else
        echo -e "${GREEN}✓ RAM: ${TOTAL_RAM_GB}GB${NC}"
    fi
    
    # Show NUMA topology
    if command -v numactl &> /dev/null; then
        echo -e "\n${BLUE}NUMA Topology:${NC}"
        numactl --hardware | grep -E "(available|node [0-9])"
    fi
    
    # Show GPU topology
    echo -e "\n${BLUE}GPU Interconnect Topology:${NC}"
    nvidia-smi topo -m | head -10
}

# System optimizations for HPC
optimize_system() {
    echo -e "\n${BLUE}Applying system optimizations...${NC}"
    
    # Set CPU governor to performance
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
        echo -e "${GREEN}✓ CPU governor set to performance${NC}"
    fi
    
    # Configure hugepages
    echo 1024 | sudo tee /proc/sys/vm/nr_hugepages > /dev/null 2>&1 || true
    echo -e "${GREEN}✓ Hugepages configured${NC}"
    
    # Disable NUMA balancing for predictable memory allocation
    echo 0 | sudo tee /proc/sys/kernel/numa_balancing > /dev/null 2>&1 || true
    echo -e "${GREEN}✓ NUMA balancing disabled${NC}"
    
    # Set memory swappiness
    echo 10 | sudo tee /proc/sys/vm/swappiness > /dev/null 2>&1 || true
    echo -e "${GREEN}✓ Swappiness reduced${NC}"
    
    # NVIDIA persistence mode
    sudo nvidia-smi -pm 1 > /dev/null 2>&1 || true
    echo -e "${GREEN}✓ GPU persistence mode enabled${NC}"
}

# Environment setup
setup_environment() {
    echo -e "\n${BLUE}Setting up environment...${NC}"
    
    # CUDA optimizations
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    
    # NCCL optimizations for NVLink
    export NCCL_P2P_LEVEL=5
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=lo
    
    # PyTorch/Tensor thread settings (single-threaded for multiprocessing)
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    
    # Rust Rayon thread pool - use primary threads only for compute
    # 252 physical cores - 12 reserved for GPU coord = 240
    export RAYON_NUM_THREADS=240
    
    # Forex bot specific
    export FOREX_BOT_HPC_MODE=1
    export FOREX_BOT_GPU_WORKERS=8
    export FOREX_BOT_CPU_THREADS=480  # 504 - 24 reserved
    export FOREX_BOT_RUST_THREADS=240  # Primary threads only
    export FOREX_BOT_RUST_ACCEL=1
    export FOREX_BOT_RUST_EVO=1
    export FOREX_BOT_RUST_FEATURES=auto
    export FOREX_BOT_RUST_FEATURES_ONLY=1
    
    echo -e "${GREEN}✓ Environment configured${NC}"
}

# Run discovery on NUMA Socket 0 (GPUs 0-3, Primary threads 0-125, SMT 252-377)
run_discovery_socket0() {
    echo -e "\n${BLUE}Starting discovery on Socket 0 (GPUs 0-3, 252 threads)...${NC}"
    
    numactl --cpunodebind=0 --membind=0 \
        env CUDA_VISIBLE_DEVICES=0,1,2,3 \
        FOREX_BOT_CPU_THREADS=240 \
        RAYON_NUM_THREADS=120 \
        python forex-ai.py discover \
        --population 250000 \
        --generations 200 \
        --island-id 0 \
        "$@" &
    
    SOCKET0_PID=$!
    echo -e "${GREEN}✓ Socket 0 discovery started (PID: $SOCKET0_PID)${NC}"
}

# Run discovery on NUMA Socket 1 (GPUs 4-7, Primary threads 126-251, SMT 378-503)
run_discovery_socket1() {
    echo -e "\n${BLUE}Starting discovery on Socket 1 (GPUs 4-7, 252 threads)...${NC}"
    
    numactl --cpunodebind=1 --membind=1 \
        env CUDA_VISIBLE_DEVICES=4,5,6,7 \
        FOREX_BOT_CPU_THREADS=240 \
        RAYON_NUM_THREADS=120 \
        python forex-ai.py discover \
        --population 250000 \
        --generations 200 \
        --island-id 1 \
        "$@" &
    
    SOCKET1_PID=$!
    echo -e "${GREEN}✓ Socket 1 discovery started (PID: $SOCKET1_PID)${NC}"
}

# Run unified discovery with all GPUs
run_discovery_unified() {
    echo -e "\n${BLUE}Starting unified discovery on all 8 GPUs...${NC}"
    
    export FOREX_BOT_ISLAND_MODEL=1
    export FOREX_BOT_ISLAND_MIGRATION_INTERVAL=10
    export FOREX_BOT_POPULATION=500000
    
    python forex-ai.py discover \
        --population 500000 \
        --generations 200 \
        --hpc-mode \
        "$@"
}

# Run CPU validation
run_validation() {
    echo -e "\n${BLUE}Starting CPU validation (using 480 threads)...${NC}"
    
    # Use all primary threads + SMT for validation
    # Reserve 24 threads (12 physical) for GPU coordination
    export RAYON_NUM_THREADS=240
    export FOREX_BOT_CPU_THREADS=480
    export FOREX_BOT_VALIDATION_MODE=1
    
    python forex-ai.py validate \
        --workers 480 \
        "$@"
}

# Run full 20-hour pipeline
run_full_pipeline() {
    echo -e "\n${YELLOW}Starting full 20-hour HPC pipeline...${NC}"
    
    START_TIME=$(date +%s)
    
    # Phase 1: Massive GPU discovery (0-18 hours)
    echo -e "\n${BLUE}Phase 1: GPU-accelerated discovery (18 hours)${NC}"
    run_discovery_unified --time-limit 64800  # 18 hours in seconds
    
    # Phase 2: CPU validation (18-19 hours)
    echo -e "\n${BLUE}Phase 2: CPU validation (1 hour)${NC}"
    run_validation --time-limit 3600
    
    # Phase 3: Ensemble construction (19-20 hours)
    echo -e "\n${BLUE}Phase 3: Ensemble construction (1 hour)${NC}"
    python forex-ai.py ensemble \
        --time-limit 3600
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Pipeline completed in $(date -u -d @${ELAPSED} +%H:%M:%S)${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
}

# Main execution
main() {
    COMMAND="${1:-full}"
    shift || true
    
    # Verify and optimize
    verify_hardware
    optimize_system
    setup_environment
    
    case "$COMMAND" in
        discovery)
            run_discovery_unified "$@"
            ;;
        validate)
            run_validation "$@"
            ;;
        dual)
            run_discovery_socket0 "$@"
            run_discovery_socket1 "$@"
            wait
            ;;
        full)
            run_full_pipeline
            ;;
        check)
            echo -e "\n${GREEN}Hardware check complete. System ready for HPC mode.${NC}"
            ;;
        *)
            echo "Usage: $0 [discovery|validate|dual|full|check]"
            echo ""
            echo "Commands:"
            echo "  discovery  - Run GPU discovery on all 8 GPUs"
            echo "  validate   - Run CPU validation"
            echo "  dual       - Run dual-socket discovery (2 processes)"
            echo "  full       - Run full 20-hour pipeline"
            echo "  check      - Verify hardware only"
            exit 1
            ;;
    esac
}

main "$@"
