#!/bin/bash
# Hyperstack N3-RTX-A6000x8 Launch Script (100% Pure Rust Edition)
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
echo -e "${GREEN}  Hyperstack N3-RTX-A6000x8 HPC Launcher (RUST)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

# CLI Binary Path
CLI_BIN="./target/release/forex-cli"

# Verify hardware
verify_hardware() {
    echo -e "\n${BLUE}Verifying hardware configuration...${NC}"
    
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
    echo -e "${GREEN}✓ GPUs: $GPU_COUNT detected${NC}"
    
    CPU_THREADS=$(nproc)
    echo -e "${GREEN}✓ CPU: $CPU_THREADS logical threads${NC}"
}

# Environment setup
setup_environment() {
    echo -e "\n${BLUE}Setting up Rust HPC environment...${NC}"
    
    export RAYON_NUM_THREADS=$(nproc)
    export FOREX_BOT_HPC_MODE=1
    
    # Build release binary if not present
    if [ ! -f "$CLI_BIN" ]; then
        echo -e "${YELLOW}Building release binary...${NC}"
        cargo build --release -p forex-cli
    fi
}

run_discovery() {
    echo -e "\n${BLUE}Starting Pure-Rust Discovery...${NC}"
    $CLI_BIN discover \
        --symbol EURUSD \
        --base M1 \
        --population 500000 \
        --generations 200 \
        "$@"
}

run_batch_discovery() {
    echo -e "\n${BLUE}Starting Batch Discovery...${NC}"
    $CLI_BIN batch-discover \
        --symbols EURUSD,GBPUSD,AUDUSD \
        --timeframes M1,M5,M15,H1 \
        "$@"
}

main() {
    COMMAND="${1:-discovery}"
    shift || true
    
    verify_hardware
    setup_environment
    
    case "$COMMAND" in
        discovery)
            run_discovery "$@"
            ;;
        batch)
            run_batch_discovery "$@"
            ;;
        *)
            echo "Usage: $0 [discovery|batch]"
            exit 1
            ;;
    esac
}

main "$@"
