#!/bin/bash
# Search + training run for the A6000 VPS (2026-06-10).
# Discovery (fused GPU, parity-proven on all TFs incl. M1) + ML training per
# combo, 5 symbols x 7 timeframes, time-bounded to ~78h, resumable.
source "$HOME/.cargo/env" 2>/dev/null
export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.2/bin:$PATH"
export LIBTORCH="$HOME/libtorch"
export LIBTORCH_BYPASS_VERSION_CHECK=1
export CUDA_PATH=/usr/local/cuda-12.2 CUDA_ROOT=/usr/local/cuda-12.2 CUDA_HOME=/usr/local/cuda-12.2
export LD_PRELOAD="$HOME/libtorch/lib/libtorch_cuda.so $HOME/libtorch/lib/libc10_cuda.so"
export LD_LIBRARY_PATH="$HOME/libtorch/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
# Parity-proven fused signal->backtest reduction (eliminates the signal-matrix
# readback) on every TF + the OOS validation that routes through the same eval.
export NEOETHOS_GPU_FUSED_EVAL=1

BIN="$HOME/Neoethos-src/target/release/neoethos-cli"
[ -x "$BIN" ] || { echo "NO BINARY at $BIN"; exit 1; }

# Risky/prop-firm discovery config (same knobs as run-risky-discover.sh).
sed -e 's/prop_search_max_hours: 2.0/prop_search_max_hours: 0.75/' \
    -e 's/prop_search_population: 200/prop_search_population: 512/' \
    "$HOME/Neoethos/config-risky.yaml" > "$HOME/Neoethos/config-risky-discover.yaml"

cd "$HOME/Neoethos" || exit 1
# Fresh search+training run: clear prior discover-only artifacts + checkpoint so
# every combo is (re)discovered AND trained. Resumable via the checkpoint after.
rm -f cache/risky_stop.flag cache/auto_loop_checkpoint.json
rm -rf cache/auto_loop cache/auto_loop_models
rm -f /tmp/neoethos_feature_store/*.fstore 2>/dev/null

SYMS="AUDUSD,EURGBP,EURJPY,EURUSD,GBPUSD"
TFS="H1,H4,M30,M15,M5,M3,M1"
echo "=== SEARCH+TRAIN START $(date -u) (5 syms x 7 TF, fused, WITH training) ==="
echo "config: $(grep -E 'discovery_mode|prop_search_population|prop_search_max_hours' "$HOME/Neoethos/config-risky-discover.yaml" | tr '\n' ' ')"
# NO --skip-training => auto-loop discovers AND trains each combo.
# 280000s ~= 77.8h hard cap; the stop-flag (cache/risky_stop.flag) stops it
# gracefully between combos for a clean manual halt.
timeout 280000 "$BIN" auto-loop \
  --symbols "$SYMS" --timeframes "$TFS" \
  --config "$HOME/Neoethos/config-risky-discover.yaml" \
  --root "$HOME/Neoethos/data" \
  --stop-flag cache/risky_stop.flag > "$HOME/search-train-run.log" 2>&1
echo "RC=$? at $(date -u)"
rm -f /tmp/neoethos_feature_store/*.fstore 2>/dev/null
echo "=== SEARCH+TRAIN DONE $(date -u) ==="
