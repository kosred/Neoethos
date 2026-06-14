#!/bin/bash
# PARALLEL search + training for the A6000 VPS (2026-06-10).
# KEY INSIGHT: ML training (cmd_train --symbol --base) is INDEPENDENT of discovery
# and runs on the CPU (parallel_trainer), leaving the A6000 idle. Discovery runs
# on the GPU. So run them as TWO PARALLEL LANES — wall-clock ~= max(disco, train)
# instead of sum. Training is mode-independent => trained ONCE, shared by BOTH
# risky and prop_firm strategies. Fits both modes + training in ~78h.
source "$HOME/.cargo/env" 2>/dev/null
export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.2/bin:$PATH"
export LIBTORCH="$HOME/libtorch"
export LIBTORCH_BYPASS_VERSION_CHECK=1
export CUDA_PATH=/usr/local/cuda-12.2 CUDA_ROOT=/usr/local/cuda-12.2 CUDA_HOME=/usr/local/cuda-12.2
export LD_PRELOAD="$HOME/libtorch/lib/libtorch_cuda.so $HOME/libtorch/lib/libc10_cuda.so"
export LD_LIBRARY_PATH="$HOME/libtorch/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
export NEOETHOS_GPU_FUSED_EVAL=1            # parity-proven fused reduction (all TFs)
export NEOETHOS_BOT_DATA_ROOT="$HOME/Neoethos/data"   # cmd_train reads this

BIN="$HOME/Neoethos-src/target/release/neoethos-cli"
[ -x "$BIN" ] || { echo "NO BINARY at $BIN"; exit 1; }
cd "$HOME/Neoethos" || exit 1
SYMS="AUDUSD,EURGBP,EURJPY,EURUSD,GBPUSD"
TFS="H1,H4,M30,M15,M5,M3,M1"
SYM_LIST="AUDUSD EURGBP EURJPY EURUSD GBPUSD"
TF_LIST="H1 H4 M30 M15 M5 M3 M1"
STOP="cache/risky_stop.flag"
rm -f "$STOP"
mkdir -p cache

# Mode configs: GA max_hours 0.75->0.5 (margin for both lanes); population 512.
mk_cfg() { # $1=src-mode-value $2=out
  sed -e "s/discovery_mode: risky/discovery_mode: $1/" \
      -e 's/prop_search_max_hours: 2.0/prop_search_max_hours: 0.5/' \
      -e 's/prop_search_population: 200/prop_search_population: 512/' \
      "$HOME/Neoethos/config-risky.yaml" > "$2"
}
mk_cfg prop_firm "$HOME/Neoethos/config-propfirm-discover.yaml"
mk_cfg risky     "$HOME/Neoethos/config-risky-discover.yaml"

echo "=== PARALLEL SEARCH+TRAIN START $(date -u) ==="

# ── LANE 1: DISCOVERY (GPU) — prop_firm all, then risky all (features cached). ──
(
  for MODE in propfirm risky; do
    rm -f cache/auto_loop_checkpoint.json
    rm -rf cache/auto_loop
    CFG="$HOME/Neoethos/config-${MODE}-discover.yaml"
    echo "[disco] === $MODE discovery $(date -u) ==="
    timeout 280000 "$BIN" auto-loop --skip-training \
      --config "$CFG" --symbols "$SYMS" --timeframes "$TFS" \
      --root "$HOME/Neoethos/data" --stop-flag "$STOP"
    rm -rf "cache/auto_loop_${MODE}"
    mv cache/auto_loop "cache/auto_loop_${MODE}" 2>/dev/null
    [ -f "$STOP" ] && { echo "[disco] stop-flag set; halting"; break; }
  done
  echo "[disco] DONE $(date -u)"
) > "$HOME/disco-lane.log" 2>&1 &
DISCO_PID=$!
echo "disco lane pid=$DISCO_PID"

# Stagger: let discovery build the first combos' features before training reads
# them (avoids a feature-store write race on combo 1).
sleep 600

# ── LANE 2: TRAINING (CPU) — every combo once, mode-independent => models/. ──
(
  for SYM in $SYM_LIST; do
    for TF in $TF_LIST; do
      [ -f "$STOP" ] && { echo "[train] stop-flag set; halting"; break 2; }
      echo "[train] === $SYM $TF $(date -u) ==="
      "$BIN" train --symbol "$SYM" --base "$TF" --models-dir cache/auto_loop_models
    done
  done
  echo "[train] DONE $(date -u)"
) > "$HOME/train-lane.log" 2>&1 &
TRAIN_PID=$!
echo "train lane pid=$TRAIN_PID"

wait "$DISCO_PID" "$TRAIN_PID"
rm -f /tmp/neoethos_feature_store/*.fstore 2>/dev/null
echo "=== PARALLEL SEARCH+TRAIN DONE $(date -u) ==="
