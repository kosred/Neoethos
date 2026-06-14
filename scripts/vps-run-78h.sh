#!/bin/bash
# 78h production run on the A6000 VPS (2026-06-10, post GPU-training-fix).
# - BOTH discovery modes (prop_firm + risky) discovered.
# - Training done ONCE (mode-independent) → shared ML ensemble (now ~31/32
#   models, 12 on the GPU via burn-cuda, the rest CPU; CPU+GPU overlap inside
#   one rayon pool per the audit).
# - LANES: discovery (GPU-saturating) || training (mostly CPU + microsecond GPU
#   bursts) run concurrently — training's GPU footprint is tiny so contention is
#   negligible, and the otherwise-idle CPU does the training "for free".
source "$HOME/.cargo/env" 2>/dev/null
export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.2/bin:$PATH"
export LIBTORCH="$HOME/libtorch" LIBTORCH_BYPASS_VERSION_CHECK=1
export CUDA_PATH=/usr/local/cuda-12.2 CUDA_ROOT=/usr/local/cuda-12.2 CUDA_HOME=/usr/local/cuda-12.2
export LD_PRELOAD="$HOME/libtorch/lib/libtorch_cuda.so $HOME/libtorch/lib/libc10_cuda.so"
export LD_LIBRARY_PATH="$HOME/Neoethos-src/target/release/deps:$HOME/libtorch/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda-12.2/lib64:$HOME/neoethos-cuda-bundle/lib:$LD_LIBRARY_PATH"
export NEOETHOS_GPU_FUSED_EVAL=1                      # parity-proven fused reduction (all TFs)
export NEOETHOS_BOT_DATA_ROOT="$HOME/Neoethos/data"    # cmd_train reads this
export NEOETHOS_BOT_CATBOOST_EXECUTABLE="$HOME/catboost"  # catboost CLI binary
# burn-cuda 0.21 has a buggy bf16 path (DTypeMismatch panic in burn-ir). Ampere
# gains ~nothing from bf16 on these tiny models — force fp32 = stable + on-GPU.
export FOREX_BURN_MODEL_SUPPORTS_BF16=0

# burn 0.21 GPU memory-leak workaround (relevant once many burn models share the
# card concurrently) — cap cubecl streams. Harmless if the key is ignored.
export CUBECL_MAX_STREAMS=1

# ── DEEP-SEARCH knobs (2026-06-10) ───────────────────────────────────────────
# The fast pass was shallow: it capped at 50k candidates, early-stopped at 50%
# of the time budget, and the funnel only searched 25% of rows. Relax all three
# so each combo searches BROADLY+DEEPLY within its max_hours throttle.
export NEOETHOS_BOT_PROP_ARCHIVE_CAP=5000000          # high enough that it NEVER fills inside
                                                      # max_hours → the GA runs the full per-combo
                                                      # time budget (more generations) instead of
                                                      # stopping early when the archive caps.
export NEOETHOS_BOT_PROP_CONVERGENCE_MIN_ELAPSED_FRAC=1.0  # never early-stop before max_hours
export NEOETHOS_BOT_PROP_CONVERGENCE_GENS=0            # 0 = disable convergence early-stop (was 250)
export NEOETHOS_BOT_FUNNEL_STAGE1_PCT=1.0             # search ALL in-sample rows, not just earliest 25%
# DO NOT set NEOETHOS_BOT_MIN_HISTORY_YEARS — leave default 0 so short pairs
# (XAUUSD ~2.7y) are never hard-rejected; the split stays percentage-based 70/30.

BIN="$HOME/Neoethos-src/target/release/neoethos-cli"
[ -x "$BIN" ] || { echo "NO BINARY at $BIN"; exit 1; }
cd "$HOME/Neoethos" || exit 1
# 7 robust pairs only (>=2.7y history). The 7 test pairs (~2y, cTrader test
# downloads) were dropped — too thin for trustworthy validation.
SYMS="AUDUSD,EURGBP,EURJPY,EURUSD,GBPUSD,GBPJPY,XAUUSD"
TFS="H1,H4,M30,M15,M5,M3,M1"
SYM_LIST="AUDUSD EURGBP EURJPY EURUSD GBPUSD GBPJPY XAUUSD"
TF_LIST="H1 H4 M30 M15 M5 M3 M1"
STOP="cache/run78_stop.flag"
rm -f "$STOP"; mkdir -p cache

# Mode discovery configs (population 512, per-combo GA cap 0.5h to bound both
# modes inside the window; stop-flag halts gracefully between combos).
mk_cfg() { sed -e "s/discovery_mode: risky/discovery_mode: $1/" \
               -e 's/prop_search_max_hours: 2.0/prop_search_max_hours: 0.5/' \
               -e 's/prop_search_population: 200/prop_search_population: 2000/' \
               -e 's/prop_search_train_years: [0-9.]*/prop_search_train_years: 0/' \
               -e 's/prop_search_val_years: [0-9.]*/prop_search_val_years: 0/' \
               "$HOME/Neoethos/config-risky.yaml" > "$2"; }
mk_cfg prop_firm "$HOME/Neoethos/config-propfirm-disc.yaml"
mk_cfg risky     "$HOME/Neoethos/config-risky-disc.yaml"

echo "=== 78h RUN START $(date -u) ==="

# ── LANE 1: DISCOVERY (GPU) — prop_firm all combos, then risky all combos. ──
(
  for MODE in propfirm risky; do
    rm -f cache/auto_loop_checkpoint.json; rm -rf cache/auto_loop
    echo "[disco] === $MODE $(date -u) ==="
    timeout 280000 "$BIN" auto-loop --skip-training \
      --config "$HOME/Neoethos/config-${MODE}-disc.yaml" \
      --symbols "$SYMS" --timeframes "$TFS" \
      --root "$HOME/Neoethos/data" --stop-flag "$STOP"
    rm -rf "cache/auto_loop_${MODE}"; mv cache/auto_loop "cache/auto_loop_${MODE}" 2>/dev/null
    [ -f "$STOP" ] && { echo "[disco] stop-flag; halting"; break; }
  done
  echo "[disco] DONE $(date -u)"
) > "$HOME/run78-disco.log" 2>&1 &
DISCO_PID=$!; echo "disco lane pid=$DISCO_PID"

# Stagger so discovery builds combo-1 features before training reads them.
sleep 300

# ── LANE 2: TRAINING (CPU + tiny GPU) — every combo once → shared models. ──
(
  for SYM in $SYM_LIST; do for TF in $TF_LIST; do
    [ -f "$STOP" ] && { echo "[train] stop-flag; halting"; break 2; }
    echo "[train] === $SYM $TF $(date -u) ==="
    "$BIN" train --symbol "$SYM" --base "$TF" --models-dir cache/auto_loop_models
  done; done
  echo "[train] DONE $(date -u)"
) > "$HOME/run78-train.log" 2>&1 &
TRAIN_PID=$!; echo "train lane pid=$TRAIN_PID"

wait "$DISCO_PID" "$TRAIN_PID"
echo "=== 78h RUN DONE $(date -u) ==="
