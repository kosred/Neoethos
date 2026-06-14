#!/bin/bash
# Smoke-test ONE training combo on the rebuilt (burn-cuda) binary and prove:
#   (1) the GPU is actually used during training (nvidia-smi util > 0), and
#   (2) the previously-failing models now train (xgboost CPU-hist fallback,
#       dqn CPU fallback, catboost via the downloaded CLI binary).
set -u
cd "$HOME/Neoethos" || exit 1
rm -f "$HOME/smoke_gpu.txt" "$HOME/smoke-train.log"
rm -rf "$HOME/smoke_models"

# GPU sampler: util% + mem every 2s for up to ~20 min.
( for i in $(seq 1 600); do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    sleep 2
  done ) > "$HOME/smoke_gpu.txt" 2>&1 &
SMI=$!

START=$(date +%s)
NEOETHOS_BOT_CATBOOST_EXECUTABLE="$HOME/catboost" \
NEOETHOS_BOT_DATA_ROOT="$HOME/Neoethos/data" \
FOREX_BURN_MODEL_SUPPORTS_BF16=0 \
RUST_LOG=info \
  "$HOME/run-cli.sh" train --symbol AUDUSD --base H1 \
  --models-dir "$HOME/smoke_models" > "$HOME/smoke-train.log" 2>&1
RC=$?
END=$(date +%s)
kill "$SMI" 2>/dev/null

echo "TRAIN_RC=$RC  wall=$((END-START))s" >> "$HOME/smoke-train.log"
echo "===== SMOKE SUMMARY ====="
echo "train RC=$RC  wall=$((END-START))s"
echo "--- GPU util: max + nonzero-sample-count ---"
awk -F',' '{gsub(/ %/,"",$1); if($1+0>max)max=$1; if($1+0>0)nz++} END{print "max_util="max"%  nonzero_samples="nz+0"  total="NR}' "$HOME/smoke_gpu.txt"
echo "--- GPU mem: peak ---"
awk -F',' '{gsub(/ MiB/,"",$2); if($2+0>max)max=$2} END{print "peak_mem="max" MiB"}' "$HOME/smoke_gpu.txt"
echo "--- model outcomes ---"
grep -aoE "(Completed|Failed|trained) [A-Za-z_0-9]+" "$HOME/smoke-train.log" | sort | uniq -c | sort -rn
echo "--- COMPLETED count / FAILED count ---"
echo "completed=$(grep -ac 'Completed ' "$HOME/smoke-train.log")  failed=$(grep -ac 'Failed ' "$HOME/smoke-train.log")"
echo "--- any failures (detail) ---"
grep -aE "Failed [a-z]" "$HOME/smoke-train.log" | sed -E 's/.*Thread[^:]*: //' | head -20
echo "--- burn backend / cuda evidence ---"
grep -aiE "cuda|burn.*backend|execution_backend|ndarray" "$HOME/smoke-train.log" | grep -ivE "warning|ndarray =" | head -8
