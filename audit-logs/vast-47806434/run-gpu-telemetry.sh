#!/usr/bin/env bash
set -uo pipefail

log=/workspace/neoethos/audit-logs/compatible/gpu-telemetry.log
mkdir -p "$(dirname "$log")"

while true; do
  {
    echo "TELEMETRY_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader
    cat /proc/loadavg
    df -h /workspace | tail -n 1
  } >> "$log" 2>&1
  sleep 10
done
