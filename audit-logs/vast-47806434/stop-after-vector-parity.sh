#!/usr/bin/env bash
set -uo pipefail

summary=/workspace/neoethos/audit-logs/compatible/gpu-unified-arch-and-tsi/summary.log
while ! grep -q 'PROBE_EXIT label=vector-ta-tsi-real-parity' "$summary" 2>/dev/null; do
    sleep 1
done

tmux kill-session -t dep-gpu-unified 2>/dev/null || true
