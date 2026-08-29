#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/neoethos/dependency-upgrade-probe
log_root=/workspace/neoethos/audit-logs/compatible/gpu-stage3-attempt2
mkdir -p "$log_root"
cd "$repo"

export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_TARGET_DIR=/workspace/neoethos/targets/compatible-gpu/target
export CARGO_BUILD_JOBS=62
export CUDA_ARCHS=89
export NEOETHOS_CUDA_ARCH='compute_89,code=sm_89'
export CUDA_FAST_MATH=0
export NEOETHOS_REQUIRE_GPU=1

run_probe() {
    local label=$1
    shift
    local log="$log_root/$label.log"
    printf 'PROBE_START label=%s utc=%s\n' "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_root/summary.log" "$log"
    set +e
    "$@" 2>&1 | tee -a "$log"
    local status=${PIPESTATUS[0]}
    set -e
    printf 'PROBE_EXIT label=%s status=%s utc=%s\n' "$label" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_root/summary.log" "$log"
    return 0
}

run_probe_count_nvcc() {
    local label=$1
    shift
    local log="$log_root/$label.log"
    printf 'PROBE_START label=%s utc=%s\n' "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_root/summary.log" "$log"
    set +e
    "$@" > >(tee -a "$log") 2>&1 &
    local command_pid=$!
    local peak_nvcc=0
    while kill -0 "$command_pid" 2>/dev/null; do
        local active_nvcc
        active_nvcc=$(pgrep -x nvcc 2>/dev/null | wc -l)
        if (( active_nvcc > peak_nvcc )); then
            peak_nvcc=$active_nvcc
        fi
        sleep 0.1
    done
    wait "$command_pid"
    local status=$?
    set -e
    printf 'PROBE_NVCC_PEAK label=%s peak=%s\n' "$label" "$peak_nvcc" | tee -a "$log_root/summary.log" "$log"
    printf 'PROBE_EXIT label=%s status=%s utc=%s\n' "$label" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_root/summary.log" "$log"
    return 0
}

run_probe_count_nvcc gpu-schema-regression-build \
    cargo test -p neoethos-data --features gpu-cuda -j 62 \
    warmup_skips_preserve_the_canonical_column_schema_without_launch_rows -- \
    --nocapture --test-threads=1

run_probe_count_nvcc gpu-real-parity-after-schema-fix \
    cargo test -p neoethos-data --features gpu-cuda -j 62 \
    gpu_cpu_indicator_sweep_parity -- --nocapture --test-threads=1

run_probe_count_nvcc gpu-real-parity-identical-repeat-vv \
    cargo test -vv -p neoethos-data --features gpu-cuda -j 62 \
    gpu_cpu_indicator_sweep_parity -- --nocapture --test-threads=1

run_probe data-f64-cpu-reference \
    cargo test -p neoethos-data --features gpu-cuda --test f64_lane_cpu_reference \
    -j 62 -- --nocapture --test-threads=1

run_probe search-cuda-all-targets-compile \
    cargo test -p neoethos-search --features gpu-cuda --all-targets --no-run -j 62

run_probe cli-gpu-nvidia-all-targets-compile \
    cargo test -p neoethos-cli --features gpu-nvidia --all-targets --no-run -j 62

run_probe app-gpu-nvidia-all-targets-compile \
    cargo test -p neoethos-app --features gpu-nvidia --all-targets --no-run -j 62

run_probe cli-gpu-native-aggregate-compile \
    cargo test -p neoethos-cli --features gpu-nvidia,gpu-bench-cuda --all-targets --no-run -j 62

printf 'STAGE3_DONE utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_root/summary.log"
