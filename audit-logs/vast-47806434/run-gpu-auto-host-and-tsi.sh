#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/neoethos/dependency-upgrade-probe
log_root=/workspace/neoethos/audit-logs/compatible/gpu-unified-arch-and-tsi-adaptive-regression
mkdir -p "$log_root"
cd "$repo"

export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_TARGET_DIR=/workspace/neoethos/targets/compatible-gpu-adaptive/target
export CUDA_FAST_MATH=0
export NEOETHOS_REQUIRE_GPU=1

host_probe=/tmp/neoethos-build-host-probe
rustc +nightly-2026-04-07 --edition 2024 \
    -D warnings \
    scripts/build/resolve_host.rs \
    -o "$host_probe"
host_evidence=$("$host_probe")
available_threads=$(sed -n 's/^available_parallelism=//p' <<<"$host_evidence")
worker_limit=$(sed -n 's/^automatic_worker_limit=//p' <<<"$host_evidence")
cuda_architectures=$(sed -n 's/^cuda_architectures=//p' <<<"$host_evidence")
accelerator_mode=$(sed -n 's/^accelerator_mode=//p' <<<"$host_evidence")
if [[ ! "$available_threads" =~ ^[1-9][0-9]*$ \
    || ! "$worker_limit" =~ ^[1-9][0-9]*$ \
    || "$accelerator_mode" != nvidia \
    || ! "$cuda_architectures" =~ ^[1-9][0-9]*(\;[1-9][0-9]*)*$ ]]; then
    printf 'invalid host probe: available=%q workers=%q mode=%q cuda_architectures=%q\n' \
        "$available_threads" "$worker_limit" "$accelerator_mode" "$cuda_architectures" >&2
    exit 2
fi
export CARGO_BUILD_JOBS=$worker_limit
export NEOETHOS_CUDA_ARCHS=$cuda_architectures
printf '%s\n' "$host_evidence" | tee -a "$log_root/summary.log"

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

run_probe source-contracts \
    python3 scripts/gpu-bench/test_cuda_model_arch_contract.py

run_probe cuda-arch-parser \
    bash -lc 'rustc +nightly-2026-04-07 --edition 2021 --test vendor/cuda_build_arch.rs -A dead_code -o /tmp/neoethos-cuda-build-arch-tests && /tmp/neoethos-cuda-build-arch-tests'

run_probe_count_nvcc vector-ta-tsi-initial-nonfinite-real-parity \
    cargo test -p neoethos-data --features gpu-cuda -j "$CARGO_BUILD_JOBS" \
    gpu_tsi_resumes_after_initial_nonfinite_bar_for_every_swept_period -- --nocapture --test-threads=1

run_probe_count_nvcc vector-ta-tsi-real-parity \
    cargo test -p neoethos-data --features gpu-cuda -j "$CARGO_BUILD_JOBS" \
    gpu_cpu_indicator_sweep_parity -- --nocapture --test-threads=1

run_probe_count_nvcc vector-ta-tsi-identical-repeat \
    cargo test -vv -p neoethos-data --features gpu-cuda -j "$CARGO_BUILD_JOBS" \
    gpu_cpu_indicator_sweep_parity -- --nocapture --test-threads=1

run_probe_count_nvcc cli-gpu-nvidia-host-compile \
    cargo test -p neoethos-cli --features gpu-nvidia --all-targets --no-run \
    -j "$CARGO_BUILD_JOBS"

run_probe_count_nvcc app-gpu-nvidia-host-compile \
    cargo test -p neoethos-app --features gpu-nvidia --all-targets --no-run \
    -j "$CARGO_BUILD_JOBS"

printf 'HOST_AUTO_STAGE_DONE utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_root/summary.log"
