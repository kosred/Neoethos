#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/neoethos/dependency-upgrade-probe
log_root=/workspace/neoethos/audit-logs/compatible/vector-ta-parallel-builder-sm89-nvcc-thread1-cgroup-aware
target_root=/workspace/neoethos/targets/compatible-gpu-adaptive/target

mkdir -p "$log_root"
cd "$repo"

export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_TARGET_DIR=$target_root
export CUDA_FAST_MATH=0
export NEOETHOS_REQUIRE_GPU=1

cpu_accounting_mode=
cpu_usage_path=
cpu_stat_path=

detect_cpu_accounting() {
    if [[ -r /sys/fs/cgroup/cpu.stat ]] \
        && grep -q '^usage_usec ' /sys/fs/cgroup/cpu.stat; then
        cpu_accounting_mode=v2
        cpu_stat_path=/sys/fs/cgroup/cpu.stat
    elif [[ -r /sys/fs/cgroup/cpu,cpuacct/cpuacct.usage \
        && -r /sys/fs/cgroup/cpu,cpuacct/cpu.stat ]]; then
        cpu_accounting_mode=v1
        cpu_usage_path=/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage
        cpu_stat_path=/sys/fs/cgroup/cpu,cpuacct/cpu.stat
    else
        printf 'no supported cgroup CPU accounting interface; refusing incomplete utilization evidence\n' >&2
        return 1
    fi
}

read_cpu_stat_value() {
    local key=$1
    awk -v key="$key" '$1 == key {print $2; found=1} END {if (!found) exit 1}' \
        "$cpu_stat_path"
}

read_cpu_usage_usec() {
    if [[ "$cpu_accounting_mode" == v2 ]]; then
        read_cpu_stat_value usage_usec
    else
        local usage_ns
        read -r usage_ns < "$cpu_usage_path"
        printf '%s\n' "$(( usage_ns / 1000 ))"
    fi
}

read_nr_throttled() {
    read_cpu_stat_value nr_throttled
}

read_throttled_usec() {
    if [[ "$cpu_accounting_mode" == v2 ]]; then
        read_cpu_stat_value throttled_usec
    else
        local throttled_ns
        throttled_ns=$(read_cpu_stat_value throttled_time)
        printf '%s\n' "$(( throttled_ns / 1000 ))"
    fi
}

detect_cpu_accounting

host_probe=/tmp/neoethos-build-host-parallel-probe
rustc +nightly-2026-04-07 --edition 2024 -D warnings \
    scripts/build/resolve_host.rs -o "$host_probe"
host_evidence=$($host_probe)
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
printf '%s\n' "$host_evidence" | tee "$log_root/summary.log"
printf 'cpu_accounting_mode=%s cpu_stat_path=%s\n' \
    "$cpu_accounting_mode" "$cpu_stat_path" | tee -a "$log_root/summary.log"

run_probe() {
    local label=$1
    shift
    local log="$log_root/$label.log"
    printf 'PROBE_START label=%s utc=%s\n' "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        | tee -a "$log_root/summary.log" "$log"
    set +e
    "$@" 2>&1 | tee -a "$log"
    local status=${PIPESTATUS[0]}
    set -e
    printf 'PROBE_EXIT label=%s status=%s utc=%s\n' \
        "$label" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        | tee -a "$log_root/summary.log" "$log"
    return "$status"
}

run_expected_failure() {
    local label=$1
    local expected_text=$2
    shift 2
    local log="$log_root/$label.log"
    printf 'PROBE_START label=%s expected_failure=true utc=%s\n' \
        "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        | tee -a "$log_root/summary.log" "$log"
    set +e
    "$@" 2>&1 | tee -a "$log"
    local status=${PIPESTATUS[0]}
    set -e
    local verdict=0
    if (( status == 0 )); then
        printf 'expected command failure but command succeeded\n' | tee -a "$log"
        verdict=1
    elif ! grep -Fq -- "$expected_text" "$log"; then
        printf 'expected failure text not found: %s\n' "$expected_text" | tee -a "$log"
        verdict=1
    fi
    printf 'PROBE_EXIT label=%s command_status=%s verdict=%s utc=%s\n' \
        "$label" "$status" "$verdict" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        | tee -a "$log_root/summary.log" "$log"
    return "$verdict"
}

run_parallel_probe() {
    local label=$1
    shift
    local log="$log_root/$label.log"
    printf 'PROBE_START label=%s utc=%s\n' "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        | tee -a "$log_root/summary.log" "$log"
    local started_at
    started_at=$(date +%s)
    set +e
    "$@" > >(tee -a "$log") 2>&1 &
    local command_pid=$!
    local peak_nvcc=0
    local peak_ptxas=0
    local peak_running_compiler_threads=0
    local peak_active_cpu_milli=0
    local minimum_mem_available_kib
    minimum_mem_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    local previous_usage_usec previous_sample_ns
    previous_usage_usec=$(read_cpu_usage_usec)
    previous_sample_ns=$(date +%s%N)
    local initial_nr_throttled initial_throttled_usec
    initial_nr_throttled=$(read_nr_throttled)
    initial_throttled_usec=$(read_throttled_usec)
    while kill -0 "$command_pid" 2>/dev/null; do
        local active_nvcc active_ptxas running_compiler_threads current_mem_available_kib
        local current_usage_usec current_sample_ns elapsed_sample_ns used_sample_usec active_cpu_milli
        active_nvcc=$(pgrep -x nvcc 2>/dev/null | wc -l)
        active_ptxas=$(pgrep -x ptxas 2>/dev/null | wc -l)
        running_compiler_threads=$(
            ps -eLo state=,comm= \
                | awk '$1 == "R" && $2 ~ /^(nvcc|cicc|ptxas|cc1|cc1plus|fatbinary|cudafe[+][+]|rustc|rust-lld|ld[.]lld|cc|c[+][+]|gcc|g[+][+]|clang|clang[+][+])$/ {count++} END {print count + 0}'
        )
        current_mem_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        current_usage_usec=$(read_cpu_usage_usec)
        current_sample_ns=$(date +%s%N)
        elapsed_sample_ns=$(( current_sample_ns - previous_sample_ns ))
        used_sample_usec=$(( current_usage_usec - previous_usage_usec ))
        active_cpu_milli=$(( used_sample_usec * 1000000 / elapsed_sample_ns ))
        (( active_nvcc > peak_nvcc )) && peak_nvcc=$active_nvcc
        (( active_ptxas > peak_ptxas )) && peak_ptxas=$active_ptxas
        (( running_compiler_threads > peak_running_compiler_threads )) \
            && peak_running_compiler_threads=$running_compiler_threads
        (( active_cpu_milli > peak_active_cpu_milli )) \
            && peak_active_cpu_milli=$active_cpu_milli
        (( current_mem_available_kib < minimum_mem_available_kib )) \
            && minimum_mem_available_kib=$current_mem_available_kib
        previous_usage_usec=$current_usage_usec
        previous_sample_ns=$current_sample_ns
        sleep 0.1
    done
    wait "$command_pid"
    local status=$?
    set -e
    local elapsed_seconds=$(( $(date +%s) - started_at ))
    local final_nr_throttled final_throttled_usec
    final_nr_throttled=$(read_nr_throttled)
    final_throttled_usec=$(read_throttled_usec)
    printf 'PROBE_RESOURCES label=%s peak_nvcc=%s peak_ptxas=%s peak_running_compiler_threads=%s peak_active_cpu_milli=%s worker_limit=%s elapsed_seconds=%s minimum_mem_available_kib=%s throttled_periods_delta=%s throttled_usec_delta=%s\n' \
        "$label" "$peak_nvcc" "$peak_ptxas" "$peak_running_compiler_threads" \
        "$peak_active_cpu_milli" "$worker_limit" "$elapsed_seconds" \
        "$minimum_mem_available_kib" \
        "$(( final_nr_throttled - initial_nr_throttled ))" \
        "$(( final_throttled_usec - initial_throttled_usec ))" \
        | tee -a "$log_root/summary.log" "$log"
    printf 'PROBE_EXIT label=%s status=%s utc=%s\n' \
        "$label" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        | tee -a "$log_root/summary.log" "$log"
    return "$status"
}

run_probe source-contracts \
    python3 scripts/gpu-bench/test_cuda_model_arch_contract.py

run_probe cuda-arch-parser \
    bash -lc 'rustc +nightly-2026-04-07 --edition 2021 --test vendor/cuda_build_arch.rs -A dead_code -o /tmp/neoethos-cuda-build-arch-parallel-tests && /tmp/neoethos-cuda-build-arch-parallel-tests'

cargo clean -p vector-ta
run_expected_failure hostile-nvcc-args \
    'external NVCC_ARGS is unsupported for vector-ta CUDA builds' \
    env NVCC_ARGS=--use_fast_math CUDA_FILTER=neoethos_f64_kernels \
    cargo test -p neoethos-data --features gpu-cuda --lib --no-run \
    -j "$CARGO_BUILD_JOBS"

cargo clean -p vector-ta
unset NVCC_ARGS CUDA_FILTER

run_parallel_probe vector-ta-parallel-build-and-tsi \
    cargo test -p neoethos-data --features gpu-cuda -j "$CARGO_BUILD_JOBS" \
    gpu_tsi_resumes_after_initial_nonfinite_bar_for_every_swept_period \
    -- --nocapture --test-threads=1

run_parallel_probe vector-ta-parallel-artifact-parity \
    cargo test -p neoethos-data --features gpu-cuda -j "$CARGO_BUILD_JOBS" \
    gpu_cpu_indicator_sweep_parity -- --nocapture --test-threads=1

run_parallel_probe cli-gpu-nvidia-parallel-artifact-compile \
    cargo test -p neoethos-cli --features gpu-nvidia --all-targets --no-run \
    -j "$CARGO_BUILD_JOBS"

run_parallel_probe app-gpu-nvidia-parallel-artifact-compile \
    cargo test -p neoethos-app --features gpu-nvidia --all-targets --no-run \
    -j "$CARGO_BUILD_JOBS"

printf 'PARALLEL_BUILDER_STAGE_DONE utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "$log_root/summary.log"
