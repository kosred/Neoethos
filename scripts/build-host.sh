#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    printf 'usage: %s <cargo-subcommand> [cargo-arguments...]\n' "$0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/.." && pwd)
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/neoethos-build-host.XXXXXX")
probe="$probe_dir/resolve-host"

cleanup() {
    rm -f -- "$probe"
    rmdir -- "$probe_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$repo"
rustc --edition 2024 -D warnings scripts/build/resolve_host.rs -o "$probe"
host_evidence=$($probe)
available_threads=$(sed -n 's/^available_parallelism=//p' <<<"$host_evidence")
worker_limit=$(sed -n 's/^automatic_worker_limit=//p' <<<"$host_evidence")
cuda_architectures=$(sed -n 's/^cuda_architectures=//p' <<<"$host_evidence")
accelerator_mode=$(sed -n 's/^accelerator_mode=//p' <<<"$host_evidence")

if [[ ! "$available_threads" =~ ^[1-9][0-9]*$ \
    || ! "$worker_limit" =~ ^[1-9][0-9]*$ ]]; then
    printf 'invalid build-host plan: available=%q workers=%q mode=%q cuda_architectures=%q\n' \
        "$available_threads" "$worker_limit" "$accelerator_mode" "$cuda_architectures" >&2
    exit 2
fi

export CARGO_BUILD_JOBS=$worker_limit
case "$accelerator_mode" in
    cpu_only)
        [[ "$cuda_architectures" == none ]] || exit 2
        unset NEOETHOS_CUDA_ARCHS
        ;;
    nvidia)
        [[ "$cuda_architectures" =~ ^[1-9][0-9]*(\;[1-9][0-9]*)*$ ]] || exit 2
        export NEOETHOS_CUDA_ARCHS=$cuda_architectures
        ;;
    *)
        printf 'invalid accelerator mode in build-host plan: %q\n' "$accelerator_mode" >&2
        exit 2
        ;;
esac
printf '%s\n' "$host_evidence"
cargo "$@"
