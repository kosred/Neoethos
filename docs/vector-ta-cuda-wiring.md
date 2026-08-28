# vector-ta exact native-SASS CUDA lane

This document describes the current Rust-only CUDA contract. Historical build
recipes and alternate artifact loaders are intentionally not compatibility
interfaces.

## Build contract

`neoethos-data/gpu-cuda` enables `vector-ta/cuda-build-native`. The vector-ta
build script:

1. resolves the exact architecture set from `CUDA_ARCHS` or every visible
   NVIDIA device;
2. intersects that set with architectures reported by the installed nvcc;
3. compiles one ELF `.cubin` for every kernel/architecture pair using
   `--cubin` and `-gencode=arch=compute_X,code=sm_X`;
4. uses `cuobjdump` to prove that each artifact contains the requested native
   SASS image and no driver-compiled payload;
5. generates a registry containing every reviewed stem/architecture pair and
   compiler provenance.

Missing tools, an unsupported requested architecture, an incomplete Cartesian
artifact set, a wrong architecture image, or an unexpected payload stops the
build. The generated registry is the only runtime artifact source.

Example for the current RTX 5090 verification lane:

```sh
CUDA_ARCHS=120 CUDA_FAST_MATH=0 \
cargo +nightly-2026-04-07 test \
  --manifest-path vendor/vector-ta-0.2.9-patched/Cargo.toml \
  --lib --features cuda-build-native \
  rust_only_contract::vector_ta_cuda_distribution_is_native_sass_only \
  -- --exact --nocapture
```

Omitting `CUDA_ARCHS` is supported only when the build process can see one or
more NVIDIA devices. A cardless build host must name every exact
deployment architecture explicitly.

`CUDA_FAST_MATH` must remain unset or `0` for NeoEthos parity. The
`neoethos-data` build script refuses any other value when `gpu-cuda` is enabled.

## Runtime contract

`vector_ta::cuda::module_loader` reads the current CUDA device compute
capability and selects the matching `(kernel stem, sm_X)` registry entry. It
loads those bytes through the single central `Module::from_cubin` call.

There is no approximate architecture selection or host execution substitution.
A missing exact pair, invalid device capability, rejected image, missing kernel,
or failed launch returns an error. `GpuOnly` therefore either executes the
requested native kernel or fails closed.

The following exported provenance is authoritative:

- `COMPILED_ARCHS`
- `COMPILED_ARCH_SOURCE`
- `COMPILED_NVCC_VERSION`
- `NATIVE_CUBIN_COUNT`

`neoethos-data::core::indicator_telemetry` republishes the architecture set and
source for run logs. It does not independently probe or infer an architecture.

## Device verification

A successful compilation is not hardware validation. Acceptance on each card
requires all of the following from the same build and process:

- the build log names the requested architecture and complete verified registry;
- `cuobjdump` validation succeeds for every produced artifact;
- the native availability probe loads and launches on the assigned device;
- an indicator module loads through the central registry and launches;
- GPU-required tests fail rather than falling back when the exact artifact is
  absent;
- CPU/GPU f64 parity tests report their complete measured outputs.

Use `CUDA_MODULE_LOAD_DEBUG=1` only when module-selection diagnostics are
needed. Use `NEOETHOS_REQUIRE_GPU=1` for tests whose absence of a usable card
must be an error rather than a skip.

## Rust-only boundary

vector-ta publishes an `rlib` only. Python, WASM, JavaScript, C FFI, unsigned
pattern bitmasks, and their optional dependencies or features are not product
interfaces. The signed pattern path preserves `{-100, -80, 0, 80, 100}` through
Rust dispatch and NeoEthos feature columns.
