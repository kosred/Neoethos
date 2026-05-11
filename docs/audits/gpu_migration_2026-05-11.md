# GPU code migration plan — 2026-05-11

The GPU kernel modules in `forex-search` were written against older
versions of `cubecl` (≤ 0.7) and `tch` (≤ 0.18) and have not been
updated since the dependencies were bumped to **cubecl 0.9.0** and
**tch 0.22.0**. The first end-to-end GPU build attempt was made on
2026-05-11 against an L40 GPU on Hyperstack with libtorch
2.9.0+cu130 and the NVIDIA 580 driver — `cargo check --features gpu`
reports **126 compile errors** across 5 files.

To unblock the rest of the GPU dep chain (libtorch + CUDA 13 +
cubecl) the broken modules are now gated behind a separate
`gpu-experimental` feature instead of `gpu`. The CPU-fallback
`discovery_gpu` shim under `cfg(not(feature = "gpu-experimental"))`
takes their place — `--features gpu` still compiles cleanly and
ships libtorch in the release artifact, just without the
non-functional GPU kernels.

## Error budget by file

| File | Error count | Lines | Dominant categories |
|------|-------------|-------|---------------------|
| `cubecl_eval.rs` | 72 | 1007 | `Return` not supported in cubecl 0.9 kernels (use `terminate!()`); `usize`/`u32` index strictness; `F: CubeElement` trait bound missing; `cubecl::CubeElement` not implemented for generic |
| `hpc_gpu_discovery.rs` | 22 | 886 | tch `Scalar: From<f32>` removed; `IntListOption` trait change; method arity changes |
| `discovery_gpu.rs` | 21 | 1015 | tch tensor->Scalar/Vec conversions; mismatched types from `Tensor` deref |
| `cubecl_ga.rs` | 8 | 323 | `ExpandElementTyped<usize>` no longer From `<u32>` |
| `hpc.rs` | 3 | 331 | small int-cast strictness |
| **Total** | **126** | **3,562** | |

## Error patterns by error code

| Code | Count | Pattern | Fix |
|------|-------|---------|-----|
| E0308 | 51 | `mismatched types` | mostly `usize` ↔ `u32` casts on kernel indices |
| E0277 | 14 | `IntListOption` trait | `&[N as i64]` slice form, or wrap in `[N]` |
| E0277 | 12 | `usize` arithmetic with `u32` | explicit `as usize` / `as u32` casts |
| E0277 | 8 | `Scalar: From<f32>` | use `Scalar::float(x as f64)` or `Scalar::from(x as f64)` |
| E0277 | 6 | `Scalar`/`Vec`: `From<&Tensor>` | use `Vec::<f32>::try_from(t.shallow_clone())` |
| custom | 3 | `Return not supported yet` | replace `return X;` with `terminate!()` (cubecl 0.9 kernel constraint) |
| E0277 | 3 | `ExpandElementTyped<usize>: From<…<u32>>` | explicit cast inside cubecl DSL |
| E0277 | 2 | `CubeElement` trait bound on generic | add `where F: cubecl::CubeElement + Float` |
| Other | ~27 | misc method arity / pattern matching | per-call inspection |

## Migration sequencing

1. **`cubecl_eval.rs` first** (72 errors, 57% of total). Pure cubecl
   DSL — no tch dependency. Fixing the index types and replacing
   `return` with `terminate!()` should clear most of it. Verify
   numerical equivalence against the CPU `simulate_trades_core`
   path on a fixed seed.
2. **`cubecl_ga.rs`** (8 errors). Same flavor as cubecl_eval; small
   blast radius.
3. **`hpc_gpu_discovery.rs`** (22 errors). Touches tch tensors;
   needs the `Scalar` / `IntListOption` migration. Run side-by-side
   against the CPU island-model output and require fitness-rank
   parity within numeric tolerance.
4. **`discovery_gpu.rs`** (21 errors). Bridges genome export to
   tch; reuse the helpers introduced in step 3.
5. **`hpc.rs`** (3 errors). Trivial casts; do last when everything
   else compiles so we stop fighting cascades.

## Acceptance criteria for closing the migration

- `cargo build --features gpu-experimental --release -p forex-cli`
  succeeds on a CUDA 13.0+ runner.
- `forex-search/parity` cross-check between CPU and GPU evaluators
  passes within `1e-3` relative tolerance on EURUSD H4 fixture.
- `cargo bench` (or a hand-rolled benchmark) shows ≥ 5× speedup vs
  CPU on a population-1000 / generations-50 discovery cycle.
- The `gpu-experimental` feature is renamed back to plain `gpu` and
  the CPU-fallback `discovery_gpu` shim is removed.
- `release.yml` is updated to build with `gpu-experimental`.

## Why we did not fix this in the same session as the audit

The 126 errors are mostly mechanical (type casts + macro renames)
but the kernel return-flow rewrite is a logic-changing edit on a
metered GPU VM. The cost of an undetected runtime bug in the GPU
fitness function is direct: bad strategies get promoted, real
money gets lost. A focused migration sprint with proper parity
testing is the right shape for that work — not a hurried fix at
the tail end of a release cycle.

The CPU paths (`forex-search::discovery`, `eval`,
`finalize_candidates_with_progress`) are validated and produce
correct fitness values; the release ships GPU-ready dependencies
but uses the verified CPU evaluator until the migration lands.
