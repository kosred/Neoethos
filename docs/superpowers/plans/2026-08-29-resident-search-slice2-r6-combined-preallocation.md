# Resident Search Slice 2 R6 Combined Preallocation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land one warning-clean behavioral RED commit that proves the exact combined Search preallocation contract without invoking CUDA hardware.

**Architecture:** The real Rust admission seam receives an exact layout, reserve/workspace authority, calibration binding and an independent allocation-recorder facade. All input validation precedes native create; the otherwise-valid path remains a test-only `ImplementationPending` stub, so precisely five named tests fail at runtime while compiling cleanly.

**Tech Stack:** Rust unit tests, checked `u64` arithmetic, move-only fixture owner, test-only allocator recorder, Cargo offline verification.

---

Authority: `docs/superpowers/specs/2026-08-28-resident-search-slice2-archive-knn-design.md`, version 6. Do not reinterpret the three-entry physical ledger, four reserve/workspace authorities, error/axis names, test names or mutation list in this plan.

## File map and hard scope

- Modify: `crates/neoethos-gpu-cuda/src/resident_search_v2.rs`
  - own only the `#[cfg(test)]` error/axis/request/receipt/allocation-call DTOs,
    allocator-facade trait, move-only owner shape, admission/enqueue RED stub and
    child-module registration;
  - do not alter an existing production item or FFI declaration.
- Create: `crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs`
  - own the sole recorder implementation/private state/read-only accessors,
    valid and mutated fixture builders, mutation register and exactly five
    `#[test]` functions;
- Forbidden: `lib.rs`, every `Cargo.toml`, `Cargo.lock`, `build.rs`, every native/CUDA file, R1-R5 files and every other source/test.
- Hardware/network boundary: no device enumeration, CUDA call, VPS call, registry access or network access occurs in this RED.

## Chunk 1: Exact warning-clean R6 RED

### Task 1: Add the test-only admission vocabulary and pending seam

**Files:**

- Modify: `crates/neoethos-gpu-cuda/src/resident_search_v2.rs`
- Test: `crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs`

- [ ] **Step 1: Add the exact error and axis vocabulary under `#[cfg(test)]`**

Use these exact enums; payloads are part of the assertion surface:

```rust
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AlignedFieldV2 {
    ArchiveGeneScalars,
    ArchiveTermIndices,
    ArchiveTermWeights,
    ArchiveMetricRows,
    ArchiveSignatures,
    ArchiveHashes,
    CurrentPopulationSignatures,
    NoveltyScores,
    ExactTopKKeys,
    AdmissionFlags,
    AdmissionOffsets,
    ArchiveControlAndSeal,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReserveAuthorityAxisV2 {
    AllocatorContextHeadroomIdentity,
    FullWorkspaceAuthorityIdentity,
    RetainedPreSearchWorkspaceIdentity,
    RemainingSearchAllocationAfterTrimIdentity,
    RetainedPlusRemainingEqualsFullWorkspace,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReserveArithmeticV2 {
    WorkspacePartitionAdd,
    RequestedDeviceSumAdd,
    SameContextFreeMinusHeadroom,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AllocationBudgetAxisV2 {
    RemainingSearchAllocationAfterTrim,
    SameContextFreeAfterAllocatorHeadroom,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2CalibrationAxisV2 {
    DeviceUuid,
    PrimaryContext,
    SearchStream,
    ActivePool,
    CudaBuildIdentity,
    KernelSemanticsIdentity,
    Binary64MathIdentity,
    PlanIdentity,
    RunIdentity,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AdmissionErrorV2 {
    ImplementationPending,
    MissingArchiveArena,
    ZeroArchiveArenaBytes,
    AlignedLayoutFieldMismatch {
        field: ResidentSearchSlice2AlignedFieldV2,
        expected_aligned_bytes: u64,
        observed_aligned_bytes: u64,
    },
    ReserveAuthorityMismatch {
        axis: ResidentSearchSlice2ReserveAuthorityAxisV2,
    },
    ReserveArithmeticOverflow {
        operation: ResidentSearchSlice2ReserveArithmeticV2,
    },
    InsufficientAllocationBudget {
        axis: ResidentSearchSlice2AllocationBudgetAxisV2,
        required_bytes: u64,
        available_bytes: u64,
    },
    ForeignCalibration {
        axis: ResidentSearchSlice2CalibrationAxisV2,
    },
}
```

- [ ] **Step 2: Add parent contract DTOs and the sole child recorder**

The request has named fields for all twelve aligned components; do not represent them as two arrays later compared by `zip`. It carries a generation receipt whose `total_device_bytes` is checked from every generation component, and a combined scoring/archive receipt whose `total_device_bytes` is checked from the unchanged scoring components plus all twelve Slice 2 fields exactly once. It also carries archive presence/bytes, the four byte authorities and their four distinct identities, same-context free bytes, and the nine calibration axes. The expected ledger reads the two totals from those exact receipts; it never accepts independently copied total arguments.

Define a test-only `ResidentSearchSlice2AllocationFacadeV2` trait in `resident_search_v2.rs` with only three effects: `begin_native_create`, `cuda_host_alloc` and `cuda_malloc_async`. The latter two receive the actual call metadata below. The admission seam receives `&mut dyn ResidentSearchSlice2AllocationFacadeV2`; it receives no expected vector and there is no generic append/setter method.

Implement that trait in `resident_search_v2_tests.rs` as `ResidentSearchSlice2AllocationRecorderV2`. Keep every counter and `observed` field private to the child test module and expose read-only accessors/snapshot only. Thus the parent admission seam can produce an observed entry only by invoking an actual facade allocation method; it cannot copy the declared ledger into recorder state.

Parent-owned allocator facade in `resident_search_v2.rs`:

```rust
#[cfg(test)]
pub(crate) trait ResidentSearchSlice2AllocationFacadeV2 {
    fn begin_native_create(&mut self);
    fn cuda_host_alloc(&mut self, actual: ResidentSearchSlice2AllocationCallV2);
    fn cuda_malloc_async(&mut self, actual: ResidentSearchSlice2AllocationCallV2);
}
```

Parent-owned allocation-call DTOs in `resident_search_v2.rs`:

```rust
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AllocationSymbolV2 {
    CudaHostAlloc,
    CudaMallocAsync,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AllocationCategoryV2 {
    TerminalHostReceipt,
    GenerationArena,
    ScoringArchiveArena,
    ArchiveOnlyArena,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AllocationCallV2 {
    pub(crate) ordinal: u8,
    pub(crate) symbol: ResidentSearchSlice2AllocationSymbolV2,
    pub(crate) category: ResidentSearchSlice2AllocationCategoryV2,
    pub(crate) requested_bytes: u64,
    pub(crate) aligned_bytes: u64,
    pub(crate) alignment_bytes: u64,
    pub(crate) flags: u32,
    pub(crate) stream_identity: Option<u64>,
    pub(crate) pool_identity: Option<u64>,
}
```

The sole child-owned recorder implementation in
`resident_search_v2_tests.rs`:

```rust
#[cfg(test)]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AllocationRecorderV2 {
    native_create_count: u64,
    physical_allocator_count: u64,
    generation_arena_count: u64,
    scoring_archive_arena_count: u64,
    archive_only_arena_count: u64,
    observed: Vec<ResidentSearchSlice2AllocationCallV2>,
}
```

The valid expected vector is exactly:

```rust
[
    // ordinal, symbol, category, requested, aligned, alignment, flags, stream, pool
    (0, CudaHostAlloc, TerminalHostReceipt, 104, 104, 8, 0x01, None, None),
    (1, CudaMallocAsync, GenerationArena,
        generation_total, generation_total, 256, 0, Some(stream), Some(pool)),
    (2, CudaMallocAsync, ScoringArchiveArena,
        scoring_archive_total, scoring_archive_total, 256, 0,
        Some(stream), Some(pool)),
]
```

There is no `ArchiveOnlyArena` call. "Archive arena" in the frozen error/test names means the logical archive subreceipt inside `ScoringArchiveArena`, never a fourth physical allocation. `0x01` is the sealed `cudaHostAllocPortable` flag. Event creation is absent from this vocabulary.

- [ ] **Step 3: Add the actual test admission stub and move-only returned owner**

The function signature must take the full request and the independent recorder. It returns a move-only owner whose `queue_generation_v2(self, ordinal, recorder)` transition consumes and returns ownership. The RED implementation returns only the typed pending error before mutating the recorder:

```rust
#[cfg(test)]
pub(crate) fn admit_slice2_combined_fixture_v2(
    _request: ResidentSearchSlice2AdmissionRequestV2,
    _allocator: &mut dyn ResidentSearchSlice2AllocationFacadeV2,
) -> Result<ResidentSearchSlice2AdmissionOwnerV2, ResidentSearchSlice2AdmissionErrorV2> {
    Err(ResidentSearchSlice2AdmissionErrorV2::ImplementationPending)
}
```

The later GREEN implementation order is fixed even though this commit remains RED:

1. validate archive presence and nonzero bytes;
2. compare all twelve named aligned fields, then recompute the checked subtotal;
3. validate all four reserve/workspace identities and checked equations;
4. validate every calibration axis;
5. increment native-create once and invoke the recorder facade for the three actual calls in order;
6. return the move-only owner; its three generation queues make no allocation call.

- [ ] **Step 4: Register the child module locally**

Add only this registration to `resident_search_v2.rs`:

```rust
#[cfg(test)]
#[path = "resident_search_v2_tests.rs"]
mod resident_search_v2_tests;
```

Do not register it in `lib.rs` and do not change feature topology.

- [ ] **Step 5: Check formatting on only the two R6 Rust files**

Run:

```powershell
rustfmt --edition 2024 --check crates/neoethos-gpu-cuda/src/resident_search_v2.rs crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs
```

Expected: only the allowed two Rust paths differ.

### Task 2: Add exactly five behavioral tests

**Files:**

- Modify: `crates/neoethos-gpu-cuda/src/resident_search_v2.rs`
- Create: `crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs`

- [ ] **Step 1: Build one exact valid fixture constructor**

Use `P=200`, `A=50_000`, `W=4`, `K=15`, `M=16`, terminal host bytes `104`, alignment `8`, Slice 2 alignment `256`, and the twelve values:

```text
3_600_128, 6_400_000, 6_400_000, 5_200_128,
1_600_000, 400_128, 6_400, 1_792, 96_000,
1_024, 1_792, 256
```

Their checked replacement subtotal is `23_707_648`. Build the generation total from the current generation-receipt components and the scoring/archive total from its unchanged scoring components plus these twelve values; never invent either total separately. Choose nonzero, distinct fixture identities for device UUID, context, stream, pool, build, kernel semantics, binary64 math, plan, run and each reserve/workspace authority. Compute all totals with `checked_add`, never with literals that bypass the contract.

- [ ] **Step 2: Add the missing/zero archive test**

Add exactly:

```rust
#[test]
fn slice2_combined_admission_rejects_missing_or_zero_archive_arena_before_allocation()
```

Run the absent and zero cases through `admit_slice2_combined_fixture_v2`. Assert the exact error for each and, after each call, assert native-create `0`, every allocator counter `0` and empty observed ledger.

- [ ] **Step 3: Add the twelve-field mismatch test**

Add exactly:

```rust
#[test]
fn slice2_combined_admission_rejects_each_aligned_layout_field_mismatch_before_allocation()
```

Create twelve named mutations. Decrement the selected aligned field by one, then recompute every derived subtotal and the scoring/archive arena total from that mutated declaration. This keeps its internal subtotal arithmetic self-consistent so subtotal-only validation would pass, while the named field still differs from the authoritative layout. Assert the exact `AlignedLayoutFieldMismatch` field and byte payload plus the zero-before-native-create audit after every case. Assert the case count is literally `12`.

- [ ] **Step 4: Add reserve/workspace arithmetic and boundary controls**

Add exactly:

```rust
#[test]
fn slice2_combined_admission_rejects_insufficient_reserve_before_allocation()
```

Cover each of the five authority/partition axes, the three checked arithmetic operations, exact fit on both budget inequalities, and one-byte-short cases on each independent budget. Exact fit is the positive control and reaches `ImplementationPending` in this RED; each rejection asserts the exact axis/operation and bytes plus the zero audit. Assert literal case counts so deleting a relation cannot reduce coverage.

- [ ] **Step 5: Add independent calibration-axis controls**

Add exactly:

```rust
#[test]
fn slice2_combined_admission_rejects_foreign_calibration_before_allocation()
```

Starting from the same valid fixture, change only one of the nine calibration axes per case. Assert `ForeignCalibration { axis }`, zero native-create/allocator counters and an empty recorder after each case. Assert the literal count `9`; UUID-only foreign calibration is mandatory.

- [ ] **Step 6: Add the real-ledger and three-generation test**

Add exactly:

```rust
#[test]
fn slice2_valid_combined_admission_executes_declared_ledger_once_and_later_generations_allocate_nothing()
```

Open an empty recorder before calling the actual admission API. On GREEN, require native-create `1`, physical allocator count `3`, generation count `1`, scoring/archive count `1`, archive-only count `0`, observed length exactly `3`, then compare the complete vector. Explicitly construct `ArchiveOnlyArena` in the assertion that no observed entry has that category, keeping the forbidden category warning-clean. Snapshot the recorder, queue generations `1`, `2` and `3` through move-only transitions, and assert equality with the snapshot after every queue.

- [ ] **Step 7: Verify the exact five test names and no hidden pass test**

Run:

```powershell
$r6TestPath = 'crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs'
$r6ExpectedTests = @(
    'slice2_combined_admission_rejects_missing_or_zero_archive_arena_before_allocation',
    'slice2_combined_admission_rejects_each_aligned_layout_field_mismatch_before_allocation',
    'slice2_combined_admission_rejects_insufficient_reserve_before_allocation',
    'slice2_combined_admission_rejects_foreign_calibration_before_allocation',
    'slice2_valid_combined_admission_executes_declared_ledger_once_and_later_generations_allocate_nothing'
)
$r6Source = Get-Content -Raw -LiteralPath $r6TestPath
$r6ActualTests = [regex]::Matches(
    $r6Source,
    '(?m)^\s*#\[test\]\r?\n\s*fn\s+([a-z0-9_]+)\s*\('
) | ForEach-Object { $_.Groups[1].Value }
if ($r6ActualTests.Count -ne 5 -or (Compare-Object $r6ExpectedTests $r6ActualTests)) {
    throw "R6 test-name set differs from the frozen five: $($r6ActualTests -join ', ')"
}
```

Expected: no output and exit zero. The assertion compares the names, not only
the number of `#[test]` attributes.

### Task 3: Capture the exact RED and mutation evidence

**Files:**

- Verify: `crates/neoethos-gpu-cuda/src/resident_search_v2.rs`
- Verify: `crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs`

- [ ] **Step 1: Run the focused warning-clean offline RED**

Run in PowerShell from the isolated repository:

```powershell
$env:CARGO_INCREMENTAL = '0'
$env:RUSTFLAGS = '-D warnings'
cargo test -p neoethos-gpu-cuda --locked --offline --no-default-features --features cuda --lib 'resident_search_v2::resident_search_v2_tests::slice2_' -- --nocapture
```

Expected: compilation succeeds warning-clean; exactly `0 passed; 5 failed`; every failure shows `ImplementationPending`; there is no ignored test, unrelated failure or device call.

- [ ] **Step 2: Freeze the complete mutation register without claiming RED kills**

Add one test-owned constant mutation-name register and assert its literal
cardinality from the relevant five tests. It must enumerate every control below:

- remove missing or zero archive validation;
- replace every-field comparison with subtotal-only comparison while keeping each mutated total self-consistent;
- replace checked reserve arithmetic with wrapping or saturating arithmetic;
- remove each of the five reserve/headroom/workspace relations;
- change `<=` to `<` so exact fit is rejected, and accept a budget one byte
  short on either independent boundary;
- remove each calibration-axis comparison;
- copy expected ledger into observed instead of recording calls;
- skip or reorder one call; independently change its ordinal, symbol, category,
  requested bytes, aligned bytes, alignment, flags, stream or pool;
- prepend and append an extra observed entry to kill zip-without-length comparisons;
- allocate on generation two and, separately, generation three.

Do not apply or claim to kill implementation mutants in this RED commit: the
stub at this stage returns `ImplementationPending` before those implementations
exist, so such a claim would be vacuous. The mutation register becomes an
executable mandatory gate in the first GREEN implementation commit that removes
`ImplementationPending`, after the canonical R1-R9 pure-RED checkpoint. At that
point apply/revert each mutant, rerun the five tests and require the intended
test to fail. This deferred GREEN receipt does not block the subsequent R7-R9
RED scaffolds after R6 review; it blocks advancing the production implementation
past combined-admission GREEN and blocks the authorized RTX run.

- [ ] **Step 3: Inspect exact scope and diff**

Run:

```powershell
git status --short
git diff --check
git diff --name-only
git diff -- crates/neoethos-gpu-cuda/src/resident_search_v2.rs crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs
git diff --no-index -- NUL crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs
```

Expected: the tracked diff and explicit untracked-file diff together cover only
the two allowed R6 Rust paths; no whitespace errors; existing untracked
`vendor/` and the historical rustc ICE report remain untouched. Exit `1` from
the `--no-index` command means a content difference was shown and is expected.

- [ ] **Step 4: Commit the bounded RED**

```powershell
git add -- crates/neoethos-gpu-cuda/src/resident_search_v2.rs crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs
git diff --cached --check
git diff --cached --name-only
git diff --cached -- crates/neoethos-gpu-cuda/src/resident_search_v2.rs crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs
git commit -m "test(search): add exact R6 combined preallocation RED"
```

Expected before commit: the cached path set is exactly the two allowed paths,
and the cached diff includes the entire newly created test file.

- [ ] **Step 5: Run the required review loop**

Assign the commit first to a spec reviewer and fix/re-review until P0/P1/P2 are `0/0/0`. Then assign the resulting commit to a fresh code-quality reviewer and repeat until `0/0/0`. Do not begin R7 before both reviews approve the same final R6 commit.
