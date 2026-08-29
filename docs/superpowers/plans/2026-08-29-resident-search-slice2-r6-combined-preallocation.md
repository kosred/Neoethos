# Resident Search Slice 2 R6 Combined Preallocation Implementation Plan v7

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land one warning-clean behavioral RED commit that proves the exact combined Search preallocation contract without invoking CUDA hardware.

**Architecture:** One private shared module is compiled either by CUDA production or by a dependency-empty host-contract unit-test feature. It owns the real admission seam, exact layout/reserve/calibration vocabulary, move-only owner and recorder facade; the otherwise-valid RED path returns `ImplementationPending` before native create, so precisely five named host tests fail without resolving CUDA tooling.

**Tech Stack:** Rust unit tests, checked `u64` arithmetic, move-only fixture owner, test-only allocator recorder, Cargo offline verification.

---

Authority: `docs/superpowers/specs/2026-08-28-resident-search-slice2-archive-knn-design.md`, version 7. Do not reinterpret the shared-module topology, three-entry physical ledger, four reserve/workspace authorities, error/axis names, test names or mutation list in this plan.

## File map and hard scope

- Modify: `crates/neoethos-gpu-cuda/Cargo.toml`
  - add only `resident-search-slice2-host-contract = []` under `[features]`.
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
  - add only the exact private shared-module `cfg(any(...))` declaration.
- Create: `crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs`
  - own the exact error/axis/request/receipt/allocation-call DTOs,
    allocator-facade trait, move-only owner shape, pending seam and child-test
    registration; this is the sole future production authority, not a mirror.
- Create: `crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs`
  - own the sole recorder implementation/private state/read-only accessors,
    valid and mutated fixture builders, mutation register and exactly five
    `#[test]` functions;
- Forbidden: `crates/neoethos-gpu-cuda/src/resident_search_v2.rs`, `Cargo.lock`,
  `build.rs`, every native/CUDA file, R1-R5 files and every other source/test.
- Hardware/network boundary: no device enumeration, CUDA call, VPS call, registry access or network access occurs in this RED.

## Why v6 could not produce the RED

The attempted v6 command enabled `cuda` and stopped in
`cust_raw`/`find_cuda_helper` with `Could not find a cuda installation`. This
host has no `CUDA_PATH`, `CUDA_ROOT`, `nvcc` or `cuobjdump`; the crate's own
`build.rs` also resolves `nvcc` whenever `CARGO_FEATURE_CUDA` is set and has no
`DOCS_RS` bypass. Do not repeat that command or report its build failure as an
R6 test failure. Version 7 separates pure Rust contract compilation from the
unchanged CUDA production build.

## Chunk 1: Exact warning-clean R6 RED

### Task 1: Add the shared R6 admission authority and pending seam

**Files:**

- Modify: `crates/neoethos-gpu-cuda/Cargo.toml`
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Create: `crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs`
- Test: `crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs`

- [ ] **Step 1: Add the dependency-empty host-contract feature**

Under `[features]` in `crates/neoethos-gpu-cuda/Cargo.toml`, add exactly:

```toml
resident-search-slice2-host-contract = []
```

Do not add it to `default`, `cuda`, `cuda-device-fixtures` or any other feature.
Do not add a dependency or change an existing feature value.

- [ ] **Step 2: Register the one private shared authority**

Add exactly one declaration to `crates/neoethos-gpu-cuda/src/lib.rs`:

```rust
#[cfg(any(
    feature = "cuda",
    all(test, feature = "resident-search-slice2-host-contract")
))]
#[cfg_attr(
    all(
        feature = "cuda",
        not(all(test, feature = "resident-search-slice2-host-contract"))
    ),
    allow(dead_code)
)]
mod resident_search_slice2_admission_v2;
```

Keep the existing `#[cfg(feature = "cuda")] pub mod resident_search_v2;`
unchanged. Do not create separate host and CUDA declarations; the `any` gate is
the single-owner all-features ratchet. The narrow `cfg_attr` is required only
because CUDA production does not consume this RED authority yet; it does not
apply to host-contract or all-features unit tests and must be removed when the
later CUDA binding lands.

- [ ] **Step 3: Add the exact error and axis vocabulary to the shared authority**

Create `crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs`.
Use these exact enums there without `#[cfg(test)]`; payloads are part of both
the host contract and future CUDA production assertion surface:

The shared file may import only `core`, `std` and non-CUDA crate authorities.
It must not import `cust`, `vector-ta`, `resident_search_v2`, another CUDA-gated
sibling or native FFI; future CUDA production adapts into these shared DTOs.

```rust
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReserveAuthorityAxisV2 {
    AllocatorContextHeadroomIdentity,
    FullWorkspaceAuthorityIdentity,
    RetainedPreSearchWorkspaceIdentity,
    RemainingSearchAllocationAfterTrimIdentity,
    RetainedPlusRemainingEqualsFullWorkspace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReserveArithmeticV2 {
    WorkspacePartitionAdd,
    RequestedDeviceSumAdd,
    SameContextFreeMinusHeadroom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AllocationBudgetAxisV2 {
    RemainingSearchAllocationAfterTrim,
    SameContextFreeAfterAllocatorHeadroom,
}

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

- [ ] **Step 4: Add shared contract DTOs and the sole child recorder**

The request has named fields for all twelve aligned components; do not represent them as two arrays later compared by `zip`. It carries a generation receipt whose `total_device_bytes` is checked from every generation component, and a combined scoring/archive receipt whose `total_device_bytes` is checked from the unchanged scoring components plus all twelve Slice 2 fields exactly once. It also carries archive presence/bytes, the four byte authorities and their four distinct identities, same-context free bytes, and the nine calibration axes. The expected ledger reads the two totals from those exact receipts; it never accepts independently copied total arguments.

Define `ResidentSearchSlice2AllocationFacadeV2` in the shared authority with only three effects: `begin_native_create`, `cuda_host_alloc` and `cuda_malloc_async`. The latter two receive the actual call metadata below. The admission seam receives `&mut dyn ResidentSearchSlice2AllocationFacadeV2`; it receives no expected vector and there is no generic append/setter method.

Implement that trait in `resident_search_v2_tests.rs` as `ResidentSearchSlice2AllocationRecorderV2`. Keep every counter and `observed` field private to the child test module and expose read-only accessors/snapshot only. Thus the parent admission seam can produce an observed entry only by invoking an actual facade allocation method; it cannot copy the declared ledger into recorder state.

Shared-authority allocator facade:

```rust
pub(crate) trait ResidentSearchSlice2AllocationFacadeV2 {
    fn begin_native_create(&mut self);
    fn cuda_host_alloc(&mut self, actual: ResidentSearchSlice2AllocationCallV2);
    fn cuda_malloc_async(&mut self, actual: ResidentSearchSlice2AllocationCallV2);
}
```

Shared-authority allocation-call DTOs:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AllocationSymbolV2 {
    CudaHostAlloc,
    CudaMallocAsync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AllocationCategoryV2 {
    TerminalHostReceipt,
    GenerationArena,
    ScoringArchiveArena,
    ArchiveOnlyArena,
}

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

The unconditional RED seam must not invoke any facade method before returning
`ImplementationPending`. Under host-contract and all-features unit tests the
narrow module-level `dead_code` allowance is intentionally absent, so the child
must reference all three concrete trait method items without calling them. Put
these references inside the existing `assert_mutation_register_is_frozen`
helper, which every one of the five tests calls:

```rust
let _begin_native_create =
    <ResidentSearchSlice2AllocationRecorderV2 as ResidentSearchSlice2AllocationFacadeV2>::begin_native_create
        as fn(&mut ResidentSearchSlice2AllocationRecorderV2);
let _cuda_host_alloc =
    <ResidentSearchSlice2AllocationRecorderV2 as ResidentSearchSlice2AllocationFacadeV2>::cuda_host_alloc
        as fn(
            &mut ResidentSearchSlice2AllocationRecorderV2,
            ResidentSearchSlice2AllocationCallV2,
        );
let _cuda_malloc_async =
    <ResidentSearchSlice2AllocationRecorderV2 as ResidentSearchSlice2AllocationFacadeV2>::cuda_malloc_async
        as fn(
            &mut ResidentSearchSlice2AllocationRecorderV2,
            ResidentSearchSlice2AllocationCallV2,
        );
```

Method-item references make the trait surface warning-clean but have no runtime
effect: they do not call native create, allocate, or mutate any recorder field.
A broad `allow(dead_code)` on the host-contract or all-features unit-test branch
is forbidden.

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

- [ ] **Step 5: Add the actual shared admission stub and move-only returned owner**

The function signature in the shared authority takes the full request and the independent recorder. It returns a move-only owner whose `queue_generation_v2(self, ordinal, recorder)` transition consumes and returns ownership. The RED implementation returns only the typed pending error before mutating the recorder:

```rust
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

- [ ] **Step 6: Register the child only from the shared authority**

Add only this registration to
`resident_search_slice2_admission_v2.rs`:

```rust
#[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
#[path = "resident_search_v2_tests.rs"]
mod resident_search_v2_tests;
```

Do not register the child directly in `lib.rs` or `resident_search_v2.rs`.

- [ ] **Step 7: Check formatting on only the three allowed Rust files**

Run:

```powershell
rustfmt --edition 2024 --check crates/neoethos-gpu-cuda/src/lib.rs crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs
```

Expected: only the three allowed Rust paths and `Cargo.toml` differ;
`resident_search_v2.rs` remains byte-identical to the pre-R6 commit.

### Task 2: Add exactly five behavioral tests

**Files:**

- Create: `crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs`
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
$r6TestAttributeCount = [regex]::Matches($r6Source, '(?m)^\s*#\[test\]\s*$').Count
$r6ActualTests = [regex]::Matches(
    $r6Source,
    '(?m)^\s*#\[test\]\r?\n\s*fn\s+([a-z0-9_]+)\s*\('
) | ForEach-Object { $_.Groups[1].Value }
if ($r6TestAttributeCount -ne 5 -or $r6ActualTests.Count -ne 5 -or
    (Compare-Object $r6ExpectedTests $r6ActualTests)) {
    throw "R6 test-name set differs from the frozen five: $($r6ActualTests -join ', ')"
}
```

Expected: no output and exit zero. The assertion compares the names, not only
the number of `#[test]` attributes.

### Task 3: Capture the exact RED and mutation evidence

**Files:**

- Verify: `crates/neoethos-gpu-cuda/Cargo.toml`
- Verify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Verify: `crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs`
- Verify: `crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs`
- Verify unchanged: `crates/neoethos-gpu-cuda/build.rs`
- Verify unchanged: `crates/neoethos-gpu-cuda/src/resident_search_v2.rs`

- [ ] **Step 1: Run the focused warning-clean offline RED**

Run in PowerShell from the isolated repository:

```powershell
$env:CARGO_INCREMENTAL = '0'
$env:RUSTFLAGS = '-Dwarnings'
cargo +nightly-2026-04-07 test --locked --offline -j 7 -p neoethos-gpu-cuda --no-default-features --features resident-search-slice2-host-contract --lib 'resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_' -- --nocapture
```

Expected: compilation succeeds warning-clean; exactly `0 passed; 5 failed`; every failure shows `ImplementationPending`; there is no ignored test, unrelated failure, CUDA dependency resolution, native CUDA build or device call. `-j 7` is before the test-runner `--` and is parsed by Cargo.

- [ ] **Step 2: Prove the feature graph, build log and dual-feature topology**

Run Cargo metadata without building:

```powershell
$r6Metadata = cargo +nightly-2026-04-07 metadata --locked --offline --no-deps --format-version 1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'R6 metadata failed' }
$r6Package = $r6Metadata.packages | Where-Object { $_.name -eq 'neoethos-gpu-cuda' }
if (@($r6Package).Count -ne 1) { throw 'expected exactly one neoethos-gpu-cuda package' }
$r6Features = $r6Package.features
$r6RequiredFeatureKeys = @(
    'resident-search-slice2-host-contract',
    'default',
    'cuda',
    'cuda-device-fixtures'
)
foreach ($r6RequiredFeatureKey in $r6RequiredFeatureKeys) {
    if ($r6Features.PSObject.Properties.Name -notcontains $r6RequiredFeatureKey) {
        throw "missing literal feature key: $r6RequiredFeatureKey"
    }
}
$r6HostEdges = @($r6Features.'resident-search-slice2-host-contract')
if ($r6HostEdges.Count -ne 0) { throw "host contract has feature edges: $r6HostEdges" }
if (@($r6Features.default).Count -ne 0) { throw 'default feature semantics changed' }
if (Compare-Object @('dep:cust', 'dep:vector-ta') @($r6Features.cuda)) {
    throw 'cuda feature semantics changed'
}
if (Compare-Object @('cuda') @($r6Features.'cuda-device-fixtures')) {
    throw 'cuda-device-fixtures semantics changed'
}
$r6Inclusions = foreach ($r6FeatureProperty in $r6Features.PSObject.Properties) {
    if (@($r6FeatureProperty.Value) -contains 'resident-search-slice2-host-contract') {
        $r6FeatureProperty.Name
    }
}
if (@($r6Inclusions).Count -ne 0) { throw "host feature included by: $r6Inclusions" }
cargo +nightly-2026-04-07 metadata --locked --offline --no-deps --format-version 1 --all-features | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'all-features metadata resolution failed' }
```

Then repeat the focused RED from a guaranteed-new task-specific target with
verbose logs. Preserve and restore a caller-owned `CARGO_TARGET_DIR` even when
an assertion fails:

```powershell
if ($env:CARGO_INCREMENTAL -ne '0' -or $env:RUSTFLAGS -ne '-Dwarnings') {
    throw 'R6 proof requires CARGO_INCREMENTAL=0 and RUSTFLAGS=-Dwarnings'
}
$r6ProofId = [guid]::NewGuid().ToString('N')
$r6ProofTarget = Join-Path (Get-Location) "target/codex-r6-host-contract-proof-$r6ProofId"
$r6BuildLogPath = Join-Path (Get-Location) "target/codex-r6-host-contract-proof-$r6ProofId.log"
if ((Test-Path -LiteralPath $r6ProofTarget) -or (Test-Path -LiteralPath $r6BuildLogPath)) {
    throw 'fresh R6 proof path already exists'
}
$r6HadTargetDir = Test-Path Env:CARGO_TARGET_DIR
$r6PriorTargetDir = $env:CARGO_TARGET_DIR
$r6BuildLog = @()
$r6Exit = $null
try {
    $env:CARGO_TARGET_DIR = $r6ProofTarget
    $r6BuildLog = & cargo +nightly-2026-04-07 test --locked --offline -vv -j 7 -p neoethos-gpu-cuda --no-default-features --features resident-search-slice2-host-contract --lib 'resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_' -- --nocapture 2>&1
    $r6Exit = $LASTEXITCODE
} finally {
    if ($r6HadTargetDir) {
        $env:CARGO_TARGET_DIR = $r6PriorTargetDir
    } else {
        Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
    }
}
$r6BuildText = $r6BuildLog -join "`n"
[IO.File]::WriteAllText(
    $r6BuildLogPath,
    $r6BuildText + "`n",
    [Text.UTF8Encoding]::new($false)
)
$r6BuildLogSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $r6BuildLogPath).Hash.ToLowerInvariant()
Write-Output "R6_BUILD_LOG=$r6BuildLogPath"
Write-Output "R6_BUILD_LOG_SHA256=$r6BuildLogSha256"
if ($r6Exit -ne 101 -or $r6BuildText -notmatch 'test result: FAILED\. 0 passed; 5 failed; 0 ignored') {
    throw "unexpected R6 RED result, exit=$r6Exit"
}
$r6ExpectedFailures = @(
    'resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_combined_admission_rejects_missing_or_zero_archive_arena_before_allocation',
    'resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_combined_admission_rejects_each_aligned_layout_field_mismatch_before_allocation',
    'resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_combined_admission_rejects_insufficient_reserve_before_allocation',
    'resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_combined_admission_rejects_foreign_calibration_before_allocation',
    'resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_valid_combined_admission_executes_declared_ledger_once_and_later_generations_allocate_nothing'
)
$r6PanicHeaderPattern = "(?m)^[ \t]*thread '(?<name>[^']+)'(?: \(\d+\))? panicked at "
$r6PanicProbeName = 'resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_probe'
$r6PanicProbe = "thread '$r6PanicProbeName' (20536) panicked at <anon>:1:1"
$r6PanicProbeMatch = [regex]::Match($r6PanicProbe, $r6PanicHeaderPattern)
if (-not $r6PanicProbeMatch.Success -or
    $r6PanicProbeMatch.Groups['name'].Value -ne $r6PanicProbeName) {
    throw 'panic-header regex rejects the pinned-nightly thread-id shape'
}
$r6PanicNames = @(
    [regex]::Matches($r6BuildText, $r6PanicHeaderPattern) |
        ForEach-Object { $_.Groups['name'].Value }
)
$r6FailedStatusNames = @(
    [regex]::Matches($r6BuildText, '(?m)^test (?<name>\S+) \.\.\. FAILED\r?$') |
        ForEach-Object { $_.Groups['name'].Value }
)
$r6FailureListPattern = '(?ms)^failures:\r?\n(?<entries>(?:[ \t]{4}\S+\r?\n)+)(?=\r?\n(?:note:[^\r\n]*\r?\n\r?\n)?test result:)'
$r6FailureListProbe = @'
failures:
    resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_probe

note: run with RUST_BACKTRACE=1 environment variable to display a backtrace

test result: FAILED. 0 passed; 1 failed; 0 ignored
'@
if (-not [regex]::Match($r6FailureListProbe, $r6FailureListPattern).Success) {
    throw 'failure-list regex rejects the pinned-nightly nocapture shape'
}
$r6FinalFailureList = [regex]::Match($r6BuildText, $r6FailureListPattern)
if (-not $r6FinalFailureList.Success) {
    throw 'missing final libtest failure list'
}
$r6FinalFailureNames = @(
    [regex]::Matches(
        $r6FinalFailureList.Groups['entries'].Value,
        '(?m)^[ \t]{4}(?<name>\S+)\r?$'
    ) | ForEach-Object { $_.Groups['name'].Value }
)
$r6PendingCount = [regex]::Matches($r6BuildText, '\bImplementationPending\b').Count
if ((Compare-Object $r6ExpectedFailures $r6PanicNames) -or
    (Compare-Object $r6ExpectedFailures $r6FailedStatusNames) -or
    (Compare-Object $r6ExpectedFailures $r6FinalFailureNames) -or
    $r6PanicNames.Count -ne 5 -or $r6FailedStatusNames.Count -ne 5 -or
    $r6FinalFailureNames.Count -ne 5 -or $r6PendingCount -ne 5) {
    throw "R6 failures were not exactly the five Pending panics: panic=$r6PanicNames status=$r6FailedStatusNames final=$r6FinalFailureNames pending=$r6PendingCount"
}
$r6ForbiddenBuildPattern = @'
(?ix)
(?<![A-Za-z0-9_])cust_raw(?![A-Za-z0-9_])
|(?<![A-Za-z0-9_])find_cuda_helper(?![A-Za-z0-9_])
|(?<![A-Za-z0-9_])cust(?![A-Za-z0-9_])
|(?<![A-Za-z0-9_-])vector-ta(?![A-Za-z0-9_-])
|(?<![A-Za-z0-9_])nvcc(?:\.exe)?(?![A-Za-z0-9_])
|(?<![A-Za-z0-9_])cuobjdump(?:\.exe)?(?![A-Za-z0-9_])
|rustc-link-lib\s*=\s*(?:(?:static|dylib|framework)\s*=\s*)?(?:cuda|cudart|cudart_static)(?![A-Za-z0-9_])
|(?<![A-Za-z0-9_])-l\s*(?:(?:static|dylib|framework)\s*=\s*)?(?:cuda|cudart|cudart_static)(?![A-Za-z0-9_])
|/DEFAULTLIB:\s*["']?(?:cuda|cudart|cudart_static)(?:\.lib)?["']?(?![A-Za-z0-9_])
|(?<![A-Za-z0-9_])(?:cuda|cudart|cudart_static)\.lib(?![A-Za-z0-9_])
'@
$r6ForbiddenSamples = @(
    'cust_raw',
    'find_cuda_helper',
    'cust',
    'vector-ta',
    '"nvcc.exe"',
    'cuobjdump.exe',
    '-lcuda',
    '-l static=cudart_static',
    'rustc-link-lib=static=cudart_static',
    '/DEFAULTLIB:cuda.lib'
)
foreach ($r6ForbiddenSample in $r6ForbiddenSamples) {
    if ($r6ForbiddenSample -notmatch $r6ForbiddenBuildPattern) {
        throw "forbidden-build regex misses: $r6ForbiddenSample"
    }
}
$r6AllowedSamples = @('neoethos-gpu-cuda', 'rerun-if-env-changed=CUDA_PATH')
foreach ($r6AllowedSample in $r6AllowedSamples) {
    if ($r6AllowedSample -match $r6ForbiddenBuildPattern) {
        throw "forbidden-build regex rejects intended exemption: $r6AllowedSample"
    }
}
if ($r6BuildText -match $r6ForbiddenBuildPattern) {
    throw "host-contract build reached forbidden CUDA edge: $($Matches[0])"
}
```

The crate/path name `neoethos-gpu-cuda` and `rerun-if-env-changed=CUDA_PATH`
are not CUDA dependency/link evidence and are intentionally not forbidden.
The commands persist the complete log, print its SHA-256 receipt and keep both
outside the four-path R6 source commit.

Finally assert one shared declaration and unchanged production semantics:

```powershell
$r6Lib = Get-Content -Raw crates/neoethos-gpu-cuda/src/lib.rs
$r6LibLf = $r6Lib.Replace("`r`n", "`n").Replace("`r", "`n")
$r6ExpectedSharedDeclaration = @'
#[cfg(any(
    feature = "cuda",
    all(test, feature = "resident-search-slice2-host-contract")
))]
#[cfg_attr(
    all(
        feature = "cuda",
        not(all(test, feature = "resident-search-slice2-host-contract"))
    ),
    allow(dead_code)
)]
mod resident_search_slice2_admission_v2;
'@
$r6ExpectedSharedDeclaration = $r6ExpectedSharedDeclaration.Replace("`r`n", "`n").Replace("`r", "`n")
$r6SharedDeclarationCount = [regex]::Matches(
    $r6LibLf,
    [regex]::Escape($r6ExpectedSharedDeclaration)
).Count
$r6SharedModuleCount = [regex]::Matches(
    $r6LibLf,
    '(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?mod\s+resident_search_slice2_admission_v2\s*;[ \t]*$'
).Count
if ($r6SharedDeclarationCount -ne 1 -or $r6SharedModuleCount -ne 1) {
    throw 'exact shared admission gate must occur once'
}
$r6ExpectedProductionDeclaration = @'
#[cfg(feature = "cuda")]
pub mod resident_search_v2;
'@
$r6ExpectedProductionDeclaration = $r6ExpectedProductionDeclaration.Replace("`r`n", "`n").Replace("`r", "`n")
$r6ProductionDeclarationCount = [regex]::Matches(
    $r6LibLf,
    [regex]::Escape($r6ExpectedProductionDeclaration)
).Count
$r6ProductionModuleCount = [regex]::Matches(
    $r6LibLf,
    '(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?mod\s+resident_search_v2\s*;[ \t]*$'
).Count
if ($r6ProductionDeclarationCount -ne 1 -or $r6ProductionModuleCount -ne 1) {
    throw 'production resident_search_v2 CUDA gate changed'
}
$r6SharedSource = Get-Content -Raw crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs
$r6SharedSourceLf = $r6SharedSource.Replace("`r`n", "`n").Replace("`r", "`n")
$r6ExpectedChildDeclaration = @'
#[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
#[path = "resident_search_v2_tests.rs"]
mod resident_search_v2_tests;
'@
$r6ExpectedChildDeclaration = $r6ExpectedChildDeclaration.Replace("`r`n", "`n").Replace("`r", "`n")
$r6ChildDeclarationCount = [regex]::Matches(
    $r6SharedSourceLf,
    [regex]::Escape($r6ExpectedChildDeclaration)
).Count
$r6ChildModuleCount = [regex]::Matches(
    $r6SharedSourceLf,
    '(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?mod\s+resident_search_v2_tests\s*;[ \t]*$'
).Count
if ($r6ChildDeclarationCount -ne 1 -or $r6ChildModuleCount -ne 1) {
    throw 'exact host-contract child gate must occur once'
}
$r6ChildOwners = @(rg -l 'mod resident_search_v2_tests;' crates/neoethos-gpu-cuda/src)
if ($r6ChildOwners.Count -ne 1 -or $r6ChildOwners[0] -notlike '*resident_search_slice2_admission_v2.rs') {
    throw "test child has wrong or duplicate owner: $r6ChildOwners"
}
git diff --exit-code 4f0880148677df0d8f58c11373b42d0bd87e5b13 -- crates/neoethos-gpu-cuda/build.rs crates/neoethos-gpu-cuda/src/resident_search_v2.rs Cargo.lock
if ($LASTEXITCODE -ne 0) { throw 'CUDA/default production semantics or lockfile changed' }
```

Metadata proves both features can be selected together and the single
`cfg(any(...))` prevents duplicate module ownership. Do not claim local
all-features compilation; repeat that build later on the authorized CUDA host.

- [ ] **Step 3: Freeze the complete mutation register without claiming RED kills**

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

- [ ] **Step 4: Inspect exact scope and diff**

Run:

```powershell
git status --short
git diff --check
git diff --name-only
git diff -- crates/neoethos-gpu-cuda/Cargo.toml crates/neoethos-gpu-cuda/src/lib.rs
git diff --no-index -- NUL crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs
git diff --no-index -- NUL crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs
```

Expected: the tracked diff and explicit untracked-file diffs together cover only
the exact four R6 paths; no whitespace errors; existing untracked
`vendor/` and the historical rustc ICE report remain untouched. Exit `1` from
each `--no-index` command means a content difference was shown and is expected.

- [ ] **Step 5: Commit the bounded RED**

```powershell
git add -- crates/neoethos-gpu-cuda/Cargo.toml crates/neoethos-gpu-cuda/src/lib.rs crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'cached R6 diff has whitespace errors' }
$r6AllowedPaths = @(
    'crates/neoethos-gpu-cuda/Cargo.toml',
    'crates/neoethos-gpu-cuda/src/lib.rs',
    'crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs',
    'crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs'
)
$r6CachedPaths = @(git diff --cached --name-only)
if ($LASTEXITCODE -ne 0 -or (Compare-Object $r6AllowedPaths $r6CachedPaths)) {
    throw "cached R6 scope is not the exact four paths: $r6CachedPaths"
}
git diff --cached -- crates/neoethos-gpu-cuda/Cargo.toml crates/neoethos-gpu-cuda/src/lib.rs crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs
git commit -m "test(search): add exact R6 combined preallocation RED"
```

Expected before commit: the cached path set is exactly the four allowed paths,
and the cached diff includes the entire newly created test file.

- [ ] **Step 6: Run the required review loop**

Assign the commit first to a spec reviewer and fix/re-review until P0/P1/P2 are `0/0/0`. Then assign the resulting commit to a fresh code-quality reviewer and repeat until `0/0/0`. Do not begin R7 before both reviews approve the same final R6 commit.
