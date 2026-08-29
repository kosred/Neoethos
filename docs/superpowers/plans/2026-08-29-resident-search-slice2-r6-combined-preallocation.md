# Resident Search Slice 2 R6 Combined Preallocation Implementation Plan v8

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land one crate-warning-clean behavioral RED commit that proves the exact combined Search preallocation contract without invoking CUDA hardware or trusting caller-declared totals, symbols or reserve authorities.

**Architecture:** One private shared module is compiled either by CUDA production or by a dependency-empty host-contract unit-test feature. It owns the real admission seam, exact layout/receipt/reserve/calibration vocabulary, move-only owner and method-derived recorder facade; the otherwise-valid RED path returns `ImplementationPending` before native create, so precisely five named host tests fail without resolving CUDA tooling.

**Tech Stack:** Rust unit tests, checked `u64` arithmetic, move-only fixture owner, test-only allocator recorder, Cargo offline verification.

---

Authority: `docs/superpowers/specs/2026-08-28-resident-search-slice2-archive-knn-design.md`, version 8. Do not reinterpret the shared-module topology, three-entry physical ledger, opaque trusted reserve/workspace authority, error/axis names, five test names or 132-entry mutation list in this plan.

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

## Chunk 1: Exact crate-warning-clean R6 RED

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
pub(crate) enum ResidentSearchSlice2ReceiptTotalAxisV2 {
    ReplacementSubtotal,
    GenerationReceiptTotal,
    ScoringArchiveReceiptTotal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReceiptArithmeticV2 {
    ReplacementSubtotalAdd,
    GenerationReceiptTotalAdd,
    ScoringArchiveReceiptTotalAdd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReserveAuthorityKindV2 {
    AllocatorContextHeadroom,
    FullWorkspaceAuthority,
    RetainedPreSearchWorkspace,
    RemainingSearchAllocationAfterTrim,
    SameContextFree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2AuthorityBindingAxisV2 {
    DeviceUuid,
    PrimaryContext,
    SearchStream,
    ActivePool,
    RunIdentity,
    FullWorkspaceReceiptIdentity,
    PostTrimReceiptIdentity,
    AuthorityIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentSearchSlice2ReserveRelationV2 {
    FourReserveAuthorityIdentitiesDistinct,
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
    ReceiptArithmeticOverflow {
        operation: ResidentSearchSlice2ReceiptArithmeticV2,
    },
    ReceiptTotalMismatch {
        axis: ResidentSearchSlice2ReceiptTotalAxisV2,
        expected_total_bytes: u64,
        observed_total_bytes: u64,
    },
    ReserveAuthorityBytesMismatch {
        authority: ResidentSearchSlice2ReserveAuthorityKindV2,
        expected_bytes: u64,
        observed_bytes: u64,
    },
    ReserveAuthorityBindingMismatch {
        authority: ResidentSearchSlice2ReserveAuthorityKindV2,
        axis: ResidentSearchSlice2AuthorityBindingAxisV2,
    },
    ReserveAuthorityRelationMismatch {
        relation: ResidentSearchSlice2ReserveRelationV2,
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

The request has named fields for all twelve aligned components; do not represent
them as two arrays later compared by `zip`. It carries a generation receipt
whose `total_device_bytes` is recomputed with checked addition from every
generation component, and a combined scoring/archive receipt whose total is
recomputed from the unchanged scoring components plus all twelve Slice 2 fields
exactly once. The separately passed opaque trusted reserve seal—not the observed
request—owns expected bytes, expected bindings, expected calibration and
full-workspace/post-trim provenance for all five authorities.

Add the exact observed/trusted reserve DTOs from design version 8. The full
binding contains UUID, context, stream, pool, run, full-workspace receipt,
post-trim receipt and authority identity. `ResidentSearchSlice2AdmissionRequestV2`
contains `ResidentSearchSlice2ObservedReserveSetV2` plus only the observed
`calibration`; remove the old `expected_identity` fields,
`expected_calibration` and bare `same_context_free_bytes`.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AuthorityBindingV2 {
    pub(crate) device_uuid: [u8; 16],
    pub(crate) primary_context_identity: u64,
    pub(crate) search_stream_identity: u64,
    pub(crate) active_pool_identity: u64,
    pub(crate) run_identity: u64,
    pub(crate) full_workspace_receipt_identity: u64,
    pub(crate) post_trim_receipt_identity: u64,
    pub(crate) authority_identity: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2ObservedReserveAuthorityV2 {
    pub(crate) bytes: u64,
    pub(crate) binding: ResidentSearchSlice2AuthorityBindingV2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2ObservedReserveSetV2 {
    pub(crate) allocator_context_headroom: ResidentSearchSlice2ObservedReserveAuthorityV2,
    pub(crate) full_workspace_authority: ResidentSearchSlice2ObservedReserveAuthorityV2,
    pub(crate) retained_pre_search_workspace: ResidentSearchSlice2ObservedReserveAuthorityV2,
    pub(crate) remaining_search_allocation_after_trim: ResidentSearchSlice2ObservedReserveAuthorityV2,
    pub(crate) same_context_free: ResidentSearchSlice2ObservedReserveAuthorityV2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2CalibrationBindingV2 {
    pub(crate) device_uuid: [u8; 16],
    pub(crate) primary_context_identity: u64,
    pub(crate) search_stream_identity: u64,
    pub(crate) active_pool_identity: u64,
    pub(crate) cuda_build_identity: u64,
    pub(crate) kernel_semantics_identity: u64,
    pub(crate) binary64_math_identity: u64,
    pub(crate) plan_identity: u64,
    pub(crate) run_identity: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2TrustedReserveAuthorityV2 {
    expected_bytes: u64,
    expected_binding: ResidentSearchSlice2AuthorityBindingV2,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2TrustedReserveSetV2 {
    allocator_context_headroom: ResidentSearchSlice2TrustedReserveAuthorityV2,
    full_workspace_authority: ResidentSearchSlice2TrustedReserveAuthorityV2,
    retained_pre_search_workspace: ResidentSearchSlice2TrustedReserveAuthorityV2,
    remaining_search_allocation_after_trim: ResidentSearchSlice2TrustedReserveAuthorityV2,
    same_context_free: ResidentSearchSlice2TrustedReserveAuthorityV2,
}

pub(crate) struct ResidentSearchSlice2TrustedReserveSealV2 {
    trusted_reserve: ResidentSearchSlice2TrustedReserveSetV2,
    expected_calibration: ResidentSearchSlice2CalibrationBindingV2,
    sealed_full_workspace_receipt_identity: u64,
    sealed_post_trim_receipt_identity: u64,
}
```

The trusted authority, trusted set and seal fields are private. They do not
derive or implement `Clone`, `Copy` or `Default`; no R6 production/raw
constructor, mutable accessor or inner-set accessor exists. Admission consumes
the seal by value. The only R6 construction site is the zero-argument
`mint_r6_trusted_reserve_seal_for_fixture_v2()` in the already cfg-gated child
test module. It computes expected bytes, calibration and full/post-trim
provenance from independent fixture constants and never accepts or reads the
observed request.

Add exact source/topology assertions inside the third existing test: freeze the
private-field blocks above; require zero `Clone`/`Copy`/`Default` implementation
or derive for the trusted authority/set/seal; require no shared-source function
that accepts raw values and returns the seal; require the admission signature to
take the seal by value; require the child minter's exact cfg plus zero-argument
signature; and require its seal struct literal to be the sole child construction
site. No dependency, extra target or sixth test is allowed. This internal R6
ratchet does not claim R7 public opacity.

Before every by-value admission call, invoke the child-only
`assert_r6_trusted_reserve_seal_fixture_v2(&seal)`. It must directly read and
assert all five private expected byte values, every field of all five private
expected bindings, all nine private expected-calibration fields and both
private sealed provenance fields against independent constants. Freeze the
inspector in the child source census and forbid a production/raw inspector or
accessor. This both consumes the test-only field surface under `-Dwarnings` and
keeps the evidence independent of the observed request.

The test minter makes all five expected bindings share the expected calibration
UUID/context/stream/pool/run tuple and the seal's full-workspace/post-trim
provenance. The later CUDA minter is explicitly deferred and must consume the
actual opaque calibration, full-workspace and post-trim authorities; it may not
accept a request or raw bytes/bindings.

Define `ResidentSearchSlice2AllocationFacadeV2` with only three effects:
`begin_native_create`, `cuda_host_alloc` and `cuda_malloc_async`. The two
allocator methods receive the method-specific argument DTOs below; neither DTO
has a symbol field. The admission seam receives no expected ledger and the
recorder has no generic append/setter method.

Shared-authority allocator facade:

```rust
pub(crate) trait ResidentSearchSlice2AllocationFacadeV2 {
    fn begin_native_create(&mut self);
    fn cuda_host_alloc(&mut self, actual: ResidentSearchSlice2HostAllocationArgsV2);
    fn cuda_malloc_async(&mut self, actual: ResidentSearchSlice2AsyncAllocationArgsV2);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2HostAllocationArgsV2 {
    pub(crate) ordinal: u8,
    pub(crate) category: ResidentSearchSlice2AllocationCategoryV2,
    pub(crate) requested_bytes: u64,
    pub(crate) aligned_bytes: u64,
    pub(crate) alignment_bytes: u64,
    pub(crate) flags: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AsyncAllocationArgsV2 {
    pub(crate) ordinal: u8,
    pub(crate) category: ResidentSearchSlice2AllocationCategoryV2,
    pub(crate) requested_bytes: u64,
    pub(crate) aligned_bytes: u64,
    pub(crate) alignment_bytes: u64,
    pub(crate) flags: u32,
    pub(crate) stream_identity: u64,
    pub(crate) pool_identity: u64,
}
```

The method implementation constructs `ResidentSearchSlice2AllocationCallV2`
and derives its symbol from the invoked method. It never accepts the symbol from
the admission implementation.

The sole child-owned recorder implementation in
`resident_search_v2_tests.rs`:

```rust
#[cfg(test)]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2AllocationRecorderV2 {
    phase: ResidentSearchSlice2RecorderPhaseV2,
    native_create_count: u64,
    host_allocator_method_count: u64,
    async_allocator_method_count: u64,
    physical_allocator_count: u64,
    generation_arena_count: u64,
    scoring_archive_arena_count: u64,
    archive_only_arena_count: u64,
    chronology: Vec<ResidentSearchSlice2RecorderEventV2>,
    observed: Vec<ResidentSearchSlice2AllocationCallV2>,
}
```

Define the exact `BeforeNativeCreate`, `NativeCreateBegun` and
`AllocationsComplete` phases and `NativeCreate { phase_before }` /
`Allocation { phase_at_call, call }` chronology events from the design. Keep
all state private and expose read-only accessors/snapshot only. Native create is
an ordered chronology event, not a physical allocation-ledger row. Derive
`Default` only for the phase enum and mark `BeforeNativeCreate` as its sole
`#[default]`, so the recorder's derived default remains exact.

The unconditional RED seam must not invoke any facade method before returning
`ImplementationPending`. Under host-contract and all-features unit tests the
narrow module-level `dead_code` allowance is intentionally absent. The fifth
test uses exactly five separate recorder-control fixtures: async
`TerminalHostReceipt`, host `GenerationArena`, host `ScoringArchiveArena`,
direct `begin_native_create`, and one allocation before native create. The
first three prove every category-specific symbol/count is derived from the
invoked method; the fourth proves the exact `NativeCreate` event/phase; the
fifth proves `BeforeNativeCreate` allocation chronology. It then opens a new
empty recorder for the real RED admission call. These controls make the trait
surface crate-warning-clean without mutating the admission recorder. A broad
`allow(dead_code)` on the host-contract or all-features unit-test branch is
forbidden.

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

The valid expected chronology has exactly four entries: `NativeCreate` from
`BeforeNativeCreate`, then the three rows above in `NativeCreateBegun` phase.
Host-method count is `1`, async-method count is `2`, and final phase is
`AllocationsComplete`.

There is no `ArchiveOnlyArena` call. "Archive arena" in the frozen error/test names means the logical archive subreceipt inside `ScoringArchiveArena`, never a fourth physical allocation. `0x01` is the sealed `cudaHostAllocPortable` flag. Event creation is absent from this vocabulary.

- [ ] **Step 5: Add the actual shared admission stub and move-only returned owner**

The function signature in the shared authority takes the full request and the independent recorder. It returns a move-only owner whose `queue_generation_v2(self, ordinal, recorder)` transition consumes and returns ownership. The RED implementation returns only the typed pending error before mutating the recorder:

```rust
pub(crate) fn admit_slice2_combined_fixture_v2(
    _request: ResidentSearchSlice2AdmissionRequestV2,
    _trusted_seal: ResidentSearchSlice2TrustedReserveSealV2,
    _allocator: &mut dyn ResidentSearchSlice2AllocationFacadeV2,
) -> Result<ResidentSearchSlice2AdmissionOwnerV2, ResidentSearchSlice2AdmissionErrorV2> {
    Err(ResidentSearchSlice2AdmissionErrorV2::ImplementationPending)
}
```

The later GREEN implementation order is fixed even though this commit remains RED:

1. validate archive presence and nonzero bytes;
2. compare all twelve named aligned fields;
3. recompute the replacement subtotal with checked addition, returning overflow
   before mismatch; repeat that exact precedence for generation and
   scoring/archive totals;
4. compare every authority byte and every binding field with full-width
   equality. Single-axis negatives require their exact typed errors but define
   no mutual order. For simultaneous mismatches freeze only the executable
   partial precedence: headroom before other authorities, full workspace before
   retained, headroom bytes before its binding and headroom `DeviceUuid` before
   its other binding axes;
5. prove the distinctness, relation, checked-arithmetic and budget invariants.
   Their single-fault cases require exact typed errors and existing
   checked-overflow-before-total-mismatch rules, but make no additional mutual
   first-error ordering claim;
6. validate every calibration axis with full-width equality;
7. invoke native create once, then host allocation once and async allocation
   twice in the exact order; the recorder derives symbol/method/phase evidence;
8. return the move-only owner; its three generation queues make no allocation
   call.

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

Their checked replacement subtotal is `23_707_648`. The deterministic
generation component vector is exactly:

```text
14_592, 25_600, 25_600, 65_792, 20_992, 8_192,
5_120, 9_472, 65_536, 256, 256
```

The order is logical scalar, index, weight, offspring, metric, rank, selection,
dedup, CUB scratch, retained evaluation coverage and terminal device receipt.
The checked host-contract generation total is `241_408`. The `65_536` CUB
scratch input is deliberately opaque host-fixture data; it is not a CUDA query
claim. The later CUDA adapter supplies the real runtime-query value. The
scoring/archive receipt uses fitness `1_792`, decision keys `1_792`, opaque CUB
scratch `65_536` and the twelve layout values exactly once, for checked total
`23_776_768`. Compute all totals with `checked_add`, never independent total
literals.

Freeze the canonical requested sum and reserve bytes exactly as:

```text
requested_device_sum                         = 24_018_176
allocator_context_headroom                   =  8_388_608
retained_pre_search_workspace                = 67_108_864
remaining_search_allocation_after_trim       = 24_018_176
full_workspace_authority                     = 91_127_040
same_context_free                            = 32_406_784
```

The canonical fixture is exact-fit on both device-budget inequalities and on
the retained-plus-remaining workspace partition; it contains no unlabelled
slack.

Freeze authority identities as headroom `0x3101`, full workspace `0x3202`,
retained `0x3303`, remaining `0x3404` and same-context free `0x3505`. Define
`PAIR_ALIAS_IDENTITY_V2 = 0xA11A_5E00_0000_0001`; assert it differs from every
canonical authority identity before running the six pair-only controls.

Build the observed reserve set separately, then call the zero-argument fixture
minter for a fresh opaque trusted seal on every admission attempt. Choose
nonzero identities and make the original four authority identities pairwise
distinct. Every binding contains the same UUID, context, stream, pool, run,
full-workspace-receipt and post-trim-receipt facts plus its
authority-specific identity. Same-context free also has independent expected
bytes and a full binding. Keep calibration IDs nonzero and distinct enough that
cross-field comparisons cannot pass by value collision. The request contains
no expected calibration and cannot pass plain values into the minter.

- [ ] **Step 2: Add the missing/zero archive test**

Add exactly:

```rust
#[test]
fn slice2_combined_admission_rejects_missing_or_zero_archive_arena_before_allocation()
```

Run the absent and zero cases through `admit_slice2_combined_fixture_v2` with a
fresh independently minted trusted seal for each by-value call. Assert the
exact error for each and, after each call, assert
native-create/host-method/async-method `0`, every allocator counter `0`, initial
phase and empty chronology/ledger.

- [ ] **Step 3: Add the twelve-field mismatch test**

Add exactly:

```rust
#[test]
fn slice2_combined_admission_rejects_each_aligned_layout_field_mismatch_before_allocation()
```

Create twelve named mutations. Decrement the selected aligned field by one, then recompute every derived subtotal and the scoring/archive arena total from that mutated declaration. This keeps its internal subtotal arithmetic self-consistent so subtotal-only validation would pass, while the named field still differs from the authoritative layout. Assert the exact `AlignedLayoutFieldMismatch` field and byte payload plus the zero-before-native-create audit after every case. Assert the case count is literally `12`.

In the same test, keep all twelve observed components equal to authority and
change only `replacement_subtotal_bytes` by one; require
`ReceiptTotalMismatch { axis: ReplacementSubtotal, ... }`. Then create an
authoritative/observed layout whose first two components overflow checked
addition and whose declared total also differs. Require
`ReceiptArithmeticOverflow { operation: ReplacementSubtotalAdd }`, proving
overflow precedence. Both cases retain the zero audit. The exact control count
inside this test is `14`: twelve named rows, one total-only mismatch and one
overflow-precedence case.

- [ ] **Step 4: Add reserve/workspace arithmetic and boundary controls**

Add exactly:

```rust
#[test]
fn slice2_combined_admission_rejects_insufficient_reserve_before_allocation()
```

First add generation and scoring/archive total controls. For each receipt, keep
every component unchanged and change only `total_device_bytes` by one; require
the exact `ReceiptTotalMismatch` axis and bytes. Then overflow its component sum
while also making the declared total disagree, and require the exact
`ReceiptArithmeticOverflow` before mismatch.

For each of the five authority kinds, independently mutate observed bytes and
each of the eight binding axes while trusted bytes/binding stay fixed. Every
`u64` binding field has bit-0 and bit-63 subcases; UUID has byte-0 and byte-15
subcases. Assert literal authority count `5` and binding-axis count `8`.

Add an all-alias case in which observed and trusted authority identities for the
original four authorities are the same nonzero value; require
`FourReserveAuthorityIdentitiesDistinct`. Mint the canonical seal, run the full
pristine inspector, then use descendant-only private-field access to change
exactly the four trusted `expected_binding.authority_identity` fields plus the
four observed counterparts to that common value. Keep same-context free and
every other byte/binding/provenance field canonical; do not add a second minter
or accessor. Then add all six one-pair-only cases `HF`, `HR`, `HRem`, `FR`,
`FRem`, `RRem`. After pristine inspection, mutate only the chosen pair's
trusted and observed authority identities to
`PAIR_ALIAS_IDENTITY_V2 = 0xA11A_5E00_0000_0001`, keep the other two distinct
and canonical, and require `FourReserveAuthorityIdentitiesDistinct`.

Add four coordinated cases:
retained+full bytes move together, headroom+same-context-free bytes move
together, all five observed binding tuples move together, and all observed
authorities plus a child-local graph of plain expected-looking values move
together while the zero-argument minter remains unchanged. Each old equation
remains self-consistent. Freeze the exact first errors as, respectively,
`ReserveAuthorityBytesMismatch { authority: FullWorkspaceAuthority, ... }`,
`ReserveAuthorityBytesMismatch { authority: AllocatorContextHeadroom, ... }`
and `ReserveAuthorityBindingMismatch { authority: AllocatorContextHeadroom,
axis: DeviceUuid }`; the fourth reports `ReserveAuthorityBytesMismatch {
authority: AllocatorContextHeadroom, ... }` and proves the plain graph cannot be
passed or minted into a seal.

Then cover the partition relation, the three reserve checked-arithmetic
operations, exact fit on both budget inequalities and one-byte-short cases on
each independent budget. Mint and pristine-inspect a fresh seal for every case.
Use descendant-only private-field mutation in exact lockstep with observed
bytes: full workspace `91_127_041` for partition mismatch; retained and
remaining both `u64::MAX` for `WorkspacePartitionAdd`; same-context free
`0` plus headroom `1` for `SameContextFreeMinusHeadroom`; remaining
`24_018_175` plus full workspace `91_127_039` for the remaining-budget short
case; and same-context free `32_406_783` for the same-context-budget short case.
Do not change a binding or provenance field and do not add a constructor or
accessor. `RequestedDeviceSumAdd` mutates only the receipt components, so its
seal remains pristine. The canonical fixture itself is the exact-fit positive
control and reaches `ImplementationPending` without seal mutation. Every
rejection asserts its exact typed axis/operation/payload plus the zero audit.
Assert the exact submatrix counts:
two generation-total controls, two scoring-total controls, five expected-byte
controls, `5 * 8 * 2 = 80` binding-width controls, one all-alias control, six
pair-only alias controls, four coordinated controls, one partition relation,
three reserve arithmetic cases, one exact-fit control and two one-byte-short
controls. These sum to exactly `107` controls in the third test.

- [ ] **Step 5: Add independent calibration-axis controls**

Add exactly:

```rust
#[test]
fn slice2_combined_admission_rejects_foreign_calibration_before_allocation()
```

Starting from the same valid fixture, change only one of the nine calibration
axes per case. For each `u64` field run bit-0 and bit-63 subcases. For UUID run
byte-0 and byte-15 subcases. Assert `ForeignCalibration { axis }`, zero
native-create/allocator counters and empty chronology/ledger after each case.
Assert the literal outer axis count `9` and exact mutation count `18`.

- [ ] **Step 6: Add the real-ledger and three-generation test**

Add exactly:

```rust
#[test]
fn slice2_valid_combined_admission_executes_declared_ledger_once_and_later_generations_allocate_nothing()
```

Before the real admission call, use exactly five separate local recorders to
prove method and phase evidence: an async call categorized
`TerminalHostReceipt` records `CudaMallocAsync` and async count `1`; a host call
categorized `GenerationArena` records `CudaHostAlloc` and host count `1`; a
separate host call categorized `ScoringArchiveArena` also records
`CudaHostAlloc` and host count `1`; direct `begin_native_create` records one
`NativeCreate { phase_before: BeforeNativeCreate }`, advances to
`NativeCreateBegun` and makes no allocation; and an allocation before native
create records `BeforeNativeCreate`. These controls must not share state with
one another or with the admission recorder.

Open a fresh recorder before calling the actual admission API. On GREEN,
require native-create `1`, host-method count `1`, async-method count `2`,
physical allocator count `3`, generation count `1`, scoring/archive count `1`,
archive-only count `0`, chronology length exactly `4` with native create first,
and observed allocation length exactly `3` before complete vector equality.
Explicitly construct `ArchiveOnlyArena` in the assertion that no observed entry
has that category. Snapshot phase, chronology, ledger and counters, queue
generations `1`, `2` and `3` through move-only transitions, and assert equality
with the snapshot after every queue.

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

- [ ] **Step 1: Run the focused crate-warning-clean offline RED**

Run in PowerShell from the isolated repository:

```powershell
$env:CARGO_INCREMENTAL = '0'
$env:RUSTFLAGS = '-Dwarnings'
cargo +nightly-2026-04-07 test --locked --offline -j 7 -p neoethos-gpu-cuda --no-default-features --features resident-search-slice2-host-contract --lib 'resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_' -- --nocapture
```

Expected: `neoethos-gpu-cuda` has zero warning diagnostics under `-Dwarnings`;
exactly `0 passed; 5 failed`; every failure shows `ImplementationPending`; there
is no ignored test, unrelated failure, CUDA dependency resolution, native CUDA
build or device call. Do not call the complete verbose log warning-free: locked
dependencies are subject to Cargo cap-lints. The current fresh log contains 12
warning-prefixed third-party lines—`generic-array` 6, `windows-core` 1,
`windows` 2 and three Cargo warning summaries—and the evidence must preserve and
report them. `-j 7` is before the test-runner `--` and is parsed by Cargo.

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

$r6ExpectedTrustedAuthority = @'
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2TrustedReserveAuthorityV2 {
    expected_bytes: u64,
    expected_binding: ResidentSearchSlice2AuthorityBindingV2,
}
'@
$r6ExpectedTrustedSet = @'
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentSearchSlice2TrustedReserveSetV2 {
    allocator_context_headroom: ResidentSearchSlice2TrustedReserveAuthorityV2,
    full_workspace_authority: ResidentSearchSlice2TrustedReserveAuthorityV2,
    retained_pre_search_workspace: ResidentSearchSlice2TrustedReserveAuthorityV2,
    remaining_search_allocation_after_trim: ResidentSearchSlice2TrustedReserveAuthorityV2,
    same_context_free: ResidentSearchSlice2TrustedReserveAuthorityV2,
}
'@
$r6ExpectedTrustedSeal = @'
pub(crate) struct ResidentSearchSlice2TrustedReserveSealV2 {
    trusted_reserve: ResidentSearchSlice2TrustedReserveSetV2,
    expected_calibration: ResidentSearchSlice2CalibrationBindingV2,
    sealed_full_workspace_receipt_identity: u64,
    sealed_post_trim_receipt_identity: u64,
}
'@
foreach ($r6ExpectedPrivateBlock in @(
    $r6ExpectedTrustedAuthority,
    $r6ExpectedTrustedSet,
    $r6ExpectedTrustedSeal
)) {
    $r6ExpectedPrivateBlockLf = $r6ExpectedPrivateBlock.Replace("`r`n", "`n").Replace("`r", "`n")
    if ([regex]::Matches($r6SharedSourceLf, [regex]::Escape($r6ExpectedPrivateBlockLf)).Count -ne 1) {
        throw 'trusted capability private-field block drifted'
    }
}
$r6ForbiddenTrustedCapabilityPattern = @'
(?xs)
(?:\#\[derive\([^\]]*\b(?:Clone|Copy|Default)\b[^\]]*\)\]\s*)
pub\(crate\)\s+struct\s+ResidentSearchSlice2Trusted(?:ReserveAuthority|ReserveSet|ReserveSeal)V2
|
impl\s+(?:Clone|Copy|Default)\s+for\s+ResidentSearchSlice2Trusted(?:ReserveAuthority|ReserveSet|ReserveSeal)V2
|
impl\s+ResidentSearchSlice2Trusted(?:ReserveAuthority|ReserveSet|ReserveSeal)V2
|
(?:pub\(crate\)\s+)?fn\s+[a-z0-9_]+\s*\([^)]*\)\s*->\s*&?(?:mut\s+)?ResidentSearchSlice2Trusted(?:ReserveAuthority|ReserveSet|ReserveSeal)V2
'@
if ($r6SharedSourceLf -match $r6ForbiddenTrustedCapabilityPattern) {
    throw "trusted capability gained a forbidden derive/impl/raw constructor/accessor: $($Matches[0])"
}
$r6ExpectedAdmissionSealParameter = '_trusted_seal: ResidentSearchSlice2TrustedReserveSealV2,'
if ([regex]::Matches($r6SharedSourceLf, [regex]::Escape($r6ExpectedAdmissionSealParameter)).Count -ne 1) {
    throw 'admission must consume exactly one opaque trusted seal by value'
}
$r6ChildSource = Get-Content -Raw crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs
$r6ChildSourceLf = $r6ChildSource.Replace("`r`n", "`n").Replace("`r", "`n")
$r6CfgOnlyMinter = @'
#[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
fn mint_r6_trusted_reserve_seal_for_fixture_v2() -> ResidentSearchSlice2TrustedReserveSealV2 {
'@
$r6CfgOnlyInspector = @'
#[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
fn assert_r6_trusted_reserve_seal_fixture_v2(seal: &ResidentSearchSlice2TrustedReserveSealV2) {
'@
foreach ($r6ExactChildSignature in @($r6CfgOnlyMinter, $r6CfgOnlyInspector)) {
    $r6ExactChildSignatureLf = $r6ExactChildSignature.Replace("`r`n", "`n").Replace("`r", "`n")
    if ([regex]::Matches($r6ChildSourceLf, [regex]::Escape($r6ExactChildSignatureLf)).Count -ne 1) {
        throw 'cfg-only zero-input minter or read-only inspector signature drifted'
    }
}
if ([regex]::Matches(
    $r6ChildSourceLf,
    '(?m)^[ \t]*ResidentSearchSlice2TrustedReserveSealV2[ \t]*\{'
).Count -ne 1) {
    throw 'trusted seal must have one child-only construction site'
}
if ($r6ChildSourceLf -match '(?m)^\s*type\s+\w+\s*=\s*(?:super::)?ResidentSearchSlice2TrustedReserveSealV2\s*;' -or
    $r6ChildSourceLf -match '(?m)^\s*use\s+[^;]*ResidentSearchSlice2TrustedReserveSealV2\s+as\s+' -or
    $r6ChildSourceLf -match '(?m)^\s*macro_rules!') {
    throw 'trusted seal construction census has an alias or macro bypass'
}
$r6InspectorBegin = '// BEGIN R6 TRUSTED SEAL INSPECTOR V2'
$r6InspectorEnd = '// END R6 TRUSTED SEAL INSPECTOR V2'
$r6InspectorBeginIndex = $r6ChildSourceLf.IndexOf($r6InspectorBegin)
$r6InspectorEndIndex = $r6ChildSourceLf.IndexOf($r6InspectorEnd)
if ($r6InspectorBeginIndex -lt 0 -or $r6InspectorEndIndex -le $r6InspectorBeginIndex) {
    throw 'trusted seal inspector markers missing or reordered'
}
$r6Inspector = $r6ChildSourceLf.Substring(
    $r6InspectorBeginIndex,
    $r6InspectorEndIndex - $r6InspectorBeginIndex
)
$r6InspectorTokens = @(
    'allocator_context_headroom', 'full_workspace_authority',
    'retained_pre_search_workspace', 'remaining_search_allocation_after_trim',
    'same_context_free', 'expected_bytes', 'expected_binding', 'device_uuid',
    'primary_context_identity', 'search_stream_identity', 'active_pool_identity',
    'run_identity', 'full_workspace_receipt_identity',
    'post_trim_receipt_identity', 'authority_identity', 'expected_calibration',
    'cuda_build_identity', 'kernel_semantics_identity', 'binary64_math_identity',
    'plan_identity', 'sealed_full_workspace_receipt_identity',
    'sealed_post_trim_receipt_identity'
)
foreach ($r6InspectorToken in $r6InspectorTokens) {
    if ($r6Inspector -notmatch "(?<![A-Za-z0-9_])$([regex]::Escape($r6InspectorToken))(?![A-Za-z0-9_])") {
        throw "trusted seal inspector does not read: $r6InspectorToken"
    }
}
git diff --exit-code 4f0880148677df0d8f58c11373b42d0bd87e5b13 -- crates/neoethos-gpu-cuda/build.rs crates/neoethos-gpu-cuda/src/resident_search_v2.rs Cargo.lock
if ($LASTEXITCODE -ne 0) { throw 'CUDA/default production semantics or lockfile changed' }
```

Metadata proves both features can be selected together and the single
`cfg(any(...))` prevents duplicate module ownership. Do not claim local
all-features compilation; repeat that build later on the authorized CUDA host.

- [ ] **Step 3: Freeze the complete mutation register without claiming RED kills**

Add one test-owned constant mutation-name register and assert its literal
cardinality `132` from the relevant five tests. Retain the existing 63 names
byte-for-byte and in order, then append the exact 69-name block below. It must
enumerate every control below:

- remove missing or zero archive validation;
- replace every-field comparison with subtotal-only comparison while keeping each mutated total self-consistent;
- trust a declared replacement/generation/scoring total, use wrapping or
  saturating component addition, or return mismatch before checked overflow;
- replace checked reserve arithmetic with wrapping or saturating arithmetic;
- remove independently trusted expected bytes, a whole binding, any one binding
  field, any one of the six pairwise distinctness checks,
  coordinated-substitution refusal or partition;
- violate any of the four frozen simultaneous-mismatch boundaries: headroom
  before other authorities, full workspace before retained, headroom bytes
  before its binding, or headroom UUID before its other binding axes;
- expose/clone/copy/default/raw-construct the opaque trusted capability graph,
  add a raw accessor, accept an unsealed set or minter arguments, ungate the
  minter, omit expected calibration/full-workspace/post-trim provenance, accept
  coordinated plain expected-looking values, or remove child full-field
  inspection;
- change `<=` to `<` so exact fit is rejected, and accept a budget one byte
  short on either independent boundary;
- remove each calibration-axis comparison or truncate identities/UUID equality;
- copy expected ledger into observed instead of recording calls;
- skip or reorder one call; independently change its ordinal, symbol, category,
  requested bytes, aligned bytes, alignment, flags, stream or pool;
- trust a declared symbol instead of the invoked method, omit a host/async
  method count, remove native-create event recording or allocate before native
  create;
- prepend and append an extra observed entry to kill zip-without-length comparisons;
- allocate on generation two and, separately, generation three.

```text
trust_declared_replacement_subtotal
trust_declared_generation_total
trust_declared_scoring_archive_total
replacement_subtotal_add_wrapping
replacement_subtotal_add_saturating
generation_total_add_wrapping
generation_total_add_saturating
scoring_archive_total_add_wrapping
scoring_archive_total_add_saturating
return_replacement_total_mismatch_before_overflow
return_generation_total_mismatch_before_overflow
return_scoring_archive_total_mismatch_before_overflow
remove_allocator_context_headroom_expected_bytes
remove_full_workspace_authority_expected_bytes
remove_retained_pre_search_workspace_expected_bytes
remove_remaining_search_allocation_expected_bytes
remove_same_context_free_expected_bytes
remove_allocator_context_headroom_full_binding
remove_full_workspace_authority_full_binding
remove_retained_pre_search_workspace_full_binding
remove_remaining_search_allocation_full_binding
remove_same_context_free_full_binding
remove_reserve_binding_device_uuid
remove_reserve_binding_primary_context
remove_reserve_binding_search_stream
remove_reserve_binding_active_pool
remove_reserve_binding_run_identity
remove_reserve_binding_full_workspace_receipt_identity
remove_reserve_binding_post_trim_receipt_identity
remove_reserve_binding_authority_identity
accept_four_way_reserve_identity_alias
accept_headroom_full_workspace_authority_identity_alias
accept_headroom_retained_authority_identity_alias
accept_headroom_remaining_authority_identity_alias
accept_full_workspace_retained_authority_identity_alias
accept_full_workspace_remaining_authority_identity_alias
accept_retained_remaining_authority_identity_alias
accept_coordinated_workspace_byte_substitution
accept_coordinated_context_budget_byte_substitution
accept_coordinated_reserve_binding_substitution
truncate_reserve_binding_identities_to_u32
compare_reserve_binding_uuid_byte_zero_only
trust_terminal_declared_symbol_instead_of_host_method
trust_generation_declared_symbol_instead_of_async_method
trust_scoring_archive_declared_symbol_instead_of_async_method
allocate_before_native_create
remove_host_allocator_method_count
remove_async_allocator_method_count
truncate_calibration_identities_to_u32
compare_calibration_uuid_byte_zero_only
swap_allocator_context_headroom_and_full_workspace_precedence
swap_full_workspace_and_retained_precedence
validate_reserve_binding_before_bytes
swap_device_uuid_and_primary_context_precedence
expose_trusted_capability_fields
derive_clone_for_trusted_capability_graph
derive_copy_for_trusted_capability_graph
derive_default_for_trusted_capability_graph
add_raw_trusted_reserve_constructor
add_raw_trusted_reserve_accessor
pass_unsealed_trusted_reserve_set
allow_trusted_fixture_minter_arguments
ungate_trusted_fixture_minter
omit_expected_calibration_from_trusted_seal
omit_full_workspace_provenance_from_trusted_seal
omit_post_trim_provenance_from_trusted_seal
accept_coordinated_observed_and_plain_trusted_substitution
remove_trusted_fixture_field_inspection
remove_native_create_event_recording
```

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
