# Resident Search Slice 2 R8 Transactional-Fault Contract Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the exact four R8 GPU-gated tests and their tests-first contract,
without implementing or claiming the later production or R9 device proof.

**Architecture:** Keep all R8 case ledgers, test-only audit types, and four test
entry points in the already CUDA-gated `resident_archive_knn_v2_tests.rs`.
The RED boundary is explicitly `ImplementationPending`; later production work
must replace that boundary with the real admitted V3 owner and device-visible
fixture before the exact serial command can turn GREEN.

**Tech Stack:** Rust 2024, libtest, the existing NeoEthos CUDA crate, pinned
nightly `2026-04-07`, Cargo offline/locked mode, and a later eligible NVIDIA
CUDA host.

**Authority:**
`docs/superpowers/specs/2026-08-29-resident-search-slice2-r8-transactional-faults-design-v10.md`.
Do not reinterpret its case counts, payloads, owner dispositions, command, or
R9 boundary.

---

## Chunk 1: Exact R8 tests-first contract

### Task 1: Freeze the case ledgers

**Files:**
- Modify: `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs`

- [ ] **Step 1: Add the explicit metric array**

Transcribe the authority's exact `[MetricFaultCase; 22]`; do not generate or
concatenate it. Validate the exact ordinal formula and bit pattern, reject
duplicate `[value_class][metric_slot]` cells through `[[bool; 11]; 2]`, and
require every cell exactly once.

- [ ] **Step 2: Add the six literal structural cases**

Transcribe the authority's explicit `[StructuralFaultCase; 6]` with six
distinct variants and exact payload fields; do not generate it. Validate the
exact ordinal, reject duplicates through a `[bool; 6]` variant seen-set, and
require every variant exactly once.

- [ ] **Step 3: Add the combined recovery ledger**

Map metric identities to `0..=21` and structural identities to `22..=27`.
Before each device case, checked-increment its `[u8; 28]` cell and reject a
duplicate. Finish by requiring every cell equal one and exact executed-fault,
fresh-admission, and successful-fresh-publication counters of `28`. A combined
vector length is not evidence.

- [ ] **Step 4: Format the one file**

Run:

```powershell
cargo +nightly-2026-04-07 fmt --package neoethos-gpu-cuda -- --check
```

Expected: exit `0`; no source or Cargo file changes from formatting. If and
only if this checkout returns Windows `os error 206`, capture that exact
failure and run the pinned, one-file fallback:

```powershell
rustup run nightly-2026-04-07 rustfmt --edition 2024 --check crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs
```

Label the fallback only as direct one-file rustfmt evidence, not package Cargo
fmt evidence. No `--all` formatting command is permitted.

### Task 2: Write the four failing R8 tests

**Files:**
- Modify: `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs`

- [ ] **Step 1: Add exactly the four authority names**

Add one `#[test]` for each exact name in the v10 design. Add no other `r8_`
test. Each body validates its literal ledger before reaching the missing
device driver.

- [ ] **Step 2: Add the fail-closed required-card gate**

At the start of every R8 test, require
`std::env::var("NEOETHOS_REQUIRE_GPU").as_deref() == Ok("1")`. Missing or
different values panic with the bounded required-card diagnostic; never skip.

- [ ] **Step 3: Add one explicit RED boundary**

Use a test-only typed `ImplementationPending` result shared by the four tests.
It must not synthesize a terminal fault, device identity, event proof, cleanup,
allocation receipt, or passing result. Each test fails with its own exact name
and the same bounded `R8ImplementationPending` reason.

- [ ] **Step 4: Ratchet the test inventory without compiling CUDA**

Run:

```powershell
rg -n '^fn r8_' crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs
```

Expected: exactly four rows, with the exact four names and no fifth row.

- [ ] **Step 5: Review the RED honestly**

Record that this host cannot compile or run the CUDA-gated module because
`nvcc` and `nvidia-smi` are absent. Do not label the source inventory as an
executed RED, GPU evidence, or 4/4 test evidence.

- [ ] **Step 6: Commit the bounded R8 RED only after approval**

Stage only:

```powershell
git add -- crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs
git diff --cached --name-only
```

Expected: the one exact test path. Commit only after root approval; preserve
all R7 paths, `vendor/`, and the ICE receipt.

### Task 3: Bind the real first-fault device fixture in later production work

**Files:**
- Modify: `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs`
- Production files: governed by a separate reviewed production plan; not
  authorized for modification by this R8 contract plan

- [ ] **Step 1: Replace only the pending driver boundary**

After the separate production implementation is reviewed, bind the tests to
the real valid combined admission and the test-only same-stream device-visible
injection seam. A test-local CPU transaction model or host-fabricated terminal
receipt is forbidden.

- [ ] **Step 2: Prove first-fault immutability in every case**

For all 28 primary injections, submit the v10 later-overwrite probe and assert
the terminal typed latch retains the primary discriminant and exact payload.

- [ ] **Step 3: Prove atomic non-publication**

Compare the post-admission snapshot with the terminal observation: identical
packed commit word and allocation ledger, no generation/store/epoch/archive
advance, and no reachable staged tail.

- [ ] **Step 4: Prove the four cleanup states**

Assert semantic-fault cleanup exactly once in this order: scoring/archive
arena; generation arena; population evaluator/session workspace; trim
map/count/event arena; retained parent import/schema/full-admission Rust owners.
Require native release/tombstone acknowledgement before each Rust disarm/drop.
NotReady returns the same pending owner across two polls with zero cleanup;
unknown outcome and unproved event poison and retain the whole owner with zero
cleanup and no reuse.

- [ ] **Step 5: Prove all 28 fresh-run pairs**

After each event-proved recoverable fault cleanup, admit a valid run with a
different run token, box identity, and every allocation identity. Require the
fresh run to publish successfully and assert the literal pair count `28`.

### Task 4: Capture the one later real-CUDA receipt

**Files:**
- Test: `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs`
- Evidence: external fresh run directory; do not add generated build output to Git

- [ ] **Step 1: Preflight an eligible NVIDIA host**

Require the pinned Rust/Cargo toolchain, working CUDA compiler/runtime, an
eligible discrete NVIDIA device, offline dependency closure, and a fresh target
directory. Stop before Cargo if any preflight fails.

- [ ] **Step 2: Run the exact serial command**

```powershell
$env:NEOETHOS_REQUIRE_GPU = "1"
$env:CARGO_INCREMENTAL = "0"
$env:RUSTFLAGS = "-Dwarnings"
$env:CARGO_NET_OFFLINE = "true"
cargo +nightly-2026-04-07 test --locked --offline -j 7 -p neoethos-gpu-cuda --no-default-features --features cuda-device-fixtures --lib r8_ -- --test-threads=1 --nocapture
```

Expected GREEN: exit `0`, exact line `running 4 tests`, and these full-name rows
once each with status `ok`:

```text
test resident_archive_knn_v2_tests::r8_all_eleven_metric_slots_reject_nan_and_infinity_atomically ... ok
test resident_archive_knn_v2_tests::r8_structural_fault_matrix_is_atomic ... ok
test resident_archive_knn_v2_tests::r8_fault_cleanup_is_checked_once_and_owner_never_reused ... ok
test resident_archive_knn_v2_tests::r8_every_recoverable_fault_allows_a_fresh_unrelated_run ... ok
```

The exact GREEN aggregate is
`test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 24 filtered out; finished in <duration>`.
Only the parsed duration may vary. The exact RED form has the same four full
names ending in `FAILED` and aggregate
`test result: FAILED. 0 passed; 4 failed; 0 ignored; 0 measured; 24 filtered out; finished in <duration>`.

- [ ] **Step 3: Reject vacuous or partial execution**

Fail the receipt on `running 0 tests`, any non-exact aggregate, fewer or more
than four unique full-name rows, a renamed/ignored/duplicate row, missing raw
stream, missing GPU-required environment, or any warning/error outside an
explicitly labelled RED result.

- [ ] **Step 4: Review logs in required order**

Preserve and hash raw stdout/stderr. Classify and review INFO, then WARNING,
then ERROR, then other lines. Record exact command/environment, GPU identity,
source/binary hashes, 4/4 name set, status, and exit code.

- [ ] **Step 5: Stop at the R8 boundary**

Report only R8 transactional-fault and cleanup evidence. Do not claim the R9
calibration, multi-generation math, CUDA interception, D2H/synchronization,
headless/readiness, deadline, integration, or release gates.

### Task 5: Independent review and handoff

**Files:**
- Review: `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs`
- Read-only authority: `docs/superpowers/specs/2026-08-29-resident-search-slice2-r8-transactional-faults-design-v10.md`

- [ ] **Step 1: Dispatch one independent read-only reviewer**

Require exact names/counts/payloads, post-admission device visibility,
first-fault behavior, cleanup ownership, fresh-run identity isolation, command,
zero-test rejection, and R9 non-claims. Record P0/P1/P2.

- [ ] **Step 2: Fix only concrete contract deviations**

Do not recursively redesign R8. If the reviewer finds a deviation, make one
bounded correction and obtain root direction before any broader change.

- [ ] **Step 3: Verify the final diff**

Run:

```powershell
git diff --check -- crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs
git diff --name-only
git diff --cached --name-only
```

Expected: no whitespace errors; only approved paths appear in either diff.

- [ ] **Step 4: Handoff without overclaiming**

Report exact paths/hashes, test status, raw-log hashes if a GPU run occurred,
review counts, untouched R7/R9 paths, and the remaining production/R9 gates.
