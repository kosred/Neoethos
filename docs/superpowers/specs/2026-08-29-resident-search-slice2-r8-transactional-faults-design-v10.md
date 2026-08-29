# Resident Search Slice 2 R8 transactional-fault correction v10

**Status:** approved docs-only authority for the R8 contract; no implementation
or CUDA result is claimed

**Prepared source observation:**
`512f2f68ef68b63ebf6469f67e0b749e77666309`

**Publication rule:** the design, execution plan, and v10 SHA manifest are the
only new paths in the authority commit. The commit has the prepared source
observation above as its sole parent. Existing dirty R7 implementation paths,
`vendor/`, and the ICE receipt remain unstaged and byte-identical.

## Supersession boundary

This document corrects only R8, “transactional faults and cleanup,” from the
version-5 Slice 2 authority at commit
`a17a2091e094f9064695de1f6a9e3247d995dfc6` and the unchanged copy of that R8
text in the later shared design. It does not rewrite the immutable v5/v8
design or manifests. It does not modify or weaken R1-R7, the R7 v9 nominal API,
R9, readiness, headless routing, production binding, or the one-hour deadline.

The following authorities remain byte-identical:

- `docs/superpowers/specs/2026-08-28-resident-search-slice2-archive-knn-design.md`
  (normalized-LF SHA-256
  `52b166cc52a09358e47e9da3ce1daad5a692783fea820027fb4db491d2b1431a`);
- `audit/resident-search-slice2-design-v5.sha256`
  (`3f370ffe7561dc26e99b1834b482d0399a188befa2bd68da2b661e771b7de144`);
- `docs/superpowers/specs/2026-08-29-resident-search-slice2-r7-compile-contract-design-v9.md`
  (`a8159742fa56c958e0e16c1d54bf2ef2b61ec28cf7bdba4b26222336cb6205bd`);
- `docs/superpowers/plans/2026-08-29-resident-search-slice2-r7-compile-contract-v9.md`
  (`b6298c4401247370e45f45264b05d0b7aee35874ee42c9a5d2f2838d1a6239fd`);
- `audit/resident-search-slice2-design-v9.sha256`
  (`7e03d57aa8b718990f506feb912e4134594c09c043a575395953c793d2cfbac4`).

Normalized-LF digests use the v9 algorithm: reject BOM/invalid UTF-8, replace
CRLF and then bare CR with LF, preserve every other byte and final-newline
presence, then SHA-256 the result.

## Exact R8 topology

R8 adds contract code only to
`crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs`. The existing
registration remains exactly:

```rust
#[cfg(all(test, feature = "cuda"))]
mod resident_archive_knn_v2_tests;
```

Exactly four test functions whose names start with `r8_` are permitted:

1. `r8_all_eleven_metric_slots_reject_nan_and_infinity_atomically`;
2. `r8_structural_fault_matrix_is_atomic`;
3. `r8_fault_cleanup_is_checked_once_and_owner_never_reused`;
4. `r8_every_recoverable_fault_allows_a_fresh_unrelated_run`.

Helpers and case tables are not tests. A fifth `#[test]` whose name starts with
`r8_`, a renamed test, or a selected zero-test run is a contract failure.
Each test itself requires the environment value to be exactly
`NEOETHOS_REQUIRE_GPU=1`; absence or any other value fails loudly and never
skips or returns success.

## The 28 post-admission device cases

Every case first completes one valid combined admission. Only then may the
test-only fixture alter the device-visible value consumed by the production
validator on the admitted stream. A host-only preflight rejection, a fixture
that mutates the request before admission, or a host-synthesized terminal
fault does not satisfy R8.

The metric test owns this explicit `[MetricFaultCase; 22]`; it is not generated
or concatenated:

```text
[(0, 0, QNaN), (1, 0, PosInf),
 (2, 1, QNaN), (3, 1, PosInf),
 (4, 2, QNaN), (5, 2, PosInf),
 (6, 3, QNaN), (7, 3, PosInf),
 (8, 4, QNaN), (9, 4, PosInf),
 (10, 5, QNaN), (11, 5, PosInf),
 (12, 6, QNaN), (13, 6, PosInf),
 (14, 7, QNaN), (15, 7, PosInf),
 (16, 8, QNaN), (17, 8, PosInf),
 (18, 9, QNaN), (19, 9, PosInf),
 (20, 10, QNaN), (21, 10, PosInf)]
```

Each tuple is `(case_ordinal, metric_slot, value_class)`. `QNaN` has exact bits
`0x7ff8_0000_0000_0000`; `PosInf` has exact bits
`0x7ff0_0000_0000_0000`. Execution validates
`case_ordinal == 2 * metric_slot + value_class_ordinal`, records a
`[[bool; 11]; 2]` value-class-by-slot seen-set, rejects a duplicate before execution, and requires
every cell exactly once afterward. The typed latch carries the metric slot and
exact observed bits.

The structural test owns this explicit `[StructuralFaultCase; 6]`, in the
listed ordinal order:

| Case | Exact typed payload |
| --- | --- |
| low signature width | expected `4`, observed `3` |
| high signature width | expected `4`, observed `5` |
| zero union | observed `0`, sealed allowed range `1..=32` |
| archive count overflow | sealed capacity `50_000`, observed `50_001` |
| boxed receipt token corruption | after valid admission, observed token is the sealed opaque token XOR `1_u64`; the actual box and CUDA allocation are not rewritten |
| comparator-bound drift | sealed `(union_max=32, cross_product_max=1_024)`, observed `(union_max=33, cross_product_max=1_024)` |

```text
[SignatureWordCountLow { ordinal: 0, expected: 4, observed: 3 },
 SignatureWordCountHigh { ordinal: 1, expected: 4, observed: 5 },
 ZeroUnion { ordinal: 2, observed: 0, allowed_min: 1, sealed_max: 32 },
 ArchiveCountOverflow { ordinal: 3, sealed_capacity: 50_000, observed: 50_001 },
 BoxedReceiptTokenMismatch { ordinal: 4, observed_xor_mask: 1_u64 },
 ComparatorBoundMismatch { ordinal: 5,
   sealed_union_max: 32, sealed_cross_product_max: 1_024,
   observed_union_max: 33, observed_cross_product_max: 1_024 }]
```

The six entries are distinct enum variants, not a common string payload. The
test records a `[bool; 6]` exact-variant seen-set, rejects a duplicate or wrong
ordinal before execution, and requires every variant exactly once afterward.

The recovery test does not prove coverage by concatenated length. It maps the
22 metric ordinals to executed identities `0..=21` and the six structural
ordinals to `22..=27`, increments a checked `[u8; 28]` execution counter before
each device case, rejects any value above one, and finally requires all 28
cells equal one plus exact fault-run and successful-fresh-run counters of 28.

## First-fault and atomic-publication authority

The device fault authority starts empty and accepts exactly the first typed
fault with a single-winner latch. Its discriminant and the case payload above
are immutable after that transition. Each case schedules a deterministic
later overwrite probe: `ZeroUnion` unless the primary is `ZeroUnion`, then
`ArchiveCountOverflow { observed: 50_001 }`. The terminal receipt must still
contain the primary typed fault. A last-fault value, bit mask without payload,
panic string, or host reconstruction is rejected.

For every primary case:

- the packed commit word equals its post-admission/pre-injection snapshot;
- committed generation, current store, commit epoch, and archive count do not
  advance;
- staged tail bytes are not published or reachable;
- the exact declared allocation ledger is unchanged; and
- the compact terminal projection returns the latched typed fault.

## Completion and cleanup matrix

`r8_fault_cleanup_is_checked_once_and_owner_never_reused` executes exactly four
terminal-state fixtures:

1. **Event-proved semantic fault.** The event proof succeeds, the exact typed
   semantic fault is projected, and checked cleanup runs once in the inherited
   order: scoring/archive arena; generation arena; population evaluator/session
   workspace; trim map/count/event arena; retained parent import, schema, and
   full-admission Rust owners. Native must acknowledge each exact release or
   tombstone state before the corresponding Rust owner is disarmed or dropped.
   A second cleanup or reuse attempt is refused by the tombstone.
2. **NotReady.** `try_complete_v3` returns `NotReady` with the same pending
   owner. A second poll receives the same run, boxed-token, and allocation
   identities. Cleanup deltas remain zero across both polls. NotReady is not a
   fault, tombstone, poison, cleanup, or owner replacement.
3. **Unknown asynchronous CUDA outcome.** The whole composite is poisoned and
   retained deliberately. No child owner is detached, dropped, cleaned, or
   reusable, and cleanup deltas remain zero.
4. **Unproved event.** Even if terminal bytes look plausible, absence or
   identity drift of the exact event proof poisons and retains the whole
   composite with zero cleanup and no reuse.

The unknown and unproved paths may retain/leak by design; they may not turn an
ambiguous device lifetime into successful cleanup. Test-only audit receipts
must observe the whole-owner disposition without adding a production raw
pointer or release API.

## Fresh-run isolation

`r8_every_recoverable_fault_allows_a_fresh_unrelated_run` repeats all 22 metric
and six structural semantic faults as event-proved recoverable faults. After
the one checked cleanup for each fault, it admits exactly one valid fresh run.
The fresh run must publish successfully and must have a different run token,
boxed-receipt identity, and every allocation identity from the faulted run.
The test retains only the old audit tombstone long enough to compare identities;
it does not retain released CUDA storage. There are exactly 28 fault/fresh-run
pairs and no shared owner, box, allocation receipt, or staged tail between a
pair.

## Exact CUDA command and evidence

The sole R8 execution shape is serial and GPU-required:

```powershell
$env:NEOETHOS_REQUIRE_GPU = "1"
$env:CARGO_INCREMENTAL = "0"
$env:RUSTFLAGS = "-Dwarnings"
$env:CARGO_NET_OFFLINE = "true"
cargo +nightly-2026-04-07 test --locked --offline -j 7 -p neoethos-gpu-cuda --no-default-features --features cuda-device-fixtures --lib r8_ -- --test-threads=1 --nocapture
```

The exact fully qualified names are:

```text
resident_archive_knn_v2_tests::r8_all_eleven_metric_slots_reject_nan_and_infinity_atomically
resident_archive_knn_v2_tests::r8_structural_fault_matrix_is_atomic
resident_archive_knn_v2_tests::r8_fault_cleanup_is_checked_once_and_owner_never_reused
resident_archive_knn_v2_tests::r8_every_recoverable_fault_allows_a_fresh_unrelated_run
```

The evidence collector preserves raw stdout/stderr and requires the exact
standalone line `running 4 tests` plus exactly one line shaped
`test <fully-qualified-name> ... <status>` for each name above. The observed
name set must be exactly 4/4, with no duplicate, missing, additional, ignored,
or filtered-in R8 test. The GREEN aggregate is exactly
`test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 24 filtered out; finished in <duration>`;
the explicit RED aggregate is exactly
`test result: FAILED. 0 passed; 4 failed; 0 ignored; 0 measured; 24 filtered out; finished in <duration>`.
Only the parsed duration field may vary. `running 0 tests`, any other aggregate,
a zero-pass success, a missing name row, or success without
`NEOETHOS_REQUIRE_GPU=1` is rejected. RED and GREEN receipts label their status;
GREEN requires all four full-name rows to end in `ok`, while RED requires all
four to end in `FAILED` with the exact bounded pending reason.

This local Windows host has neither `nvcc` nor `nvidia-smi` on its executable
path and cannot execute that command. Therefore v10 provides authority and an
implementation plan only. It is not local CUDA, real-card, RED, GREEN,
performance, or integration evidence.

## R9 boundary

R8 proves only the 28 fault cases, first-fault latch, atomic non-publication,
four-state cleanup behavior, and 28 unrelated fresh-run recoveries. It does
not prove calibration representativeness or rate, multi-generation kNN/math
parity, allocation/API interception, zero intermediate D2H/synchronization,
the terminal projection count, interposer non-vacuity, RTX identity,
prepared-headless behavior, readiness, or a one-hour deadline. Those remain
R9 or later production/integration gates. R8 does not create or modify the R9
device-test or CUDA-interceptor paths.

## Review and publication bound

This authority receives one author pass and one independent read-only review,
bounded to 40 minutes total. The reviewer reports P0/P1/P2 counts against the
same three new bytes. There is no recursive design expansion. Staging and
commit require root approval after exact paths, normalized-LF hashes, and
review counts are reported.
