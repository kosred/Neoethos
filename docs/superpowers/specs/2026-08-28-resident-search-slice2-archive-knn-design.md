# Resident Search Slice 2: permanent archive and exact kNN novelty

Status: design and executable RED plan only; production implementation is not
authorized by this document.

Version: 3

Authoritative base: `7824e191c04b4eb78e547728ad7cdb78f915a2af`

Branch: `codex/resident-search-novelty-slice2`

Version 3 supersedes the review candidate at commit
`def68c428b625126e48302f1443d210e64846e84`. Its version-2 manifest remains an
immutable historical receipt; the version-3 manifest binds this corrected
document.

## Outcome and boundary

Slice 2 will add a device-resident, permanent, profitable archive and exact
mean k-nearest-neighbor Jaccard novelty to the existing resident Search owner.
It will not expose CUDA pointers, materialize per-generation genes or metrics
on the host, synchronize between generations, enter the CPU Search numerical
path, enable the headless entry point, or change any readiness bit.

The shipped current-config identity is fixed to:

- population `P = 200`;
- maximum generations `G = 20_000`;
- maximum configured run time `3_600_000 ms`;
- maximum terms per gene `M = 16`;
- trimmed feature count `F = 240`, hence signature words `W = 4`;
- novelty weight with the exact `0.2_f64` bit pattern;
- novelty neighbor count `K = 15`;
- permanent archive capacity `A = 50_000`;
- archive mode `net`, minimum net with the exact `+0.0_f64` bit pattern;
- normalized full-gene first-seen deduplication.

Passing Slice 2 tests will prove only this archive/novelty transaction. Full
multi-generation evaluator orchestration, the end-to-end one-hour deadline,
terminal portfolio projection, validation/finalization, Training handoff, and
headless `DiscoveryResult` remain later gates.

## Primary authorities

- Lehman and Stanley, ALIFE XI Eq. (1), defines novelty as mean distance to the
  `K` nearest neighbors drawn from the current population plus the permanent
  archive:
  <https://www.cs.ucf.edu/~gitars/cap6671-2010/Presentations/lehman_alife08.pdf>.
- Lehman's dissertation Appendix A records `K=15` for the referenced novelty
  experiments; Slice 2 exposes and seals that value rather than hiding it as a
  kernel constant: <https://www.joellehman.com/lehman-dissertation.pdf>.
- CUDA stream-ordered allocation and asynchronous-execution contracts govern
  same-stream lifetime, terminal-event proof and the prohibition on host access
  to an in-flight D2H destination:
  <https://docs.nvidia.com/cuda/cuda-program-guide/04-special-topics/stream-ordered-memory-allocation.html>
  and
  <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html>.
- NVIDIA CCCL determinism guidance does not by itself promise the Slice 2 total
  order; therefore the fixed-K and rank tie keys are explicit versioned inputs:
  <https://nvidia.github.io/cccl/unstable/cccl/determinism.html>.

## Review corrections incorporated through version 3

| Finding | Versioned decision |
| --- | --- |
| Chat-only freeze was not reviewable | This tracked document and its tracked SHA-256 manifest are the sole design authority. |
| Ownership omitted trim lifetimes | The Search owner consumes and retains the whole `ResidentTrimmedPopulationSessionV1`, including its native map, ready event, parent import, schema and full-admission owners. |
| Rational comparison could overflow | The comparator widens to `u64` only after a checked host preflight proves the maximum cross product; device inputs are range-validated against that receipt. |
| Existing composite cannot insert post-rank/pre-rotation | A three-state typed transaction separates score/rank, archive staging, and evolution/publication. |
| Readiness was listed as a RED | Unchanged readiness is a continuously GREEN invariant, not a missing behavior. |
| Opacity was a source-string assertion | Public opacity and move-only ownership use executable `compile_fail` doctests; missing archive allocation is a behavioral pre-allocation RED. |
| Hash collision behavior was unspecified | Hashes are accelerators only; equality falls back to the exact full gene and unequal colliding genes both survive. |
| `23_707_648` bytes was ambiguous | It replaces three V1 scoring regions and is a net `23_699_200`-byte increase before the unchanged fitness/decision/CUB regions. |
| Calibration could benchmark an easy case | The exact production fixed-K kernel is benchmarked with `P=200`, a prefilled `A=50_000` archive, `W=4`, and `K=15`. |
| Production counters were self-reported | RTX acceptance independently intercepts every D2H and synchronization API in addition to checking production counters. |
| One-hour claim was too broad | Slice 2 proves a necessary novelty-stage lower bound only; a later combined deadline gate must cover every Search stage. |
| GREEN readiness/headless assertions named no executable authority | Version 3 binds the existing behavioral readiness function in `resident_search_generation_v2_production_contract` and adds a prepared-native behavioral refusal test with an independent post-trim allocation scope. |
| Calibration rejection covered only exact and under-rate receipts | Version 3 executes empty-archive, smaller-K, proxy-kernel, stale-build, foreign-device and shape-drift rejections and proves a valid novelty-only receipt cannot satisfy the separate headless deadline/readiness gate. |
| Opacity/interception tests could pass vacuously | Version 3 requires exact rustc diagnostic codes plus compile-pass controls for R7, and live positive-control calls plus an interposer-disabled rejection for the independent CUDA interception layer. |

## Current source constraints

The CPU implementation at
`crates/neoethos-search/src/genetic/search_engine.rs:3167-3200` computes mean
Jaccard distance against all other current-population members. It has neither
`K` nor a permanent archive and is not the Slice 2 mathematical authority.

The current CUDA scoring translation unit already contains useful independent
parts:

- checked current-population bitset construction at
  `resident_scoring_novelty_v1.cu:382-419`;
- canonical finite objective scoring at `:421-445`;
- finite normalization, blending and decision-key encoding at `:569-609`.

Its old novelty kernel at `:448-492` is also mean-to-all-current and must not be
made reachable as the Slice 2 implementation. The current production bridge
correctly rejects any non-zero novelty weight at `:1126-1166`.

The generation composite at
`resident_generation_v1.cu:2626-2766` currently binds scoring, ranks, selects,
crosses, mutates, deduplicates, publishes the store rotation, copies the
terminal receipt to the host and records the completion event in one private
function. There is no valid insertion seam between rank and rotation. Slice 2
therefore versions this orchestration rather than inserting an archive side
effect around the existing function.

## Ownership model

### The actual incoming owner

`ResidentTrimmedPopulationSessionV1` is not just a population session. At
`resident_trim_prefilter_v1.rs:1160-1174` it owns:

- `ResidentPopulationSessionV3`;
- `NativeResidentTrimPrefilterRunV1`, which owns the selected compact-column
  map, selected-count scalar and trim-ready event;
- `ResidentTrimPrefilterParentImportV1`, whose retained owner carries the V3
  feature-store/context/stream lifetime;
- `SealedResidentColumnClassificationV1`;
- `ResidentTrimPrefilterFullDiscoveryAdmissionV1`;
- the private device views and exact ready receipt.

The move at `:1263-1321` deliberately keeps all of those owners together. Its
Drop at `:1551-1571` deliberately leaks every armed lifetime because no Search
consumer currently exists. Extracting only `PopulationSession` would lose the
column-map/event/retained-owner authority and is forbidden.

### Search owner

The new crate-private constructor is implemented on the trimmed carrier. It
consumes `ResidentTrimmedPopulationSessionV1` by value and passes its map and
ready-event facts directly to the native combined admission without returning
those facts to its caller.

The resulting move-only Search owner retains:

```text
ResidentSearchGenerationChainV3
  trimmed: ResidentTrimmedPopulationSessionV1
    population session
    trim native map/count/event owner
    parent feature-store/context/stream owner
    sealed schema owner
    full-discovery admission owner
  generation owner
  persistent scoring/archive owner
  exact boxed enqueue receipts
  terminal completion lease
```

There is no standalone public `PopulationSession` route, raw map getter, raw
archive handle, or detachable CUDA event. Rust may expose bounded summaries of
counts and identities only.

The native begin call must validate that the trim-ready event, population
stream, generation stream, scoring/archive stream, device, context, pool,
full-workspace identity and parent owner all name the same admitted run. It
waits for the trim-ready event on that stream before any Search kernel.

### Completion and cleanup

The terminal lease retains the entire trimmed carrier until the Search
completion event proves every consumer is done. On a clean terminal result,
release is enqueued in dependency order:

1. scoring/archive arena;
2. generation arena;
3. population evaluator/session workspace;
4. trim native map/count/event arena;
5. retained parent import, schema and full-admission Rust owners.

Rust owners are disarmed or dropped only after native acknowledges their exact
release/tombstone state. `NotReady`, an unknown asynchronous CUDA outcome, or
an unproven event retains or deliberately leaks the entire composite owner;
none of its pieces may be retried or reused separately. A device semantic
fault after exact event proof may run the normal checked cleanup while keeping
generation, store and archive count uncommitted.

## Typed split transaction

The existing one-shot composite is replaced for this versioned path by a
move-only generation chain with ranked and archive-staged intermediate states.
Transition methods enqueue on one admitted stream and do not wait for device
completion. The chain may enqueue the next generation immediately because
same-stream order preserves the device dependency; Rust records only the
planned ordinal and never treats the generation as committed before terminal
proof.

### 1. Active to ranked

`enqueue_score_and_rank_v3` consumes `ResidentSearchGenerationChainV3` and
returns `ResidentSearchRankEnqueuedV3`. The initial chain is minted by combined
admission; every later chain is returned by the prior generation enqueue.

It performs evaluation import, population signatures, all-finite objective
scoring, exact kNN, normalization/blending and deterministic rank. It does not
write archive records, select parents, create offspring, rotate stores, copy a
terminal receipt or record the terminal event.

The private fixed-width ranked receipt binds the original boxed receipt
address, run token, generation, store epoch, committed archive snapshot count,
rank-semantics identity and same-stream enqueue count. It carries no raw device
pointer.

### 2. Ranked to archive-staged

`enqueue_stage_archive_from_rank_v3` consumes the ranked state and returns
`ResidentSearchArchiveStagedV3`.

It reads the exact ranked population order before any store rotation. Eligible
unique records are copied into unused tail slots beginning at the committed
archive count. The committed count does not change. The staged receipt binds
the ranked receipt, staged count, target archive count and target commit word.

### 3. Archive-staged to next-generation-enqueued

`enqueue_evolve_and_publish_v3` consumes the staged state. It reuses the
existing selection, crossover, mutation and offspring full-gene dedup kernels.
After all producers have written complete sentinel-safe ranges, one final
single-thread device kernel either publishes both the offspring store and the
staged archive or publishes neither.

The method returns `ResidentSearchGenerationChainV3`, still owning the whole
run and binding the planned next ordinal to the exact prior staged receipt. It
does not enqueue a D2H copy, record a completion event or inspect a device
result. The next generation can consume that chain immediately on the same
stream. This preserves the per-generation ordering:

```text
score -> exact kNN -> rank -> stage archive tail -> evolve/dedup
      -> combined commit -> next generation on the same stream
```

After the final planned generation, a separate
`enqueue_terminal_seal_v3` consumes the chain and enqueues exactly one compact
terminal D2H followed by exactly one completion event. The resulting
`ResidentSearchTerminalPendingV3` is the only state that may be queried by the
host. Thus three or 20,000 generations still have zero intermediate D2H and
zero intermediate synchronization.

### Single publication authority

Generation/store and archive count are not two independently readable control
fields. The current-config V3 path packs them into one 64-bit commit word:

| Bits | Meaning | Preflight bound |
| --- | --- | --- |
| `0` | current store index | two stores |
| `1..16` | generation index | at most `65_535` |
| `17..32` | committed archive count | at most `65_535` |
| `33..63` | commit epoch | at most `2^31 - 1` |

Current values `20_000` and `50_000` fit. Any future generic configuration
outside those bounds is rejected before allocation.

The final kernel completes tail and offspring writes, executes the required
device fence, and publishes the one commit word with a 64-bit atomic exchange.
Every next-generation consumer decodes the same word; mirrored seal fields are
diagnostics, not authority. A fault leaves the word unchanged, latches the
device fault authority and makes staged tail bytes unreachable. Later queued
kernels total-write deterministic sentinel inputs but cannot publish; terminal
cleanup retires the composite owner instead of reusing it.

## Exact novelty semantics

### Neighbor set and timing

For candidate `i` in generation `g`, neighbors are:

- every current population slot except the exact slot `(current, i)`;
- every committed archive slot in `[0, archive_count_at_start_of_g)`.

Entries staged during `g` are not visible until the combined commit and first
participate in `g + 1`. An archive copy identical to the current gene is still
a valid zero-distance neighbor. Other identical current genes are also valid
zero-distance neighbors; self-exclusion is by source and ordinal, never by
content identity.

When fewer than `K` neighbors exist, the denominator is
`min(K, available_neighbors)`. Zero available neighbors is a device fault.
The shipped population always supplies at least 199.

### Jaccard representation and proven comparator

Each behavior is a `W=4` word bitset of selected feature indices. For a pair:

```text
intersection = popcount(left AND right)
union        = popcount(left OR right)
distance     = (union - intersection) / union
```

Zero union is a device fault. Ordering is performed on the integer fraction,
not rounded `f64` values.

The plan preflight computes with checked arithmetic:

```text
union_max = checked_mul(2, max_terms_per_gene)
cross_product_max = checked_mul(u64(union_max), u64(union_max))
```

It also requires `union_max <= u32::MAX`. For `M=16`, `union_max=32` and
`cross_product_max=1_024`. The receipt binds both values.

Device code validates `1 <= denominator <= union_max` and
`numerator <= denominator`, widens every operand to `u64` before multiplication,
and compares `lhs_num * rhs_den` with `rhs_num * lhs_den`. The preflight proof
therefore makes both products bounded; there is no unchecked same-width
multiplication. The CPU oracle uses checked `u128` multiplication independently.

Equal fractions use the total tie key:

```text
(gene_identity, source_kind, source_ordinal)
```

The selected `min(K, available)` keys are accumulated in that exact order and
then converted to `f64`. CPU/device comparison uses a documented tolerance for
the final division; it does not claim bitwise host-log parity.

### Archive admission and full-gene equality

All eleven metric values must be finite or the whole transaction faults.
Current-config admission then requires positive trade count and
`net > +0.0`. Candidates are considered in the versioned blended rank order
`(score descending, gene_identity, population_ordinal)`. Across generations,
the earlier committed generation is first seen. At capacity, only the earliest
eligible unique records fill the remaining slots.

Full-gene equality covers normalized term count, every fixed-stride feature
index and weight, long/short thresholds, TP/SL/volatility stop and SMC flags.
It excludes ephemeral gene identity, content hash, generation and metric fields.
Unused fixed-stride terms must be canonical zeroes.

The 64-bit full-gene hash is only an accelerator. On an equal hash the archive
performs exact full-gene equality:

- equal hash and equal full gene: duplicate, retain the first entry;
- equal hash and unequal full gene: record a collision counter and retain both;
- unequal hash: distinct without a full comparison.

No collision may silently collapse a gene or fault a mathematically valid run.

## Memory and computational bounds

### Work bounds

At full capacity, one generation has:

```text
P * ((P - 1) + A) = 200 * 50_199 = 10_039_800 distances
10_039_800 * W = 40_159_200 64-bit popcount words
```

The already sealed conservative 20,000-generation bounds are:

```text
200_796_000_000 distances
803_184_000_000 popcount words
```

If the archive starts empty and can add at most 200 entries per generation, the
tighter maximum is `199_541_000_000` distances and `798_164_000_000` words.
Admission keeps the larger sealed bound.

The necessary novelty-stage one-hour rates are therefore:

```text
55_776_667 complete distance/top-K items per second
223_106_667 popcount words per second
```

The algorithm is exact streaming fixed-K selection. It scans each available
neighbor once, keeps a deterministic block-local top 15, and materializes no
`P * (P + A)` distance matrix. Time is
`O(G * P * (P + A) * W)` plus bounded fixed-K selection; persistent auxiliary
space is `O(A * (gene + metric + signature + hash) + P * K)`.

### Exact replacement layout subtotal

All regions use the existing 256-byte scoring alignment. The Slice 2 V2
replacement subtotal is exactly `23_707_648` bytes:

| Region | Formula | Aligned bytes |
| --- | ---: | ---: |
| archive gene scalars | `50_000 * 72` | `3_600_128` |
| archive term indices | `50_000 * 16 * 8` | `6_400_000` |
| archive term weights | `50_000 * 16 * 8` | `6_400_000` |
| archive metric rows | `50_000 * 104` | `5_200_128` |
| archive signatures | `50_000 * 4 * 8` | `1_600_000` |
| archive hashes | `50_000 * 8` | `400_128` |
| current population signatures | `200 * 4 * 8` | `6_400` |
| novelty scores | `200 * 8` | `1_792` |
| exact top-K keys | `200 * 15 * 32` | `96_000` |
| admission flags | `200 * 4` | `1_024` |
| admission offsets | `200 * 8` | `1_792` |
| archive control and seal | bounded raw control | `256` |

The last three rows are the exact `3_072`-byte control subtotal:
`1_024 + 1_792 + 256`.

This is a replacement, not an additive arena layered over all V1 scoring
regions. It replaces the existing V1 population bitmap (`6_400`), novelty
scores (`1_792`) and scoring control (`256`), totaling `8_448` bytes. The net
increase is therefore exactly:

```text
23_707_648 - 8_448 = 23_699_200 bytes
```

Existing fitness-score, decision-key and CUB-scratch regions remain and are
added once by the native V2 layout query. Generation, evaluator, trim and
terminal allocations are separate fields of the full combined admission. The
query returns every aligned component and the final total; Rust recomputes and
compares each field before the first full Search allocation.

## Calibration and deadline claims

The calibration receipt is bound to the exact device UUID, context, stream,
memory pool, CUDA build/math identity, kernel semantics, `P=200`, `A=50_000`,
`W=4`, `K=15`, `M=16`, union/cross-product bounds and plan identity.

Calibration must use the production fixed-K kernel with an archive actually
prefilled to 50,000 valid deterministic signatures. Empty-archive execution,
inactive-tail sentinels, a smaller K, or a proxy popcount-only kernel cannot
mint the receipt. A bounded preflight allocation may be used and released at
this explicit pre-run boundary.

The receipt records elapsed CUDA-event time for at least one warmup and a
versioned number of measured full-capacity iterations, complete distance items,
popcount words, fixed-K comparisons and output digest. Admission requires both
minimum rates above and rejects missing, zero, stale, foreign or under-rate
receipts before the first full Search allocation.

Calibration and admission expose typed, opaque receipts. The executable
negative matrix independently requires:

- an empty active archive, inactive-tail-only work or `archive_count != 50_000`
  cannot mint a calibration receipt;
- `K != 15`, including the exact `K=14` smaller-neighbor control, cannot mint
  the current-config receipt;
- a popcount-only/proxy kernel or a foreign kernel-semantics identity cannot
  mint it even if its measured rate is high;
- a receipt from a stale CUDA build/math identity, another device UUID,
  context, stream or memory pool is rejected by admission;
- independent `P`, `A`, `W` and `M` shape mutations are rejected; and
- a correctly bound but deliberately under-rate receipt is rejected.

Every rejection asserts the exact typed stage/reason and independently observes
zero generation, scoring and archive allocations. Tests mutate opaque receipt
fields only from a crate-private fixture module; no production caller gains a
constructor or raw field access.

This is only a necessary novelty-stage lower bound. Passing it does not prove
that evaluation, scoring reductions, GA, archive staging, launch overhead and
terminal work all fit within one hour. Headless execution remains fail-closed
until a later combined deadline receipt sums conservative bounds for every
stage and proves the entire current-config run fits `3_600_000 ms`.

`ResidentArchiveKnnCalibrationReceiptV2` and the later
`FullResidentDiscoveryDeadlineReceiptV1` are distinct opaque types and distinct
run-identity domains. Slice 2 never implements a conversion between them. A
compile-fail contract rejects passing the novelty receipt where the full
deadline receipt is required, and the prepared-native behavioral invariant
passes a valid novelty receipt while omitting the full deadline receipt and
still requires fail-before-Search-allocation. Thus a novelty benchmark cannot
open headless execution by type confusion or by a readiness-bit shortcut.

## Independent no-boundary evidence

Production counters are necessary but cannot validate themselves. The real RTX
sequence therefore also links a test-only CUDA interception translation unit
around:

- synchronous and stream-ordered allocation entry points reached by the
  combined Search admission;
- runtime and driver D2H copies, synchronous and asynchronous;
- stream, event, context and device synchronization calls.

The intercept scope begins after the one-time trim/Search admission and ends at
the terminal event. It records call kind, byte count and phase independently of
the Search receipt. The multi-generation oracle requires zero D2H and zero
synchronization inside that scope. At the terminal boundary it permits exactly
one bounded compact receipt D2H after the combined commit and before the event;
no gene, metric, signature or archive payload is permitted. Production receipt
counters must independently agree with the interception log.

The test binary link-wraps the exact runtime and driver symbols referenced by
the production native objects. The interception state lives only in the
test-only translation unit; production kernels, owners and receipt writers have
no symbol, pointer or callback through which they can reset it. The runner
records the wrapped-symbol manifest and fails if a production object resolves a
covered call to an unwrapped symbol or an unexpected CUDA transfer/sync symbol.

Zero calls are accepted only after a non-vacuity handshake in the same process.
Before the measured scope, the test issues one bounded known allocation followed
by its checked release through each wrapped allocator entry point, one bounded
known D2H and one known synchronization through each runtime/driver API family
that the production wrapper can reach. The interceptor must record the nonce,
process/thread identity, exact symbol, direction, byte count and successful
return for every control call. It then seals a fresh measurement epoch; Search
cannot reset or write that epoch. A separate child run with the interposer
disabled, a missing symbol hook, a wrong PID/nonce or a dropped control record
must be rejected even if its reported measured counters are all zero. Controls
occur outside the Search timing and zero-boundary scope.

The RTX oracle does not evade that boundary to inspect arrays. Before the
intercept scope, the CPU oracle creates the deterministic fixture and its
expected per-generation neighbor, novelty, rank and archive digests. A
test-only device validator receives those expected values by H2D, follows each
combined commit on the admitted stream, checks exact integer/order fields and
the documented novelty tolerance, and latches only bounded mismatch bits and
digests into the terminal seal. The single compact D2H returns those results.
The validator has no production symbol and cannot authorize admission; the
production kernels and independent API intercept remain separate authorities.

## Executable behavioral RED plan

REDs are added before production. Test-only CPU oracles are not production
fallbacks and use a named ChaCha8 seed/draw/order contract.

### R1: exact current-config plan and layout

Path: `crates/neoethos-search/src/gpu_resident_current_config_plan_v1_tests.rs`

Assertions:

- exact current dimensions, work bounds, rational bounds and memory fields;
- novelty/archive/layout/calibration identities alter the run identity;
- checked overflow and any current-config extent drift fail before allocation.

### R2: exact rational kNN CPU oracle

Path: `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs`

Hand-computed fixtures cover current plus archive neighbors, exact self
exclusion, fewer than K, duplicate zero-distance neighbors, zero union, total
tie ordering and checked comparator limits. The oracle uses checked `u128`
cross products.

### R3: archive timing, ordering and cap

Same test module. Generation `g` cannot see admissions staged by `g`; generation
`g+1` must see them. Rank-ordered first-seen behavior, positive-trade/net gates,
cap-minus-one admission and duplicate handling are asserted exactly.

### R4: adversarial hash collision

A fixture forces two unequal full genes to the same 64-bit hash and proves both
are admitted. An exact duplicate under the same forced hash is admitted once.
The collision counter increments without setting the device fault word.

### R5: typed split transaction

Path:
`crates/neoethos-gpu-cuda/tests/resident_archive_knn_v2_transaction_contract.rs`

Behavioral fixture ABIs prove that rank state cannot publish, archive staging
cannot rotate, evolution cannot run without consuming the exact staged receipt,
and only the per-generation final commit changes the packed
store/generation/archive word. Chained generations queue without a D2H or
event. Only the separately consumed terminal-seal state may enqueue one compact
D2H and record one event after the last combined commit.

### R6: combined preallocation

Path: `crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs`

A combined admission with a missing/zero archive arena, mismatched aligned
field, insufficient reserve or foreign calibration returns a typed admission
error while generation/scoring/archive allocation counters all remain zero.
The exact valid receipt performs the declared allocation count once; no later
generation allocates.

### R7: executable move-only opacity

Path: public doctests in
`crates/neoethos-gpu-cuda/src/resident_archive_knn_v2.rs`, executed by the exact
`neoethos-gpu-cuda` doc-test target with `--no-default-features --features
cuda-device-fixtures`.

Public owner documentation contains paired compile-pass and `compile_fail`
doctests under the exact contract feature that exports the opaque public owner.
The compile-pass control imports and moves each public owner and receipt through
its supported typed API, proving that crate resolution, feature selection and
the positive surface are live without constructing a GPU resource. Every
negative fence is annotated `compile_fail,E####`, never bare `compile_fail`, so
an unrelated compilation error is not a pass. The exact authored diagnostics
cover eight negative fences:

- cloning a move-only owner fails with `E0599`, and a generic `Copy` bound fails
  with `E0277`;
- four separate trim-map, event, archive-pointer and population-field probes
  each fail with `E0616`;
- calling the private staged-receipt constructor fails with `E0624`, and
  directly constructing a ranked receipt with private fields fails with
  `E0451`.

The gate requires exactly one positive and eight negative doctest results. It
rejects an unrelated diagnostic, missing crate/feature, skipped or `no_run`
negative snippet, warning-only output, or a negative snippet that compiles. Thus
a broken test topology cannot masquerade as opacity. The normal R6 behavioral
test supplies the separate missing combined archive-preallocation RED. No
source-string/token assertion is accepted for either property.

### R8: transactional faults and cleanup

All eleven metric positions are independently replaced with NaN and infinity;
additional cases cover malformed/zero-union signatures, archive bounds,
receipt-address drift and comparator-range drift. Every case leaves the packed
commit word unchanged, publishes no staged tail, returns a compact terminal
fault, performs checked terminal cleanup and permits a fresh unrelated run.

### R9: real RTX behavior and calibration

Paths:

- `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_device_tests.rs`;
- test-only CUDA interception translation unit
  `crates/neoethos-gpu-cuda/native/resident_archive_knn_v2_cuda_intercept_test.cu`.

One self-authenticating sequence runs:

1. `resident_archive_knn_v2_calibration_rejects_nonrepresentative_receipts_on_real_cuda`:
   exact production fixed-K calibration with a genuinely prefilled 50,000-entry
   archive, plus executable rejection of empty/inactive-tail archive, `K=14`,
   proxy/popcount kernel and deliberately under-rate controls;
2. `resident_archive_knn_v2_admission_rejects_stale_foreign_or_shape_drift_receipts_before_allocation`:
   current-capacity allocation admission, including executable rejection of a
   stale CUDA build/math identity, foreign UUID/context/stream/pool receipt and
   independent `P`, `A`, `W` and `M` shape drift, with zero Search allocation
   deltas on every rejection;
3. at least three resident generations compared to the independent CPU oracle
   by the preloaded test-only device validator for neighbor identities, novelty
   values, rank, archive content/count and `g+1` visibility;
4. duplicate, adversarial collision, cap and all fault/cleanup cases;
5. interception assertions for zero intermediate D2H/synchronization and one
   compact terminal D2H, accepted only after the positive-control handshake has
   observed every wrapped runtime/driver family and an interposer-disabled child
   has been rejected.

The separately classified prepared-headless GREEN invariant below runs before
and after this sequence. It is evidence in the final receipt, but it is never an
R9 RED and a failure stops the sequence rather than authorizing implementation.

The device sequence records GPU UUID/name/compute capability/memory, exact
command and environment, source and binary hashes, receipt/counter values and
exit status. It runs once after an independent source review says safe to run.

## Continuously GREEN invariants

Readiness and prepared-headless refusal are guards, not REDs. They run before
RED capture, after every implementation step, and after RTX validation.

The existing exact target is
`crates/neoethos-gpu-cuda/tests/resident_search_generation_v2_production_contract.rs`,
Cargo test target `resident_search_generation_v2_production_contract`, function
`h_implementation_patch_keeps_all_five_unproven_readiness_facts_false`. It must
continue to assert the source truth: `exact_generation_semantics`,
`device_resident_generation_advance`, `immutable_scenario_admission`,
`whole_workspace_preallocated`, `unified_device_fault_authority`,
`terminal_cleanup_lease` and `production_ready` are false, while
`device_owned_search_control` and `native_bridge_production_sealed` are true.
Its exact focused invocation is:

```text
cargo test -p neoethos-gpu-cuda --no-default-features --features cuda --test resident_search_generation_v2_production_contract h_implementation_patch_keeps_all_five_unproven_readiness_facts_false -- --exact
```

Slice 2 also adds the GREEN behavioral target
`crates/neoethos-search/src/prepared_discovery_run_input_v3/resident_slice2_headless_invariant_tests.rs`,
function
`resident_slice2_valid_novelty_receipt_still_refuses_prepared_headless_before_search_allocation`.
It is a descendant test module, so it calls the private
`run_native_cuda_prepared_discovery_v3` entry directly rather than accepting a
public surrogate. The fixture first creates the real current-config plan and
whole `ResidentTrimmedPopulationSessionV1`, then supplies a correctly sealed
novelty calibration receipt but no `FullResidentDiscoveryDeadlineReceiptV1`.
Only after trim setup does it open an independent Search-allocation interception
epoch. The call must reach the private entry, return the exact missing-full-run-
deadline/readiness error, leave the carrier under its checked cleanup lease, and
show zero combined-admission calls and zero generation, scoring or archive
allocation deltas in that epoch. The test also rejects CPU Search, host
materialization and post-trim Search-generation progress. The exact existing
`gpu_native_trim_prefilter` progress event is allowed and asserted. Trim
allocations are explicitly outside the measured epoch, so the assertion is
precisely
fail-before-**Search**-allocation rather than a vacuous zero-allocation claim.

This target is registered only under the dedicated
`resident-search-slice2-device-fixtures` feature, which includes
`gpu-b-native` and `neoethos-gpu-cuda/cuda-device-fixtures` and is absent from
default, application and production feature closures. Its positive control
proves that the allocation interceptor records a known test allocation in the
same process before the epoch is sealed; disabled or missing interception makes
the invariant test fail. No full archive or terminal portfolio projection is
added by Slice 2.

Its exact focused invocation on the admitted RTX runner is:

```text
NEOETHOS_REQUIRE_GPU=1 cargo test -p neoethos-search --no-default-features --features resident-search-slice2-device-fixtures --lib prepared_discovery_run_input_v3::resident_slice2_headless_invariant_tests::resident_slice2_valid_novelty_receipt_still_refuses_prepared_headless_before_search_allocation -- --exact --nocapture
```

This second invariant is introduced only when the typed novelty calibration
receipt and its crate-private fixture constructor exist. It must be GREEN in the
same commit that first introduces those types; it is never captured as RED.

## Planned production files after design approval

The first implementation may touch only the bounded archive/transaction seam:

- new `native/resident_archive_knn_v2_abi.cuh` and
  `native/resident_archive_knn_v2.cu`;
- new `src/resident_archive_knn_v2.rs` plus focused tests;
- versioned combined admission and split-transaction additions in
  `resident_search_generation_v2_abi.cuh`, `resident_generation_v1.cu`,
  `resident_scoring_novelty_v1.cu`, `resident_scoring_v2.rs` and
  `resident_search_v2.rs`;
- the crate-private whole-carrier consumer in
  `resident_trim_prefilter_v1.rs`;
- build registration and focused contract/device test modules, including the
  test-only prepared-headless GREEN invariant module and its non-production
  fixture feature in `crates/neoethos-search/Cargo.toml`, plus only the
  `#[cfg(test)]` child-module registration in
  `prepared_discovery_run_input_v3.rs`.

The old current-only mean-Jaccard path is not modified into a look-alike. It
remains unreachable from current-config production and may be removed only
after the versioned replacement is fully verified.

## Implementation and review gates

1. Run the existing exact readiness GREEN invariant, then land R1-R9 and capture
   pure RED while that invariant continues to pass.
2. Implement checked ABI/layout/query and exact calibration receipt; in the same
   boundary add the prepared-headless invariant and require its first run GREEN.
3. Implement whole-trim-carrier ownership and transactional create/unwind.
4. Implement signatures, exact fixed-K and collision-safe archive equality.
5. Implement the ranked/staged/evolve states and packed combined commit.
6. Run both GREEN invariants, focused contracts and the exact no-link feature
   gates only.
7. Freeze source for independent lifecycle/math/source-contract review.
8. Run the single authorized RTX sequence, including both GREEN invariants,
   only after a safe-to-run verdict.
9. Freeze bounded Slice 2 evidence without changing readiness or headless routing.
