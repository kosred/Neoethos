# Resident Search Slice 2: permanent archive and exact kNN novelty

Status: design and executable RED plan only; production implementation is not
authorized by this document.

Version: 8

Authoritative base: `7824e191c04b4eb78e547728ad7cdb78f915a2af`

Branch: `codex/resident-search-novelty-slice2`

Version 8 supersedes the version-7 design at commit
`492c3eaa61c0709e381c2359c2b249558118d4b6`. The version-2 through version-7
manifests remain immutable historical receipts; the version-8 manifest alone
binds this corrected document and its R6 implementation plan. The prerequisite
test-only cfg correction at `4837ff9b06dd6df8d276ba83aed8fa3ed3e52feb`
does not change this design authority or authorize production implementation.

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
- Cargo's feature reference governs the empty, non-default host-contract
  feature and proves that its feature array activates no optional dependency:
  <https://doc.rust-lang.org/cargo/reference/features.html>.
- Cargo's target reference governs the library/build-script separation used by
  the host-contract verification topology:
  <https://doc.rust-lang.org/cargo/reference/cargo-targets.html>.
- The CUDA Runtime API memory-management contract governs the exact
  `cudaHostAlloc(..., cudaHostAllocPortable)` terminal allocation:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>.
- The CUDA Runtime API stream-ordered allocator contract governs the two exact
  `cudaMallocAsync` device allocations:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>.
  Same-stream lifetime, terminal-event proof and the prohibition on host access
  to an in-flight D2H destination additionally follow:
  <https://docs.nvidia.com/cuda/cuda-program-guide/04-special-topics/stream-ordered-memory-allocation.html>
  and
  <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html>.
- NVIDIA CCCL determinism guidance does not by itself promise the Slice 2 total
  order; therefore the fixed-K and rank tie keys are explicit versioned inputs:
  <https://nvidia.github.io/cccl/unstable/cccl/determinism.html>.

## Review corrections incorporated through version 8

| Finding | Versioned decision |
| --- | --- |
| Chat-only freeze was not reviewable | This tracked document and its tracked SHA-256 manifest are the sole design authority. |
| Ownership omitted trim lifetimes | The Search owner consumes and retains the whole `ResidentTrimmedPopulationSessionV1`, including its native map, ready event, parent import, schema and full-admission owners. |
| Rational comparison could overflow | The comparator widens to `u64` only after a checked host preflight proves the maximum cross product; device inputs are range-validated against that receipt. |
| Existing composite cannot insert post-rank/pre-rotation | A three-state typed transaction separates score/rank, archive staging, and evolution/publication. |
| Readiness was listed as a RED | Unchanged readiness is a continuously GREEN invariant, not a missing behavior. |
| Opacity was a source-string assertion | Public opacity and move-only ownership use the executable compiler-UI contract sealed in R7; a missing logical archive subreceipt inside the combined arena is a behavioral pre-allocation RED. |
| Hash collision behavior was unspecified | Hashes are accelerators only; equality falls back to the exact full gene and unequal colliding genes both survive. |
| `23_707_648` bytes was ambiguous | It replaces three V1 scoring regions and is a net `23_699_200`-byte increase before the unchanged fitness/decision/CUB regions. |
| Calibration could benchmark an easy case | The exact production fixed-K kernel is benchmarked with `P=200`, a prefilled `A=50_000` archive, `W=4`, and `K=15`. |
| Production counters were self-reported | RTX acceptance independently intercepts every D2H and synchronization API in addition to checking production counters. |
| One-hour claim was too broad | Slice 2 proves a necessary novelty-stage lower bound only; a later combined deadline gate must cover every Search stage. |
| GREEN readiness/headless assertions named no executable authority | Version 3 binds the existing behavioral readiness function in `resident_search_generation_v2_production_contract` and adds a prepared-native behavioral refusal test with an independent post-trim allocation scope. |
| Calibration rejection covered only exact and under-rate receipts | Version 3 executes empty-archive, smaller-K, proxy-kernel, stale-build, foreign-device and shape-drift rejections and proves a valid novelty-only receipt cannot satisfy the separate headless deadline/readiness gate. |
| Opacity/interception tests could pass vacuously | R7 requires compiler JSON, tracked stderr and a compile-pass control; the independent CUDA interception layer requires live positive controls plus an interposer-disabled rejection. |
| The interception epoch began after Search admission | Version 4 seals the epoch after the whole trim carrier exists but before combined Search admission, accounts for every declared admission allocation, classifies the one trim-ready stream wait as an asynchronous device dependency, and keeps the boundary live through the one terminal projection. |
| Cross-source ties had no wire encoding | Version 4 fixes `Current = 0` and `Archive = 1`, seals disjoint ordinal domains, binds them into run identity, and adds a cutoff tie fixture with equal distance and equal gene identity across sources. |
| Final novelty floating-point semantics were open | Version 4 fixes rational selection, integer-to-binary64 conversion, division and accumulation order, strict round-to-nearest-even math mode, and an absolute/relative/ULP acceptance tuple bound into every plan, calibration receipt and run identity. |
| Calibration negatives did not independently exercise all sealed bounds | Version 4 adds `A=49_999`, `K=16`, and orthogonal distance-rate/popcount-rate failures, with exact typed refusals and zero combined-Search allocations. |
| R7 and R8 lacked executable topology/count authority | Version 4 replaces doctest counting with compiler JSON plus tracked stderr for one positive and nine negative UI fixtures, seals the exact feature/cfg topology, and names the R8 files, tests and fixture counts. |
| Calibration capacity and active population were conflated | Version 5 independently seals capacity `A=50_000` and `calibration_active_count=50_000`, with a one-inactive-tail `49_999` active-count control distinct from the retained `A=49_999` capacity control. |
| The plan/deadline-marker ownership file was absent from may-touch scope | Version 5 names `gpu_resident_current_config_plan_v1.rs` and fixes the declaration-only deadline marker's exact source location without implementing a deadline receipt. |
| The nested UI fixture package could join the parent workspace or select the wrong target | Version 5 requires an empty `[workspace]`, `autobins=false`, ten explicit `[[bin]]` mappings and one exact `--bin` selection per compiler invocation. |
| The disabled-interposer child could be a weaker/different run | Version 5 re-spawns the exact current executable/test/fixture/hashes/argv with only the test-only interposer mode changed, and requires the exact typed handshake error and sealed non-zero exit. |
| The R6 physical allocation ledger was implied rather than enumerated | Version 6 freezes exactly three physical calls in order: one portable 104-byte terminal host receipt, one generation arena and one combined scoring/archive arena. There is no separate archive allocation; event creation remains R9 evidence and is outside the allocator ledger. |
| Allocator reserve and full-workspace authority were conflated | Version 6 separates allocator/context headroom, full-workspace authority, retained pre-Search workspace and remaining Search allocation bytes, then requires checked partition and availability equations before native create. |
| An R6 receipt could self-report a plausible ledger | Version 6 requires a separate recorder facade on the actual Rust admission path, exact cardinality before element comparison and mutation controls for missing, extra, reordered and field-drifted actual calls. |
| R6 could fail to compile or fan out into unrelated production changes | Version 6 permits only a test-only stub/child registration in `resident_search_v2.rs` plus `resident_search_v2_tests.rs`, and freezes exactly five warning-clean runtime `ImplementationPending` failures. |
| The v6 RED command activated CUDA on a host with no CUDA installation | Version 7 adds the empty non-default `resident-search-slice2-host-contract` feature, which activates neither `cust` nor `vector-ta` and leaves the CUDA feature/build semantics unchanged. |
| A host-only copy could drift from the future production validator | Version 7 places DTOs, facade, move-only owner, pending seam and later validator in one private shared `resident_search_slice2_admission_v2` module. Host tests and future CUDA production use that same authority; no mirror is permitted. |
| The v6 plan put test declarations inside the CUDA-gated production module | Version 7 keeps `resident_search_v2.rs` CUDA-gated and unchanged in R6 RED. The shared module owns the seam and registers its child tests only under the host-contract test feature. |
| An empty feature could later acquire hidden CUDA edges | Version 7 adds Cargo-metadata, build-log, default/CUDA preservation and dual-feature cfg ratchets. Actual all-features compilation remains a later CUDA-toolchain gate rather than a false local claim. |
| The R6 generation fixture mixed incompatible population sizes and invented a retained workspace | Version 8 derives every deterministic generation component from `P=200`, `M=16` and the current native layout formulas. Only the host-contract CUB scratch value is opaque; the CUDA implementation must use the real runtime query. |
| Caller totals could be trusted without recomputation | Version 8 adds typed subtotal/receipt mismatch axes, checked-add overflow axes and same-components/one-total-only plus overflow-precedence negatives for the replacement, generation and scoring/archive totals. |
| Reserve identities could self-report, alias or detach bytes from their source receipts | Version 8 separates observed authorities from an opaque, by-value trusted capability with private fields, independently trusted expected bytes/calibration/provenance and no raw constructor. It binds all five byte facts to the full CUDA/run/workspace tuple and adds the coarse four-way alias, all six one-pair-only aliases, and coordinated byte/binding/plain-value negatives. |
| The allocation recorder trusted a declared symbol and did not order native create | Version 8 derives the symbol and method count from the invoked facade method and records native create plus allocations in one phase-bearing chronology. Wrong-method and allocate-before-create implementations cannot reproduce the expected trace. |
| Identity mutations exercised only one low bit and UUID byte zero | Version 8 requires low- and high-bit controls for every `u64` identity and both byte zero and the tail byte for every UUID equality boundary. |
| The verbose host proof contained cap-linted dependency warnings | Version 8 calls only the `neoethos-gpu-cuda` crate warning-clean under `-Dwarnings`; it preserves and reports all dependency warning lines instead of claiming a globally warning-free log. |

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

### Version 8 local host-toolchain evidence

The v6 focused command selected `--features cuda`. On the current Windows host
it stopped during the `cust_raw`/`find_cuda_helper` build with the exact terminal
diagnostic `Could not find a cuda installation`; no R6 test binary was produced.
At that observation point `CUDA_PATH` and `CUDA_ROOT` were absent and neither
`nvcc` nor `cuobjdump` resolved on `PATH`.

The crate's own `build.rs:626-652` independently confirms the topology: when
`CARGO_FEATURE_CUDA` is present it calls `resolve_cuda_build` before compiling
the host ABI, and that path resolves/runs `nvcc` and `cuobjdump`. There is no
`DOCS_RS` bypass. Therefore a command that enables `cuda` is not a truthful
host-only R6 contract command on this machine. This evidence says nothing about
the correctness of CUDA production code; it proves only that R6 needs a
dependency-empty host-contract feature to compile and execute its pure Rust RED.

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
enqueues exactly one `cudaStreamWaitEvent` or `cuStreamWaitEvent`, according to
the one symbol resolved by the linked production object, on the admitted Search
stream before any Search kernel. This is an asynchronous device dependency: it
does not query or wait for the event on the host.

The independent interception epoch is sealed only after the complete trimmed
carrier and its ready-event identity exist and after the test validator's
bounded expected-value H2D upload, but before the first combined Search
admission call. Consequently all admission allocations are inside the epoch.
The layout receipt declares the exact three-entry physical allocation ledger
frozen below. The independent recorder must observe the same cardinality,
allocator symbol, order, category, requested/aligned byte count, alignment,
flags, stream and resolved pool; equality of a shared prefix is insufficient.
The epoch does not close at event record. It remains live through successful
nonblocking query of the one terminal event and host projection of the one
compact terminal receipt.

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

`source_kind` is a versioned one-byte wire value: `Current = 0` and
`Archive = 1`; every other value is a device fault. `Current` ordinals are
sealed unsigned values in `[0, P)`. `Archive` ordinals are sealed unsigned
values in `[0, archive_count_at_start_of_g)`. They are deliberately disjoint
domains even when their numeric values match. Self-exclusion applies only to
`(Current, candidate_ordinal)`; it never excludes an archive record. The
encoding, both ordinal-domain rules and the tie-key order have separate
semantics identities and all three feed the current-config plan identity,
calibration receipt and final run identity.

R2 and R9 contain an exact cutoff fixture with fourteen strictly nearer
neighbors followed by one current and one archive neighbor having the same
distance fraction and the same `gene_identity`. With `K=15`, the current
neighbor must occupy slot fifteen because `Current(0) < Archive(1)`; swapping
the source encoding must alter run identity and the old receipt must be
rejected.

### Sealed binary64 mean

Rational numerators and denominators remain integers through comparison and
top-K selection. After the selected keys have been sorted by the complete
neighbor order `(rational_distance, gene_identity, source_kind,
source_ordinal)`, the device computes their mean in this sole allowed order,
where every operation rounds once to IEEE-754 binary64 round-to-nearest,
ties-to-even:

```text
q = min(K, available_neighbors)
sum = +0.0_f64
for j in 0..q in selected total-key order:
    n = exact_u32_to_f64(numerator[j])
    d = exact_u32_to_f64(denominator[j])
    term = rn_div(n, d)
    sum = rn_add(sum, term)
novelty = rn_div(sum, exact_u32_to_f64(q))
```

All integers are at most 32 (or 15 for `q`) and therefore convert exactly. The
CUDA implementation uses `__ddiv_rn` and `__dadd_rn`, is compiled without
`--use_fast_math` and with exact switches `--ftz=false --prec-div=true
--fmad=false`, and forbids reassociation. The complete compiler/flag digest is
bound as `binary64_rn_strict_v1`. No reduction tree, FMA, reciprocal
approximation, extended accumulator, conversion before top-K selection or
alternate order is equivalent.

The CPU oracle independently selects fractions with checked `u128` cross
products, verifies their exact checked rational sum, and then emulates the
sealed binary64 sequence above. Expected `+0.0` must match its exact bit pattern.
For every non-zero finite expected value, device acceptance requires all three
conditions simultaneously:

```text
absolute_error <= 2^-50
relative_error <= 2^-48
nonnegative_binary64_ulp_distance <= 4
```

ULP distance is the difference between the ordered `u64` encodings of two
non-negative finite values. NaN, infinity, negative zero, a sign difference or
failure of any one bound is a mismatch. The operation-sequence identity, math
mode identity and exact `(2^-50, 2^-48, 4 ULP)` policy are fields of the plan,
calibration receipt, validator receipt and run identity; none is a test-runner
constant or host-log convention.

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
added once by the native V2 layout query. The query returns every aligned
component and the final total; Rust recomputes and compares each field before
the first full Search allocation.

The R6 host contract uses one deterministic generation receipt derived from the
current `checked_physical_layout_v1` formulas at `P=200`, `M=16` and 256-byte
device alignment:

| Generation component | Host-contract bytes |
| --- | ---: |
| logical gene scalars | `14_592` |
| logical gene indices | `25_600` |
| logical gene weights | `25_600` |
| offspring store | `65_792` |
| metric rows | `20_992` |
| rank keys | `8_192` |
| selection state | `5_120` |
| dedup hashes/control/seal | `9_472` |
| CUB scratch | `65_536` |
| retained evaluation coverage | `256` |
| terminal device receipt | `256` |
| checked host-contract total | `241_408` |

The `65_536` CUB value is an opaque, deterministic host-contract fixture value,
not a claim about a particular CUDA/CUB query. All other rows are fixed by the
current formulas and ABI sizes. The later CUDA implementation must substitute
the actual same-stream runtime-query result, recompute the checked total and
bind the exact returned receipt; it may not require the device query to equal
the host fixture. The scoring/archive host fixture analogously uses fitness
`1_792`, decision keys `1_792`, opaque CUB scratch `65_536` and the exact
`23_707_648` replacement subtotal, for checked total `23_776_768`.

For each of the replacement subtotal, generation total and scoring/archive
total, validation first recomputes the component sum with `checked_add`. An
overflow returns its typed arithmetic error before any declared-total mismatch.
Only a successful sum is compared with the declared total. Each relation has a
negative that leaves every component byte-for-byte unchanged and changes only
the declared total, plus a separate overflow-precedence negative.

### Exact combined-admission physical allocation ledger

R6 freezes exactly three physical allocation calls, in this order:

| Ordinal | Exact symbol | Category | Requested bytes | Aligned bytes | Alignment | Flags | Stream | Resolved pool |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 0 | `cudaHostAlloc` | `TerminalHostReceipt` | `104` | `104` | `8` | `cudaHostAllocPortable` (`0x01`) | none | none |
| 1 | `cudaMallocAsync` | `GenerationArena` | `generation_receipt.total_device_bytes` | the same exact receipt total | `256` | `0` | exact admitted Search stream | exact admitted active pool |
| 2 | `cudaMallocAsync` | `ScoringArchiveArena` | `scoring_archive_receipt.total_device_bytes` | the same exact receipt total | `256` | `0` | exact admitted Search stream | exact admitted active pool |

`ScoringArchiveArena` is one physical allocation. Its checked receipt includes
the unchanged fitness-score, decision-key and CUB-scratch components exactly
once and all twelve aligned Slice 2 rows in the table above exactly once. There
is no fourth physical allocation and specifically no standalone archive arena.
On CUDA, the generation receipt total and scoring/archive receipt total are
exact runtime-query facts, not estimates or independently rounded copies. In
the host RED they are the checked deterministic fixture totals above; only the
two explicitly labelled CUB inputs are opaque test values.

The legacy phrase "missing/zero archive arena" retained in the R6 test and error
names means a missing or zero logical archive subreceipt inside this one
`ScoringArchiveArena`; it never authorizes an `ArchiveOnlyArena` allocation.

Event creation is not an allocation-ledger entry. The terminal event and its
record/query behavior remain exclusively within the R9 CUDA interception and
completion proof. R6 neither creates an event category nor treats an event API
call as an allocator call.

### Exact reserve and workspace authority

The combined admission carries five observed byte facts and consumes a
separately typed, opaque `ResidentSearchSlice2TrustedReserveSealV2` by value. An
observed authority never carries its own expected value. The seal privately
owns the trusted reserve set, expected calibration, common binding tuple and
full-workspace/post-trim provenance. No ordinary crate caller can construct,
clone, copy or default this capability from raw values.

The R6 host fixture uses these canonical reserve bytes, with no hidden slack:

| Canonical fact | Bytes |
| --- | ---: |
| requested device sum | `24_018_176` |
| allocator-context headroom | `8_388_608` |
| retained pre-Search workspace | `67_108_864` |
| remaining Search allocation after trim | `24_018_176` |
| full-workspace authority | `91_127_040` |
| same-context free | `32_406_784` |

Thus retained plus remaining equals full workspace, and both independent device
budgets are exact fits in the canonical positive fixture.

The canonical authority identities are headroom `0x3101`, full workspace
`0x3202`, retained `0x3303`, remaining `0x3404` and same-context free `0x3505`.
`PAIR_ALIAS_IDENTITY_V2 = 0xA11A_5E00_0000_0001` is distinct from all five.
The original four reserve/workspace authority identities are pairwise distinct;
none is an alias for another:

- `allocator_context_headroom_bytes`: bytes intentionally left free in the
  same CUDA context for allocator/runtime operation;
- `full_workspace_authority_bytes`: the sealed full-run workspace authority;
- `retained_pre_search_workspace_bytes`: bytes already retained by data,
  evaluator, population and trim owners when the Search allocation epoch opens;
- `remaining_search_allocation_bytes_after_trim`: the exact device budget left
  for the two Search device arenas after trim.

The fifth fact is `same_context_free_bytes`, obtained from the same admitted
primary CUDA context. It also has independently trusted expected bytes and the
same complete binding, but it is not counted as one of the four sealed
reserve/workspace receipt identities in the pairwise-distinct control.

Every observed authority binds exactly this tuple:

```text
device_uuid[16]
primary_context_identity: u64
search_stream_identity: u64
active_pool_identity: u64
run_identity: u64
full_workspace_receipt_identity: u64
post_trim_receipt_identity: u64
authority_identity: u64
```

The opaque seal owns the expected bytes and expected tuple. The eventual CUDA
minter is deferred beyond R6 and must consume the actual opaque calibration,
full-workspace and post-trim authorities; it may not accept raw bytes, raw
bindings or the observed request. The seal is a separate by-value admission
argument, not nested expected fields that the request can self-report. Every
single-authority or single-binding-axis negative requires its own exact typed
error but makes no mutual precedence claim against the other single-axis cases.
Only these simultaneous-mismatch boundaries are frozen: `AllocatorContextHeadroom`
wins over the other authorities in the plain-graph control;
`FullWorkspaceAuthority` wins over `RetainedPreSearchWorkspace` in the
workspace-shift control; headroom bytes win over a headroom binding mismatch in
the plain-graph control; and headroom `DeviceUuid` wins over the other headroom
binding axes in the all-binding control.

The all-alias negative sets the four observed and trusted authority identities
to the same nonzero value so pairwise observed/expected equality alone would
pass; it must fail the typed distinctness relation. To make that negative
reachable without weakening the capability, the cfg-gated descendant test
first inspects a pristine newly minted seal, then directly mutates exactly the
four private trusted
`expected_binding.authority_identity` fields and the four observed counterparts
to one common nonzero value. It changes no other field. Descendant privacy makes
this test-only fault injection legal; production siblings cannot do it, and no
second constructor or raw accessor is added.

Six additional pair-only controls cover every unordered pair among headroom
(`H`), full workspace (`F`), retained (`R`) and remaining (`Rem`): `HF`, `HR`,
`HRem`, `FR`, `FRem`, `RRem`. For each control the descendant first inspects a
pristine seal, then changes only that pair's trusted and observed
`authority_identity` fields to `PAIR_ALIAS_IDENTITY_V2 =
0xA11A_5E00_0000_0001`. The other two identities stay distinct and canonical.
Every case requires `FourReserveAuthorityIdentitiesDistinct`; a validator that
checks only one inequality or omits one member of the uniqueness set fails.

Coordinated negatives preserve the old equations while changing authority:

- retained and full-workspace observed bytes both move by one;
- allocator headroom and same-context-free observed bytes both move by one;
- every observed authority receives the same foreign-but-internally-consistent
  binding tuple while the independently trusted tuple stays unchanged;
- every observed authority and a child-local graph of plain expected-looking
  bytes/bindings move together, while the zero-argument minter still produces
  the unchanged opaque seal.

The first case must report `ReserveAuthorityBytesMismatch` for
`FullWorkspaceAuthority` because `AllocatorContextHeadroom` remains unchanged
and full workspace precedes retained. The second must report the same error for
`AllocatorContextHeadroom`. The third must report
`ReserveAuthorityBindingMismatch` for `AllocatorContextHeadroom` and
`DeviceUuid`. The fourth must report `ReserveAuthorityBytesMismatch` for
`AllocatorContextHeadroom`; the plain value graph is not a capability and is
never accepted by admission or the minter. All fail before native create. No
validator may infer authority merely because the arithmetic remains
self-consistent.

Relation, checked-arithmetic and budget fixtures that alter authority bytes
first inspect the pristine seal, then use descendant-only private-field access
to update exactly these affected trusted expected bytes in lockstep with the
observed values:

- partition mismatch: full workspace becomes `91_127_041`;
- `WorkspacePartitionAdd`: retained and remaining both become `u64::MAX`;
- `SameContextFreeMinusHeadroom`: same-context free becomes `0` and headroom
  becomes `1`;
- remaining-budget one-byte-short: remaining becomes `24_018_175` and full
  workspace becomes `91_127_039`;
- same-context-budget one-byte-short: same-context free becomes `32_406_783`.

`RequestedDeviceSumAdd` changes only generation/scoring receipt components and
needs no seal mutation. Canonical exact fit uses the untouched constants above.
Every case inspects the pristine seal before descendant mutation. This makes
authority equality pass so the intended relation/arithmetic/budget error is
reachable. No binding/provenance field, constructor or accessor changes for
this fault injection.

Before native create or any allocator call, checked arithmetic must prove:

```text
retained_pre_search_workspace_bytes
  + remaining_search_allocation_bytes_after_trim
  == full_workspace_authority_bytes

requested_device_sum
  = generation_receipt.total_device_bytes
  + scoring_archive_receipt.total_device_bytes

requested_device_sum <= remaining_search_allocation_bytes_after_trim
requested_device_sum
  <= same_context_free_bytes - allocator_context_headroom_bytes
```

Every addition and subtraction above is checked; wrapping and saturating
arithmetic are forbidden. Exact fit on both inequalities is GREEN. A budget
that is one byte short on either independent inequality returns the
corresponding typed refusal before native create. The 104-byte host receipt is
governed by the host allocation ledger and is not added to
`requested_device_sum`.

### R6 independent admission recorder

The R6 test facade sits on the actual Rust combined-admission path. Host and
async allocation methods accept method-specific argument DTOs that contain no
symbol field. The recorder derives `CudaHostAlloc` from `cuda_host_alloc` and
`CudaMallocAsync` from `cuda_malloc_async`, counts those method invocations
separately and constructs the observed allocation row itself. Calling the wrong
method therefore cannot be hidden by passing a name that claims the other API.

The recorder owns one ordered chronology. Its first expected event is
`NativeCreate`, followed by the three derived allocation events. Each allocation
event records the phase at call time; the only accepted phase is
`NativeCreateBegun`, and the third allocation advances to
`AllocationsComplete`. Thus three allocations followed by native create cannot
match the expected trace even if the final counters and allocation rows look
plausible. The recorder does not accept the declared ledger, cannot copy a
receipt into observed state and cannot be backfilled after the fact. Its state
is private to the test implementation and has no append/setter API.

Every negative calls the real Rust admission API once, but validation completes
before native create: native-create, host-method, async-method, physical
allocator, generation-arena, scoring/archive-arena and forbidden archive-only
counts all remain zero, and both chronology and observed allocation ledger are
empty. The valid case requires chronology length `4`, with native create first,
then host-method count `1`, async-method count `2` and the exact three entries
above. After that admission it queues three generations and proves the phase,
chronology, complete ledger and every allocation counter remain byte-for-byte
unchanged.

## Calibration and deadline claims

The calibration receipt is bound to the exact device UUID, context, stream,
memory pool, CUDA build/math identity, kernel semantics, `P=200`, `A=50_000`,
`calibration_active_count=50_000`, `W=4`, `K=15`, `M=16`,
union/cross-product bounds, `Current=0`/`Archive=1` source encoding and ordinal
domains, binary64 operation sequence, `binary64_rn_strict_v1`, the exact
`(2^-50, 2^-48, 4 ULP)` policy and plan/run identity. Capacity and active count
are separate receipt fields and separate identity inputs.

Calibration must use the production fixed-K kernel with an archive actually
prefilled to 50,000 valid deterministic signatures, so every slot in
`[0, 50_000)` is active. The host-sealed calibration preflight owns and validates
both capacity and active count before any calibration allocation or kernel.
Only after those fields equal their exact current-config values may a bounded
preflight allocation be used and released at this explicit pre-run boundary.
Empty-archive execution, an inactive tail, a smaller K, or a proxy popcount-only
kernel cannot mint the receipt.

The receipt records elapsed CUDA-event time for at least one warmup and a
versioned number of measured full-capacity iterations, complete distance items,
popcount words, fixed-K comparisons and output digest. Admission requires both
minimum rates above and rejects missing, zero, stale, foreign or under-rate
receipts before the first full Search allocation.

Calibration and admission expose typed, opaque receipts. The executable
negative matrix independently requires:

- an empty active archive or inactive-tail-only work fails with
  `ResidentArchiveKnnCalibrationErrorV2::ArchiveActiveCountMismatch` during the
  host count preflight, before a calibration allocation, kernel or receipt and
  before every Search allocation;
- with capacity still exactly `A=50_000`, the independent
  `calibration_active_count=49_999` fixture marks slots `[0, 49_999)` (ordinals
  `0..=49_998`) as valid active records and slot `49_999` as inactive. It must
  fail with
  `ResidentArchiveKnnCalibrationErrorV2::ArchiveActiveCountMismatch` during the
  same host preflight, with zero calibration allocations, zero calibration
  kernel launches, no receipt and zero Search allocations;
- `A=49_999` is an exact capacity control and fails with
  `ResidentArchiveKnnCalibrationErrorV2::ArchiveCapacityMismatch` before any
  calibration or Search allocation. This control supplies
  `calibration_active_count=49_999`, valid for its mutated capacity, and is not
  the `A=50_000`/active-count `49_999` fixture above; capacity validation has
  deterministic precedence over active-count validation;
- `K != 15`, including exact `K=14` and `K=16` controls, fails with
  `ResidentArchiveKnnCalibrationErrorV2::NeighborCountMismatch` before any
  calibration or Search allocation;
- a popcount-only/proxy kernel or a foreign kernel-semantics identity cannot
  mint it even if its measured rate is high;
- a receipt from a stale CUDA build/math identity, another device UUID,
  context, stream or memory pool is rejected by admission;
- independent `P`, `A`, `W` and `M` shape mutations are rejected; and
- a test-only sealed receipt with distance rate `55_776_666` while popcount
  rate remains at least `223_106_667` fails with
  `ResidentSearchAdmissionErrorV2::ArchiveKnnDistanceRateBelowMinimum`;
- a separate sealed receipt with popcount rate `223_106_666` while distance
  rate remains at least `55_776_667` fails with
  `ResidentSearchAdmissionErrorV2::ArchiveKnnPopcountRateBelowMinimum`.

Every rejection asserts the exact typed stage/reason and independently observes
zero combined-admission calls and zero generation, scoring and archive Search
allocations. Capacity, active-count, K and shape preflight refusals additionally
observe zero calibration allocations and zero calibration kernel launches. The
two rate controls derive from a real completed
calibration but are independently re-sealed only by a crate-private fixture;
their already released calibration scratch is not a Search allocation. Tests
mutate opaque receipt fields only from that fixture module; no production caller
gains a constructor or raw field access.

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

Slice 2 introduces only the public, uninhabited compile-contract marker
`FullResidentDiscoveryDeadlineReceiptV1` in
`crates/neoethos-search/src/gpu_resident_current_config_plan_v1.rs`, immediately
after the `impl SealedCurrentConfigResidentSearchPlanV1` block and before
`seal_current_config_resident_search_plan_v1`. Its sole field is private
`_not_minted_in_slice2: core::convert::Infallible`; it has no constructor,
sealer, trait conversion, receipt fields or production consumer. R7 obtains
typed expressions only through divergent fixture functions. A later slice must
replace this declaration-only marker with a separately reviewed, identity-bound
receipt and executable deadline proof. Merely naming the marker cannot change
readiness, headless routing or Search admission in Slice 2.

## Independent no-boundary evidence

Production counters are necessary but cannot validate themselves. The real RTX
sequence therefore also links a test-only CUDA interception translation unit
around:

- synchronous and stream-ordered allocation entry points reached by the
  combined Search admission;
- runtime and driver D2H copies, synchronous and asynchronous;
- stream, event, context and device synchronization calls.

The measured epoch begins after the whole trimmed carrier and trim-ready event
are sealed but before the first combined Search admission call. The expected
validator payload has already been uploaded H2D. The epoch records call kind,
symbol, byte count, stream/event identity and phase independently of the Search
receipt. Combined admission must perform exactly the declared allocation ledger
from the layout receipt; every allocator call and aligned byte count is observed
inside the epoch. Any undeclared allocation, missing declared allocation,
different symbol/order/size or any allocation after admission is a failure.

The trim-ready dependency is exactly one intercepted `cudaStreamWaitEvent` or
`cuStreamWaitEvent` on the admitted Search stream and exact sealed ready event.
It is classified as an asynchronous stream dependency, not host synchronization.
`cudaEventSynchronize`, `cudaStreamSynchronize`, `cudaDeviceSynchronize`,
`cuEventSynchronize`, `cuStreamSynchronize`, `cuCtxSynchronize` and equivalent
covered host-blocking calls remain forbidden.

From epoch seal through admission, the async trim-ready dependency and every
generation enqueue, the oracle requires zero D2H and zero host synchronization.
After the final combined commit, terminal seal may enqueue exactly one bounded
compact asynchronous D2H on the admitted stream and immediately record exactly
one terminal event. No gene, metric, signature or archive payload is permitted.
The host may only poll that exact event nonblockingly; it may not inspect the D2H
destination until the query returns success. The epoch remains live through
that success and the single bounded host projection, then closes. There is no
second D2H, event, host wait or synchronization at the boundary. Production
receipt counters, the declared admission ledger and the interception log must
all agree independently.

The test binary link-wraps the exact runtime and driver symbols referenced by
the production native objects. The interception state lives only in the
test-only translation unit; production kernels, owners and receipt writers have
no symbol, pointer or callback through which they can reset it. The runner
records the wrapped-symbol manifest and fails if a production object resolves a
covered call to an unwrapped symbol or an unexpected CUDA transfer/sync symbol.

Zero calls are accepted only after a non-vacuity handshake in the same process.
Before the measured epoch, the test issues one bounded known allocation followed
by its checked release through each wrapped allocator entry point, one bounded
known D2H and one known synchronization through each runtime/driver API family
that the production wrapper can reach. The interceptor must record the nonce,
process/thread identity, exact symbol, direction, byte count and successful
return for every control call. It then seals a fresh measurement epoch; Search
cannot reset or write that epoch.

The disabled-interposer negative is paired with the enabled control by a
supervisor. Both children are spawned from `std::env::current_exe()` and require
the same executable SHA-256, exact child test
`resident_archive_knn_v2_interceptor_spans_admission_to_terminal_projection_child`,
fixture seed and fixture digest, source/tree/design/plan/run/binary/wrapped-symbol
hashes, inherited environment and byte-for-byte argv:

```text
<current_exe> resident_archive_knn_v2_interceptor_spans_admission_to_terminal_projection_child --exact --nocapture
```

The sole permitted difference is the test-only environment value
`NEOETHOS_SLICE2_INTERPOSER_MODE_V1=enabled` versus `disabled`; it is recorded
outside argv and included in the supervisor receipt. The enabled child must pass
the complete control handshake and exit zero. The disabled child must stop
before calibration/Search admission, emit the exact bounded typed discriminant
`ResidentArchiveKnnInterceptionErrorV2::MissingInterposerControlHandshake` bound
to the same fixture/run hashes, and exit with the sealed test-only code
`MISSING_INTERPOSER_CONTROL_HANDSHAKE_EXIT_V1 = 86`. A CLI parse failure, absent
test, different executable/test/fixture/hash/argv, panic/exit 101, generic
non-zero exit or all-zero self-report is not this negative and must fail the
supervisor.

Missing-symbol, wrong PID/nonce and dropped-control-record negatives remain
separate exact fault modes of that same current executable and fixture and must
also be rejected before Search admission. Controls occur outside the Search
timing and measured epoch. The epoch counter can be sealed once but cannot be
reset; a second seal attempt is an exact rejection.

The RTX oracle does not evade that boundary to inspect arrays. Before the
measured epoch, the CPU oracle creates the deterministic fixture and its
expected per-generation neighbor, novelty, rank and archive digests. A
test-only device validator receives those expected values by H2D, follows each
combined commit on the admitted stream, checks exact integer/order fields and
the receipt-bound binary64 tolerance, and latches only bounded mismatch bits and
digests into the terminal seal. The single compact D2H returns those results.
The validator has no production symbol and cannot authorize admission; the
production kernels and independent API intercept remain separate authorities.

## Executable behavioral RED plan

REDs are added before production. Test-only CPU oracles are not production
fallbacks and use a named ChaCha8 seed/draw/order contract.

### R1: exact current-config plan and layout

Path: `crates/neoethos-search/src/gpu_resident_current_config_plan_v1_tests.rs`

Assertions:

- exact current dimensions, work bounds, rational bounds, memory fields,
  distinct capacity `A=50_000`, calibration active count `50_000`,
  `Current=0`/`Archive=1` encoding and sealed ordinal domains;
- exact binary64 operation-sequence/math-mode identities and the
  `(2^-50, 2^-48, 4 ULP)` tuple;
- novelty/archive-capacity/calibration-active-count/layout/calibration/
  source-kind/ordinal/math/tolerance identities independently alter the run
  identity and reject an old receipt;
- checked overflow and any current-config extent drift fail before allocation.

### R2: exact rational kNN CPU oracle

Path: `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs`

Hand-computed fixtures cover current plus archive neighbors, exact self
exclusion, fewer than K, duplicate zero-distance neighbors, zero union, total
tie ordering and checked comparator limits. The exact `K=15` cutoff fixture has
fourteen nearer neighbors plus equal-distance/equal-identity current and archive
neighbors and proves `Current(0)` wins the last slot. Out-of-domain current,
archive and source-kind values fault. The oracle uses checked `u128` cross
products, independently verifies the exact rational sum, and emulates the
sealed binary64 sequence. Exact `+0.0`, each tolerance boundary, four-ULP
acceptance, five-ULP rejection, and independent absolute/relative-bound
rejections are executable cases rather than comments.

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

The R6 RED may touch exactly four paths:

- `crates/neoethos-gpu-cuda/Cargo.toml`;
- `crates/neoethos-gpu-cuda/src/lib.rs`;
- new private shared authority
  `crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs`;
- `crates/neoethos-gpu-cuda/src/resident_search_v2_tests.rs`.

`Cargo.toml` adds exactly the non-default empty feature
`resident-search-slice2-host-contract = []`. It enables no dependency and is
not included by `default`, `cuda`, `cuda-device-fixtures` or any production
aggregate. `lib.rs` registers the private shared module exactly as:

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

The narrow `cfg_attr` applies only while the shared authority is compiled by a
CUDA build that cannot run the host-contract child. It prevents the deliberately
unbound RED authority from breaking CUDA `-Dwarnings`; it is removed in the
later commit that binds `resident_search_v2.rs`. Host-contract tests and
all-features unit tests do not receive the allowance, so their dead-code
coverage remains strict.

The shared module registers its child tests exactly as:

```rust
#[cfg(all(test, feature = "resident-search-slice2-host-contract"))]
#[path = "resident_search_v2_tests.rs"]
mod resident_search_v2_tests;
```

The shared module contains the exact R6 DTOs, allocator facade, move-only owner
and pending seam frozen below. The later validator is implemented there too;
there is no host-only mirror. To remain available under the empty host feature,
the module imports only `core`, `std` and non-CUDA crate authorities; it may not
import `cust`, `vector-ta`, a CUDA-gated sibling or native FFI. Production
`resident_search_v2.rs` remains under its existing `#[cfg(feature = "cuda")]`
gate and is byte-unchanged by R6 RED.
Later CUDA implementation must call the private shared authority, but this RED
does not claim that production binding yet. No build/native file, other source,
R1-R5 test or dependency feature may change. This remains a pure host contract
and makes no CUDA hardware claim.

Because the RED seam returns `ImplementationPending` before any facade call,
the fifth test uses exactly five separate control recorders before opening a
fresh empty recorder for the real admission call: async `TerminalHostReceipt`,
host `GenerationArena`, host `ScoringArchiveArena`, direct
`begin_native_create`, and one allocation before native create. The first three
prove all category-specific symbols and method counts are derived from the
invoked facade method; the fourth proves the exact single `NativeCreate` event
and `NativeCreateBegun` transition; the fifth proves pre-create phase evidence.
The RED seam itself therefore leaves native-create and allocator counters zero.
This makes the trait methods crate-warning-clean and non-vacuously proves
method-derived evidence; a broad `dead_code` allowance on either unit-test
branch is forbidden.

The shared R6 and future-production error authority is
`ResidentSearchSlice2AdmissionErrorV2`, with these exact discriminants and
payload shapes:

```text
ImplementationPending
MissingArchiveArena
ZeroArchiveArenaBytes
AlignedLayoutFieldMismatch {
    field: ResidentSearchSlice2AlignedFieldV2,
    expected_aligned_bytes: u64,
    observed_aligned_bytes: u64,
}
ReceiptArithmeticOverflow {
    operation: ResidentSearchSlice2ReceiptArithmeticV2,
}
ReceiptTotalMismatch {
    axis: ResidentSearchSlice2ReceiptTotalAxisV2,
    expected_total_bytes: u64,
    observed_total_bytes: u64,
}
ReserveAuthorityBytesMismatch {
    authority: ResidentSearchSlice2ReserveAuthorityKindV2,
    expected_bytes: u64,
    observed_bytes: u64,
}
ReserveAuthorityBindingMismatch {
    authority: ResidentSearchSlice2ReserveAuthorityKindV2,
    axis: ResidentSearchSlice2AuthorityBindingAxisV2,
}
ReserveAuthorityRelationMismatch {
    relation: ResidentSearchSlice2ReserveRelationV2,
}
ReserveArithmeticOverflow {
    operation: ResidentSearchSlice2ReserveArithmeticV2,
}
InsufficientAllocationBudget {
    axis: ResidentSearchSlice2AllocationBudgetAxisV2,
    required_bytes: u64,
    available_bytes: u64,
}
ForeignCalibration {
    axis: ResidentSearchSlice2CalibrationAxisV2,
}
```

`ResidentSearchSlice2AlignedFieldV2` has exactly the twelve table-row variants
`ArchiveGeneScalars`, `ArchiveTermIndices`, `ArchiveTermWeights`,
`ArchiveMetricRows`, `ArchiveSignatures`, `ArchiveHashes`,
`CurrentPopulationSignatures`, `NoveltyScores`, `ExactTopKKeys`,
`AdmissionFlags`, `AdmissionOffsets` and `ArchiveControlAndSeal`.

`ResidentSearchSlice2ReceiptTotalAxisV2` has exactly
`ReplacementSubtotal`, `GenerationReceiptTotal` and
`ScoringArchiveReceiptTotal`. `ResidentSearchSlice2ReceiptArithmeticV2` has the
parallel `ReplacementSubtotalAdd`, `GenerationReceiptTotalAdd` and
`ScoringArchiveReceiptTotalAdd` operations.

`ResidentSearchSlice2ReserveAuthorityKindV2` has exactly
`AllocatorContextHeadroom`, `FullWorkspaceAuthority`,
`RetainedPreSearchWorkspace`, `RemainingSearchAllocationAfterTrim` and
`SameContextFree`. `ResidentSearchSlice2AuthorityBindingAxisV2` has exactly
`DeviceUuid`, `PrimaryContext`, `SearchStream`, `ActivePool`, `RunIdentity`,
`FullWorkspaceReceiptIdentity`, `PostTrimReceiptIdentity` and
`AuthorityIdentity`. `ResidentSearchSlice2ReserveRelationV2` has exactly
`FourReserveAuthorityIdentitiesDistinct` and
`RetainedPlusRemainingEqualsFullWorkspace`. The reserve arithmetic operation
enum retains `WorkspacePartitionAdd`, `RequestedDeviceSumAdd` and
`SameContextFreeMinusHeadroom`. The insufficient-budget axes are
`RemainingSearchAllocationAfterTrim` and
`SameContextFreeAfterAllocatorHeadroom`.

`ResidentSearchSlice2CalibrationAxisV2` has the independent axes `DeviceUuid`,
`PrimaryContext`, `SearchStream`, `ActivePool`, `CudaBuildIdentity`,
`KernelSemanticsIdentity`, `Binary64MathIdentity`, `PlanIdentity` and
`RunIdentity`.

The exact reserve DTO and capability split is:

```rust
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

pub(crate) struct ResidentSearchSlice2ObservedReserveAuthorityV2 {
    pub(crate) bytes: u64,
    pub(crate) binding: ResidentSearchSlice2AuthorityBindingV2,
}

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

`ResidentSearchSlice2AdmissionRequestV2` contains only the observed reserve set
and observed `calibration`; remove `expected_calibration`, every
`expected_identity` field and bare `same_context_free_bytes`. Admission consumes
`ResidentSearchSlice2TrustedReserveSealV2` by value. The trusted authority, set
and seal fields are private; none implements `Clone`, `Copy` or `Default`; there
is no raw constructor, mutable accessor or accessor returning the inner set.

The only R6 minter is the zero-argument
`mint_r6_trusted_reserve_seal_for_fixture_v2()` inside the descendant test
module already gated by
`#[cfg(all(test, feature = "resident-search-slice2-host-contract"))]`. It derives
the exact expected bytes/calibration/provenance from independent fixture
constants and never accepts or reads a request. Static source/topology ratchets
freeze the private field spellings, by-value admission signature, absence of
the three forbidden derives and production/raw constructors, the minter's
zero-argument signature and its sole construction site. This is R6 internal
constructor opacity, not an R7 public-opacity claim.

Before moving each newly minted seal into admission, the child-only
`assert_r6_trusted_reserve_seal_fixture_v2(&seal)` inspects and asserts every
private expected byte, all eight fields of every expected authority binding,
all nine expected-calibration fields and both sealed provenance fields against
independent constants. This keeps the warning-clean host-contract branch
non-vacuous without adding a production accessor. The topology ratchet requires
that inspector and forbids any shared-source/raw inspector.

Every expected authority binding inside the seal shares the seal's calibration
UUID/context/stream/pool/run tuple and its sealed full-workspace/post-trim
receipt identities. The deferred CUDA minter must establish that invariant by
consuming the real opaque authorities. A foreign observed set plus coordinated
plain expected-looking values cannot be converted to the seal and must still
fail against the independently minted capability.

Each foreign calibration case starts from the otherwise-valid receipt. Every
`u64` calibration axis is exercised twice, once with bit `0` changed and once
with bit `63` changed. `DeviceUuid` is exercised at byte `0` and byte `15`.
The same low/high and byte-zero/tail controls apply to every corresponding field
in each observed reserve binding before the independently trusted comparison.

The facade accepts method-specific arguments without a caller-declared symbol:

```rust
pub(crate) struct ResidentSearchSlice2HostAllocationArgsV2 {
    pub(crate) ordinal: u8,
    pub(crate) category: ResidentSearchSlice2AllocationCategoryV2,
    pub(crate) requested_bytes: u64,
    pub(crate) aligned_bytes: u64,
    pub(crate) alignment_bytes: u64,
    pub(crate) flags: u32,
}

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

pub(crate) trait ResidentSearchSlice2AllocationFacadeV2 {
    fn begin_native_create(&mut self);
    fn cuda_host_alloc(&mut self, actual: ResidentSearchSlice2HostAllocationArgsV2);
    fn cuda_malloc_async(&mut self, actual: ResidentSearchSlice2AsyncAllocationArgsV2);
}
```

The child recorder derives the symbol from the method and records these exact
test-owned chronology authorities:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ResidentSearchSlice2RecorderPhaseV2 {
    #[default]
    BeforeNativeCreate,
    NativeCreateBegun,
    AllocationsComplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentSearchSlice2RecorderEventV2 {
    NativeCreate {
        phase_before: ResidentSearchSlice2RecorderPhaseV2,
    },
    Allocation {
        phase_at_call: ResidentSearchSlice2RecorderPhaseV2,
        call: ResidentSearchSlice2AllocationCallV2,
    },
}
```

The exact valid chronology is `NativeCreate`, host allocation ordinal `0`,
async allocation ordinal `1`, async allocation ordinal `2`; its method counts
are `1/2`. Exactly five direct recorder controls inside the fifth existing test
prove: a terminal-shaped call through the async method records
`CudaMallocAsync`; a `GenerationArena` call through the host method records
`CudaHostAlloc`; a distinct `ScoringArchiveArena` call through the host method
also records `CudaHostAlloc`; direct `begin_native_create` records exactly one
`NativeCreate { phase_before: BeforeNativeCreate }` event and advances to
`NativeCreateBegun`; and an allocation before native create retains
`BeforeNativeCreate` in the event. Each control owns a separate recorder and
completes before the fresh real-admission recorder reaches
`ImplementationPending`.

Exactly these five named tests are the R6 authority:

1. `slice2_combined_admission_rejects_missing_or_zero_archive_arena_before_allocation`
   covers both absent archive authority and present-but-zero archive bytes. It
   requires `MissingArchiveArena` and `ZeroArchiveArenaBytes`, respectively.
2. `slice2_combined_admission_rejects_each_aligned_layout_field_mismatch_before_allocation`
   mutates each of the twelve aligned fields independently while recomputing the
   declared subtotal so that subtotal-only validation would pass. Every case
   requires `AlignedLayoutFieldMismatch` with the exact field and byte payload.
   The same test keeps all twelve fields fixed while changing only the declared
   replacement subtotal, then separately overflows the checked subtotal and
   requires overflow to take precedence over total mismatch.
3. `slice2_combined_admission_rejects_insufficient_reserve_before_allocation`
   keeps generation components fixed while changing only their total, repeats
   that control for scoring/archive, and separately proves checked-add overflow
   precedence for both receipts. It then covers expected bytes and every tuple
   field for all five observed/trusted reserve authorities, low/high `u64` and
   UUID byte-zero/tail controls, four-way all-alias refusal, all six pair-only
   alias refusals, the four coordinated substitution controls, the workspace
   partition relation, all
   three reserve checked-arithmetic failures, exact fit on both independent
   budgets and a one-byte-short failure on each budget. Exact fit reaches the
   otherwise-valid `ImplementationPending` RED seam; each rejection returns its
   exact typed axis/operation and payload. This third test contains exactly
   `107` controls.
4. `slice2_combined_admission_rejects_foreign_calibration_before_allocation`
   changes each calibration axis independently and requires
   `ForeignCalibration` with that exact axis. Every `u64` axis has low- and
   high-bit cases; UUID has byte-zero and tail-byte cases. This fourth test has
   exactly `18` mutations across the nine axes.
5. `slice2_valid_combined_admission_executes_declared_ledger_once_and_later_generations_allocate_nothing`
   first runs exactly five separate controls: async terminal, host generation,
   host scoring/archive, direct native create and allocate-before-create. It
   then opens a fresh recorder before actual admission. On GREEN it requires
   chronology length `4` beginning with native create, host/async method counts
   `1/2`, exact allocation-ledger length `3` before full vector equality, queues
   generations one, two and three, and requires phase, chronology, ledger and
   all counters to remain byte-for-byte equal to the post-admission snapshot.

Every input-rejection case invokes the real Rust admission API exactly once but
must stop before native create. The native-create, host-method, async-method,
physical allocator, generation-arena, scoring/archive-arena and forbidden
archive-only-arena counts are all zero, and chronology plus observed ledger are
empty.

The R6 mutation gate must kill every one of these edits:

- delete missing-archive validation or zero-archive validation;
- validate only the subtotal: each of the twelve fields is independently
  changed while the subtotal remains self-consistent;
- trust any declared subtotal/receipt total, use wrapping/saturating component
  addition, or report total mismatch before the corresponding checked-add
  overflow;
- replace checked reserve math with wrapping or saturating math;
- delete any independently trusted expected-byte comparison, whole-authority
  binding, individual binding-tuple field, any one of the six pairwise
  distinctness checks, coordinated-substitution refusal or partition relation;
- violate any of the four frozen simultaneous-mismatch boundaries: headroom
  before other authorities, full workspace before retained, headroom bytes
  before its binding, or headroom UUID before its other binding axes;
- expose, clone, copy, default or raw-construct the trusted capability graph;
  accept an unsealed set, minter input or coordinated plain expected-looking
  values; omit expected calibration or full/post-trim provenance; or remove the
  child-only full-field inspection;
- accept exact fit incorrectly or accept a one-byte-short budget;
- delete any calibration binding, with UUID-only foreign receipt mandatory and
  context, stream, pool, build, kernel-semantics, binary64-math, plan and run
  axes independently controlled; truncated-`u32` identity and UUID-byte-zero-
  only comparisons are explicit mutants;
- copy the declared ledger into observed state instead of recording actual
  allocator calls;
- skip or reorder one call; independently change its ordinal, symbol, category,
  requested bytes, aligned bytes, alignment, flags, stream or resolved pool;
- trust a declared symbol instead of the invoked host/async method, omit either
  method count, remove native-create event recording, or allocate before native
  create;
- prepend or append an extra observed entry, proving a `zip` comparison without
  exact cardinality cannot pass;
- allocate again while queueing generation two or generation three.

The existing 63 names stay byte-for-byte and in the same order. Version 8
appends exactly these 69 unique names, making
`R6_MUTATION_NAMES.len() == 132`:

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

The RED commit freezes this complete register and the input mutations inside
the five test bodies, but it cannot truthfully claim to kill implementation
mutants while the only implementation is the unconditional
`ImplementationPending` stub. Actual apply/revert mutation kills are mandatory
against the first passing GREEN implementation that removes that stub, after
the canonical R1-R9 pure-RED checkpoint. The deferred GREEN receipt does not
block R7-R9 RED scaffolding after R6 review; it blocks advancing production
implementation beyond combined-admission GREEN and blocks the authorized RTX
sequence.

The first R6 commit compiles the `neoethos-gpu-cuda` crate with zero warning
diagnostics under `-Dwarnings` and runs exactly these five tests. All five fail
at runtime with the typed `ImplementationPending` discriminant; there is no
crate warning, compile failure, sixth test, unrelated failure or device
execution. A verbose Cargo log may contain cap-linted warnings from locked
third-party dependencies and must preserve/count them rather than describe the
entire log as warning-free. The current fresh log has twelve warning-prefixed
third-party lines: six from `generic-array`, one from `windows-core`, two from
`windows`, and three Cargo warning summaries.
The exact Windows PowerShell command is:

```powershell
$env:CARGO_INCREMENTAL = '0'
$env:RUSTFLAGS = '-Dwarnings'
cargo +nightly-2026-04-07 test --locked --offline -j 7 -p neoethos-gpu-cuda --no-default-features --features resident-search-slice2-host-contract --lib 'resident_search_slice2_admission_v2::resident_search_v2_tests::slice2_' -- --nocapture
```

`-j 7` is a Cargo option before the test-runner `--`. The exact result is a
successful crate-warning-clean compile followed by `0 passed; 5 failed`, with
all five failures caused only by `ImplementationPending`. The command must not
resolve `cust`, `cust_raw`, `vector-ta`, `nvcc`, `cuobjdump` or a CUDA link.

### R7: executable move-only opacity

Authority is the tracked compiler-UI harness, not rustdoc:

- runner:
  `crates/neoethos-search/tests/resident_search_slice2_compile_contract.rs`;
- shared fixture package:
  `crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml`;
- sources and normalized stderr:
  `crates/neoethos-search/tests/ui/resident_search_slice2/{pass,fail}/`.

The nested manifest is a standalone fixture workspace. It contains the literal
empty table `[workspace]`, sets `autobins = false`, has `default = []`, forwards
only `resident-search-slice2-device-fixtures`, declares the two local path
dependencies with `default-features = false`, and contains exactly ten explicit
`[[bin]]` entries matching the target/source table below. It has no implicit
`src/main.rs`, glob target or workspace member. Its generated `Cargo.lock` is
tracked and bound by the R7 receipt.

The non-target portion is exactly:

```toml
[package]
name = "neoethos-resident-search-slice2-ui"
version = "0.0.0"
edition = "2024"
publish = false
autobins = false

[workspace]

[features]
default = []
resident-search-slice2-device-fixtures = [
    "neoethos-search/resident-search-slice2-device-fixtures",
    "neoethos-gpu-cuda/cuda-device-fixtures",
]

[dependencies]
neoethos-search = { path = "../../..", default-features = false }
neoethos-gpu-cuda = { path = "../../../../neoethos-gpu-cuda", default-features = false }
```

The ten `[[bin]]` tables contain only the exact `name` and `path` pair in the
table below. Each negative source has exactly one co-located same-stem
`.stderr`; the positive target has none.

The runner performs exactly ten isolated compiler invocations. For each row it
uses the same command shape with that row's one exact target substituted:

```text
cargo check --manifest-path crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml --locked --offline --no-default-features --features resident-search-slice2-device-fixtures --bin <exact-target> --message-format=json
```

It never invokes `--bins`, an inferred default target or a package-wide check. It
parses compiler JSON, requires the expected primary diagnostic code and exact
authored source span, and compares the compiler's normalized rendered diagnostic
against the tracked `.stderr` receipt. Any extra primary error, unresolved
crate, feature drift, skipped fixture, warning-only result or stderr drift is a
failure. The positive fixture must exit successfully with zero diagnostics.
Rustdoc prose may illustrate the API but is not evidence and no doctest count is
claimed.

The exact fixture set is one positive plus nine negatives:

| Exact `--bin` target | Exact source | Required result |
| --- | --- | --- |
| `pass_typed_surface` | `pass/typed_surface.rs` | imports and moves every owner/receipt through the supported typed surface without creating a GPU resource |
| `fail_clone_owner_e0599` | `fail/clone_owner_e0599.rs` | `E0599` |
| `fail_copy_owner_e0277` | `fail/copy_owner_e0277.rs` | `E0277` |
| `fail_read_trim_map_e0616` | `fail/read_trim_map_e0616.rs` | `E0616` |
| `fail_read_trim_event_e0616` | `fail/read_trim_event_e0616.rs` | `E0616` |
| `fail_read_archive_pointer_e0616` | `fail/read_archive_pointer_e0616.rs` | `E0616` |
| `fail_read_population_field_e0616` | `fail/read_population_field_e0616.rs` | `E0616` |
| `fail_call_staged_constructor_e0624` | `fail/call_staged_constructor_e0624.rs` | `E0624` |
| `fail_construct_ranked_receipt_e0451` | `fail/construct_ranked_receipt_e0451.rs` | `E0451` |
| `fail_novelty_receipt_as_full_deadline_e0308` | `fail/novelty_receipt_as_full_deadline_e0308.rs` | `E0308` at the argument that passes `ResidentArchiveKnnCalibrationReceiptV2` to the sink requiring `FullResidentDiscoveryDeadlineReceiptV1` |

The positive fixture names both receipt types and calls correctly typed sinks
through divergent value suppliers, so private constructors are not needed and
the `E0308` negative cannot pass because a type is missing. The gate requires
exactly `1` positive and `9` negative results. The normal R6 behavioral test
supplies the separate missing combined archive-preallocation RED. No
source-string/token assertion is accepted for either property.

### R8: transactional faults and cleanup

Path: `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_tests.rs`, registered
as `#[cfg(all(test, feature = "cuda"))] mod resident_archive_knn_v2_tests;`.
Exactly these four named tests are the R8 authority:

1. `r8_all_eleven_metric_slots_reject_nan_and_infinity_atomically` executes
   exactly 22 injections: each metric slot `0..=10` is replaced once with the
   canonical quiet NaN and once with positive infinity.
2. `r8_structural_fault_matrix_is_atomic` executes exactly six injections:
   signature word count `3`, signature word count `5`, zero union,
   `archive_count=50_001`, boxed receipt-address drift and comparator
   union/cross-product-bound drift.
3. `r8_fault_cleanup_is_checked_once_and_owner_never_reused` executes exactly
   four terminal states: event-proved semantic fault, `NotReady`, unknown CUDA
   outcome and unproved event. Only the first may perform normal checked cleanup;
   the other three retain or deliberately leak the armed composite. A second
   cleanup or reuse attempt is rejected in all four cases.
4. `r8_every_recoverable_fault_allows_a_fresh_unrelated_run` repeats the 22
   metric and six structural fixtures, for exactly 28 fault/fresh-run pairs,
   and requires a separately admitted valid run with a different run token to
   publish successfully after each pair.

Every one of the 28 metric/structural cases leaves the packed commit word
unchanged, publishes no staged tail, returns the exact compact terminal fault
and preserves the declared allocation ledger. The module also contains one
valid baseline control; it is not counted as an injection or fault/fresh pair.
The test runner asserts the literal case counts so deleting a loop member cannot
silently reduce coverage.

### R9: real RTX behavior and calibration

Paths:

- `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_device_tests.rs`;
- test-only CUDA interception translation unit
  `crates/neoethos-gpu-cuda/native/resident_archive_knn_v2_cuda_intercept_test.cu`.

One self-authenticating sequence runs:

1. `resident_archive_knn_v2_calibration_rejects_nonrepresentative_receipts_on_real_cuda`:
   exact production fixed-K calibration with a genuinely prefilled 50,000-entry
   archive and a receipt binding both capacity `A=50_000` and
   `calibration_active_count=50_000`. Independent executable controls include
   empty/inactive-tail archive, capacity `A=50_000` with valid active slots
   `0..=49_998` and inactive slot `49_999`, capacity `A=49_999` with its bounded
   active count also `49_999`, `K=14`, `K=16`, proxy/popcount kernel, a
   distance-under-rate while popcount passes control and a popcount-under-rate
   while distance passes control. The
   one-inactive-tail control returns exactly
   `ArchiveActiveCountMismatch`; the capacity control returns exactly
   `ArchiveCapacityMismatch`. Every case returns its listed typed refusal;
   active-count/capacity/K/shape preflight refusals have zero calibration
   allocations and kernel launches, and every case has zero combined-Search
   allocations;
2. `resident_archive_knn_v2_admission_rejects_stale_foreign_or_shape_drift_receipts_before_allocation`:
   current-capacity allocation admission, including executable rejection of a
   stale CUDA build/math identity, foreign UUID/context/stream/pool receipt and
   independent `P`, `A`, `W` and `M` shape drift, source-kind/ordinal identity
   drift, binary64 operation/math-mode drift and each tolerance-field drift,
   with the exact typed binding/field refusal, zero combined-admission calls and
   zero generation-arena/scoring-archive-arena/archive-only-arena allocation
   deltas on every rejection;
3. at least three resident generations compared to the independent CPU oracle
   by the preloaded test-only device validator for neighbor identities, novelty
   values under the receipt-bound `(2^-50, 2^-48, 4 ULP)` policy, rank, archive
   content/count and `g+1` visibility. The exact cross-source cutoff fixture
   proves `Current(0)` precedes `Archive(1)` and is bound to the run identity;
4. duplicate, adversarial collision, cap and all fault/cleanup cases;
5. interception assertions spanning pre-admission epoch seal through terminal
   projection: the exact declared admission allocation ledger, exactly one
   asynchronously classified trim-ready stream wait, zero pre-terminal D2H and
   host synchronization, one compact terminal D2H/event/projection, and no later
   allocation. Acceptance requires the positive-control handshake to have
   observed every wrapped runtime/driver family. The supervisor then re-spawns
   the exact same current executable/child test/fixture/hashes/argv with only
   `NEOETHOS_SLICE2_INTERPOSER_MODE_V1=disabled`; acceptance requires the exact
   `MissingInterposerControlHandshake` discriminant and exit code `86`, not a
   generic child failure.

The separately classified prepared-headless GREEN invariant below runs before
and after this sequence. It is evidence in the final receipt, but it is never an
R9 RED and a failure stops the sequence rather than authorizing implementation.

The device sequence records GPU UUID/name/compute capability/memory, exact
command and environment, source and binary hashes, receipt/counter values and
exit status. It runs once after an independent source review says safe to run.

## Exact host-contract feature and cfg topology

The implementation may use only this topology:

- `neoethos-gpu-cuda/Cargo.toml` declares exactly
  `resident-search-slice2-host-contract = []`; it is non-default, has no
  `dep:` or feature edge, and neither includes nor is included by `cuda`;
- `neoethos-gpu-cuda/src/lib.rs` has exactly one private declaration of
  `resident_search_slice2_admission_v2`, under
  `cfg(any(feature="cuda", all(test,
  feature="resident-search-slice2-host-contract")))`. The `any` means enabling
  both features still declares the module once. The adjacent narrow
  `cfg_attr` above permits dead code only for a CUDA compile without the host
  child, until production binding removes it;
- the shared module alone registers `resident_search_v2_tests` under
  `cfg(all(test, feature="resident-search-slice2-host-contract"))`;
- production `resident_search_v2` retains its existing public module
  declaration under only `cfg(feature="cuda")`. R6 RED does not modify that
  file or claim that it already consumes the shared validator;
- `neoethos-gpu-cuda/src/lib.rs` registers production
  `resident_archive_knn_v2` under `#[cfg(feature = "cuda")]`, R8 under
  `#[cfg(all(test, feature = "cuda"))]`, and
  `resident_archive_knn_v2_device_tests` under
  `#[cfg(all(test, feature = "cuda-device-fixtures"))]`;
- the external-compile and device fixture façade is
  `resident_archive_knn_v2_device_fixture`, registered under
  `#[cfg(feature = "cuda-device-fixtures")]` because integration/UI targets
  compile the library without `cfg(test)`; the façade contains no production
  constructor and is absent unless the non-default fixture feature is explicit;
- the native interception translation unit is compiled only when
  `CARGO_FEATURE_CUDA_DEVICE_FIXTURES` is present and is never linked by the
  plain `cuda`, default, application or production feature closures;
- `neoethos-search` adds exactly
  `resident-search-slice2-device-fixtures = ["gpu-b-native",
  "neoethos-gpu-cuda/cuda-device-fixtures"]`; it is absent from every default,
  application and production aggregate;
- `prepared_discovery_run_input_v3.rs` registers the headless invariant exactly
  as `#[cfg(all(test, feature = "resident-search-slice2-device-fixtures"))]`
  with its explicit `#[path = ...]` child module;
- Cargo target `resident_search_slice2_compile_contract` has
  `required-features = ["resident-search-slice2-device-fixtures"]`; its nested
  fixture package has its own literal empty `[workspace]`, `autobins=false`, a
  tracked lockfile, forwards only that feature and exposes only the ten explicit
  `[[bin]]` targets selected individually by R7. It never joins the repository
  workspace or becomes a default/package-wide target.

No `cfg(test)`-only item is treated as visible to an integration test, and no
test-only feature is allowed to unify into a production or application build.
Source-contract tests assert these exact attributes and negative feature
closures.

R6 acceptance also records four host-topology ratchets:

1. `cargo +nightly-2026-04-07 metadata --locked --offline --no-deps
   --format-version 1` proves the host-contract feature value is the literal
   empty array, `default` remains `[]`, `cuda` remains exactly
   `["dep:cust", "dep:vector-ta"]`, and `cuda-device-fixtures` remains
   `["cuda"]`. No feature includes the host-contract feature.
2. The focused R6 command is repeated from a GUID-named target whose
   nonexistence is asserted first, with verbose build logging. Its persisted
   complete log and SHA-256 must show the five exact named tests in the panic
   headers, failed-status lines and final failure list, with exactly five
   `ImplementationPending` discriminants. Because the exact command uses
   `--nocapture`, acceptance does not rely on absent captured-stdout sections.
   Neither resolved packages nor build/link command lines may contain `cust`,
   `cust_raw`, `find_cuda_helper`, `vector-ta`, an `nvcc`/`cuobjdump`
   invocation, `-lcuda`, `cudart`, CUDA `rustc-link-lib` (including static
   forms), or Windows `DEFAULTLIB` CUDA forms. A package/path containing the
   crate name `neoethos-gpu-cuda` and the build-script declaration
   `rerun-if-env-changed=CUDA_PATH` are not themselves CUDA links.
3. Relative to v6 authority commit
   `4f0880148677df0d8f58c11373b42d0bd87e5b13`, the R6 commit has zero diff in
   `build.rs`, `resident_search_v2.rs` and `Cargo.lock`; normalized literal
   source ratchets assert exactly one complete shared-module `cfg`/`cfg_attr`
   declaration, one complete child-test gate and one continued CUDA-only
   production module declaration, with no duplicate declaration of any of the
   three modules.
4. Cargo metadata with `--all-features` must resolve both features together,
   and the one `cfg(any(...))` declaration prevents duplicate module ownership.
   This host proves graph/topology compatibility only; actual all-features
   compilation is repeated later on the authorized CUDA toolchain and is not
   claimed by R6.

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
epoch, after the whole carrier is sealed and before the first possible combined
Search-admission call. The call must reach the private entry, return the exact missing-full-run-
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
- private `src/resident_search_slice2_admission_v2.rs` as the single shared
  R6 admission/validation authority for host contract and future CUDA
  production, plus the empty non-default host-contract feature and exact
  `lib.rs` gate above;
- versioned combined admission and split-transaction additions in
  `resident_search_generation_v2_abi.cuh`, `resident_generation_v1.cu`,
  `resident_scoring_novelty_v1.cu`, `resident_scoring_v2.rs` and
  `resident_search_v2.rs`; the latter later calls the shared authority under
  `cuda`, but that binding is not present or claimed in R6 RED;
- the crate-private whole-carrier consumer in
  `resident_trim_prefilter_v1.rs`;
- `crates/neoethos-search/src/gpu_resident_current_config_plan_v1.rs` and its
  focused test module, solely to bind the distinct archive capacity/calibration
  active count and other Slice 2 identities into the plan, and to place the
  uninhabited `FullResidentDiscoveryDeadlineReceiptV1` compile marker at the
  exact declaration-only location specified above; no deadline sealer, proof or
  readiness input is authorized;
- build registration and focused contract/device test modules, including the
  test-only prepared-headless GREEN invariant module and its non-production
  fixture feature and compiler-UI target in
  `crates/neoethos-search/Cargo.toml`, plus only the exact
  `#[cfg(all(test, feature = "resident-search-slice2-device-fixtures"))]`
  child-module registration in `prepared_discovery_run_input_v3.rs`;
- the tracked R7 fixture `Cargo.toml`, `Cargo.lock`, ten sources/stderr receipts
  and the non-default
  `resident_archive_knn_v2_device_fixture` façade required by the exact topology
  above.

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
