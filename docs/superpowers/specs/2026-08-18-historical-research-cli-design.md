# Historical Research CLI Design

## Goal

Replace the legacy `search` command with one production caller for the
receipt-bound, CPU-only historical candidate scan. The command produces only
gross-reference-R research evidence and can never enter broker-cost, PnL,
promotion, live, model, or PropFirm code.

The production logic lives in `neoethos-search`, not the model-dependent legacy
CLI crate. A lightweight `neoethos-historical-search` binary in that package is
the runnable proof. The legacy `neoethos-cli search` branch is only a thin
delegate to the same adapter, so there is one implementation and no model build
dependency on the critical search path.

## Command contract

`neoethos-cli search` requires an exact canonical input receipt, an explicit
seed, a non-zero candidate count, a non-zero maximum indicators-per-candidate,
positive stop/target multiples, an output path, and a canonical data root. The
receipt is the authority for the anchor identity and every directly downloaded
timeframe. Symbol/base/higher display selectors and the old genes/generations
flags are removed.

The entrypoint classifies `search` before configuration loading and runtime
installation. Strict receipt search does not require or read `config.yaml` and
skips every config-dependent hardware, data, model, and generic search-runtime
installer. Its only process knob is the non-semantic CPU capacity boundary:
automatic host detection optionally narrowed by the shared, validated parent
`--cpu-threads` assignment. Every semantic search knob therefore comes only
from strict command arguments and the exact receipt, and is recorded in the
artifact contract/hash.

The lightweight binary installs that immutable process budget before any pool
can exist. The shared adapter acquires an exact lease and performs exact source
loading plus feature construction inside a private `BudgetedCpuExecutor`, so
feature computation cannot initialize or escape to unbudgeted global Rayon.
The lease may be returned after candidate evaluation; atomic output I/O does
not reserve CPU workers.

The command first decodes and validates the receipt. It converts every receipt
binding into a `SelectedDatasetGenerationV1` and opens that exact immutable
generation, so generation and manifest binding mismatches fail before feature
computation. All identities must name one symbol/source/account series and
bar-open, directly sourced timeframes. The feature frame is then rebuilt, its
new receipt must equal the expected receipt byte-for-byte, and only then is a
`CanonicalSearchRunInputV1` constructed.

Every loaded `CanonicalOhlcvFrame` and its generation lease remains alive until
the scan and atomic output commit finish. No current-generation loader,
symbol-only lookup, missing-timeframe fallback, or resampling API is reachable.

## Deterministic candidates and validity

The operator seed is domain-separated with the exact receipt SHA-256 and the
versioned generator policy. The existing typed `Gene` generator creates signal
rules with structural/MTF flags disabled. Candidate feature columns may contain
typed warmup and gap cells. Invalid cells remain ineligible/flat through
`signals_for_gene`; they are not treated as fatal data corruption. A candidate
is admitted only when the intersection of its selected columns contains the
fixed minimum number of causal, valid, finite rows.

Candidate signal identities are computed by the historical-research domain
owner. Deterministic collisions are regenerated until exactly the requested N
unique identities exist. A bounded retry exhaustion returns a typed error and
never silently produces fewer candidates.

Candidate evaluations use indexed Rayon only inside a `BudgetedCpuExecutor`
that owns a transfer from the exact broker which installed the process budget.
Every worker produces one `Result` and the joined vector is interpreted only in
input-ordinal order, including `FailEntireScan` error selection. A transfer
from another broker is rejected. Worker width is an execution detail and is
not serialized or hashed: worker=1 and automatic-width runs must produce
byte-identical full JSON artifacts and rankings.

## Distance and result artifact

The price-native distance is a fixed causal true-range policy. Each row uses
only that closed bar and the previous close. A zero range carries the last
positive distance; before the first positive range it uses one ULP at the
current positive close. The semantic policy ID is versioned and the historical
scan binds both its values hash and the input receipt hash.

The atomically created JSON artifact contains a versioned candidate-generation
contract and the complete `HistoricalCandidateScanResultV1`. It therefore
contains the exact receipt scope, search identity, ranking-policy identity,
`research_only`, `not_promotion_eligible`, CPU-only backend, and gross-R metrics.
It contains no `net_profit`, financial cost, broker-real, or promotion label.

## Tests

The first integration test is intentionally RED and asserts the end-to-end
source/behavior boundary: `search` delegates only to the new module; the legacy
`evolve_search` and old flags/help are absent; strict arguments are mandatory;
an exact canonical fixture produces a receipt-bound ResearchOnly artifact.
Focused tests cover receipt drift before feature computation, typed validity,
deterministic exact-N uniqueness, typed search-space exhaustion, causal distance,
wrong-broker lease rejection, worker=1 versus automatic-width full-JSON byte
parity, and create-new atomic output behavior.
