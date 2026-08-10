# Streaming parameter search — sweeping the (indicator, period) space over time

**Status:** design. Two of its four mechanisms already exist and are running in
`crates/neoethos-data/src/core/`. The other two are a change to the discovery
loop, which lives in forbidden territory for the workstream that wrote this, so
they are specified here rather than applied.

---

## 1. Why the obvious answer is wrong, and why the memory objection dissolves

The vocabulary repair took the base timeframe from **66 columns to 825**. The
next question is parameters: a period is not a detail of an indicator, it
selects which market phenomenon the indicator measures. RSI(4) is a scalping
oscillator; RSI(54) is a regime filter. A vocabulary that can reach only one of
them is not 342 indicators, it is 342 arbitrary points in a space nobody chose.

The obvious extension is to sweep everything across a period ladder. Its
arithmetic:

```
342 indicators x 5 periods x 3 timeframes  = 5,130 columns
5,130 x 843,456 bars x 8 bytes             = 34.6 GB   (one symbol)
```

That does not fit, and the number is often used to close the discussion. It
should not be, because it prices something nobody asked for: **materialising all
5,130 columns simultaneously**. The operator's design does not do that.

> "A LOOP, mix + SMC, with about 32 indicators at a time whose parameters
> CHANGE — then if it sees the result is 0, or that the result is against our
> goals, it must move on to the NEXT indicators and do the same, change after
> change, until we see a result. They do not need to reach the OOS/validation
> stage at all if they are below target from the start."

Its arithmetic is a different problem:

```
32 indicators x 5 periods x 3 timeframes   = 480 columns
480 x 843,456 bars x 8 bytes               = 3.24 GB   -> fits comfortably
```

The full space is swept over **time**, in batches, never held at once. Peak
memory becomes a function of the working-set size — which is itself derived from
available hardware — and the reachable parameter space becomes effectively
unbounded. That is the never-OOM invariant applied correctly rather than
circumvented.

The second half matters as much as the first. **Early rejection.** A batch whose
candidates are below target must be abandoned and replaced immediately, never
promoted to the quality screen, the walk-forward, or OOS validation. Today the
opposite happens: the measured run spent **50.4% of its wall time** in a quality
screen that rejected 174 of 174 candidates — work that was knowably wasted
before it started.

---

## 2. What already exists

Three of the four parts of this design are already built, because the
vocabulary repair needed them for its own reasons. This is the useful fact: the
design does not start from a blank page.

| Mechanism it needs | What plays that role today | File |
|---|---|---|
| A hardware-derived working-set sizer | `VocabularyBudget::for_run` — turns free RAM into a maximum column count, never a constant, never a user parameter | `core/feature_budget.rs` |
| A deterministic, non-repeating advance through the id space | `admit_indicators` / `extended_sweep_plan` — take the prefix of `ALL_INDICATORS` that fits, in declaration order | `core/feature_budget.rs`, `core/hpc_ta.rs` |
| Per-batch accounting that cannot drop silently | `IndicatorLedger` — 14 typed drop reasons, per-id tallies, one census per pass, a hard floor | `core/indicator_ledger.rs` |
| **A cheap early-reject predicate + a loop that swaps the working set** | **missing** | `neoethos-search` |

`extended_sweep_plan` is deliberately shaped as the prototype of the advance
step: it is a pure function of `(ALL_INDICATORS, budget_columns)` with no
dependence on the frame, so a later change can call it with an *offset* and get
batch 2, batch 3, … without the column layout becoming a function of
scheduling.

---

## 3. The design

### 3.1 The working set

A batch is a set of `(indicator, period)` pairs plus the always-resident
families (SMC, session, regime, footprint — the "mix + SMC" half of the
operator's sentence). Its width is chosen by `VocabularyBudget`, not by a
constant:

```
max_columns = 0.25 * free_bytes / (rows * 8)      # feature_budget.rs
batch_pairs = max_columns - resident_columns      # what is left after the mix
```

On the operator's box (20.6 GB free, M5 at 1,054,320 bars, 8.43 MB per column)
that is 611 columns total; with ~180 resident it leaves ~430 swept pairs per
batch — comfortably more than the 480-column figure the design was sketched at,
and it shrinks automatically on a laptop instead of OOMing.

### 3.2 The advance

The `(indicator, period)` space is enumerated in one canonical order:

```
for period in ladder:            # outer, so a batch mixes timescales
    for id in ALL_INDICATORS:    # inner, declaration order
        if sweepable(id): yield (id, period)
```

Batch *k* is the slice `[k * batch_pairs, (k+1) * batch_pairs)`. Properties this
buys, all of which the search needs and none of which are free:

* **no repeats** — a pair appears in exactly one batch;
* **deterministic** — batch *k* is the same set on every machine and every run,
  so a result is reproducible from `(seed, batch index)` alone;
* **frame-independent** — the batch does not depend on how many bars the frame
  has, so per-timeframe widths stay equal and the cube can still be assembled;
* **resumable** — the cursor is one integer in the run artifact.

The ladder is a ladder only as a first cut. See §5(d): making the period part of
the gene turns this into a continuous space and the batch into a *region* rather
than a slice.

### 3.3 The early-reject predicate

This is the part that does not exist, and it is the part that pays.

The predicate must be cheap enough to run on every batch and honest enough that
discarding on it is not throwing away signal. Three candidates, in increasing
cost:

1. **Does the batch produce any candidate above the expectancy floor at all?**
   Run the GA for a small fixed number of generations (`probe_generations`,
   sized from the batch, not a constant) and read the best candidate's payoff
   and expectancy *with real broker costs already charged* — the costs are what
   make the 174/174 rejection knowable in advance. Reject the batch if the best
   probe candidate cannot clear the floor.
2. **Does the batch's best candidate improve on the incumbent?** Stricter,
   cheaper to justify, and the right predicate once a first survivor exists.
3. **Does any column in the batch carry information the resident set does not?**
   A redundancy test, not a correlation-with-return test. See §4 — this is where
   the univariate trap lives.

The measured justification for doing this at all: the last run reached the
quality screen with 174 candidates and **zero** survived a 2.0 payoff floor with
real costs charged, after spending half the run's wall time getting there. Any
predicate that would have rejected those 174 before the screen is worth more
than the predicate's own cost by a wide margin.

**The predicate must be logged and counted like every other discard.** A batch
rejected with no record is the silent drop again, one level up. Reuse
`IndicatorLedger`'s shape: reason, count, named examples, one census line per
batch.

### 3.4 The loop

```
resident = mix + SMC                       # always present
cursor   = 0
while budget_remains and not converged:
    batch = advance(cursor, batch_pairs)   # 3.2
    cols  = materialise(resident + batch)  # existing feature build, one batch wide
    probe = ga_probe(cols, probe_generations)
    if reject(probe):                      # 3.3
        ledger.batch_rejected(cursor, why, probe.best)
        cursor += batch_pairs
        continue                           # never reaches quality screen / WF / OOS
    survivors += full_search(cols)
    cursor += batch_pairs
```

---

## 4. The trap this must not walk into

The prefilter that already exists (`discovery.rs:3891-4029`) ranks features by
`|Pearson(feature, 1-bar forward close return)|` over an 80% in-sample prefix.
It is **univariate with no redundancy step**, and the code already knows the
criterion is wrong: `regime_` columns had to be exempted with `f32::INFINITY`
because a regime state has no standalone directional correlation. That exemption
is the criterion admitting its own flaw — and it was granted to exactly one
family. SMC, session and footprint have no such exemption, which is why they die
first when `prefilter_top_k` is small.

A stage-one screen built on the same criterion would amputate the same families
before the search ever saw them. So the batch-level predicate in §3.3 is
deliberately **not** a correlation screen. It is "did the GA find anything",
which is multivariate by construction: a feature that only matters in
combination is still reachable, because the GA is what evaluates it.

If a cheaper screen is ever needed, the honest version is a redundancy test
against the resident set (does this batch span any direction the resident
columns do not), not a relevance test against forward return.

---

## 5. Options (a)-(d) as ingredients, not alternatives

* **(a) Wider sweep on a bounded set** — this is what `extended_sweep_plan`
  already does, and it is the right *first* landing: it materialises one cube,
  so nothing about the discovery loop changes. It is strictly weaker than (e):
  the space it can reach is capped by one machine's RAM.
* **(b) Two-stage coarse-then-refine** — the coarse stage is exactly where the
  univariate trap in §4 bites. Usable only with §3.3's GA-probe predicate, at
  which point it is (e) with a non-uniform ladder.
* **(c) Compute on demand on the device** — the multiplier. If indicator columns
  are derived in VRAM from resident bars, `materialise()` costs a kernel launch
  instead of a host allocation, and `batch_pairs` stops being bounded by host
  RAM at all. The f64 arch-agnostic indicator kernels another workstream is
  building are exactly the prerequisite. Cost to weigh honestly: recomputation
  per generation versus caching — a column referenced by many genes across many
  generations may be cheaper to keep resident than to re-derive, so this needs a
  measured cache policy, not an assumption.
* **(d) Period inside the gene** — the most faithful answer to "dynamic", and
  the most invasive. It makes the space continuous instead of a ladder: the gene
  already carries CSR indices and weights, and a period per selected term is one
  more parallel array. The cost is in the kernel — a per-term period means the
  indicator value can no longer be a column lookup, so this only makes sense
  **after** (c), where the value is derived on the device anyway. Say plainly:
  this is a redesign of the gene encoding and the evaluation kernel together.

---

## 6. What lands first, and what is a restructuring

**Small, already landed:** the hardware-derived budget, the deterministic
admission over the id space, the per-batch ledger, and the extended sweep that
uses all three to spend leftover memory on `(indicator, period)` pairs. Value
today: the search reaches periods it could not reach, sized by the machine.

**Small, next:** the batch-rejection ledger — the accounting for §3.3 — can be
written and tested before the predicate exists. It is the thing that stops the
new loop from becoming a new silent drop.

**A restructuring:** §3.4. The discovery loop today builds one feature cube up
front and then searches it; the batch loop builds a cube per batch and abandons
most of them. That inverts the relationship between the feature build and the
search, touches `discovery.rs` and `genetic/search_engine.rs`, and changes what
a run artifact means (a run is now a sequence of batches with a cursor, not one
search over one cube). It should not be attempted in the same change as
anything else.

**Non-negotiable when it lands:** every batch rejection is logged with a reason
and counted; the working-set width is read from `VocabularyBudget` and never
from a config constant; and the run artifact records the cursor and the batch
list, because a result that cannot say which parameter region produced it is not
a result.
