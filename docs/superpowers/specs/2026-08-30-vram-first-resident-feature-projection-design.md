# VRAM-first resident feature projection design

**Status:** Approved by the operator on 2026-08-30.

## Goal

Make the canonical native CUDA path fit the GPU that is actually selected. Host
RAM may stage CPU work, but it must never authorize a resident CUDA shape. The
selected device's one sealed free-VRAM snapshot and topology determine the Data
projection, minimum Search reservation, population extent, and launch geometry.

## Root cause

The current five-timeframe recipe resolves 2,660 columns over 3,924,096 rows.
Its CUDA store requires about 82.9 GiB steady and at least 98.5 GiB peak. The
32-GiB RTX 5090 therefore rejects it before allocation. Population auto-sizing
runs after this fixed Data extent and cannot reduce rows or columns. The existing
resident prefilter runs only after the full parent store exists, so it cannot
solve the initial VRAM admission.

## Architecture

Use two deterministic CUDA producer passes with at most one producer batch live.

1. **Screening pass.** Retain the source/HTF working set and one at-most-64-column
   scratch batch. Apply the existing robust-normalization semantics, compute the
   existing first-passage labels/CPCV two-pass-f64 correlations, and retain only
   O(column-count) scores and schema metadata. Stable rank, state/template keeps,
   and timeframe quotas produce an ascending compact-to-parent map. Read back
   only this bounded control map; no feature values cross to the host.
2. **Projection seal.** Seal the selected map, selected routes, exact compact
   extent, and both pass peaks against the original recipe and the single native
   admission snapshot. Reserve the configured minimum population plus one
   scenario before accepting the projection.
3. **Materialization pass.** Replay the identical immutable producer recipe and
   pack only selected columns into the compact bar-major f64/u4 store. Normalize
   and hash the compact store with the existing exact authorities.
4. **Adaptive Search.** Size population/scenarios from the remaining VRAM under
   the same snapshot. Capture SM count and warp size in the existing selected
   device admission pass; kernels use them for launch geometry, while memory and
   time caps choose the maximum population without artificial divisibility.

The shared admission predicate is:

`max(data_peak, data_steady + views + genes + scenarios) + reserve <= free_vram_snapshot`.

## Required invariants

- No allocation proportional to `rows * unfiltered_columns`.
- No RAM-derived CUDA admission and no second free-memory query.
- Producer batch boundaries cannot change scores, order, selected map, values,
  validity, normalization fits, or content identity.
- Preserve pairwise-complete ordered two-pass f64 Pearson, stable parent-index
  tie-breaks, state/template keeps, timeframe quotas, and fail-closed label rules.
- Only a bounded selected index-map/control receipt may cross between passes.
- The bounded readback uses a new V2 orchestration receipt; V1's zero-host-boundary
  identity remains unchanged.
- The CPU route continues to use process/cgroup RAM minus its OS reserve.
- No CPU fallback in the native CUDA path.

## Verification boundary

Focused contracts must first fail on the existing full-store path, then prove:
batch-partition invariance, compact selection identity, exact compact values and
validity versus the small full-store oracle, shared VRAM authority, and absence
of full-cube allocation. Local warning-clean compile precedes one real-card run.
