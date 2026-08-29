# Exact Classic-TA CUDA Routing Design

## Scope

The existing canonical feature schema is preserved. `GpuOnly` does not create a
smaller research vocabulary and never computes an admitted Classic/vector-ta
output on the CPU or through f32. The five non-Classic producer families remain
outside this policy; they are explicit producer implementations, not a fallback
inside the Classic lane.

## Current failure

`compute_classic_ta_columns_sized_report` rejects every `GpuOnly` call before it
builds the actual vocabulary plan. Its gap scan is broader and narrower than the
real request at the same time: it scans raw registry outputs for every
`ALL_INDICATORS` row, but does not account for budget admission, production
output exclusions, the library-declared pattern outputs, historical periods, or the installed
extended working set. Removing that rejection would still run the base and both
sweeps on the CPU because no production code constructs `GpuIndicatorEngine`.

## Design

Planning remains allocation-free and happens before either CPU or CUDA work.
The planner captures the exact post-budget, post-working-set Classic request as
ordered typed nodes:

- base nodes: admitted indicator plus each production output identity;
- historical nodes: the existing period plan in its stable emission order;
- extended nodes: the installed batch or budget-prefix groups in their stable
  emission order;
- pattern outputs: each typed discrete column, never a fake f64 matrix alias.

Each node carries its emitted column name, indicator/output identity, parameter
request, value kind, and expected row count. Preflight resolves every node to an
exact CUDA route. Any missing route returns one ordered, complete error before
`GpuIndicatorEngine::new` and before CPU dispatch. It is forbidden to delete,
rename, or defer a node merely because its CUDA route is missing.

After successful preflight, one `GpuIndicatorEngine` owns the frame upload,
context, and stream. It launches nodes in plan order while retaining device
results. At the existing `FeatureFrame` boundary, f64 matrices are downloaded
without narrowing and converted to the same `FeatureColumnF64` values and
validity classifications as the CPU reference. A launch, shape, download, or
validity mismatch is fatal; no node is retried on CPU.

The execution report records the exact planned/admitted/deferred identities and
the exact produced count from this same plan. The feature contract continues to
derive final outputs from the produced canonical columns, so receipts cannot
claim a different schema from the one executed.

## Verification boundary

Source contracts first prove that `GpuOnly` no longer contains an unconditional
rejection or CPU dispatch and that production owns a single engine. Pure planner
tests then prove ordering, budget/working-set fidelity, output exclusions, and
complete gap reporting without a card. Cargo compilation waits for the shared
lane. Completion additionally requires a real RTX 3090 run proving launches,
telemetry, exact f64 parity, and fail-loud behavior for an injected unsupported
output.
