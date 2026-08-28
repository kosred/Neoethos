# Canonical Trendbar Bulk Acquisition Design

## Scope

NeoEthos must acquire broker history only through direct cTrader
`ProtoOAGetTrendbarsReq` canonical timeframes. Tick/Bid/Ask capture is not a
dataset source for search or model training. The existing financial-truth
quote path remains sealed and uninvoked as a separate promotion boundary.

The first bounded repair is same-session `hasMore` pagination. Later slices add
an exact account/environment authority and an immutable resumable
symbol/timeframe matrix. Each slice must be independently RED-first and
warning-clean before the next begins.

## Current Defect

`neoethos-historical-fetch` correctly publishes one content-addressed direct
timeframe generation and refuses a response marked `hasMore`. Because cTrader
states that the backend chunk size is configuration-dependent, refusal is
honest but cannot guarantee complete ten-year acquisition. The current CLI
also selects an enabled or first configured account, defaults the upper bound
to wall-clock now, has no durable page checkpoint, and emits no multi-symbol
matrix authority.

## Architecture

### Same-session paging

One logical timeframe chunk may contain multiple broker responses. On
`hasMore=true`, the next request retains the same lower bound and moves the
inclusive wire upper bound strictly before the oldest returned bar. Every
subpage stays on the already authenticated session and uses a unique
`clientMsgId`. Pages must be non-empty, strictly increasing internally,
strictly older than the previously accepted subpage, range-bound, and exact in
account/symbol/timeframe identity. No sorting, deduplication, reconstruction,
CPU fallback, or partial publication is allowed.

The complete logical chunk is materialized as one oldest-first canonical chunk
only after the terminal `hasMore=false` response. Cancellation or a bounded
page-limit failure leaves no published generation.

### Exact acquisition authority

A later versioned plan binds explicit broker environment, exact account id,
fixed lower bound `2016-01-01T00:00:00Z`, explicit exclusive upper bound,
ordered unique symbols, ordered typed canonical timeframes, and the paging
policy version. Credentials remain internal and non-debuggable; there is no
enabled/first-account, current-generation, environment, or resampling fallback.

### Durable matrix resume

Every completed cell is an immutable receipt that reopens and hash-verifies its
Vortex generation. A content-addressed checkpoint links the exact plan and the
ordered completed cells. Resume may skip only cells whose receipts reopen and
match the plan exactly. Cancellation may preserve completed immutable cells but
cannot emit a final matrix authority.

For each symbol, existing `CanonicalDatasetSeriesReceiptV1` binds all direct
timeframe generations. A new strict matrix receipt binds every required symbol
series exactly once. Search and training accept only explicit final
series/matrix receipts, never checkpoint or current-state discovery.

## Error Handling

- `hasMore=true` with an empty page, non-retreating cursor, overlap, identity
  drift, or page-limit exhaustion fails before publication.
- Missing broker bars remain visible through requested/returned provenance;
  the acquisition report must distinguish requested ten-year scope from actual
  broker coverage.
- Rate limiting remains below the official five historical requests per second
  per connection.
- All paths are cancellation-aware before every request and before publication.

## Verification

1. Private connector tests prove same-session paging, exact cursors, terminal
   completeness, cancellation, and no partial generation.
2. Warning-denied broker-history tests prove exact account/plan/checkpoint and
   matrix receipt contracts.
3. A real broker smoke captures one bounded direct timeframe, reopens the
   immutable generation, and compares receipt row/range/hash values.
4. The production matrix run records every symbol/timeframe receipt and final
   matrix authority before search or training starts.

