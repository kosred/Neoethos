# Autoresearch Batch Promotion Bindings Design

## Goal

Make autoresearch promotion evidence preserve the exact identity of every
streaming batch that selected a gene, and reject any incomplete or substituted
binding before the single out-of-sample touch is journalled as spent.

## Approved design

Promotion evidence advances from schema v3 to v4. The top-level artifact keeps
the durable session, sweep, slot, proposal stamp, and frozen
`DatasetReceiptV1`. It no longer carries a flattened feature-name list or a
flattened gene list. Instead, it carries an ordered list of per-batch bindings.

Each binding records its ordinal and source cursor, the exact
`CanonicalSearchInputReceiptV1`, the receipt's recomputed SHA-256, the exact
`CanonicalSearchEvaluatedWindowV1`, the effective
`DiscoveryResult.search_config_hash`, the batch-local effective feature names,
and genes tagged with the same ordinal and cursor. The genes retain their local
indices; the out-of-sample evaluator projects each batch by its own recorded
names.

The proposal stamp and effective search-config hash remain distinct identities.
The writer derives the expected effective hash from the exact charged runtime
configuration handed to discovery, exact-compares every result against it, and
only then writes promotion evidence. It also validates every receipt against the
frozen dataset receipt before persistence.

The loader rejects unknown fields, schema/count/order/tag mismatches, empty
bindings, duplicate or non-monotonic cursors, invalid receipt digests, windows
that are not the exact entire receipt scope, effective config mismatches,
out-of-range gene indices, and any receipt whose direct source bindings or
feature provenance do not validate against the frozen dataset receipt. All of
these checks run before `OosTouchSpent` is appended.

Session and journal v2/v3 contracts are unchanged. The change is Rust-only,
models-free, and does not reconstruct authority from symbol/timeframe display
identity.

## Rejected alternatives

- A flattened canonical portfolio plus parallel batch sidecars leaves genes and
  bindings independently swappable.
- Generic artifact envelopes hide the explicit batch cursor/tag invariants and
  make the promotion-specific fail-closed rules less direct.

## Verification

Strict red-green tests cover a valid multi-batch artifact and failures for a
missing binding, swapped order or cursor, stale digest, changed feature plan,
wrong window, wrong effective config, and a flattened untagged gene. Focused
autoresearch tests run warning-denied, followed by rustfmt and full crate tests.
