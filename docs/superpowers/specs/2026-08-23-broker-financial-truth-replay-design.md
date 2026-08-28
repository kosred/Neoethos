# Broker Financial Truth Replay Design

**Date:** 2026-08-23  
**Decision:** A — explicit run-scoped immutable broker replay bundle  
**Status:** approved architecture; implementation is deliberately chunked

## Problem

`BrokerFinancialTruthCapabilityV1::require` correctly fails closed, but no
production path can supply the five evidence classes it names. Existing
financial evaluation is OHLC plus operator/config scalars. Opening the gate for
that evaluator would fabricate broker truth rather than supply it.

The separate strict historical-search CLI remains `ResearchOnly`,
`NotPromotionEligible`, and gross reference-R. It is not a broker-real PnL
lane and must not gain a financial-truth permit.

## Dependency boundary

Create `neoethos-broker-truth` below core, broker-history, search and app.

- The leaf owns versioned evidence contracts, exact receipts, immutable bundle
  storage, semantic validation and the eventual run-scoped capability.
- `neoethos-core` re-exports the leaf for compatibility; it does not install
  global state.
- `neoethos-broker-history` is the sole production producer of retained raw
  cTrader envelopes and decoded Vortex tables.
- `neoethos-search` owns a new exact quote replay evaluator. The legacy
  scalar/OHLC evaluator remains sealed.
- App/CLI orchestration selects one exact broker bundle receipt and passes the
  validated run-scoped authority explicitly.

This direction avoids a core -> broker-history -> core dependency cycle and
prevents app/search from becoming evidence authorities.

## Bundle contract

One bundle is bound to all of the following:

- canonical cTrader dataset identity;
- canonical search input receipt SHA-256;
- exact half-open evaluated window;
- primary base/quote and account assets;
- explicit primary Bid and Ask captures;
- explicit conversion routes, including an explicit zero-leg identity route
  only when source and destination assets are exactly the same;
- retained raw page/envelope tables and decoded tables;
- quote-session/synchronization rule observations and decoded replay rules;
- exact `ProtoOASymbol` response/contract tables;
- broker position unrealized-PnL response/decoded tables; and
- close/deal response/reconciliation tables.

Every artifact is a named Vortex file with an exact byte length, row count,
schema discriminator and SHA-256. The manifest is deny-unknown-fields and the
bundle directory name is `bft1-<manifest SHA-256>`.

## Integrity is not authority

The immutable store proves only that an exact receipt reopens the exact
manifest/file set and bytes. It explicitly does not produce a permit.

The later semantic validator must reopen the Vortex schemas, re-decode retained
raw envelopes, compare raw and decoded rows, prove Bid/Ask and conversion
synchronization under evidenced rules, validate the full symbol/account money
contract, and prove broker-PnL and close/deal reconciliation. Only that complete
path may construct a run-scoped capability.

Synthetic fixtures may test storage refusal and tamper detection. They may
never yield a production capability or permit.

## Non-negotiable invariants

- No global registry or mutable `current/latest` success pointer.
- No public permit constructor, test-only permit, environment switch or config
  boolean.
- No legacy scalar/OHLC promotion path after validation.
- No inferred quote side, default spread/commission/swap/pip/money digits,
  reconstructed conversion leg or repaired ordering.
- No mutable `data/symbol_metadata.json` authority.
- No fallback from a missing/mismatched/corrupt bundle.
- Unknown schema versions, unknown fields, extra files, symlinks and mismatched
  hashes fail closed.

## Delivery sequence

1. Leaf contracts/store plus core re-export; the release gate remains closed.
2. Broker-history exact evidence capture and immutable bundle publication.
3. Leaf semantic Vortex validator and run-scoped permit.
4. Search exact quote replay evaluator, with no legacy financial adapter.
5. App/CLI exact receipt selection and explicit authority threading.
6. Promotion/live parity proof and removal of superseded scalar paths only
   after the replacement is proven.

