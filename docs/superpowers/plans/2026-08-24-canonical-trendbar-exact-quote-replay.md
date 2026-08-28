# Canonical Trendbar Broker Quote-Validated Replay Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep every predictive input broker-native canonical-timeframe data while using historical Bid/Ask ticks only as immutable execution evidence for causal quoted spread and executable-side reference prices.

**Architecture:** Discovery features and model inputs never consume ticks or tick-derived candles. A separate broker-truth replay lane binds complete per-side Bid and Ask evidence for bounded, locked final-evaluation windows to the same account, symbol, canonical dataset, and half-open window. Signals are produced from closed canonical bars; execution references consume only the first executable-side quote after the permitted decision time. A versioned reviewed top-of-book reconstruction rule defines causal book seeding, cross-side merge, same-timestamp handling, staleness, and gap-through behavior. Historical quotes prove the broker's quoted spread and side price, not realized slippage for a hypothetical order: latency/slippage remains a separately named, immutable per-fill policy unless an actual broker deal receipt exists. Intrabar stop/target validation replays quote events only in the locked outer-holdout and one-touch autoresearch OOS windows.

**Tech Stack:** Rust workspace, cTrader Open API quote history, Vortex immutable evidence, SHA-256 receipts, CPU reference replay, CUDA candidate search, serde deny-unknown-fields contracts.

---

## Chunk 1: Preserve the data/execution boundary

### Task 1: Freeze source-level isolation

**Files:**
- Modify: `crates/neoethos-cli/tests/canonical_trendbar_full_run_source_contract.rs`
- Add/modify: a focused source contract under `crates/neoethos-search/tests/`

- [ ] Require feature construction to use only `CanonicalDatasetSeriesReceiptV1` and direct broker `CanonicalTimeframe` generations.
- [ ] Forbid quote/tick rows from feature matrices, normalization, indicators, labels, and model artifacts.
- [ ] Permit quote evidence only at a new explicitly named execution-replay boundary.
- [ ] Require `ResearchOnly`/`NotPromotionEligible` until exact quote replay and every remaining broker-truth class pass.
- [ ] Run the source test and record RED because no exact execution-replay boundary exists.

### Task 2: Correct the existing scalar assumption provenance

**Files:**
- Modify: `crates/neoethos-cli/src/canonical_full_run.rs`
- Modify: `crates/neoethos-cli/tests/canonical_trendbar_full_run_source_contract.rs`
- Modify: `crates/neoethos-core/src/symbol_metadata.rs`

- [ ] Remove the false complete-cost identity `neoethos.exact-broker-canonical-d1-costs.v1`.
- [ ] Persist separate operator-assumed spread and per-fill slippage fields for broad-search screening; do not call either broker-exact.
- [ ] Keep exact broker attribution only on fields proven by exact broker contracts/receipts; compute commission and conversion per fill in account-currency units.
- [ ] Remove the false `ProtoOASymbol::spread` source claim.
- [ ] Version the changed wire and fail closed on old artifacts.
- [ ] Keep this scalar policy screening-only; it cannot satisfy final quote-validated replay or promotion.

## Chunk 2: Causal bounded Bid/Ask execution evidence

### Task 3: Define exact decision/fill contracts

**Files:**
- Create: `crates/neoethos-broker-truth/src/execution_replay_v1.rs`
- Modify: `crates/neoethos-broker-truth/src/lib.rs`
- Create: `crates/neoethos-broker-truth/tests/execution_replay_v1_contract.rs`

- [ ] Write RED tests proving a signal from canonical row `i-1` cannot execute before `decision_at = timestamps[i]`, the next canonical bar-open.
- [ ] Define the entry reference as the first eligible executable-side quote at or after `decision_at`: long uses Ask, short uses Bid.
- [ ] Define exit references symmetrically: long uses Bid, short uses Ask.
- [ ] Permit an opposite-side quote only as a causal pre-decision/pre-event book seed under an explicit maximum age; it can establish broker spread but can never become the fill price.
- [ ] Reject pre-decision fills, crossed books, missing/stale sides, non-finite/non-positive prices, non-monotonic per-side events, and mismatched account/symbol/window receipts.
- [ ] Bind the reviewed cross-side merge and same-timestamp rule into the receipt; never invent an ordering not proven by the two broker streams.
- [ ] Make maximum entry wait, quote staleness, evidence coverage, latency, and per-fill slippage policy explicit, typed, and receipt-bound; missing/incomplete evidence is an error, not a no-fill.
- [ ] Never describe a policy-derived hypothetical execution as an exact broker fill; only an actual broker deal receipt may make that claim.
- [ ] Ensure no API exposes tick values to feature/model consumers.

### Task 4: Define exact intrabar stop/target replay

**Files:**
- Extend: `crates/neoethos-broker-truth/src/execution_replay_v1.rs`
- Extend: `crates/neoethos-broker-truth/tests/execution_replay_v1_contract.rs`

- [ ] RED: one OHLC bar touches stop and target but the quote sequence proves only one occurred first.
- [ ] Replay synchronized quotes in strict broker event order over the trade window.
- [ ] Apply stops/targets on the executable side of the book, including gaps through the level.
- [ ] Update trailing thresholds only from completed canonical bars; quote events cannot change features, signals, or strategy parameters.
- [ ] Reject an evaluation that attempts to infer intrabar order from OHLC high/low alone.
- [ ] Record exact signal, decision, first-eligible-quote, trigger, and exit-reference timestamps plus broker Bid/Ask prices and the distinct policy-derived simulated fill prices in the replay receipt.

## Chunk 3: Acquisition and bounded window planning

### Task 5: Capture complete quotes for bounded locked evaluation windows

**Files:**
- Extend: `crates/neoethos-broker-truth-acquire/src/lib.rs`
- Extend: `crates/neoethos-broker-history/src/production_broker_truth_v2.rs`
- Add focused tests beside existing acquisition/preflight contracts

- [ ] Accept only the locked final portfolio's explicit outer-holdout window and the single autoresearch one-touch OOS window; GA/CPCV screening never requests quotes.
- [ ] Add explicit causal book-seed and exit-wait padding, then bind the exact half-open coverage window and padding policy into the receipt.
- [ ] Bind every requested window to account, symbol, locked portfolio/gene SHA-256, canonical dataset receipt, and replay-policy version.
- [ ] Reuse one authenticated broker session and paginate exact Bid and Ask history independently.
- [ ] Require complete continuous per-side coverage in bounded broker-sized chunks, explicit terminal-page proof, and explicit zero-row-window proof; never fabricate a tick to make an empty artifact serializable.
- [ ] Publish immutable raw+decoded evidence and retain the reviewed top-of-book reconstruction rule; no current/default discovery.
- [ ] Cancellation preserves completed immutable chunks but cannot emit a complete authority from a partial set.

## Chunk 4: Search and OOS integration

### Task 6: Add a finalist-only quote-validated replay gate

**Files:**
- Create: `crates/neoethos-search/src/exact_quote_replay.rs`
- Modify: `crates/neoethos-search/src/lib.rs`
- Modify: `crates/neoethos-search/src/discovery.rs`
- Modify: `crates/neoethos-search/src/validation.rs`
- Modify: `crates/neoethos-search/src/validation_snapshot.rs`
- Modify: `crates/neoethos-search/src/live_portfolio.rs`
- Add: `crates/neoethos-search/tests/exact_quote_replay_contract.rs`
- Add: `crates/neoethos-search/tests/exact_quote_finalist_integration.rs`
- Add: `crates/neoethos-autoresearch/tests/exact_quote_oos_gate.rs`

- [ ] Broad CUDA discovery may rank under an explicitly conservative, hashed screening-cost envelope only.
- [ ] Keep GA and CPCV entirely canonical-trendbar-only; do not quote-replay candidates or CPCV survivors.
- [ ] After the portfolio is locked, quote-validate each final gene once on the outer holdout; separately replay only the one tagged autoresearch OOS touch.
- [ ] Recompute the complete metric tuple from broker quote references plus the bound latency/slippage policy; do not patch only net profit or spread.
- [ ] Treat quoted spread as already embedded in Ask/Bid execution references and never charge a scalar spread a second time.
- [ ] Reuse one immutable exact ledger per gene for forward metrics and prop-firm risk summaries instead of running two divergent simulations.
- [ ] Reject candidates whose trade events cannot be replayed exactly, including missing/stale quote windows.
- [ ] Bind the quote-evidence receipt, execution-policy receipt, replay ledger, and metric tuple into the candidate/artifact identity.
- [ ] Require the new exact receipt/ledger artifacts as a separate promotion-evidence set; legacy bar-derived V2 forward/prop artifacts remain `ResearchOnly` and cannot unlock promotion alone.
- [ ] In autoresearch, validate quote receipt/bindings before consuming `OosTouchSpent`, and decode/replay quotes only after the single touch is authorized.
- [ ] Keep this lane unavailable to promotion until commission/swap/conversion/deal reconciliation capability is also complete.

### Task 7: Prove CPU replay, CUDA-search isolation, and look-ahead boundaries

**Files:**
- Add focused CPU reference and CUDA routing tests in `neoethos-search`

- [ ] Prove signals are identical before/after quote evidence is attached; ticks cannot affect features or signal generation.
- [ ] Prove first-at/after fill selection never reads a predecision or future feature row.
- [ ] Compare broad-search candidate ordering separately from exact-replayed finalist metrics; never claim they are the same evaluation.
- [ ] Keep sequential quote replay on the CPU: finalists are few, event ordering is stateful, and this boundary must not modify or duplicate the mass-search CUDA kernels.
- [ ] Run warning-denied CPU tests plus source contracts proving the CUDA search inputs and kernels remain quote-free.

## Completion gates

- [ ] Canonical OHLCV remains direct broker timeframe data; no tick candle/resampling path exists.
- [ ] Bid/Ask ticks are reachable only through execution evidence and never through features/models.
- [ ] Every accepted entry/exit has exact executable-side quote provenance and separately identified policy-derived fill semantics.
- [ ] Every reported spread is reconstructed from causal broker Bid/Ask evidence under the reviewed merge/staleness rule, never from a scalar advertised as broker-exact.
- [ ] Intrabar ordering is proven from quote sequence or the trade is unsupported.
- [ ] Missing/stale/mismatched evidence fails before metrics/publication.
- [ ] Broad-search assumed-cost results and exact-replay results are separately named and hashed.
- [ ] Legacy V2 bar-derived validation artifacts alone can never satisfy promotion.
- [ ] Full INFO -> WARNING -> ERROR logs and immutable receipts are retained.
