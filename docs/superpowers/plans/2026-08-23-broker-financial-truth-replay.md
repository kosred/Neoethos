# Broker Financial Truth Replay Implementation Plan

> Execute in strict TDD chunks. Do not advance a chunk without its RED,
> focused verification, source hashes and review checkpoint.

**Goal:** Replace the permanently unavailable financial-search boundary with a
run-scoped capability backed only by exact immutable broker evidence and an
exact quote replay evaluator.

**Architecture:** A dependency-leaf evidence authority is produced by
`neoethos-broker-history`, semantically validated below core/search/app, and
passed explicitly to a new search evaluator. Research-only gross-R remains a
separate non-promotable lane.

**Stack:** Rust 2024, serde/serde_json, SHA-256, Vortex 0.67 artifacts, existing
canonical dataset/search receipts and existing cTrader transport.

---

## Chunk 1: leaf contracts, immutable store and core re-export

**Files**

- Create `crates/neoethos-broker-truth/Cargo.toml`
- Create `crates/neoethos-broker-truth/src/{lib,contracts,store,gate}.rs`
- Create `crates/neoethos-broker-truth/tests/evidence_gate.rs`
- Modify root `Cargo.toml`
- Modify `crates/neoethos-core/Cargo.toml`
- Modify `crates/neoethos-core/src/{broker_truth,lib}.rs`

**RED**

- Require exact receipt reopen to reject same-length artifact tampering.
- Require a different search receipt binding to reject bundle reuse.
- Prove a synthetic storage fixture cannot create/install a permit.

**Implementation**

- Add deny-unknown-fields V1 binding/manifest/receipt types.
- Require all five evidence families and explicit Bid/Ask/conversion routes.
- Publish to an exact content-addressed directory with no mutable selector.
- On reopen, verify directory/file types, exact file set, manifest/artifact
  lengths/hashes and exact run binding.
- Move the existing permanent-refusal gate to the leaf and re-export it from
  core. Do not add a semantic success state in this chunk.

**Verification**

- Direct `rustfmt --check`, scoped `git diff --check`, static no-bypass/source
  contract and SHA-256 census.
- Cargo remains explicitly unrun until root approval.

## Chunk 2: broker-history evidence producer

**Owned area**

- `crates/neoethos-broker-history/src/` new disjoint capture modules and tests.
- Add only the leaf dependency to broker-history Cargo.

**RED**

- Bid and Ask page requests retain explicit side and every raw response/page
  boundary; missing/truncated/overlapping/out-of-order pages refuse publication.
- Every required conversion route is captured for the exact account/window.
- Full symbol/account money contracts, position unrealized PnL and close/deal
  evidence are retained raw and decoded.
- Cancellation or capture failure leaves no published bundle.

**Implementation**

- Extend the existing authenticated broker-history service; do not create a
  second transport or promote mutable symbol JSON.
- Encode exact raw and decoded tables as Vortex artifacts, publish once through
  the leaf store and return the exact bundle receipt.

## Chunk 3: semantic validator and run-scoped capability

**RED**

- Refuse unknown schemas/fields, raw-decoded mismatch, wrong side, changed
  account/symbol/window/search receipt, stale/unsynchronized quotes, missing
  conversion paths, omitted money digits/contract fields, PnL mismatch and
  close/deal reconciliation mismatch.
- A reviewed captured real-broker fixture is the first success fixture. No
  synthetic success fixture or test constructor.

**Implementation**

- Decode exact Vortex schemas and compare retained raw envelopes to decoded
  rows.
- Construct capability only inside the complete validator; permit owns the
  validated bundle view and operation scope.

## Chunk 4: exact quote replay evaluator

**Owned area**

- New disjoint `neoethos-search` exact replay modules and tests.

**RED**

- Buy entry/exit uses Ask/Bid and sell entry/exit uses Bid/Ask.
- Conversion uses the exact directed leg and quote side at event time.
- Missing initial quote, evidenced staleness breach, closed session, contract
  omission or exhausted event stream fails the run.
- Exact replay cannot call legacy scalar `evaluation_config(price_hint)` or
  OHLC spread/commission/swap adapters.

**Implementation**

- Stream synchronized quote events under the validated replay-rule contract.
- Compute fills, commission, swap, conversion and realized PnL from exact
  contract/quote rows and preserve reconciliation provenance in results.

## Chunk 5: app/CLI orchestration

**RED**

- Financial discovery requires one exact broker-bundle receipt before loading
  search inputs.
- Receipt/binding mismatch fails before launching work.
- No implicit `current/latest`, fallback, environment bypass or CTrader-identity
  shortcut.
- Historical research CLI remains `ResearchOnly` and
  `NotPromotionEligible`.

**Implementation**

- Pass receipt -> semantic validator -> capability -> exact evaluator config
  explicitly through each financial entrypoint.

## Chunk 6: end-to-end evidence and cleanup

- Run focused crate tests, standalone feature combinations and full relevant
  workspaces with complete logs after approval.
- Capture and replay real broker evidence only with explicit live-broker GO.
- Compare evaluator totals to broker PnL/deals at exact money precision.
- After replacement proof, remove superseded scalar/OHLC financial success
  routes, stale flags/aliases/tests/docs. Preserve only explicit versioned
  fail-closed migrations.
