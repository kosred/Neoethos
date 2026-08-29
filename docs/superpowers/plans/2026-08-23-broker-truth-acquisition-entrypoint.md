# Broker Truth Acquisition Entrypoint Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-shot, fail-closed production acquisition command that binds an exact canonical Search receipt/root/scope/window and immutable reviewed cTrader inputs to a real V2 broker-truth bundle without verifying semantic trust or minting a permit.

**Architecture:** A new thin `neoethos-broker-truth-acquire` crate sits above Search, broker-history, and the broker-truth leaf. The leaf owns only versioned content-addressed acquisition authority/link contracts and storage; broker-history owns the exact authenticated same-session runner; the bridge owns strict non-secret input preflight and orchestration. Review/protocol/trust bytes are frozen and hash-linked, but signature/policy validation remains blocked for the later semantic chunk.

**Tech Stack:** Rust 2024 workspace, serde/serde_json, sha2, clap, existing NeoEthos canonical dataset receipts, existing cTrader JSON-WSS session, Vortex 0.67 evidence, filesystem content-addressed publication.

---

**Root-authorized RED boundary:** The shared Cargo lane is serialized by root. For every remaining production checkpoint, write the wished-for test first, run its exact root-approved warning-denied Cargo command, record the expected compiler/test RED, and stop. Production implementation is prohibited until that executable RED exists and root gives a separate repair GO. A source-absence probe is useful census evidence only; it is never RED. Task 1 predates this correction and has a warning-clean GREEN but no retrospective RED claim. Every later task follows the corrected order below.

## Chunk 1: Leaf acquisition authority and immutable store

### Task 1: Define versioned evidence-only acquisition contracts

**Files:**
- Create: `crates/neoethos-broker-truth/tests/acquisition_authority_v1_contract.rs`
- Create: `crates/neoethos-broker-truth/src/acquisition_v1.rs`
- Modify: `crates/neoethos-broker-truth/src/lib.rs`

- [ ] **Step 1: Write the failing contract test**

The test must describe these public types before they exist:

```rust
use neoethos_broker_truth::{
    BrokerTruthAcquisitionArtifactRoleV1,
    BrokerTruthAcquisitionArtifactV1,
    BrokerTruthAcquisitionAuthorityManifestV1,
    BrokerTruthReviewedSynchronizationBindingV1,
    EvidenceWindowV1,
    ReviewedQuoteReplayRuleIdentityV2,
};
```

Construct one manifest with exactly one canonical receipt, evaluated scope, exact-root verification receipt, explicit scope-to-half-open-window binding, capture plan, review record, protocol evidence, trust root, and an ordered observations/rules pair. Assert:

- semantic status is permanently `UnvalidatedEvidenceOnly`;
- promotion eligibility is permanently `NotPromotionEligible`;
- the manifest has no success-permit or capability conversion;
- receipt/scope/root-verification/window-binding/plan/trust digests are exact lowercase SHA-256;
- synchronization account, symbol, window, ordinal, and review identity are preserved.

- [x] **Step 2: Historical Task 1 source census (not RED)**

Run:

```powershell
rg -n "BrokerTruthAcquisitionAuthorityManifestV1" crates/neoethos-broker-truth/src
```

Historical observation: the production definition was absent when the test was written. This is retained only as source-order evidence and is not called RED. Task 1 subsequently reached warning-clean GREEN; the corrected executable-RED rule applies to every remaining production task.

- [ ] **Step 3: Add the minimal contracts**

Implement in `acquisition_v1.rs`:

```rust
pub enum BrokerTruthAcquisitionSemanticStatusV1 {
    UnvalidatedEvidenceOnly,
}

pub enum BrokerTruthAcquisitionPromotionEligibilityV1 {
    NotPromotionEligible,
}

pub enum BrokerTruthAcquisitionArtifactRoleV1 {
    CanonicalSearchInputReceipt,
    CanonicalSearchArtifactScope,
    CanonicalRootVerificationReceipt,
    CanonicalScopeWindowBinding,
    CapturePlan,
    ReviewRecord,
    ProtocolEvidence,
    TrustRoot,
    QuoteSessionObservations { ordinal: u32 },
    ReviewedQuoteReplayRules { ordinal: u32 },
}
```

Add strict `#[serde(deny_unknown_fields)]` wire forms, bounded safe basenames, exact byte length, lowercase SHA-256, unique roles/paths, contiguous synchronization ordinals, one observations/rules pair per binding, and exact digest cross-links. The manifest carries both the domain hash of the canonical receipt/scope and the exact byte hash of a versioned root-verification receipt that lists every exact-opened generation identity/manifest/generation/Vortex binding. It also carries the exact byte hash of a versioned window-binding receipt containing scope identity, role, row bounds, first/last consumed bar timestamps, the explicit `EvidenceWindowV1`, and reviewed window-policy id. Constructors validate; deserialization revalidates; no default implementation and no capability/permit API.

- [ ] **Step 4: Export only the evidence contract**

Add `mod acquisition_v1;` and explicit re-exports in `lib.rs`. Do not change `gate.rs` or `semantic_v2.rs`.

- [ ] **Step 5: Freeze source and request the focused leaf gate**

Future command, only after root GO:

```powershell
$env:RUSTFLAGS='-Dwarnings'; cargo test -p neoethos-broker-truth --test acquisition_authority_v1_contract -- --nocapture
```

Expected: all tests pass, zero warnings/errors. Classify the complete log INFO then WARNING then ERROR.

### Task 2: Publish and reopen immutable authority/link receipts

**Files:**
- Extend test: `crates/neoethos-broker-truth/tests/acquisition_authority_v1_contract.rs`
- Create test: `crates/neoethos-broker-truth/tests/acquisition_store_v1_contract.rs`
- Create: `crates/neoethos-broker-truth/src/acquisition_store_v1.rs`
- Modify: `crates/neoethos-broker-truth/src/lib.rs`

- [x] **Step 1: Write tamper/refusal REDs**

First extend the authority RED to require the root-verification and scope-window-binding artifacts/digests described in Task 1. Then cover exact source set, modified byte, truncated file, symlink source, extra published file, unsafe basename, existing mismatched content-addressed directory, link to a missing authority package, link to a missing BFT2 bundle, and absence of any mutable `current` path.

- [x] **Step 2: Record the executable leaf RED before production**

Confirm the desired store symbols are absent with `rg` as census only. Then request and run exactly:

```powershell
$env:RUSTFLAGS='-Dwarnings'; cargo test -p neoethos-broker-truth --test acquisition_authority_v1_contract --test acquisition_store_v1_contract -- --nocapture
```

Capture the complete log under `target/audit-logs/financial-truth-gate/`, classify INFO then WARNING then ERROR, and stop at the first missing-symbol/compiler or failing-test RED. Do not create `acquisition_store_v1.rs` or repair the authority contract until root separately approves the repair.

- [x] **Step 3: Implement minimal content-addressed storage**

Add:

- `BrokerTruthAcquisitionAuthorityReceiptV1` with `bfta1-<manifest_sha256>`;
- `BrokerTruthAcquisitionLinkManifestV1` with schema version 1, exact authority receipt, exact `BrokerFinancialTruthBundleReceiptV2`, exact `BrokerFinancialTruthBindingV1`, and fixed evidence-only/not-promotion-eligible status;
- `BrokerTruthAcquisitionLinkReceiptV1` with `bftl1-<manifest_sha256>`;
- `BrokerTruthAcquisitionStoreV1::{publish_authority,open_authority,publish_link,open_link}`;
- atomic private staging + rename, bounded regular-file reads, exact file-set reopen, no overwrite, no current pointer;
- authority manifest filename `broker-truth-acquisition-authority.manifest.json` and link filename `broker-truth-acquisition-link.manifest.json`;
- exact `publish_link(&authority_receipt, &bft2_receipt, &expected_binding)` inputs. It must reopen the authority and call `BrokerFinancialTruthBundleStoreV1::open_exact_v2(bft2_receipt, expected_binding)` under the same explicit store root before link publication.

- [x] **Step 4: Freeze source and request one replacement leaf GREEN gate**

Use the same focused command after explicit GO. Do not start history edits until this checkpoint is accepted.

## Chunk 2: Broker-history exact authenticated same-session runner

### Task 3: Add a secret-opaque production runner

**Files:**
- Create: `crates/neoethos-broker-history/src/production_broker_truth_v2_tests.rs`
- Create: `crates/neoethos-broker-history/src/production_broker_truth_v2.rs`
- Modify: `crates/neoethos-broker-history/src/lib.rs`
- Modify: `crates/neoethos-broker-history/src/service.rs` only to add the private exact-account credential loader used by the runner; no existing defaulting API changes.

- [x] **Step 1: Write runner REDs**

The public surface contains no factory or transport trait. Specify these exact non-secret APIs before production exists:

```rust
pub struct ProductionBrokerTruthCaptureRequestV2 {
    /* private, constructor-validated non-secret fields only:
       exact environment/account, authority receipt, capture request,
       reviewed synchronizations, work parent and store root */
}
pub struct ProductionBrokerTruthCaptureOutcomeV2 {
    /* private BFT2 receipt with getter */
}
pub enum ProductionBrokerTruthCaptureStageV2 { Admission, Credentials, Connect, ApplicationAuth, AccountAuth, Adapter, Publication }
pub enum ProductionBrokerTruthCaptureErrorCodeV2 {
    ConfigurationMismatch,
    CredentialsUnavailable,
    TransportFailed,
    AuthenticationFailed,
    CaptureFailed,
    PublicationFailed,
    Cancelled,
}
pub struct ProductionBrokerTruthCaptureErrorV2 { /* stage + code + fixed sanitized detail */ }
pub struct ProductionBrokerTruthCancellationV2 { /* run-scoped Arc<AtomicU8>; no registry */ }
pub fn capture_production_broker_financial_truth_v2(
    request: ProductionBrokerTruthCaptureRequestV2,
    cancellation: &ProductionBrokerTruthCancellationV2,
) -> Result<ProductionBrokerTruthCaptureOutcomeV2, ProductionBrokerTruthCaptureErrorV2>;
```

The implementation has a crate-private, credential-opaque same-object state machine, tested by unit tests in `production_broker_truth_v2.rs` rather than exported for integration tests:

```rust
trait CTraderBrokerTruthAuthenticationWireV2 {
    type Session: CTraderBrokerTruthSameSessionV2;
    fn connect(&mut self, endpoint_host: &str) -> Result<(), OpaqueAuthenticationFailureV2>;
    fn application_auth(&mut self) -> Result<(), OpaqueAuthenticationFailureV2>;
    fn exact_account_auth(
        &mut self,
        expected_account_id: i64,
    ) -> Result<(), OpaqueAuthenticationFailureV2>;
    fn into_authenticated_session(self) -> Result<Self::Session, OpaqueAuthenticationFailureV2>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OpaqueAuthenticationFailureV2;

fn establish_exact_authenticated_session_v2<W: CTraderBrokerTruthAuthenticationWireV2>(
    wire: W,
    endpoint_host: &str,
    expected_account_id: i64,
    cancellation: &ProductionBrokerTruthCancellationV2,
) -> Result<W::Session, ProductionBrokerTruthCaptureErrorV2>;
```

The concrete wire privately owns non-`Debug` credential material plus one optional `ProductionCTraderOpenApiSession`. `connect` creates that single session; application auth and exact-account auth send on that same session; `into_authenticated_session` consumes the wire only after all three stages. A private exact-account loader takes the requested environment/account, rejects any mismatch, and never selects `enabled_for_execution`, `.first()`, another account, or a default. Unit spies record the stage order and a connection identity without receiving secret arguments. Assert:

- configured environment/account must exactly match the receipt-derived request before connect;
- no enabled-account/first-account fallback;
- exactly one session connect, then application auth, exact account auth, and every adapter RPC on that session;
- `CTraderBrokerTruthAdapterV2` gets fixed DealList `maxRows=100` and `returnProtectionOrders=true`;
- credential values never cross the public runner API and authentication failures return sanitized stage codes;
- cancellation before publication produces no BFT2 receipt.

The request constructor, not public fields, enforces environment/account/binding equality. DealList `maxRows=100`, `returnProtectionOrders=true`, and the client-message namespace derived from the authority manifest digest are internal constants/derivations, not caller choices. `ProductionBrokerTruthCancellationV2` is lexical and run-scoped: its atomic `begin_publication` transition yields a private guard; cancel-before-transition stops the next stage, while cancel-after-transition reports `PublicationInProgress`. It has no process-global registry or reuse across runs.

- [x] **Step 2: Record the executable history RED before production**

Wire `production_broker_truth_v2_tests.rs` into the library under `#[cfg(test)]`, then confirm the public runner and private state-machine symbols are absent as census only. Request exactly the command below, record the expected missing-symbol/compiler RED, and stop before creating `production_broker_truth_v2.rs`:

```powershell
$env:RUSTFLAGS='-Dwarnings'; cargo test -p neoethos-broker-history --lib production_broker_truth_v2_tests -- --nocapture
```

- [x] **Step 3: Implement the minimal runner**

Keep credential loading private. Select the exact configured account required by the request; do not call the existing defaulting public loader. The concrete private wire reuses `ProductionCTraderOpenApiTransport::connect_session`, the exact application/account auth request builders, response correlation, and `ProductionCTraderOpenApiSession`. Derive the adapter namespace from the authority digest, borrow the authenticated session into `CTraderBrokerTruthAdapterV2`, and call `capture_and_publish_broker_financial_truth_v2`. The only test double is the private wire spy; no public factory can return an unaudited already-authenticated session.

Implement a lexical/run-scoped cancellation state whose atomic `begin_publication` transition yields a real publication guard; do not use a process-global registry, `()`, or any semantic permit type.

- [x] **Step 4: Freeze source and request the focused history gate**

Future command after root GO:

```powershell
$env:RUSTFLAGS='-Dwarnings'; cargo test -p neoethos-broker-history --lib production_broker_truth_v2_tests -- --nocapture
```

## Chunk 3: Thin acquisition bridge and CLI preflight

### Task 4: Bind canonical Search authority before session access

**Files:**
- Create: `crates/neoethos-broker-truth-acquire/Cargo.toml`
- Create: `crates/neoethos-broker-truth-acquire/src/lib.rs`
- Create: `crates/neoethos-broker-truth-acquire/tests/acquisition_preflight_contract.rs`

- [x] **Step 1: Write preflight REDs**

Define the wished-for API around `BrokerTruthAcquisitionArgsV1` and `prepare_acquisition_v1`. Cover:

- every explicit path/window/trust digest argument required;
- unknown/duplicate arguments rejected;
- no secret flags, env lookup, sibling discovery, or default paths;
- receipt and scope strict decode + `validate_against_receipt`;
- exact root binding open/pin for every receipt generation, with no current-generation fallback;
- explicit half-open window equality across flags, the immutable scope-window-binding receipt, plan, review record, and every synchronization;
- receipt cTrader environment/server/account/symbol equality with the plan;
- exact ordered primary/conversion synchronization set;
- any file/digest/tamper/symlink/path mismatch fails before a spy session runner is called.

- [ ] **Step 2: Register only the empty bridge harness and record executable RED**

Confirm the new library implementation does not yet exist as census only. Because a workspace-inheriting crate cannot be compiled honestly before it is a workspace member, request a narrowly coordinated exception to the earlier "root Cargo last" boundary: re-read and preserve the CatBoost patch, add only `crates/neoethos-broker-truth-acquire` to `[workspace].members`, and do not change `default-members`, workspace dependencies, vendor files, or `Cargo.lock` manually. With only `Cargo.toml`, an empty `src/lib.rs`, and the preflight test present, run exactly:

```powershell
$env:RUSTFLAGS='-Dwarnings'; cargo test -p neoethos-broker-truth-acquire --test acquisition_preflight_contract -- --nocapture
```

Record the expected missing-API compiler RED and stop. Do not implement preflight until root gives a separate repair GO. This minimal registration is the only root-Cargo edit before bridge source stability.

- [ ] **Step 3: Implement strict bounded preflight**

Parse strict versioned capture-plan/review-record/root-verification/window-binding request formats. Treat JSON only as frozen request bytes: authority comes from canonical types, exact immutable file bytes, digests, and exact root generation opens. Reconstruct `SelectedDatasetGenerationV1` from every receipt binding and call `open_exact_dataset_generation`; record the reopened manifest/generation/Vortex identities in the root-verification receipt and hold every lease through authority publication and capture/link publication.

The window rule performs no OHLC/timeframe reconstruction: the binding receipt must exactly copy `CanonicalSearchArtifactScopeV1` identity, role, row start/end, `timestamp_start_ms`, and `timestamp_end_ms`; its explicit `EvidenceWindowV1.from_unix_ms_inclusive` must equal `timestamp_start_ms`; its explicit exclusive end must equal the command flag, capture plan, review record and every reviewed synchronization and must be strictly greater than the scope's last consumed bar timestamp. The reviewed `window_policy_id` authenticates why that exclusive end is correct; acquisition merely freezes/hash-links it.

Recompute `ReviewedQuoteReplayRuleIdentityV2` from exact review/protocol/observation bytes; require the review record to bind the rules digest, trust-root digest, receipt/scope digests, account/instrument/window, and window-policy id. Freeze/hash-link only; do not verify a signature or call semantic ingress.

- [ ] **Step 4: Freeze source and request the exact GREEN rerun**

Hash the new crate sources and test, then request one rerun of the exact Task 4 command.

### Task 5: Add the one-shot CLI and evidence-only output

**Files:**
- Create: `crates/neoethos-broker-truth-acquire/src/main.rs`
- Create: `crates/neoethos-broker-truth-acquire/tests/acquisition_cli_contract.rs`

- [ ] **Step 1: Write CLI REDs**

Pin required help flags and prove the help/source contain no client-secret/token/credential-path inputs. Assert stdout contains only authority/link/BFT2 IDs, safe paths, and fixed:

```text
semantic_status=UnvalidatedEvidenceOnly
promotion_eligibility=NotPromotionEligible
permit_issued=false
```

Assert sanitized nonzero failures do not contain fixture secret values or raw broker JSON.

- [ ] **Step 2: Record the executable CLI RED before production**

Confirm no binary entrypoint exists as census only. Then request and run exactly:

```powershell
$env:RUSTFLAGS='-Dwarnings'; cargo test -p neoethos-broker-truth-acquire --test acquisition_cli_contract -- --nocapture
```

Record the expected missing-binary/compiler or failing-behavior RED and stop before creating `src/main.rs`.

- [ ] **Step 3: Implement the one-shot orchestration**

Preflight and authority publication must finish before invoking the secret-opaque history runner. Publish the final immutable link only after BFT2 publication. An orphan authority or BFT2 directory is incomplete and never auto-discovered.

- [ ] **Step 4: Freeze source and request the exact GREEN rerun**

Run rustfmt only on the exact changed Rust files, inspect the full diff, hash sources, and update root's required local audit checkpoint `target/audit-logs/agent-checkpoints/financial-truth-gate.md` (explicitly local/disposable build evidence, not tracked source authority). Request one rerun of the exact Task 5 command.

## Chunk 4: Coordinated final wiring proof and focused gates

### Task 6: Prove the exact workspace wiring and focused suite

**Files:**
- Modify: `Cargo.toml`
- Modify only if dependency resolution does so during an approved Cargo gate: `Cargo.lock`

- [ ] **Step 1: Re-read shared root Cargo**

Verify and preserve the existing CatBoost `[patch.crates-io]` hunk and every unrelated dirty edit.

- [ ] **Step 2: Write and run a workspace-wiring RED before any additional root edit**

Add a leaf source-contract test that reads the root manifest and asserts the exact bridge member plus the absence of a default member, global/current selector, and unnecessary workspace dependency. Run that focused test before any additional root manifest edit. Task 4 should already have added the sole required member; if the test is unexpectedly GREEN, no root edit is authorized. If another workspace package genuinely needs to name the bridge, write that exact dependency expectation into the test, record its RED, and only then add `neoethos-broker-truth-acquire = { path = "crates/neoethos-broker-truth-acquire" }`. Otherwise omit it. Do not touch `default-members`, vendor CatBoost, or any app/data/model/CUDA path.

- [ ] **Step 3: Request exact warning-denied gates one at a time**

Only after root GO, run and fully classify separate logs for:

```powershell
$env:RUSTFLAGS='-Dwarnings'; cargo test -p neoethos-broker-truth --test acquisition_authority_v1_contract -- --nocapture
$env:RUSTFLAGS='-Dwarnings'; cargo test -p neoethos-broker-history --lib production_broker_truth_v2_tests -- --nocapture
$env:RUSTFLAGS='-Dwarnings'; cargo test -p neoethos-broker-truth-acquire --test acquisition_preflight_contract --test acquisition_cli_contract -- --nocapture
```

Each command is a post-RED GREEN verification for its checkpoint, never the first execution of its tests. Stop at the first compiler/test error. For every RED and GREEN log, expected WARNING=0 and unrelated ERROR=0; record complete INFO, WARNING, and ERROR counts plus log SHA-256 in `target/audit-logs/agent-checkpoints/financial-truth-gate.md`.

- [ ] **Step 4: Report the honest completion boundary**

Offline gates establish structural acquisition behavior only. Real cTrader capture and semantic signature/policy validation remain blocked until the operator supplies the complete real receipt/scope/root/window/review/protocol/trust/synchronization set and separately authorizes live/demo access.
