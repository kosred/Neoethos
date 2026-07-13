# Audit Remediation Wave 4 Implementation Plan

> **For agentic workers:** REQUIRED: Use `superpowers:subagent-driven-development`. Write production code, tests, fixtures, manifests, and documentation now, but execute no build, test, lint, formatter, or package-manager verification command until every task in Waves 1-4 is authored and reviewed.

**Goal:** Enforce authenticated local control, mandatory MCP approval, real live-risk gates, and trusted mesh artifact exchange.

**Architecture:** Split each external boundary into parsing, authentication/authorization, and side-effect layers. Public health remains isolated; all private API routes fail closed. MCP approvals are action-bound and one-use. Every real order-open path reaches one risk gate immediately before broker submission. Mesh peers are explicitly trusted, work is lease-bound, and artifacts are extracted into bounded staging before atomic promotion.

**Tech Stack:** Axum/Tower middleware, Tauri 2 managed state/commands, OS randomness, HMAC-SHA256, iroh/QUIC endpoint identities, serde protocols, shared Wave 1 atomic persistence.

---

## Chunk 1: Local API trust boundary

### Task 1: API session types, token lifecycle, and bind policy (S01)

**Files:**
- Create: `crates/neoethos-app/src/server/auth.rs`
- Modify: `crates/neoethos-app/src/server/mod.rs` (`build_router`, `serve`, `serve_on`)
- Modify: `crates/neoethos-app/src/app_services/env_overrides.rs` (`AppRuntimeOverrides`, `server_bind_addr`)
- Modify: `crates/neoethos-core/src/config.rs` (`SystemSettings`)
- Modify: `config.yaml`
- Modify: `desktop/src-tauri/resources/config.yaml`
- Modify: `crates/neoethos-app/Cargo.toml` only if an already-locked crypto dependency must become direct
- Create: `crates/neoethos-app/tests/server_contract_tests.rs`

- [ ] Re-read every router layer, `serve`/`serve_on` caller, bind/origin setting producer, state constructor, and nearby server/config test; confirm the unauthenticated/CORS/bind defects from executable code before editing.
- [ ] Resolve current Axum 0.8 and Tower HTTP 0.6 middleware/CORS documentation with Context7 before choosing layer order; record the official APIs used in code comments only where the ordering is security-critical.
- [ ] Define `ApiSession` in `server/auth.rs`: exactly 32 bytes from the OS RNG, encoded only as unpadded base64url (43 ASCII characters) at the client boundary, constant-time byte comparison, redacted `Debug`, and no `Display`/serialization that can leak the secret. `NEOETHOS_API_TOKEN` must decode from that same canonical form to exactly 32 bytes; reject whitespace, padding, malformed alphabet, shorter/longer values, and noncanonical re-encodings before any bind.
- [ ] Define `ApiAuthConfig` with exact allowed origins and an explicit `remote_api_enabled` flag. Add backward-compatible defaults to `SystemSettings`: loopback bind, remote disabled, and development origin `http://localhost:5173`; reject wildcard, opaque, credential-bearing, path/query, or unsupported-scheme origins. The bundled origin is compile-time exact per Tauri's custom protocol: Windows/Android `http://tauri.localhost`, macOS/iOS/Linux `tauri://localhost` (no HTTPS override is configured).
- [ ] Refuse every non-loopback `SocketAddr` unless `remote_api_enabled == true` and an active non-empty token source exists. Return a startup error naming the unsafe setting, never silently fall back to loopback or open mode.
- [ ] In headless mode, consume `NEOETHOS_API_TOKEN` when present; otherwise create a fresh token, write it through the Wave 1 same-directory atomic primitive to the per-user config directory, restrict the file to the current user before exposure, log only the path, replace it on each start, and remove it best-effort on clean shutdown.
- [ ] Write named tests `server_contract_tests::headless_token_rotates_without_logging_secret`, `server_contract_tests::unsafe_non_loopback_bind_is_rejected`, `server_contract_tests::invalid_allowed_origin_is_rejected`, and `server_contract_tests::legacy_config_defaults_to_loopback_remote_disabled`.

### Task 2: Router authorization, exact CORS, and Tauri bootstrap (S01)

**Files:**
- Modify: `crates/neoethos-app/src/server/mod.rs`
- Modify: `crates/neoethos-app/src/lib.rs`
- Modify: `desktop/src-tauri/src/lib.rs` (`backend`, setup, invoke handler)
- Create: `desktop/src-tauri/src/api_proxy.rs`
- Modify: `desktop/src-tauri/capabilities/default.json`
- Modify: `desktop/src/api.ts`
- Modify: `desktop/src-tauri/tauri.conf.json`
- Modify: `crates/neoethos-app/tests/server_contract_tests.rs`
- Create: `desktop/src/api.contract.test.ts` if Wave 1's selected dependency-free test harness does not already own this coverage

- [ ] Re-read all HTTP clients (desktop API, mesh, MCP-facing app code), every route declaration, Tauri window label/capability, and the final Wave 1 CSP before editing; confirm which calls require the bearer token.
- [ ] Resolve current Tauri 2 command/window-origin and `ipc::Channel` behavior with Context7. Store `{ base_url, ApiSession }` only in Rust managed state; do not return either to JavaScript. Add typed `api_request(method, path, json_body)` and `api_subscribe(path, event, Channel)` commands that validate the path against a static app-route allowlist, inject the bearer inside Rust, proxy to the ephemeral backend, bound response/event sizes, and surface structured terminal errors.
- [ ] Both proxy commands require all three: label exactly `main`, current `WebviewWindow::url()` origin exactly the platform bundled origin (or debug `http://localhost:5173` only in debug), and a local-only capability whose `windows` list is only `main` with no `remote.urls`. Reject secondary, remote, data/blob, navigated hostile, or mismatched-scheme URLs.
- [ ] Deny external main-window navigation in the webview navigation callback except explicit system-browser opens handled outside the webview. Capability checks and the command's current-URL check remain mandatory defence in depth even when navigation is denied.
- [ ] Build an unauthenticated router containing only `GET /healthz`, a local-private router for desktop/headless/operator/internal-sidecar routes with `ApiSession` bearer middleware, and the two remote worker routes `/federation/job` plus `/federation/submit` with Task 7's distinct `FederationCredentials` bearer middleware. No route accepts a union/fallback of both token types. Merge them beneath exact CORS; browser preflight cannot make any private request executable without its scoped bearer.
- [ ] Allow only `http://tauri.localhost` on Windows/Android or `tauri://localhost` on macOS/iOS/Linux; in debug builds additionally allow exactly `http://localhost:5173`. Permit only the required methods/headers (`Authorization`, `Content-Type`) and never use `Any`, reflected origins, or credentials.
- [ ] Make `desktop/src/api.ts` call `api_request` for every JSON operation and `api_subscribe` channels for the two SSE streams. No `fetch`, `EventSource`, backend URL, port, bearer, local/session storage, DOM injection, exception, or log contains the secret.
- [ ] Set production CSP `connect-src` exactly to `'self' ipc: http://ipc.localhost`; development adds only `http://localhost:5173 ws://localhost:5173`. No loopback wildcard/dynamic port is present because the webview has no direct backend network access.
- [ ] Add named tests `server_contract_tests::healthz_is_public`, `server_contract_tests::private_get_requires_bearer`, `server_contract_tests::state_change_rejects_missing_wrong_and_malformed_bearer`, `server_contract_tests::valid_desktop_bearer_succeeds`, `server_contract_tests::weak_or_noncanonical_env_token_is_rejected_before_bind`, `server_contract_tests::hostile_origin_has_no_cors_permission`, `server_contract_tests::configured_origins_receive_exact_cors_headers`, Tauri-local `api_proxy_rejects_secondary_and_hostile_navigated_windows`, `api_proxy_injects_bearer_without_exposing_url_or_token`, `api_proxy_channels_surface_terminal_sse_errors`, and frontend `api_contract_uses_only_tauri_proxy_without_fetch_eventsource_or_secret_storage`.

## Chunk 2: MCP approval enforcement

### Task 3: Exact tool classification and action-bound approval (S02)

**Files:**
- Modify: `crates/neoethos-app/src/app_services/supervisor.rs` (`SupervisorAction::McpCall`, `execute`)
- Modify: `crates/neoethos-app/src/app_services/pending_actions.rs` (`ActionKind`, transitions, journal serialization)
- Modify: `crates/neoethos-app/src/server/pending_actions.rs` (`confirm` dispatch)
- Modify: `crates/neoethos-app/src/server/mcp.rs` (config validation)
- Modify: `desktop/src-tauri/src/lib.rs` (`mcp_sidecar` secret handoff)
- Modify: `mcp/mcp_servers.example.json`
- Create: `mcp/tests/fixtures/approval_vectors.json`
- Modify: `crates/neoethos-app/Cargo.toml` only for direct use of already-locked HMAC/SHA dependencies
- Create: `crates/neoethos-app/tests/mcp_approval_contract.rs`

- [ ] Re-read the complete Supervisor action parse/execute path, pending-action state machine and HTTP confirm path, MCP config writer, Tauri sidecar spawn, and all direct `/call` clients; confirm there is no second invocation path before editing.
- [ ] Define config `readOnlyTools` as an exact, duplicate-free array of `{ server, tool }`. Empty/missing means no direct calls. Reject wildcards, blank names, duplicate pairs, and any claim inferred from tool descriptions or prompts.
- [ ] Define one cross-workspace argument canonicalization: recursively sort object keys by UTF-8 byte order, preserve array order, serialize null/bool/string with `serde_json`, serialize integers in minimal decimal form, serialize finite floats with serde_json/ryu minimal round-trip form, and reject duplicate keys at parsing plus non-finite/unrepresentable numbers. `args_sha256` is lowercase hex SHA-256 of those UTF-8 canonical bytes.
- [ ] Keep `McpTools` read-only. For `McpCall`, call directly only when the exact pair is configured read-only; convert every unknown or non-read-only pair into persisted `ActionKind::McpToolCallMetadata { server, tool, args_sha256 }` and return the pending action ID without contacting `/call`.
- [ ] Keep raw executable MCP arguments only in a bounded in-memory `PendingMcpPayloadStore` keyed by pending-action ID and expiring with the action. Never serialize them to JSONL/disk/logs; after restart, an orphaned metadata action is explicitly non-executable/expired. Preserve legacy close-position actions.
- [ ] At operator confirmation, mint envelope v1 fields in fixed order: `version`, `action_id`, `server`, `tool`, `args_sha256`, `issued_at_ms`, `expires_at_ms`, `nonce_b64url`. HMAC input is an unambiguous sequence of 8-byte big-endian byte lengths followed by each UTF-8 field/value in that order; integer timestamps are minimal ASCII decimal. Require `0 <= expires-issued <= PENDING_ACTION_TTL_MS`, clock skew at most 5 seconds, 16-byte canonical nonce, and lowercase digests. Signature is lowercase hex HMAC-SHA256.
- [ ] Obtain the per-process 32-byte signing secret from OS randomness in Tauri and pass it to both app state and sidecar child without logging; in headless mode require canonical 32-byte `NEOETHOS_MCP_APPROVAL_SECRET` or disable mutating MCP execution with an explicit status.
- [ ] Put fixed key/args/canonical-bytes/digest/envelope/signature known-answer vectors in `mcp/tests/fixtures/approval_vectors.json`; both the app contract test and isolated sidecar test must consume the same file and produce byte-identical results.
- [ ] Replace the Supervisor prompt claim with the exact enforced rule: configured read-only pairs may execute; everything else is queued for operator confirmation.
- [ ] Add named tests `mcp_approval_contract::unknown_tool_is_queued_without_sidecar_call`, `mcp_approval_contract::mutating_tool_is_queued`, `mcp_approval_contract::exact_read_only_pair_may_call`, `mcp_approval_contract::description_cannot_grant_read_only`, and `mcp_approval_contract::confirmation_binds_server_tool_args_and_expiry`.

### Task 4: Sidecar verification, replay defence, and lifetime integration (S02 with M02)

**Files:**
- Modify: `mcp/src/main.rs`
- Modify: `mcp/Cargo.toml` and `mcp/Cargo.lock` only if direct cryptographic dependencies are required
- Modify: `mcp/mcp_servers.example.json`
- Modify: `mcp/tests/fixtures/approval_vectors.json`
- Create: `mcp/tests/lifecycle_and_approval.rs`

- [ ] Re-read the final Wave 1 `ConnectionHolder`/shutdown design and current rmcp request model before editing. Resolve rmcp 2.1 APIs with Context7; do not infer authorization from MCP metadata.
- [ ] Parse the same exact `readOnlyTools` pairs in the sidecar. `/call` permits an unapproved request only for an exact configured read-only pair; every other pair requires the signed approval envelope.
- [ ] Implement the exact Task 3 canonicalization/envelope v1 contract and shared known-answer vectors. Verify HMAC in constant time, exact server/tool/args digest binding, issued/expiry/skew bounds, canonical encodings, and nonce uniqueness before `peer.call_tool`. Mark a nonce used atomically before the external side effect; reject replay even if the first tool call later fails.
- [ ] Return structured non-2xx authorization errors for missing, malformed, expired, tampered, wrong-action, or replayed approval; never encode authorization failure as HTTP 200 with `ok: false`.
- [ ] Extend `lifecycle_and_approval` with named cases `http_tool_callable_after_connect_returns`, `stdio_tool_callable_after_connect_returns`, `explicit_shutdown_closes_both_transports`, `mutating_call_without_approval_never_reaches_peer`, `tampered_or_expired_approval_never_reaches_peer`, and `approval_nonce_is_one_use`.

## Chunk 3: Real live-risk enforcement

### Task 5: One mandatory pre-send open-order gate (S03)

**Files:**
- Create: `crates/neoethos-app/src/app_services/live_risk.rs`
- Modify: `crates/neoethos-app/src/app_services/mod.rs`
- Modify: `crates/neoethos-app/src/app_services/broker_api.rs` (`prepare_new_order`, `submit_market_order_blocking`, `submit_pending_order_blocking`)
- Modify: `crates/neoethos-app/src/app_services/live_trading.rs` (automatic open intent context)
- Modify: `crates/neoethos-app/src/server/orders.rs` (manual market/pending context)
- Modify: `crates/neoethos-trader/src/risk.rs`
- Modify: `crates/neoethos-trader/src/contracts.rs`
- Modify: `crates/neoethos-trader/src/engine.rs`
- Modify: `crates/neoethos-core/src/domain/risk.rs` (`TradeGateInput`, clock/equity refresh)
- Modify: `crates/neoethos-core/src/domain/risky_mode.rs` (validated persisted snapshot)
- Create: `crates/neoethos-trader/tests/live_risk_contract.rs`
- Create: `crates/neoethos-app/tests/live_risk_contract.rs`

- [ ] Re-read every construction of `CTraderExecutionRequest::NewOrder`, every call to both submit helpers, manual and automatic adapters, account snapshot units, SL/volume conversion, core `RiskManager`, `RiskyModeManager`, and nearby broker/risk tests; document in the code review that no real broker open bypass remains.
- [ ] Define `OrderOrigin::Manual | Autonomous { confidence, strategy_sharpe, strategy_rank }` and trader-level `LiveRiskInput` carrying finite equity/balance/daily PnL/drawdown, UTC seconds/hour/minute/weekday/date/month IDs, market volatility from the latest completed symbol bars, symbol, side, entry/SL/TP, requested units/lots and computed risk USD, existing exposure, account mode, and origin. Autonomous evidence is required/finite/in range; manual origin is explicit rather than inventing confidence/rank. Missing chart/account/clock/strategy evidence needed by an active rule fails closed.
- [ ] Change core `TradeGateInput` so confidence is explicitly optional only for `Manual`; keep all session, volatility, Sharpe/rank, drawdown, and trade-count rules active. Add `RiskManager::refresh_equity_clock(equity, utc_date_id, utc_month_id)` to update total/day peaks and perform deterministic day/month rollover before checking; never reconstruct a manager per order.
- [ ] Define `LiveRiskRegistry` as `Mutex<HashMap<AccountKey, AccountRiskState>>`. Each entry owns one long-lived `RiskManager`, optional `RiskyModeManager`, last broker/account evidence, and versioned persistence snapshot. Initialize once from valid account/settings plus persisted peaks/counters; atomically persist after rollover, open, close, rejection trip, and re-arm. Corrupt/mismatched state disables opens for that account.
- [ ] Add a validated serializable `RiskyModeSnapshot` in core that round-trips bankroll, stage, daily/weekly/monthly loss accumulators, consecutive losses, and sticky halt without exposing private fields. Restore only through a validator against the current config/account; never deserialize an unchecked permissive manager.
- [ ] Implement `live_risk::authorize_open_order` as the sole app adapter to the registry/core managers. Call it inside both broker submit helpers immediately before constructing/sending `NewOrder`, after final broker normalization, so manual market, manual pending, and automatic live opens cannot bypass it. On broker acceptance call `on_trade_opened`; feed broker close outcomes/equity into `on_trade_closed` and `record_trade_outcome` exactly once.
- [ ] Keep closes, SL tightening, cancellations, and other de-risking operations available; the gate must classify and reject any attempt to route an exposure-increasing amend as de-risking.
- [ ] Return structured rejection tiers and operator-visible reasons. Never replace missing account/risk evidence with zero or a permissive default.
- [ ] Add named tests `live_risk_contract::manual_market_and_pending_orders_share_gate`, `live_risk_contract::automatic_open_shares_gate`, `live_risk_contract::missing_or_nonfinite_inputs_fail_closed`, `live_risk_contract::registry_preserves_peak_day_month_and_trade_state_across_orders_and_restart`, `live_risk_contract::broker_open_and_close_update_state_exactly_once`, `live_risk_contract::each_prop_firm_and_risky_tier_rejects`, and `live_risk_contract::closes_and_true_derisking_amends_remain_allowed`.

### Task 6: Risky-mode trip persistence, restart, and re-arm (S03)

**Files:**
- Modify: `crates/neoethos-app/src/app_services/risky_mode_persistence.rs`
- Modify: `crates/neoethos-app/src/app_services/live_risk.rs`
- Modify: `crates/neoethos-app/src/server/risky.rs`
- Modify: `crates/neoethos-app/src/server/bridge.rs` (remove automatic re-arm polling)
- Modify: `crates/neoethos-app/src/server/mod.rs` (authenticated `POST /risky/rearm`)
- Modify: `desktop/src/api.ts`
- Modify: `desktop/src/screens/RiskyMode.tsx`
- Modify: `crates/neoethos-app/tests/live_risk_contract.rs`

- [ ] Re-read every risky-state load/save/reset caller and existing corrupt-state behavior; confirm the exact trip/re-arm lifecycle before editing.
- [ ] On risky-mode rejection, record the tier, reason, account identity, timestamp, and non-secret evidence through the Wave 1 locked atomic primitive before returning the broker rejection. If durable trip recording fails, keep trading disabled and return both failures.
- [ ] Treat missing legacy state according to existing defaults, but treat present unreadable, corrupt, wrong-schema, or account-mismatched state as disabled/fail-closed with explicit operator status.
- [ ] Delete `auto_re_arm_if_ready` and its bridge polling path. Add authenticated `POST /risky/rearm`, a typed desktop API wrapper, and an explicit Risky Mode screen action. Re-arm requires elapsed cooldown, current valid matching account/equity evidence, persisted acknowledgement, and an operator request in this session; a timer or restart alone cannot clear a trip.
- [ ] Add named tests `live_risk_contract::rejection_trip_survives_restart`, `live_risk_contract::corrupt_state_disables_opens`, `live_risk_contract::trip_write_failure_remains_closed`, `live_risk_contract::cooldown_blocks_early_rearm`, and `live_risk_contract::explicit_rearm_after_cooldown_restores_valid_opens`.

## Chunk 4: Mesh trust and artifact isolation

### Task 7: Trusted endpoints, coordinator bearer, and lease protocol (S04)

**Files:**
- Create: `mesh/src/lib.rs`
- Create: `mesh/src/protocol.rs`
- Create: `mesh/src/trust.rs`
- Modify: `mesh/src/main.rs`
- Modify: `mesh/Cargo.toml` and `mesh/Cargo.lock` only if a required direct dependency is not already locked
- Modify: `desktop/src-tauri/src/lib.rs` (managed mesh-sidecar child credentials/lifecycle)
- Modify: `crates/neoethos-app/src/app_services/federation.rs`
- Modify: `crates/neoethos-app/src/server/federation.rs`
- Modify: `crates/neoethos-core/src/config.rs`
- Modify: `config.yaml`
- Modify: `desktop/src-tauri/resources/config.yaml`
- Create: `mesh/tests/mesh_trust_contract.rs`

- [ ] Re-read the entire mesh RPC/gossip/job loop, iroh identity persistence, app HTTP federation handlers, worker token forwarding, coordinator state, and existing tests; confirm every trust and lease bypass before editing.
- [ ] Resolve iroh 1.0 endpoint identity/connection APIs with Context7. Parse a non-empty configured set of trusted `EndpointId`s; reject RPC work/result traffic from all other peers before reading a large body. Discovery metadata cannot grant trust.
- [ ] Define two non-interchangeable credentials. `LocalApiCredentials` contains the per-start S01 loopback URL/token and authorizes only this machine's private API. `FederationCredentials` contains a separately configured 32-byte cross-node bearer and authorizes only coordinator worker endpoints. They use distinct types, headers, env/config names, rotation logs, and redacted debug; neither may fall back to the other.
- [ ] In Tauri, own a `mesh_sidecar` child like the MCP sidecar and pass `NEOETHOS_LOCAL_API_URL` plus `NEOETHOS_LOCAL_API_TOKEN` only in the child environment; stop it on window close. In headless/manual mesh mode require both local API env values explicitly. The public temp port file is no longer sufficient and carries no credential.
- [ ] Make the federation bearer mandatory whenever cross-node federation/mesh work is enabled. Attach local credentials to local engine/status/migration/artifact calls and federation credentials to remote `/federation/job` and `/federation/submit`; never log either or place either in a query/body.
- [ ] Wrap every gossip announcement/migrant in `SignedGossipEnvelopeV1 { origin_endpoint_id, kind, issued_at_ms, nonce_b64url, payload, signature }`. Sign length-prefixed version/origin/kind/timestamp/nonce plus canonical payload bytes with the origin iroh/Ed25519 secret. Verify against the claimed origin public key, require the origin in the trusted set, require ±5-minute freshness and a one-use nonce, then process. The immediate forwarding peer is never treated as the author.
- [ ] Replace tuple leases with `FedLease { lease_id, worker_endpoint_id, job, issued_at_ms, expires_at_ms, expected_outputs }`. Generate 128-bit OS-random lease IDs; bind symbol, timeframe, work type, schema version, exact output filenames, and coordinator identity. TTL is exactly 12 hours; allow at most 1,024 active leases globally and 4 per worker endpoint.
- [ ] Persist queue, active leases, and completed replay records in versioned `federation_leases.json` through the Wave 1 locked atomic primitive. Keep completed IDs for 30 days up to 4,096 newest records. On restart validate schema/bindings, requeue only expired active jobs, retain unexpired ownership, and retain/prune completed replays deterministically; corrupt state disables leasing/submission until operator repair.
- [ ] Include the full lease in `GetJob` responses and require `lease_id` on every discovery/training result. Atomically reject unknown, expired, duplicate/completed, wrong-peer, wrong-job, or wrong-schema submissions before file I/O.
- [ ] Add mesh tests `mesh_trust_contract::unknown_endpoint_cannot_claim_or_submit`, `mesh_trust_contract::untrusted_origin_cannot_register_or_inject_through_trusted_forwarder`, `mesh_trust_contract::tampered_stale_or_replayed_gossip_signature_fails`, `mesh_trust_contract::local_and_federation_credentials_are_not_interchangeable`, `mesh_trust_contract::coordinator_requests_carry_federation_bearer_without_logging_it`, `mesh_trust_contract::lease_is_bound_to_peer_job_schema_and_expiry`, `mesh_trust_contract::lease_capacity_and_per_peer_limits_fail_closed`, and `mesh_trust_contract::completed_replay_survives_restart_and_retention_pruning`.
- [ ] Add Tauri-local tests under filter `mesh_sidecar_contract`: `child_receives_local_api_credentials_only_in_environment` and `window_close_stops_child`; these do not belong to the isolated mesh integration test.

### Task 8: Bounded extraction, staged validation, and atomic promotion (S04)

**Files:**
- Create: `mesh/src/artifacts.rs`
- Modify: `mesh/src/protocol.rs`
- Modify: `mesh/src/main.rs`
- Modify: `mesh/Cargo.toml` and `mesh/Cargo.lock` (zstd and streaming support)
- Create: `crates/neoethos-app/src/app_services/federation_artifacts.rs`
- Modify: `crates/neoethos-app/src/app_services/mod.rs`
- Modify: `crates/neoethos-app/src/app_services/federation.rs`
- Modify: `crates/neoethos-app/src/server/federation.rs` (protected local staging endpoint)
- Modify: `crates/neoethos-app/src/server/engines_control.rs` (lease-bound start DTO/context)
- Modify: `crates/neoethos-app/src/app_services/discovery.rs` (binding and produced outputs)
- Modify: `crates/neoethos-app/src/app_services/training.rs` (binding and produced outputs)
- Modify: `crates/neoethos-search/src/live_portfolio.rs` (optional federation binding metadata)
- Modify: `crates/neoethos-models/src/runtime/training_artifact.rs` (optional federation binding metadata)
- Modify: `crates/neoethos-models/src/runtime/artifacts.rs` and `crates/neoethos-models/src/ensemble_inference/bootstrap.rs` (namespaced promoted model roots)
- Create: `crates/neoethos-app/tests/federation_artifact_contract.rs`
- Modify: `mesh/tests/mesh_trust_contract.rs`

- [ ] Re-read every model/artifact discovery path, current 512 MiB `MAX_RPC`, `tar_dir`, `untar_to`, `models_dir`, portfolio collection, and active-store scanner; confirm exactly which output files each work type legitimately produces.
- [ ] Add `FederationRunContext` to the local-auth-only engine-start DTO and carry it through app discovery/training into artifact writers. On completion the app records and returns the exact output paths created by that lease through a local-auth-only endpoint; the worker mesh may read only those paths. Add optional `FederationBinding { lease_id, coordinator_endpoint_id, symbol, timeframe, work_type, schema_version }` to the general schema so ordinary local/legacy artifacts keep `None`, but a federated run writer must emit `Some(binding)`.
- [ ] Replace whole-store tar transfer with a per-lease manifest containing only those expected new output files: normalized relative path, decoded byte length, SHA-256, schema version, symbol/timeframe, and lease ID. Reject zero-file and unexpected-file results.
- [ ] Replace base64 JSON artifacts with bounded binary framing on a dedicated QUIC stream: fixed magic/version, unsigned 32-bit manifest length (max 1 MiB), unsigned 64-bit zstd length (max 144 MiB), manifest bytes, then exactly that many compressed bytes. Cap the complete artifact frame at 160 MiB and ordinary JSON control RPC at 1 MiB. Keep decoded total 256 MiB, per-file 160 MiB, and file count 96; enforce every bound while streaming.
- [ ] Split ownership explicitly. Mesh owns trusted transport, binary framing/streaming caps, zstd decoding, manifest/hash verification, and archive path screening; it forwards the bounded binary frame plus authenticated lease/peer metadata to the local app staging endpoint with `LocalApiCredentials`. It never calls main-workspace typed loaders or writes active stores.
- [ ] The app owns filesystem staging and semantic validation. Stream to a newly-created `<cache>/federation_staging/<lease_id>.<nonce>/` outside active scanners using Wave 1 atomic-file writes; reject absolute, parent, alternate-prefix, duplicate, NUL, Windows-device, symlink, hard-link, and non-regular entries before creating output paths.
- [ ] Parse each staged file through the main-workspace canonical portfolio/model typed loader; verify manifest hash/size and require `Some(FederationBinding)` exactly matching expected filename/schema/symbol/timeframe/lease/coordinator. Reject `None` for every federated submission and reject undeclared companions. On failure remove only the verified nonce staging directory and leave the active store untouched.
- [ ] Publish by same-volume atomic rename of the complete unique staging directory to a previously nonexistent namespaced destination: discovery `<cache>/federation_inbox/<lease_id>/`, training `<models>/federation/<lease_id>/`. Sync files and parent before rename; fail if the destination exists; never replace/delete an active directory. Update portfolio/model discovery to read these namespaced roots only after publication, then mark the lease completed.
- [ ] Add mesh tests `mesh_trust_contract::binary_frame_manifest_compressed_and_decoded_caps_fail_closed`, `mesh_trust_contract::absolute_parent_symlink_hardlink_and_duplicate_paths_fail`, and `mesh_trust_contract::only_app_reported_job_outputs_are_transferred`.
- [ ] Add app tests `federation_artifact_contract::federated_run_propagates_required_binding_and_exact_outputs`, `federation_artifact_contract::missing_or_mismatched_binding_fails`, `federation_artifact_contract::typed_hash_schema_symbol_timeframe_lease_tamper_fails`, `federation_artifact_contract::failed_stage_never_touches_active_store`, and `federation_artifact_contract::validated_directory_publishes_once_atomically_to_namespaced_store`.

## Deferred verification gate for Wave 4

- [ ] Do not execute anything in this section until production code, tests, fixtures, manifests, lockfiles, and documentation for every M01-M10, D01-D09, B01-B15, and S01-S05 row are authored and reviewed.
- [ ] At focused-verification step 5, run `cargo test -p neoethos-app server_contract_tests -- --nocapture`; require exit code 0, nonzero execution of every named S01 test, no warnings, and complete log inspection from first to last byte.
- [ ] Run `cargo test --manifest-path mcp/Cargo.toml lifecycle_and_approval -- --nocapture`; require exit code 0, nonzero HTTP/stdio/approval test execution, no warnings, and complete log inspection.
- [ ] Run `cargo test -p neoethos-app live_risk_contract -- --nocapture` and `cargo test -p neoethos-trader live_risk_contract -- --nocapture`; require exit code 0, nonzero tests for every open path/tier/restart/corruption/re-arm case, no warnings, and complete logs.
- [ ] Run `cargo test --manifest-path mesh/Cargo.toml mesh_trust_contract -- --nocapture`; require exit code 0, nonzero trust/lease/limits/path/tamper/isolation cases, no warnings, and complete logs.
- [ ] Run `cargo test -p neoethos-app federation_artifact_contract -- --nocapture`; require exit code 0, nonzero typed-staging/publication/isolation cases, no warnings, and complete logs.
- [ ] Run `cargo test -p neoethos-desktop mesh_sidecar_contract -- --nocapture`; require both child-credential/lifecycle tests, exit code 0, no warnings, and a complete log.
- [ ] Continue with the approved full command order. Any diagnostic reopens authoring; fix root causes without suppression, rerun the affected layer, and repeat the final full suite until all complete logs are clean.
