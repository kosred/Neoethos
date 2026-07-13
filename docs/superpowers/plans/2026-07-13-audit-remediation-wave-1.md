# Audit Remediation Wave 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use `superpowers:subagent-driven-development`. Every step uses checkbox syntax. Documentation lookups and source/diff inspection are allowed during authoring; no build, test, lint, formatter, audit, or package-install verification command may run until Waves 1-4 are fully authored.

**Goal:** Correct M01-M10 and S05 with low-blast-radius contract, persistence, CI, packaging, desktop, and frontend changes without altering trading mathematics.

**Architecture:** Put reusable atomic-file and runtime-initialization behavior in focused modules. Keep wire contracts camelCase and typed at both ends. Preserve legacy persisted reads, but make corrupt canonical state explicit and make all new writes deterministic and atomic.

---

## Chunk 1: Wire contracts and MCP lifetime

### Task 1: Define and test the camelCase HTTP contract (M01)

**Files:**
- Add: `desktop/src/apiContracts.ts`
- Add: `desktop/test/apiContracts.test.ts`
- Modify: `desktop/src/api.ts`
- Modify: `crates/neoethos-app/src/server/orders.rs`
- Modify: `crates/neoethos-app/src/server/data_control.rs`
- Modify: `crates/neoethos-app/src/server/strategy_lab.rs`
- Add: `crates/neoethos-app/tests/server_contract_tests.rs`

- [ ] Extract pure builders `amendProtectionBody`, `dataImportBody`, and `promoteStrategyBody` into `apiContracts.ts`; their exact emitted keys are respectively `{positionId, stopLossPrice, takeProfitPrice, trailingStopLoss}`, `{sourcePath, symbol, timeframe}`, and `{symbol, baseTf}`. Keep the read-only promotion query key `base_tf`, because `PromotionQuery` is explicitly snake_case.
- [ ] Make `api.ts::{amendProtection,dataImport,promoteStrategy}` use those builders and remove their snake_case JSON bodies.
- [ ] Apply `#[serde(deny_unknown_fields, rename_all = "camelCase")]` to `NewOrderBody`, `NewPendingOrderBody`, `ClosePositionBody`, `CancelOrderBody`, `AmendPositionProtectionBody`, `FetchBody`, `ImportBody`, and `PromoteBody`; existing camelCase clients remain compatible, while ignored snake_case money fields are rejected.
- [ ] In `server_contract_tests.rs`, deserialize the three exact frontend fixtures, assert all price/timeframe/path values, and assert snake_case aliases plus unknown money fields fail.
- [ ] In `apiContracts.test.ts`, assert the pure builders emit the same fixtures byte-for-byte after `JSON.stringify`.
- [ ] Review the producer/consumer diff only; do not execute either test file.

### Task 2: Own and shut down rmcp services (M02)

**Files:**
- Add: `mcp/src/lib.rs`
- Modify: `mcp/src/main.rs`
- Add: `mcp/tests/lifecycle_and_approval.rs`

- [ ] Resolve current rmcp documentation with `npx ctx7@latest library rmcp "rmcp 2.1 RunningService peer is_closed cancel HTTP stdio lifecycle"`, then fetch the selected ID with `npx ctx7@latest docs <id> "rmcp 2.1 RunningService peer is_closed cancel HTTP stdio lifecycle"`; supplement only with the resolved 2.1.0 dependency source. This is documentation lookup, not verification.
- [ ] Move connection/state logic to `mcp/src/lib.rs`. Define `ConnectedService` with a cloned `Peer<RoleClient>` and `tokio::sync::Mutex<Option<RunningService<RoleClient, ()>>>`; `connect` returns this holder, never a bare peer.
- [ ] Store `Arc<ConnectedService>` values in `AppState`; `list_tools`/`call_tool` use `peer()`, while `/health` returns each server's `connected: !is_closed()` and counts only open transports.
- [ ] Add `AppState::shutdown_all` that takes each running service and awaits `cancel`; wrap Axum serving with graceful shutdown and call it before `main` returns.
- [ ] In `lifecycle_and_approval.rs`, use an in-process rmcp Streamable HTTP fixture and a self-spawned test-binary stdio fixture. Add `lifecycle_and_approval_http_holder_keeps_tool_callable` and `lifecycle_and_approval_stdio_holder_keeps_tool_callable`, each calling an echo tool after `connect` has returned.
- [ ] Add `lifecycle_and_approval_shutdown_closes_transport_and_health` and assert explicit cancellation closes both transports and changes health. The self-spawn fixture exits immediately in ordinary full-suite execution and enters server mode only under its child-only environment flag; do not add ignored tests.

---

## Chunk 2: CI, release metadata, and vendor integration

### Task 3: Make CI references truthful (M03)

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.cargo/config.toml`
- Modify: `crates/neoethos-app/tests/packaging_smoke.rs`

- [ ] Remove the `check_no_python_legacy.sh` step. Source/manifests contain no production PyO3/Python dependency and the referenced script does not exist; do not create a new script.
- [ ] Change every Linux/Windows/macOS Rust setup to `dtolnay/rust-toolchain@nightly-2026-04-07`, retaining required `rustfmt`/`clippy` components and matching `rust-toolchain.toml` plus `.cargo/config.toml` nightly flags.
- [ ] Add `ci_workflow_uses_pinned_nightly_and_existing_paths` to `packaging_smoke.rs`; it asserts no `@stable`, exactly the pinned nightly at every setup step, no legacy-script reference, and existence of every repository-relative path used by `run:` commands.
- [ ] Trace the duplicate MSVC symbols hidden by `/FORCE:MULTIPLE` and `/IGNORE:4006` through the final native source/link list. Remove the duplicate compilation/library input in the LightGBM build integration, then delete both linker suppressions from `.cargo/config.toml`; do not replace them with another ignore flag. Add a static assertion that no `/FORCE`, `/IGNORE`, warning-disable, or diagnostic-suppression link arg remains.

### Task 4: Align all first-party licenses (M05)

**Files:**
- Modify: `crates/neoethos-{app,cli,codex,core,data,models,search,trader}/Cargo.toml`
- Modify: `crates/neoethos-app/tests/packaging_smoke.rs`

- [ ] Replace package-level proprietary declarations in app/CLI/Codex with `license.workspace = true`; add the same inheritance to core/data/models/search/trader.
- [ ] Change app and CLI `package.metadata.generate-rpm.license` to `AGPL-3.0-or-later` and both `package.metadata.deb.copyright` values from `proprietary; all rights reserved` to the exact project copyright plus `AGPL-3.0-or-later`; retain the already-correct desktop, MCP, and mesh declarations.
- [ ] Add `first_party_metadata_is_agpl` to `packaging_smoke.rs`, covering all eight workspace crates, Tauri, MCP, mesh, rpm/deb metadata, and release-manager templates; case-insensitively reject `LicenseRef-Proprietary`, `proprietary`, `all rights reserved`, and every other non-AGPL first-party declaration.

### Task 5: Separate unpublished Windows templates from supported packages (M04)

**Files:**
- Remove: `packaging/scoop/neoethos.json`
- Remove: `packaging/winget/manifests/k/kosred/neoethos/0.4.20/**`
- Remove: `packaging/chocolatey/neoethos/**`
- Add: `packaging/templates/windows/scoop/neoethos.json.in`
- Add: `packaging/templates/windows/winget/0.5.3/kosred.neoethos.{yaml,installer.yaml,defaultLocale.yaml}.in`
- Add: `packaging/templates/windows/chocolatey/neoethos/{neoethos.nuspec,chocolateyinstall.ps1,chocolateyuninstall.ps1}.in`
- Remove: `crates/neoethos-app/packager.json`
- Modify: `crates/neoethos-app/tests/packaging_smoke.rs`

- [ ] Move the unpublished external-manager definitions to the exact `.in` paths above. Use literal version `0.5.3`, AGPL metadata, the v0.5.3 installer URL, and one explicit `@SHA256@` substitution token; no `.json`, `.yaml`, `.nuspec`, or installable PowerShell manifest remains at the old recognized locations.
- [ ] Remove `packager.json` after confirming the repository has no consumer; Tauri `desktop/src-tauri/tauri.conf.json` remains the sole desktop bundler definition.
- [ ] Replace `portable_build_script_is_present_and_executable` and stale Chocolatey/Scoop/WinGet installability tests with `unpublished_windows_manifests_are_templates`, asserting 0.5.3, AGPL, one hash token, no empty/all-zero digest, and non-installable `.in` suffixes.
- [ ] Keep supported-package tests explicit for `packaging/appimage/build.sh`, its AppDir assets, app/CLI deb+rpm metadata, and Tauri bundle resources/targets.

### Task 6: Make the LightGBM build source-sensitive and target-correct (M10)

**Files:**
- Modify: `vendor/lightgbm3-sys/build.rs`
- Add: `vendor/lightgbm3-sys/build_support.rs`
- Modify: `crates/neoethos-core/tests/vendor_provenance_contract.rs`

- [ ] Resolve current Cargo documentation with `npx ctx7@latest library Cargo "build scripts rerun-if-changed rustc-link-search rustc-link-lib TARGET bindgen cross compilation"`, then fetch the selected ID with `npx ctx7@latest docs <id> "build scripts rerun-if-changed rustc-link-search rustc-link-lib TARGET bindgen cross compilation"`.
- [ ] Put recursive source fingerprinting and target-triple decisions in `build_support.rs`. Hash file paths and contents under `lightgbm/` in sorted order; compare with an `OUT_DIR` stamp, delete/re-copy only a mismatched stage, and publish the stamp only after a complete copy.
- [ ] Emit `cargo::rerun-if-changed=build.rs`, `build_support.rs`, `Cargo.toml`, and `lightgbm`; derive Windows generator, macOS deployment/architecture, C++ runtime, and OpenMP paths from `TARGET`, never build-script host `cfg`.
- [ ] Pass `--target=<TARGET>` to bindgen and use valid `cargo::rustc-link-search=native=...`, `cargo::rustc-link-lib=...`, and singular `cargo::rustc-link-arg=...` directives.
- [ ] Add static/unit contract assertions for source-change invalidation, sorted fingerprints, target decisions, bindgen target propagation, and absence of host `cfg(target_*)` decisions.

### Task 7: Record vendor provenance and licenses (M10)

**Files:**
- Add: `vendor/{lightgbm3-sys,sklears-core,rlkit}/PROVENANCE.toml`
- Add: `vendor/lightgbm3-sys/LICENSE`
- Add: `vendor/sklears-core/LICENSE`
- Modify: `crates/neoethos-core/tests/vendor_provenance_contract.rs`

- [ ] Give each provenance file fixed fields `crate`, `version`, `upstream_repository`, `upstream_revision` (40 lowercase hex), `source_archive_sha256` (64 lowercase hex), `vendored_at_repository_commit`, `license_files`, and `[[patches]] {path, rationale}`. LightGBM records wrapper 1.0.8 and embedded LightGBM 4.6.0 as separate components.
- [ ] Inventory the existing NeoEthos changes: LightGBM build/staging integration, sklears non-Linux `read_perf_counters` fallback, and rlkit precision/backend changes; never claim an empty patch list.
- [ ] Copy exact upstream MIT and Apache-2.0 license texts into the two missing root license files; reference LightGBM's embedded `lightgbm/LICENSE` and rlkit's existing `LICENSE` too.
- [ ] Make `vendor_provenance_contract` parse all three TOML files and validate identities, hashes/revisions, existing license paths, and every root `[patch.crates-io]` vendor override.

---

## Chunk 3: Deterministic and atomic canonical state

### Task 8: Add a focused atomic-file primitive (M07)

**Files:**
- Add: `crates/neoethos-core/src/storage/atomic_file.rs`
- Modify: `crates/neoethos-core/src/storage.rs`
- Modify: `crates/neoethos-core/src/storage/json.rs`
- Add: `crates/neoethos-core/tests/persistence_contract.rs`

- [ ] Implement `write_bytes_atomic` and `update_bytes_atomic`. Both share a normalized-path lock registry; update holds the same lock across read/parse/mutate/write.
- [ ] Use same-directory `create_new` temp names containing PID plus an atomic nonce, `write_all`, file `sync_all`, atomic replace, parent-directory sync where supported, and cleanup guards. On Windows use `MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)` as already proven in `neoethos-data::vortex_io`; never delete the destination before replacement.
- [ ] Make `write_json_atomic` serialize deterministically and delegate only byte replacement to `atomic_file`, keeping JSON backup/directory-artifact responsibilities in `json.rs`.
- [ ] Put replace/fault-injection coverage in private `#[cfg(test)]` unit tests inside `storage/atomic_file.rs`: factor a private implementation that accepts a test-only replace closure, without any production global hook or public fault API. Keep `crates/neoethos-core/tests/persistence_contract.rs` for public concurrent-writer/round-trip behavior. Name the tests `persistence_contract_concurrent_writers_leave_complete_payload`, `persistence_contract_interrupted_replace_preserves_previous_file`, and `persistence_contract_unique_temp_files_are_removed`; use no timing sleeps.

### Task 9: Migrate every canonical singleton writer (M06, M07)

**Files:**
- Modify: `crates/neoethos-core/src/{config.rs,broker_config.rs,symbol_metadata.rs}`
- Modify: `crates/neoethos-app/src/server/{settings.rs,mcp.rs}`
- Modify: `crates/neoethos-app/src/app_services/{risky_mode_persistence.rs,strategy_blacklist.rs,spread_stats.rs,supervisor.rs}`
- Add: `crates/neoethos-app/tests/persistence_contract.rs`

- [ ] Route `Settings::save`, `broker_config::save_to_disk`, `SymbolMetadataTable::save_to_disk`, raw/merged settings writes, MCP config, risky-mode state, strategy blacklist, spread stats, and supervisor config through the shared primitive. These are the complete canonical singleton families found by the direct-writer trace; immutable discovery/model/data artifacts and append-only JSONL journals remain outside M07.
- [ ] Use `update_bytes_atomic` for settings merge/raw backup, `record_kill_switch_trip`, `auto_re_arm_if_ready`, and blacklist retirement so their entire read-modify-write is locked. Refactor merged settings mutation into a pure `apply_settings_update` helper.
- [ ] Preserve missing-file and legacy-schema behavior, but distinguish NotFound from permission/I/O/parse/schema errors. Corrupt risky state and blacklist fail closed; observational spread stats may reset only after emitting a structured error.
- [ ] Serialize symbol metadata through a sorted `BTreeMap` view while retaining its public `HashMap` lookup API.
- [ ] Add app `persistence_contract` cases for concurrent settings/risky/blacklist updates, corrupt-state rejection, raw-settings backup, and legacy reads; add core cases for broker TOML/YAML/JSON round-trips and byte-identical repeated symbol saves.

---

## Chunk 4: Shared startup and frontend quality

### Task 10: Share runtime initialization (M08, S05)

**Files:**
- Add: `crates/neoethos-app/src/runtime_initialization.rs`
- Modify: `crates/neoethos-app/src/{lib.rs,main.rs}`
- Modify: `desktop/src-tauri/src/lib.rs`
- Modify: `crates/neoethos-app/tests/server_contract_tests.rs` (CLI/app half)
- Modify: `desktop/src-tauri/src/lib.rs` test module (Tauri half)

- [ ] Implement `initialize_runtime_from_settings(path) -> Result<InitializedRuntime>`: load once, install config path plus search/tree/hardware/data/app overrides exactly once, and return the loaded settings and resolved path.
- [ ] Call it in CLI before `AppRuntimeConfig`/state/jobs and in Tauri immediately after data-root preparation but before `backend::start`, broker, MCP, or any background task. Remove Tauri's duplicate `install_config_path` call.
- [ ] Make `prepare_data_root` return the prepared root and store it for `resolve_data_root`; remove `C:\Users\konst\development\forex-ai\data`. Missing configured data falls back only to the prepared per-user root.
- [ ] Add app test `server_contract_tests_cli_resolves_nondefault_runtime` and Tauri crate tests under module/filter `runtime_initialization_contract`: `tauri_resolves_same_nondefault_runtime` and `portable_fallback_uses_prepared_root`. Compare both to one serialized expected `InitializedRuntime` fixture without making `neoethos-app` import the desktop crate or creating a dependency cycle.

### Task 11: Remove frontend suppressions, unsafe types, and silent failures (M09)

**Files:**
- Add: `desktop/src/apiTypes.ts`
- Modify: `desktop/src/{api.ts,hooks.ts,discoveryQueue.ts}`
- Modify: `desktop/src/components/{KChart.tsx,Select.tsx}`
- Modify: `desktop/src/screens/{Account,Actions,Advanced,AiDesk,Autopilot,Cockpit,Data,Discovery,Files,News,RiskyMode,Settings,StrategyLab,StrategyReport,Supervisor}.tsx`

- [ ] Move domain response/error DTOs to `apiTypes.ts` (keep it under 400 lines) and replace every source-confirmed explicit `any` in the listed files with concrete types or `unknown` plus narrowing.
- [ ] Rename `Settings.useAccount` to `selectBrokerAccount`; replace `KChart` render-time ref writes with effects; make `usePoll` dependency/ref synchronization explicit; remove the suppressions in `hooks.ts` and `Discovery.tsx` by fixing dependencies.
- [ ] Replace synchronous cache-to-state effects in `Select.tsx` and derived tuning state in `Settings.tsx` with lazy state/derived values; keep asynchronous subscriptions cancellable.
- [ ] Replace empty catches in the listed API, queue, component, and screen paths with typed UI state or structured `console.error`/`console.warn` for genuinely best-effort operations. Malformed SSE frames, config reads, broker refresh, and open-path failures may not disappear silently.
- [ ] Do not edit lint configuration, add disables, weaken recommended rules, or rename legitimate React hooks.

### Task 12: Add dependency-free frontend contracts and CSP (M09)

**Files:**
- Modify: `desktop/package.json`
- Add: `desktop/test/{apiContracts,configContracts}.test.ts`
- Modify: `desktop/src-tauri/tauri.conf.json`

- [ ] Add `npm test` as `node --experimental-strip-types --test test/*.test.ts`; add no dependency and do not change `package-lock.json`.
- [ ] Resolve Tauri 2 CSP fields through Context7 before editing. Set production CSP to: `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' asset: http://asset.localhost data: blob:; font-src 'self' data:; connect-src 'self' ipc: http://ipc.localhost; object-src 'none'; base-uri 'self'; frame-ancestors 'none'`. Wave 4's Rust proxy owns the ephemeral backend, so no loopback host/port appears in webview CSP.
- [ ] Set `devCsp` to the same policy plus only `http://localhost:5173` and `ws://localhost:5173`; never add `*`, `unsafe-eval`, or unsafe script sources.
- [ ] In `configContracts.test.ts`, parse Tauri/package config and assert production/dev separation, exact IPC-only production connect origins, no loopback wildcard, non-null CSP, no dependency change, and exact test command.

### Task 13: Establish explicit lazy-load boundaries (M09)

**Files:**
- Modify: `desktop/src/App.tsx`
- Modify: `desktop/vite.config.ts`

- [ ] Replace all eager screen imports with `React.lazy` component loaders and render the selected component inside one `Suspense`; do not pre-create a module-level map of React elements.
- [ ] In Vite Rollup output, define stable `react-vendor`, `tauri-vendor`, and `chart-vendor` chunks for React/ReactDOM, `@tauri-apps`, and `klinecharts`; leave other code split by lazy screen.
- [ ] Add config-contract assertions that every `View` has one lazy loader and the three vendor boundaries exist; final acceptance is every minified JS chunk at most 500 kB with no Vite size warning.

---

## Deferred verification gate for Wave 1

- [ ] **DEFERRED:** Confirm every production/test/config/documentation change for Waves 1-4 is authored and reviewed before executing any command below.
- [ ] **DEFERRED:** Run the approved focused command map for M01-M10/S05 through the full-log capture mechanism. For each Cargo filter, inspect the complete log and require a summed nonzero `running N tests`, zero failures, zero warnings, and exit code zero; a filter that runs only zero tests is a failure.
- [ ] **DEFERRED:** Run `cargo test -p neoethos-desktop runtime_initialization_contract -- --nocapture` as the Tauri half of M08/S05; require both named Tauri tests, exit zero, no warnings, and a complete log.
- [ ] **DEFERRED:** Run `npm run lint`, `npm run build`, and `npm test` separately; require zero warnings/errors, nonzero Node test count, every JS chunk at most 500 kB, and no Vite size warning.
- [ ] **DEFERRED:** Run the design's full workspace, no-default-feature, MCP, mesh, packaging, and audit sequence in order. Capture stdout/stderr from the first byte to EOF under `%TEMP%\forex-ai-remediation-logs`; read every log in bounded start-to-finish chunks, never with tail-only filters or ignored exit codes.
- [ ] **DEFERRED:** If verification prompts any edit, mark authoring open again, finish that edit, and rerun the affected layer plus the final full suite; never add a suppression to make a command green.
