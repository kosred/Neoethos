# Audit Remediation Design

## Goal

Eliminate every confirmed correctness, security, reliability, packaging, lint, and warning issue found by the final repository audit without hiding diagnostics or destabilizing currently working flows.

## Constraints

- Preserve existing working behavior unless the behavior is the confirmed defect.
- Re-read each producer, consumer, persistence path, and nearby test before editing.
- Prefer the smallest local fix over broad refactors.
- Do not add `allow(...)`, `eslint-disable`, warning filters, `|| true`, ignored failures, or weaker lint/test settings.
- Complete all source and test-file writing before running any build, test, lint, formatter, or package-manager verification command.
- During verification, capture and read complete outputs rather than only the final lines.
- Reuse the existing Cargo target directory and existing Node dependency store to minimize disk use; never run `cargo clean`.
- Keep the operator's existing `data/symbol_metadata.json` change untouched on `master`.

## Architecture of the remediation

The work is split into four ordered waves. Each wave contains small, independently reviewable patches, but verification is deliberately deferred until all four waves are written.

### Wave 1: Mechanical and contract safety

Fix low-blast-radius defects first: frontend/backend DTO casing, MCP service lifetime, CI toolchain/script references, package versions/hashes/licenses, deterministic JSON output, atomic persistence, fixed developer paths, and existing lint failures. These changes do not intentionally alter trading mathematics.

### Wave 2: Data and mathematical correctness

Unify timestamp units as milliseconds, model higher-timeframe availability explicitly, reject malformed market rows, make timeframe-sensitive features use actual timeframe/calendar information, correct Hurst and capped-simplex math, and remove process-global dataset state.

### Wave 3: Backtest, validation, model, and trader parity

Close or mark open positions consistently, preserve SL/TP/confidence through replay, evaluate disjoint validation islands independently, prevent train/validation leakage, require complete fold coverage, tighten probability/artifact contracts, and make checkpoint completion truthful.

### Wave 4: Security and live-risk enforcement

Authenticate state-changing local API calls, restrict CORS, enforce MCP approval at both caller and sidecar boundaries, connect the real risk manager to every live intent, make risky-mode persistence fail closed, and authenticate/stage/validate mesh work and artifacts.

## Scope and ownership

In scope are all project-owned Rust, TypeScript/React, TOML, JSON, YAML, CI, packaging, MCP, and mesh files implicated by the audit. NeoEthos-specific vendored build scripts and patches are in scope. Untouched upstream vendor implementations, binary images/fonts, generated Cargo output, and platform signing credentials are out of scope unless verification demonstrates that NeoEthos integration is broken.

Warnings owned by NeoEthos source, tests, manifests, build scripts, or vendored patches must be fixed. Diagnostics emitted solely inside an unchanged third-party dependency are recorded and evaluated, but are not hidden or patched blindly; they become an escalation item if they make a required command non-zero. Optional CUDA/Vulkan/ROCm builds are verified only when the required local SDK/hardware is present; feature-graph compilation that does not require unavailable hardware remains required.

Every high-risk change is independently revertible by task/commit. Compatibility defaults preserve current local operation: legacy clients can discover the new API session token through the desktop bootstrap path, legacy persisted files remain readable, and new writes use the safer canonical format. If a safe compatibility path cannot be proved from existing code, the affected feature is disabled fail-closed with an explicit operator error rather than guessed.

## Finding traceability

| ID | Confirmed defect and producer/consumer | Intended correction | Required written regression coverage / acceptance evidence |
|---|---|---|---|
| M01 | `desktop/src/api.ts` emits snake_case bodies while Axum DTOs in orders/data-control/strategy-lab require camelCase | Use one camelCase HTTP contract and reject unknown money-sensitive fields | Router tests deserialize each real frontend payload; typed TS wrappers match the DTO fields |
| M02 | `mcp::connect` drops `RunningService` while `AppState` retains only `Peer` | Store running services for process lifetime and close them on shutdown; health checks transport state | HTTP and stdio lifecycle tests prove tools remain callable after `connect` returns |
| M03 | CI invokes a missing script and selects stable while `.cargo/config.toml` passes nightly-only flags | Restore/remove the stale guard intentionally and use the pinned nightly on every job | Workflow/static contract test verifies referenced files and pinned toolchain |
| M04 | Package versions/hashes disagree and packaging smoke expects a missing portable builder | Generate coherent 0.5.3 manifests, remove placeholders, and remove the obsolete portable-builder expectation in favor of the currently shipped AppImage/deb/rpm/Tauri paths | Packaging smoke and manifest parsers pass; no empty/all-zero digest remains |
| M05 | Root/NOTICE say AGPL while Cargo/deb/rpm/Scoop/WinGet say proprietary | Align all first-party metadata to AGPL-3.0-or-later | Metadata search finds no contradictory first-party license declaration |
| M06 | `SymbolMetadataTable.entries` is a `HashMap`, producing order-only rewrites | Serialize through a sorted representation without changing the public lookup API | Two saves of equivalent maps are byte-identical and round-trip |
| M07 | Settings/risky/symbol state uses direct or fixed-temp writes and unlocked read-modify-write | Shared atomic-write primitives, unique same-directory temp files, fsync/replace, writer lock | Concurrent-write, legacy-read, corruption, and replacement tests |
| M08 | Desktop data fallback embeds the developer workstation path | Derive fallback from the prepared per-user data root | Path test contains no workstation-specific component |
| M09 | Frontend lint is red, CSP disabled, no frontend contract tests, oversized single chunk | Correct hook/type/render issues, define production CSP, add contract tests, split large routes/components only where needed | ESLint/build/test zero-warning output and CSP config validation |
| M10 | LightGBM build can combine stale copied C++ with fresh bindings; vendor provenance/licenses incomplete | Content-aware fresh staging, target-correct bindgen/build decisions, correct Cargo directives, provenance/license artifacts | Build-script unit/static tests and clean native rebuild evidence |
| D01 | Canonical ms timestamps are bucketed/aged with ns constants | Millisecond-only duration helpers and `_ms` naming | Epoch-ms M1→M5/H1 resample and stale-age tests |
| D02 | Finished higher-TF values are stamped at open and aligned immediately | Separate candle open time from availability-at-close; exclude incomplete final bucket | Appending future intra-candle bars cannot change already-available feature rows |
| D03 | Malformed imports coerce required prices/timestamps to zero | Checked parsing, positive OHLC/plausible timestamp validation, counted rejection | Ragged/blank/invalid CSV and JSON fixtures are rejected explicitly |
| D04 | Day/week/ORB/pivot/volatility features assume fixed bar counts | Pass timeframe/calendar context and derive real session windows/annualization | Equivalent calendar fixtures across M1/M5/H1/D1 have consistent semantics |
| D05 | Hurst operates on differences of returns rather than log-price increments | Implement validated log-price scaling estimator | Random-walk, persistent, and anti-persistent fixtures classify correctly |
| D06 | Dataset-derived adaptive ladder is process-global first-write-wins | Carry ladder in per-run state or key it by dataset identity | Multi-symbol order-independence test |
| D07 | “Block bootstrap” shuffles without replacement and uses unseeded RNG | Sample blocks with replacement using persisted deterministic seed | Same seed is reproducible; different seeds vary; samples can repeat/omit blocks |
| D08 | Weight clamp followed by renormalization violates `max_weight` | Capped-simplex/water-fill allocation with explicit infeasible residual handling | Property tests enforce nonnegative weights, cap, and legal sum |
| D09 | Feature normalization fits median/MAD on the full series | Fit on training prefix/fold and persist/apply immutable stats | Appending future rows leaves historical normalized rows unchanged |
| B01 | Final open backtest position disappears from trade/PnL/cost metrics | Force-liquidate at the final close through the normal exit-cost/carry path, shared by metric and trade paths | Last-bar and long-open fixtures reconcile metrics and trade ledger |
| B02 | Walk-forward reuses already-selected full-series genes; CPCV concatenates islands | Reserve an outer chronological split: search/train prefix, bounded validation segment, and untouched final OOS segment separated by configured purge/embargo; freeze candidates before either evaluation segment; reset state per contiguous island | Mutating validation/OOS observations cannot change selected genes; island-boundary position reset test |
| B03 | Genetic evolution repeatedly selects on its validation tail | Train-only evolution plus bounded validation selection and untouched purged OOS test | Mutation of OOS tail cannot change evolved population |
| B04 | Replay loses gene confidence/native brackets and default decision math diverges | Carry one combined signal/confidence/SL/TP structure through replay and live adapters | Replay intents match source gene brackets, confidence, sizing, and pip conversion |
| B05 | Gap stops fill optimistically; invalid close volume/fill mutates positions | Conservative gap fills and strict finite/positive/within-position validation before mutation | Gap-through, negative/NaN/oversized partial-close tests |
| B06 | Sharpe duration uses active trade days rather than elapsed calendar span | Derive elapsed trading horizon from first/last timestamps | Sparse year-long strategy annualizes from the full year |
| B07 | Promotion aggregation can turn NaN drawdown into zero | Reject every non-finite promotion metric before aggregation | NaN/+∞/-∞ tests fail closed for every criterion |
| B08 | Regime indices from filtered/budgeted frames are applied to original OHLCV rows | Carry original row identity/timestamp through all filters and join by identity | Shifted/sparse fixture preserves correct regime labels |
| B09 | Model CPCV accepts partial fold sets and records the first fold as representative | Require every planned fold to succeed; reject the candidate on any fold failure; aggregate means for score-like metrics and worst-case values for loss/drawdown metrics | Injected later-fold failure rejects the candidate with recorded coverage |
| B10 | HPO accepts invalid probability rows and small-data path emits an invalid empty report | Normalize/validate one probability representation; use `None` or a valid base trial for no-HPO | Invalid simplex tests and small-dataset persistence test |
| B11 | DQN no-feature artifact integrity differs from feature-enabled path; precision tests drift from FP32 policy | Enforce snapshot claims in every feature branch and make one explicit supported precision contract | Default/no-default artifact tests and FP32 policy tests agree |
| B12 | Swarm live adapter gets one row although it requires history; degradation reason is overwritten | Pass historical context and preserve the specific degradation reason | Loaded swarm artifact emits a non-neutral last-row vote; reason round-trips |
| B13 | `exit_agent` is trained without a production consumer | Remove automatic `exit_agent` training while preserving an explicit operator-requested training path; do not register it as a voter until a real consumer exists | Default model inventory produces no orphan exit-agent output; explicit request remains truthful |
| B14 | CLI auto-loop records failed discovery/training as completed | Checkpoint only complete successful stages and persist retryable failure state | Resume retries failed work and skips only successful work |
| B15 | Codex auth freshness/401, raw `Debug`, SSE failure parsing, and callback read timeout are unsafe | One locked refresh/retry, redacted debug/raw extras, strict terminal SSE errors, overall callback deadline | Expired-token retry, formatted-secret redaction, failed/truncated SSE, slow-client tests |
| S01 | Loopback Axum API has no auth and `CorsLayer::Any` | Per-launch session token for state-changing/private routes, exact origins, refuse unsafe non-loopback mode | Hostile-origin and missing/wrong-token requests fail; desktop bootstrap succeeds |
| S02 | Free-form Supervisor MCP calls bypass pending approval | Read-only `(server, tool)` allowlist; all other tools become signed pending actions; sidecar rechecks | Mutating/unknown MCP calls cannot execute without confirmation |
| S03 | Real `RiskManager`/`RiskyModeManager` has no production live call path | Adapt every live open intent through the real gate, record trip, fail closed on corrupt state | Live-intent integration tests cover allow/reject/restart/corruption/re-arm |
| S04 | Mesh lacks trusted membership/leases/quotas/staging and leaks the whole model store; federation token is not forwarded | Authenticate endpoints, bind jobs to leases, cap streams, stage/hash/schema-check artifacts, transfer only job outputs, forward protected credentials | Unauthorized/oversized/tampered/wrong-lease submissions fail; job output isolation test |
| S05 | Tauri startup omits the runtime override installers used by CLI | Extract one shared settings initialization path used by both entry points | Non-default runtime knob produces identical CLI/Tauri resolved state |

Each matrix row must end as fixed with evidence or rejected as an invalid audit finding with a written code-path proof. No row may disappear from the implementation plan.

## Data-flow rules

- Canonical OHLCV timestamps remain milliseconds everywhere.
- A higher-timeframe feature becomes available at candle close, never at candle open.
- Validation/test observations cannot participate in fitting, search, normalization, survivor selection, or hyperparameter selection.
- Non-contiguous validation islands reset position state at every boundary.
- Live and replay paths carry the same signal direction, confidence, bracket, sizing, and cost metadata.
- Invalid or non-finite financial values are rejected rather than coerced to zero or silently ignored.
- State-changing operations require authenticated and authorized requests; loopback alone is not authentication.

## Local API authentication contract

The token is a 256-bit OS-random value rotated on every backend start. It is never printed. In Tauri mode, the Rust setup stores it in managed state and a custom command returns `{ baseUrl, bearerToken }` only to the bundled `main` window; the production CSP permits only bundled scripts and the generated loopback backend. In headless mode, the operator may provide `NEOETHOS_API_TOKEN`; otherwise the backend writes the generated token atomically to the user-config directory with owner-only permissions and logs only that file's path. The file is replaced on startup and removed best-effort on clean shutdown.

`/healthz` remains unauthenticated. All other private reads and every state-changing route require `Authorization: Bearer <token>`. Production CORS allows only the Tauri application origin; development may additionally allow the exact configured Vite origin. Non-loopback binding is refused unless an explicit remote-enable setting is present and token authentication is active. There is no endpoint that returns the token over the unauthenticated HTTP API, so a browser origin cannot “discover” it.

## Error handling

- Fail closed for authentication, risk-state corruption, artifact-integrity failures, and unknown mutating MCP tools.
- Surface structured errors to the UI/logs instead of converting them to `None`, empty output, or success.
- Use same-directory unique temporary files, flush/sync, atomic replace, and shared writer locks for canonical state.
- Preserve specific degradation/failure reasons instead of replacing them with generic fallbacks.

## Verification design

“Initial authoring complete” means production code, test code, fixtures, manifests, and documentation for every matrix row have been written and reviewed without executing build/test/lint/format commands. Only then does the first verification cycle begin. Formatter changes and fixes prompted by verification reopen the authoring state; after those edits, the entire affected verification layer and final full suite run again. This edit/reverify loop continues until clean.

The required command order is:

1. `cargo fmt --all -- --check`
2. `cargo build --workspace --all-targets --locked`
3. `cargo check --workspace --all-targets --locked`
4. `cargo clippy --workspace --all-targets --locked -- -D warnings`
5. Focused regression commands in the command map below.
6. `cargo test --workspace --all-targets --locked --no-fail-fast`
7. `cargo check --workspace --all-targets --locked --no-default-features`
8. `cargo test --workspace --all-targets --locked --no-default-features --no-fail-fast`
9. In `desktop`: `npm run lint`, `npm run build`, `npm test`, `npm audit --omit=dev`.
10. For each of `mcp/Cargo.toml` and `mesh/Cargo.toml`: `cargo fmt --manifest-path <manifest> -- --check`, `cargo build --manifest-path <manifest> --all-targets --locked`, `cargo check --manifest-path <manifest> --all-targets --locked`, `cargo clippy --manifest-path <manifest> --all-targets --locked -- -D warnings`, and `cargo test --manifest-path <manifest> --all-targets --locked --no-fail-fast`.
11. `cargo test -p neoethos-app --test packaging_smoke -- --nocapture` and `cargo test -p neoethos-core --test vendor_provenance_contract -- --nocapture`.
12. `cargo audit --file Cargo.lock`, `cargo audit --file mcp/Cargo.lock`, `cargo audit --file mesh/Cargo.lock`, plus `npm audit --omit=dev`.

### Focused regression command map

The named tests are written during authoring before any command is executed.

| IDs | Exact command after authoring |
|---|---|
| M01, S01, S05 | `cargo test -p neoethos-app server_contract_tests -- --nocapture` |
| M02, S02 | `cargo test --manifest-path mcp/Cargo.toml lifecycle_and_approval -- --nocapture` |
| M03, M04, M05, M10 | `cargo test -p neoethos-core --test vendor_provenance_contract -- --nocapture` and `cargo test -p neoethos-app --test packaging_smoke -- --nocapture` |
| M06, M07, M08 | `cargo test -p neoethos-core persistence_contract -- --nocapture` and `cargo test -p neoethos-app persistence_contract -- --nocapture` |
| M09 | Run `npm run lint`, `npm run build`, and `npm test` as separate commands; acceptance additionally requires every generated JS chunk to be at most 500 kB minified and no Vite size warning |
| D01, D02 | `cargo test -p neoethos-data timestamp_and_mtf_contract -- --nocapture` |
| D03, D04 | `cargo test -p neoethos-data import_and_calendar_contract -- --nocapture` |
| D05, D06, D07, D08 | `cargo test -p neoethos-search corrected_math_contract -- --nocapture` |
| D09, B08 | `cargo test -p neoethos-data causal_preprocessing_contract -- --nocapture` and `cargo test -p neoethos-models causal_preprocessing_contract -- --nocapture` |
| B01, B02, B06 | `cargo test -p neoethos-search validation_and_terminal_position_contract -- --nocapture` |
| B03, B09, B10 | `cargo test -p neoethos-models selection_and_hpo_contract -- --nocapture` |
| B04, B05 | `cargo test -p neoethos-trader replay_and_position_contract -- --nocapture` |
| B07 | `cargo test -p neoethos-core promotion_nonfinite_contract -- --nocapture` |
| B11, B12, B13 | `cargo test -p neoethos-models artifact_and_inventory_contract -- --nocapture` and `cargo test -p neoethos-models --no-default-features artifact_and_inventory_contract -- --nocapture` |
| B14 | `cargo test -p neoethos-cli auto_loop_checkpoint_contract -- --nocapture` |
| B15 | `cargo test -p neoethos-codex auth_stream_callback_contract -- --nocapture` |
| S03 | `cargo test -p neoethos-app live_risk_contract -- --nocapture` and `cargo test -p neoethos-trader live_risk_contract -- --nocapture` |
| S04 | `cargo test --manifest-path mesh/Cargo.toml mesh_trust_contract -- --nocapture` |

Commands run without `-q`, `|| true`, tail-only filters, ignored exit codes, or warning suppression. Complete stdout/stderr is written per command to `%TEMP%\forex-ai-remediation-logs\<cycle>-<command>.log`. Logs are read from start to finish in bounded chunks; old successful-cycle logs are deleted only after their diagnostic summary is recorded. A single log is capped operationally at 100 MiB: if a command would exceed that, output is split by byte range without discarding any segment.

Disk policy: set `CARGO_TARGET_DIR=C:\Users\konst\development\forex-ai\target` to reuse the existing target; never run `cargo clean`; junction the worktree `desktop/node_modules` to the existing dependency directory rather than reinstall; install audit-only tools under `%TEMP%\forex-ai-tools`; remove superseded temporary logs/tools after verification. The junction is permitted only while `desktop/package.json` and `desktop/package-lock.json` remain byte-identical to `master`. No dependency change is planned. If either lockfile or manifest changes, remove the junction and run `npm ci` in the worktree using the existing npm cache after the disk check. Before a command expected to add more than 2 GiB, measure free disk and skip/escalate only if fewer than 8 GiB remain.

Time policy: mechanical and focused checks precede full rebuilds so cheap diagnostics are exhausted first. A command with active CPU/disk progress is allowed to finish. A command with no output and no CPU/disk progress for 10 minutes is diagnosed rather than killed blindly. Missing proprietary signing credentials or unavailable GPU SDK/hardware are reported as external blockers, not silenced and not treated as permission to weaken required CPU/default checks.

## Completion criteria

- No confirmed audit defect remains unfixed or explicitly proven invalid with evidence.
- Full workspace build/test/lint/format commands exit zero.
- Frontend, MCP, mesh, packaging, and dependency checks exit zero.
- No warnings, ignored new failures, hidden diagnostics, stale placeholder hashes, or contradictory release metadata remain.
- The remediation branch contains only intentional source/test/config/documentation changes.
