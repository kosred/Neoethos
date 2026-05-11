# Post-release technical debt — 2026-05-12

This file collects items found during the pre-release audit
(2026-05-11 through 2026-05-12) that were intentionally **not** fixed
before cutting the release. Each item has a "why-deferred" rationale
so the next maintainer can make informed prioritisation calls.

## God files — 5 production source files >90 KB each

The audit + test-extraction passes already moved ~6,800 LOC of in-source
test code into sibling `_tests.rs` files (commit `f01bb4aa`). The
remaining bulk is genuine implementation code that benefits from being
split into smaller modules — but each split is a 1-3 hour focused
refactor that adds zero functional value to the release.

| File | Size | Suggested split |
| ---- | ---- | --------------- |
| `crates/forex-models/src/training_orchestrator.rs` | 153 KB | `training_orchestrator/{mod.rs, helpers.rs, params.rs, model_dispatch.rs}` — extract the param-parsing helpers (lines 1887+) and the indicator helpers (lines 1812-1858) first; the main `impl TrainingOrchestrator` block is one long sequence of training pipelines that splits naturally by model family (tree / deep / statistical / ensemble). |
| `crates/forex-app/src/app_services/trading.rs` | 132 KB | `trading/{mod.rs, session.rs, snapshot.rs, panel_state.rs, runtime.rs}` — `TradingSession` is the central struct; the snapshot + panel-state pieces are independent. |
| `crates/forex-models/src/forecasting/swarm_impl.rs` | 117 KB | Single coherent model — split by training vs inference vs serialisation vs the metric helpers. |
| `crates/forex-search/src/discovery.rs` | 110 KB | `discovery/{mod.rs, config.rs, result.rs, cycle.rs, validation_gates.rs, prop_firm.rs, correlation.rs}` — the structure is already in well-named sections separated by `// ─────` comment dividers, which makes mechanical extraction straightforward. |
| `crates/forex-models/src/rl/dqn_impl.rs` | 104 KB | Single coherent RL agent — the cleanest split is by network code vs replay buffer vs training loop vs serialisation. |

**Estimated effort**: 6-12 hours total. **Not** blocking release.

## Audit findings deferred (medium/low severity)

From the parallel audits run on 2026-05-12:

### Worth doing (medium)

- `crates/forex-app/src/app_services/ctrader_data.rs:278, 363` — `unwrap_or(false)` on optional `enabled` and `has_more` fields. Same shape as the money_digits cascade we already fixed; recommend the same `required_*` helper pattern that emits `tracing::error!` on the missing-field path.
- `crates/forex-app/src/app_services/trading.rs:1131, 1191` — `.parse::<i64>().ok()` silently drops invalid order IDs. Add a `tracing::warn!` with the actual raw value before returning `None`.
- `crates/forex-app/src/app_services/broker_persistence.rs:447` — `unwrap_or(0)` on what may be account balance. Verify if this can hide an empty account state at startup.
- `crates/forex-app/src/app_services/ctrader_execution.rs:982` — `unwrap_or(false)` on a trade execution status field. If broker omits the status, we report a successful order as failed (or the reverse).
- `crates/forex-core/src/config.rs:966, 1014, 1024, 1030` — model-config env-var parsing falls back to hardcoded defaults silently when env-var is set but unparseable. Add a `tracing::warn!` with the env-var name and original value.
- `crates/forex-search/src/discovery.rs:1366` — `(total_rows as f64 * stage1_pct) as usize` uses float math then truncates. Tiny precision loss; recommend integer arithmetic with explicit `.ceil()`.
- `crates/forex-search/src/portfolio.rs:62` — magic number `if returns.len() > 5`. Move to a config field on `PortfolioOptimizer`.

### Won't fix (low / cosmetic)

- `crates/forex-search/src/eval.rs:40-52` — `let _ = rayon::ThreadPoolBuilder::new()...build_global();` silently ignores the `AlreadyInit` error. This is intentional — second init attempts are common in test/binary boundaries and the existing pool is fine.
- `crates/forex-search/src/genetic/runtime_overrides.rs:764, 770` — test-only `panic!("expected …")` messages. Could be richer, but they're in `#[cfg(test)]`.
- `crates/forex-app/src/app_services/live_journal.rs:145` — `_guard` naming convention. Cosmetic.
- `crates/forex-search/src/discovery.rs:1468-1475` — comment that documents a bugfix. The fix is in git history; comment is fine.

## Already-fixed in pre-release pass

For the record, these were caught and shipped:

- ✅ `broker_persistence.rs` — env-override path was bypassing temp dirs and **writing to user's real `broker_credentials.toml`** during tests
- ✅ `broker_persistence.rs` — `Mutex` poisoning blew up sibling tests
- ✅ `ctrader_account.rs` + `ctrader_execution.rs` — `money_digits.unwrap_or(0)` would have **scaled balance / equity / P&L 100×** when the cTrader payload omitted the field
- ✅ `ctrader_execution.rs` + `ctrader_streaming.rs` — duplicate exponential-backoff impl, dedup'd into `app_services/backoff.rs`
- ✅ `forex-models/src/base.rs` — `partial_cmp(.).unwrap()` panicked on first NaN sample in distribution-fitting
- ✅ `forex-search/src/genetic/evolution_math.rs` — silent flush failure could drop unsynced data
- ✅ `forex-search/src/cubecl_eval.rs` — silently fell back to GPU 0 when `FOREX_BOT_SEARCH_EVAL_CUDA_DEVICE` was set but unparseable

## Dependabot

The `kosred/forex-ai` repo currently shows **14 open dependabot
vulnerabilities** (6 high, 1 moderate, 7 low) per the GitHub remote.
The two open dependabot PRs on origin are:

- `dependabot/cargo/openssl-0.10.79`
- `dependabot/cargo/rustls-webpki-0.103.13`

Both are security bumps with low merge risk. Recommend merging both
and resolving any remaining advisories before the 0.2 release window.

## Clippy

`cargo clippy --workspace --all-targets --release` is clean of errors
(was failing before). It still emits ~50 warnings (mostly
`dead_code`, `casting to the same type is unnecessary`, and unused
constants from the UI theme module). None of them are correctness
issues. The next refactor pass should `#[allow(...)]` the intentional
ones and remove the rest.
