# Codex Control Plane — MCP Server Design

Status: DESIGN (committed decisions, no open options)
Date: 2026-08-08
Scope: give OpenAI Codex CLI (and any MCP-speaking agent runtime) full operational
control of the NeoEthos backend — open/close/modify trades, autopilot, discovery,
training, promotion, settings, journal analytics, account state, system health —
under a hard, server-side, fail-closed DEMO-ONLY guard.

---

## 1. Architecture

### 1.1 Crate and binary

- **Crate:** `crates/neoethos-mcp` (new member of the root workspace, added to
  `members` and `default-members` so plain `cargo build --release` produces it).
  The top-level `mcp/` directory (the existing MCP **client** aggregator sidecar,
  its own excluded workspace) is unrelated and untouched; the name collision is
  resolved by path: `mcp/` = outbound client sidecar, `crates/neoethos-mcp` =
  inbound control plane server.
- **Binary:** `neoethos-mcp` → `target/release/neoethos-mcp.exe`.
- **SDK:** official `rmcp`, pinned exactly: `rmcp = { version = "=3.1.2",
  features = ["server", "transport-io", "macros"] }` (MCP spec 2026-07-28; same
  protocol implementation family Codex CLI itself ships). Pin is exact because
  rmcp 3.x moved 3.0.0→3.1.2 in ten days; upgrades are deliberate, not drive-by.
- **Other deps:** `reqwest` (workspace pin 0.13.3, rustls), `tokio`, `serde`,
  `serde_json`, `anyhow`, `schemars` (schema generation), `tracing` +
  `tracing-subscriber` writing to **stderr only**.

### 1.2 Transport and backend access

**stdio MCP transport; the server is a thin HTTP client of the running backend at
`http://127.0.0.1:7423`.** It does NOT link the engine crates. Rationale: the
backend process is the single owner of all state — broker session, engine
lifecycles, config resolution, journal store. Linking `neoethos-core`/`-app`
in-process would create a second, independent `Settings`/state resolution in a
second process (the exact "two Settings in one process" defect class this repo
has already been burned by, now across two processes racing the same
`config.yaml` and `broker_credentials.toml`), would let the MCP process start
engines the desktop UI cannot see or stop, and would drag the entire GPU/ML
dependency tree into what must be a small, instantly-starting binary whose
stdout is a JSON-RPC channel. Wrapping HTTP reuses exactly the surface the
desktop frontend already exercises (same DTOs, same validation, same loud
errors), so the control plane can never diverge from what the UI does.

**Backend-not-running is a first-class, loud failure.** Every tool call goes
through one HTTP helper. On `ECONNREFUSED`/timeout it returns an MCP tool error
(`isError: true`) with this exact text:

> `NeoEthos backend is not reachable at http://127.0.0.1:7423 (connection
> refused). Start the desktop app or run 'neoethos-app --server', then retry.
> This control plane never falls back to acting on its own.`

No retry loops, no silent degradation, no in-process fallback — by design.

**Configuration:** one CLI flag, `--base-url` (default
`http://127.0.0.1:7423`), passed via `args` in the Codex config — a flag, not an
env var, per the one-config/no-env-vars directive. If the operator sets
`NEOETHOS_API_TOKEN` on the backend, the same value is passed to the MCP server
as `--token <value>` and sent as a bearer header; absent by default because the
backend requires no auth on loopback.

**stdout discipline:** stdout carries JSON-RPC exclusively. `tracing_subscriber`
is initialized with a stderr writer before anything else; the crate contains no
`println!`. A CI test asserts the binary's stdout stays protocol-clean (§6.1).

### 1.3 Long-running work: job pattern, never blocking calls

Codex's default `tool_timeout_sec` is 60s. Discovery and training are exposed
as **start / poll / stop** triples (`discovery_start` returns as soon as the
backend accepts the job; progress is read via `engine_status`; the backend's
`/engines/*` endpoints already work exactly this way). The only deliberately
long synchronous tool is `replay_backtest` (offline dry-run, zero broker calls);
the shipped Codex config raises the server's `tool_timeout_sec` to 600 to cover
it (and slow `fetch_history` downloads).

### 1.4 Explicitly NOT exposed (committed exclusions)

| Backend surface | Why it is excluded |
|---|---|
| `POST /broker/credentials`, `POST /broker/account/select`, `POST /broker/reauth` | These can flip the whole system Demo→Live or change which account trades. Operator-only, via the desktop app. The control plane must be structurally incapable of switching itself to live money. |
| `GET/POST /settings/raw` | Verbatim `config.yaml` write can change any knob including broker-adjacent behavior, bypassing the typed DTO's field allowlist. The typed `get_settings`/`update_settings` + `knob_catalog` cover every legitimate need. |
| `/supervisor/*` | One agent brain at a time. Codex IS the supervisor when this control plane is in use; exposing the internal LLM supervisor's tick/chat would create two autonomous actors issuing StartLive-class actions concurrently. |
| `/auth/codex/*`, `POST /codex/chat` | OAuth/browser flows are operator-only; proxying chat through the app is pointless from inside Codex. |
| `/federation/*`, `/mesh/*`, `/mcp/*` (client-sidecar mgmt) | Not needed to operate trading/discovery/training; add later as a separate tool group if distributed search control is ever wanted from Codex. |
| Tauri commands | In-process only, unreachable by design; HTTP covers everything they do (and the HTTP `POST /orders` has the naked-order guard the Tauri `place_order` lacks). |

---

## 2. Demo guard (the security boundary)

**Where it lives:** inside the MCP server binary, in every trade-affecting tool
handler, enforced by the compiler. The only function able to reach a
trade-affecting backend route is a private helper whose signature requires a
`DemoProof` value:

```rust
struct DemoProof(());            // not Clone, not Send across calls, private ctor

impl Backend {
    /// The ONLY way to obtain a DemoProof.
    async fn ensure_demo(&self) -> Result<DemoProof, ToolError> { ... }

    /// The ONLY function that may POST to /orders, /orders/pending,
    /// /orders/cancel, /positions/*, /actions/*/confirm, /autonomous/start,
    /// /strategy_lab/promote, /risk/preset.
    async fn trade_post<B: Serialize>(&self, path: &str, body: &B, _proof: DemoProof)
        -> Result<Value, ToolError> { ... }
}
```

A handler that forgets the guard does not compile. Annotations
(`destructive_hint`) are honest UX hints for Codex's approval routing, but they
are **never** the protection — per the MCP spec clients treat annotations as
untrusted, and symmetrically this server never assumes the client prompted a
human. If the operator sets `approval_mode = "auto"` on the Codex side, the
guard still holds.

**What `ensure_demo()` checks, fresh on every call (no caching):**

1. `GET /broker/status` → require `connected == true` **and**
   `environment == "Demo"` (this string is derived from
   `broker_credentials.toml`'s `CTraderBrokerEnvironment`, the same source the
   execution path uses — `app_services/live_gate.rs::active_env_is_live`).
2. `GET /broker/accounts` → resolve the active `account_id` from step 1 and
   require that account's `is_live == Some(false)`. `None` (unknown) **fails** —
   fail-closed: absence of proof-of-demo is treated as live.
3. Any HTTP failure in steps 1–2 fails the guard with the backend-unreachable
   message (§1.2).

**Failure messages (verbatim, returned as `isError: true`):**

- Environment is Live:
  > `DEMO GUARD REFUSED: the active broker environment is 'Live' (account
  > <id>). This control plane operates demo accounts only — no trade-affecting
  > tool will execute against a live environment, ever. Switch the active
  > account to Demo in the desktop app, then retry.`
- Cannot verify:
  > `DEMO GUARD REFUSED (fail-closed): cannot positively verify the active
  > account is a demo account (broker connected=<bool>, environment=<str>,
  > is_live=<opt>). Trade-affecting tools stay locked until /broker/status
  > reports connected=true + environment=Demo and /broker/accounts reports
  > is_live=false for the active account.`

**One deliberate exemption:** `autopilot_stop` (`POST /autonomous/stop`) does
not require the guard. It executes no broker order — it halts engines and
leaves positions untouched — and the agent must always be able to stop trading,
including in exactly the scenario where the guard has started refusing.

**Additional hard rails inside handlers:**

- `open_position` and `place_pending_order` schemas **require**
  `stop_loss_pips` (stricter than the backend, which allows naked orders with
  `risky:true`). The MCP server hardcodes `risky: false` and has no parameter to
  change it.
- `close_position` takes `volume_lots` (optional; omitted = full close) and
  performs the lots→centi-lot wire-unit conversion itself after reading the
  position's current volume from `/account/snapshot`, so the agent never
  touches wire units.
- `update_settings` forwards only the typed `POST /settings` DTO fields; there
  is no passthrough for unknown keys.

---

## 3. Tool catalog (complete)

Side-effect classes: **RO** = readonly (`read_only_hint=true`), **MUT** =
mutating non-trade (`read_only_hint=false, destructive_hint=false`), **TRADE** =
trade-affecting (`read_only_hint=false, destructive_hint=true`, demo guard
required). `open_world_hint=false` on everything except `news_briefing`
(outbound RSS fetch). `idempotent_hint=true` only where noted.

### System and health

| # | Tool | Params (JSON schema, required in **bold**) | Class | Wraps |
|---|---|---|---|---|
| 1 | `system_health` | `{}` | RO | `GET /healthz` (+ round-trip latency ms) |
| 2 | `hardware_info` | `{}` | RO | `GET /hardware` |
| 3 | `engine_status` | `{}` | RO | `GET /engines/status` — the poll half of the discovery/training job pattern |
| 4 | `storage_paths` | `{}` | RO | `GET /storage/paths` |
| 5 | `diagnostics_report` | `{}` | MUT (writes report file) | `POST /diagnostics/report` |

### Account and broker state

| # | Tool | Params | Class | Wraps |
|---|---|---|---|---|
| 6 | `broker_status` | `{}` | RO | `GET /broker/status` + `GET /broker/accounts` merged: adapter, environment, connected, active account, all accounts with `is_live` — the agent-visible face of the demo guard |
| 7 | `account_snapshot` | `{refresh?: bool}` | RO (refresh does a broker round-trip, no trade) | `GET /account/snapshot`; when `refresh` → `POST /account/snapshot/refresh` first |
| 8 | `list_positions` | `{}` | RO | positions array of `GET /account/snapshot` |
| 9 | `broker_symbols` | `{}` | RO | `GET /broker/symbols` |
| 10 | `order_history` | `{from_ms?: int, to_ms?: int}` | RO | `GET /broker/orders/history` |
| 11 | `cashflow` | `{}` | RO | `GET /broker/cashflow` |
| 12 | `expected_margin` | `{symbol: str, volume_lots: num}` | RO | `GET /broker/margin/expected` |

### Market data

| # | Tool | Params | Class | Wraps |
|---|---|---|---|---|
| 13 | `get_chart` | `{**symbol**, **timeframe**, limit?: int}` | RO | `GET /chart` |
| 14 | `get_indicator` | `{**symbol**, **timeframe**, **indicator**, period?, std_dev?, fast?, slow?, signal?, k_period?, k_slow?, d_period?, limit?}` | RO | `GET /indicators` |
| 15 | `live_spots` | `{}` | RO | `GET /live/spots` (single read of the SSE source) |
| 16 | `spread_stats` | `{}` | RO | `GET /data/spread-stats` |
| 17 | `news_briefing` | `{force?: bool}` | RO, open-world | `GET /news/feed?force=` |
| 18 | `get_watchlist` | `{}` | RO | `GET /watchlist` |
| 19 | `set_watchlist` | `{**symbols**: [str]}` | MUT | `POST /watchlist` (live spot stream re-subscribes ~5s) |

### Trading — direct orders (ALL demo-guarded)

| # | Tool | Params | Class | Wraps |
|---|---|---|---|---|
| 20 | `open_position` | `{**symbol**, **side**: buy\|sell, **volume_lots**: num, **stop_loss_pips**: num, take_profit_pips?: num, comment?: str}` | TRADE | `POST /orders` with `risky:false` hardcoded; SL required by schema (stricter than backend) |
| 21 | `close_position` | `{**position_id**, volume_lots?: num}` (omit = full close; server converts lots→wire centi-lots) | TRADE | `POST /positions/close` |
| 22 | `modify_position_protection` | `{**position_id**, stop_loss_price?: num, take_profit_price?: num, trailing_stop?: bool}` — at least one, enforced | TRADE | `POST /positions/protection` (absolute prices) |
| 23 | `place_pending_order` | `{**symbol**, **side**, **order_type**: limit\|stop, **volume_lots**, **trigger_price**, **stop_loss_pips**, take_profit_pips?, expiry_unix_ms?, comment?}` | TRADE | `POST /orders/pending` (GTC when no expiry) |
| 24 | `list_pending_orders` | `{}` | RO | `GET /orders/pending` |
| 25 | `cancel_pending_order` | `{**order_id**}` | TRADE | `POST /orders/cancel` |

### Trading — human-confirm queue (#136)

| # | Tool | Params | Class | Wraps |
|---|---|---|---|---|
| 26 | `list_pending_actions` | `{}` | RO | `GET /actions/pending` |
| 27 | `confirm_pending_action` | `{**action_id**, volume_units_override?: int}` | TRADE (confirm EXECUTES the broker call) | `POST /actions/{id}/confirm` |
| 28 | `reject_pending_action` | `{**action_id**, reason?: str}` | MUT (audit only) | `POST /actions/{id}/reject` |

### Autopilot (autonomous trader)

| # | Tool | Params | Class | Wraps |
|---|---|---|---|---|
| 29 | `autopilot_start` | `{**portfolio_paths**: [str], **lot_size**: num, stop_loss_pips?, take_profit_pips?, warmup_bars?: int, cull_after_consecutive_losses?: int, cull_min_win_rate_pct?: num, cull_window_trades?: int}` | TRADE | `POST /autonomous/start` |
| 30 | `autopilot_stop` | `{}` | MUT, idempotent — **guard-exempt** (halts engines, no broker order; must work even when the guard refuses) | `POST /autonomous/stop` |
| 31 | `autopilot_status` | `{}` | RO | `GET /autonomous/status` |
| 32 | `demo_gate_status` | `{**portfolio**: str}` | RO | `GET /autonomous/gate` |
| 33 | `replay_backtest` | `{symbol?, base_tf?}` | RO compute (long; zero broker calls) | `POST /autonomous/replay` |
| 34 | `parity_check` | `{**portfolio**, window?: int, reference?: str}` | RO | `GET /autonomous/parity` |
| 35 | `tail_risk` | `{**portfolio**, iterations?: int, risk?: num}` | RO | `GET /autonomous/tailrisk` (Monte-Carlo p95 DD/ruin) |
| 36 | `challenge_sim` | `{**portfolio**, iterations?: int}` | RO | `GET /autonomous/challenge` |

### Discovery and training (job pattern: start → `engine_status` → stop)

| # | Tool | Params | Class | Wraps |
|---|---|---|---|---|
| 37 | `discovery_start` | `{symbol?, base_tf?, higher_tfs?: [str], population?: int, generations?: int, max_indicators?: int, target_candidates?: int, portfolio_size?: int}` — omitted fields resolve from config.yaml exactly like the CLI | MUT (long compute) | `POST /engines/discovery/start` |
| 38 | `discovery_stop` | `{}` | MUT, idempotent | `POST /engines/discovery/stop` |
| 39 | `training_start` | `{symbol?, base_tf?, higher_tfs?: [str]}` | MUT (long compute) | `POST /engines/training/start` |
| 40 | `training_stop` | `{}` | MUT, idempotent | `POST /engines/training/stop` |
| 41 | `experience_train` | `{}` | MUT (writes models/report; never touches live) | `POST /experience/train` |

### Strategy lab and portfolios

| # | Tool | Params | Class | Wraps |
|---|---|---|---|---|
| 42 | `promotion_gate_status` | `{}` | RO | `GET /strategy_lab/promotion` |
| 43 | `promote_strategies` | `{symbol?, base_tf?}` | TRADE (changes what the autopilot trades) | `POST /strategy_lab/promote` |
| 44 | `list_portfolios` | `{}` | RO | `GET /portfolios/list` |
| 45 | `list_strategies` | `{}` | RO | `GET /strategy/list` |
| 46 | `strategy_report` | `{dir?: str, base?: str}` | RO | `GET /strategy/report` |
| 47 | `strategy_blacklist` | `{}` | RO | `GET /strategy/blacklist` |

### Settings and risk

| # | Tool | Params | Class | Wraps |
|---|---|---|---|---|
| 48 | `get_settings` | `{}` | RO | `GET /settings` (typed DTO) |
| 49 | `update_settings` | partial mirror of the `POST /settings` DTO: `{trading_mode?, compute_mode?, risk_per_trade?, search_population?, search_generations?, search_max_hours?, max_indicators?, portfolio_size?, corr_threshold?, max_rows?, prefilter_top_k?, convergence_patience?, stagnation_patience?, novelty_weight?, disable_smc_gate?, max_portfolio_risk?, live_ml_gate?, risky_*?, news_*?}` — typed fields only, no passthrough | MUT (several knobs alter live sizing → `destructive_hint=true` as the honest hint, but classed MUT: it executes no trade) | `POST /settings` |
| 50 | `knob_catalog` | `{}` | RO | `GET /settings/knob-catalog` + `GET /settings/presets` — the machine-readable help Codex reads before touching `update_settings` |
| 51 | `get_risk` | `{}` | RO | `GET /risk` |
| 52 | `apply_risk_preset` | `{**preset**: str}` | TRADE (rewrites live sizing config) | `POST /risk/preset` |
| 53 | `risky_scenarios` | `{starting_usd?, target_usd?, risk_fraction?, win_rate?, reward_to_risk?}` | RO | `GET /risky/scenarios` |

### Journal and analytics

| # | Tool | Params | Class | Wraps |
|---|---|---|---|---|
| 54 | `journal_trades` | `{from_ms?: int, to_ms?: int, limit?: int (default 500)}` | RO | `GET /journal/trades` |
| 55 | `journal_stats` | `{}` | RO | `GET /journal/stats` |
| 56 | `journal_analytics` | `{}` | RO | `GET /journal/analytics` (pips/R/MFE-MAE per hour/symbol/day — not even in the UI yet; Codex gets it first) |

### Data management

| # | Tool | Params | Class | Wraps |
|---|---|---|---|---|
| 57 | `data_coverage` | `{}` | RO | `GET /data/bootstrap` |
| 58 | `fetch_history` | `{**symbol**, **timeframe**, **from_ms**: int, to_ms?: int}` | MUT (disk; downloads broker history into Vortex) | `POST /data/fetch` |
| 59 | `import_data` | `{**source_path**, **symbol**, **timeframe**}` | MUT (disk) | `POST /data/import` |

**Guarded set (exact):** `open_position`, `close_position`,
`modify_position_protection`, `place_pending_order`, `cancel_pending_order`,
`confirm_pending_action`, `autopilot_start`, `promote_strategies`,
`apply_risk_preset`. A unit test asserts this list is identical to the set of
tools annotated `destructive_hint=true ∧ read_only_hint=false` minus
`update_settings` (hint-only) — any drift fails CI (§6.1).

---

## 4. Codex-side configuration (operator pastes verbatim)

File: `~/.codex/config.toml` (user-global — the project-scoped
`.codex/config.toml` only applies to trusted projects and is easy to silently
lose; use the global file). Alternative setup: `codex mcp add neoethos --
C:/Users/konst/development/forex-ai/target/release/neoethos-mcp.exe`, then edit
the per-tool blocks in the file. Verify with `codex mcp list` and `/mcp` in the
Codex TUI.

```toml
# Auto-approve read-only tools, prompt for everything that writes.
default_tools_approval_mode = "writes"

[mcp_servers.neoethos]
command = "C:/Users/konst/development/forex-ai/target/release/neoethos-mcp.exe"
args = ["--base-url", "http://127.0.0.1:7423"]
# Fail loudly at Codex startup if the MCP server cannot initialize.
required = true
startup_timeout_sec = 20
# replay_backtest / fetch_history can exceed the 60s default. Discovery and
# training are job-pattern (start/poll/stop) and never block this long.
tool_timeout_sec = 600

# Trade-affecting tools always prompt, regardless of the default mode.
# (Defense in depth only — the server re-verifies the demo account on every
# one of these calls and refuses on anything but a verified demo account.)
[mcp_servers.neoethos.tools.open_position]
approval_mode = "approve"
[mcp_servers.neoethos.tools.close_position]
approval_mode = "approve"
[mcp_servers.neoethos.tools.modify_position_protection]
approval_mode = "approve"
[mcp_servers.neoethos.tools.place_pending_order]
approval_mode = "approve"
[mcp_servers.neoethos.tools.cancel_pending_order]
approval_mode = "approve"
[mcp_servers.neoethos.tools.confirm_pending_action]
approval_mode = "approve"
[mcp_servers.neoethos.tools.autopilot_start]
approval_mode = "approve"
[mcp_servers.neoethos.tools.promote_strategies]
approval_mode = "approve"
[mcp_servers.neoethos.tools.apply_risk_preset]
approval_mode = "approve"
```

Known limitation (do not work around): non-interactive `codex exec` cannot
approve MCP tool calls at all (openai/codex #24135). Use interactive Codex
sessions for this system. The bypass flag that "fixes" it also disables the
sandbox and MUST NOT be used or documented as a workaround.

---

## 5. Implementation notes

- One `Backend` struct holding the `reqwest::Client` and base URL; all tools are
  `#[tool]` methods on it under `#[tool_router]`/`#[tool_handler]`, with typed
  parameter structs deriving `schemars::JsonSchema` and annotations set in the
  macro (`#[tool(annotations(read_only_hint = true, ...))]`).
- Every handler returns structured JSON (`Json<T>` / raw backend DTO) on
  success; every failure path is `isError:true` with the backend's error body
  passed through verbatim plus the HTTP status — no unwrap, no panic, no silent
  fallback anywhere (repo standing rule).
- The `DemoProof` choke point (§2) lives next to the HTTP helper module;
  trade-affecting routes are listed once, in that module, and `trade_post` is
  the only function referencing them.
- Logging: `tracing_subscriber::fmt().with_writer(std::io::stderr)` +
  `EnvFilter`; installed first thing in `main`.

---

## 6. Test plan

### 6.1 Without a running backend (cargo, runs in CI)

1. **Schema/catalog snapshot test** — instantiate the tool router, dump
   `tools/list` to JSON, snapshot-assert: every tool present, every schema has
   descriptions, annotations match the class table in §3 exactly.
2. **Guard-coverage test** — assert the guarded set (§3) equals the set of
   `destructive_hint=true` trade tools; a new trade tool without a guard fails
   the test even before the `DemoProof` compile error (belt and braces).
3. **Demo-guard behavior against a mock backend** (axum test server on an
   ephemeral port): environment=Live → refused with the exact §2 message;
   environment=Demo but `is_live=null` → refused (fail-closed);
   `connected=false` → refused; backend down (server dropped) → the §1.2
   unreachable message; environment=Demo + `is_live=false` → forwarded, and the
   mock asserts `risky:false` and the lots→wire-unit conversion on
   `close_position`.
4. **stdout hygiene test** — spawn the real binary, run
   `initialize` → `tools/list` over stdio, assert every stdout line parses as
   JSON-RPC (catches any stray print/log — the classic Windows MCP corruption
   failure).

### 6.2 With the app running (real data, per standing verification rule)

`scripts/mcp-smoke.ps1` — requires the desktop app or `neoethos-app --server`
up with the demo cTrader account connected; drives the binary over stdio:

1. `initialize`, `tools/list` (assert 59 tools).
2. `system_health`, `broker_status` — **assert environment=Demo, connected**;
   abort the script loudly otherwise.
3. `account_snapshot`, `journal_stats`, `journal_analytics`, `get_chart`
   EURUSD M5 — read path sanity.
4. `open_position` EURUSD buy 0.01 with SL 20 / TP 20 → `list_positions` shows
   it → `modify_position_protection` (widen SL) → `close_position` (full) →
   `journal_trades` shows the round trip.
5. `place_pending_order` far-from-market limit → `list_pending_orders` →
   `cancel_pending_order`.
6. `discovery_start` (population 64, generations 1) → poll `engine_status`
   until running → `discovery_stop`.
7. `replay_backtest` on the default symbol (proves the >60s path within the
   600s budget).

The Live-refusal path is **never** tested against the real system (that would
require flipping the real environment to Live); it is covered exhaustively by
the mock in 6.1.3.

### 6.3 Codex end-to-end (manual, once)

`codex mcp add neoethos -- <exe path>` → `codex mcp list` → interactive TUI
`/mcp` shows the server → ask Codex "what is the broker status?" (auto-runs,
read-only) → ask it to open a 0.01 demo position (approval prompt appears →
approve → position opens → guard messages visible in the transcript).

---

## 7. Build and rollout order

1. Scaffold `crates/neoethos-mcp` + workspace membership; `Backend`, HTTP
   helper, `DemoProof`, error mapping.
2. Read-only tool group + tests 6.1.1/6.1.4 — ship value immediately
   (journal analytics, status, charts).
3. Demo guard + mock tests 6.1.2/6.1.3.
4. Trade tools + confirm-queue + autopilot.
5. Discovery/training/promotion/settings/data groups.
6. `scripts/mcp-smoke.ps1` against the real app on the demo account (6.2).
7. Operator pastes §4 config; 6.3 walkthrough; done.
