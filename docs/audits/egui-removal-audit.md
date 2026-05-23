# egui Removal Quarantine Audit

Date: 2026-05-23

Scope: `crates/neoethos-app/src/ui/**` and `crates/neoethos-app/src/workspace/**`

Decision rule: Flutter remains server-driven. Nothing from egui is copied into Flutter. Any reusable behavior must already exist in `app_services`, `server`, `neoethos-core`, or `neoethos-data`, or it needs a separate test-first migration.

| legacy file or area | category | candidate behavior | existing source of truth | decision | tests needed |
| --- | --- | --- | --- | --- | --- |
| `src/workspace/**` | render-only | Dock layout, tabs, last focused panel | Flutter shell navigation | Delete | None |
| `src/ui/theme.rs`, `components.rs`, `dashboard.rs`, `ai_insights.rs` | render-only | egui widgets, colors, panel state | Flutter widgets and backend DTOs | Delete | None |
| `src/ui/wizard/oauth.rs` | broker-auth | OAuth launch, token exchange, account discovery | `app_services::reauth`, `app_services::ctrader_*`, `/broker/reauth`, `/broker/accounts` | Reject direct migration; server routes already own this | Existing server/app service tests only |
| `src/ui/wizard/summary.rs` | config-writer/risk-safety | config writer, broker credentials writer, risk ack ledger, risky mode state write | `server::{settings,risk,broker_control}`, `app_services::broker_credentials`, `app_services::risky_mode_persistence`, `neoethos_core::Settings` | Do not salvage in this pass; old wizard apply flow is high-risk and needs a separate tested service API | Separate migration tests before reuse |
| `src/ui/wizard/historical.rs` | data-history | historical download chunking, rate limiting, sentinels | `app_services::ctrader_bootstrap`, `server::data_control`, `neoethos-data` | Delete egui path; use `/data/fetch` and bootstrap services | Existing/future data service tests |
| `src/ui/wizard/autonomy_risk.rs` | risk-safety | risk quiz and acknowledgement hash | `neoethos_core::config::RiskConfig`, `app_services::risky_mode_persistence`, `/risk` | Reject direct migration; no Flutter/server contract in this step | Separate risk-ack service tests if restored |
| `src/ui/system/bootstrap.rs` | data-history | symbol/timeframe list parsing, bootstrap start | `/data/bootstrap`, `/data/fetch`, `app_services::trading::start_ctrader_bootstrap_batch` | Delete egui panel; server endpoint is source of truth | Route/service tests if endpoint changes |
| `src/ui/system/brokers.rs` | broker-auth | save credentials, discover accounts, open broker pages | `/broker/credentials`, `/broker/accounts`, `/broker/reauth`, broker credential services | Delete egui panel; keep server API | Route/service tests if endpoint changes |
| `src/ui/trading/execution_panel.rs` | business-logic | order ticket state, auto lot display, cancel/close/place actions | `/orders`, `/orders/cancel`, `/positions/close`, `app_services::trading::{orders,risk_gate,snapshots}` | Delete egui panel; do not reuse auto-lot UI math without service test | Existing order/risk tests; add tests only for new server behavior |
| `src/ui/chrome/halt_button.rs` | risk-safety | manual halt close/cancel behavior | `app_services::trading::trip_manual_halt`, `app_services::broker_control` | Delete render code; service logic remains | Existing trading tests |
| `src/ui/system/intelligence.rs`, `src/ui/ai_helper.rs` | business-logic/render | Gemma status/chat/news UI | `/gemma/status`, `/gemma/chat`, `/gemma/news`, `/intelligence` | Delete egui UI; server owns intelligence surface | Existing/future route tests |

Conclusion: no code from `src/ui/**` is migrated into Flutter during this removal. The remaining backend/server behavior is already represented outside the egui tree, and unrepresented wizard-only flows are explicitly deferred until they have test-first service contracts.
