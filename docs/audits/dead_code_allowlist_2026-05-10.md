# `allow(dead_code)` active-source audit - 2026-05-10

Scope: active Rust source under `crates/`. Documentation references are excluded.

## Removed

- `crates/forex-data/src/core/session_features.rs`: removed the struct-level
  `#[allow(dead_code)]` on `SessionAccum`. Its helpers are used by the session
  feature implementation, so the suppression was no longer justified.

## Retained with rationale

- `crates/forex-app/src/app_services/ctrader_proto_messages.rs`:
  module-level generated cTrader protobuf surface. The OpenAPI schema contains
  messages that are valid protocol members even when the current app path does
  not construct every variant.
- `crates/forex-app/src/app_services/ctrader_session.rs`:
  cTrader session boundary surface. Retained until the broker runtime panel and
  execution workflows finish selecting the live subset.
- `crates/forex-app/src/app_services/ctrader_bootstrap.rs`:
  range bootstrap helpers retained for cTrader historical-data workflows and
  tests that exercise injected fetchers.
- `crates/forex-app/src/app_services/ctrader_data.rs` and
  `crates/forex-app/src/app_services/ctrader_execution.rs`: broker service
  endpoints kept behind the cTrader integration boundary.
- `crates/forex-app/src/app_services/broker_config.rs` and
  `crates/forex-app/src/app_services/trading.rs`: `cfg_attr(not(test),
  allow(dead_code))` is limited to test-only helper exposure and should be
  revisited with the app-service split, not removed blindly.
- `crates/forex-app/src/ui/theme.rs`: theme constants are part of the UI style
  contract even when not all are referenced by the current egui screens.
- `crates/forex-models/src/tree_models/{catboost,lightgbm,xgboost}.rs`:
  backend FFI handles are intentionally feature-gated. The `cfg_attr`
  suppressions apply when the native feature is disabled.
- `crates/forex-models/src/forecasting/swarm_impl.rs`: retained for now because
  the large forecasting/swarm module is already called out for a separate
  runtime-metadata and large-file pass. Removing those suppressions here risks
  conflating warning cleanup with behavior changes.

## Follow-up

The retained entries are now explicit debt instead of silent suppressions. The
next cleanup should target them only when the owning boundary is already being
changed: cTrader integration, UI theme contract, tree-model feature gates, or
forecasting/swarm runtime metadata.
