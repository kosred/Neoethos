//! Live cTrader API test harness — `neoethos-app --api-test`.
//!
//! Runs a curated set of integration flows against the cTrader demo
//! environment using the OAuth token already saved by the wizard, and
//! produces a structured JSON report that captures, per flow:
//!
//!   - status (PASS / FAIL / SKIP)
//!   - wall-clock duration
//!   - request / response payload sizes
//!   - error message + classified error kind on FAIL
//!   - first 2 KB of the wire frame for forensic debug
//!
//! The output is consumed by the V0.4 audit Phase A so the fix list is
//! driven by EMPIRICAL evidence from this machine + this broker, not by
//! theoretical static analysis. Each flow is intentionally additive: a
//! later flow assumes the earlier flows passed (e.g. the order-modify
//! flow only runs after `orders.market_buy_001` succeeded and the
//! position id is known).
//!
//! ## Safety
//!
//! - Defaults to `--env demo`; switching to `live` requires an explicit
//!   `--api-test-i-really-mean-live` companion flag (TODO when live is
//!   needed; not implemented in this revision — demo only).
//! - Order size is hardcoded to `0.01` lot on EURUSD (~$1 of risk for a
//!   25-pip stop on a USD account).
//! - The runner emits a `cleanup.flatten_all` flow at the end of any
//!   run that touched orders / positions. The cleanup fires even on
//!   earlier failure so a partial run does not leave a position hanging.
//! - `cancel`-style flows are best-effort: a failure to cancel during
//!   cleanup is logged but does not flip a passing run to FAIL.

pub mod flows;
pub mod report;
pub mod runner;

pub use runner::{ApiTestConfig, ApiTestEnvironment, run_api_test_suite};
