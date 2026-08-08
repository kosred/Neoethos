//! NeoEthos Codex control plane — an MCP *server* (stdio transport) that
//! wraps the localhost backend HTTP API so OpenAI Codex CLI (or any
//! MCP-speaking agent runtime) can operate the whole system: account state,
//! market data, journal analytics, discovery/training jobs, autopilot,
//! settings, and DEMO-ONLY trade execution.
//!
//! Design of record: `docs/codex-control-plane.md` (commit c6431311).
//!
//! Hard rules enforced here:
//! - **Demo-only, fail-closed**: every trade-affecting tool re-verifies the
//!   active broker account is a demo account on every call ([`backend`]).
//! - **Loud failure, no fallback**: a backend that is not running produces a
//!   fixed actionable error on every call; nothing retries or degrades.
//! - **stdout is protocol-only**: logging goes to stderr; this crate
//!   contains no `println!`.

pub mod backend;
pub mod ops;
pub mod params;
pub mod server;

pub use backend::{Backend, DemoProof, GUARDED_TOOLS, TRADE_ROUTES, ToolError};
pub use server::ControlPlane;
