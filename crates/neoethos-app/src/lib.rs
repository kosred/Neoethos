//! `neoethos-app` as a LIBRARY.
//!
//! The same modules the `neoethos-app` binary uses are exposed here so OTHER
//! crates (notably the Tauri desktop shell, `neoethos-desktop`) can call the
//! broker / cTrader / account / chart logic **in-process** — as ordinary Rust
//! function calls, with NO second process, NO HTTP server, NO port. This is the
//! mechanism that collapses the old Flutter "UI process + backend process over
//! HTTP" split into a single binary.
//!
//! The `bin` (`main.rs`) consumes these same modules via `neoethos_app::…`, so
//! there is exactly one copy of the code — the library — linked into whichever
//! front-end (the legacy headless server bin, or the Tauri app) needs it.

pub mod app_services;
pub mod app_state;
pub mod server;
