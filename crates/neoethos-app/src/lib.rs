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

/// Install EVERY runtime override from the operator's settings — the SINGLE
/// path both front-ends must call so a non-default config knob resolves to the
/// SAME engine state everywhere (audit S05).
///
/// The headless `main.rs` binary did all five of these inline; the in-process
/// Tauri desktop shell (`neoethos-desktop`) did NONE of them, so config.yaml
/// runtime knobs — search population/generation overrides, hardware CPU
/// budget, feature normalization + stale-HTF rebuild, tree-model threads, and
/// the app-server runtime — were silently IGNORED in the shipped desktop app
/// while working in the CLI/headless binary. Both entry points now call this.
pub fn install_runtime_overrides_from_settings(settings: &neoethos_core::Settings) {
    neoethos_search::install_search_runtime_overrides_from_settings(settings);
    neoethos_models::tree_models::config::install_tree_runtime_from_settings(settings);
    neoethos_core::system::install_hardware_runtime_overrides_from_settings(settings);
    neoethos_data::install_data_runtime_overrides(
        settings.models.data_runtime.normalize_features,
        settings.models.data_runtime.rebuild_stale_higher_tfs,
    );
    app_services::env_overrides::install_app_runtime_overrides(settings.app_runtime.clone());
}
