//! Step 4 — cTrader broker sign-in (operator OAuth → account picker).
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 4 + §9.2 mockup, with
//! the **2026-05-17 operator-directive correction**:
//!
//! > "Το wizard installer ζητάει user για ctrader api id ενώ αυτό
//! > είναι developer credentials" / "the wizard installer asks the
//! > user for the cTrader API id when those are developer credentials,
//! > not user credentials."
//!
//! cTrader Open API uses a *registered-application* OAuth model: the
//! `client_id` + `client_secret` identify the **bot binary** (one app
//! registered once on connect.spotware.com by the developer who built
//! the binary), and the OAuth flow lets each end-user authorise that
//! app against their own broker account. Asking the end-user to type
//! the app credentials was a misread of the spec. This module now
//! reads the embedded developer credentials baked into the binary at
//! build time by `crates/neoethos-app/build.rs` /
//! [`crate::app_services::embedded_credentials`] and only asks the
//! end-user to:
//!
//! 1. Pick environment (Demo vs Live).
//! 2. Click "Sign in to your broker" → OAuth.
//! 3. Pick which `ctidTraderAccountId` to use.
//!
//! The actual OAuth flow is driven by
//! [`crate::app_services::ctrader_live_auth::ProductionCTraderLiveAuthBackend`].
//! This step is responsible for:
//!
//! 1. Resolving the embedded `client_id` + `client_secret`. If the
//!    binary was built without them (developer building from source
//!    without setting `NEOETHOS_EMBED_CTRADER_CLIENT_ID` /
//!    `_CLIENT_SECRET`), the step renders an explanatory diagnostic
//!    banner with the env-var names — there's no operator-facing
//!    text field for this.
//! 2. Spawning the background thread that runs the loopback listener,
//!    opens the system browser, captures the callback with CSRF-state
//!    validation, and exchanges the authorization code for a token
//!    bundle.
//! 3. Issuing `ProtoOAGetAccountListByAccessTokenReq` (payload 2149)
//!    against the picked environment and rendering the returned
//!    accounts as a dropdown.
//! 4. Recording the picked `ctidTraderAccountId` on `WizardConfig`.
//!    Spec §11 acceptance criterion 4: "OAuth tokens are persisted
//!    only after the flow completes — no half-written
//!    `broker_credentials.toml`."

use eframe::egui;
use secrecy::SecretString;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Mutex, OnceLock};

use super::state::WizardError;
use super::{CTraderEnvironment, StepResult, WizardController};
use crate::app_services::ctrader_auth::CTraderDiscoveredAccount;
use crate::app_services::ctrader_live_auth::{
    CTRADER_DEFAULT_SCOPE, CTraderAccountDiscoveryRequest, CTraderAccountDiscoveryResult,
    CTraderEnvironment as AuthCTraderEnvironment, CTraderLiveAuthBackend, CTraderLiveAuthRequest,
    CTraderLiveAuthResult, CTraderLoopbackConfig, ProductionCTraderLiveAuthBackend,
    discover_ctrader_accounts,
};
use crate::app_services::embedded_credentials::{
    EMBEDDED_CTRADER_CLIENT_ID, EMBEDDED_CTRADER_CLIENT_SECRET,
};
use crate::ui::theme;

/// Env-var name the developer sets at build time to bake the cTrader
/// app `client_id` into the binary. Spec mirror of
/// `crates/neoethos-app/build.rs::emit_embedded_credentials`. Surfaced
/// in the developer-setup banner when the embedded constant is empty.
pub const BUILD_ENV_CLIENT_ID: &str = "NEOETHOS_EMBED_CTRADER_CLIENT_ID";

/// Env-var name the developer sets at build time to bake the cTrader
/// app `client_secret` into the binary.
pub const BUILD_ENV_CLIENT_SECRET: &str = "NEOETHOS_EMBED_CTRADER_CLIENT_SECRET";

/// Spec §2 Step 4.2 — loopback port allocator. RFC 8252 §7.3 fallback
/// list. Must match `CTraderLoopbackConfig` at
/// `app_services/ctrader_live_auth.rs:28`.
// v0.4.12 — was [7777, 7878, 8989] (RFC 8252 §7.3 fallback list). Wired
// walkthrough on 2026-05-19 confirmed cTrader rejected the OAuth request
// with "Application does not contain provided URI" because the
// developer-registered cTrader Open API app has only
// `http://127.0.0.1:43001/callback` in its allowed-redirect list. The
// wizard's fallback list now leads with 43001 to match what the app
// dashboard has registered; the legacy ports remain as alternates in
// case the operator re-registers them on a forked dev app.
pub const WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS: &[u16] = &[43001, 7777, 7878, 8989];

/// Spec §2 Step 4.2 — browser callback timeout (matches
/// `CTRADER_CALLBACK_TIMEOUT` at `ctrader_live_auth.rs:24`).
pub const WIZARD_DEFAULT_OAUTH_CALLBACK_TIMEOUT_SECONDS: u64 = 300;

/// Loopback callback path. Must match the redirect URI registered in
/// the operator's cTrader Open API app (spec §2 Step 4.1 mockup).
// v0.4.12 — was "/ctrader/callback". Same root cause as the port-list
// change above: the registered redirect URI in the cTrader Open API
// app is `http://127.0.0.1:43001/callback`, so the wizard's loopback
// listener must answer the bare `/callback` path or the OAuth exchange
// 404s out before reaching the token-exchange leg.
pub const WIZARD_DEFAULT_OAUTH_CALLBACK_PATH: &str = "/callback";

// Earlier builds exposed WIZARD_DEFAULT_OAUTH_CLIENT_ID_MIN_LEN /
// WIZARD_DEFAULT_OAUTH_CLIENT_SECRET_MIN_LEN as validation bounds for
// the operator-facing text fields in sub-step 4.1. Those fields were
// retired by the 2026-05-17 directive — the embedded credentials are
// the only source — so the constants are no longer surfaced. If a
// future build-time sanity check needs them, re-introduce as crate-
// local items in `build.rs` rather than this module.

/// Sub-step within the OAuth screen. The wizard re-renders the same
/// step until the user clicks "Continue" — the sub-step is internal.
///
/// The legacy `RegisterApp` sub-step (where the operator typed
/// `client_id`/`client_secret`) was retired by the 2026-05-17
/// directive; the binary now reads those values from the embedded
/// constants and the operator only sees the OAuth handoff, account
/// picker, and probe sub-steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthSubStep {
    /// Sign in with cTID — browser handoff in progress.
    SignIn,
    /// Account picker — token bundle obtained, fetching accounts.
    PickAccount,
    /// Per-account auth probe — account picked, ready to continue.
    AuthProbe,
}

/// Wizard-local sub-step state. Held in a process-global `OnceLock`
/// because egui re-renders this step on every frame and the running
/// background thread's `Receiver` cannot be re-created across frames.
///
/// Cleared on a fresh wizard run via [`reset_oauth_runtime`]. Tests
/// reach the inner state via [`force_runtime_state_for_tests`].
#[derive(Default)]
struct OAuthRuntime {
    sub_step: OAuthSubStep,
    /// Held-in-memory access token from the OAuth exchange.
    access_token: Option<SecretString>,
    /// Held-in-memory refresh token from the OAuth exchange.
    refresh_token: Option<SecretString>,
    /// v0.4.17 — the full token bundle stays in memory until Apply
    /// writes it to the secure store. The standalone `access_token`
    /// and `refresh_token` fields stay for backwards-compat with the
    /// older `expose_access_token` / `expose_refresh_token` getters
    /// that take secret-string copies; the bundle here is the
    /// authoritative source for `expose_token_bundle` which the Apply
    /// step calls to persist tokens across app restarts.
    token_bundle: Option<crate::app_services::ctrader_auth::CTraderTokenBundle>,
    /// Receiver for the spawned auth worker.
    auth_rx: Option<Receiver<Result<CTraderLiveAuthResult, String>>>,
    /// Receiver for the spawned account-discovery worker.
    accounts_rx: Option<Receiver<Result<CTraderAccountDiscoveryResult, String>>>,
    /// Most-recent account list (populated on success).
    accounts: Vec<CTraderDiscoveredAccount>,
    /// Last error surfaced verbatim per spec §3 rule 1.
    last_error: Option<String>,
}

impl Default for OAuthSubStep {
    fn default() -> Self {
        OAuthSubStep::SignIn
    }
}

fn runtime_mutex() -> &'static Mutex<OAuthRuntime> {
    static RUNTIME: OnceLock<Mutex<OAuthRuntime>> = OnceLock::new();
    RUNTIME.get_or_init(|| Mutex::new(OAuthRuntime::default()))
}

/// Read-only access to the developer-embedded cTrader app
/// `client_id` for the Apply step (Step 10) and the history step.
/// Returns `None` when the binary was built without the embedded
/// credentials (developer-build mode); callers must then either
/// abort or surface the developer-setup banner.
pub fn expose_client_id() -> Option<String> {
    let trimmed = EMBEDDED_CTRADER_CLIENT_ID.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Read-only access to the developer-embedded cTrader app
/// `client_secret`. Mirrors [`expose_client_id`]. The secret is held
/// as a `&'static str` from the compile-time include, which is the
/// best we can do for a value the linker bakes into the binary —
/// every consumer copies it into a [`SecretString`] at the boundary
/// before any logging or persistence.
pub fn expose_client_secret() -> Option<String> {
    let trimmed = EMBEDDED_CTRADER_CLIENT_SECRET.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Read-only access to the access token for the Apply step.
pub fn expose_access_token() -> Option<String> {
    let runtime = runtime_mutex().lock().ok()?;
    use secrecy::ExposeSecret;
    runtime
        .access_token
        .as_ref()
        .map(|s| s.expose_secret().to_string())
}

/// v0.4.17 — Read-only access to the full token bundle for the Apply
/// step. Returns `None` until the OAuth exchange in Step 4 succeeds.
/// The bundle carries every field the secure-store needs
/// (`access_token`, `refresh_token`, `token_type`, `expires_in`,
/// `scope`, `created_at_unix`) so the Apply writer can persist a
/// fully-formed `CTraderTokenBundle` directly to the keyring without
/// reconstructing the missing fields from defaults.
pub fn expose_token_bundle() -> Option<crate::app_services::ctrader_auth::CTraderTokenBundle> {
    let runtime = runtime_mutex().lock().ok()?;
    runtime.token_bundle.clone()
}

/// Read-only access to the refresh token for the Apply step.
#[allow(dead_code)] // restored 2026-05-21 — the wizard rerun + Apply
                    // path still needs this when the operator chooses
                    // to persist only the refresh token (e.g. for a
                    // CLI-driven later authorization). Pairs with
                    // `expose_access_token` above.
pub fn expose_refresh_token() -> Option<String> {
    let runtime = runtime_mutex().lock().ok()?;
    use secrecy::ExposeSecret;
    runtime
        .refresh_token
        .as_ref()
        .map(|s| s.expose_secret().to_string())
}

/// `true` iff the binary was built with non-empty embedded cTrader
/// app credentials. The wizard's Step 4 renders the OAuth flow when
/// this is `true` and the developer-setup banner when it isn't.
pub fn embedded_credentials_present() -> bool {
    expose_client_id().is_some() && expose_client_secret().is_some()
}

/// v0.4.16 — build a stable picker label for one discovered account.
/// Format: `<traderLogin> · <broker> (Live|Demo) · ctid #<ctidTraderAccountId>`.
///
/// v0.5.1.1 — the operator should be able to match the picker entry
/// 1-to-1 with what cTrader's consent page shows. The consent page
/// uses `<broker> · <Type> · <traderLogin>` as its line shape (e.g.
/// "FTMO Platform • Live • 17111418"), so we lead with the
/// `traderLogin` to keep visual parity. The `ctidTraderAccountId` is
/// kept at the tail (after a `·`) so the wire-protocol identifier is
/// still readable for debugging without taking over the line.
fn account_picker_label(
    account: &crate::app_services::ctrader_auth::CTraderDiscoveredAccount,
) -> String {
    let trader_login_or_id = account
        .trader_login
        .map(|n| n.to_string())
        .unwrap_or_else(|| account.account_id.clone());
    let broker = if account.broker_title.trim().is_empty() {
        "(broker unknown)".to_string()
    } else {
        account.broker_title.clone()
    };
    let env_label = match account.is_live {
        Some(true) => "Live",
        Some(false) => "Demo",
        None => "Unknown",
    };
    format!(
        "{} · {} ({}) · ctid #{}",
        trader_login_or_id, broker, env_label, account.account_id,
    )
}

/// Clear the process-global runtime — call when starting a fresh
/// wizard run (e.g. from `Settings → Wizard`).
#[allow(dead_code)] // restored 2026-05-21 — needed for the planned
                    // "Settings → Re-run wizard" entry point and by
                    // the integration test that asserts the runtime
                    // resets cleanly between wizard launches.
pub fn reset_oauth_runtime() {
    if let Ok(mut runtime) = runtime_mutex().lock() {
        *runtime = OAuthRuntime::default();
    }
}

/// Translate the wizard's `CTraderEnvironment` to the auth-module
/// equivalent. The wizard enum is part of `mod.rs` so the persisted
/// `WizardConfig` schema does not depend on the app_services crate
/// path; we map at the IO boundary.
fn map_environment(env: CTraderEnvironment) -> AuthCTraderEnvironment {
    match env {
        CTraderEnvironment::Live => AuthCTraderEnvironment::Live,
        CTraderEnvironment::Demo => AuthCTraderEnvironment::Demo,
    }
}

/// Spawn the OAuth-flow background thread. The wizard thread retains
/// the `Receiver`; the worker drives the production backend and sends
/// the result back. Returns the configured `Receiver` so the caller
/// stores it on the runtime.
fn spawn_oauth_worker(
    backend: ProductionCTraderLiveAuthBackend,
    request: CTraderLiveAuthRequest,
) -> Receiver<Result<CTraderLiveAuthResult, String>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("wizard-oauth-worker".to_string())
        .spawn(move || {
            let result = backend.run(request).map_err(|err| err.to_string());
            let _ = tx.send(result);
        })
        .expect("spawn wizard-oauth-worker");
    rx
}

/// Spawn the account-discovery background thread. Issued only after
/// the OAuth flow returns a token bundle.
fn spawn_account_discovery_worker(
    request: CTraderAccountDiscoveryRequest,
) -> Receiver<Result<CTraderAccountDiscoveryResult, String>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("wizard-oauth-accounts-worker".to_string())
        .spawn(move || {
            let result = discover_ctrader_accounts(&request).map_err(|err| err.to_string());
            let _ = tx.send(result);
        })
        .expect("spawn wizard-oauth-accounts-worker");
    rx
}

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;
    // Note — recover from a poisoned mutex instead of
    // panicking the render thread. A panic inside an OAuth background
    // worker (e.g. inside `poll_auth_worker` chain) used to poison this
    // global mutex, and the next wizard frame would `.expect()` and bring
    // down the entire UI. We now log and continue with the inner guard so
    // the operator can at least see the error banner + retry.
    let mut runtime = match runtime_mutex().lock() {
        Ok(g) => g,
        Err(poisoned) => {
            tracing::error!(
                target: "neoethos_app::wizard::oauth",
                "OAuth runtime mutex was poisoned (a background worker panicked while \
                 holding it); recovering inner guard so the render thread can continue"
            );
            poisoned.into_inner()
        }
    };

    // Poll background workers. Both pollers replace the receiver with
    // `None` after success/failure so the next frame doesn't re-fire.
    poll_auth_worker(&mut runtime, controller);
    poll_account_discovery_worker(&mut runtime, controller);

    // Live/Demo mode banner — competitive analysis §1.1 (TradingView
    // colour-codes live vs paper at login). The wizard tints the
    // surrounding label accordingly.
    let env_color = match controller.config.ctrader_environment {
        CTraderEnvironment::Live => theme::DANGER,
        CTraderEnvironment::Demo => theme::TEXT_MUTED,
    };
    let env_label = match controller.config.ctrader_environment {
        CTraderEnvironment::Live => "LIVE",
        CTraderEnvironment::Demo => "DEMO",
    };
    // Task #67 — was rendering THREE visual elements that read as a
    // triple button to operators ("Demo | Live | DEMO"): two
    // selectable labels for the choice and a third coloured status
    // badge that mirrored the selection. The badge was confusing
    // (looked like a third clickable option) and provided no extra
    // signal. The safety colour now lives on the selected button
    // itself via the `_` prefix on the inactive label below.
    ui.horizontal(|ui| {
        ui.label("cTrader environment:");
        let demo_selected = controller.config.ctrader_environment == CTraderEnvironment::Demo;
        let live_selected = controller.config.ctrader_environment == CTraderEnvironment::Live;
        if ui.selectable_label(demo_selected, "Demo").clicked() {
            controller.config.ctrader_environment = CTraderEnvironment::Demo;
        }
        if ui.selectable_label(live_selected, "Live").clicked() {
            controller.config.ctrader_environment = CTraderEnvironment::Live;
        }
        // Single inline safety indicator (replaces the old triple-
        // badge layout). Reads "→ DEMO" in muted text for paper, or
        // "→ LIVE" in danger red for real money. Operators perceive
        // this as a status read-out, not a third button.
        ui.label(
            egui::RichText::new(format!("→ {env_label}"))
                .color(env_color)
                .strong(),
        );
    });

    ui.separator();
    let embedded_present = embedded_credentials_present();
    if !embedded_present {
        // Developer-build diagnostic. End users on a release binary
        // never see this — the release build pipeline always sets the
        // embed env vars / writes the workspace `.local` TOML. If
        // they DO see it, the binary was built from source without
        // the dev's app credentials and the OAuth flow cannot proceed.
        ui.label(
            egui::RichText::new("Developer build: cTrader app credentials not embedded")
                .strong()
                .color(theme::DANGER),
        );
        ui.label(
            egui::RichText::new(format!(
                "This binary was built without the cTrader Open API \
                 app credentials baked in. Re-build with the two \
                 environment variables set:\n\
                 \u{00A0}\u{00A0}{}=<your app client_id>\n\
                 \u{00A0}\u{00A0}{}=<your app client_secret>\n\
                 or place a TOML at .local/neoethos/broker_credentials.toml \
                 with [ctrader] client_id / client_secret keys, then \
                 run cargo build again. End users never see this banner \
                 — they receive a binary that already has them baked in.",
                BUILD_ENV_CLIENT_ID, BUILD_ENV_CLIENT_SECRET,
            ))
            .color(theme::TEXT_MUTED)
            .size(theme::FONT_CAPTION),
        );
    } else {
        ui.label(
            egui::RichText::new("Sign in to your broker")
                .strong()
                .color(theme::TEXT_PRIMARY),
        );
        ui.label(
            egui::RichText::new(format!(
                "The wizard will bind a loopback listener on the first \
                 free port of {:?} and open your system browser at the \
                 cTrader authorize page. You'll sign in with your \
                 broker cTID — the bot's app credentials are baked in \
                 to the binary, so there is nothing else for you to \
                 type here. Callback timeout: {} s.",
                WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS, WIZARD_DEFAULT_OAUTH_CALLBACK_TIMEOUT_SECONDS
            ))
            .color(theme::TEXT_MUTED)
            .size(theme::FONT_CAPTION),
        );
    }

    let auth_in_flight = runtime.auth_rx.is_some();
    let accounts_in_flight = runtime.accounts_rx.is_some();
    let can_start_oauth = embedded_present && !auth_in_flight && !accounts_in_flight;
    ui.horizontal(|ui| {
        let label = if auth_in_flight {
            "Waiting for browser callback…"
        } else {
            "Sign in to your broker"
        };
        let response = ui.add_enabled(can_start_oauth, egui::Button::new(label));
        if response.clicked() && can_start_oauth {
            start_oauth_flow(&mut runtime, controller);
        }
    });

    if runtime.access_token.is_some() {
        ui.label(
            egui::RichText::new(
                "Token bundle received — held in memory as SecretString until Apply.",
            )
            .color(theme::TEXT_PRIMARY)
            .size(theme::FONT_CAPTION),
        );
    }

    // ─── 4.3 / 4.4 Account picker + auth probe ─────────────────────
    ui.separator();
    ui.label(
        egui::RichText::new("4.3 / 4.4 Account picker + auth probe")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );
    if accounts_in_flight {
        ui.label(
            egui::RichText::new("Fetching account list from broker…")
                .color(theme::TEXT_MUTED)
                .size(theme::FONT_CAPTION),
        );
    } else if !runtime.accounts.is_empty() {
        // v0.5.1.1 — show how many accounts came back so the operator
        // can compare with cTrader's consent page at a glance. If a
        // newly-activated account (e.g. an FTMO Free Trial moved from
        // "Ready" to "Active") is missing, the fix is a fresh OAuth
        // round — the broker only includes activated accounts in
        // `ProtoOAGetAccountListByAccessTokenRes`.
        let live_count = runtime
            .accounts
            .iter()
            .filter(|a| a.is_live == Some(true))
            .count();
        let demo_count = runtime
            .accounts
            .iter()
            .filter(|a| a.is_live == Some(false))
            .count();
        ui.label(
            egui::RichText::new(format!(
                "{} accounts returned by broker · {} Live · {} Demo. \
                 Missing one? Re-run \"Sign in\" after activating it on \
                 the broker portal.",
                runtime.accounts.len(),
                live_count,
                demo_count,
            ))
            .color(theme::TEXT_MUTED)
            .size(theme::FONT_CAPTION),
        );
        // v0.4.16 — the dropdown's `selected_text` previously rendered
        // just the bare `ctidTraderAccountId` (e.g. "47149192"), which
        // looked like a raw integer floating in the UI. Surface the
        // same "#id broker_title (demo|live)" formatting the dropdown
        // options use so the operator sees something legible even
        // after selecting an account.
        let current = controller
            .config
            .selected_ctid_trader_account_id
            .and_then(|id| {
                runtime
                    .accounts
                    .iter()
                    .find(|a| a.account_id == id.to_string())
                    .map(|a| account_picker_label(a))
            })
            .or_else(|| {
                controller
                    .config
                    .selected_ctid_trader_account_id
                    .map(|id| format!("#{id}"))
            })
            .unwrap_or_else(|| "(none)".to_string());
        egui::ComboBox::from_id_salt("wizard_ctrader_account_picker")
            .selected_text(current.clone())
            .show_ui(ui, |ui| {
                for account in &runtime.accounts {
                    let label = account_picker_label(account);
                    let parsed = account.account_id.parse::<u64>().ok();
                    let mut current_id = controller.config.selected_ctid_trader_account_id;
                    if ui
                        .selectable_value(&mut current_id, parsed, label)
                        .clicked()
                    {
                        controller.config.selected_ctid_trader_account_id = current_id;
                        // v0.5.1 — every cTrader account is routed by
                        // the broker pool to either `demo.ctraderapi.com`
                        // or `live.ctraderapi.com`. Sending app/account
                        // auth for a Live account to the demo endpoint
                        // (or vice versa) gets rejected with
                        // `CANT_ROUTE_REQUEST: Cannot route request`.
                        // Auto-sync the wizard's `ctrader_environment`
                        // to the picked account's `is_live` flag so the
                        // workspace, the historical-bars bootstrap, and
                        // every later session call land on the right
                        // endpoint without the operator having to flip
                        // a toggle they didn't know existed.
                        match account.is_live {
                            Some(true) => {
                                controller.config.ctrader_environment =
                                    super::CTraderEnvironment::Live;
                            }
                            Some(false) => {
                                controller.config.ctrader_environment =
                                    super::CTraderEnvironment::Demo;
                            }
                            None => {}
                        }
                    }
                }
            });
    } else if let Some(account_id) = controller.config.selected_ctid_trader_account_id {
        ui.label(format!(
            "Primary account: #{} ({})",
            account_id,
            controller.config.ctrader_environment.as_str()
        ));
    } else {
        ui.label(
            egui::RichText::new("No account picked yet. Complete 4.2 to populate this list.")
                .color(theme::TEXT_MUTED)
                .size(theme::FONT_CAPTION),
        );
    }

    if let Some(err) = runtime.last_error.as_ref() {
        ui.separator();
        ui.label(
            egui::RichText::new(format!("OAuth error: {}", err))
                .color(theme::DANGER)
                .size(theme::FONT_CAPTION),
        );
    }

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("← Back").clicked() {
            result = StepResult::BackRequested;
        }
        if ui.button("Skip cTrader").clicked() {
            result = StepResult::SkipRequested;
        }
        if ui.button("Continue →").clicked() {
            result = StepResult::NextRequested;
        }
    });

    // egui re-renders this step on a timer — request a repaint while
    // background workers are running so the user sees status changes
    // without having to nudge the mouse.
    if auth_in_flight || accounts_in_flight {
        ui.ctx().request_repaint();
    }

    result
}

/// Issue the OAuth `run()` call on a worker thread. The wizard thread
/// keeps the rx half on the runtime; `poll_auth_worker` consumes it.
///
/// Pulls the developer-embedded `client_id` + `client_secret` from
/// the compile-time constants (`crates/neoethos-app/build.rs` writes
/// these via `emit_embedded_credentials`). The end user never sees
/// or supplies these values — that was the 2026-05-17 directive fix.
fn start_oauth_flow(runtime: &mut OAuthRuntime, controller: &mut WizardController) {
    let _ = controller; // controller is unused in this function now;
    // every legacy ctrader_client_id read was retired.
    runtime.last_error = None;
    runtime.sub_step = OAuthSubStep::SignIn;
    let client_id = match expose_client_id() {
        Some(id) => id,
        None => {
            runtime.last_error = Some(format!(
                "Binary was built without the embedded cTrader app client_id \
                 (build-time env {}). Cannot start OAuth.",
                BUILD_ENV_CLIENT_ID
            ));
            return;
        }
    };
    let client_secret = match expose_client_secret() {
        Some(secret) => secret,
        None => {
            runtime.last_error = Some(format!(
                "Binary was built without the embedded cTrader app client_secret \
                 (build-time env {}). Cannot start OAuth.",
                BUILD_ENV_CLIENT_SECRET
            ));
            return;
        }
    };
    // Pick the first port out of the spec'd port list. The
    // production backend tries each in order via
    // `bind_loopback_listener`.
    let primary_port = *WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS
        .first()
        .expect("WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS is non-empty");
    let fallback_ports: Vec<u16> = WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS
        .iter()
        .skip(1)
        .copied()
        .collect();
    let loopback = CTraderLoopbackConfig::new(
        primary_port,
        fallback_ports,
        WIZARD_DEFAULT_OAUTH_CALLBACK_PATH,
    );
    let redirect_uri = format!(
        "http://127.0.0.1:{}{}",
        primary_port, WIZARD_DEFAULT_OAUTH_CALLBACK_PATH
    );
    let request = CTraderLiveAuthRequest {
        client_id,
        client_secret,
        redirect_uri,
        scope: CTRADER_DEFAULT_SCOPE.to_string(),
        loopback,
    };
    let rx = spawn_oauth_worker(ProductionCTraderLiveAuthBackend, request);
    runtime.auth_rx = Some(rx);
    tracing::info!(
        target: "neoethos_app::wizard::oauth",
        "wizard OAuth flow spawned"
    );
}

/// Poll the OAuth-flow worker. On success, store the token bundle and
/// kick off the account-discovery worker. On failure, surface verbatim.
fn poll_auth_worker(runtime: &mut OAuthRuntime, controller: &mut WizardController) {
    let outcome = {
        let Some(rx) = runtime.auth_rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(outcome) => outcome,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                runtime.auth_rx = None;
                runtime.last_error =
                    Some("OAuth worker thread disconnected before returning a result".to_string());
                controller.last_error = Some(WizardError::OAuthTokenExchange(
                    "worker disconnected".to_string(),
                ));
                return;
            }
        }
    };
    runtime.auth_rx = None;

    match outcome {
        Ok(result) => {
            tracing::info!(
                target: "neoethos_app::wizard::oauth",
                callback_port = result.callback_port,
                "wizard OAuth flow returned token bundle"
            );
            runtime.access_token =
                Some(SecretString::from(result.token_bundle.access_token.clone()));
            runtime.refresh_token = Some(SecretString::from(
                result.token_bundle.refresh_token.clone(),
            ));
            // v0.4.17 — stash the full bundle for the Apply-step
            // persist call. Cloning the inner String here is fine —
            // the bundle is small, and copying it lets the discovery
            // request below take ownership of access_token without
            // moving the value out of the runtime.
            runtime.token_bundle = Some(result.token_bundle.clone());
            runtime.sub_step = OAuthSubStep::PickAccount;
            // Kick off account discovery immediately. The wizard's
            // chosen Demo/Live radio decides the endpoint host. The
            // app credentials come from the embedded constants — the
            // operator never sees them.
            let Some(client_id) = expose_client_id() else {
                runtime.last_error = Some(
                    "Embedded cTrader client_id missing after OAuth — \
                     binary appears to have been rebuilt without it"
                        .to_string(),
                );
                return;
            };
            let Some(client_secret) = expose_client_secret() else {
                runtime.last_error = Some(
                    "Embedded cTrader client_secret missing after OAuth — \
                     binary appears to have been rebuilt without it"
                        .to_string(),
                );
                return;
            };
            let request = CTraderAccountDiscoveryRequest {
                client_id,
                client_secret,
                access_token: result.token_bundle.access_token,
                environment: map_environment(controller.config.ctrader_environment),
            };
            runtime.accounts_rx = Some(spawn_account_discovery_worker(request));
        }
        Err(err) => {
            tracing::error!(
                target: "neoethos_app::wizard::oauth",
                error = %err,
                "wizard OAuth flow failed"
            );
            // Categorise on substring — the production backend's
            // anyhow chains include the step labels (e.g.
            // "step 4/5 (wait_for_callback)" / "callback `state` did
            // not match").
            controller.last_error = Some(classify_oauth_failure(&err));
            runtime.last_error = Some(err);
        }
    }
}

/// Map an `anyhow` string into a typed `WizardError` so the spec §3
/// error matrix can surface the right banner copy.
pub fn classify_oauth_failure(err: &str) -> WizardError {
    let lower = err.to_ascii_lowercase();
    if lower.contains("bind") && lower.contains("port") {
        WizardError::OAuthLoopbackBindFailed {
            tried_ports: WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS.to_vec(),
        }
    } else if lower.contains("timed out") && lower.contains("callback") {
        WizardError::OAuthCallbackTimeout
    } else if lower.contains("state") && (lower.contains("mismatch") || lower.contains("csrf")) {
        WizardError::OAuthCsrfMismatch
    } else if lower.contains("token") {
        WizardError::OAuthTokenExchange(err.to_string())
    } else {
        WizardError::Other(err.to_string())
    }
}

/// Poll the account-discovery worker. On success, populate the
/// account picker; if there's exactly one account, auto-select it
/// (spec §2 Step 4.3 mockup).
fn poll_account_discovery_worker(runtime: &mut OAuthRuntime, controller: &mut WizardController) {
    let outcome = {
        let Some(rx) = runtime.accounts_rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(outcome) => outcome,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                runtime.accounts_rx = None;
                runtime.last_error = Some(
                    "Account-discovery worker disconnected before returning a result".to_string(),
                );
                return;
            }
        }
    };
    runtime.accounts_rx = None;

    match outcome {
        Ok(result) => {
            tracing::info!(
                target: "neoethos_app::wizard::oauth",
                account_count = result.accounts.len(),
                "wizard account discovery returned account list"
            );
            // The discovery call may rotate the access token — keep
            // the in-memory copy in sync so Apply persists the right
            // value (Step 10).
            runtime.access_token = Some(SecretString::from(result.access_token));
            // If exactly one account was returned and the operator
            // hasn't already picked, auto-select it. Spec §2 Step 4.3
            // "auto-select if only one account".
            if result.accounts.len() == 1
                && controller.config.selected_ctid_trader_account_id.is_none()
            {
                if let Ok(id) = result.accounts[0].account_id.parse::<u64>() {
                    controller.config.selected_ctid_trader_account_id = Some(id);
                    // v0.5.1 — auto-select also has to sync the
                    // routing environment to the account's
                    // `is_live` flag; otherwise the demo default
                    // sends a Live account into the wrong endpoint
                    // (CANT_ROUTE_REQUEST). Mirrors the picker
                    // dropdown handler above.
                    match result.accounts[0].is_live {
                        Some(true) => {
                            controller.config.ctrader_environment = super::CTraderEnvironment::Live;
                        }
                        Some(false) => {
                            controller.config.ctrader_environment = super::CTraderEnvironment::Demo;
                        }
                        None => {}
                    }
                    runtime.sub_step = OAuthSubStep::AuthProbe;
                }
            }
            runtime.accounts = result.accounts;
            if runtime.accounts.is_empty() {
                // The cTrader Open API discovery endpoint succeeded but
                // returned zero accounts. Common, distinct causes:
                //
                // 1. **Wrong environment** — the user picked Demo above but
                //    the account is Live (or vice versa). The endpoint is
                //    environment-scoped, so a Live account is invisible to
                //    the Demo discovery URL even with a valid token.
                // 2. **Open API not enabled on the broker side** — the
                //    operator must visit ctrader.com → Account → API and
                //    flip the "Open API" toggle. Without it, ANY discovery
                //    call returns zero accounts.
                // 3. **Token scope missing `accounts`** — the bot always
                //    requests the right scope, but a token rotated by hand
                //    or pulled from another tool may lack it.
                // 4. **No account exists at all** — fresh cTID with no
                //    demo provisioned yet.
                //
                // We surface all four because the user has no way to
                // distinguish them from the symptom alone.
                let other_env = match controller.config.ctrader_environment {
                    super::CTraderEnvironment::Demo => "Live",
                    super::CTraderEnvironment::Live => "Demo",
                };
                runtime.last_error = Some(format!(
                    "Sign-in succeeded but the broker returned 0 trading accounts. \
                     Try one of:\n\
                     • Switch the environment toggle above to {} and click Sign in again \
                     (this is the most common cause — accounts are environment-scoped).\n\
                     • At ctrader.com → Account → API, confirm that 'Open API' access is \
                     enabled for the account you signed in with.\n\
                     • If you do not yet have a demo account, open one for free at \
                     https://ctrader.com and retry."
                    , other_env
                ));
            }
        }
        Err(err) => {
            tracing::error!(
                target: "neoethos_app::wizard::oauth",
                error = %err,
                "wizard account discovery failed"
            );
            runtime.last_error = Some(err);
        }
    }
}

#[cfg(test)]
pub(crate) fn force_runtime_state_for_tests(
    sub_step: OAuthSubStep,
    accounts: Vec<CTraderDiscoveredAccount>,
) {
    if let Ok(mut runtime) = runtime_mutex().lock() {
        runtime.sub_step = sub_step;
        runtime.accounts = accounts;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::ctrader_live_auth::parse_callback_request_with_state;
    use crate::ui::wizard::{StepResult, WizardController, WizardState};

    #[test]
    fn default_loopback_ports_lead_with_registered_redirect_port() {
        // v0.4.12 — the cTrader Open API app dashboard has only port
        // 43001 in its allowed-redirect list, so the primary port has
        // to be 43001. The legacy [7777, 7878, 8989] from the
        // RFC 8252 §7.3 three-port fallback list stays as the tail so
        // a fork that re-registers them still works.
        assert!(!WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS.is_empty());
        assert_eq!(WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS[0], 43001);
    }

    #[test]
    fn callback_timeout_matches_spec_300_seconds() {
        assert_eq!(WIZARD_DEFAULT_OAUTH_CALLBACK_TIMEOUT_SECONDS, 300);
    }

    #[test]
    fn oauth_step_advances_to_symbols_on_next() {
        let mut c = WizardController::new();
        c.current = WizardState::OAuth;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::Symbols);
    }

    #[test]
    fn oauth_step_skip_records_under_ctrader_oauth_key() {
        let mut c = WizardController::new();
        c.current = WizardState::OAuth;
        c.apply(StepResult::SkipRequested);
        assert!(c.state_file.skipped_steps.contains(&WizardState::OAuth));
        assert_eq!(c.current, WizardState::Symbols);
    }

    #[test]
    fn oauth_back_returns_to_account_profile() {
        let mut c = WizardController::new();
        c.current = WizardState::OAuth;
        c.apply(StepResult::BackRequested);
        assert_eq!(c.current, WizardState::AccountProfile);
    }

    #[test]
    fn embedded_credentials_present_matches_constant_state() {
        // The function under test exists to drive the Step 4 UI
        // branch. With non-empty embedded constants it returns true;
        // with empty constants it returns false. We pin the
        // round-trip rather than the specific value because the
        // build pipeline decides what to embed and tests must work
        // in both modes (developer source-build and release CI).
        let expected = !EMBEDDED_CTRADER_CLIENT_ID.trim().is_empty()
            && !EMBEDDED_CTRADER_CLIENT_SECRET.trim().is_empty();
        assert_eq!(embedded_credentials_present(), expected);
    }

    #[test]
    fn expose_client_id_returns_none_iff_embedded_is_empty() {
        let trimmed = EMBEDDED_CTRADER_CLIENT_ID.trim();
        if trimmed.is_empty() {
            assert_eq!(expose_client_id(), None);
        } else {
            assert_eq!(expose_client_id(), Some(trimmed.to_string()));
        }
    }

    #[test]
    fn expose_client_secret_returns_none_iff_embedded_is_empty() {
        let trimmed = EMBEDDED_CTRADER_CLIENT_SECRET.trim();
        if trimmed.is_empty() {
            assert_eq!(expose_client_secret(), None);
        } else {
            assert_eq!(expose_client_secret(), Some(trimmed.to_string()));
        }
    }

    /// Audit-fix F2: a callback whose `state` query parameter does not
    /// match the value issued to the authorize URL must be refused
    /// before any token exchange. This drives the same code path the
    /// wizard relies on at 4.2.
    #[test]
    fn oauth_state_csrf_rejects_mismatched_state() {
        let issued = "issued-state-token-abc123";
        let received = "attacker-state-token-xyz999";
        let target = format!("/ctrader/callback?code=AUTHCODE&state={}", received);
        let err = parse_callback_request_with_state(&target, "/ctrader/callback", issued)
            .expect_err("CSRF mismatch must error");
        let msg = err.to_string();
        assert!(
            msg.contains("state") && msg.contains("mismatch"),
            "expected mismatch error, got: {msg}"
        );
    }

    #[test]
    fn classify_oauth_failure_recognises_bind_error() {
        let err = "OAuth step 1/5 (bind_loopback) failed — could not bind any of the allowed callback ports [7777, 7878, 8989]";
        assert!(matches!(
            classify_oauth_failure(err),
            WizardError::OAuthLoopbackBindFailed { .. }
        ));
    }

    #[test]
    fn classify_oauth_failure_recognises_callback_timeout() {
        let err =
            "OAuth step 4/5 (wait_for_callback) failed — timed out waiting for cTrader callback";
        assert_eq!(
            classify_oauth_failure(err),
            WizardError::OAuthCallbackTimeout
        );
    }

    #[test]
    fn classify_oauth_failure_recognises_csrf_mismatch() {
        let err = "cTrader callback `state` mismatch — possible CSRF";
        assert_eq!(classify_oauth_failure(err), WizardError::OAuthCsrfMismatch);
    }

    #[test]
    fn map_environment_round_trips() {
        assert_eq!(
            map_environment(CTraderEnvironment::Live),
            AuthCTraderEnvironment::Live
        );
        assert_eq!(
            map_environment(CTraderEnvironment::Demo),
            AuthCTraderEnvironment::Demo
        );
    }

    #[test]
    fn expose_tokens_return_none_when_runtime_is_fresh() {
        // access/refresh tokens live in the in-memory runtime (they
        // come back from OAuth). client_secret is sourced from the
        // embedded constant and is covered by a separate test —
        // `expose_client_secret_returns_none_iff_embedded_is_empty`.
        reset_oauth_runtime();
        assert_eq!(expose_access_token(), None);
        assert_eq!(expose_refresh_token(), None);
    }

    /// Full OAuth flow against a captured cTrader fixture. Ignored —
    /// drives the real `ProductionCTraderLiveAuthBackend` over a live
    /// loopback socket, which is only feasible with manual browser
    /// interaction.
    #[test]
    #[ignore = "needs cTrader fixture"]
    fn oauth_flow_with_captured_callback_url() {
        // The intended fixture is a captured response from a real
        // cTrader callback URL of the shape
        // `http://127.0.0.1:7777/ctrader/callback?code=…&state=…`
        // plus the subsequent `/apps/token` JSON response. The fixture
        // must be re-captured per refresh-token rotation (see
        // `ctrader_api_full_reference.md` §2.5) — not committable.
    }
}
