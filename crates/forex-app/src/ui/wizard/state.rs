//! Wizard state machine — pure data, no IO. Both the egui (`mod.rs`)
//! and the ratatui (`forex-cli`) front-ends drive this enum.
//!
//! References:
//! - `docs/audits/research/installer_wizard_ux_spec.md` §2 (10 steps),
//!   §5 (persisted state file schema), §11 (acceptance criteria).
//! - `docs/audits/research/wizard_onboarding_competitive_analysis.md`
//!   §9.2 (new Step 9.5 — Autonomy & Risk acknowledgement).
//!
//! All defaults are surfaced as `pub const` in their step file so a
//! reviewer can grep `WIZARD_DEFAULT_` and audit operator-policy
//! conformance in one pass.

use serde::{Deserialize, Serialize};

/// Persisted-file schema version. Bump on any breaking change to
/// `WizardStateFile`.
pub const WIZARD_STATE_FILE_VERSION: u32 = 1;

/// Filename inside `<config_dir>` for the persisted wizard state.
/// Matches `installer_wizard_ux_spec.md` §5 ("`wizard_state.json`").
pub const WIZARD_STATE_FILENAME: &str = "wizard_state.json";

/// Filename inside `<config_dir>` for the persisted in-progress state.
/// Spec §5.2 — separate from the completed sentinel so a half-finished
/// wizard does not look "complete".
pub const WIZARD_PROGRESS_FILENAME: &str = "wizard_progress.json";

/// 11 steps (10 numbered + 9.5 Autonomy & Risk Acknowledgement).
///
/// Spec §2 owns the 10 numbered steps; competitive analysis §9.2 owns
/// step 9.5. The order is load-bearing: `WizardController` advances
/// linearly through `WizardState::ordered()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WizardState {
    /// Step 1 — Welcome + License. NOT skippable (only mandatory step).
    Welcome,
    /// Step 2 — Path selection (data dir).
    Path,
    /// Step 3 — Account & profile (extended per competitive analysis
    /// §9.1: risk profile slider, SL toggle, beginner/advanced).
    AccountProfile,
    /// Step 4 — cTrader OAuth onboarding (4.1–4.4 sub-steps).
    OAuth,
    /// Step 5 — Symbol & timeframe defaults (+ template gallery per
    /// competitive analysis §8.4).
    Symbols,
    /// Step 6 — Historical-data download (rate-limited).
    Historical,
    /// Step 7 — Hardware compatibility probe.
    Hardware,
    /// Step 8 — News / sentiment provider + risk knobs (news window,
    /// maintenance window, correlation cap, volatility σ).
    NewsApi,
    /// Step 9 — Auto-start at login.
    Autostart,
    /// Step 9.5 — Autonomy & Risk Acknowledgement (competitive
    /// analysis §9.2). Mandatory iff Step 3 trading_mode = Live OR
    /// autonomous mode is enabled.
    AutonomyRisk,
    /// Step 10 — Summary & Apply (terminal).
    Summary,
}

impl WizardState {
    /// Canonical step order. The wizard advances strictly forward
    /// through this slice (with Back going one entry left).
    pub const fn ordered() -> &'static [WizardState] {
        &[
            WizardState::Welcome,
            WizardState::Path,
            WizardState::AccountProfile,
            WizardState::OAuth,
            WizardState::Symbols,
            WizardState::Historical,
            WizardState::Hardware,
            WizardState::NewsApi,
            WizardState::Autostart,
            WizardState::AutonomyRisk,
            WizardState::Summary,
        ]
    }

    /// 0-based index in `ordered()`.
    pub fn index(self) -> usize {
        Self::ordered()
            .iter()
            .position(|s| *s == self)
            .expect("WizardState::ordered() must contain every variant")
    }

    /// Next state in the ordered sequence, or `None` if this is the
    /// terminal step.
    pub fn next(self) -> Option<WizardState> {
        let idx = self.index();
        Self::ordered().get(idx + 1).copied()
    }

    /// Previous state, or `None` if this is the first step.
    pub fn previous(self) -> Option<WizardState> {
        let idx = self.index();
        if idx == 0 {
            None
        } else {
            Self::ordered().get(idx - 1).copied()
        }
    }

    /// Stable string key — used for `skipped_steps` / `incomplete_steps`
    /// fields in `WizardStateFile`.
    pub fn key(self) -> &'static str {
        match self {
            WizardState::Welcome => "welcome",
            WizardState::Path => "path",
            WizardState::AccountProfile => "account_profile",
            WizardState::OAuth => "ctrader_oauth",
            WizardState::Symbols => "symbols",
            WizardState::Historical => "historical_download",
            WizardState::Hardware => "hardware_probe",
            WizardState::NewsApi => "news_api",
            WizardState::Autostart => "autostart",
            WizardState::AutonomyRisk => "autonomy_risk",
            WizardState::Summary => "summary",
        }
    }

    /// Whether the step can be skipped. Spec §5 — only Welcome is
    /// non-skippable globally; AutonomyRisk is conditionally
    /// non-skippable (see `WizardController::is_skippable`).
    pub const fn is_skippable_default(self) -> bool {
        !matches!(self, WizardState::Welcome | WizardState::Summary)
    }

    /// Human-readable label for the breadcrumb / progress tracker.
    pub fn label(self) -> &'static str {
        match self {
            WizardState::Welcome => "Welcome & License",
            WizardState::Path => "Data Path",
            WizardState::AccountProfile => "Account & Profile",
            WizardState::OAuth => "cTrader Sign-in",
            WizardState::Symbols => "Symbols & Timeframes",
            WizardState::Historical => "Historical Data",
            WizardState::Hardware => "Hardware Probe",
            WizardState::NewsApi => "News & Safeguards",
            WizardState::Autostart => "Auto-start",
            WizardState::AutonomyRisk => "Autonomy & Risk",
            WizardState::Summary => "Summary & Apply",
        }
    }
}

/// Per-step status, persisted to `wizard_state.json` under
/// `completed_steps`. Spec §5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WizardStepStatus {
    Pending,
    Completed,
    Skipped,
    /// Step was reached, an error fired, and the user chose to
    /// continue past it (spec §3 "Never silently skip" → recorded as
    /// `Incomplete` rather than `Skipped`).
    Incomplete,
}

/// Wizard-level errors — surfaces in the UI verbatim per spec §3
/// rule 1 ("Never silently skip" — always log).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WizardError {
    LicenseMissing,
    PathNoWritePermission(String),
    PathLowDisk { free_gib: u64, threshold_gib: u64 },
    MonthlyTargetBelowFloor { requested: f32, floor: f32 },
    OAuthLoopbackBindFailed { tried_ports: Vec<u16> },
    OAuthCallbackTimeout,
    OAuthCsrfMismatch,
    OAuthTokenExchange(String),
    SymbolsListTimeout,
    HistoricalRateLimited,
    HistoricalCancelled,
    HardwareNoGpu,
    NewsApiPingFailed(String),
    AutostartWriteFailed(String),
    SummaryDiskFull,
    KeychainLocked,
    /// Generic catch-all — the wizard surfaces this verbatim so the
    /// operator sees the raw broker/OS error.
    Other(String),
}

impl WizardError {
    pub fn message(&self) -> String {
        match self {
            WizardError::LicenseMissing => {
                "LICENSE file missing — falling back to embedded copy.".to_string()
            }
            WizardError::PathNoWritePermission(p) => format!("No write permission: {}", p),
            WizardError::PathLowDisk {
                free_gib,
                threshold_gib,
            } => format!(
                "Low disk: {} GiB free (recommended ≥ {} GiB)",
                free_gib, threshold_gib
            ),
            WizardError::MonthlyTargetBelowFloor { requested, floor } => format!(
                "Minimum {:.0}% per operator policy (requested {:.2}%)",
                floor * 100.0,
                requested * 100.0
            ),
            WizardError::OAuthLoopbackBindFailed { tried_ports } => format!(
                "Could not bind any of the loopback ports {:?} — use copy-paste flow.",
                tried_ports
            ),
            WizardError::OAuthCallbackTimeout => {
                "cTrader sign-in timed out (5 min). Retry or skip.".to_string()
            }
            WizardError::OAuthCsrfMismatch => {
                "CSRF state mismatch — sign-in refused for safety.".to_string()
            }
            WizardError::OAuthTokenExchange(s) => format!("Broker rejected token exchange: {}", s),
            WizardError::SymbolsListTimeout => {
                "Symbol-list request timed out — broker maintenance window?".to_string()
            }
            WizardError::HistoricalRateLimited => {
                "Broker rate limit hit; backing off 30 s before resume.".to_string()
            }
            WizardError::HistoricalCancelled => {
                "Download cancelled — partial files preserved.".to_string()
            }
            WizardError::HardwareNoGpu => "No GPU detected — falling back to CPU.".to_string(),
            WizardError::NewsApiPingFailed(s) => format!("News API ping failed: {}", s),
            WizardError::AutostartWriteFailed(s) => format!("Autostart write failed: {}", s),
            WizardError::SummaryDiskFull => {
                "Disk full while writing config — free space and retry.".to_string()
            }
            WizardError::KeychainLocked => {
                "macOS keychain locked — falling back to file storage (mode 0o600).".to_string()
            }
            WizardError::Other(s) => s.clone(),
        }
    }
}

/// Risk acknowledgement record — spec §5 + competitive analysis §9.2.
/// The 5-question quiz answers are hashed with SHA-256 along with the
/// ISO-8601 timestamp; the wizard does not store the raw answers
/// long-term to keep the file small.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskAcknowledgement {
    /// SHA-256 of the concatenation of (question_id, chosen_option_id)
    /// pairs in canonical order, plus the timestamp string.
    pub answers_sha256: String,
    /// ISO-8601 UTC string at the moment of acknowledgement.
    pub timestamp_utc: String,
    /// Quiz version — bump if the question set changes.
    pub quiz_version: u32,
    /// Number of correct answers (out of 5). The wizard refuses to
    /// advance unless `correct == 5` per competitive analysis §9.2
    /// "Cannot Continue until 5/5 correct".
    pub correct_count: u8,
}

/// Install-time metadata, persisted alongside the wizard state so a
/// re-run can tell whether the install was a fresh MSI/pkg, an in-
/// place upgrade, or a `cargo install`. Spec §1.3.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstallMetadata {
    pub installer_version: Option<String>,
    pub installed_at_utc: Option<String>,
    pub install_path: Option<String>,
    pub data_path: Option<String>,
}

/// Persisted state file. Spec §5.
///
/// Serialised to `<config_dir>/wizard_state.json` on Apply (Step 10)
/// or on any explicit Skip. The `last_updated_utc_ms` lets a
/// concurrent forex-app instance detect a stale file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WizardStateFile {
    /// Schema version — bumped on breaking changes.
    pub version: u32,
    /// Steps the wizard has completed (in advance order).
    #[serde(default)]
    pub completed_steps: Vec<WizardState>,
    /// Steps the user explicitly skipped (spec §3 rule 1 — never
    /// silently skip).
    #[serde(default)]
    pub skipped_steps: Vec<WizardState>,
    /// Steps that were attempted but errored; the main app banners
    /// these on next launch.
    #[serde(default)]
    pub incomplete_steps: Vec<WizardState>,
    /// Install-time metadata.
    #[serde(default)]
    pub install_metadata: InstallMetadata,
    /// Risk acknowledgement (Step 9.5). `None` unless the user
    /// completed the quiz.
    #[serde(default)]
    pub risk_acknowledgement: Option<RiskAcknowledgement>,
    /// Last write time as Unix-milliseconds, UTC.
    #[serde(default)]
    pub last_updated_utc_ms: i64,
}

impl WizardStateFile {
    pub fn new() -> Self {
        Self {
            version: WIZARD_STATE_FILE_VERSION,
            ..Self::default()
        }
    }

    /// Returns the first incomplete step in `WizardState::ordered()`.
    /// On a fresh install, returns `Welcome`. Spec §5.2.
    pub fn first_incomplete_step(&self) -> WizardState {
        for state in WizardState::ordered() {
            if !self.completed_steps.contains(state) && !self.skipped_steps.contains(state) {
                return *state;
            }
        }
        // Wizard finished; default to terminal step (caller should
        // gate via `is_complete()`).
        WizardState::Summary
    }

    pub fn is_complete(&self) -> bool {
        self.completed_steps.contains(&WizardState::Summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_contains_eleven_states() {
        // 10 numbered + 9.5 Autonomy & Risk
        assert_eq!(WizardState::ordered().len(), 11);
    }

    #[test]
    fn next_and_previous_chain_in_order() {
        let states = WizardState::ordered();
        for window in states.windows(2) {
            assert_eq!(window[0].next(), Some(window[1]));
            assert_eq!(window[1].previous(), Some(window[0]));
        }
        assert_eq!(states.first().unwrap().previous(), None);
        assert_eq!(states.last().unwrap().next(), None);
    }

    #[test]
    fn welcome_and_summary_are_not_skippable_by_default() {
        assert!(!WizardState::Welcome.is_skippable_default());
        assert!(!WizardState::Summary.is_skippable_default());
        for other in WizardState::ordered() {
            if !matches!(other, WizardState::Welcome | WizardState::Summary) {
                assert!(other.is_skippable_default(), "{:?} should be skippable", other);
            }
        }
    }

    #[test]
    fn first_incomplete_step_returns_welcome_on_fresh_state() {
        let file = WizardStateFile::new();
        assert_eq!(file.first_incomplete_step(), WizardState::Welcome);
    }

    #[test]
    fn first_incomplete_step_skips_completed_and_skipped_entries() {
        let mut file = WizardStateFile::new();
        file.completed_steps
            .extend_from_slice(&[WizardState::Welcome, WizardState::Path]);
        file.skipped_steps.push(WizardState::AccountProfile);
        assert_eq!(file.first_incomplete_step(), WizardState::OAuth);
    }
}
