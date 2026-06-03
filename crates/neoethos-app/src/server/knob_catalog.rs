//! `/settings/knob-catalog` — machine-readable catalog of every runtime
//! knob the bot honours.
//!
//! This is the JSON counterpart to `docs/CONFIG-KNOBS-REFERENCE.md` —
//! the operator-facing markdown reference. The Flutter "Advanced
//! Settings" screen calls `GET /settings/knob-catalog`, gets back a
//! flat list of `KnobEntry` records, and renders each with a tooltip
//! (description), a numeric / dropdown / toggle widget (driven by
//! `kind`), and a "current value" badge (live from the typed runtime
//! overrides).
//!
//! ## Why this exists
//!
//! Operator directive 2026-05-25: "βάζουμε στο UI όλες τις πιθανές
//! επιλογές που υπάρχουν μαζί με ένα help section που να εξηγεί τι
//! είναι το κάθε τι και πως επιρεαζει το bot και τις λειτουργίες
//! του. Φυσικά θα υπάρχουν κάποια presets για ευκολία και ασφάλεια."
//!
//! Translation: "we put in the UI every possible option that exists
//! plus a help section that explains what each one is and how it
//! affects the bot and its functions. Of course there will be some
//! presets for convenience and safety."
//!
//! ## Layering
//!
//! - **Catalog** (this module): the schema + help text + defaults +
//!   ranges + preset values. Compiled into the binary so the
//!   Flutter UI never has to bundle a copy of the help text.
//! - **Current values**: read at request time from the installed
//!   `*RuntimeOverrides` structs via the existing
//!   `current_*_runtime_overrides()` accessors. This is the
//!   "what is the bot using right now" badge in the UI.
//! - **Write path** (future): `POST /settings/knobs` will write the
//!   operator's changes to `config.yaml` and tell them to restart.
//!   Hot-reloading typed overrides is out of scope for Phase 1 (it
//!   would require converting OnceLock → RwLock across every
//!   override struct, which is a bigger refactor).

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};

use super::state::AppApiState;

/// One knob in the catalog. Serializes as JSON for the Flutter UI.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KnobEntry {
    /// Stable machine-readable id (e.g. `"ctrader.max_attempts"`).
    /// The Flutter UI keys on this; renaming a knob requires
    /// updating the front-end too.
    pub id: &'static str,

    /// Section the knob belongs to (matches the headings in
    /// `docs/CONFIG-KNOBS-REFERENCE.md`).
    pub section: &'static str,

    /// Human-readable display name for the Settings card.
    pub label: &'static str,

    /// Legacy env-var name (still honoured for backward compat).
    /// `None` when the knob has no env-var equivalent.
    pub env_var: Option<&'static str>,

    /// What kind of widget the UI should render. **2026-05-26**: this
    /// field is `#[serde(flatten)]` so the `KnobKind` variant's tag
    /// (`kind`) and constraint fields (`min`, `max`, `enumChoices`)
    /// surface as top-level JSON keys instead of a nested object —
    /// matching what Flutter's `KnobDescriptor.fromJson` expects.
    #[serde(flatten)]
    pub kind: KnobKind,

    /// Default value (as a string — the front-end parses by `kind`).
    pub default: &'static str,

    /// Current value (as a string — read at request time from the
    /// installed runtime overrides; falls back to the default when
    /// no install has happened).
    pub current: String,

    /// Short help text (1-2 sentences) — shown as a tooltip.
    pub help_short: &'static str,

    /// Long help text — shown in an expanded info box. Includes the
    /// "effect on the bot" explanation from the markdown reference.
    pub help_long: &'static str,

    /// Recommended value for each safety preset. Empty values mean
    /// "use the default" or "this knob isn't preset-driven".
    pub preset_conservative: &'static str,
    pub preset_balanced: &'static str,
    pub preset_aggressive: &'static str,
}

/// **2026-05-26 fix (Κωνσταντίνος)**: was previously serialized via the
/// default externally-tagged + kebab-case representation, producing
/// JSON like `{"int": {"min": 0, "max": 3600}}` for struct variants
/// and `"bool"` for unit variants. The Flutter Advanced Settings
/// screen (`advanced_settings_screen.dart:646-674`) parses with:
///   ```dart
///   kind: j['kind'] as String? ?? 'Text',
///   minValue: (j['min'] as num?)?.toDouble(),
///   enumChoices: (j['enumChoices'] as List?)?.cast<String>(),
///   ```
/// — i.e. expects a **flat** shape with `kind` as a String and
/// `min`/`max`/`enumChoices` as siblings of `kind`. Under the old
/// serde, `j['kind']` for any Int/Float/Enum knob was a `Map`, so the
/// `as String?` cast fell through to `'Text'`, every numeric input
/// rendered as a plain TextField with no clamping, every enum lost
/// its dropdown, and the Save button silently dropped most edits.
///
/// Switching to `#[serde(tag = "kind")]` + `#[serde(flatten)]` on the
/// containing `KnobEntry.kind` field gives us exactly the shape the
/// UI wants: `{"id": "...", "kind": "Int", "min": 0, "max": 3600,
/// ...}`. Unit variants serialize as `{"kind": "Bool"}` with no
/// extras. The `variants` field renames to `enumChoices` to match
/// the camelCase wire convention used by the rest of the DTOs.
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(tag = "kind")]
pub enum KnobKind {
    /// Integer with optional min/max clamp.
    Int {
        #[serde(skip_serializing_if = "Option::is_none")]
        min: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max: Option<i64>,
    },
    /// Floating-point with optional min/max clamp.
    Float {
        #[serde(skip_serializing_if = "Option::is_none")]
        min: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max: Option<f64>,
    },
    /// Boolean checkbox.
    Bool,
    /// Free-text string.
    Text,
    /// One of a fixed enum set. Flutter side reads this as
    /// `enumChoices`, not `variants`.
    Enum {
        #[serde(rename = "enumChoices")]
        variants: &'static [&'static str],
    },
    /// Filesystem path.
    Path,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KnobCatalogResponse {
    pub schema_version: u32,
    pub generated_at_unix_ms: i64,
    pub knobs: Vec<KnobEntry>,
}

const SCHEMA_VERSION: u32 = 1;

/// Build the catalog. Each entry reads its `current` value from the
/// installed runtime overrides; the static parts come from the
/// catalog array.
fn build_catalog() -> Vec<KnobEntry> {
    use neoethos_search::current_genetic_search_runtime_overrides as ga;
    use neoethos_search::current_quality_runtime_overrides as quality;
    use neoethos_search::current_strategy_evaluation_runtime_overrides as strat;

    let ga_overrides = ga();
    let quality_overrides = quality();
    let strat_overrides = strat();

    let cost = &strat_overrides.cost_profile;
    let smc = &ga_overrides.smc_gate;

    vec![
        // ── Section 1 — Broker connectivity ──────────────────────────────
        KnobEntry {
            id: "ctrader.read_timeout_secs",
            section: "Broker connectivity (cTrader)",
            label: "Read timeout (seconds)",
            env_var: Some("NEOETHOS_BOT_CTRADER_READ_TIMEOUT_SECS"),
            kind: KnobKind::Int { min: Some(0), max: Some(3600) },
            default: "30",
            current: "30".to_string(), // Read inline in execute_via_session; no typed cache yet.
            help_short: "Caps the TCP read for cTrader execution; 0 disables.",
            help_long: "Without this cap, a broker stall could wedge the trading loop indefinitely. With it, the I/O error bubbles up, the session is dropped, and the next attempt re-authenticates. Lower (15s) for low-latency colo; raise for slow consumer networks.",
            preset_conservative: "30",
            preset_balanced: "30",
            preset_aggressive: "15",
        },
        KnobEntry {
            id: "ctrader.max_attempts",
            section: "Broker connectivity (cTrader)",
            label: "Max execution attempts",
            env_var: Some("NEOETHOS_BOT_CTRADER_MAX_ATTEMPTS"),
            kind: KnobKind::Int { min: Some(1), max: Some(5) },
            default: "3",
            current: "3".to_string(),
            help_short: "Initial + retries per cTrader order. Retry safety relies on the broker deduping by clientOrderId.",
            help_long: "Lower (2) is safer for the prop-firm gate; higher (5) gives more retry resilience at the cost of duplicate-order risk if the broker's dedup misbehaves.",
            preset_conservative: "2",
            preset_balanced: "3",
            preset_aggressive: "5",
        },
        KnobEntry {
            id: "ctrader.backoff_base_ms",
            section: "Broker connectivity (cTrader)",
            label: "Retry backoff base (ms)",
            env_var: Some("NEOETHOS_BOT_CTRADER_BACKOFF_BASE_MS"),
            kind: KnobKind::Int { min: Some(10), max: Some(2000) },
            default: "200",
            current: "200".to_string(),
            help_short: "Base backoff in ms; doubles per attempt with 0-99ms jitter, capped at 5s total.",
            help_long: "Slower retries (500ms) are gentler on the broker; faster (100ms) recovers quicker from transient errors but risks rate-limiting.",
            preset_conservative: "500",
            preset_balanced: "200",
            preset_aggressive: "100",
        },
        KnobEntry {
            id: "ctrader.allow_partial_fill",
            section: "Broker connectivity (cTrader)",
            label: "Accept partial fills",
            env_var: Some("NEOETHOS_BOT_CTRADER_ALLOW_PARTIAL_FILL"),
            kind: KnobKind::Bool,
            default: "false",
            current: "false".to_string(),
            help_short: "When off, partial fills error out; when on, they're accepted as final.",
            help_long: "Conservative trading rejects partial fills so the risk-per-trade math stays consistent. Aggressive mode accepts whatever the broker can fill — useful on illiquid pairs where partial is better than nothing.",
            preset_conservative: "false",
            preset_balanced: "false",
            preset_aggressive: "true",
        },
        KnobEntry {
            id: "ctrader.chart_merge_side",
            section: "Broker connectivity (cTrader)",
            label: "Chart merge side",
            env_var: Some("NEOETHOS_BOT_CHART_MERGE_SIDE"),
            kind: KnobKind::Enum { variants: &["mid", "bid", "ask"] },
            default: "mid",
            current: "mid".to_string(),
            help_short: "Which side of the spread the chart layer uses when one price is needed.",
            help_long: "Mid is the broker-standard convention. Bid/Ask let you model worst-case slippage explicitly for backtests.",
            preset_conservative: "mid",
            preset_balanced: "mid",
            preset_aggressive: "mid",
        },

        // ── Section 2 — Risk & PnL ─────────────────────────────────────
        KnobEntry {
            id: "risk.account_currency",
            section: "Risk & PnL safety",
            label: "Account currency (ISO-4217)",
            env_var: Some("NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY"),
            kind: KnobKind::Text,
            default: "(unset → hard-fail at risk gate)",
            current: cost
                .account_currency
                .clone()
                .unwrap_or_else(|| "(unset)".to_string()),
            help_short:
                "ISO-4217 code (USD/EUR/GBP/JPY/CHF/CAD/AUD/NZD…) for the funded account. Required.",
            help_long:
                "Drives the pip-value math in the risk gate. With a wrong account currency, position sizing is wrong (e.g. treating GBP as USD overstates risk by ~20%). Operator-supplied via Settings → Broker Setup once; never auto-defaulted.",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "paths.symbol_metadata_override",
            section: "Logging / persistence",
            label: "Symbol-metadata path override",
            env_var: Some("NEOETHOS_BOT_SYMBOL_METADATA"),
            kind: KnobKind::Path,
            default: "data/symbol_metadata.json",
            current: "data/symbol_metadata.json".to_string(),
            help_short: "Overrides the on-disk symbol-metadata JSON file.",
            help_long:
                "By default the bot reads `data/symbol_metadata.json` (auto-populated from the cTrader ProtoOASymbol records). Override to point at a frozen snapshot for reproducible offline backtests.",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "paths.user_data_dir_override",
            section: "Logging / persistence",
            label: "User data dir override",
            env_var: Some("NEOETHOS_USER_DATA_DIR"),
            kind: KnobKind::Path,
            default: "(platform default — %LOCALAPPDATA% on Windows)",
            current: "(platform default)".to_string(),
            help_short:
                "Where logs and persistent state live. Override to redirect to a portable drive or RAM disk.",
            help_long:
                "By default `dirs::data_local_dir()` resolves to `%LOCALAPPDATA%/NeoEthos` on Windows. Override only if you have a specific reason (portable installation, faster disk, segregated backup target).",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "risk.prop_firm_preset",
            section: "Risk & PnL safety",
            label: "Prop-firm preset",
            env_var: Some("NEOETHOS_PROP_FIRM_PRESET"),
            kind: KnobKind::Enum {
                variants: &["ftmo", "myforexfunds", "fundednext", "the5ers", "none"],
            },
            default: "ftmo",
            current: "ftmo".to_string(),
            help_short: "Seeds RiskConfig defaults from the chosen prop-firm's published rules.",
            help_long: "Sets daily-loss / max-drawdown / profit-target / min-trading-days defaults. Operator can override individual fields. Pick the preset that matches your funded account.",
            preset_conservative: "ftmo",
            preset_balanced: "ftmo",
            preset_aggressive: "none",
        },
        KnobEntry {
            id: "risk.pnl_audit_drift_fraction",
            section: "Risk & PnL safety",
            label: "PnL audit drift threshold",
            env_var: Some("NEOETHOS_BOT_PNL_AUDIT_DRIFT_FRACTION"),
            kind: KnobKind::Float { min: Some(1e-5), max: Some(0.05) },
            default: "0.001",
            current: "0.001".to_string(),
            help_short: "Drift threshold (fraction of notional) above which a PnL audit warning is logged.",
            help_long: "When broker-side unrealized PnL diverges from local mark-to-market by more than this fraction, a warning is logged. Helps catch bad pip-value mappings before they cost money. 5bp = paranoid; 10bp = default; 50bp = quiet.",
            preset_conservative: "0.0005",
            preset_balanced: "0.001",
            preset_aggressive: "0.005",
        },
        KnobEntry {
            id: "risk.pnl_circuit_breaker_fraction",
            section: "Risk & PnL safety",
            label: "PnL circuit breaker",
            env_var: Some("NEOETHOS_BOT_PNL_CIRCUIT_BREAKER_FRACTION"),
            kind: KnobKind::Float { min: Some(1e-4), max: Some(0.20) },
            default: "0.01",
            current: "0.01".to_string(),
            help_short: "Drift threshold (fraction of notional) that halts the auto-trader for operator review.",
            help_long: "Upper bound 20% so the breaker can't be silenced by a typo. Lower bound 1bp avoids tripping on float epsilon when broker + local agree. 50bp = paranoid; 1% = default; 5% = only-loud-drift (NOT recommended for prop-firm).",
            preset_conservative: "0.005",
            preset_balanced: "0.01",
            preset_aggressive: "0.05",
        },
        KnobEntry {
            id: "risk.require_stop_loss",
            section: "Risk & PnL safety",
            label: "Require Stop-Loss on every order",
            env_var: None,
            kind: KnobKind::Bool,
            default: "true",
            current: "true".to_string(),
            help_short: "When on, the risk gate REJECTS any order without a stop_loss.",
            help_long: "**F-249/F-271 closure (2026-05-25 — operator-approved configurable preset)**: Conservative preset turns this ON so prop-firm validation never trades without SL. Balanced/Aggressive turn it OFF for scalp strategies that fill first and place SL second. NOTE: Risky Mode ALWAYS requires SL+TP regardless of this flag (kill-switch math depends on it).",
            preset_conservative: "true",
            preset_balanced: "false",
            preset_aggressive: "false",
        },
        KnobEntry {
            id: "risk.reject_pip_fallback",
            section: "Risk & PnL safety",
            label: "Reject cross-pair pip-value fallback",
            env_var: Some("NEOETHOS_BOT_REJECT_PIP_FALLBACK"),
            kind: KnobKind::Bool,
            default: "false",
            current: format!("{}", cost.reject_pip_fallback),
            help_short: "When on, cross-pair pip-value fallback bails out instead of using a possibly-wrong quote-currency value.",
            help_long: "Recommended on prop-firm runs: fail loudly if a cross pair lacks an FX rate. Off keeps the historical tolerant behaviour.",
            preset_conservative: "true",
            preset_balanced: "false",
            preset_aggressive: "false",
        },

        // ── Section 3 — Discovery / GA ─────────────────────────────────
        KnobEntry {
            id: "ga.seed",
            section: "Discovery / GA search",
            label: "RNG seed (deterministic)",
            env_var: Some("NEOETHOS_BOT_SEARCH_SEED"),
            kind: KnobKind::Int { min: Some(0), max: None },
            default: "(unset — non-deterministic)",
            current: ga_overrides.seed.map(|s| s.to_string()).unwrap_or_else(|| "(unset)".to_string()),
            help_short: "Setting any value makes the GA run deterministic. Unset → OS-RNG seed.",
            help_long: "Set during validation runs to compare changes apples-to-apples. Leave unset for production search so the GA gets fresh randomness.",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "ga.novelty_weight",
            section: "Discovery / GA search",
            label: "Novelty bonus weight",
            env_var: Some("NEOETHOS_BOT_NOVELTY_WEIGHT"),
            kind: KnobKind::Float { min: Some(0.0), max: Some(1.0) },
            default: "0.0",
            current: format!("{}", ga_overrides.novelty_weight),
            help_short: "Favours diverse genes during selection; 0 = pure fitness ranking.",
            help_long: "Useful when the GA gets stuck in a local optimum. 0.05 = mild diversity nudge; 0.15 = strong novelty push.",
            preset_conservative: "0.0",
            preset_balanced: "0.05",
            preset_aggressive: "0.15",
        },
        KnobEntry {
            id: "ga.stagnation_patience",
            section: "Discovery / GA search",
            label: "Stagnation patience (generations)",
            env_var: Some("NEOETHOS_BOT_PROP_STAGNATION_GENS"),
            kind: KnobKind::Int { min: Some(1), max: Some(50) },
            default: "2",
            current: format!("{}", ga_overrides.stagnation_patience),
            help_short: "Generations of no-progress before early-stop / gate relaxation triggers.",
            help_long: "Conservative (1) stops fast to save compute. Aggressive (5+) lets the search push through stagnation.",
            preset_conservative: "1",
            preset_balanced: "2",
            preset_aggressive: "5",
        },
        KnobEntry {
            id: "ga.tournament_size",
            section: "Discovery / GA search",
            label: "Tournament size override",
            env_var: Some("NEOETHOS_BOT_PROP_TOURNAMENT_SIZE"),
            kind: KnobKind::Int { min: Some(2), max: Some(64) },
            default: "(derived: max(pop/12, 3))",
            current: ga_overrides
                .tournament_size_override
                .map(|s| s.to_string())
                .unwrap_or_else(|| "(derived)".to_string()),
            help_short: "Larger tournaments → stronger selection → faster convergence, less diversity.",
            help_long: "Default is population-derived. Manually pin only if your search is converging too slowly (raise) or losing diversity (lower).",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "ga.smc_gate_start",
            section: "Discovery / GA search",
            label: "SMC gate start threshold",
            env_var: Some("NEOETHOS_BOT_PROP_SMC_GATE_START"),
            kind: KnobKind::Float { min: Some(0.0), max: Some(1.0) },
            default: "0.75",
            current: format!("{}", smc.start),
            help_short: "Where the SMC confluence gate begins each run.",
            help_long: "Higher (0.85) = only strong SMC confluence passes. Lower (0.65) = permissive. Gate decays toward `end` over the run.",
            preset_conservative: "0.85",
            preset_balanced: "0.75",
            preset_aggressive: "0.65",
        },
        KnobEntry {
            id: "ga.smc_gate_end",
            section: "Discovery / GA search",
            label: "SMC gate end threshold",
            env_var: Some("NEOETHOS_BOT_PROP_SMC_GATE_END"),
            kind: KnobKind::Float { min: Some(0.0), max: Some(1.0) },
            default: "0.35",
            current: format!("{}", smc.end),
            help_short: "Floor for the SMC confluence gate.",
            help_long: "The gate decays from `start` to `end` along a power curve. Higher (0.45) = strict floor; lower (0.25) = permissive.",
            preset_conservative: "0.45",
            preset_balanced: "0.35",
            preset_aggressive: "0.25",
        },
        KnobEntry {
            id: "ga.disable_smc_gate",
            section: "Discovery / GA search",
            label: "Disable SMC gate (diagnostic)",
            env_var: Some("NEOETHOS_BOT_DISABLE_SMC_GATE"),
            kind: KnobKind::Bool,
            default: "false",
            current: format!("{}", smc.disable_gate),
            help_short: "DIAGNOSTIC ONLY: forces the SMC gate to bypass (active sum = 0).",
            help_long: "Useful for isolating 'SMC indicators don't trigger on this symbol' from 'signal generation is broken'. Should NEVER be enabled in production — your strategies become unfiltered.",
            preset_conservative: "false",
            preset_balanced: "false",
            preset_aggressive: "false",
        },

        // ── Section 5 — Quality / acceptance ───────────────────────────
        KnobEntry {
            id: "quality.min_trades_per_month",
            section: "Quality / acceptance filtering",
            label: "Min trades per month",
            env_var: Some("NEOETHOS_BOT_PROP_MIN_TRADES_PER_MONTH"),
            kind: KnobKind::Int { min: Some(1), max: Some(200) },
            default: "4",
            current: format!("{}", quality_overrides.min_trades_per_month),
            help_short: "Strategies with fewer trades/month than this are rejected as undersampled.",
            help_long: "Conservative (8) keeps only well-sampled strategies. Aggressive (2) tolerates rare-but-edge-rich strategies.",
            preset_conservative: "8",
            preset_balanced: "4",
            preset_aggressive: "2",
        },
        KnobEntry {
            id: "quality.trading_days_per_month",
            section: "Quality / acceptance filtering",
            label: "Trading days per month",
            env_var: Some("NEOETHOS_BOT_TRADING_DAYS_PER_MONTH"),
            kind: KnobKind::Float { min: Some(1.0), max: Some(31.0) },
            default: "21.0",
            current: format!("{}", quality_overrides.trading_days_per_month),
            help_short: "Used to normalize trade frequency across calendars.",
            help_long: "Forex trades ~22 days/month; 21 is the conservative round. Only matters for cross-strategy comparison.",
            preset_conservative: "21.0",
            preset_balanced: "21.0",
            preset_aggressive: "21.0",
        },

        // ── Section 1 cont. — Broker streaming + transport ────────────
        KnobEntry {
            id: "ctrader.stream_max_attempts",
            section: "Broker connectivity (cTrader)",
            label: "Streaming max attempts",
            env_var: Some("NEOETHOS_BOT_CTRADER_STREAM_MAX_ATTEMPTS"),
            kind: KnobKind::Int { min: Some(1), max: Some(5) },
            default: "3",
            current: "3".to_string(),
            help_short: "Max attempts for `load_live_chart_update` poll (stateless, safe to retry).",
            help_long: "Streaming polls are idempotent, so retries are safe. Default 3 is sensible for typical WiFi/VPS latencies; raise (5) on unstable links.",
            preset_conservative: "3",
            preset_balanced: "3",
            preset_aggressive: "5",
        },
        KnobEntry {
            id: "ctrader.stream_backoff_base_ms",
            section: "Broker connectivity (cTrader)",
            label: "Streaming backoff base (ms)",
            env_var: Some("NEOETHOS_BOT_CTRADER_STREAM_BACKOFF_BASE_MS"),
            kind: KnobKind::Int { min: Some(10), max: Some(2000) },
            default: "200",
            current: "200".to_string(),
            help_short: "Streaming-layer retry backoff base; doubles per attempt, capped at 5s total.",
            help_long: "Mirror of the execution backoff. Slower (500ms) on unstable networks; faster (100ms) on colo / low-latency.",
            preset_conservative: "500",
            preset_balanced: "200",
            preset_aggressive: "100",
        },

        // ── Section 2 cont. — Risk knobs ─────────────────────────────
        KnobEntry {
            id: "risk.quote_to_account_rate",
            section: "Risk & PnL safety",
            label: "Quote→account FX rate override",
            env_var: Some("NEOETHOS_BOT_PROP_QUOTE_TO_ACCOUNT_RATE"),
            kind: KnobKind::Float { min: Some(0.000_001), max: None },
            default: "(unset — broker-derived)",
            current: cost
                .quote_to_account_rate
                .map(|v| format!("{v}"))
                .unwrap_or_else(|| "(unset)".to_string()),
            help_short: "Live quote→account FX rate for cross-pair pip math. Only used during initial session before broker feeds it.",
            help_long: "Cross pairs (e.g. EURGBP on a USD account) need a quote→account rate to convert pip value. The broker feeds this once the session is up; this override is mainly for offline backtests / paper trading when no live broker is connected.",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "risk.pip_value",
            section: "Cost model / pip-value",
            label: "Per-pip account-currency value (override)",
            env_var: Some("NEOETHOS_BOT_PROP_PIP_VALUE"),
            kind: KnobKind::Float { min: Some(0.000_001), max: None },
            default: "(broker symbol metadata)",
            current: cost
                .pip_value
                .map(|v| format!("{v}"))
                .unwrap_or_else(|| "(broker)".to_string()),
            help_short: "Manual pip-value override. Leave unset; let broker symbol_metadata.json drive.",
            help_long: "Only set when you're stress-testing a specific pip assumption (e.g. simulating a JPY pair with non-standard pip semantics). Production runs leave this empty and read from the broker's ProtoOASymbol records.",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "risk.pip_value_per_lot",
            section: "Cost model / pip-value",
            label: "Per-lot pip value (override)",
            env_var: Some("NEOETHOS_BOT_PROP_PIP_VALUE_PER_LOT"),
            kind: KnobKind::Float { min: Some(0.000_001), max: None },
            default: "(broker symbol metadata)",
            current: cost
                .pip_value_per_lot
                .map(|v| format!("{v}"))
                .unwrap_or_else(|| "(broker)".to_string()),
            help_short: "Pip value per standard lot, in account currency. Override only for stress-testing.",
            help_long: "Companion to `risk.pip_value`. Same recommendation: leave unset for production.",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "cost.spread_pips",
            section: "Cost model / pip-value",
            label: "Spread override (pips)",
            env_var: Some("NEOETHOS_BOT_PROP_SPREAD_PIPS"),
            kind: KnobKind::Float { min: Some(0.0), max: Some(100.0) },
            default: "(broker-quoted)",
            current: cost
                .spread_pips
                .map(|v| format!("{v}"))
                .unwrap_or_else(|| "(broker)".to_string()),
            help_short: "Stress-test override for the spread used in backtest cost math.",
            help_long: "Conservative: +0.5 over broker-quoted (buffer for worst-case). Balanced: leave unset (use broker quote). Aggressive: 0.0 (zero-friction theoretical baseline). For prop-firm validation, stress-test with the widest spread your broker has quoted in the last 30 days.",
            preset_conservative: "0.5",
            preset_balanced: "",
            preset_aggressive: "0.0",
        },
        KnobEntry {
            id: "cost.commission_per_trade",
            section: "Cost model / pip-value",
            label: "Commission per trade (override)",
            env_var: Some("NEOETHOS_BOT_PROP_COMMISSION"),
            kind: KnobKind::Float { min: Some(0.0), max: Some(50.0) },
            default: "(broker-quoted)",
            current: cost
                .commission_per_trade
                .map(|v| format!("{v}"))
                .unwrap_or_else(|| "(broker)".to_string()),
            help_short: "Override commission per round-trip per standard lot.",
            help_long: "Stress-test with the worst-case commission your broker quotes for your account class. cTrader typically charges $3-7 per round-trip per standard lot for raw-spread accounts.",
            preset_conservative: "7.0",
            preset_balanced: "",
            preset_aggressive: "0.0",
        },

        // ── Section 3 cont. — GA archive / selection / SMC gate curve ─
        KnobEntry {
            id: "ga.archive_cap",
            section: "Discovery / GA search",
            label: "Archive capacity override",
            env_var: Some("NEOETHOS_BOT_PROP_ARCHIVE_CAP"),
            kind: KnobKind::Int { min: Some(0), max: Some(200_000) },
            default: "(derived: min(pop × gens, 50000))",
            current: ga_overrides
                .archive_cap_override
                .map(|v| v.to_string())
                .unwrap_or_else(|| "(derived)".to_string()),
            help_short: "Maximum genes kept in the archive. Override only if RAM-constrained or deep-tuning.",
            help_long: "Larger archive = more candidates for the final picker, more RAM. Capped at 200_000 to prevent blowups on very-long HPC runs.",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "ga.smc_gate_curve",
            section: "Discovery / GA search",
            label: "SMC gate curve exponent",
            env_var: Some("NEOETHOS_BOT_PROP_SMC_GATE_CURVE"),
            kind: KnobKind::Float { min: Some(0.1), max: Some(5.0) },
            default: "1.0",
            current: format!("{}", smc.curve),
            help_short: "Power-curve exponent for the SMC gate decay between start and end.",
            help_long: "1.0 = linear decay. Higher (1.5-2.0) keeps the gate strict longer (more concave). Lower (0.5-0.7) relaxes faster (more convex).",
            preset_conservative: "1.5",
            preset_balanced: "1.0",
            preset_aggressive: "0.7",
        },
        KnobEntry {
            id: "ga.smc_gate_stagnation_step",
            section: "Discovery / GA search",
            label: "SMC gate stagnation-relax step",
            env_var: Some("NEOETHOS_BOT_PROP_SMC_GATE_STAGNATION_STEP"),
            kind: KnobKind::Float { min: Some(0.0), max: Some(0.5) },
            default: "0.03",
            current: format!("{}", smc.stagnation_step),
            help_short: "How much the SMC gate relaxes per stagnant generation (after patience exceeded).",
            help_long: "Lower = gate stays strict during stagnation (good for selective searches). Higher = gate opens up faster to escape local optima.",
            preset_conservative: "0.01",
            preset_balanced: "0.03",
            preset_aggressive: "0.05",
        },
        KnobEntry {
            id: "ga.archive_mode",
            section: "Discovery / GA search",
            label: "Archive scoring mode",
            env_var: Some("NEOETHOS_BOT_PROP_ARCHIVE_MODE"),
            kind: KnobKind::Enum { variants: &["net", "pf", "sharpe"] },
            default: "net",
            current: ga_overrides.archive_scoring.mode.clone(),
            help_short: "Which metric gates archive admission: net P&L, profit factor, or Sharpe.",
            help_long: "`net` is the default-stable choice. `pf` is more tolerant of low-trade strategies. `sharpe` rewards consistency over total return.",
            preset_conservative: "net",
            preset_balanced: "net",
            preset_aggressive: "pf",
        },
        KnobEntry {
            id: "ga.archive_min_net",
            section: "Discovery / GA search",
            label: "Archive min net P&L",
            env_var: Some("NEOETHOS_BOT_PROP_ARCHIVE_MIN_NET"),
            kind: KnobKind::Float { min: None, max: None },
            default: "0.0",
            current: format!("{}", ga_overrides.archive_scoring.min_net),
            help_short: "Floor for net P&L below which strategies are NOT archived.",
            help_long: "Conservative (500): keeps only strategies that won at least $500 over the in-sample window. Balanced (0): admits any net-positive. Aggressive (negative): includes near-break-even strategies for further inspection.",
            preset_conservative: "500.0",
            preset_balanced: "0.0",
            preset_aggressive: "-100.0",
        },
        KnobEntry {
            id: "ga.archive_min_pf",
            section: "Discovery / GA search",
            label: "Archive min profit factor",
            env_var: Some("NEOETHOS_BOT_PROP_ARCHIVE_MIN_PF"),
            kind: KnobKind::Float { min: Some(0.0), max: Some(10.0) },
            default: "1.0",
            current: format!("{}", ga_overrides.archive_scoring.min_pf),
            help_short: "Floor for profit factor (gross_win / gross_loss) below which strategies are NOT archived.",
            help_long: "1.0 = break-even. Conservative (1.5) keeps only solid edges. Aggressive (1.1) admits more candidates.",
            preset_conservative: "1.5",
            preset_balanced: "1.0",
            preset_aggressive: "1.1",
        },
        KnobEntry {
            id: "ga.archive_min_sharpe",
            section: "Discovery / GA search",
            label: "Archive min Sharpe ratio",
            env_var: Some("NEOETHOS_BOT_PROP_ARCHIVE_MIN_SHARPE"),
            kind: KnobKind::Float { min: None, max: None },
            default: "0.0",
            current: format!("{}", ga_overrides.archive_scoring.min_sharpe),
            help_short: "Floor for Sharpe (only enforced when `archive_mode = sharpe`).",
            help_long: "Conservative (0.5) is institutional-quality consistency. Balanced (0.0) is no floor. Aggressive (-0.5) admits noisy strategies for diagnostic inspection.",
            preset_conservative: "0.5",
            preset_balanced: "0.0",
            preset_aggressive: "-0.5",
        },
        KnobEntry {
            id: "ga.parent_selection",
            section: "Discovery / GA search",
            label: "Parent selection policy",
            env_var: Some("NEOETHOS_BOT_PROP_PARENT_SELECTION"),
            kind: KnobKind::Enum {
                variants: &["rank_weighted", "tournament", "truncation"],
            },
            default: "rank_weighted",
            current: format!("{:?}", ga_overrides.selection.parent).to_lowercase(),
            help_short: "How the GA picks parents for crossover.",
            help_long: "`rank_weighted` (default) is stable + diverse. `tournament` is faster on large populations. `truncation` is deterministic top-K (most aggressive).",
            preset_conservative: "rank_weighted",
            preset_balanced: "rank_weighted",
            preset_aggressive: "tournament",
        },
        KnobEntry {
            id: "ga.survivor_selection",
            section: "Discovery / GA search",
            label: "Survivor selection policy",
            env_var: Some("NEOETHOS_BOT_PROP_SURVIVOR_SELECTION"),
            kind: KnobKind::Enum {
                variants: &["rank_weighted", "tournament", "truncation"],
            },
            default: "rank_weighted",
            current: format!("{:?}", ga_overrides.selection.survivor).to_lowercase(),
            help_short: "How the GA picks survivors for the next generation.",
            help_long: "Mirror of `parent_selection`. Mixing policies (e.g. tournament parents + truncation survivors) is permitted but rarely useful.",
            preset_conservative: "rank_weighted",
            preset_balanced: "rank_weighted",
            preset_aggressive: "truncation",
        },
        KnobEntry {
            id: "ga.random_immigrants",
            section: "Discovery / GA search",
            label: "Random immigrants ratio",
            env_var: Some("NEOETHOS_BOT_PROP_RANDOM_IMMIGRANTS"),
            kind: KnobKind::Float { min: Some(0.0), max: Some(0.95) },
            default: "0.25",
            current: format!("{}", ga_overrides.selection.immigrant_ratio),
            help_short: "Fraction of each generation replaced with fresh random genes (diversity injection).",
            help_long: "Higher (0.4) = more exploration. Lower (0.1) = more exploitation of existing genes. 0.25 is the audit baseline.",
            preset_conservative: "0.10",
            preset_balanced: "0.25",
            preset_aggressive: "0.40",
        },
        KnobEntry {
            id: "ga.survivor_fraction",
            section: "Discovery / GA search",
            label: "Survivor fraction (elite carry-over)",
            env_var: Some("NEOETHOS_BOT_PROP_SURVIVOR_FRACTION"),
            kind: KnobKind::Float { min: Some(0.0), max: Some(0.95) },
            default: "0.10",
            current: format!("{}", ga_overrides.selection.survivor_fraction),
            help_short: "Fraction of top genes carried unchanged to the next generation.",
            help_long: "Conservative (0.20) preserves more elites = slower convergence but safer. Aggressive (0.05) recycles more = faster turnover but riskier.",
            preset_conservative: "0.20",
            preset_balanced: "0.10",
            preset_aggressive: "0.05",
        },
        KnobEntry {
            id: "ga.selection_temperature",
            section: "Discovery / GA search",
            label: "Selection temperature",
            env_var: Some("NEOETHOS_BOT_PROP_SELECTION_TEMPERATURE"),
            kind: KnobKind::Float { min: Some(0.001), max: Some(10.0) },
            default: "0.75",
            current: format!("{}", ga_overrides.selection.temperature),
            help_short: "Softness of the selection probability distribution.",
            help_long: "Lower (<0.5) = more random (every gene has a similar chance). Higher (>1.0) = more deterministic (top genes dominate).",
            preset_conservative: "0.5",
            preset_balanced: "0.75",
            preset_aggressive: "1.5",
        },

        // ── Section 6 — Backtest runtime ───────────────────────────────
        KnobEntry {
            id: "backtest.initial_equity",
            section: "Backtest runtime",
            label: "Initial equity",
            env_var: Some("NEOETHOS_BOT_BACKTEST_INITIAL_EQUITY"),
            kind: KnobKind::Float { min: Some(100.0), max: Some(10_000_000.0) },
            default: "100000.0",
            current: format!("{}", neoethos_search::current_backtest_runtime_overrides().initial_equity),
            help_short: "Starting equity for the backtest simulation, independent of any live account.",
            help_long: "100k USD is the prop-firm baseline (most challenges fund at $100k). Use 10k for a smaller-account stress test, or 1M to see how compounding scales.",
            preset_conservative: "100000.0",
            preset_balanced: "100000.0",
            preset_aggressive: "100000.0",
        },
        KnobEntry {
            id: "backtest.max_month_buckets",
            section: "Backtest runtime",
            label: "Max month buckets",
            env_var: Some("NEOETHOS_BOT_BACKTEST_MAX_MONTH_BUCKETS"),
            kind: KnobKind::Int { min: Some(12), max: Some(1200) },
            default: "240",
            current: format!("{}", neoethos_search::current_backtest_runtime_overrides().month_capacity),
            help_short: "Cap on per-month statistics buckets (240 = 20 years).",
            help_long: "Caps RAM on very-long history runs. Default 240 covers a 20-year sweep; raise to 600 for 50-year MT5 historical imports.",
            preset_conservative: "240",
            preset_balanced: "240",
            preset_aggressive: "600",
        },
        KnobEntry {
            id: "backtest.rayon_threads",
            section: "Backtest runtime",
            label: "Rayon worker threads",
            env_var: Some("NEOETHOS_BOT_RUST_THREADS"),
            kind: KnobKind::Int { min: Some(1), max: Some(256) },
            default: "(num CPU cores)",
            current: neoethos_search::current_backtest_runtime_overrides()
                .rayon_threads
                .map(|n| n.to_string())
                .unwrap_or_else(|| "(num CPU cores)".to_string()),
            help_short: "Rayon worker thread count (also honours `RAYON_NUM_THREADS`).",
            help_long: "Default uses all logical cores. Lower (cpu_cores - 2) to keep CPU available for other tasks (Conservative). Pin to your physical-core count for prop-firm validation runs (no hyperthread contention).",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },

        // ── Section 7 — Logging / server ─────────────────────────────
        KnobEntry {
            id: "log.rust_log",
            section: "Logging / persistence",
            label: "RUST_LOG filter",
            env_var: Some("RUST_LOG"),
            kind: KnobKind::Text,
            default: "(production default from Settings)",
            current: "(see startup banner)".to_string(),
            help_short: "tracing-subscriber filter (e.g. `info,sqlx=warn`).",
            help_long: "Lower to `error,neoethos=info` for quieter logs; raise to `debug` for diagnostics. The production default ships in `Settings.system.log_filter`.",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "log.log_dir",
            section: "Logging / persistence",
            label: "Log directory override",
            env_var: Some("LOG_DIR"),
            kind: KnobKind::Path,
            default: "(platform default — %APPDATA%/neoethos/logs on Windows)",
            current: "(platform default)".to_string(),
            help_short: "Override the on-disk log directory.",
            help_long: "By default logs go under the platform user-data-dir. Override to redirect to a fast SSD, a network share, or a tmpfs for ephemeral runs.",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
        KnobEntry {
            id: "server.bind_addr",
            section: "Server / network",
            label: "HTTP server bind address",
            env_var: Some("NEOETHOS_SERVER_BIND"),
            kind: KnobKind::Text,
            default: "127.0.0.1:7423",
            current: "127.0.0.1:7423".to_string(),
            help_short: "host:port for the backend HTTP server.",
            help_long: "Default binds to localhost only (the Flutter front-end on the same machine). Override to `0.0.0.0:7423` to expose to LAN, or change the port if 7423 conflicts with another app. Both sides (backend + Flutter `backend_client.dart`) must agree.",
            preset_conservative: "",
            preset_balanced: "",
            preset_aggressive: "",
        },
    ]
}

/// `GET /settings/knob-catalog`
pub async fn get_knob_catalog(State(_state): State<AppApiState>) -> Response {
    let response = KnobCatalogResponse {
        schema_version: SCHEMA_VERSION,
        generated_at_unix_ms: chrono::Utc::now().timestamp_millis(),
        knobs: build_catalog(),
    };
    Json(response).into_response()
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PresetSummary {
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PresetsResponse {
    pub presets: Vec<PresetSummary>,
}

/// `GET /settings/presets` — returns the three safety presets the UI
/// surfaces as one-click switches.
pub async fn get_presets(State(_state): State<AppApiState>) -> Response {
    let response = PresetsResponse {
        presets: vec![
            PresetSummary {
                id: "conservative",
                label: "Conservative",
                description:
                    "Capital-preservation defaults. 0.5% risk/trade, strict SMC gate, \
                     tight PnL circuit breaker, fewer cTrader retries. Best for \
                     prop-firm passing and new operators. Risky Mode disabled.",
            },
            PresetSummary {
                id: "balanced",
                label: "Balanced",
                description:
                    "Production-recommended defaults. 1% risk/trade, default SMC gate, \
                     1% PnL circuit breaker. Best for funded accounts and multi-month \
                     campaigns. Risky Mode disabled.",
            },
            PresetSummary {
                id: "aggressive",
                label: "Aggressive (advanced)",
                description:
                    "Higher risk, permissive filters. 2% risk/trade, relaxed SMC gate, \
                     5% PnL circuit breaker. Risky Mode AVAILABLE but requires the \
                     signed §6.4 acknowledgement (99% ruin probability ceiling). Only \
                     for operators who understand Kelly mathematics and run a separate \
                     prop-firm-passing account on the Conservative preset.",
            },
        ],
    };
    Json(response).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_non_empty_and_ids_unique() {
        let catalog = build_catalog();
        assert!(!catalog.is_empty(), "knob catalog should not be empty");
        let mut ids: Vec<&str> = catalog.iter().map(|k| k.id).collect();
        ids.sort();
        let len_before = ids.len();
        ids.dedup();
        assert_eq!(
            len_before,
            ids.len(),
            "knob catalog ids must be unique — a duplicate would clobber \
             the Flutter UI's per-id state."
        );
    }

    #[test]
    fn catalog_serializes_to_json_cleanly() {
        let catalog = build_catalog();
        let response = KnobCatalogResponse {
            schema_version: SCHEMA_VERSION,
            generated_at_unix_ms: 0,
            knobs: catalog,
        };
        let json =
            serde_json::to_string(&response).expect("catalog must serialize without error");
        assert!(json.contains("\"schemaVersion\":1"));
        assert!(json.contains("\"id\":\"ctrader.max_attempts\""));
    }
}
