// Core configuration structures for Forex trading system
// Project configuration loader.

use crate::contracts::CANONICAL_TIMEFRAMES;
use crate::domain::prop_firm::{PropFirmConstraints, PropFirmPreset, PropFirmRuntimeDefaults};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Serialize a `HashMap` in SORTED key order (audit M06/M07 follow-up).
/// HashMap iteration order is randomized per process, so `Settings::save`
/// reshuffled these config maps on every write — dirtying config.yaml in git
/// with no real change. Sorting on serialize makes two saves of equivalent
/// settings byte-identical. Lookup semantics (the public API) are unchanged.
fn serialize_sorted_map<S, V>(map: &HashMap<String, V>, ser: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
    V: Serialize,
{
    let sorted: std::collections::BTreeMap<&String, &V> = map.iter().collect();
    serde::Serialize::serialize(&sorted, ser)
}

/// Sorted serialization for a nested map (both levels ordered).
fn serialize_sorted_nested_map<S>(
    map: &HashMap<String, HashMap<String, String>>,
    ser: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let sorted: std::collections::BTreeMap<&String, std::collections::BTreeMap<&String, &String>> =
        map.iter().map(|(k, v)| (k, v.iter().collect())).collect();
    serde::Serialize::serialize(&sorted, ser)
}
use std::path::PathBuf;

/// Public, no-API-key financial NEWS RSS feeds for the AI news desk
/// (`GET /news/feed`). Verified reachable 2026-06-30 (HTTP 200 + XML). The
/// economic *calendar* is separate (`news_calendar_source`) — ForexFactory's
/// ffcal XML is a calendar format, not RSS, so it does NOT belong here. Used
/// both as the default and as the runtime fallback when a user's configured
/// feeds are all unreachable, so a stale config never leaves the desk blank.
pub fn default_news_rss_feeds() -> Vec<String> {
    vec![
        "https://www.investing.com/rss/news.rss".to_string(),
        "https://www.fxstreet.com/rss/news".to_string(),
        "https://www.cnbc.com/id/100003114/device/rss/rss.html".to_string(),
    ]
}

/// The one economic-calendar provider that is actually implemented
/// (`app_services::news_calendar` fetches ForexFactory's `ff_calendar` JSON).
///
/// `news_calendar_source` used to be validated for non-emptiness, persisted,
/// echoed back by `GET /settings`, and offered as a free-text box in
/// Advanced → News — while the fetcher hardcoded the ForexFactory URL and
/// never read the field. Typing `investing` there saved successfully and
/// changed nothing. Both the write path and the fetch path now check against
/// this list, so an unsupported provider fails loudly instead of pretending.
pub const NEWS_CALENDAR_FOREXFACTORY: &str = "forexfactory";

/// Every calendar provider id the runtime can actually serve.
pub const SUPPORTED_NEWS_CALENDAR_SOURCES: &[&str] = &[NEWS_CALENDAR_FOREXFACTORY];

/// Normalise + validate an economic-calendar provider id.
///
/// `Ok(canonical_id)` when the runtime has an implementation for it;
/// `Err(message)` — phrased for direct display to the operator — otherwise.
pub fn validate_news_calendar_source(raw: &str) -> Result<String, String> {
    let id = raw.trim().to_ascii_lowercase();
    if id.is_empty() {
        return Err("news_calendar_source cannot be blank".to_string());
    }
    if SUPPORTED_NEWS_CALENDAR_SOURCES.contains(&id.as_str()) {
        Ok(id)
    } else {
        Err(format!(
            "unknown news_calendar_source `{raw}`. This build implements only: {}. \
             Setting any other value would be ignored — the calendar fetcher has no \
             implementation for it — so it is rejected instead of silently accepted.",
            SUPPORTED_NEWS_CALENDAR_SOURCES.join(", ")
        ))
    }
}

/// System-level configuration
///
/// **Sealed against a second load path.** `remote = "Self"` makes the derive
/// emit an *inherent* `SystemConfig::deserialize` instead of an
/// `impl Deserialize for SystemConfig`, so this type cannot be reached by
/// `serde_*::from_str::<SystemConfig>` nor by a `#[derive(Deserialize)]`
/// struct that merely holds one. See the SUB-STRUCT SEAL block further down
/// this file, above the hand-written `Serialize` impls.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(remote = "Self", default, deny_unknown_fields)]
pub struct SystemConfig {
    pub symbol: String,
    /// Market Watch live-tick subscription set (F-338). Empty → the spot
    /// streamer falls back to `DEFAULT_STREAMED_SYMBOLS` (the 8 majors).
    /// The operator edits this from Market Watch; it takes effect on the
    /// next backend start.
    #[serde(default)]
    pub watchlist: Vec<String>,
    /// Operator's account currency for cost-model FX conversions
    /// (commission, swap, pnl_conversion_fee → account ccy) and the
    /// risk-gate sizing math.
    ///
    /// Population paths:
    ///  1. Manual via `config.yaml` `system.account_currency: "USD"`.
    ///  2. Auto from the cTrader trader profile when the broker session
    ///     is alive — the `/account/snapshot` bridge resolves
    ///     `ProtoOATrader.depositAssetId` → currency name via the
    ///     asset table and writes it back here (Phase D follow-up).
    ///
    /// (The legacy `NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY` env fallback was
    /// retired in v0.4.36 — config is the single source.)
    ///
    /// **Empty string (`""`) is the deliberate fail-loud default**
    /// matching the `symbol` field's policy — `DiscoveryConfig::
    /// from_settings()` propagates it to `evaluation_account_currency`
    /// and the cost-model NaN-sentinel guard then rejects backtests
    /// rather than silently lying about commission/swap values.
    /// Operators must populate before running discovery.
    #[serde(default)]
    pub account_currency: String,
    pub data_dir: PathBuf,
    /// Top-level **trading mode** — the single master switch the operator picks
    /// in the Risk screen. Two mutually-exclusive values:
    ///   - `"risky"`     → aggressive capital multiplication (small balance →
    ///     large target, ASAP). Drives discovery into `DiscoveryMode::Risky`
    ///     (high-risk filter floors, growth-tilted ranking, no prop-firm gate).
    ///   - `"prop_firm"` → safety / stability: pass prop-firm challenges and
    ///     bank a steady monthly return. Drives `DiscoveryMode::PropFirm` (FTMO
    ///     window-pass gate) and the active `risk.preset` constraints.
    /// Search/discovery + risk framing orient around this one choice. An
    /// explicit `models.discovery_mode = "strict"` is a power-user escape hatch
    /// that overrides the discovery side only. Default `"prop_firm"`.
    pub trading_mode: String,
    /// Risky-Mode goal — capital multiplication. The operator sets where to
    /// start, where to reach, and by when; in Risky mode these PRESSURE the
    /// strategy search to surface portfolios that can compound from start to
    /// target within the horizon (see `DiscoveryConfig::risky_*` + the
    /// target-aware candidate ranking). Sizing is a fraction of the *current*
    /// balance, so risk compounds with the bankroll. Defaults 100 -> 50,000 in
    /// 180 days (~6 months — beyond that it is closer to normal trading, per
    /// the operator's Risky-vs-normal distinction). Fully operator-editable.
    pub risky_start_balance_usd: f64,
    pub risky_target_balance_usd: f64,
    pub risky_horizon_days: u32,
    /// When auto-cull permanently retires a live strategy, automatically
    /// queue a fresh Discovery run on the same symbol + base timeframe to
    /// refill the gap (the retired strategy itself can never come back —
    /// its fingerprint stays blacklisted). The Symbiotic-GP retraining-trigger
    /// loop (2026-07-02). Default ON; toggle in Settings.
    pub auto_rediscover_on_cull: bool,
    pub multi_resolution_enabled: bool,
    pub multi_resolution_timeframes: Vec<String>,
    pub multi_resolution_prefix_base: bool,
    pub base_timeframe: String,
    pub higher_timeframes: Vec<String>,
    pub poll_interval_seconds: u64,
    pub metrics_db_path: PathBuf,
    pub cache_dir: PathBuf,
    pub enable_gpu_preference: String,
    // agent 2026-06-05 overfitting fix: removed three dead `discovery_*` fields
    // (`discovery_auto_cap` / `discovery_max_rows` / `discovery_stream`). They
    // were never read anywhere in the workspace — the REAL discovery row cap is
    // `models.prop_search_max_rows` (→ DiscoveryConfig.max_rows, discovery.rs).
    //
    // CORRECTED 2026-08-10: this block used to end "SystemConfig does NOT derive
    // `#[serde(deny_unknown_fields)]`, so any stale copies of these keys are
    // ignored, not errors." It does now, and that sentence was the whole defect
    // written down as a feature — the same permissiveness accepted
    // `trailing_enabeld:` and reported it saved. The three keys are listed in
    // `load_seal::RETIRED_KEYS`, so a file that still carries them loads with
    // each one NAMED at WARN; anything not on that list is refused.
    pub enable_gpu: bool,
    /// WARNING DERIVED FROM HARDWARE - NOT AN INPUT. See `n_jobs` above.
    /// `num_gpus: 0` in the operator's live store, on a box with a 3090, is
    /// the same frozen detector output. `#[serde(skip)]` 2026-08-10.
    #[serde(skip)]
    pub num_gpus: usize,
    pub device: String,
    pub max_training_rows_per_tf: usize,
    /// Hardware / accelerator runtime knobs. See [`HardwareConfig`].
    #[serde(default)]
    pub hardware: HardwareConfig,
}

/// Hardware / accelerator runtime knobs — the ONLY source for these settings.
///
/// These replace six env vars that used to be read by
/// `HardwareRuntimeOverrides::from_env`: `NEOETHOS_BOT_CPU_BUDGET`,
/// `NEOETHOS_BOT_TRAIN_PRECISION` (plus a legacy `FOREX_TRAIN_PRECISION`
/// alias), `NEOETHOS_BOT_{CUDA,ROCM,WGPU}_PRECISIONS` and
/// `NEOETHOS_BOT_WGPU_DEVICES`. That function was deleted on 2026-08-03 because
/// it had zero callers — so those env vars had already stopped doing anything
/// long before, while these doc comments went on pointing readers at them.
/// Setting one and watching nothing change was the intended experience of a
/// function nobody called.
///
/// All-`None`/empty defaults reproduce the historical env-absent behaviour.
///
/// CPU capacity is resolved by `ExecutionBudgetInputs` from this typed setting,
/// an optional legacy read-only cap, and an optional parent assignment. No
/// caller mutates persistent settings to carry ephemeral process capacity.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HardwareConfig {
    /// Maximum workers for every CPU-heavy workload in this process. `None` =
    /// effective logical threads minus the fixed two-thread stability reserve.
    /// An explicit value can only narrow that automatic ceiling.
    pub cpu_budget: Option<usize>,
    /// Forced training precision; `None` = auto per accelerator.
    pub training_precision: Option<crate::system::TrainingPrecision>,
    /// Per-backend precision ladders; `None` = engine defaults.
    pub cuda_precisions: Option<Vec<crate::system::TrainingPrecision>>,
    pub rocm_precisions: Option<Vec<crate::system::TrainingPrecision>>,
    pub wgpu_precisions: Option<Vec<crate::system::TrainingPrecision>>,
    /// Explicit Vulkan/WGPU device names; empty = auto-enumerate.
    pub wgpu_device_names: Vec<String>,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            // F-129 fix (2026-05-25): the previous defaults hardcoded
            // `symbol = "EURUSD"` + `symbols = vec!["EURUSD"]`. Both
            // are synthetic-data violations per the operator's
            // real-data directive 2026-05-24. Empty defaults force
            // the loader / caller to populate from real `config.yaml`
            // (which is the production path — `SystemConfig::default()`
            // is only the seed for serde defaults). Any production code
            // that runs against the all-empty default will hit the
            // downstream guard that rejects empty-symbol orders
            // (see `risk_gate::prop_firm_pre_trade_check` Batch B Pass 3).
            symbol: String::new(),
            watchlist: Vec::new(),
            // F-304 fix (2026-05-28): empty default forces operator/
            // broker-session population, matching `symbol`. The
            // cost-model NaN-sentinel guard rejects empty values
            // downstream, so a bare-install run fails LOUD instead of
            // silently producing zero-trade GA results from a NaN pip
            // value.
            account_currency: String::new(),
            data_dir: PathBuf::from("data"),
            trading_mode: "prop_firm".to_string(),
            risky_start_balance_usd: 100.0,
            risky_target_balance_usd: 50000.0,
            risky_horizon_days: 180,
            auto_rediscover_on_cull: true,
            multi_resolution_enabled: true,
            multi_resolution_timeframes: CANONICAL_TIMEFRAMES
                .iter()
                .map(|tf| (*tf).to_string())
                .collect(),
            multi_resolution_prefix_base: false,
            base_timeframe: "M1".to_string(),
            higher_timeframes: CANONICAL_TIMEFRAMES
                .iter()
                .map(|tf| (*tf).to_string())
                .collect(),
            poll_interval_seconds: 60,
            metrics_db_path: PathBuf::from("metrics.sqlite"),
            cache_dir: PathBuf::from("cache"),
            enable_gpu_preference: "auto".to_string(),
            // agent 2026-06-05 overfitting fix: dead `discovery_*` fields removed
            // (see struct decl). The real row cap is `models.prop_search_max_rows`.
            enable_gpu: false,
            num_gpus: 0,
            device: "cpu".to_string(),
            max_training_rows_per_tf: 0,
            hardware: HardwareConfig::default(),
        }
    }
}

impl SystemConfig {
    /// Resolve the effective **base timeframe** from config.
    ///
    /// THE single source of truth shared by BOTH the CLI (`default_base_tf`)
    /// and the app server (`/engines/*/start`) so the two never diverge.
    /// Operator mandate (2026-06-04): the bot must behave identically whether
    /// driven from the UI or the CLI — no difference anywhere.
    pub fn resolve_base_timeframe(&self) -> String {
        self.base_timeframe.trim().to_string()
    }

    /// Resolve the effective **symbol** from config (shared by CLI + server).
    pub fn resolve_symbol(&self) -> String {
        self.symbol.trim().to_string()
    }

    /// Resolve the effective **higher timeframes** for an already-resolved
    /// `base`, honouring `multi_resolution_enabled` / `multi_resolution_timeframes`
    /// / `higher_timeframes` exactly. SHARED by CLI + server.
    ///
    /// - When multi-resolution is on and a non-empty explicit list is set, that
    ///   list wins (minus any entry equal to `base`).
    /// - Otherwise the configured `higher_timeframes` are filtered to those
    ///   strictly *above* `base` in canonical order (never a lower/equal TF).
    ///
    /// The filter is relative to the **effective** `base` passed in (which may be
    /// a CLI `--base` / payload override), not necessarily `self.base_timeframe`
    /// — so an overridden base always gets the correct top-down ladder above it.
    pub fn resolve_higher_timeframes(&self, base: &str) -> Vec<String> {
        let base_trim = base.trim();
        if self.multi_resolution_enabled && !self.multi_resolution_timeframes.is_empty() {
            self.multi_resolution_timeframes
                .iter()
                .map(|tf| tf.trim().to_string())
                .filter(|tf| !tf.is_empty() && !tf.eq_ignore_ascii_case(base_trim))
                .collect()
        } else {
            let above = crate::contracts::canonical_higher_timeframes(base_trim);
            self.higher_timeframes
                .iter()
                .map(|tf| tf.trim().to_string())
                .filter(|tf| !tf.is_empty() && above.iter().any(|a| a.eq_ignore_ascii_case(tf)))
                .collect()
        }
    }
}

/// Risk management configuration
///
/// **Sealed against a second load path — and this is the one that was proved to
/// be a MONEY divergence.** The same bytes `risk: {preset: the5ers}` gave
/// `daily_drawdown_limit` 0.032 through [`Settings`] (which re-derives the
/// preset seeds, see `reconcile_preset`) and 0.040 through a
/// `#[derive(Deserialize)] struct Bypass { risk: RiskConfig }` that never runs
/// it — total 0.042 vs 0.070. **The bypass got the LOOSER limit under the
/// correct firm label.** `remote = "Self"` makes the derive emit an *inherent*
/// `RiskConfig::deserialize` instead of an `impl Deserialize for RiskConfig`,
/// so that bypass no longer compiles:
///
/// ```compile_fail
/// // The verifier's own bypass. It must NOT compile.
/// #[derive(serde::Deserialize)]
/// struct Bypass {
///     risk: neoethos_core::config::RiskConfig,
/// }
/// ```
///
/// The control: a `#[derive(Deserialize)]` container is otherwise perfectly
/// legal here, so the block above fails for the seal and not for the
/// environment.
///
/// ```
/// #[derive(serde::Deserialize)]
/// struct Control {
///     a: String,
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(remote = "Self", default, deny_unknown_fields)]
pub struct RiskConfig {
    /// Named prop-firm preset that seeds every other field in this
    /// struct. The runtime is firm-agnostic; this field just selects
    /// which lookup table populates the numeric thresholds at default
    /// construction. Operators can override any field below — preset
    /// values are seeds, not locks. Setting `preset: none` disables
    /// the external-challenge gate without touching the other fields.
    #[serde(default)]
    pub preset: PropFirmPreset,
    pub initial_balance: f64,
    /// WARNING UNWIRED - `RiskManager` has no production constructor; every
    /// `RiskManager::new` in the workspace is inside its own test module. This
    /// is RETAINED AS INTENT (it records the operator's monthly floor and is
    /// seeded from the active preset) but no live decision reads it.
    /// See `tests/config_has_recipient.rs::UNWIRED`.
    pub monthly_profit_target_pct: f64,
    pub min_risk_per_trade: f64,
    pub max_risk_per_trade: f64,
    pub risk_per_trade: f64,
    /// PER-MODE risk band. `None` (the default, and what every pre-2026-07-21
    /// config has) falls back to the shared `min/max_risk_per_trade` above, so
    /// existing setups are untouched.
    ///
    /// Why this exists: Risky and Prop-firm are two *different products* that
    /// happen to share one engine — aggressive compounding vs surviving a
    /// challenge whose daily-loss rule is a few percent. A single shared risk
    /// band silently carried one mode's sizing into the other: switching
    /// `system.trading_mode` to `prop_firm` while the band still said 30% made
    /// every candidate break the firm's daily rule on its first loss, so the
    /// search could never return anything — with nothing on screen explaining
    /// why. Set these once and each mode keeps its own sizing forever.
    #[serde(default)]
    pub risky_min_risk_per_trade: Option<f64>,
    #[serde(default)]
    pub risky_max_risk_per_trade: Option<f64>,
    #[serde(default)]
    pub prop_firm_min_risk_per_trade: Option<f64>,
    #[serde(default)]
    pub prop_firm_max_risk_per_trade: Option<f64>,
    /// Portfolio-level cap on TOTAL concurrent risk across all running live
    /// engines, as a balance fraction (e.g. 0.05 = at most ~5% of the account
    /// at risk across every open autopilot position at once). Each engine
    /// budgets its entry against `cap − (open positions × risk_per_trade)`,
    /// sizing down or skipping when the budget is spent. `0.0` disables the
    /// cap (per-engine sizing only — the pre-2026-07 behavior).
    pub max_portfolio_risk: f64,
    pub daily_drawdown_limit: f64,
    pub total_drawdown_limit: f64,
    pub min_risk_reward: f64,
    pub max_lot_size: f64,
    /// Manual-order authority: when `true`, `POST /orders` and
    /// `POST /orders/pending` REFUSE an order with no `stopLossPips`, and the
    /// body's `risky: true` flag does not override it.
    ///
    /// Wired 2026-08-09 (W1) in `neoethos-app/src/server/orders.rs`. Before
    /// that it was displayed in the Settings knob catalog and echoed by
    /// `GET /risk` and read by nothing; the `config_has_recipient` guard passed
    /// it on `RiskDto`'s same-named field.
    ///
    /// Does NOT affect order SIZE on the manual path — by operator decision
    /// there is no `max_lot_size` clamp there. It also does not affect the
    /// autopilot, which always places a bracket (the gene's, or the kernel's
    /// 20/40-pip defaults).
    pub require_stop_loss: bool,
    /// WARNING UNWIRED - `RiskManager` has no production constructor, so
    /// nothing reads this. Both repo YAMLs ship `challenge_mode: true` against
    /// a `Default` of `false`: a mode that does not exist has been deliberately
    /// armed. RETAINED AS INTENT; do not read the `true` as an active regime.
    pub challenge_mode: bool,
    /// ⚠ UNWIRED — nothing reads this field.
    ///
    /// The mechanism it was written for exists:
    /// `PropFirmPhaseRiskDefaults::for_preset(preset, challenge_phase)` in
    /// `domain/prop_firm.rs` takes exactly this string and returns per-phase
    /// risk defaults. Its only callers are that module's own `#[cfg(test)]`
    /// block. Wiring the two together would change live sizing, so it is NOT
    /// done silently here — see `tests/config_has_recipient.rs::UNWIRED`.
    pub challenge_phase: String,
    // `prop_firm_rules: bool` DELETED 2026-08-10 (knob-second-pass D6). It was
    // literally `preset != PropFirmPreset::None` — one write from the Risk
    // preset dropdown, one read into the display DTO, and ZERO decisions:
    // every discovery call passes a hardcoded `PropFirmRiskRules::default()`.
    // Because it tracked the PRESET while the engine takes its regime from
    // `system.trading_mode`, the card could announce "Prop-firm rules: ON"
    // during a risky run. The display is now derived from `system.trading_mode`
    // (`neoethos-app/src/server/risk.rs::derive_prop_firm_rules_active`), which
    // is what the engine actually reads. The key is in `RETIRED_KEYS`, so a
    // live store still carrying it loads with the key NAMED at WARN.
    pub kill_zones_enabled: bool,
    /// Daily entry cap, counted per ACCOUNT — see the arming flag below.
    pub max_trades_per_day: usize,
    /// Arms live enforcement of `max_trades_per_day`. Default `false`, so
    /// today's behaviour is unchanged until the operator flips it.
    ///
    /// When `true`, the cap binds ACCOUNT-WIDE per UTC day: one counter
    /// (`domain::daily_entry_cap`, held in a `static` by
    /// `live_trading.rs`) is shared by EVERY running engine — four engines do
    /// NOT get 8 each; the account gets 8 total, which is what an operator
    /// reading "8" expects. Every refusal logs the rule, the day's count and
    /// the cap. The counter resets at UTC midnight and on app restart, and
    /// `max_trades_per_day: 0` disables the cap like the other risk caps.
    #[serde(default)]
    pub max_trades_per_day_enabled: bool,
    /// ⚠ UNWIRED — nothing reads this field; setting it `false` does NOT
    /// disable recovery mode.
    ///
    /// `RiskManager::update_recovery_state` (domain/risk.rs) flips
    /// `RiskManager.recovery_mode` purely from the drawdown vs
    /// `daily_dd_warning_pct`, consulting no operator toggle. `RiskManager`
    /// has no production constructor at all — every `RiskManager::new` call
    /// in the workspace is inside its own test module — so there is no live
    /// call site to wire this into. See
    /// `tests/config_has_recipient.rs::UNWIRED`.
    pub recovery_mode_enabled: bool,
    pub feature_drift_threshold: f64,
    pub high_quality_confidence: f64,
    pub atr_period: usize,
    pub atr_stop_multiplier: f64,
    pub triple_barrier_max_bars: usize,
    // ---------------------------------------------------------------------
    // `trailing_enabled` / `trailing_atr_multiplier` / `trailing_be_trigger_r`
    // / `trailing_min_lock_pips` DELETED HERE 2026-08-10 (audit #206).
    //
    // They were shadowed duplicates of `models.exit_policy.*`: the search has
    // always read that copy (`strategy_gene.rs`), so these four reached no
    // evaluator, CPU or CUDA, while the operator's live store set them —
    // including a hand-tuned `trailing_atr_multiplier: 0.4` and
    // `trailing_be_trigger_r: 0.1` that moved nothing.
    //
    // The documented precondition for deleting them is now MET. Until today
    // live execution trailed unconditionally with no config recipient, so
    // removing the visible-but-dead keys would have turned a wrong value into an
    // invisible hardcode on the path that spends real money — the required order
    // was "wire live to `models.exit_policy` FIRST, then delete". Live reads it:
    // `live_trading.rs:747-782` resolves the policy and logs which branch it
    // took, and `:1479-1493` applies `trailing_be_trigger_r` /
    // `trailing_stop_multiplier` / `trailing_min_lock_pips` from it.
    //
    // All four are in `RETIRED_KEYS`, so a store that still carries them loads
    // with each key NAMED at WARN and the rename to `models.exit_policy` spelled
    // out (`trailing_atr_multiplier` → `trailing_stop_multiplier`; despite the
    // old name it was never an ATR multiple).
    // ---------------------------------------------------------------------
    /// Adverse slippage assumption in pips **per fill**. Broad flat-cost
    /// screening charges it once at entry and once at exit. This is an operator
    /// assumption, not historical Bid/Ask or broker-deal evidence.
    pub slippage_pips: f64,
    pub commission_per_lot: f64,
    /// Is `commission_per_lot` the charge for ONE SIDE, or for the round trip?
    ///
    /// Brokers quote it per side. A cTrader FX account at 45 USD per million
    /// per side is about **0.62 pips per side** on EURUSD, so **1.24 pips round
    /// trip** before any spread. Every evaluator in this workspace subtracts
    /// `commission_per_trade` exactly ONCE per closed trade (CPU `eval.rs`, the
    /// CUDA kernel `prototype_b_population.cu`, the C prototype) — so with a
    /// per-side number in the field the backtest charged HALF the commission a
    /// live fill pays, on every trade, forever.
    ///
    /// `true` (the default, and the correct reading of a broker schedule) makes
    /// the resolvers double the number once, at the two boundaries named in
    /// `neoethos_search::genetic::strategy_gene::round_trip_commission_per_lot`,
    /// so that everything downstream of those boundaries means ROUND TRIP and
    /// the single subtraction is right. Set `false` only if the number you put
    /// in `commission_per_lot` is already the round trip.
    ///
    /// **Behaviour change (2026-08-09).** At the shipped `commission_per_lot:
    /// 7.0` this raises the charged commission from $7 to $14 per lot per
    /// closed trade — about 1.4 pips on a EURUSD standard lot instead of 0.7.
    /// It refuses nothing new by itself; it makes every net figure smaller and
    /// truer. It has ZERO expected value in money: charging the real cost does
    /// not create edge, it stops the search from selecting on a subsidy.
    pub commission_per_lot_is_per_side: bool,
    pub backtest_spread_pips: f64,
    /// Session-aware backtest spread, in pips, for the three UTC buckets the
    /// evaluator resolves per bar (`eval::SessionSpreadProfile::spread_pips_at`):
    /// Asian = hours 22–07, Overlap (London/NY) = 07–16, Late NY = 16–22.
    ///
    /// The mechanism has existed on both the CPU path (`eval.rs:843`) and the
    /// CUDA kernel (`prototype_b_population.cu:47 spread_pips_for_bar`) for
    /// months, and **it was never populated outside `#[cfg(test)]`**: every
    /// production construction site left `session_spread_profile: None`, so a
    /// flat `backtest_spread_pips` was charged at 03:00 Tokyo and at the London
    /// open alike. Setting all three of these keys is what turns the curve on.
    ///
    /// All three must be set together, or none. A partial setting is a config
    /// ERROR, not a silent fall-back to flat — half a curve is a cost model
    /// nobody can reason about. `slippage_pips` is added to each bucket exactly
    /// as it is added to `backtest_spread_pips`.
    ///
    /// Left unset by default because this project does not invent broker
    /// numbers (see the F-301 fail-loud note on the synthetic-spread removal).
    ///
    /// The measurement already exists: `neoethos-app`'s `spread_stats` service
    /// samples the live tick cache once a minute and accumulates per-(symbol,
    /// UTC-hour) mean and max spread into `<data_dir>/spread_stats.json`. Its
    /// module header names the eval kernels as its intended consumer — this is
    /// that consumer. Average the hourly means over 22–07, 07–16 and 16–22 and
    /// set the three keys. Until you do, the run WARNs that it is charging a
    /// flat spread.
    pub backtest_spread_pips_asian: Option<f64>,
    pub backtest_spread_pips_overlap: Option<f64>,
    pub backtest_spread_pips_late_ny: Option<f64>,
    /// Round-trip cost band, in pips, that every reported result is measured
    /// against — never a single "true cost" number.
    ///
    /// A backtest result is a function of the cost you charged it, and nobody
    /// knows their real all-in cost to better than a few tenths of a pip
    /// (spread varies by hour and by news, commission is quoted per side, and
    /// slippage is not a constant). Reporting one number invites the reader to
    /// believe it. So the screen evaluates each survivor at BOTH edges and a
    /// candidate that is profitable at `cost_band_optimistic_pips` but not at
    /// `cost_band_pessimistic_pips` is flagged `optimistic_edge_only` — which
    /// is to say: **not a result**.
    ///
    /// These are TOTAL round-trip costs (spread + commission + slippage,
    /// expressed in pips), not spreads. 1.6–2.4 is the band the 2026-08-09
    /// cost review settled on for FX majors on a retail ECN account.
    pub cost_band_optimistic_pips: f64,
    pub cost_band_pessimistic_pips: f64,
    pub conformal_enabled: bool,
    pub conformal_alpha: f64,
    pub conformal_abstain_min_set_size: usize,
    pub meta_label_max_hold_bars: usize,
    pub meta_label_min_dist: f64,
    pub meta_label_fixed_sl: f64,
    pub meta_label_fixed_tp: f64,
    pub vol_horizon_bars: usize,
}

impl Default for RiskConfig {
    fn default() -> Self {
        // Config is the single source: `config.yaml`'s `risk.preset` key drives
        // the preset (serde fills it post-construction). The legacy
        // `NEOETHOS_PROP_FIRM_PRESET` env override was retired in v0.4.36 —
        // headless deployments set `risk.preset` in config.yaml instead. The
        // default (`PropFirmPreset::default()`) is unchanged from the prior
        // env-absent behaviour, so existing config.yaml / default users are
        // unaffected; only env-only deployments must move the preset to YAML.
        let preset = PropFirmPreset::default();
        let constraints = PropFirmConstraints::for_preset(preset);
        let runtime = PropFirmRuntimeDefaults::for_preset(preset);
        Self {
            preset,
            // Account starting balance is broker-specific. Operators
            // override this via `config.yaml`'s `risk.initial_balance`.
            initial_balance: 10_000.0,
            // Monthly profit floor (operator directive 2026-05-14)
            // tracks the active preset's published target.
            monthly_profit_target_pct: constraints.monthly_profit_target(),
            min_risk_per_trade: 0.0,
            max_risk_per_trade: 0.030,
            risk_per_trade: 0.030,
            // Unset by default: each mode inherits the shared band until the
            // operator gives it its own.
            //
            // Operator decision (2026-08-09): Risky mode chases 100 -> 50k in
            // ~6 months, so its risk CEILING is 30% per trade (the "20-pip
            // challenge" style, stop 10-30 pips, target 2RR). This is a CEILING,
            // not a fixed size: the log-growth objective sizes each gene toward
            // its own growth-optimal fraction and only approaches 30% when the
            // edge justifies it — below ~43% win rate at 2RR, 30% is geometric
            // ruin, and the objective correctly declines it. The honest report
            // surfaces P(ruin) so this stays a visible bet, not a hidden one.
            risky_min_risk_per_trade: None,
            risky_max_risk_per_trade: Some(0.30),
            prop_firm_min_risk_per_trade: None,
            prop_firm_max_risk_per_trade: None,
            // Portfolio-level concurrent-risk cap. WAS 0.0 UNTIL 2026-08-10,
            // where 0 meant "disabled" — i.e. a knob named max_ shipped meaning
            // NO CAP AT ALL, on every install, chosen by nobody. Nothing about
            // that was opt-in: the operator opted into a limit by the act of
            // running a prop-firm mode, and got none.
            //
            // The daily stop is the right seed because it is the same number
            // read the other way round: if every open position stops out
            // together — the honest worst case for correlated FX pairs — the
            // day's loss IS the total open risk. `reconcile_preset` re-seeds
            // this per preset AND per trading_mode; the risky ladder gets
            // RISKY_PORTFOLIO_RISK_CAP instead, which is a tolerance for ruin
            // rather than a rulebook.
            max_portfolio_risk: runtime.daily_dd_stop_trading_pct,
            // Internal early stop sits 20% below the firm's published
            // daily-loss ceiling so a guard-rail trips before a real
            // breach. Operators override in YAML if their firm gives
            // tighter / looser tolerance.
            daily_drawdown_limit: runtime.daily_dd_stop_trading_pct,
            // Internal trailing total cap at 70% of the firm's
            // overall-drawdown ceiling for the same buffer reason.
            //
            // #269: the buffer used to be a bare `0.7` here and a private
            // `TOTAL_DRAWDOWN_BUFFER` in `neoethos-app/src/server/risk.rs`.
            // One number, one spelling — the helper on the constraints.
            total_drawdown_limit: constraints.buffered_total_drawdown_limit(),
            min_risk_reward: 2.0,
            max_lot_size: runtime.max_lot_size,
            require_stop_loss: true,
            challenge_mode: false,
            challenge_phase: "phase_1".to_string(),
            kill_zones_enabled: true,
            // Cap is preset-driven. FTMO defaults to 15; The5%ers is
            // tighter; "own money" raises it. Operators can override
            // via YAML when their style demands a different cap.
            max_trades_per_day: runtime.max_trades_per_day,
            // OFF: arming the account-wide daily entry cap is a deliberate
            // operator act, never a default — measured on the real journal a
            // cap of 8 would have refused 68.1 % of historical entries.
            max_trades_per_day_enabled: false,
            recovery_mode_enabled: true,
            feature_drift_threshold: 0.30,
            high_quality_confidence: 0.65,
            atr_period: 14,
            atr_stop_multiplier: 1.5,
            triple_barrier_max_bars: 35,
            slippage_pips: 0.5,
            commission_per_lot: 7.0,
            // Per side — that is how a broker quotes it, and how the number 7.0
            // was obtained. See the field doc for the arithmetic and for what
            // this changes.
            commission_per_lot_is_per_side: true,
            backtest_spread_pips: 1.5,
            // Unset: no broker per-hour curve has been measured for this
            // install. The run WARNs and charges the flat spread.
            backtest_spread_pips_asian: None,
            backtest_spread_pips_overlap: None,
            backtest_spread_pips_late_ny: None,
            cost_band_optimistic_pips: 1.6,
            cost_band_pessimistic_pips: 2.4,
            conformal_enabled: true,
            conformal_alpha: 0.10,
            conformal_abstain_min_set_size: 3,
            meta_label_max_hold_bars: 100,
            meta_label_min_dist: 0.0005,
            meta_label_fixed_sl: 0.0020,
            meta_label_fixed_tp: 0.0040,
            vol_horizon_bars: 5,
        }
    }
}

/// Session-spread curve in pips, already ordered the way
/// `neoethos_search::eval::SessionSpreadProfile` reads it.
///
/// This crate cannot name that type (it does not depend on the search crate),
/// so the resolved curve travels as a plain triple and the search crate builds
/// the profile from it. The ORDER is part of the contract: `[asian, overlap,
/// late_ny]`, matching the UTC buckets 22–07 / 07–16 / 16–22.
pub type SessionSpreadPips = [f64; 3];

impl RiskConfig {
    /// Resolve the operator's session-spread curve.
    ///
    /// - `Ok(None)` — none of the three keys is set. The evaluator charges the
    ///   flat `backtest_spread_pips` at every hour of the day. This is the
    ///   shipped default and the caller is expected to say so out loud.
    /// - `Ok(Some(curve))` — all three set, finite and non-negative.
    /// - `Err(reason)` — a PARTIAL curve, or a non-finite / negative bucket.
    ///   Refused rather than repaired: a cost model that is two-thirds
    ///   configured charges numbers nobody chose.
    pub fn session_spread_pips(&self) -> Result<Option<SessionSpreadPips>, String> {
        crate::current_broker_financial_truth_capability_v1()
            .require(crate::BrokerFinancialOperationV1::HistoricalEvaluation)
            .map_err(|error| error.to_string())?;

        let named = [
            (
                "backtest_spread_pips_asian",
                self.backtest_spread_pips_asian,
            ),
            (
                "backtest_spread_pips_overlap",
                self.backtest_spread_pips_overlap,
            ),
            (
                "backtest_spread_pips_late_ny",
                self.backtest_spread_pips_late_ny,
            ),
        ];
        let set: Vec<&str> = named
            .iter()
            .filter(|(_, v)| v.is_some())
            .map(|(k, _)| *k)
            .collect();
        if set.is_empty() {
            return Ok(None);
        }
        if set.len() != named.len() {
            let missing: Vec<&str> = named
                .iter()
                .filter(|(_, v)| v.is_none())
                .map(|(k, _)| *k)
                .collect();
            return Err(format!(
                "session spread curve is partially configured: {} set, {} missing. All three \
                 buckets must be given together (Asian 22–07 UTC, Overlap 07–16, Late NY 16–22) \
                 or none — a partial curve would charge an unchosen number for a third of every \
                 trading day.",
                set.join(", "),
                missing.join(", ")
            ));
        }
        let mut out = [0.0f64; 3];
        for (slot, (key, value)) in out.iter_mut().zip(named.iter()) {
            let v = value.unwrap_or(f64::NAN);
            if !v.is_finite() || v < 0.0 {
                return Err(format!(
                    "{key} = {v} is not a usable spread (must be finite and >= 0)"
                ));
            }
            *slot = v;
        }
        Ok(Some(out))
    }

    /// The commission this account pays for a COMPLETE round trip on one lot,
    /// in account currency.
    ///
    /// Every evaluator subtracts this exactly once per closed trade, so this —
    /// not the per-side quote — is what belongs in `commission_per_trade`.
    pub fn round_trip_commission_per_lot(&self) -> Result<f64, crate::BrokerFinancialTruthErrorV1> {
        crate::current_broker_financial_truth_capability_v1()
            .require(crate::BrokerFinancialOperationV1::HistoricalEvaluation)?;
        let per_lot = self.commission_per_lot.max(0.0);
        Ok(if self.commission_per_lot_is_per_side {
            per_lot * 2.0
        } else {
            per_lot
        })
    }

    /// The cost band, ordered `(optimistic, pessimistic)`, sanitised.
    ///
    /// Returns `None` when the band is unusable (non-finite, negative, or
    /// inverted) so the caller can refuse rather than silently report a
    /// one-sided cost. An operator who genuinely wants a point estimate has to
    /// set both edges to the same number, and the report will then say the band
    /// is degenerate instead of pretending it measured a range.
    pub fn cost_band_pips(&self) -> Option<(f64, f64)> {
        let lo = self.cost_band_optimistic_pips;
        let hi = self.cost_band_pessimistic_pips;
        if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi < 0.0 || hi < lo {
            return None;
        }
        Some((lo, hi))
    }
}

/// Models and training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelsConfig {
    pub ml_models: Vec<String>,
    pub use_rl_agent: bool,
    pub use_sac_agent: bool,
    /// Legacy Ray/RLlib request retained only so older configuration files can
    /// be read and rejected with an actionable migration error. NeoEthos does
    /// not ship a Ray runtime; `true` fails before model dispatch.
    pub use_rllib_agent: bool,
    /// Legacy RLlib worker count retained for config migration only. A non-zero
    /// value is an RLlib request and fails at the same pre-dispatch boundary.
    pub rllib_num_workers: usize,
    /// Legacy auto-RLlib request. This must remain `false`; `true` is rejected
    /// before dispatch instead of silently substituting the native rlkit DQN.
    pub auto_enable_rllib: bool,
    pub use_neuroevolution: bool,
    /// ⚠ UNWIRED — nothing reads this field.
    ///
    /// `NeatTrainer` has a `population_size`, but it is hardcoded (96 in
    /// `NeatConfig::default`, floored at 24 in `with_config`) and never
    /// sourced from config. Connecting this field's default of 5 to it would
    /// collapse the NEAT population 19-fold, so it is NOT wired silently —
    /// see `tests/config_has_recipient.rs::UNWIRED`.
    pub rl_population_size: usize,
    pub rl_timesteps: usize,
    pub rl_eval_episodes: usize,
    pub rl_network_arch: Vec<usize>,
    pub rl_parallel_envs: usize,
    pub rl_state_bins: usize,
    pub rl_state_encoding: String,
    pub rl_update_interval: usize,
    pub rl_update_freq: usize,
    pub rl_learning_rate: f64,
    pub rl_gamma: f64,
    pub rl_epsilon_start: f64,
    pub rl_epsilon_end: f64,
    pub rl_epsilon_decay: f64,
    pub rl_buffer_capacity: usize,
    pub rl_reward_horizon: usize,
    pub rl_episode_len: usize,
    pub rl_train_seconds: u64,
    pub exit_agent_hidden_dim: usize,
    pub exit_agent_gamma: f64,
    pub exit_agent_epsilon: f64,
    pub exit_agent_epsilon_min: f64,
    pub exit_agent_epsilon_decay: f64,
    pub exit_agent_memory_capacity: usize,
    pub exit_agent_reward_horizon: usize,
    pub exit_agent_warmup_steps: usize,
    pub evo_train_seconds: u64,
    pub evo_hidden_size: usize,
    pub evo_population: usize,
    pub evo_islands: usize,
    pub evo_sigma: f64,
    pub prop_search_enabled: bool,
    pub prop_search_population: usize,
    /// Size the GA population from the card instead of from
    /// `prop_search_population`.
    ///
    /// This is the SEARCH-MORE knob, and it is the opposite of a batching
    /// knob: a bigger GA population evaluates DIFFERENT candidates and
    /// selects different survivors — it changes results on purpose. When
    /// `true` and a CUDA card is present, discovery raises the population to
    /// the card's fits ceiling (never above 16 384, never below
    /// `prop_search_population`) and logs the resolved value. When no card
    /// ceiling is readable, the configured population is kept and a warning
    /// says so. Default `true`: the card-sized result is sealed into the run
    /// receipt, so every search records the exact population it evaluated.
    pub prop_search_population_auto: bool,
    pub prop_search_generations: usize,
    pub prop_search_max_hours: f64,
    pub prop_search_max_rows: usize,
    #[serde(serialize_with = "serialize_sorted_map")]
    pub prop_search_max_rows_by_tf: HashMap<String, usize>,
    pub prop_search_portfolio_size: usize,
    pub prop_search_max_indicators: usize,
    pub prop_search_checkpoint: PathBuf,
    pub prop_search_device: String,
    pub prop_search_val_candidates: usize,
    pub prop_search_val_min_positive_months: usize,
    pub prop_search_val_min_trades_per_month: usize,
    pub prop_search_val_min_trades_per_day: f64,
    /// Target profile: the lowest win rate a candidate may have, as a fraction.
    ///
    /// Stated separately from `prop_search_min_payoff_ratio` because
    /// `profit_factor` folds the two together — 30 % of trades at 5:1 and 70 %
    /// at 0.6:1 both give about 2.1, and they are completely different systems
    /// to hold. `0.0` disables the gate.
    pub prop_search_min_win_rate: f64,
    /// Target profile: the lowest average-win over average-loss a candidate may
    /// have. `0.0` disables the gate.
    ///
    /// SECONDARY ONLY since 2026-08-09. This number is a SHAPE preference and it
    /// is not, and can never be, evidence of profit. Measured: a candidate at
    /// payoff 2.53 had an expectancy of -4.18 pips per trade. It clears a 2.0
    /// floor and loses money on every trade it takes. The gate that decides
    /// survival is `prop_search_min_net_expectancy_per_trade` below; this one can
    /// only narrow what that gate already admitted.
    pub prop_search_min_payoff_ratio: f64,
    /// The PRIMARY survival gate: the lowest cost-charged net expectancy per
    /// trade, in account currency, a candidate may have.
    ///
    /// `0.0` does NOT mean "no preference" — unlike every other field on the
    /// target profile, it means "must be strictly greater than zero". There is
    /// no configuration in which a negative-expectancy candidate is admitted.
    ///
    /// The number is the mean of the per-trade net P&L the backtest actually
    /// booked, so spread, commission, swap and the conversion fee are already
    /// subtracted. It answers the only question that decides whether an account
    /// grows: after paying the broker, does the average trade make money?
    pub prop_search_min_net_expectancy_per_trade: f64,
    /// How many standard errors above zero that expectancy must sit.
    ///
    /// `mean / (sd / sqrt(n))`. `0.0` requires only the sign, which is the
    /// default because a significance floor is a separate decision from a
    /// correctness bound and the operator has not made it yet. Set it to ~2.0 to
    /// refuse candidates whose positive expectancy is inside its own noise —
    /// note that this is an IN-SAMPLE t-statistic on overlapping trades, so it
    /// bounds sampling noise, not selection bias. Only DSR/PBO over the full
    /// trial set can do the latter.
    ///
    /// The per-trial return series that DSR/PBO need IS now persisted — every
    /// screened candidate, captured before any gate, written to
    /// `{SYMBOL}_{TF}.trial_returns.bin` beside the ledger. What does not exist
    /// yet is a READER: nothing in the workspace computes a deflated Sharpe or a
    /// CSCV/PBO from that matrix. So the precondition has landed and the
    /// correction has not, and until it does, a candidate that clears this floor
    /// has been checked against its own sampling noise and NOT against the
    /// thousands of trials it was selected from.
    pub prop_search_min_expectancy_t_stat: f64,
    /// Target profile: the most of the evaluated span a candidate may spend
    /// holding a position, as a fraction.
    ///
    /// A strategy in the market almost always is not selecting entries, and its
    /// win rate converges on the market's base rate however the entry rule is
    /// written. `0.0` disables the gate.
    pub prop_search_max_in_market: f64,
    pub prop_search_val_min_monthly_profit_pct: f64,
    pub prop_search_val_log_trades: bool,
    pub prop_search_val_trade_log_max: usize,
    pub prop_search_async: bool,
    pub prop_search_async_wait: bool,
    pub tree_device_preference: String,
    /// ML overfit-reduction (v0.5 ML-integration Stage 1). When `true`
    /// (default), the gradient boosters (xgboost/lightgbm/catboost + variants)
    /// train with regularized, bar-scaled defaults (shallower trees, column +
    /// row subsampling, L1/L2, leaf-size floors, bar-scaled tree counts) instead
    /// of the legacy full-depth / full-data / no-shrinkage defaults that
    /// memorize thin-TF (D1/W1/MN) targets. Set `false` to restore the legacy
    /// unregularized defaults for a controlled before/after OOS comparison.
    /// `#[serde(default)]` on `ModelsConfig` makes a missing key fall back to
    /// the `Default` impl below (= `true`).
    pub regularized_model_defaults: bool,
    /// ML overfit-reduction: minimum per-(symbol,TF) bar count below which the
    /// heavy gradient boosters are forced onto a shrunk preset (shallow depth,
    /// few trees, strong L2) and per-bar HPO is disabled (a thin holdout cannot
    /// select 5+ hyperparameters). Default 4000 — D1 (~2700 bars) and coarser
    /// TFs fall below it. Below an absolute floor (800) an even tinier preset is
    /// used. Set to 0 to disable the gate entirely.
    pub heavy_booster_min_bars: usize,
    /// ML overfit-reduction: when `true` (default) ML hyperparameter selection
    /// uses CombinatorialPurgedCV (purge+embargo, 15 paths) scored by
    /// mean-minus-stdev of the objective across folds — penalizing params that
    /// only generalize to one lucky window — instead of a single time-series
    /// holdout. Gated to `bars >= heavy_booster_min_bars` and `trials > 1` to
    /// bound the 15×-fold fit cost; below that it falls back to the single
    /// holdout. Set `false` to restore the single-holdout HPO.
    /// WARNING NAMING TWIN of `models.enable_cpcv` - NOT a duplicate. This one
    /// gates TRAINING CPCV (`training_orchestrator.rs:4432`); `enable_cpcv`
    /// gates the SEARCH admission gate (`discovery.rs:2586-2591`). Disarming
    /// the wrong one admits candidates that never passed purged CV. Renaming
    /// this to `training_cpcv_enabled` needs `neoethos-models` in the same wave
    /// - routed to `docs/pending-edits-forbidden-territory.md`.
    pub ml_cpcv_enabled: bool,
    pub prop_search_parent_selection: String,
    pub prop_search_survivor_selection: String,
    pub prop_search_survivor_fraction: f64,
    pub prop_search_immigrant_fraction: f64,
    pub prop_search_selection_temperature: f64,
    pub prop_search_tournament_size: usize,
    pub prop_search_opportunistic_enabled: bool,
    pub prop_search_opportunistic_min_positive_months: usize,
    pub prop_search_opportunistic_min_trades_per_month: usize,
    pub prop_search_opportunistic_min_trade_return_pct: f64,
    pub prop_search_opportunistic_max_dd: f64,
    pub prop_search_use_opportunistic: bool,
    /// 2026-05-26 operator directive (dual-mode product): correlation
    /// threshold for portfolio diversification (Pearson + Spearman both
    /// checked). Strategies with |correlation| ≥ this value against any
    /// portfolio member are rejected. Previously hardcoded 0.85 in
    /// `discovery.rs` — surfaced here so the operator can tune dedup
    /// aggressiveness from config / Settings UI without rebuilding.
    pub prop_search_corr_threshold: f64,
    /// Monte-Carlo perturbation runs per surviving candidate. The MC test
    /// re-evaluates each gene with random ±15-25% noise on thresholds,
    /// weights, and SL/TP and requires a configurable minimum to be
    /// profitable. Previously hardcoded 100 in discovery.rs.
    pub prop_search_mc_runs: u32,
    /// Minimum number of profitable MC runs required for a candidate to
    /// survive (out of `prop_search_mc_runs`). Previously hardcoded 70/100
    /// in discovery.rs (i.e. 70% threshold).
    pub prop_search_mc_min_profitable: u32,
    /// Spread (in pips) used in the sensitivity test — re-runs the
    /// candidate's backtest with a wider spread to verify the strategy
    /// stays profitable under degraded execution. Previously hardcoded
    /// 2.0 in discovery.rs.
    pub prop_search_sensitivity_spread_pips: f64,
    /// Commission per lot used in the sensitivity test, quoted on the SAME side
    /// convention as `risk.commission_per_lot` — `DiscoveryConfig::from_settings`
    /// puts it through the same `round_trip_commission_per_lot` conversion,
    /// gated on `risk.commission_per_lot_is_per_side`, and then clamps it UP to
    /// the resolved baseline commission: a stress pass may charge more than the
    /// run it stresses, never less.
    ///
    /// Until 2026-08-10 it skipped the conversion entirely, so at the shipped
    /// defaults (7.0 here; 7.0 per side → 14.0 round trip there) the "higher
    /// commission" scenario charged HALF the baseline and every candidate passed
    /// it. Previously hardcoded $7/lot in discovery.rs.
    pub prop_search_sensitivity_commission_per_lot: f64,
    pub train_batch_size: usize,
    /// WARNING DERIVED FROM HARDWARE - NOT AN INPUT. `#[serde(skip)]`
    /// 2026-08-10. Zero readers (`config_has_recipient.rs:210-233`);
    /// `HardwareExecutionPlan::inference_batch_size` (`system.rs:1466-1477`)
    /// computes it from the probe and hands it to the consumer as a parameter
    /// rather than writing it back into `Settings`. Deleting the field is
    /// blocked on `system.rs` - routed to `pending-A.md`.
    #[serde(skip)]
    pub inference_batch_size: usize,
    pub enable_transformer_expert: bool,
    /// WARNING ARITHMETIC TWIN of `transformer_n_heads` (below): both are read
    /// and collapsed by `.max()` at `training_orchestrator.rs:886-892`, with no
    /// winner named. They ship equal, agreeing by luck. One field should
    /// survive; the call site is in `neoethos-models` - routed to
    /// `docs/pending-edits-forbidden-territory.md`.
    pub transformer_heads: usize,
    pub transformer_layers: usize,
    pub transformer_hidden_dim: usize,
    pub transformer_dropout: f64,
    pub transformer_seq_len: usize,
    pub transformer_train_seconds: u64,
    pub nbeats_train_seconds: u64,
    pub tide_train_seconds: u64,
    pub tabnet_train_seconds: u64,
    pub kan_train_seconds: u64,
    pub mlp_train_seconds: u64,
    pub num_transformers: usize,
    pub swarm_memory_limit_mb: f64,
    pub swarm_horizon: usize,
    pub swarm_frequency: String,
    pub swarm_strategy: String,
    pub swarm_online_learning: bool,
    pub swarm_interpretability_needed: bool,
    pub swarm_latency_ms: usize,
    pub hpo_backend: String,
    pub hpo_trials: usize,
    #[serde(serialize_with = "serialize_sorted_map")]
    pub hpo_trials_by_model: HashMap<String, usize>,
    pub hpo_max_rows: usize,
    #[serde(serialize_with = "serialize_sorted_map")]
    pub max_epochs_by_model: HashMap<String, usize>,
    pub ray_tune_max_concurrency: usize,
    pub calibration_enabled: bool,
    pub calibration_method: String,
    pub calibration_min_rows: usize,
    /// LIVE ML gate (Stage 3 blend in the live autopilot). When true, the
    /// live loop loads the symbol's soft-voting ensemble once at engine
    /// start and, on every closed bar, scales the per-trade risk by the
    /// ensemble's agreement × regime gate × anomaly scale (MlScale mode:
    /// the genes ALWAYS pick the direction; ML can only SHRINK size or
    /// skip a bar on a hard regime/anomaly collapse — never flip, never
    /// manufacture a trade). Default FALSE: live sizing must never change
    /// silently; the operator flips this knowingly. Fail-soft: any
    /// ensemble error on a bar falls back to gene-only sizing, loudly.
    pub live_ml_gate: bool,

    /// Floor on the ML agreement term in the live blend (`models.live_ml_gate`).
    /// A gene bar the ensemble is only lukewarm about still trades at THIS
    /// fraction of its size, so the validated gene edge is never gated to
    /// nothing by a lukewarm model. Range [0,1]; default 0.34. Out of range,
    /// non-finite, or below `blend_veto_below` ⇒ REFUSED back to the default and
    /// logged with both numbers (`BlendConfig::from_config_values`) — this
    /// multiplier scales every entry's risk.
    ///
    /// Kept numerically equal to `neoethos_trader::DEFAULT_BLEND_GATE_FLOOR`.
    /// `neoethos-core` cannot depend on `neoethos-trader`, so the literal is
    /// duplicated deliberately rather than imported; changing one without the
    /// other is the defect this note exists to prevent.
    pub blend_gate_floor: f64,

    /// Effective-multiplier floor below which the live blend SKIPS the bar
    /// entirely (Flat, not confidence 0 — the sizing floor would otherwise open
    /// min volume). In `MlConfirm` it also vetoes when the raw ML `p_side` is
    /// below it. Range [0,1]; default 0.15. Must be <= `blend_gate_floor`, else
    /// every floored bar would be vetoed and the pair is REFUSED back to the
    /// defaults, loudly.
    ///
    /// Kept numerically equal to `neoethos_trader::DEFAULT_BLEND_VETO_BELOW`.
    pub blend_veto_below: f64,
    #[serde(serialize_with = "serialize_sorted_nested_map")]
    pub model_param_overrides: HashMap<String, HashMap<String, String>>,
    pub regime_router_enabled: bool,
    pub regime_router_min_models: usize,
    pub regime_trend_models: Vec<String>,
    pub regime_range_models: Vec<String>,
    pub regime_neutral_models: Vec<String>,
    pub l1_feature_selection_enabled: bool,
    pub l1_feature_selection_per_regime: bool,
    pub l1_feature_selection_min_features: usize,
    pub l1_feature_selection_max_features: usize,
    pub l1_feature_selection_sample_limit: usize,
    pub l1_feature_selection_c: f64,
    pub filter_to_base_signal: bool,
    pub global_max_rows: usize,
    pub global_max_rows_per_symbol: usize,
    pub symbol_hash_buckets: usize,
    pub global_train_ratio: f64,
    pub train_holdout_pct: f64,
    pub label_use_triple_barrier: bool,
    /// Bracket geometry for training-label derivation (2026-08-08).
    ///
    /// * `"symmetric"` (default) — target distance == stop distance and the
    ///   round-trip cost is charged the same way in both directions, so the
    ///   label is a fair direction race. Real EURUSD M15 bars measure a
    ///   0.49/0.51 class prior here — a coin flip, which is what a label with
    ///   no manufactured bias looks like on noise.
    /// * `"asymmetric"` — the pre-2026-08-08 geometry, kept for comparison
    ///   runs: `risk.meta_label_fixed_tp` (0.0040) against
    ///   `risk.meta_label_fixed_sl` (0.0020) with `risk.min_risk_reward`
    ///   flooring the target at 2× the stop. On M15 those floors bind on
    ///   88.4 % of bars — a 40-pip target racing a 20-pip stop — and
    ///   MANUFACTURE a 66/34 class prior. 14 models recorded validation
    ///   accuracy bit-identical to that prior: constant predictors trained on
    ///   a label whose skew was geometry, not signal.
    ///
    /// Any other value fails label derivation loudly. The resolved geometry is
    /// recorded in each model's training profile/artifact so a model can name
    /// the labels it was trained on.
    pub label_geometry: String,
    pub label_horizon_bars: usize,
    pub label_neutral_band_atr_fraction: f64,
    /// MONEY WARNING ARITHMETIC TWIN of `risk.atr_stop_multiplier`: collapsed
    /// by `.max()` at `training_orchestrator.rs:2316-2321`, no winner named,
    /// and a THIRD hardcoded 1.5 lives at `stop_target.rs:226`. Survivor should
    /// be `risk.atr_stop_multiplier`. Call site is in `neoethos-models` -
    /// routed to `docs/pending-edits-forbidden-territory.md`.
    pub label_stop_atr_multiplier: f64,
    /// MONEY WARNING ARITHMETIC TWIN of `risk.min_risk_reward`: collapsed by
    /// `.max()` at `training_orchestrator.rs:2481-2487` - but ONLY when
    /// `label_geometry` selects the Asymmetric arm. The shipped `symmetric` arm
    /// reads NEITHER, which is why a 2RR floor may never reach the labels.
    /// Survivor should be `risk.min_risk_reward`. Routed to
    /// `docs/pending-edits-forbidden-territory.md`.
    pub label_take_profit_rr: f64,
    pub walkforward_splits: usize,
    pub embargo_minutes: usize,
    /// Discovery search regime: `"prop_firm"` (default — permissive
    /// quality floors so the prop-firm gauntlet does the heavy lifting)
    /// or `"strict"` (full FilteringConfig floors). Was the env-only
    /// `NEOETHOS_BOT_DISCOVERY_MODE`; now a first-class config knob the
    /// operator sets from the UI / TUI — never the environment.
    /// WARNING RESTRICT, DO NOT MERGE (refuter overturn).
    /// `discovery.rs:5755-5760` maps only `strict|legacy`; every other value
    /// falls through to `system.trading_mode`. It reaches `Strict`, which
    /// `trading_mode` structurally cannot, so the two are NOT one knob. The
    /// accepted values must be narrowed to `strict|legacy` and the fall-through
    /// logged by name - the caller is `neoethos-search::discovery` and the TUI
    /// that offers `risky`/`prop_firm` and rejects `legacy` is app-side. Routed
    /// to `docs/pending-edits-forbidden-territory.md`.
    pub discovery_mode: String,
    /// agent 2026-06-05 overfitting fix: when `true` (default), a discovered
    /// portfolio is only export-ready in PropFirm mode if it ALSO passes the
    /// walk-forward gate (not just the prop-firm window gate). Previously the
    /// walk-forward result was purely informational in PropFirm mode, so
    /// overfit strategies (in-sample Sharpe 3-11 / PF up to 62) that failed
    /// out-of-sample still exported. Set `false` to restore the old behaviour
    /// (prop-firm-window gate only). `#[serde(default)]` on `ModelsConfig`
    /// makes a missing key fall back to the `Default` impl below (= `true`).
    pub require_walkforward_for_export: bool,
    /// Hard floor for the prop-firm window-pass rate, applied on top of
    /// `discovery_runtime.prop_firm_gate.pass_rate` (effective floor = max of the
    /// two). RE-CALIBRATED 2026-06-06 from 0.65 → **0.40** when the per-window
    /// profit target was set to the operator's bar (8%/60-day window = >=4%/month,
    /// in `derive_prop_firm_gate`). 0.40 = a candidate must hit >=4%/month in at
    /// least 40% of the random 60-day windows to survive — a genuine persistent
    /// edge, with the live models lifting the rest (discovery=edge, models=grow).
    /// The base-filter max-DD + walk-forward export gate still reject blow-ups /
    /// overfit. Raise toward 0.65 for stricter selection; lower for more candidates.
    /// MONEY WARNING ARITHMETIC TWIN of
    /// `models.discovery_runtime.prop_firm_gate.pass_rate`: collapsed by
    /// `.max()` at `discovery.rs:7812`. The 2026-06-06 mandate written into
    /// both repo YAMLs names only THIS field, so raising the other silently
    /// overrides the disarm. One field should survive; while both exist the
    /// SAFER (higher) value wins and the disagreement must be logged with both
    /// numbers. Caller is in `neoethos-search` - routed to
    /// `docs/pending-edits-forbidden-territory.md`.
    pub prop_firm_min_pass_rate: f64,
    /// Genetic-search runtime knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_*` search env vars). See [`SearchRuntimeConfig`].
    pub search_runtime: SearchRuntimeConfig,
    /// Discovery-pipeline runtime knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_PREFILTER_*` / `NEOETHOS_BOT_FUNNEL_STAGE1_*` /
    /// `NEOETHOS_BOT_MIN_HISTORY_YEARS` / `NEOETHOS_BOT_PROP_ADAPTIVE_THRESHOLDS`
    /// env vars). See [`DiscoveryRuntimeConfig`].
    pub discovery_runtime: DiscoveryRuntimeConfig,
    /// Strategy-evaluation runtime knobs (config-driven replacement for
    /// the `NEOETHOS_BOT_PROP_*` cost + SMC-weight env vars). See
    /// [`EvalRuntimeConfig`].
    pub eval_runtime: EvalRuntimeConfig,
    /// Strategy-quality scoring knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_PROP_*` monthly-quality env vars). See
    /// [`QualityRuntimeConfig`].
    pub quality_runtime: QualityRuntimeConfig,
    /// Backtest-evaluation runtime knobs (config-driven replacement for
    /// the `NEOETHOS_BOT_BACKTEST_*` env vars). See [`BacktestRuntimeConfig`].
    pub backtest_runtime: BacktestRuntimeConfig,
    /// Adaptive-stop cost caps. See [`StopTargetRuntimeConfig`] — the
    /// recipient the hardcoded `tail_max_bars = 300_000` never had.
    pub stop_target_runtime: StopTargetRuntimeConfig,
    /// Discovery exit geometry. See [`ExitPolicyConfig`] — the recipient the
    /// hardcoded `trailing_enabled: true` in `strategy_gene.rs:851` never had.
    pub exit_policy: ExitPolicyConfig,
    /// The stop/target band gene generation and mutation draw within. See
    /// [`GeneStopBoundsConfig`] — the recipient the `[6, 20]` / `[12, 45]` pip
    /// literals in `evolution_math.rs` never had.
    pub gene_stop_bounds: GeneStopBoundsConfig,
    /// How the live soft-voting ensemble combines its experts. See
    /// [`EnsembleVotingConfig`] — the recipient
    /// `SoftVotingEnsembleConfig::default()` never had.
    pub ensemble_voting: EnsembleVotingConfig,
    /// Seen-signature dedup-memory knobs (config-driven replacement for
    /// the `NEOETHOS_BOT_PROP_SEEN_*` env vars). See
    /// [`SeenSignatureRuntimeConfig`].
    pub seen_signature_runtime: SeenSignatureRuntimeConfig,
    /// Search-memory + weekly-refresh ledger knobs (2026-06-06): persist what
    /// each discovery run found and seed the next run's seen-set so weekly runs
    /// add NEW strategies instead of re-discovering old ones. See
    /// [`DiscoveryLedgerConfig`].
    pub discovery_ledger: DiscoveryLedgerConfig,
    /// SMC search-injection knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_PROP_SMC_*` env vars). See [`SmcSearchRuntimeConfig`].
    pub smc_search_runtime: SmcSearchRuntimeConfig,
    /// Data-layer behavior knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_NORMALIZE_FEATURES`
    /// env vars). See [`DataRuntimeConfig`].
    pub data_runtime: DataRuntimeConfig,
    /// Tree-model training knobs (config-driven replacement for the
    /// `NEOETHOS_BOT_EARLY_STOP_*` env vars). See [`TreeRuntimeConfig`].
    pub tree_runtime: TreeRuntimeConfig,
    /// Device policy for the statistical models (ElasticNet / Logistic —
    /// `statistical/linear_impl.rs`). One of:
    ///
    /// - `"cpu"` (default) — always the CPU path. This is what every build
    ///   has done to date, because the CUDA softmax kernel behind it was
    ///   compiled by no shipped feature combination.
    /// - `"auto"` — the CUDA kernel when a CUDA device is present, else CPU.
    /// - `"gpu"` / `"cuda"` / `"gpu:N"` / `"cuda:N"` — the CUDA kernel,
    ///   optionally pinned to device N.
    ///
    /// The default is `"cpu"` rather than `"auto"` because the two backends
    /// are not bit-identical: the kernel optimises with subgradient-L1 SGD,
    /// so it is only used when `l1_ratio == 0` in the first place, and even
    /// then f32 GPU reductions accumulate differently from the CPU path. That
    /// changes fitted weights, which changes predictions, which changes
    /// selection. Flipping this is the operator's call.
    ///
    /// `NEOETHOS_BOT_<MODEL>_DEVICE` and `NEOETHOS_BOT_META_DEVICE` still
    /// override this, as they always have.
    pub statistical_device: String,
    /// Thresholds for the promotion gate — the bar a discovered portfolio
    /// must clear before it is reported as promotable. See
    pub prop_metric_weight: f64,
    pub prop_accuracy_weight: f64,
    pub prop_min_trades: usize,
    pub prop_conf_threshold: f64,
    /// WARNING NAMING TWIN of `models.ml_cpcv_enabled` - NOT a duplicate. THIS
    /// one is the SEARCH purged-CV admission gate (`discovery.rs:2586-2591`).
    /// Rename to `search_cpcv_gate_enabled` needs `neoethos-search` in the same
    /// wave - routed to `docs/pending-edits-forbidden-territory.md`.
    pub enable_cpcv: bool,
    pub cpcv_n_splits: usize,
    pub cpcv_n_test_groups: usize,
    pub cpcv_embargo_pct: f64,
    pub cpcv_purge_pct: f64,
    pub cpcv_min_phi: f64,
    pub cpcv_max_rows: usize,
    pub enable_ddp: bool,
    pub enable_fsdp: bool,
    pub ddp_world_size: usize,
    /// WARNING ARITHMETIC TWIN - see `transformer_hidden_dim` above. These
    /// three are the `.max()` partners at `training_orchestrator.rs:878-900`.
    pub transformer_d_model: usize,
    pub transformer_n_heads: usize,
    pub transformer_n_layers: usize,
    pub nf_hidden_dim: usize,
    pub tide_hidden_dim: usize,
    pub nbeats_hidden_dim: usize,
    pub kan_hidden_dim: usize,
    pub kan_grid_size: usize,
    pub tabnet_hidden_dim: usize,
    pub phase5_filter_meta_blender: bool,
    pub phase5_core_models: Vec<String>,
    /// Promotion-gate thresholds — the quality bar a discovered +
    /// trained portfolio must clear before `POST /strategy_lab/promote`
    /// copies it into `live_models/`.
    ///
    /// **2026-08-04 — "mechanism exists, no final recipient".** The gate
    /// itself has always run, but the thresholds it ran with were
    /// unreachable: `neoethos_app::server::strategy_lab::load_gate_config`
    /// took a `&Settings`, ignored it (`_settings`), and returned
    /// `PromotionGateConfig::default()`. Its own doc comment promised
    /// "a future `ModelsConfig.promotion_gate` field can override them
    /// here" — this is that field, and the function now reads it. Until
    /// today no portfolio had ever been judged against an operator-set
    /// bar; every promotion decision used the hardcoded moderate
    /// defaults regardless of what the operator configured.
    ///
    /// This is deliberately the SAME struct the gate evaluates and the
    /// same struct the endpoint echoes to the UI — not a mirror with its
    /// own `Default`. A mirror would be free to drift from the enforced
    /// copy, which is exactly the failure this field exists to close.
    /// `promotion_gate_config_default_matches_the_gates_own_default`
    /// pins the values so adding the knob shifted no decision today.
    pub promotion_gate: crate::domain::promotion_gate::PromotionGateConfig,
    /// Demo forward-test gate — the bar a promoted strategy must clear on
    /// REAL demo fills before the operator may move it to real money.
    ///
    /// **2026-08-04 — the same "no final recipient" shape as
    /// [`Self::promotion_gate`], one stage further down the pipeline.**
    /// `neoethos_app::app_services::live_gate::evaluate_for_portfolio`
    /// loaded a `Settings` at its top and then passed
    /// `&DemoForwardGateConfig::default()` to the gate twenty-three lines
    /// later. `min_demo_trades: 100` and `forward_tolerance: 0.20` were
    /// unreachable literals on the last gate before real money.
    pub demo_forward_gate: crate::domain::demo_gate::DemoForwardGateConfig,
}

/// Genetic-search runtime knobs — the config-driven replacement for the
/// `NEOETHOS_BOT_*` genetic-search env vars (RNG seed, novelty weighting,
/// tournament / archive sizing, SMC-gate curve, archive scoring, selection
/// policy). Mirrors `neoethos_search::genetic::GeneticSearchRuntimeOverrides`,
/// which the search crate now builds via `from_settings(&Settings)` so the
/// operator sets these from config / UI / TUI — never the environment.
///
/// Defaults here MUST match that override struct's `Default`; a
/// `from_settings(&Settings::default()) == default()` unit test in
/// `neoethos-search` enforces it. Empty strings on the policy / archive-mode
/// fields mean "use the engine default" (so the config default need not
/// duplicate the parser vocabulary).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SearchRuntimeConfig {
    pub seed: Option<u64>,
    pub novelty_weight: f64,
    /// Explicit neighborhood size for novelty scoring. This is versioned run
    /// input, not an implementation constant: the resident path computes the
    /// mean Jaccard distance to the `k` nearest neighbors drawn from the
    /// current population plus its permanent archive.
    pub novelty_neighbors: usize,
    pub stagnation_patience: usize,
    pub tournament_size_override: Option<usize>,
    pub archive_cap_override: Option<usize>,
    pub seen_retry_attempts: usize,
    pub smc_gate_start: f64,
    pub smc_gate_end: f64,
    pub smc_gate_curve: f64,
    pub smc_gate_stagnation_step: f64,
    pub disable_smc_gate: bool,
    pub archive_mode: String,
    pub archive_min_net: f64,
    pub archive_min_pf: f64,
    pub archive_min_sharpe: f64,
    pub parent_selection: String,
    pub survivor_selection: String,
    pub immigrant_ratio: f64,
    pub survivor_fraction: f64,
    pub selection_temperature: f64,
    /// Generations of no meaningful improvement before the GA hard
    /// early-stops THIS combo and returns its archive, freeing the
    /// wall-clock budget for the next symbol×timeframe. `0` disables the
    /// early-stop (run to the time / generation cap as before). This is a
    /// SEPARATE, larger threshold than `stagnation_patience`: the soft
    /// diversity kick (gate relaxation + immigrants + hypermutation) is
    /// attempted first; the hard stop fires only if the search is STILL
    /// flat after `convergence_patience` generations.
    pub convergence_patience: usize,
    /// Minimum increase in top fitness counted as "improvement" when
    /// tracking stagnation; a generation gaining less than this is
    /// stagnant. Replaces the legacy hard-coded `1e-12`.
    pub min_improvement: f64,
    /// Wall-clock floor for the convergence early-stop, as a fraction of
    /// the per-combo time budget (`prop_search_max_hours`). The early-stop
    /// (see `convergence_patience`) may fire ONLY after this fraction of
    /// the budget has elapsed. This makes the early-stop throughput-robust:
    /// generation rate varies ~300× across timeframes, so a pure
    /// generation count (e.g. 250 gens ≈ 1 s on a fast TF, ≈ 21 min on M1)
    /// would otherwise kill fast timeframes before they ever search. `0.5`
    /// = every combo gets at least half its budget; `0` = no floor (pure
    /// generation count, NOT recommended); `1.0` = effectively disables the
    /// early-stop (only the time cap stops the combo).
    pub convergence_min_elapsed_fraction: f64,
}

impl Default for SearchRuntimeConfig {
    fn default() -> Self {
        Self {
            seed: None,
            novelty_weight: 0.0,
            novelty_neighbors: 15,
            stagnation_patience: 2,
            tournament_size_override: None,
            archive_cap_override: None,
            seen_retry_attempts: 16,
            smc_gate_start: 0.75,
            smc_gate_end: 0.35,
            smc_gate_curve: 1.0,
            smc_gate_stagnation_step: 0.03,
            disable_smc_gate: false,
            archive_mode: String::new(),
            archive_min_net: 0.0,
            archive_min_pf: 1.0,
            archive_min_sharpe: 0.0,
            parent_selection: String::new(),
            survivor_selection: String::new(),
            immigrant_ratio: 0.25,
            survivor_fraction: 0.10,
            selection_temperature: 0.75,
            convergence_patience: 250,
            min_improvement: 1e-12,
            convergence_min_elapsed_fraction: 0.5,
        }
    }
}

/// Discovery-pipeline runtime knobs — the config-driven replacement for the
/// legacy `NEOETHOS_BOT_PREFILTER_TOP_K`, `NEOETHOS_BOT_PREFILTER_INSAMPLE`,
/// `NEOETHOS_BOT_FUNNEL_STAGE1_PCT`, `NEOETHOS_BOT_FUNNEL_STAGE1_WINDOW`,
/// `NEOETHOS_BOT_MIN_HISTORY_YEARS`, and `NEOETHOS_BOT_PROP_ADAPTIVE_THRESHOLDS`
/// env vars. Consumed by `DiscoveryConfig::from_settings` (via
/// `DiscoveryRuntimeOverrides::from_settings`) — the operator sets these from
/// the UI / TUI, never the environment. Defaults reproduce the previous
/// env-absent behaviour exactly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DiscoveryRuntimeConfig {
    /// Max features kept after the in-sample correlation prefilter; `0`
    /// disables the prefilter. (was `NEOETHOS_BOT_PREFILTER_TOP_K`)
    ///
    /// **The default MUST equal the value shipped in `config.yaml`.** It did
    /// not: the code said 50 while the shipped config said 240, so any path
    /// that fell back to the default searched a quarter of the intended
    /// feature pool. At 50 the base set collapses from 217 columns to roughly
    /// 64 and the SMC, session and footprint families die first, because —
    /// unlike `regime_` — they have no force-keep. That divergence is now
    /// pinned by `crates/neoethos-core/tests/shipped_config_matches_defaults.rs`,
    /// which parses the shipped YAML and fails if the two ever disagree again.
    ///
    /// **2026-08-10 — THIS IS NOW A FLOOR, NOT THE VALUE.** The effective pool
    /// is derived from GA CAPACITY in
    /// `neoethos_search::discovery::resolve_prefilter_top_k`:
    /// `clamp(population * E[indices per gene] / 46, this value, cube width)`.
    ///
    /// Why the constant could not stay a constant: it was chosen against a
    /// 217-column-per-timeframe cube, and the cube's width is now bounded by
    /// `VocabularyBudget`, i.e. by FREE RAM and the frame length. A constant
    /// against a variable cube means the FRACTION of the vocabulary the GA can
    /// see is decided by the hardware — 13.8% at the old vocabulary, ~4.9% at
    /// what a 20 GB box affords on the real M5 frame, 0.7% at the 4,096-column
    /// ceiling. The derived value is calibrated on the historical operating
    /// point (265 columns kept at population 4,096) and, correctly, does NOT
    /// grow when the box or the timeframe list grows — because the alphabet the
    /// GA can actually cover does not grow either.
    ///
    /// At the shipped populations the derived value is below this floor, so the
    /// effective pool is still exactly the 240 it has always been. `0` still
    /// disables the prefilter entirely.
    pub prefilter_top_k: usize,
    /// Fraction of rows treated as in-sample when ranking features; must be
    /// in `(0, 1]`. (was `NEOETHOS_BOT_PREFILTER_INSAMPLE`)
    pub prefilter_insample_frac: f64,
    /// Minimum number of features to force-keep from EACH present higher
    /// timeframe group during the prefilter, in addition to the global
    /// `prefilter_top_k`. The prefilter ranks by correlation with the BASE
    /// timeframe's 1-bar forward return; a slow higher-TF indicator is
    /// near-constant across many base bars so that correlation is ~0 by
    /// construction, and the global top-K therefore discards EVERY multi-TF
    /// feature — wasting the entire multi-resolution cube and starving the
    /// GA's multi-TF seed templates. This quota guarantees each higher TF
    /// (`H1_`, `H4_`, `M15_`, …) reaches the GA. `0` reproduces the legacy
    /// base-only behaviour. (new 2026-06-08)
    pub prefilter_min_per_timeframe: usize,
    /// Fraction of rows fed to the multi-stage funnel's first stage; clamped
    /// to `[0.01, 1.0]`. (was `NEOETHOS_BOT_FUNNEL_STAGE1_PCT`)
    pub funnel_stage1_pct: f64,
    /// Where to slice the stage-1 fast-eval rows: `"earliest"` (default,
    /// OOS-safe), `"latest"`, or `"random"`. (was
    /// `NEOETHOS_BOT_FUNNEL_STAGE1_WINDOW`)
    pub stage1_window: String,
    /// Minimum historical-data window (years) discovery requires before it
    /// runs; `0` skips the pre-flight check. (was
    /// `NEOETHOS_BOT_MIN_HISTORY_YEARS`)
    pub min_history_years: u32,
    /// Derive the coarse-threshold ladder from THIS run's feature cube instead
    /// of using the static one.
    ///
    /// **Default flipped to `true` on 2026-08-09.** The static ladder
    /// `[0.10, 0.20, 0.35, 0.50, 0.70, 0.90]` carries its own calibration in a
    /// comment — "Calibrated for z-score-normalised features"
    /// (`evolution_math.rs:560-563`) — and `models.data_runtime.normalize_features`
    /// was `false` in both shipped configs and in the code default. So the
    /// ladder was being compared against raw feature magnitudes spanning about
    /// 1e5:1 (RSI ~50, EMA ~1.08, M5 ATR ~5e-4). A gene's threshold at 0.35 is
    /// unreachable for an ATR term and always-on for an RSI term, whatever
    /// weight the GA gave it.
    ///
    /// `derive_adaptive_threshold_ladder_from_features` (`evolution_math.rs:648`)
    /// exists for exactly this and is installed at `discovery.rs:3492` when the
    /// flag is true; it places the six rungs at percentile points of the
    /// dataset's own per-column median magnitude, so a threshold means the same
    /// thing on XAUUSD M1 as on EURUSD D1.
    ///
    /// **What this fixes and what it does NOT.** It fixes the THRESHOLD side
    /// only. The weight ladder is still `{0.2, 0.4, 0.6, 0.8, 1.0}` — a 5:1
    /// span — set against a feature-scale span of ~1e5:1, so a multi-term gene
    /// is still arithmetically equal to its single largest-magnitude term. The
    /// complete fix is `models.data_runtime.normalize_features: true`, which
    /// puts every column on a comparable scale and makes the weight ladder
    /// decide something. See that field's docs for the trade-off.
    ///
    /// The old "leave off for multi-symbol sweeps (OnceLock)" caveat is stale:
    /// audit D06 (2026-07-13) replaced the `OnceLock` with a per-run replace
    /// (`install_adaptive_threshold_ladder` / `clear_adaptive_threshold_ladder`),
    /// so a batch sweep no longer leaks the first symbol's ladder.
    ///
    /// **This changes what is searched.** Genes initialise and mutate onto
    /// different thresholds, so a run before this flag and a run after it are
    /// not comparable. (was `NEOETHOS_BOT_PROP_ADAPTIVE_THRESHOLDS`)
    pub adaptive_thresholds: bool,
    /// Prop-firm window-pass gate parameters (FTMO baseline + overrides).
    /// See [`PropFirmGateConfig`]. (was the
    /// `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_*` env overrides)
    pub prop_firm_gate: PropFirmGateConfig,
}

impl Default for DiscoveryRuntimeConfig {
    fn default() -> Self {
        Self {
            // 240, matching config.yaml. See the field docs — the previous 50
            // silently contradicted the shipped config.
            prefilter_top_k: 240,
            prefilter_insample_frac: 0.80,
            prefilter_min_per_timeframe: 6,
            funnel_stage1_pct: 0.25,
            stage1_window: "earliest".to_string(),
            min_history_years: 0,
            // true, matching BOTH shipped config.yaml files. The static ladder
            // is calibrated for normalised features that normalisation never
            // produced; see the field docs.
            adaptive_thresholds: true,
            prop_firm_gate: PropFirmGateConfig::default(),
        }
    }
}

/// Prop-firm window-pass gate parameters — the config-driven replacement for
/// the `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_*` env overrides read by
/// `derive_prop_firm_gate`. The `Option` rule fields default to `None`,
/// meaning "use the FTMO baseline" (`PropFirmRiskRules::default` /
/// `FTMO_STANDARD`) — exactly reproducing the env-absent behaviour; set a
/// value to override that specific rule (e.g. to target a non-FTMO firm's
/// challenge from the UI / TUI).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PropFirmGateConfig {
    /// Max daily-loss fraction (e.g. `0.05` = 5%). `None` = FTMO baseline.
    /// (was `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_MAX_DAILY_LOSS_PCT`)
    pub max_daily_loss_pct: Option<f64>,
    /// Max overall-drawdown fraction (e.g. `0.10` = 10%). `None` = FTMO
    /// baseline. (was `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_MAX_DD_PCT`)
    pub max_overall_drawdown_pct: Option<f64>,
    /// Challenge profit-target fraction (e.g. `0.10` = 10%); `0` disables the
    /// target requirement. `None` = `FTMO_STANDARD` target. (was
    /// `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_PROFIT_TARGET_PCT`)
    pub profit_target_pct: Option<f64>,
    /// Minimum trading days the strategy must be active. `None` = FTMO
    /// baseline. (was `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_MIN_TRADING_DAYS`)
    pub min_trading_days: Option<usize>,
    /// Length (days) of each random evaluation window. Default `60` (the
    /// longest standard prop-firm phase). (was
    /// `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_WINDOW_DAYS`)
    pub window_days: usize,
    /// Number of random windows to score. `0` = auto-tune from dataset
    /// length. (was `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_N_WINDOWS`)
    pub n_windows: usize,
    /// Hard pass-rate floor in `[0, 1]`. `0` = ranking-only (no hard
    /// threshold). (was `NEOETHOS_BOT_DISCOVERY_PROP_FIRM_PASS_RATE`)
    pub pass_rate: f64,
}

impl Default for PropFirmGateConfig {
    fn default() -> Self {
        Self {
            max_daily_loss_pct: None,
            max_overall_drawdown_pct: None,
            profit_target_pct: None,
            min_trading_days: None,
            window_days: 60,
            n_windows: 0,
            pass_rate: 0.0,
        }
    }
}

/// Strategy-evaluation runtime knobs — the config-driven replacement for
/// the `NEOETHOS_BOT_PROP_*` cost-profile + SMC-weight env vars (symbol /
/// currency / pip-value / spread / commission overrides used by
/// `infer_market_cost_profile`, and the 12 SMC indicator weights +
/// gate threshold used by `EvaluationConfig::default`). Mirrors
/// `neoethos_search::genetic::StrategyEvaluationRuntimeOverrides`; the
/// search crate builds it via `from_settings(&Settings)`. Defaults MUST
/// match that struct's `Default` (a `from_settings(&Settings::default())
/// == default()` test enforces it). `None` cost fields mean "no
/// override" (production callers pass explicit values).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EvalRuntimeConfig {
    // ---------------------------------------------------------------------
    // WARNING LIBRARY FALLBACK, NEVER REACHED BY DISCOVERY (2026-08-10).
    //
    // `symbol` / `account_currency`: `system.symbol` and
    // `system.account_currency` win whenever non-empty (`discovery.rs:735`,
    // `:747`); these are a last-resort fallback that logs an error and returns
    // a NaN sentinel. Two `symbol:` keys ~1300 lines apart in one file.
    //
    // `spread_pips` / `commission_per_trade` (MONEY): `risk.backtest_spread_pips`
    // + `slippage_pips` + `commission_per_lot` win UNCONDITIONALLY -
    // `strategy_gene.rs:509-511` is a 4-step chain whose step (1) is filled by
    // `discovery.rs:754-756` on EVERY discovery run, and `:4086-4104` refuses a
    // non-finite override so the `.filter` can never fall through. These two
    // are nevertheless what the Settings screen renders as `cost.spread_pips` /
    // `cost.commission_per_trade`, with tuning presets - the UI advertises the
    // loser. Deleting them requires the search-side read to collapse to
    // `risk.*` in the same wave - routed to
    // `docs/pending-edits-forbidden-territory.md`.
    // ---------------------------------------------------------------------
    pub symbol: Option<String>,
    pub account_currency: Option<String>,
    pub pip_value: Option<f64>,
    pub quote_to_account_rate: Option<f64>,
    pub pip_value_per_lot: Option<f64>,
    pub spread_pips: Option<f64>,
    pub commission_per_trade: Option<f64>,
    pub reject_pip_fallback: bool,
    pub smc_gate_threshold: f64,
    pub smc_w_ob: f64,
    pub smc_w_fvg: f64,
    pub smc_w_liq: f64,
    pub smc_w_mtf: f64,
    pub smc_w_premium: f64,
    pub smc_w_inducement: f64,
    pub smc_w_bos: f64,
    pub smc_w_choch: f64,
    pub smc_w_eqh: f64,
    pub smc_w_eql: f64,
    pub smc_w_displacement: f64,
}

impl Default for EvalRuntimeConfig {
    fn default() -> Self {
        Self {
            symbol: None,
            account_currency: None,
            pip_value: None,
            quote_to_account_rate: None,
            pip_value_per_lot: None,
            spread_pips: None,
            commission_per_trade: None,
            // Refuse a pip value that cannot be converted into the account
            // currency, rather than booking a foreign-currency amount as if it
            // were account currency. That fallback inflated every JPY-quoted
            // result about 192-fold on a GBP account and every USD-quoted one by
            // 27 %, and since the search ranks on profit, the inflated
            // candidates won selection and reached live trading — where they
            // earned what they were actually worth. Nothing in the reported
            // numbers revealed it. Set to `false` only to reproduce an old run.
            reject_pip_fallback: true,
            smc_gate_threshold: 0.75,
            smc_w_ob: 1.0,
            smc_w_fvg: 1.0,
            smc_w_liq: 1.0,
            smc_w_mtf: 1.0,
            smc_w_premium: 1.0,
            smc_w_inducement: 1.0,
            smc_w_bos: 1.0,
            smc_w_choch: 1.0,
            smc_w_eqh: 1.0,
            smc_w_eql: 1.0,
            smc_w_displacement: 1.0,
        }
    }
}

/// Strategy-quality scoring knobs — config-driven replacement for the
/// `NEOETHOS_BOT_PROP_MIN_TRADES_PER_MONTH` /
/// `NEOETHOS_BOT_TRADING_DAYS_PER_MONTH` env vars. Mirrors
/// `neoethos_search::quality::QualityRuntimeOverrides`; a
/// `from_settings(&Settings::default()) == default()` test enforces the
/// matching defaults.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct QualityRuntimeConfig {
    /// Minimum trades a calendar month needs to count toward monthly
    /// win-rate / avg-return scoring.
    pub min_trades_per_month: usize,
    /// Trading days per month used to convert observed trading days into
    /// a months-traded estimate.
    pub trading_days_per_month: f64,
}

impl Default for QualityRuntimeConfig {
    fn default() -> Self {
        Self {
            min_trades_per_month: 4,
            trading_days_per_month: 21.0,
        }
    }
}

/// Adaptive-stop (stop-target) runtime knobs. Mirrors
/// `neoethos_search::stop_target::StopTargetRuntimeOverrides`; a
/// `stop_target_from_settings_default_matches_default` test enforces the
/// matching defaults.
///
/// HISTORY — why this key exists again. `tail_max_bars` was a config key from
/// v0.4.19 until commit `48abfc90` (2026-06-06) removed it, correctly, as
/// "dead in config": the value never reached `StopTargetSettings`, which used
/// a hardcoded `300_000`. What the removal did not notice is that the
/// hardcoded number was not inert. Above it the rolling expected-shortfall
/// series was skipped and the caller substituted a tail distance of ZERO, so
/// the `1.25 ×` tail term silently vanished from the stop. Measured on EURUSD
/// M5 (`data.vortex`, 1 054 320 bars): 300 000 bars ⇒ median base stop
/// 18.09 pips, 300 001 bars ⇒ 5.81 pips. One extra bar, 3.11×.
///
/// That made the production callers of the same function disagree — and the
/// first write-up of WHICH callers was itself wrong, so here is the measured
/// map (EURUSD M5, operator's own file, at the spans the code actually
/// produces; an adversarial review re-derived every number):
///
///   GA scoring     stage-1 window   210 864 bars  tail ON   21.68 pips
///   MC screen      in-sample slice  843 456 bars  tail OFF   6.57 → 20.00
///   sensitivity    in-sample slice  843 456 bars  tail OFF   6.57 → 20.00
///   walk-forward   window slice      42 172 bars  tail ON   23.35
///   live loop      rolling buffer     1 000 bars  tail ON   12.96
///
/// So scoring was NOT the odd one out (the GA never sees the full series —
/// the funnel slices it to the earliest 25 % of the 80 % in-sample). The
/// divergent lanes were the MONTE-CARLO ROBUSTNESS SCREEN and the SENSITIVITY
/// SCREEN: every candidate was quality-screened against a ~6.6-pip stop while
/// being scored on ~21.7 and traded on ~13-23. Removing the cap moves ONLY
/// those two lanes (6.57 → 20.00, 3.04×); scoring, walk-forward and live move
/// by under 0.1 %.
///
/// Cost of `tail_step = 1`, honestly: the base series is rebuilt once per GA
/// GENERATION (nothing caches it), once per MC chunk, once per sensitivity
/// chunk, per walk-forward split and per CPCV fold — measured +78 ms per
/// generation at the stage-1 span and ~533 ms per MC/sensitivity chunk. Not
/// "once per combo" as an earlier note claimed.
///
/// One behaviour change beyond the cap: a DEGENERATE slice (non-finite or
/// non-positive median distance — e.g. a run of corrupt equal-price bars) now
/// aborts discovery with a named error instead of silently continuing on the
/// gene's fixed pips. Fail-loud is deliberate; the old silence is how 14 240
/// zero-price bars once produced a £77 211 phantom move without a whisper.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct StopTargetRuntimeConfig {
    /// Hard cap on how many bars the rolling expected-shortfall series will
    /// process. `0` (the default) means NO cap — every series gets the tail
    /// term, so scoring, walk-forward and live compute the same base stop.
    ///
    /// A non-zero value no longer degrades silently: a series longer than the
    /// cap is a named error carrying both numbers, never a zero tail term.
    /// Set one only to bound cost, and expect to be told when it bites.
    pub tail_max_bars: usize,
    /// Sample the rolling expected shortfall every `tail_step` bars, carrying
    /// the value forward in between. `1` (the default) = every bar.
    ///
    /// Also a correctness knob, not only a speed one. The sampling grid is
    /// anchored at the START of whatever slice it is handed, so `> 1` makes
    /// the tail term depend on where the caller's series begins. Measured on
    /// EURUSD M5 at the old default of 5: the base over the trailing 300 001
    /// bars differs from the same bars of the full-series base by up to 86 %
    /// per bar while the medians agree to 0.006 %. The live loop's rolling
    /// buffer shifts its start every bar, so that fires continuously.
    ///
    /// `1` costs 574 ms instead of 206 ms over 1 054 320 bars, once per combo,
    /// and moves every median by under 0.02 %.
    pub tail_step: usize,
}

impl Default for StopTargetRuntimeConfig {
    fn default() -> Self {
        Self {
            tail_max_bars: 0,
            tail_step: 1,
        }
    }
}

/// The exit geometry discovery evaluates every candidate under.
///
/// WHY THIS EXISTS (2026-08-09). Until today these four numbers were literals
/// inside `EvaluationConfig::for_symbol` (`strategy_gene.rs:849-853`) —
/// `trailing_enabled: true`, `trailing_be_trigger_r: 1.0`,
/// `trailing_atr_multiplier: 1.0` — with a comment calling them an operator
/// mandate. Nothing could switch them off, so nothing ever measured the search
/// without them.
///
/// What they did, measured on real EURUSD bars: `trailing_atr_multiplier`
/// despite its name is NOT an ATR multiple, it is a multiple of the position's
/// own stop distance (`eval.rs:1030-1035`), and the trail is applied BEFORE the
/// take-profit check on every bar after entry. At trigger 1.0 / multiple 1.0 the
/// stop sits at entry the instant a trade touches +1R, so reaching 3R requires
/// climbing 1R→3R without ever giving back 1R from the running high. The
/// realised payoff was 0.87 at sl 6 / tp 45 AND 0.87 at sl 6 / tp 300 — average
/// win 6.10 vs 6.11 pips. The take-profit was dead code. Highest payoff observed
/// anywhere in the full grid, all timeframes: 1.08, against a configured floor of
/// 2.0. Zero of 174 screened candidates could survive, before a bar was read.
///
/// WHAT THIS IS NOT. Turning the trail off has ZERO prior expected value in
/// money. Measured across every trailing configuration, expectancy stayed at
/// -4.15 pips per trade while the payoff ratio moved from 0.91 to 2.53. Exit
/// geometry redistributes the (win-rate, payoff) split; on a driftless price the
/// product is fixed at -cost. The value of this knob is DIAGNOSTIC: it makes the
/// question askable. Do not read a payoff improvement here as an edge.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ExitPolicyConfig {
    /// Whether the break-even + trailing stop is active in discovery.
    ///
    /// DEFAULT `false` — the opposite of the old hardcode. What this now
    /// PERMITS: a take-profit that can actually be reached, so a payoff ratio
    /// above ~1.1 is expressible at all. What it REFUSES: the automatic
    /// break-even protection the old comment credited with lowering drawdown.
    /// Both effects are real; only the first one was ever measured.
    pub trailing_enabled: bool,
    /// Profit, in multiples of the position's initial stop distance, that must
    /// be reached before the trail arms at all.
    pub trailing_be_trigger_r: f64,
    /// How far behind the running extreme the armed trail sits, as a multiple of
    /// the position's initial stop distance.
    ///
    /// Named `trailing_atr_multiplier` in `BacktestSettings` / `EvaluationConfig`
    /// for historical reasons. It has never been an ATR multiple. The name is
    /// kept there because the CUDA kernel and the cubecl kernel both bind to it;
    /// renaming is a kernel-coupled change and does not belong in this one.
    pub trailing_stop_multiplier: f64,
    /// Floor, in pips, on the profit the armed trail locks in.
    pub trailing_min_lock_pips: f64,
}

impl Default for ExitPolicyConfig {
    fn default() -> Self {
        Self {
            // OFF. See the type doc for the measurement.
            trailing_enabled: false,
            trailing_be_trigger_r: 1.0,
            trailing_stop_multiplier: 1.0,
            // The shared constant, not a literal: since the `risk.trailing_*`
            // shadows were deleted (#206) this struct is the ONLY owner of the
            // trail geometry, and the backtest, the CUDA kernel and live
            // trading all take it from here.
            trailing_min_lock_pips: DEFAULT_TRAILING_MIN_LOCK_PIPS,
        }
    }
}

/// The stop/target band the GA is allowed to draw and mutate within, expressed
/// in MULTIPLES OF THE DATASET'S OWN TYPICAL BAR RANGE rather than absolute pips.
///
/// WHY. `evolution_math.rs` clamped every gene to `sl ∈ [6, 20]` pips and
/// `tp ∈ [12, 45]` pips, and sampled reward:risk only in `[1.5, 2.5]`. Those are
/// M5 numbers. On H1 (ATR ≈ 12 pips) a 6-pip stop is inside the spread; on H4
/// (ATR ≈ 30 pips) the entire band is below one bar's range. So "move to a higher
/// timeframe" was not advice the search could act on — the higher timeframes were
/// literally inexpressible.
///
/// It also excluded the reward:risk the payoff floor demanded. With barriers only
/// the payoff is `(tp - c) / (sl + c)`, so a floor of 2.0 needs `tp >= 2·sl + 3·c`.
/// At the charged cost of c = 2.89 pips a 20-pip stop needs `tp >= 48.67` — outside
/// the 45-pip ceiling, and the initialiser never sampled a reward:risk above 2.5
/// anyway.
///
/// The unit is the median ATR of the dataset being searched, measured per run and
/// installed by discovery. When no scale has been installed the absolute
/// pre-2026-08-09 band is used verbatim, so a caller outside discovery
/// (`neoethos-models`' own GA) behaves exactly as before.
///
/// Same warning as [`ExitPolicyConfig`]: widening the band has no prior expected
/// value in money. It changes which shapes are REACHABLE, not whether they pay.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct GeneStopBoundsConfig {
    /// When `false`, gene generation and mutation use the absolute pip band
    /// below and ignore the dataset's ATR entirely (the pre-2026-08-09 behaviour,
    /// kept reachable for reproducing an old run).
    pub atr_scaled: bool,
    /// Tightest stop the GA may draw, in ATR units.
    pub sl_min_atr: f64,
    /// Widest stop the GA may draw, in ATR units.
    pub sl_max_atr: f64,
    /// Lowest reward:risk the initialiser samples.
    pub rr_min: f64,
    /// Highest reward:risk the initialiser samples. Raised from the old 2.5 so
    /// the reward:risk a 2.0 payoff floor demands is inside the search space
    /// instead of outside it.
    pub rr_max: f64,
    /// Absolute pip band used when `atr_scaled` is false or no ATR scale has
    /// been installed for the run. These four are the literals that were in
    /// `evolution_math.rs` before this config existed.
    pub sl_min_pips: f64,
    pub sl_max_pips: f64,
    pub tp_min_pips: f64,
    pub tp_max_pips: f64,
}

impl Default for GeneStopBoundsConfig {
    fn default() -> Self {
        Self {
            atr_scaled: true,
            // EURUSD M5 ATR ≈ 5 pips, so [1.0, 4.0] ATR reproduces the old
            // [6, 20]-pip band on the timeframe it was tuned for, and means the
            // same thing on H1 (≈ [12, 48] pips) and H4 (≈ [30, 120]).
            sl_min_atr: 1.0,
            sl_max_atr: 4.0,
            rr_min: 1.5,
            // 4.0, not 2.5: `tp >= 2·sl + 3·c` at c = 2.89 pips needs 2.43 at
            // sl = 20 and 3.95 at sl = 6. A 2.0 payoff floor with rr_max = 2.5
            // is a gate no draw can clear on a small stop.
            rr_max: 4.0,
            sl_min_pips: 6.0,
            sl_max_pips: 20.0,
            tp_min_pips: 12.0,
            tp_max_pips: 45.0,
        }
    }
}

/// How the LIVE soft-voting ensemble combines its experts.
///
/// WHY THIS EXISTS (2026-08-10, audit #168). `SoftVotingEnsembleConfig` carried
/// these knobs with no way to set any of them: the only production builder was
/// `build_ensemble_for_symbol`, which handed the aggregator
/// `SoftVotingEnsembleConfig::default()`. So `expert_weights` was empty on every
/// install and all ~33 loaded experts voted at exactly 1.0 — including the ones
/// the operator would have discounted. A `_with_config` twin existed and was
/// called by nothing; it is gone, and this is its recipient.
///
/// MONEY PATH, and the size direction only. The ensemble never picks a
/// direction — the genes do (`live_trading.rs`); its output scales position
/// size. Down-weighting an expert therefore cannot open a trade that would not
/// have opened; it can change how much is committed to one.
///
/// THE DEFAULT IS EXACTLY THE OLD BEHAVIOUR: no weights, no exclusions, the same
/// two anomaly knees. Nothing moves until the operator writes a weight.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EnsembleVotingConfig {
    /// Per-expert vote weight, `canonical expert name` → `weight`. An expert
    /// not named here votes at 1.0; the weights are normalised per row, so only
    /// their RATIOS matter and a uniform map is the same as an empty one.
    ///
    /// OWNER: the operator. There is no defensible machine-chosen default —
    /// validation accuracy is measured per training run and per symbol, and
    /// baking one run's ranking into the shipped config would be exactly the
    /// invented number this codebase refuses. Empty means "no opinion", which
    /// is the honest state until he measures one.
    pub expert_weights: std::collections::BTreeMap<String, f64>,
    /// Canonical expert names that must NOT vote even when their artifact
    /// loaded. Distinct from a 0.0 weight only in intent: a 0.0 weight still
    /// counts the expert as a voter for the "N voting" log line.
    pub excluded_experts: Vec<String>,
    /// Raw `isolation_forest` anomaly score below which no size penalty is
    /// applied (scale 1.0).
    pub anomaly_lo: f64,
    /// Raw anomaly score at or above which the anomaly scale hard-vetoes the
    /// trade to size 0. Must be above [`Self::anomaly_lo`]; the two are
    /// validated together by [`Self::validate`].
    pub anomaly_hi: f64,
}

impl Default for EnsembleVotingConfig {
    fn default() -> Self {
        Self {
            // Empty: all experts at 1.0. See the field doc for why no ranking
            // is shipped.
            expert_weights: std::collections::BTreeMap::new(),
            excluded_experts: Vec::new(),
            anomaly_lo: 0.5,
            // 0.9 matches the trained ~0.95-quantile threshold.
            anomaly_hi: 0.9,
        }
    }
}

impl EnsembleVotingConfig {
    /// Reject a configuration whose knees are inverted or whose weights are not
    /// usable numbers, by name, rather than letting it scale live position size
    /// into nonsense.
    pub fn validate(&self) -> Result<(), String> {
        if !self.anomaly_lo.is_finite() || !self.anomaly_hi.is_finite() {
            return Err(format!(
                "models.ensemble_voting.anomaly_lo / anomaly_hi must be finite numbers \
                 (got {} and {})",
                self.anomaly_lo, self.anomaly_hi
            ));
        }
        if self.anomaly_hi <= self.anomaly_lo {
            return Err(format!(
                "models.ensemble_voting.anomaly_hi ({}) must be ABOVE anomaly_lo ({}) — below \
                 the low knee there is no penalty and at the high knee the trade is vetoed, so \
                 an inverted pair vetoes everything",
                self.anomaly_hi, self.anomaly_lo
            ));
        }
        for (name, weight) in &self.expert_weights {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(format!(
                    "models.ensemble_voting.expert_weights[{name}] = {weight} — a vote weight \
                     must be a finite number at or above zero"
                ));
            }
        }
        Ok(())
    }
}

/// Backtest-evaluation runtime knobs — config-driven replacement for the
/// `NEOETHOS_BOT_BACKTEST_*` + `NEOETHOS_BOT_RUST_THREADS` env vars.
/// Mirrors `neoethos_search::eval::BacktestRuntimeOverrides`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BacktestRuntimeConfig {
    /// Starting equity for canonical backtest PnL accounting (> 0).
    pub initial_equity: f64,
    /// Max monthly PnL buckets retained for consistency math (> 0).
    pub month_capacity: usize,
    /// One-release read-only compatibility cap. New configurations use
    /// `system.hardware.cpu_budget`; this legacy field is accepted with a WARN
    /// but is never written back out.
    #[serde(skip_serializing)]
    pub rayon_threads: Option<usize>,
}

impl Default for BacktestRuntimeConfig {
    fn default() -> Self {
        Self {
            initial_equity: 100_000.0,
            month_capacity: 240,
            rayon_threads: None,
        }
    }
}

/// Seen-signature memory knobs — config-driven replacement for the
/// `NEOETHOS_BOT_PROP_SEEN_*` env vars (dedup-memory flush cadence,
/// load/entry caps, and on-disk path). Mirrors
/// `neoethos_search::genetic::SeenSignatureMemoryRuntimeOverrides`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct SeenSignatureRuntimeConfig {
    pub flush_every: usize,
    pub load_max: usize,
    /// `0` → unbounded (`usize::MAX`); otherwise the entry cap.
    /// `0` means DERIVE FROM AVAILABLE MEMORY - it does NOT mean "unbounded".
    ///
    /// Changed 2026-08-10. `0` is exactly what an operator types for "no
    /// limit", and the old reading grew a `HashSet<u64>` + `VecDeque<u64>` for
    /// the whole run. WARNING: eviction is FIFO, so a LOWERED cap silently
    /// re-admits previously-seen genes and CHANGES WHAT THE RUN EXPLORES - the
    /// effective cap and the first eviction are logged by the search crate.
    pub max_entries: usize,
    /// Optional on-disk seen-signature file. Empty / unset → in-memory only.
    pub file_path: Option<String>,
}

impl Default for SeenSignatureRuntimeConfig {
    fn default() -> Self {
        Self {
            flush_every: 4096,
            load_max: 3_000_000,
            max_entries: 3_000_000,
            file_path: None,
        }
    }
}

/// Search-memory + weekly-refresh knobs (2026-06-06). When `enabled`, each
/// discovery run reads a per-symbol/TF on-disk **ledger** of previously found
/// strategies (indicator + SMC-flag combos + fitness) and seeds the GA's
/// seen-signature memory with their hashes so the next run AVOIDS
/// re-discovering them — every weekly run ADDS new diverse strategies to a
/// growing library. Mirrors the nested-config pattern of
/// [`DiscoveryRuntimeConfig`]; consumed via
/// `neoethos_search::DiscoveryConfig::from_settings`.
///
/// Cross-run dedup of the seeded hashes only takes effect for the GA when an
/// on-disk seen-signature file is configured (`seen_signature_runtime.file_path`):
/// the genetic engine builds its own `SeenSignatureMemory::current()` and reads
/// previously-persisted hashes from that file. When `file_path` is unset
/// (in-memory only, the default), the ledger is still recorded + the seed step
/// runs, but the seeded hashes are not visible to the engine's fresh in-memory
/// set — set a `file_path` to get true cross-run dedup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct DiscoveryLedgerConfig {
    /// Master switch. When `false`, discovery behaves byte-identically to a
    /// build without this feature (no ledger read, no seed, no ledger write).
    pub enabled: bool,
    /// Directory the per-symbol/TF ledger JSON files live in. Relative paths
    /// resolve against the process CWD (same convention as `cache/features`).
    /// WARNING NAMING TWIN of `system.cache_dir` - genuinely different
    /// artifacts and different types (`PathBuf` vs a relative `String`). KEEP
    /// BOTH; this one should be renamed `ledger_dir`. The rename needs
    /// `neoethos-search::discovery` in the same wave - routed to
    /// `docs/pending-edits-forbidden-territory.md`.
    pub cache_dir: String,
    /// How many top archive (non-portfolio) genes to also record per run, so
    /// the seen-set grows beyond just the promoted portfolio.
    pub archive_top_n: usize,
    /// Promotion policy for `discovery-promote-weekly`. `"additive"` (the
    /// default + only implemented policy) merges new genes by hash and keeps
    /// existing ones; unknown values fall back to additive.
    pub promotion_policy: String,
}

impl Default for DiscoveryLedgerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_dir: "cache/search".to_string(),
            archive_top_n: 20,
            promotion_policy: "additive".to_string(),
        }
    }
}

/// SMC (smart-money-concept) search-injection knobs — config-driven
/// replacement for the `NEOETHOS_BOT_PROP_SMC_*` env vars (the per-flag
/// enable probabilities, the force-ratio + min-flags that seed each GA
/// generation with SMC-aware genes, and the master `force_enabled`
/// toggle). Mirrors `neoethos_search::genetic::SmcSearchConfig`
/// (probabilities are clamped to `[0,1]`; `force_enabled = false` zeroes
/// `force_ratio` + `min_flags`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SmcSearchRuntimeConfig {
    pub force_ratio: f64,
    pub min_flags: usize,
    /// Master toggle — `false` disables SMC forcing (zeroes force_ratio +
    /// min_flags). Was `NEOETHOS_BOT_PROP_SMC_FORCE_ENABLED`.
    pub force_enabled: bool,
    pub p_ob: f64,
    pub p_fvg: f64,
    pub p_liq: f64,
    pub p_premium: f64,
    pub p_inducement: f64,
    pub p_mtf: f64,
    pub p_bos: f64,
    pub p_choch: f64,
    pub p_eqh: f64,
    pub p_eql: f64,
    pub p_displacement: f64,
}

impl Default for SmcSearchRuntimeConfig {
    fn default() -> Self {
        Self {
            force_ratio: 0.30,
            min_flags: 1,
            force_enabled: true,
            p_ob: 0.50,
            p_fvg: 0.50,
            p_liq: 0.50,
            p_premium: 0.50,
            p_inducement: 0.50,
            p_mtf: 0.85,
            p_bos: 0.50,
            p_choch: 0.50,
            p_eqh: 0.50,
            p_eql: 0.50,
            p_displacement: 0.50,
        }
    }
}

/// Where the multi-timeframe feature cube is assembled.
///
/// This replaces `NEOETHOS_FEATURE_CUBE_MODE` (retired 2026-08-10), whose
/// `ram` arm returned BEFORE the free-RAM check and so could put a cube larger
/// than the machine into memory — a failure wearing the costume of a choice.
///
/// The RAM and disk assemblies are required to produce BIT-IDENTICAL cubes, so
/// this knob is a performance/robustness choice, not an arithmetic one. It is
/// nonetheless recorded in the discovery run profile
/// (`/execution/feature_cube_mode`) because a run that silently went to disk
/// and a run that stayed in RAM are otherwise indistinguishable after the fact,
/// and because a knob that CAN move the answer must never be invisible again
/// (the `NEOETHOS_GPU_F64` failure mode).
///
/// # THERE IS NO `ram`, AND THAT IS THE POINT
///
/// The retired variable accepted `ram | disk | auto`. This field accepts
/// `auto | disk` only, and a file that says `ram` fails to load naming the
/// field and the two accepted values. Two reasons, and both matter:
///
/// 1. **Forcing RAM is the defect that was removed.** It is the only input in
///    the whole assembly that could put more bytes in memory than the machine
///    reports having.
/// 2. **A clamped `ram` would be a lever that changes nothing.** If the probe
///    still binds — and it must — then `ram` and `auto` produce the same
///    decision for every cube on every machine. Shipping it would be the same
///    disease under a new name: a control the operator believes is live, that
///    decides nothing.
///
/// `disk` is kept, and it is not symmetric with `ram`: it can only LOWER peak
/// memory, so there is nothing for the probe to veto. It is the escape hatch
/// for the case where the probe over-reports — a container whose cgroup limit
/// is below the host RAM `available_memory_bytes()` sees, which is exactly the
/// shape of a rented box.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum FeatureCubeMode {
    /// Derive from the free-RAM probe (the shipped answer, and what every run
    /// with the env var unset has always done).
    #[default]
    Auto,
    /// Always stream to the disk-mmap store, even when the cube would fit.
    /// Slower; always honoured; can only reduce peak memory.
    Disk,
}

impl FeatureCubeMode {
    /// Stable lowercase spelling, for the run profile and for logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Disk => "disk",
        }
    }
}

/// Data-layer behavior knobs — config-driven replacement for the
/// `NEOETHOS_BOT_NORMALIZE_FEATURES` / `NEOETHOS_FEATURE_CUBE_MODE` env vars.
/// Consumed by the data crate via
/// `neoethos_data::install_data_runtime_overrides(...)` and
/// `neoethos_data::install_feature_cube_policy(...)` at startup.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct DataRuntimeConfig {
    /// Per-column robust z-score normalization of the feature matrix before the
    /// GA search (was `NEOETHOS_BOT_NORMALIZE_FEATURES`).
    ///
    /// **Default flipped to `true` on 2026-08-09, and this one is not cosmetic.**
    ///
    /// The GA's signal is `combined = Σ wᵢ · featureᵢ`, with `wᵢ` drawn from
    /// `{0.2, 0.4, 0.6, 0.8, 1.0}` (optionally negated) — a 5:1 span. Raw
    /// feature magnitudes span about 1e5:1 on a single symbol: RSI ≈ 50,
    /// an EMA ≈ 1.08, an M5 ATR ≈ 5e-4. A 5:1 weight cannot reorder terms that
    /// differ by 1e5, so **a multi-indicator gene was arithmetically equal to
    /// its single largest-magnitude term** and every other weight the GA
    /// searched was decoration. `normalization.rs` says the same thing in its
    /// own module docs, and names the empty-portfolio bug it produced on EURJPY
    /// (magnitudes ±3.5e11) and XAUUSD.
    ///
    /// With this on, `normalize_feature_column_f64` applies a robust per-column
    /// z-score — `(x − median) / (1.4826·MAD)`, clipped to ±10 — fitted only on
    /// the exact in-sample row range supplied by the split contract. No 80%
    /// window or other fit range is inferred inside the normalizer.
    ///
    /// ## What this changes, stated plainly
    ///
    /// 1. **Every gene threshold now means what the static ladder always said
    ///    it meant.** `evolution_math.rs:560` calls that ladder "Calibrated for
    ///    z-score-normalised features"; until now nothing produced them.
    /// 2. **Invalid cells never become numeric zero.** Warmup, missing input,
    ///    gaps, stale alignment, zero denominators and non-finite results retain
    ///    their typed validity reason and canonical NaN payload. Fits and
    ///    transforms consume only explicitly valid cells, so alignment gaps
    ///    remain visible to search/model gates instead of becoming data.
    /// 3. **Prior artifacts are not comparable.** Anything fitted on the raw
    ///    cube — trained models, exported genes, saved thresholds — was fitted
    ///    on a different feature scale. Retrain and re-search; do not mix.
    ///
    /// It has ZERO expected value in money on its own. It does not create edge.
    /// It makes the search's own parameters mean something, which is the
    /// precondition for finding out whether there is any.
    pub normalize_features: bool,
    /// Where the multi-TF feature cube is assembled — see [`FeatureCubeMode`].
    /// Was `NEOETHOS_FEATURE_CUBE_MODE`, retired 2026-08-10 because its `ram`
    /// arm returned before the free-RAM check. `ram` is not an accepted value
    /// here; `auto | disk` are.
    pub feature_cube_mode: FeatureCubeMode,
}

impl Default for DataRuntimeConfig {
    fn default() -> Self {
        Self {
            // true, matching BOTH shipped config.yaml files. See the field docs:
            // without it the GA's 5:1 weight ladder cannot reorder terms that
            // differ by 1e5, so a multi-indicator gene equalled its largest term.
            normalize_features: true,
            // `Auto` reproduces exactly what every run got with the env var
            // unset, which is what the operator's live store has always had.
            feature_cube_mode: FeatureCubeMode::Auto,
        }
    }
}

/// Tree-model (LightGBM / XGBoost / CatBoost) device + training knobs —
/// config-driven replacement for the `NEOETHOS_BOT_TREE_DEVICE` / `_GPU_ONLY`
/// / `_EARLY_STOP_*` env vars and the `FOREX_GPU_COUNT` rebrand remnant.
/// Platform-standard GPU-selection knobs (`CUDA_VISIBLE_DEVICES`, …) are NOT
/// app config and stay honored. (The cross-cutting `cpu_threads` budget — read
/// in core/search/models, so it needs a single system-level knob — is a
/// separate follow-up.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TreeRuntimeConfig {
    /// Device preference for tree-model training: `"auto"` | `"cpu"` |
    /// `"gpu"` | `"cuda"` | `"cuda:N"`. `""` is treated as `"auto"`. Was
    /// `NEOETHOS_BOT_TREE_DEVICE` (the per-model `_{MODEL}_DEVICE` overrides
    /// are folded into this single global knob).
    pub device: String,
    /// Require GPU for tree training — no silent CPU fallback. Was
    /// `NEOETHOS_BOT_GPU_ONLY`.
    pub gpu_only: bool,
    /// Explicit GPU count; `None` = auto-detect (the standard
    /// `*_VISIBLE_DEVICES` vars, then `nvidia-smi` / `rocm`). Was the
    /// `FOREX_GPU_COUNT` rebrand remnant.
    pub gpu_count: Option<usize>,
    /// Early-stop patience override for tree-model training; `None` (the
    /// default) = use each model's built-in default. Was
    /// `NEOETHOS_BOT_EARLY_STOP_PATIENCE`.
    pub early_stop_patience: Option<usize>,
    /// Early-stop min-delta override; `None` = use the model's default.
    /// Was `NEOETHOS_BOT_EARLY_STOP_MIN_DELTA`.
    pub early_stop_min_delta: Option<f64>,
    /// Let LightGBM train on the CUDA tree learner when the build has it and
    /// the host has a card. `false` (the default) pins LightGBM to the CPU
    /// regardless of `device`.
    ///
    /// This is deliberately a separate knob from `device` and not a new
    /// spelling of it. `device` is what the operator WANTS across all tree
    /// models; this says whether LightGBM in particular is allowed to act on
    /// it. It defaults to `false` because flipping it changes which
    /// arithmetic trains the model — CUDA histogram construction sums in a
    /// different order than the CPU learner, so the same data and the same
    /// hyper-parameters produce a slightly different tree. That is a
    /// selection change, and selection changes are the operator's to make.
    ///
    /// Set `true` once a run on this host has been compared against a CPU
    /// run. On a build without the CUDA learner, or a host with no card,
    /// `true` is simply inert — it never silently degrades.
    pub lightgbm_gpu: bool,
}

impl Default for TreeRuntimeConfig {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            gpu_only: false,
            gpu_count: None,
            early_stop_patience: None,
            early_stop_min_delta: None,
            lightgbm_gpu: false,
        }
    }
}

impl Default for ModelsConfig {
    fn default() -> Self {
        let mut hpo_trials_by_model = HashMap::new();
        for (model, trials) in [
            ("lightgbm", 8),
            ("xgboost", 8),
            ("xgboost_rf", 6),
            ("xgboost_dart", 6),
            ("catboost", 8),
            ("catboost_alt", 6),
            ("mlp", 6),
            ("tabnet", 6),
            ("nbeats", 6),
            ("tide", 6),
            ("kan", 6),
            ("transformer", 6),
        ] {
            hpo_trials_by_model.insert(model.to_string(), trials);
        }

        Self {
            ml_models: vec![
                "lightgbm",
                "xgboost",
                "xgboost_rf",
                "xgboost_dart",
                "catboost",
                "catboost_alt",
                "sklears_tree",
                "mlp",
                "elasticnet",
                // Plain L2 logistic regression — trainable since day one but
                // absent from every request list, so it never existed on any
                // install (operator audit 2026-07-11). Cheap linear voter.
                "logistic",
                "bayes_logit",
                "online_pa",
                "online_hoeffding",
                "swarm_forecaster",
                "isolation_forest",
                "neat",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            use_rl_agent: true,
            use_sac_agent: true,
            use_rllib_agent: false,
            rllib_num_workers: 0,
            auto_enable_rllib: false,
            use_neuroevolution: true,
            rl_population_size: 5,
            rl_timesteps: 10_000_000,
            rl_eval_episodes: 15,
            rl_network_arch: vec![4096, 4096, 4096, 2048, 1024],
            rl_parallel_envs: 1,
            rl_state_bins: 255,
            rl_state_encoding: "normalized".to_string(),
            rl_update_interval: 0,
            rl_update_freq: 0,
            rl_learning_rate: 1e-3,
            rl_gamma: 0.99,
            rl_epsilon_start: 1.0,
            rl_epsilon_end: 0.02,
            rl_epsilon_decay: 0.995,
            rl_buffer_capacity: 0,
            rl_reward_horizon: 0,
            rl_episode_len: 0,
            rl_train_seconds: 3600,
            exit_agent_hidden_dim: 64,
            exit_agent_gamma: 0.99,
            exit_agent_epsilon: 0.20,
            exit_agent_epsilon_min: 0.05,
            exit_agent_epsilon_decay: 0.999,
            exit_agent_memory_capacity: 10_000,
            exit_agent_reward_horizon: 0,
            exit_agent_warmup_steps: 0,
            evo_train_seconds: 3600,
            evo_hidden_size: 64,
            evo_population: 32,
            evo_islands: 4,
            evo_sigma: 0.25,
            prop_search_enabled: false,
            prop_search_population: 100,
            prop_search_population_auto: true,
            prop_search_generations: 50,
            prop_search_max_hours: 0.5, // 2026-06-05: sane default (was 8.0=absurd 8h/combo); config-overridable (VPS budget run uses 0.25)
            prop_search_max_rows: 0,
            prop_search_max_rows_by_tf: HashMap::new(),
            prop_search_portfolio_size: 3000,
            prop_search_max_indicators: 12,
            prop_search_checkpoint: PathBuf::from("models/strategy_evo_checkpoint.json"),
            // Task #35 (2026-08-09): `auto` so the GA population eval uses the
            // GPU population lane (prototype B) when a card is present — `cpu`
            // pinned the whole GA (~97% of a run) to the CPU while validation
            // used the card, the 8-month asymmetry. Falls back to CPU with no card.
            prop_search_device: "auto".to_string(),
            prop_search_val_candidates: 0,
            prop_search_val_min_positive_months: 0,
            prop_search_val_min_trades_per_month: 0,
            prop_search_val_min_trades_per_day: 0.0,
            // Off by default: these express one operator's target, not a
            // universal truth about what a good strategy looks like.
            prop_search_min_win_rate: 0.0,
            // Operator decision A (2026-08-09): the search must select for
            // payoff ratio, not trade volume. 2.0 = only strategies whose
            // average win is at least 2x their average loss (the operator's
            // "ideally 2RR"). Enforced at discovery.rs (`TargetProfile`); an
            // empty portfolio at 2.0 is the honest "2RR is rare here" signal,
            // never silently relaxed. Was 0.0 (gate off) — the single reason
            // 16 months of runs kept selecting one-point-of-margin systems.
            prop_search_min_payoff_ratio: 2.0,
            // The primary gate (2026-08-09). `0.0` = strictly positive required.
            // This is the floor the payoff ratio was standing in for and could
            // not carry: payoff 2.53 at expectancy -4.18 pips/trade passes a 2.0
            // payoff floor and empties the account.
            prop_search_min_net_expectancy_per_trade: 0.0,
            // Sign only, by default. Raising this is an operator decision about
            // how much in-sample noise to tolerate, not a correctness bound.
            prop_search_min_expectancy_t_stat: 0.0,
            prop_search_max_in_market: 0.0,
            prop_search_val_min_monthly_profit_pct: 0.0,
            prop_search_val_log_trades: false,
            prop_search_val_trade_log_max: 20,
            prop_search_async: false,
            prop_search_async_wait: false,
            tree_device_preference: "auto".to_string(),
            regularized_model_defaults: true,
            heavy_booster_min_bars: 4000,
            ml_cpcv_enabled: true,
            prop_search_parent_selection: "rank".to_string(),
            prop_search_survivor_selection: "rank".to_string(),
            prop_search_survivor_fraction: 0.10,
            prop_search_immigrant_fraction: 0.18,
            prop_search_selection_temperature: 0.75,
            prop_search_tournament_size: 0,
            prop_search_opportunistic_enabled: true,
            prop_search_opportunistic_min_positive_months: 3,
            prop_search_opportunistic_min_trades_per_month: 10,
            prop_search_opportunistic_min_trade_return_pct: 4.0,
            prop_search_opportunistic_max_dd: 0.025,
            prop_search_use_opportunistic: true,
            // 2026-05-26 operator directive (dual-mode product): the 5 knobs
            // below were previously hardcoded in discovery.rs. Surfaced here
            // so the dual-mode product can tune them without rebuilds. The
            // defaults reproduce the previous hardcoded behavior.
            prop_search_corr_threshold: 0.85,
            prop_search_mc_runs: 100,
            prop_search_mc_min_profitable: 70,
            prop_search_sensitivity_spread_pips: 2.0,
            prop_search_sensitivity_commission_per_lot: 7.0,
            train_batch_size: 32,
            inference_batch_size: 32,
            enable_transformer_expert: true,
            transformer_heads: 8,
            transformer_layers: 4,
            transformer_hidden_dim: 256,
            transformer_dropout: 0.20,
            transformer_seq_len: 64,
            transformer_train_seconds: 3600,
            nbeats_train_seconds: 3600,
            tide_train_seconds: 3600,
            tabnet_train_seconds: 3600,
            kan_train_seconds: 3600,
            mlp_train_seconds: 3600,
            num_transformers: 2,
            swarm_memory_limit_mb: 256.0,
            swarm_horizon: 0,
            swarm_frequency: "H".to_string(),
            swarm_strategy: "bayesian".to_string(),
            swarm_online_learning: true,
            swarm_interpretability_needed: true,
            swarm_latency_ms: 0,
            hpo_backend: "ax".to_string(),
            hpo_trials: 8,
            hpo_trials_by_model,
            hpo_max_rows: 1_000_000,
            max_epochs_by_model: HashMap::new(),
            ray_tune_max_concurrency: 1,
            calibration_enabled: true,
            calibration_method: "platt".to_string(),
            calibration_min_rows: 300,
            live_ml_gate: false,
            // MUST stay equal to `neoethos_trader::DEFAULT_BLEND_GATE_FLOOR` /
            // `DEFAULT_BLEND_VETO_BELOW`. `neoethos-core` cannot depend on
            // `neoethos-trader`, so these are the literals, not an import.
            blend_gate_floor: 0.34,
            blend_veto_below: 0.15,
            model_param_overrides: HashMap::new(),
            regime_router_enabled: false,
            regime_router_min_models: 2,
            regime_trend_models: vec![
                "transformer",
                "patchtst",
                "timesnet",
                "nbeats",
                "nbeatsx_nf",
                "tide",
                "tide_nf",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            regime_range_models: vec![
                "tabnet",
                "lightgbm",
                "xgboost",
                "xgboost_rf",
                "xgboost_dart",
                "catboost",
                "catboost_alt",
                "elasticnet",
                "bayes_logit",
                "online_pa",
                "online_hoeffding",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            regime_neutral_models: Vec::new(),
            l1_feature_selection_enabled: false,
            l1_feature_selection_per_regime: false,
            l1_feature_selection_min_features: 20,
            l1_feature_selection_max_features: 256,
            l1_feature_selection_sample_limit: 200_000,
            l1_feature_selection_c: 0.20,
            filter_to_base_signal: true,
            global_max_rows: 0,
            global_max_rows_per_symbol: 0,
            symbol_hash_buckets: 32,
            global_train_ratio: 0.8,
            train_holdout_pct: 0.2,
            label_use_triple_barrier: true,
            // Symmetric by default: the asymmetric bracket manufactured a
            // 66/34 class prior and 14 constant-predictor models. See the
            // field docs; the old geometry stays reachable via "asymmetric".
            label_geometry: "symmetric".to_string(),
            label_horizon_bars: 0,
            label_neutral_band_atr_fraction: 0.25,
            label_stop_atr_multiplier: 0.0,
            label_take_profit_rr: 0.0,
            walkforward_splits: 10, // 2026-06-05: robust OOS default (was 20, slow); config-overridable
            embargo_minutes: 120,
            discovery_mode: "prop_firm".to_string(),
            // walk-forward export gate ON (robustness). prop-firm pass-rate floor
            // RE-CALIBRATED 0.65→0.40 (2026-06-06) to match the operator's >=4%/month
            // bar now used as the per-window target — see derive_prop_firm_gate +
            // config.yaml prop_firm_min_pass_rate. (see field docs above.)
            require_walkforward_for_export: true,
            prop_firm_min_pass_rate: 0.40,
            search_runtime: SearchRuntimeConfig::default(),
            discovery_runtime: DiscoveryRuntimeConfig::default(),
            eval_runtime: EvalRuntimeConfig::default(),
            quality_runtime: QualityRuntimeConfig::default(),
            backtest_runtime: BacktestRuntimeConfig::default(),
            stop_target_runtime: StopTargetRuntimeConfig::default(),
            exit_policy: ExitPolicyConfig::default(),
            gene_stop_bounds: GeneStopBoundsConfig::default(),
            ensemble_voting: EnsembleVotingConfig::default(),
            seen_signature_runtime: SeenSignatureRuntimeConfig::default(),
            discovery_ledger: DiscoveryLedgerConfig::default(),
            smc_search_runtime: SmcSearchRuntimeConfig::default(),
            data_runtime: DataRuntimeConfig::default(),
            tree_runtime: TreeRuntimeConfig::default(),
            statistical_device: "cpu".to_string(),
            prop_metric_weight: 1.0,
            prop_accuracy_weight: 0.1,
            prop_min_trades: 0,
            prop_conf_threshold: 0.55,
            enable_cpcv: true,
            cpcv_n_splits: 5,
            cpcv_n_test_groups: 2,
            cpcv_embargo_pct: 0.01,
            cpcv_purge_pct: 0.02,
            cpcv_min_phi: 0.80,
            cpcv_max_rows: 200000, // 2026-06-05: cap informational CPCV (was 0=full=heavy on full-data); config-overridable
            enable_ddp: false,
            enable_fsdp: false,
            ddp_world_size: 1,
            transformer_d_model: 256,
            transformer_n_heads: 8,
            transformer_n_layers: 4,
            nf_hidden_dim: 256,
            tide_hidden_dim: 256,
            nbeats_hidden_dim: 256,
            kan_hidden_dim: 256,
            kan_grid_size: 9,
            tabnet_hidden_dim: 64,
            phase5_filter_meta_blender: true,
            phase5_core_models: vec!["transformer", "nbeats", "tide", "tabnet", "kan"]
                .into_iter()
                .map(String::from)
                .collect(),
            // Delegated to the gate's own Default so there is exactly one
            // place these five thresholds are written down. A test asserts
            // the literal values, so a silent change here fails loudly.
            promotion_gate: crate::domain::promotion_gate::PromotionGateConfig::default(),
            demo_forward_gate: crate::domain::demo_gate::DemoForwardGateConfig::default(),
        }
    }
}

/// How the trading gate should treat high-impact news events.
///
/// Until #117 the only option was auto-pause; the runtime would block
/// new orders inside the kill window. Operators with directional
/// strategies (event-driven, breakout-on-news, news-fade) need the
/// opposite — explicit opt-in to trade through events. This enum
/// makes the choice an operator-driven setting instead of a baked
/// policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NewsTradingMode {
    /// Block new orders inside the blackout window of any high-impact
    /// event. Default — the safe choice.
    ///
    /// The window is fixed by the live gate, not configurable: 15 min
    /// before / 10 min after (`app_services::news_calendar::
    /// BLACKOUT_BEFORE_MS` / `BLACKOUT_AFTER_MS`), consulted from
    /// `entry_blackout_for` at `live_trading.rs:1078`. The former
    /// `news.news_kill_window_min` / `news.news_lookahead_minutes` knobs
    /// were deleted in the 2026-08-09 D3 purge: they reached only a
    /// `NewsFilter` nothing constructed, so they advertised control over
    /// this window that they never had.
    #[default]
    BlockOnNews,
    /// Allow orders through the kill window. The UI shows a banner
    /// while a high-impact event is imminent so the operator knows
    /// what they're flying into.
    AllowAlways,
    /// Don't block, but surface a prominent warning in the UI when
    /// inside the kill window. Suited to operators who want a head's-
    /// up but don't want the gate to override their judgment.
    WarnOnly,
}

impl NewsTradingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BlockOnNews => "block_on_news",
            Self::AllowAlways => "allow_always",
            Self::WarnOnly => "warn_only",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::BlockOnNews => "Pause during news (safe default)",
            Self::AllowAlways => "Play through news (event-driven strategies)",
            Self::WarnOnly => "Warn only — don't block",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "block_on_news" | "block" | "pause" => Some(Self::BlockOnNews),
            "allow_always" | "allow" | "play" => Some(Self::AllowAlways),
            "warn_only" | "warn" => Some(Self::WarnOnly),
            _ => None,
        }
    }
}

/// News and LLM configuration
///
/// Sealed against a second load path — see the SUB-STRUCT SEAL block below.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(remote = "Self", default, deny_unknown_fields)]
pub struct NewsConfig {
    /// How the trading gate handles incoming high-impact news.
    /// Operator-controlled; default `block_on_news` preserves the
    /// pre-#117 safe behaviour. See [`NewsTradingMode`].
    #[serde(default)]
    pub news_trading_mode: NewsTradingMode,
    pub news_calendar_enabled: bool,
    /// Economic-calendar provider id. The ONLY implemented provider is
    /// `forexfactory`; `news_calendar::fetch_calendar` rejects anything else
    /// with an actionable error rather than silently fetching ForexFactory
    /// while the operator believes another source is live.
    pub news_calendar_source: String,
    pub rss_feeds: Vec<String>,
}

impl Default for NewsConfig {
    fn default() -> Self {
        Self {
            news_trading_mode: NewsTradingMode::default(),
            news_calendar_enabled: true,
            news_calendar_source: NEWS_CALENDAR_FOREXFACTORY.to_string(),
            // Public, no-API-key financial NEWS feeds for the AI news
            // desk (GET /news/feed). Operator-editable in Settings → News.
            // NB: the economic *calendar* lives in `news_calendar_source`
            // (ForexFactory's ffcal XML is a custom calendar format, not
            // RSS), so it intentionally does NOT belong in this list.
            // Verified reachable 2026-06-30 (200 + XML). The old defaults
            // (dailyfx, forexlive) now 403/redirect; ForexFactory's ffcal is a
            // calendar, not RSS (see `news_calendar_source`). Reused as the
            // runtime fallback when a user's configured feeds all fail.
            rss_feeds: default_news_rss_feeds(),
        }
    }
}

/// App / server / trading-runtime knobs — config-driven replacement for the
/// `neoethos-app` env_overrides registry (HTTP server bind, cTrader
/// connection retry/backoff/timeout, partial-fill acceptance, chart-merge
/// quote side, PnL audit / circuit-breaker thresholds). The app installs
/// these into its `env_overrides` cache at startup so the trading layer reads
/// the single config instead of `std::env`. Clamping is applied by the
/// getters (same bounds the env readers used).
///
/// Sealed against a second load path — see the SUB-STRUCT SEAL block below.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(remote = "Self", default, deny_unknown_fields)]
pub struct AppRuntimeConfig {
    /// HTTP server bind address `host:port` (default `127.0.0.1:7423`).
    pub server_bind: String,
    /// cTrader execution read-timeout (seconds); 0 disables. Clamped [0,3600].
    pub ctrader_read_timeout_secs: u64,
    /// cTrader execution attempts (initial + retries). Clamped [1,5].
    pub ctrader_max_attempts: u32,
    /// cTrader retry backoff base (ms). Clamped [10,2000].
    pub ctrader_backoff_base_ms: u64,
    /// Accept partial fills as final (default false).
    pub ctrader_allow_partial_fill: bool,
    /// cTrader streaming poll attempts. Clamped [1,5].
    pub ctrader_stream_max_attempts: u32,
    /// cTrader streaming backoff base (ms). Clamped [10,2000].
    pub ctrader_stream_backoff_base_ms: u64,
    /// Chart-merge quote side (`mid`/`bid`/`ask`); empty → caller default.
    pub chart_merge_side: String,
}

impl Default for AppRuntimeConfig {
    fn default() -> Self {
        Self {
            server_bind: "127.0.0.1:7423".to_string(),
            ctrader_read_timeout_secs: 30,
            ctrader_max_attempts: 3,
            ctrader_backoff_base_ms: 200,
            ctrader_allow_partial_fill: false,
            ctrader_stream_max_attempts: 3,
            ctrader_stream_backoff_base_ms: 200,
            chart_merge_side: String::new(),
        }
    }
}

// ─── THE SUB-STRUCT SEAL ────────────────────────────────────────────────────
//
// `load_seal` seals `Settings`. Until 2026-08-10 it sealed the WRONG BOUNDARY:
// the sub-structs it is made of were `pub` with a DERIVED `Deserialize`, so a
// second load path built out of them COMPILED —
//
//     #[derive(Deserialize)]
//     struct Bypass { risk: RiskConfig, system: SystemConfig }
//     serde_yaml_ng::from_str::<Bypass>(bytes)
//
// — and it DIVERGED ON MONEY. The same bytes `risk: {preset: the5ers}` yield
// `daily_drawdown_limit` 0.032 / `total_drawdown_limit` 0.042 through the seal
// and 0.040 / 0.070 through the bypass, because the bypass never runs
// `reconcile_preset`. The bypass got the LOOSER limit under the correct firm
// label. No consumer existed in the tree, so it was latent; it is closed here
// before one appears.
//
// THE MECHANISM. `#[serde(remote = "Self")]` makes `#[derive(Deserialize)]`
// emit an *inherent* `fn X::deserialize<D>(D) -> Result<X, D::Error>` INSTEAD
// OF `impl Deserialize for X`. With no trait impl:
//
//   * `serde_yaml_ng::from_str::<RiskConfig>(..)` does not compile (E0277);
//   * a `#[derive(Deserialize)]` struct holding one does not compile either —
//     that is the exact bypass the verifier built, and it is a `compile_fail`
//     doctest on `RiskConfig` plus a source guard in
//     `tests/config_single_load_path.rs`.
//
// The ONE caller of each inherent parser is `SettingsWire`, via
// `#[serde(deserialize_with = ...)]`. That is the whole point: the parser still
// exists, but the only route into it is the loader, which then runs
// `reconcile_preset`, `validate_safety_bounds` and the money-path reports.
//
// WHAT IS STILL OPEN, NAMED RATHER THAN HIDDEN. The inherent fn inherits the
// struct's visibility, so it is `pub`: code that explicitly builds a
// `Deserializer` and names the inherent parser can still get a raw, un-
// reconciled value. That is a deliberate act naming the loader's private door,
// not the accidental `from_str` that shipped before, and
// `no_second_caller_of_the_inherent_parsers` in
// `tests/config_single_load_path.rs` fails the build if a second call site
// appears anywhere in the workspace.
//
// `ModelsConfig` is NOT sealed. Five `serde_yaml_ng::from_str::<ModelsConfig>`
// call sites live in `domain/demo_gate.rs` and `domain/promotion_gate.rs`
// (their own `#[cfg(test)]` blocks), which this change does not own. Sealing it
// without moving them is a build break, so the edit is written out in
// `docs/pending-edits-forbidden-territory.md`. `ModelsConfig` carries no
// preset re-derivation, so the divergence it exposes is the missing top-level
// retired-key prune, not a money number.
//
// The `Serialize` side is UNCHANGED behaviour: `remote` removes that trait impl
// too, so each is hand-written below to delegate to the inherent serializer.
// `X::serialize(self, s)` resolves to the INHERENT associated fn — inherent
// items are selected before trait items in `Type::name` path resolution — so
// this is delegation, not recursion. Serialization was never a load path and
// nothing about `Settings::save` changes.

impl Serialize for SystemConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        SystemConfig::serialize(self, serializer)
    }
}

impl Serialize for RiskConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        RiskConfig::serialize(self, serializer)
    }
}

impl Serialize for NewsConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        NewsConfig::serialize(self, serializer)
    }
}

impl Serialize for AppRuntimeConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        AppRuntimeConfig::serialize(self, serializer)
    }
}

pub use load_seal::{ConfigOverride, ConfigProvenance, ConfigSource, Settings};

/// **THE SINGLE RESOLUTION POINT.** The only place in the workspace where a
/// [`Settings`] value can come into existence.
///
/// ## Why this is a module and not a convention
///
/// Before 2026-08-10 `Settings` was an ordinary `#[derive(Deserialize)]`
/// struct with public fields. That meant three ways to obtain one, each
/// bypassing whatever the last one had learned:
///
/// 1. `Settings::load()` — the intended path;
/// 2. `serde_yaml_ng::from_str::<Settings>(..)` — used by the raw-YAML
///    endpoint validator, which therefore accepted `trailing_enabeld:` and
///    reported "saved (verbatim)";
/// 3. a `Settings { .. }` literal — nothing to stop a fourth loader.
///
/// The recorded lesson from the previous attempt at this fix is that
/// *migrating the call sites* reproduces the defect: the next author adds
/// path 4. So the seal is structural:
///
/// * [`Settings`] carries a **private** `provenance: ConfigProvenance` field.
///   A struct literal outside this module does not compile.
/// * [`ConfigProvenance`]'s own fields are private to this module and its only
///   constructor is `ConfigProvenance::record`, which **logs the source by
///   name**. Even a second loader written inside this file has to declare
///   where its bytes came from.
/// * `Deserialize` is **hand-written and is itself the seal** — it is not a
///   bypass around the loader, it *is* the loader. Every `from_str`,
///   `from_reader` and `from_value` in the workspace therefore runs the same
///   retired-key prune, the same unknown-key refusal, the same preset
///   re-derivation and the same money-path reports. There is nothing left to
///   route around.
///
/// The compiler enforces all three; there is no derived `Deserialize` left to
/// fall back to.
///
/// ## The boundary this originally got wrong (fixed 2026-08-10)
///
/// Sealing `Settings` is not the same as sealing the config. The five
/// sub-structs it is made of were `pub` with derived `Deserialize`, so a
/// `#[derive(Deserialize)] struct Bypass { risk: RiskConfig }` compiled and
/// read the same bytes to a DIFFERENT drawdown limit — it never ran
/// `reconcile_preset`, so `preset: the5ers` gave it FTMO's looser 0.040 under
/// The5%ers' label. Four of the five now have no `Deserialize` impl at all and
/// are parsed only by `SettingsWire`'s `deserialize_with` attributes; see the
/// SUB-STRUCT SEAL block above this module for the mechanism, the one door left
/// open by design, and why `ModelsConfig` is still unsealed.
mod load_seal {
    use super::{
        AppRuntimeConfig, ModelsConfig, NewsConfig, RiskConfig, SystemConfig, user_config_path,
    };
    use crate::domain::prop_firm::{PropFirmConstraints, PropFirmPreset, PropFirmRuntimeDefaults};
    use serde::{Deserialize, Deserializer, Serialize};
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    // ─── noise control ──────────────────────────────────────────────────────
    // `Settings::from_yaml` is called ~40 times per app session (every route
    // re-reads the file). Emitting the same money-path finding 40 times is how
    // a real finding becomes wallpaper. Each DISTINCT message is emitted once
    // per process; nothing is ever downgraded or suppressed by content.
    pub(super) fn say_once(key: String, emit: impl FnOnce()) {
        static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut guard = match seen.lock() {
            Ok(g) => g,
            // A poisoned mutex must never silence a money-path report.
            Err(poisoned) => poisoned.into_inner(),
        };
        if guard.insert(key) {
            drop(guard);
            emit();
        }
    }

    /// Which of the four historical config surfaces produced this `Settings`.
    ///
    /// §8 of the 2026-08-09 knob pass: nothing in the workspace logged which
    /// file a run had opened, and two subsystems in one process were observed
    /// reading two different files. Every value below is logged by name at the
    /// moment it is chosen.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ConfigSource {
        /// No file was read — `Settings::default()`, the compiled defaults.
        CompiledDefaults,
        /// `$CONFIG_FILE` was set and pointed here.
        EnvConfigFile,
        /// The operator's live store (`%LOCALAPPDATA%\neoethos\config.yaml`
        /// and its POSIX equivalents). **This is what a real run reads.**
        UserStore,
        /// The operator's store, RELOCATED by `NEOETHOS_USER_DATA_DIR`.
        ///
        /// This is the *second* env input that can move WHICH FILE the process
        /// reads (`$CONFIG_FILE` is the first). It gets its own name in the log
        /// instead of hiding inside [`Self::UserStore`], because "the config
        /// you edited is not the config this run read" is precisely the failure
        /// this enum exists to make impossible.
        UserStoreRedirected,
        /// A caller-supplied path (`Settings::from_yaml`).
        ExplicitPath,
        /// Deserialized from bytes with no path: the raw-YAML editor's schema
        /// check, tests, an embedded document.
        InMemory,
    }

    impl ConfigSource {
        pub fn as_str(self) -> &'static str {
            match self {
                Self::CompiledDefaults => "compiled_defaults",
                Self::EnvConfigFile => "env:CONFIG_FILE",
                Self::UserStore => "user_store",
                Self::UserStoreRedirected => "user_store:NEOETHOS_USER_DATA_DIR",
                Self::ExplicitPath => "explicit_path",
                Self::InMemory => "in_memory",
            }
        }
    }

    /// Always written to the operator's store, never pruned. Each one governs
    /// how much money can be lost or committed; none may move because a default
    /// moved. A limit he can read in his own file is worth the extra lines.
    pub(crate) const ALWAYS_PERSIST: &[&str] = &[
        "risk.daily_drawdown_limit",
        "risk.total_drawdown_limit",
        "risk.risk_per_trade",
        "risk.risky_max_risk_per_trade",
        "risk.max_portfolio_risk",
        "risk.preset",
        "risk.require_stop_loss",
        "risk.min_risk_reward",
        "system.trading_mode",
        "system.account_currency",
    ];

    /// One leaf of the operator's store set against the default it shadows.
    ///
    /// See [`Settings::overrides_against_defaults`] for why this is the only
    /// honest question a full-snapshot config file can answer.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ConfigOverride {
        /// Dotted path, e.g. `risk.max_portfolio_risk`.
        pub path: String,
        /// What the operator's file says.
        pub live: String,
        /// What the compiled default says. `None` means the current schema has
        /// no such key — a leftover from an older build.
        pub default: Option<String>,
        /// A key that governs money, carried even when it equals the default.
        pub money_key: bool,
        /// `false` for a money key that happens to equal its default; such a
        /// row is present for visibility, not because anything diverges.
        pub diverges: bool,
    }

    /// Proof that a [`Settings`] was minted by the single resolution point.
    ///
    /// The fields are private to `load_seal` and the only constructor is
    /// `record`, which is private to `load_seal`. No code outside this module
    /// can produce one, which is what makes a second load path a **compile
    /// error** rather than a code-review opinion.
    #[derive(Debug, Clone)]
    pub struct ConfigProvenance {
        source: ConfigSource,
        path: Option<PathBuf>,
    }

    impl ConfigProvenance {
        fn record(source: ConfigSource, path: Option<PathBuf>) -> Self {
            let me = Self { source, path };
            let described = me.describe();
            say_once(format!("provenance:{described}"), || {
                tracing::info!(
                    target: "neoethos_core::config",
                    source = source.as_str(),
                    config = %described,
                    "config resolved — this is the file this process reads"
                );
            });
            me
        }

        pub fn source(&self) -> ConfigSource {
            self.source
        }

        pub fn path(&self) -> Option<&Path> {
            self.path.as_deref()
        }

        fn path_display(&self) -> String {
            match &self.path {
                Some(p) => p.display().to_string(),
                None => "<none>".to_string(),
            }
        }

        /// One-line, log-safe description for a startup banner.
        pub fn describe(&self) -> String {
            format!("{} ({})", self.source.as_str(), self.path_display())
        }
    }

    impl Default for ConfigProvenance {
        fn default() -> Self {
            Self {
                source: ConfigSource::CompiledDefaults,
                path: None,
            }
        }
    }

    /// Main settings structure.
    ///
    /// Construction is sealed — see the [module docs](self). Obtain one with
    /// [`Settings::load`], [`Settings::from_yaml`] or [`Settings::default`].
    #[derive(Debug, Clone, Serialize)]
    pub struct Settings {
        pub system: SystemConfig,
        pub risk: RiskConfig,
        pub models: ModelsConfig,
        pub news: NewsConfig,
        /// App / server / trading-runtime knobs (config-driven replacement for
        /// the `neoethos-app` env_overrides registry). See [`AppRuntimeConfig`].
        pub app_runtime: AppRuntimeConfig,
        /// PRIVATE — the seal. Never serialized: it records where the value
        /// came from, which is not a value the operator chose.
        #[serde(skip)]
        provenance: ConfigProvenance,
    }

    /// The wire shape. Private, so nothing outside can deserialize around the
    /// seal. `deny_unknown_fields` here is what catches a stale TOP-LEVEL key
    /// such as the `secrets_file:` in the operator's live store.
    ///
    /// **These `deserialize_with` attributes are the sub-struct seal's ONE call
    /// site.** Four of the five sections have no `impl Deserialize` at all (see
    /// the SUB-STRUCT SEAL block above `pub use load_seal`); their parser is an
    /// inherent fn that only this struct names. Deleting an attribute here does
    /// not fall back to a derive — it stops compiling, which is the point.
    /// `models` is the exception and is ledgered there.
    #[derive(Deserialize)]
    #[serde(default, deny_unknown_fields)]
    struct SettingsWire {
        #[serde(deserialize_with = "SystemConfig::deserialize")]
        system: SystemConfig,
        #[serde(deserialize_with = "RiskConfig::deserialize")]
        risk: RiskConfig,
        models: ModelsConfig,
        #[serde(deserialize_with = "NewsConfig::deserialize")]
        news: NewsConfig,
        #[serde(deserialize_with = "AppRuntimeConfig::deserialize")]
        app_runtime: AppRuntimeConfig,
    }

    impl Default for SettingsWire {
        fn default() -> Self {
            Self {
                system: SystemConfig::default(),
                risk: RiskConfig::default(),
                models: ModelsConfig::default(),
                news: NewsConfig::default(),
                app_runtime: AppRuntimeConfig::default(),
            }
        }
    }

    impl Default for Settings {
        fn default() -> Self {
            Self::mint(
                SettingsWire::default(),
                ConfigProvenance::record(ConfigSource::CompiledDefaults, None),
            )
        }
    }

    // ─── retired keys ───────────────────────────────────────────────────────

    /// What happened to a key that a real config file still carries.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum RetiredKind {
        /// The field was deleted from `Settings`. Nothing reads it.
        Deleted,
        /// The field still exists but is now DERIVED from detected hardware,
        /// so a value in the file is a detector OUTPUT that `Settings::save`
        /// pickled back in as an INPUT.
        Derived,
    }

    struct RetiredKey {
        /// Dotted path, exactly as it appears in the YAML.
        path: &'static str,
        kind: RetiredKind,
        /// What to tell the operator.
        note: &'static str,
    }

    /// Keys a real config file may still carry that this build has no field
    /// for.
    ///
    /// **Every entry was verified present in a real file.** The operator's
    /// live store (`%LOCALAPPDATA%\neoethos\config.yaml`, 2026-07-31, 509
    /// lines) carries 51 of them. Without this table `deny_unknown_fields`
    /// would refuse to load his config and the app would not open. With it,
    /// each one is named at WARN and ignored — exactly the behaviour he has
    /// today; the difference is that it is no longer silent.
    ///
    /// A key that is NOT here and NOT a field is a hard load failure. That is
    /// the `trailing_enabeld:` case, which used to save and report success.
    const RETIRED_KEYS: &[RetiredKey] = &[
        // ── top level ──
        RetiredKey {
            path: "secrets_file",
            kind: RetiredKind::Deleted,
            note: "broker credentials live in the broker config / OS keyring; this key has zero \
                   readers anywhere in the workspace",
        },
        // ── system.* — the dead discovery_* trio removed 2026-06-05 ──
        RetiredKey {
            path: "system.discovery_auto_cap",
            kind: RetiredKind::Deleted,
            note: "the real discovery row cap is models.prop_search_max_rows",
        },
        RetiredKey {
            path: "system.discovery_max_rows",
            kind: RetiredKind::Deleted,
            note: "the real discovery row cap is models.prop_search_max_rows",
        },
        RetiredKey {
            path: "system.discovery_stream",
            kind: RetiredKind::Deleted,
            note: "never read anywhere in the workspace",
        },
        // ── system.* — pre-2026 keys with no field today ──
        RetiredKey {
            path: "system.ui_locale",
            kind: RetiredKind::Deleted,
            note: "no field; the desktop shell picks the locale",
        },
        RetiredKey {
            path: "system.indices_path",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "system.use_online_indices",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "system.use_volume_features",
            kind: RetiredKind::Deleted,
            note: "no field; volume features are chosen by the feature builder",
        },
        RetiredKey {
            path: "system.required_timeframes",
            kind: RetiredKind::Deleted,
            note: "superseded by base_timeframe + higher_timeframes + multi_resolution_timeframes",
        },
        RetiredKey {
            path: "system.enable_level2",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "system.level2_depth_levels",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "system.broker_timezone",
            kind: RetiredKind::Deleted,
            note: "no field; bar timestamps are normalised to UTC on import",
        },
        RetiredKey {
            path: "system.evo_multiproc_per_gpu",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "system.cache_training_frames",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "system.training_cache_max_bytes",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "system.downcast_training_float32",
            kind: RetiredKind::Deleted,
            note: "no field; precision is system.hardware.training_precision",
        },
        RetiredKey {
            path: "system.vortex_memory_map",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "system.smc_freshness_limit",
            kind: RetiredKind::Deleted,
            note: "no field; SMC knobs live under models.smc_search_runtime / models.eval_runtime",
        },
        RetiredKey {
            path: "system.smc_atr_displacement",
            kind: RetiredKind::Deleted,
            note: "no field; see models.smc_search_runtime",
        },
        RetiredKey {
            path: "system.smc_max_levels",
            kind: RetiredKind::Deleted,
            note: "no field; see models.smc_search_runtime",
        },
        RetiredKey {
            path: "system.smc_use_cuda",
            kind: RetiredKind::Deleted,
            note: "no field; device selection is models.prop_search_device",
        },
        // ── system.* — DERIVED, no longer an input (§4a of the knob pass) ──
        RetiredKey {
            path: "system.n_jobs",
            kind: RetiredKind::Derived,
            note: "hardware-derived. A value here is a detector output that `Settings::save` \
                   pickled back in as an input. The runtime now uses effective \
                   `available_parallelism()` minus the fixed two-thread reserve, then applies \
                   only typed narrowing caps",
        },
        RetiredKey {
            path: "system.num_gpus",
            kind: RetiredKind::Derived,
            note: "hardware-derived. `num_gpus: 0` on a box with a 3090 is the same frozen \
                   detector output",
        },
        // ── risk.* ──
        RetiredKey {
            path: "risk.prop_firm_rules",
            kind: RetiredKind::Deleted,
            note: "deleted 2026-08-10: it was `risk.preset != none` written twice, read only by \
                   the Risk card's DTO, and consulted by NO engine — every discovery call passes \
                   a hardcoded PropFirmRiskRules::default(). Which rule set actually runs is \
                   `system.trading_mode` (risky/growth = risky ladder, anything else = \
                   prop-firm); set that. The value in this file changed nothing and is ignored",
        },
        // ── risk.trailing_* — the four shadows of models.exit_policy (#206) ──
        // Deleted 2026-08-10, AFTER live execution was given the real recipient.
        // The operator's live store sets `trailing_enabled: true` plus a
        // hand-tuned `trailing_atr_multiplier: 0.4` / `trailing_be_trigger_r:
        // 0.1`, so every one of these is present in a real file and each has to
        // name its replacement rather than just disappear.
        RetiredKey {
            path: "risk.trailing_enabled",
            kind: RetiredKind::Deleted,
            note: "deleted 2026-08-10: a shadowed duplicate that reached no evaluator, CPU or \
                   CUDA. The trail is `models.exit_policy.trailing_enabled` — read by the search \
                   (strategy_gene.rs) and, since today, by live execution \
                   (live_trading.rs). Note the DEFAULT DIFFERS: this key shipped `true`, \
                   models.exit_policy ships `false`, because a trail armed at +1R capped the \
                   measured payoff at 1.08 against a configured floor of 2.0. If you want the \
                   trail, set models.exit_policy.trailing_enabled: true deliberately",
        },
        RetiredKey {
            path: "risk.trailing_atr_multiplier",
            kind: RetiredKind::Deleted,
            note: "deleted 2026-08-10: replaced by models.exit_policy.trailing_stop_multiplier. \
                   RENAMED ON PURPOSE — despite the old name it was never an ATR multiple, it is \
                   a multiple of the position's own initial stop distance. Copy your value across \
                   under the new name",
        },
        RetiredKey {
            path: "risk.trailing_be_trigger_r",
            kind: RetiredKind::Deleted,
            note: "deleted 2026-08-10: replaced by models.exit_policy.trailing_be_trigger_r, same \
                   name and same meaning (profit in multiples of the initial stop before the \
                   trail arms)",
        },
        RetiredKey {
            path: "risk.trailing_min_lock_pips",
            kind: RetiredKind::Deleted,
            note: "deleted 2026-08-10: replaced by models.exit_policy.trailing_min_lock_pips, \
                   same name and same meaning (absolute pip floor on the profit the armed trail \
                   locks)",
        },
        RetiredKey {
            path: "risk.meta_label_tp_pips",
            kind: RetiredKind::Deleted,
            note: "superseded by risk.meta_label_fixed_tp (price units, not pips)",
        },
        RetiredKey {
            path: "risk.meta_label_sl_pips",
            kind: RetiredKind::Deleted,
            note: "superseded by risk.meta_label_fixed_sl (price units, not pips)",
        },
        RetiredKey {
            path: "risk.vol_ensemble_weights_trend",
            kind: RetiredKind::Deleted,
            note: "no field; the volatility ensemble is not operator-weighted",
        },
        RetiredKey {
            path: "risk.vol_ensemble_weights_range",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "risk.vol_ensemble_weights_neutral",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        // ── models.* ──
        RetiredKey {
            path: "models.export_onnx",
            kind: RetiredKind::Deleted,
            note: "deleted in the 2026-08-09 D3 purge — it reached no exporter",
        },
        RetiredKey {
            path: "models.inference_batch_size",
            kind: RetiredKind::Derived,
            note: "hardware-derived. HardwareExecutionPlan computes the inference batch from the \
                   probe and hands it to the consumer as a parameter",
        },
        // ── news.* — the 2026-08-09 D3 purge and its predecessors ──
        RetiredKey {
            path: "news.news_kill_window_min",
            kind: RetiredKind::Deleted,
            note: "the blackout window is fixed by the live gate (15 min before / 10 min after); \
                   this knob reached a NewsFilter nothing constructed",
        },
        RetiredKey {
            path: "news.news_lookahead_minutes",
            kind: RetiredKind::Deleted,
            note: "same NewsFilter nothing constructed",
        },
        RetiredKey {
            path: "news.perplexity_enabled",
            kind: RetiredKind::Deleted,
            note: "deleted in the 2026-08-09 D3 purge",
        },
        RetiredKey {
            path: "news.perplexity_api_key_env",
            kind: RetiredKind::Deleted,
            note: "deleted with the Perplexity path",
        },
        RetiredKey {
            path: "news.perplexity_model",
            kind: RetiredKind::Deleted,
            note: "deleted with the Perplexity path",
        },
        RetiredKey {
            path: "news.perplexity_num_results",
            kind: RetiredKind::Deleted,
            note: "deleted with the Perplexity path",
        },
        RetiredKey {
            path: "news.perplexity_timeframe_hours",
            kind: RetiredKind::Deleted,
            note: "deleted with the Perplexity path",
        },
        RetiredKey {
            path: "news.news_decay_minutes",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.news_confidence_threshold",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.news_trade_on_event",
            kind: RetiredKind::Deleted,
            note: "superseded by news.news_trading_mode",
        },
        RetiredKey {
            path: "news.news_trade_confidence_threshold",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.news_event_risk_pct",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.enable_news",
            kind: RetiredKind::Deleted,
            note: "superseded by news.news_calendar_enabled + news.news_trading_mode",
        },
        RetiredKey {
            path: "news.news_sources",
            kind: RetiredKind::Deleted,
            note: "superseded by news.rss_feeds + news.news_calendar_source",
        },
        RetiredKey {
            path: "news.enable_llm_helper",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.llm_helper_enabled",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.llm_sentiment_positive_threshold",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.llm_sentiment_negative_threshold",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.news_backfill_enabled",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.news_backfill_days",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.news_local_glob",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.strategist_enabled",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.strategist_interval_minutes",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.auto_rescore_enabled",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.auto_rescore_days",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.auto_rescore_max_events",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
        RetiredKey {
            path: "news.auto_rescore_only_missing",
            kind: RetiredKind::Deleted,
            note: "no field",
        },
    ];

    /// Remove every retired key present in `raw`, naming each one.
    fn prune_retired_keys(raw: &mut serde_yaml_ng::Value, origin: &str) {
        for entry in RETIRED_KEYS {
            if remove_dotted(raw, entry.path).is_none() {
                continue;
            }
            let path = entry.path;
            let note = entry.note;
            let kind = entry.kind;
            let origin = origin.to_string();
            say_once(format!("retired:{origin}:{path}"), move || match kind {
                RetiredKind::Deleted => tracing::warn!(
                    target: "neoethos_core::config",
                    key = path,
                    config = %origin,
                    "config key is RETIRED and is being IGNORED — {note}. Delete the line: it has \
                     no effect and never will."
                ),
                // 2026-08-10 (pending-A A4): the derived text comes from
                // `crate::system::RETIRED_DERIVED_KEYS` — the ONE table — not
                // from a second copy maintained here. The two agreed by luck
                // before; `retired_derived_tables_are_one_table` below makes it
                // a mechanism. `note` is the fallback only if the lookup misses,
                // and that test makes a miss impossible.
                RetiredKind::Derived => {
                    let detail = match crate::system::retired_derived_key(path) {
                        Some(entry) => match entry.set_instead {
                            Some(knob) => format!(
                                "it is computed from {}. If you meant to constrain it, set `{knob}`",
                                entry.derived_from
                            ),
                            None => format!("it is computed from {}", entry.derived_from),
                        },
                        None => note.to_string(),
                    };
                    tracing::warn!(
                        target: "neoethos_core::config",
                        key = path,
                        config = %origin,
                        "config key is DERIVED FROM HARDWARE and is being IGNORED — {detail}. \
                         Delete the line: the runtime detects it."
                    );
                }
            });
        }
    }

    fn remove_dotted(raw: &mut serde_yaml_ng::Value, dotted: &str) -> Option<serde_yaml_ng::Value> {
        let mut parts = dotted.split('.').peekable();
        let mut node = raw;
        loop {
            let key = parts.next()?;
            let map = node.as_mapping_mut()?;
            if parts.peek().is_none() {
                return map.remove(key);
            }
            node = map.get_mut(key)?;
        }
    }

    fn get_dotted<'a>(
        raw: &'a serde_yaml_ng::Value,
        dotted: &str,
    ) -> Option<&'a serde_yaml_ng::Value> {
        let mut node = raw;
        for key in dotted.split('.') {
            node = node.as_mapping()?.get(key)?;
        }
        Some(node)
    }

    // ─── the preset ordering fix ────────────────────────────────────────────

    /// The six `risk.*` fields `RiskConfig::default()` seeds from the active
    /// preset, and whether a LOWER number is the SAFER number.
    ///
    /// `#[serde(default)]` on `RiskConfig` builds `Default` FIRST — with
    /// `PropFirmPreset::default()` (= `Ftmo`, `domain/prop_firm.rs:38`) — and
    /// only then applies the YAML keys. So `preset: the5ers` arrived AFTER the
    /// six fields it is documented to seed, and nothing re-derived them: you
    /// got The5%ers' name with FTMO's drawdown, lot and target numbers.
    const PRESET_SEEDED_FIELDS: &[(&str, bool)] = &[
        ("risk.monthly_profit_target_pct", false), // a target, not a limit
        ("risk.daily_drawdown_limit", true),
        ("risk.total_drawdown_limit", true),
        ("risk.max_lot_size", true),
        ("risk.max_trades_per_day", true),
        // Added 2026-08-10 with the mode-aware seed. Without this row the field
        // never counts as operator-set, and the seed would OVERWRITE a number
        // he typed — a preset silently becoming a lock, on a money cap.
        ("risk.max_portfolio_risk", true),
        // `risk.prop_firm_rules` removed 2026-08-10 with the field itself (D6).
    ];

    /// What the ACTIVE preset says the six fields should hold.
    struct PresetSeeds {
        monthly_profit_target_pct: f64,
        daily_drawdown_limit: f64,
        total_drawdown_limit: f64,
        max_lot_size: f64,
        max_trades_per_day: usize,
        max_portfolio_risk: f64,
    }

    /// The concurrent-risk ceiling for the RISKY ladder, where the point is to
    /// multiply a small balance and `risky_max_risk_per_trade` is already 0.30.
    /// It is deliberately not derived from a daily-drawdown stop: risky mode has
    /// no challenge to fail, so the binding constraint is the operator's
    /// tolerance for ruin, not a firm's rulebook.
    const RISKY_PORTFOLIO_RISK_CAP: f64 = 0.34;

    impl PresetSeeds {
        /// `risky_ladder` comes from `system.trading_mode`, NOT from the preset.
        /// The two are different questions — the preset names whose rulebook
        /// applies, the mode names which ladder runs — and `max_portfolio_risk`
        /// is the one seed that needs both.
        fn for_preset(preset: PropFirmPreset, risky_ladder: bool) -> Self {
            let constraints = PropFirmConstraints::for_preset(preset);
            let runtime = PropFirmRuntimeDefaults::for_preset(preset);
            Self {
                monthly_profit_target_pct: constraints.monthly_profit_target(),
                daily_drawdown_limit: runtime.daily_dd_stop_trading_pct,
                total_drawdown_limit: constraints.buffered_total_drawdown_limit(),
                max_lot_size: runtime.max_lot_size,
                max_trades_per_day: runtime.max_trades_per_day,
                // Under a prop firm the ceiling is arithmetic, not taste. FX
                // pairs correlate, so the honest worst case is every open
                // position stopping out together — and then the day's loss IS
                // the total open risk. A concurrent-risk budget above the daily
                // stop can therefore breach the daily limit in a single move,
                // which is the way a challenge is failed outright rather than
                // slowly. Seeding it AT the daily stop makes the two limits say
                // the same thing instead of contradicting each other.
                max_portfolio_risk: if risky_ladder {
                    RISKY_PORTFOLIO_RISK_CAP
                } else {
                    runtime.daily_dd_stop_trading_pct
                },
            }
        }
    }

    /// f32-widening tolerance. The preset numbers are `f32` constants widened
    /// to `f64`, and `Settings::save` writes the widened form
    /// (`0.03999999910593033`). Comparing those against `0.04` as exact `f64`
    /// would report a divergence that does not exist — the same f32 fingerprint
    /// that identified the writer behind the two drawdown divergences.
    fn same_number(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
    }

    /// One preset-seeded field. Returns `Some(new_value)` only when the field
    /// must be RE-SEEDED because the file never set it; otherwise `None` and
    /// the operator's value stands.
    ///
    /// Never silently corrects a value the operator typed. When his number is
    /// the LOOSER of the two, both readings are named at ERROR and his number
    /// is still what runs — a preset is documented as a seed, not a lock, and
    /// quietly clamping his file would be exactly the hidden fallback this
    /// pass exists to remove.
    fn reconcile_one(
        path: &'static str,
        current: f64,
        seed: f64,
        lower_is_safer: bool,
        explicit: bool,
        preset: &'static str,
    ) -> Option<f64> {
        if same_number(current, seed) {
            return None;
        }
        if !explicit {
            say_once(format!("preset-seed:{preset}:{path}"), move || {
                tracing::warn!(
                    target: "neoethos_core::config",
                    key = path,
                    preset,
                    old = current,
                    new = seed,
                    "PRESET RE-DERIVED: the file selects this preset but does not set this \
                     field, and `#[serde(default)]` had already filled it from the DEFAULT \
                     preset. Re-seeded from the SELECTED preset."
                );
            });
            return Some(seed);
        }
        let looser = lower_is_safer && current > seed;
        say_once(format!("preset-conflict:{preset}:{path}"), move || {
            if looser {
                tracing::error!(
                    target: "neoethos_core::config",
                    key = path,
                    preset,
                    your_value = current,
                    preset_value = seed,
                    "TWO READINGS OF ONE LIMIT: your config sets a value LOOSER than the \
                     selected preset's. Your value is used — a preset is a seed, not a lock — \
                     but the firm's number is the tighter one and it is the one that fails a \
                     challenge."
                );
            } else {
                tracing::info!(
                    target: "neoethos_core::config",
                    key = path,
                    preset,
                    your_value = current,
                    preset_value = seed,
                    "config overrides the preset seed (tighter, or not a limit); your value is \
                     used"
                );
            }
        });
        None
    }

    impl Settings {
        fn mint(wire: SettingsWire, provenance: ConfigProvenance) -> Self {
            Self {
                system: wire.system,
                risk: wire.risk,
                models: wire.models,
                news: wire.news,
                app_runtime: wire.app_runtime,
                provenance,
            }
        }

        /// Where this `Settings` came from. Print it in a startup banner: it is
        /// the answer to "which of the four config files did this process
        /// actually read?", which nothing in the workspace could answer before
        /// 2026-08-10.
        pub fn provenance(&self) -> &ConfigProvenance {
            &self.provenance
        }

        /// Load settings from a YAML config file.
        ///
        /// Goes through the same seal as every other path: retired keys are
        /// named and pruned, an unrecognised key is REFUSED, the prop-firm
        /// preset is re-derived, and every money-path reading is logged.
        pub fn from_yaml(path: impl AsRef<Path>) -> anyhow::Result<Self> {
            Self::from_path_tagged(path.as_ref(), ConfigSource::ExplicitPath)
        }

        fn from_path_tagged(path: &Path, source: ConfigSource) -> anyhow::Result<Self> {
            let content = std::fs::read_to_string(path).map_err(|err| {
                anyhow::anyhow!("cannot read config file {}: {err}", path.display())
            })?;
            let raw: serde_yaml_ng::Value = serde_yaml_ng::from_str(&content).map_err(|err| {
                anyhow::anyhow!("config file {} is not valid YAML: {err}", path.display())
            })?;
            Self::from_raw(
                raw,
                ConfigProvenance::record(source, Some(path.to_path_buf())),
            )
        }

        /// THE resolution order, and the only one. **Two branches, then the
        /// compiled defaults.**
        ///
        ///   1. `$CONFIG_FILE` — the one supported way to point a run at a
        ///      different file. Explicit, and named in the log.
        ///   2. `user_config_path()` — the operator's store — **if it exists**.
        ///      On his machine it does, so this is the branch a real run takes.
        ///      Tagged [`ConfigSource::UserStoreRedirected`] when
        ///      `NEOETHOS_USER_DATA_DIR` moved it.
        ///   3. Neither exists → **the compiled `Default` impls, and nothing
        ///      else.** Under the 2026-08-10 scheme the Rust `Default`s ARE the
        ///      defaults and a config file carries OVERRIDES ONLY, so "no file"
        ///      means "no overrides" — a legitimate state, not an error.
        ///
        /// # What this REFUSES that it used to do
        ///
        /// There was a third branch: the bare relative path `"config.yaml"`,
        /// resolved against the process working directory. It is **deleted**.
        /// It was the fourth config surface and it made the same binary trade
        /// differently depending on the directory it was started from — a run
        /// from the repo root silently picked up the repo's developer profile
        /// (`require_walkforward_for_export: false`,
        /// `prop_firm_min_pass_rate: 0.0`, `max_portfolio_risk: 0.34`), and the
        /// identical binary one directory up did not.
        ///
        /// A developer who WANTS the repo profile now says so:
        /// `$env:CONFIG_FILE = 'config.yaml'`. See
        /// `docs/config-single-source-of-truth.md`.
        ///
        /// Each branch logs, by name, what it opened — including branch 3,
        /// which logs that it opened nothing.
        pub fn load() -> anyhow::Result<Self> {
            if let Ok(explicit) = std::env::var("CONFIG_FILE") {
                let path = PathBuf::from(explicit);
                return Self::from_path_tagged(&path, ConfigSource::EnvConfigFile);
            }
            let user = user_config_path();
            if user.exists() {
                let source = if crate::env_overrides::user_data_dir_override().is_some() {
                    ConfigSource::UserStoreRedirected
                } else {
                    ConfigSource::UserStore
                };
                return Self::from_path_tagged(&user, source);
            }
            // No file anywhere. Say so at WARN, name the path that was absent,
            // name the escape hatch, and name the fallback that is NO LONGER
            // taken — silence here is how a run on a rented box could pick up a
            // profile nobody chose.
            let cwd_had_one = std::path::Path::new("config.yaml").exists();
            say_once("compiled-defaults-config".to_string(), move || {
                tracing::warn!(
                    target: "neoethos_core::config",
                    user_store = %user_config_path().display(),
                    cwd = ?std::env::current_dir().ok(),
                    cwd_config_yaml_present = cwd_had_one,
                    "NO CONFIG FILE WAS READ. $CONFIG_FILE is unset and the operator's store does \
                     not exist, so this run uses the COMPILED DEFAULTS and nothing else. Until \
                     2026-08-10 this branch silently read the relative path \"config.yaml\" — that \
                     fallback is DELETED, so if a ./config.yaml is present it is being IGNORED \
                     (cwd_config_yaml_present says whether one is). To use it, set \
                     CONFIG_FILE=config.yaml explicitly."
                );
            });
            Ok(Self::default())
        }

        fn from_raw(
            mut raw: serde_yaml_ng::Value,
            provenance: ConfigProvenance,
        ) -> anyhow::Result<Self> {
            let origin = provenance.describe();
            // An empty file is a valid "all defaults" document.
            if raw.is_null() {
                raw = serde_yaml_ng::Value::Mapping(serde_yaml_ng::Mapping::new());
            }
            for key in [
                "system.hardware.cpu_budget",
                "models.backtest_runtime.rayon_threads",
            ] {
                if get_dotted(&raw, key).and_then(serde_yaml_ng::Value::as_u64) == Some(0) {
                    anyhow::bail!(
                        "config key `{key}` must be greater than zero; omit it or use null for \
                         automatic effective logical threads minus the fixed two-thread reserve"
                    );
                }
            }
            let legacy_cpu_cap_present =
                get_dotted(&raw, "models.backtest_runtime.rayon_threads").is_some();
            prune_retired_keys(&mut raw, &origin);

            // Which of the preset-seeded fields the operator typed EXPLICITLY.
            // Captured before the typed parse, because afterwards every field
            // holds a value and the distinction is gone — that erasure IS the
            // ordering bug.
            let explicit: Vec<&'static str> = PRESET_SEEDED_FIELDS
                .iter()
                .filter(|(path, _)| get_dotted(&raw, path).is_some())
                .map(|(path, _)| *path)
                .collect();
            let preset_explicit = get_dotted(&raw, "risk.preset").is_some();

            let wire = serde_yaml_ng::from_value::<SettingsWire>(raw)
                .map_err(|err| unknown_key_error(err, &origin))?;
            let mut settings = Self::mint(wire, provenance);
            if legacy_cpu_cap_present {
                let legacy_value = settings.models.backtest_runtime.rayon_threads;
                let warning_origin = origin.clone();
                say_once(
                    format!(
                        "legacy-cpu-budget:{warning_origin}:{}",
                        legacy_value
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "null".to_string())
                    ),
                    move || {
                        tracing::warn!(
                            target: "neoethos_core::config",
                            key = "models.backtest_runtime.rayon_threads",
                            replacement = "system.hardware.cpu_budget",
                            value = ?legacy_value,
                            config = %warning_origin,
                            "legacy CPU cap is accepted for one read-only compatibility window; \
                             it narrows the same process budget but will be omitted on save"
                        );
                    },
                );
            }
            settings.reconcile_preset(preset_explicit, &explicit);
            settings.validate_safety_bounds();
            settings.report_ambiguous_sentinels(&origin);
            Ok(settings)
        }

        /// Re-derive the six preset-seeded fields AFTER `risk.preset` is known.
        ///
        /// * Field ABSENT from the file → seeded from the ACTIVE preset, and
        ///   the substitution is logged with old, new and why. This is the bug
        ///   fix: before today `preset: the5ers` selected the label and left
        ///   FTMO's numbers in place.
        /// * Field PRESENT and disagreeing → the operator's typed value wins
        ///   (his file is his word; the preset is documented as a seed, not a
        ///   lock) and the disagreement is reported LOUDLY naming BOTH
        ///   readings. Where his value is the LOOSER of the two it is an
        ///   ERROR, because that is a limit sitting above the firm's. Nothing
        ///   is silently corrected in either direction.
        fn reconcile_preset(&mut self, preset_explicit: bool, explicit: &[&'static str]) {
            let preset = self.risk.preset;
            // `resolved_config` collapses "risky" and "growth" onto the same
            // ladder; this must agree with it or the cap and the ranking would
            // describe two different runs.
            let risky_ladder = matches!(self.system.trading_mode.as_str(), "risky" | "growth");
            let seeds = PresetSeeds::for_preset(preset, risky_ladder);
            let name = preset.as_str();

            let is_explicit = |path: &str| explicit.iter().any(|p| *p == path);

            // Six calls, one helper. Deliberately NOT a `macro_rules!` that
            // touches `self`: a macro body referring to `self` resolves at the
            // definition site, which is a footgun sitting on the prop-firm
            // drawdown numbers.
            let fix = |path: &'static str, current: f64, seed: f64, lower_is_safer: bool| {
                reconcile_one(path, current, seed, lower_is_safer, is_explicit(path), name)
            };

            if let Some(v) = fix(
                "risk.monthly_profit_target_pct",
                self.risk.monthly_profit_target_pct,
                seeds.monthly_profit_target_pct,
                false,
            ) {
                self.risk.monthly_profit_target_pct = v;
            }
            if let Some(v) = fix(
                "risk.daily_drawdown_limit",
                self.risk.daily_drawdown_limit,
                seeds.daily_drawdown_limit,
                true,
            ) {
                self.risk.daily_drawdown_limit = v;
            }
            if let Some(v) = fix(
                "risk.total_drawdown_limit",
                self.risk.total_drawdown_limit,
                seeds.total_drawdown_limit,
                true,
            ) {
                self.risk.total_drawdown_limit = v;
            }
            if let Some(v) = fix(
                "risk.max_lot_size",
                self.risk.max_lot_size,
                seeds.max_lot_size,
                true,
            ) {
                self.risk.max_lot_size = v;
            }
            if let Some(v) = fix(
                "risk.max_trades_per_day",
                self.risk.max_trades_per_day as f64,
                seeds.max_trades_per_day as f64,
                true,
            ) {
                self.risk.max_trades_per_day = v.max(0.0).round() as usize;
            }

            // `max_portfolio_risk: 0.0` is not a decision, it is the field's own
            // empty value. Every other seeded limit here reads "at most this
            // much"; on this one a zero was read as "no ceiling", so the
            // loosest possible setting and the unset state were spelled the
            // same way. That is the disguise, not a preference, and it is why
            // it is re-seeded rather than honoured — a money cap must not be
            // removable by leaving a field alone. An operator who genuinely
            // wants no ceiling says 1.0, which is representable, readable, and
            // cannot be arrived at by accident.
            if self.risk.max_portfolio_risk <= 0.0 {
                let seed = seeds.max_portfolio_risk;
                let was = self.risk.max_portfolio_risk;
                let explicit = is_explicit("risk.max_portfolio_risk");
                say_once(format!("portfolio-cap-sentinel:{name}"), move || {
                    tracing::warn!(
                        target: "neoethos_core::config",
                        key = "risk.max_portfolio_risk",
                        preset = name,
                        old = was,
                        new = seed,
                        set_in_file = explicit,
                        "NO PORTFOLIO CAP: a knob named max_ was {was}, which this code read as \
                         UNLIMITED concurrent risk rather than as a limit. Re-seeded from the \
                         selected preset and trading mode. To run with no ceiling, say so with \
                         1.0 — a zero cannot mean it."
                    );
                });
                self.risk.max_portfolio_risk = seed;
            } else if let Some(v) = fix(
                "risk.max_portfolio_risk",
                self.risk.max_portfolio_risk,
                seeds.max_portfolio_risk,
                true,
            ) {
                self.risk.max_portfolio_risk = v;
            }

            // The `risk.prop_firm_rules` re-derivation block was DELETED here
            // 2026-08-10 together with the field (D6). It re-derived a bool
            // that no engine read; the regime the engine actually runs comes
            // from `system.trading_mode`.

            if !preset_explicit {
                say_once("preset-implicit".to_string(), move || {
                    tracing::info!(
                        target: "neoethos_core::config",
                        preset = name,
                        "config sets no `risk.preset`; the compiled default preset is in force"
                    );
                });
            }
        }

        /// Report every knob whose name says "maximum" or "minimum" but whose
        /// `0` the code reads as "no limit at all".
        ///
        /// This never changes a value. Per the operator's standing rule a
        /// money-path reading is not corrected behind his back: `0` is
        /// reported with BOTH readings named, and he decides.
        fn report_ambiguous_sentinels(&self, origin: &str) {
            /// key, what the code does at 0, what the name/UI implies at 0.
            const AMBIGUOUS: &[(&str, &str, &str)] = &[
                (
                    "risk.max_portfolio_risk",
                    "NO CAP AT ALL on total concurrent risk across every running engine",
                    "the Advanced screen calls 0 'disabled' and says entries PAUSE at the cap; \
                     live_trading.rs:1749-1770 instead SIZES DOWN, so with no open positions the \
                     first entry is resized to the cap. At 0 neither happens",
                ),
                (
                    "models.prop_search_max_in_market",
                    "NO CAP on the fraction of time a candidate may hold a position",
                    "read as a maximum, 0 would mean 'never in the market', which would reject \
                     every candidate",
                ),
                (
                    "models.prop_search_min_payoff_ratio",
                    "NO PAYOFF FLOOR — every candidate clears this gate",
                    "read as a minimum, 0 is also the literal floor 'payoff >= 0'. The compiled \
                     default is 2.0 (the 2RR mandate); a 0 here disarms it",
                ),
                (
                    "models.prop_search_min_expectancy_t_stat",
                    "NO STATISTICAL-SIGNIFICANCE FLOOR on expectancy",
                    "read as a minimum, 0 is also the literal floor 't >= 0'",
                ),
                // Added 2026-08-10 (audit #193). The autopilot is the ONE
                // enforcement reader in the workspace
                // (`live_trading.rs:694-697`) and it spells the disabled state
                // `.filter(|v| *v > 0.0).unwrap_or(f64::INFINITY)` — so a knob
                // named `max_` reads 0 as NO LOT CEILING, on the path that
                // sends orders. Same shape as `risk.max_portfolio_risk`, which
                // shipped uncapped on every install for exactly this reason.
                (
                    "risk.max_lot_size",
                    "NO LOT CEILING on autopilot entries — the size is bounded only by the \
                     broker's own max_volume",
                    "read as a maximum, 0 lots would mean 'never trade'. Neither is a setting \
                     anyone chooses on purpose; the shipped default is 10.0 lots",
                ),
            ];

            let values: [(&str, f64, f64); 5] = [
                ("risk.max_portfolio_risk", self.risk.max_portfolio_risk, 0.0),
                (
                    "models.prop_search_max_in_market",
                    self.models.prop_search_max_in_market,
                    0.0,
                ),
                (
                    "models.prop_search_min_payoff_ratio",
                    self.models.prop_search_min_payoff_ratio,
                    2.0,
                ),
                (
                    "models.prop_search_min_expectancy_t_stat",
                    self.models.prop_search_min_expectancy_t_stat,
                    0.0,
                ),
                // The seeded value is preset-derived (`PropFirmRuntimeDefaults`),
                // so the comparison is against THIS config's preset rather than
                // a literal — no preset seeds 0, so a 0 here is always a
                // deviation and is reported at ERROR rather than WARN.
                (
                    "risk.max_lot_size",
                    self.risk.max_lot_size,
                    PropFirmRuntimeDefaults::for_preset(self.risk.preset).max_lot_size,
                ),
            ];

            for (key, value, compiled_default) in values {
                if value != 0.0 {
                    continue;
                }
                let Some((_, reading_a, reading_b)) = AMBIGUOUS.iter().find(|(k, _, _)| *k == key)
                else {
                    continue;
                };
                let disarmed = compiled_default != 0.0;
                let origin = origin.to_string();
                say_once(format!("sentinel:{origin}:{key}"), move || {
                    if disarmed {
                        tracing::error!(
                            target: "neoethos_core::config",
                            key,
                            value = 0.0,
                            compiled_default,
                            config = %origin,
                            "AMBIGUOUS SENTINEL ON A MONEY PATH, AND IT IS NOT THE DEFAULT. \
                             Reading A (what the code does): {reading_a}. Reading B: {reading_b}. \
                             The compiled default is {compiled_default} — this file turns the \
                             gate OFF. Nothing has been changed; set an explicit value."
                        );
                    } else {
                        tracing::warn!(
                            target: "neoethos_core::config",
                            key,
                            value = 0.0,
                            config = %origin,
                            "AMBIGUOUS SENTINEL: a knob named as a limit is 0. Reading A (what \
                             the code does): {reading_a}. Reading B: {reading_b}. This matches \
                             the shipped default; nothing has been changed."
                        );
                    }
                });
            }
        }
    }

    impl<'de> Deserialize<'de> for Settings {
        /// The seal. There is no derived `Deserialize` to route around: every
        /// `from_str` / `from_reader` / `from_value` in the workspace lands
        /// here and gets the retired-key prune, the unknown-key refusal, the
        /// preset re-derivation and the money-path reports.
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let raw = serde_yaml_ng::Value::deserialize(deserializer)?;
            Self::from_raw(raw, ConfigProvenance::record(ConfigSource::InMemory, None))
                .map_err(serde::de::Error::custom)
        }
    }

    /// Turn serde's `unknown field` into something an operator can act on.
    ///
    /// **What this now REFUSES that it used to accept:** a key that is neither
    /// a field nor listed in `RETIRED_KEYS`. `trailing_enabeld:` used to
    /// parse, save, and report "config.yaml saved (verbatim)" through the
    /// raw-YAML editor — the only route to 364 of the 390 knobs.
    fn unknown_key_error(err: serde_yaml_ng::Error, origin: &str) -> anyhow::Error {
        let text = err.to_string();
        if !text.contains("unknown field") {
            return anyhow::anyhow!("config {origin} does not match the settings schema: {text}");
        }
        anyhow::anyhow!(
            "config {origin} contains a key this build does not recognise.\n\n  {text}\n\n\
             WHAT TO DO\n\
             \x20 1. Check the spelling against the expected-field list above. A misspelled key \
             (the classic is `trailing_enabeld` for `trailing_enabled`) was silently accepted \
             before 2026-08-10 and reported as saved.\n\
             \x20 2. If the key was retired by an earlier release, it belongs in RETIRED_KEYS in \
             crates/neoethos-core/src/config.rs — add it there with a note and it becomes a \
             warning instead of a failure.\n\
             \x20 3. Your file has NOT been modified by this error. Back it up before editing; \
             `neoethos-cli config normalize --write` takes the backup and shows every \n             override beside the default it shadows."
        )
    }

    #[cfg(test)]
    mod seal_tests {
        use super::*;

        /// pending-A A4. Two tables listed the same three hardware-derived
        /// keys and agreed **by luck**: `load_seal::RETIRED_KEYS` (this file)
        /// and `crate::system::RETIRED_DERIVED_KEYS`. The message is now taken
        /// from `system`, and this test is what stops the two lists from
        /// separating — a key derived in one and settable in the other is a
        /// hardware value an operator can freeze into a file and carry to
        /// another machine, which is the defect being closed.
        #[test]
        fn retired_derived_tables_are_one_table() {
            let here: Vec<&str> = RETIRED_KEYS
                .iter()
                .filter(|k| k.kind == RetiredKind::Derived)
                .map(|k| k.path)
                .collect();

            for path in &here {
                assert!(
                    crate::system::retired_derived_key(path).is_some(),
                    "`{path}` is marked Derived in config.rs but is absent from \
                     crate::system::RETIRED_DERIVED_KEYS, so the loader would fall back to a \
                     second, hand-maintained sentence. Add it to the system table."
                );
            }

            for entry in crate::system::RETIRED_DERIVED_KEYS {
                assert!(
                    here.contains(&entry.key),
                    "`{}` is in crate::system::RETIRED_DERIVED_KEYS but NOT in \
                     load_seal::RETIRED_KEYS as Derived — so a config file that still carries it \
                     is REFUSED as an unknown key (or worse, still accepted as an input) instead \
                     of being named and ignored.",
                    entry.key
                );
            }
        }

        /// The cwd-relative surface is deleted, and the enum is the record of
        /// it. If a variant for it reappears, the fourth config surface has
        /// come back with it.
        #[test]
        fn there_is_no_cwd_relative_config_source() {
            for source in [
                ConfigSource::CompiledDefaults,
                ConfigSource::EnvConfigFile,
                ConfigSource::UserStore,
                ConfigSource::UserStoreRedirected,
                ConfigSource::ExplicitPath,
                ConfigSource::InMemory,
            ] {
                assert_ne!(
                    source.as_str(),
                    "cwd_relative",
                    "the working-directory config branch is deleted; a run must never again \
                     depend on the directory it was started from"
                );
            }
        }
    }
}

/// Canonical user-data path for the operator's editable `config.yaml`.
///
/// **F-311 (2026-05-29) — single source of truth**. Historically four
/// separate call sites (`neoethos-core::Settings::load`, the
/// `neoethos-app` server routes, `neoethos-cli` argument parsing, the
/// `neoethos-models::registry`) each rolled their own resolution: some
/// honoured `$CONFIG_FILE`, some used a relative literal, some
/// hard-coded the install dir. That shadow made F-310 (supervisor
/// seeding the wrong file) extremely hard to diagnose because two
/// readers saw two different states for the same logical config.
///
/// Going forward every read should resolve via this helper:
///
/// * **Windows**: `%LOCALAPPDATA%\neoethos\config.yaml`
///   (`C:\Users\<u>\AppData\Local\neoethos\config.yaml`).
/// * **Linux**: `$XDG_DATA_HOME/neoethos/config.yaml` or
///   `~/.local/share/neoethos/config.yaml`.
/// * **macOS**: `~/Library/Application Support/neoethos/config.yaml`.
///
/// On startup the F-310 supervisor seeds this path from the bundle's
/// read-only config when the user file is missing; subsequent edits
/// (Settings → App tab, F-312 raw YAML editor, `/settings` POST) write
/// back to the same path. Tests that need a synthetic path can still
/// supply one via the `CONFIG_FILE` env var — `Settings::load` checks
/// that first and LOGS which branch it took. Since 2026-08-10 there are only
/// two file branches (`$CONFIG_FILE`, this store) and then the compiled
/// defaults; the bare relative `"config.yaml"` branch is deleted.
pub fn user_config_path() -> PathBuf {
    // Explicit override (NEOETHOS_USER_DATA_DIR) wins on every platform, so the
    // desktop shell / power users can point ALL config + data readers at one
    // chosen root (e.g. a project dir) — keeping every resolver consistent.
    //
    // 2026-08-10 (pending-A A9): this is the SECOND env input that can move
    // WHICH FILE the whole process reads, and it was silent. It is retained —
    // `desktop/src-tauri/src/lib.rs:545` uses it as the dev data-root escape and
    // deleting it here would split the config path from the data path — but a
    // redirect now NAMES ITSELF, with both the platform-standard path it
    // replaced and the path it chose instead. A failure must never wear the
    // costume of a choice, and neither must a redirect.
    if let Some(dir) = crate::env_overrides::user_data_dir_override() {
        let redirected = PathBuf::from(&dir).join("config.yaml");
        let shown = redirected.display().to_string();
        let standard = platform_user_config_path().display().to_string();
        load_seal::say_once(format!("user-data-dir-override:{shown}"), || {
            tracing::warn!(
                target: "neoethos_core::config",
                env_var = crate::env_overrides::ENV_USER_DATA_DIR,
                value = %dir,
                config = %shown,
                platform_default = %standard,
                "NEOETHOS_USER_DATA_DIR IS REDIRECTING THE CONFIG FILE. This process will read \
                 the config shown, NOT the platform-standard store. Unset the variable to go back \
                 to the standard path."
            );
        });
        return redirected;
    }
    platform_user_config_path()
}

/// The platform-standard store path, with no env redirect applied.
///
/// Split out so [`user_config_path`] can name BOTH paths when
/// `NEOETHOS_USER_DATA_DIR` moves the answer — "your config is elsewhere" is
/// only actionable if the message says where it would otherwise have been.
fn platform_user_config_path() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(local).join("neoethos").join("config.yaml");
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("neoethos")
                .join("config.yaml");
        }
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
            return PathBuf::from(xdg).join("neoethos").join("config.yaml");
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("neoethos")
                .join("config.yaml");
        }
    }
    // No LOCALAPPDATA (Windows), no HOME / XDG_DATA_HOME (POSIX). Unreachable
    // on any supported OS, and it must NOT resolve to the bare relative
    // `"config.yaml"`: `Settings::load` treats this path's existence as "the
    // operator has a store", so returning the repo file here would smuggle back
    // the cwd-relative surface that branch 3 of `load()` just deleted.
    load_seal::say_once("no-home-dir".to_string(), || {
        tracing::warn!(
            target: "neoethos_core::config",
            "no LOCALAPPDATA / HOME / XDG_DATA_HOME — the config store falls back to \
             ./neoethos/config.yaml relative to the working directory. This is NOT the repo's \
             ./config.yaml: adopting that file by accident is the cwd-dependent-behaviour defect \
             closed on 2026-08-10. Set NEOETHOS_USER_DATA_DIR to choose the root explicitly."
        );
    });
    PathBuf::from("neoethos").join("config.yaml")
}

impl Settings {
    /// Sanity-check loaded RiskConfig values against prop-firm-safe bounds.
    ///
    /// We can't reject the load — config consumers expect a non-fatal load —
    /// but a mistyped `risk_per_trade: 50` (meaning 50% instead of 0.5%) needs
    /// to be screamed about, otherwise the bot silently sizes 100× too big.
    /// All checks emit `tracing::error` with the field, the loaded value,
    /// and a recommended sane value. M9 in the audit.
    fn validate_safety_bounds(&self) {
        let risk = &self.risk;
        // risk_per_trade should be a fraction (0.0 — 0.05 typical, 0.10 max).
        // A YAML value > 1.0 means the user typed a percentage (e.g. 1.5 for
        // 1.5%) — we recover by interpreting it as percent and warning.
        if risk.risk_per_trade > 1.0 {
            tracing::error!(
                target: "neoethos_core::config",
                risk_per_trade = risk.risk_per_trade,
                "RiskConfig.risk_per_trade > 1.0 — looks like a percentage typo. \
                 0.005 means 0.5%, NOT 0.5 = 50%. Halt or fix the config."
            );
        } else if risk.risk_per_trade > 0.05 {
            tracing::warn!(
                target: "neoethos_core::config",
                risk_per_trade = risk.risk_per_trade,
                "RiskConfig.risk_per_trade > 5% per trade — uncommonly aggressive for a prop firm"
            );
        }
        if risk.daily_drawdown_limit <= 0.0 || risk.daily_drawdown_limit > 0.20 {
            tracing::error!(
                target: "neoethos_core::config",
                daily_drawdown_limit = risk.daily_drawdown_limit,
                "RiskConfig.daily_drawdown_limit must be in (0, 0.20]; typical prop firms set 0.04-0.05"
            );
        }
        if risk.total_drawdown_limit <= risk.daily_drawdown_limit {
            tracing::error!(
                target: "neoethos_core::config",
                total = risk.total_drawdown_limit,
                daily = risk.daily_drawdown_limit,
                "RiskConfig.total_drawdown_limit should exceed daily_drawdown_limit"
            );
        }
        if risk.total_drawdown_limit > 0.30 {
            tracing::error!(
                target: "neoethos_core::config",
                total_drawdown_limit = risk.total_drawdown_limit,
                "RiskConfig.total_drawdown_limit > 30% — exceeds every published prop-firm rule"
            );
        }
    }

    /// Save settings to YAML file

    /// The document to persist: ONLY what differs from `Settings::default()`,
    /// plus every money key, always, whatever its value.
    ///
    /// This used to be `serde_yaml_ng::to_string(self)` — the whole struct. The
    /// scheme this wave installed is "Rust defaults are the source of defaults,
    /// the operator's file holds only his overrides", and serialising the full
    /// document defeated it on the FIRST UI CLICK: a two-line store became 482
    /// lines after one unrelated mutation, at which point every default was
    /// frozen into his file as though he had chosen it. Worse, `risk.preset`
    /// stopped moving the money numbers after any save, because the six fields
    /// `reconcile_preset` seeds were now present explicitly and looked
    /// operator-set.
    ///
    /// MONEY KEYS ARE NEVER PRUNED, even when they equal the default. Pruning
    /// them would be semantically identical *today* and would silently move a
    /// drawdown limit or a risk fraction the day a default changes underneath
    /// him. A limit he can read in his own file is worth the four extra lines.
    pub fn as_override_document(&self) -> anyhow::Result<serde_yaml_ng::Value> {
        fn prune(
            current: &serde_yaml_ng::Value,
            default: &serde_yaml_ng::Value,
            path: &str,
            keep: &[&str],
        ) -> Option<serde_yaml_ng::Value> {
            if keep.contains(&path) {
                return Some(current.clone());
            }
            match (current, default) {
                (serde_yaml_ng::Value::Mapping(cur), serde_yaml_ng::Value::Mapping(def)) => {
                    let mut out = serde_yaml_ng::Mapping::new();
                    for (k, v) in cur {
                        let name = k.as_str().unwrap_or_default();
                        let child = if path.is_empty() {
                            name.to_string()
                        } else {
                            format!("{path}.{name}")
                        };
                        // A key absent from the defaults is by definition not a
                        // default, so it survives whatever its value.
                        let d = def.get(k).cloned().unwrap_or(serde_yaml_ng::Value::Null);
                        if let Some(kept) = prune(v, &d, &child, keep) {
                            out.insert(k.clone(), kept);
                        }
                    }
                    // An empty section carries no information; drop it rather
                    // than leave `risk: {}` behind.
                    (!out.is_empty()).then(|| serde_yaml_ng::Value::Mapping(out))
                }
                _ => (current != default).then(|| current.clone()),
            }
        }

        let current = serde_yaml_ng::to_value(self)?;
        let default = serde_yaml_ng::to_value(Self::default())?;
        Ok(prune(&current, &default, "", load_seal::ALWAYS_PERSIST)
            .unwrap_or(serde_yaml_ng::Value::Mapping(serde_yaml_ng::Mapping::new())))
    }

    /// Every leaf on which this `Settings` DIVERGES from the compiled defaults,
    /// each beside the default it is shadowing.
    ///
    /// This exists because a config file written by an older build is a FULL
    /// SNAPSHOT, not a record of decisions: it repeats every default of its own
    /// era, and from that day forward it silently shadows every default the
    /// codebase improves. The operator cannot tell the two apart by reading it —
    /// a deliberate choice and a fossilised default are the same line of YAML.
    ///
    /// So the file cannot answer "what did I choose?", but it CAN answer "where
    /// do I differ, and from what?" — and that question has a short, reviewable
    /// answer where the file has five hundred lines. Every entry is then either
    /// a real decision worth keeping or a fossil worth dropping, and the
    /// operator decides per line instead of per file.
    ///
    /// Money keys are reported even when they equal the default, for the same
    /// reason [`Self::as_override_document`] never prunes them.
    pub fn overrides_against_defaults(&self) -> anyhow::Result<Vec<ConfigOverride>> {
        // One line per value, always. A block-style sequence would break the
        // table this feeds and make a 28-symbol watchlist unreadable next to
        // the default it shadows.
        fn render(v: &serde_yaml_ng::Value) -> String {
            match v {
                serde_yaml_ng::Value::String(s) => s.clone(),
                serde_yaml_ng::Value::Sequence(items) => {
                    let inner: Vec<String> = items.iter().map(render).collect();
                    format!("[{}]", inner.join(", "))
                }
                serde_yaml_ng::Value::Mapping(m) => {
                    let inner: Vec<String> = m
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k.as_str().unwrap_or_default(), render(v)))
                        .collect();
                    format!("{{{}}}", inner.join(", "))
                }
                other => serde_yaml_ng::to_string(other)
                    .unwrap_or_else(|_| "<unrenderable>".to_string())
                    .trim_end()
                    .to_string(),
            }
        }

        // `default` is `None` only when the DEFAULTS MAPPING HAS NO SUCH KEY —
        // which is not the same thing as a key whose default is null. An
        // `Option<f64>` field defaulting to `None` serialises as `null` and is
        // a perfectly live setting; conflating the two would report every
        // opt-in knob the operator turned on as an unrecognised leftover.
        fn walk(
            over: &serde_yaml_ng::Value,
            default: Option<&serde_yaml_ng::Value>,
            path: &str,
            out: &mut Vec<ConfigOverride>,
        ) {
            match over {
                serde_yaml_ng::Value::Mapping(map) => {
                    for (k, v) in map {
                        let name = k.as_str().unwrap_or_default();
                        let child = if path.is_empty() {
                            name.to_string()
                        } else {
                            format!("{path}.{name}")
                        };
                        let d = default.and_then(|d| d.as_mapping()).and_then(|m| m.get(k));
                        walk(v, d, &child, out);
                    }
                }
                leaf => {
                    let is_money = load_seal::ALWAYS_PERSIST.contains(&path);
                    let default_str = default.map(render);
                    let live_str = render(leaf);
                    // A money key that equals its default is carried by
                    // `as_override_document` on purpose; report it as such
                    // instead of listing it as a divergence it is not.
                    let same = default_str.as_deref() == Some(live_str.as_str());
                    out.push(ConfigOverride {
                        path: path.to_string(),
                        live: live_str,
                        default: default_str,
                        money_key: is_money,
                        diverges: !same,
                    });
                }
            }
        }

        let over = self.as_override_document()?;
        let default = serde_yaml_ng::to_value(Self::default())?;
        let mut out = Vec::new();
        walk(&over, Some(&default), "", &mut out);
        Ok(out)
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let yaml = serde_yaml_ng::to_string(&self.as_override_document()?)?;
        // Audit M07: atomic write (temp + fsync + rename) so a crash mid-write
        // can never leave a truncated config.yaml — a corrupt config is the
        // known "app won't open" root cause. The previous std::fs::write was
        // non-atomic.
        crate::storage::json::write_bytes_atomic(path, yaml.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The successor to `NEOETHOS_FEATURE_CUBE_MODE` must not accept `ram`.
    ///
    /// The retired variable accepted it, and its `ram` arm returned BEFORE the
    /// free-RAM check — the one input that could cause an OOM was the one input
    /// that skipped the OOM guard. A migration that copies the old value across
    /// must fail LOUDLY at load, naming the field and the two legal values,
    /// rather than parse into something that quietly means `auto`.
    #[test]
    fn feature_cube_mode_refuses_ram_and_defaults_to_auto() {
        assert_eq!(
            FeatureCubeMode::default(),
            FeatureCubeMode::Auto,
            "the default must reproduce the env-unset behaviour every run has had"
        );
        assert_eq!(
            serde_yaml_ng::from_str::<FeatureCubeMode>("disk").unwrap(),
            FeatureCubeMode::Disk
        );
        assert_eq!(
            serde_yaml_ng::from_str::<FeatureCubeMode>("auto").unwrap(),
            FeatureCubeMode::Auto
        );

        let refused = serde_yaml_ng::from_str::<FeatureCubeMode>("ram");
        let err = refused
            .expect_err("`ram` must be refused — forcing RAM is the defect that was removed")
            .to_string();
        assert!(
            err.contains("ram") && err.contains("auto") && err.contains("disk"),
            "the refusal must name the offending value AND the accepted ones, or the \
             operator has to go read the source to fix his file: {err}"
        );
    }

    /// The whole point of the field is that a run can say which assembly built
    /// its cube. A spelling that drifts from the serde representation would
    /// make the run profile disagree with the config file it came from.
    #[test]
    fn feature_cube_mode_as_str_matches_its_serde_spelling() {
        for mode in [FeatureCubeMode::Auto, FeatureCubeMode::Disk] {
            let yaml = serde_yaml_ng::to_string(&mode).unwrap();
            assert_eq!(yaml.trim(), mode.as_str());
        }
    }

    #[test]
    fn config_maps_serialize_in_sorted_deterministic_order() {
        // M06/M07 follow-up: HashMap config fields must serialize in sorted
        // key order so Settings::save doesn't reshuffle config.yaml on every
        // write. Insert keys out of order and confirm two serializations
        // match and the keys come out sorted.
        let mut s = Settings::default();
        for (k, v) in [("H4", 3usize), ("M1", 1), ("D1", 5), ("M5", 2)] {
            s.models.hpo_trials_by_model.insert(k.to_string(), v);
            s.models.prop_search_max_rows_by_tf.insert(k.to_string(), v);
        }
        s.models.model_param_overrides.insert(
            "zeta".to_string(),
            HashMap::from([("b".to_string(), "1".to_string())]),
        );
        s.models.model_param_overrides.insert(
            "alpha".to_string(),
            HashMap::from([("a".to_string(), "0".to_string())]),
        );

        let a = serde_yaml_ng::to_string(&s).unwrap();
        let b = serde_yaml_ng::to_string(&s).unwrap();
        assert_eq!(a, b, "two serializations must be byte-identical");

        // hpo_trials_by_model keys appear in sorted order.
        let positions: Vec<usize> = ["D1", "H4", "M1", "M5"]
            .iter()
            .map(|k| {
                a.find(&format!("{k}:"))
                    .unwrap_or_else(|| panic!("key {k} present"))
            })
            .collect();
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "hpo_trials_by_model keys must be sorted"
        );
        // Nested override keys sorted too.
        assert!(a.find("alpha:").unwrap() < a.find("zeta:").unwrap());
    }

    #[test]
    fn test_default_settings() {
        // F-303 (2026-05-28): updated post-F-129 — the previous
        // assertion `settings.system.symbol == "EURUSD"` was a stale
        // hardcoded-default check from before the synthetic-data
        // cleanup. `SystemConfig::default()` now returns empty for
        // both `symbol` and `account_currency` (F-304), forcing the
        // operator's `config.yaml` to populate them. The pre-flight
        // bail in `run_discovery_cycle` catches the omission with an
        // actionable error.
        let settings = Settings::default();
        assert_eq!(
            settings.system.symbol, "",
            "default symbol must be empty per F-129"
        );
        assert_eq!(
            settings.system.account_currency, "",
            "default account_currency must be empty per F-304"
        );
        assert_eq!(settings.risk.initial_balance, 10_000.0);
        assert!(!settings.models.ml_models.is_empty());
    }

    // ─── UI↔CLI parity: the shared timeframe/symbol resolvers ───────────────
    // These lock the behaviour of `SystemConfig::resolve_*`, the SINGLE source
    // of truth that BOTH `neoethos-cli` and the app server call. If this drifts,
    // the two entry points would search differently from the same config —
    // exactly the divergence the 2026-06-04 parity pass removed.

    #[test]
    fn resolve_higher_timeframes_default_config_multi_resolution() {
        // Default: multi_resolution_enabled = true and the multi-res list is the
        // FULL canonical set → "every configured TF except the effective base".
        let sys = SystemConfig::default();

        let m1 = sys.resolve_higher_timeframes("M1");
        assert_eq!(
            m1,
            vec![
                "M3", "M5", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"
            ],
            "M1 base → all canonical above M1"
        );
        assert!(!m1.iter().any(|tf| tf == "M1"), "base itself is excluded");

        // base=H1: multi-resolution keeps LOWER TFs (M1..M30) as extra context
        // too — only the base is dropped. The Flutter UI cannot replicate this,
        // which is precisely why an untouched UI sends no override and lets this
        // resolver decide (parity with the CLI).
        let h1 = sys.resolve_higher_timeframes("H1");
        assert!(
            h1.contains(&"M5".to_string()),
            "lower TFs retained under multi-res"
        );
        assert!(h1.contains(&"H4".to_string()), "higher TFs retained");
        assert!(!h1.iter().any(|tf| tf == "H1"), "base itself is excluded");
        assert_eq!(h1.len(), 10, "all 11 canonical minus the base");

        // Effective-base relativity: an overridden base trims itself out even
        // when it differs from `self.base_timeframe`.
        assert!(
            !sys.resolve_higher_timeframes("H4")
                .iter()
                .any(|tf| tf == "H4")
        );
    }

    #[test]
    fn resolve_higher_timeframes_multi_resolution_off_filters_strictly_above() {
        // multi_resolution OFF → higher_timeframes filtered to strictly-above
        // the base in canonical order (never a lower/equal TF).
        let mut sys = SystemConfig::default();
        sys.multi_resolution_enabled = false;
        assert_eq!(
            sys.resolve_higher_timeframes("H1"),
            vec!["H4", "H12", "D1", "W1", "MN1"],
            "H1 base, multi-res off → only canonical TFs strictly above H1"
        );

        // An operator exclusion in higher_timeframes is honoured, and entries
        // not strictly above the base are dropped (D1/H4 kept, M5 below M1? no —
        // M5 is above M1, so it stays; M1-equal would be dropped).
        sys.higher_timeframes = vec!["H4".to_string(), "D1".to_string(), "M5".to_string()];
        assert_eq!(
            sys.resolve_higher_timeframes("H1"),
            vec!["H4", "D1"],
            "restricted higher_timeframes respected; M5 (below H1) excluded"
        );
    }

    #[test]
    fn resolve_base_and_symbol_trim_preserve_config_value() {
        let mut sys = SystemConfig::default();
        sys.base_timeframe = "  H4 ".to_string();
        sys.symbol = " EURUSD ".to_string();
        assert_eq!(sys.resolve_base_timeframe(), "H4");
        assert_eq!(sys.resolve_symbol(), "EURUSD");
    }

    #[test]
    fn test_serialize_deserialize() {
        let settings = Settings::default();
        let yaml = serde_yaml_ng::to_string(&settings).unwrap();
        let deserialized: Settings = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(deserialized.system.symbol, settings.system.symbol);
    }
}

/// Profit the trail locks once it engages, in pips.
///
/// Shared so the backtest, the GPU kernel and live trading cannot drift apart:
/// a live stop that protects a different amount than the strategy was scored on
/// is the parity break that makes a backtest optimistic.
pub const DEFAULT_TRAILING_MIN_LOCK_PIPS: f64 = 2.0;
