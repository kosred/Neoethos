use super::strategy_gene::Gene;
use neoethos_data::{FeatureFrame, Ohlcv};
use rand::Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct SmcSearchConfig {
    pub force_ratio: f64,
    pub min_flags: usize,
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

impl Default for SmcSearchConfig {
    fn default() -> Self {
        let default_p = 0.50;
        Self {
            // F-276 (2026-05-28): lowered from 0.65 → 0.30. The original
            // 0.65 forced 65% of every generation to carry at least one
            // SMC flag — disproportionately restrictive when the GA is
            // already evolving threshold + indicator weights. On
            // empty-portfolio diagnostic runs the SMC-forced subset
            // produced 4-candidate funnels (one of the AUDUSD M15
            // smoking-guns from the earlier audit). 0.30 keeps SMC
            // injection as a meaningful seed (~30% of every generation
            // is SMC-aware) without crowding out the non-SMC genome
            // pool that often discovers profitable counter-momentum
            // strategies on D1/H4.
            //
            // Operator can still pin the old value via
            // `models.smc_search_runtime.force_ratio: 0.65` (2026-08-10: was
            // the `NEOETHOS_BOT_PROP_SMC_FORCE_RATIO` env var, now retired).
            force_ratio: 0.30,
            min_flags: 1,
            p_ob: default_p,
            p_fvg: default_p,
            p_liq: default_p,
            p_premium: default_p,
            p_inducement: default_p,
            p_mtf: 0.85,
            p_bos: default_p,
            p_choch: default_p,
            p_eqh: default_p,
            p_eql: default_p,
            p_displacement: default_p,
        }
    }
}

static SMC_SEARCH_CONFIG_CACHE: OnceLock<SmcSearchConfig> = OnceLock::new();

// The 15 `NEOETHOS_BOT_PROP_SMC_*` readers (`smc_env_f64` / `_usize` / `_bool`
// and `read_smc_search_config_from_env`) were DELETED 2026-08-10. Every one of
// them is now a typed field on `models.smc_search_runtime`, installed through
// `install_smc_search_config_from_settings`. These probabilities decide which
// genes can EXIST, so an export that changed them and appeared in no artifact
// made two runs of the same config incomparable.

impl SmcSearchConfig {
    /// The installed SMC search config, or the typed defaults when nothing was
    /// installed (the `neoethos-models` GA and test fixtures land here).
    pub fn current() -> Self {
        *SMC_SEARCH_CONFIG_CACHE.get_or_init(SmcSearchConfig::default)
    }

    /// Config-driven constructor (was the `NEOETHOS_BOT_PROP_SMC_*` env
    /// vars). Probabilities are clamped to `[0,1]` and `force_enabled =
    /// false` zeroes `force_ratio` + `min_flags`, exactly like the env
    /// reader. A `smc_search_from_settings_default_matches_env_default`
    /// test guarantees a fresh `Settings` reproduces [`Self::default`].
    pub fn from_settings(s: &neoethos_core::Settings) -> Self {
        let c = &s.models.smc_search_runtime;
        let mut cfg = SmcSearchConfig {
            force_ratio: c.force_ratio.clamp(0.0, 1.0),
            min_flags: c.min_flags,
            p_ob: c.p_ob.clamp(0.0, 1.0),
            p_fvg: c.p_fvg.clamp(0.0, 1.0),
            p_liq: c.p_liq.clamp(0.0, 1.0),
            p_premium: c.p_premium.clamp(0.0, 1.0),
            p_inducement: c.p_inducement.clamp(0.0, 1.0),
            p_mtf: c.p_mtf.clamp(0.0, 1.0),
            p_bos: c.p_bos.clamp(0.0, 1.0),
            p_choch: c.p_choch.clamp(0.0, 1.0),
            p_eqh: c.p_eqh.clamp(0.0, 1.0),
            p_eql: c.p_eql.clamp(0.0, 1.0),
            p_displacement: c.p_displacement.clamp(0.0, 1.0),
        };
        if !c.force_enabled {
            cfg.force_ratio = 0.0;
            cfg.min_flags = 0;
        }
        cfg
    }
}

/// Config-driven install — reads the SMC search knobs from the single
/// `Settings` instead of the environment. Idempotent.
pub fn install_smc_search_config_from_settings(s: &neoethos_core::Settings) {
    let _ = SMC_SEARCH_CONFIG_CACHE.set(SmcSearchConfig::from_settings(s));
}

#[cfg(test)]
mod overrides_tests {
    use super::*;

    #[test]
    fn smc_search_from_settings_default_matches_env_default() {
        // Behavior-preservation gate (config-consolidation S2e): a fresh
        // `Settings` reproduces the engine SMC-search defaults exactly.
        let s = neoethos_core::Settings::default();
        assert_eq!(
            SmcSearchConfig::from_settings(&s),
            SmcSearchConfig::default()
        );
    }

    #[test]
    fn smc_search_config_default_matches_documented_defaults() {
        // F-276 (2026-05-28): updated for the new 0.30 force_ratio
        // (previously 0.65). See `SmcSearchConfig::default` for the
        // rationale — 0.65 was crowding out non-SMC genome paths on
        // D1/H4 discovery and contributing to the 4-candidate
        // funnel failure mode.
        let defaults = SmcSearchConfig::default();
        assert!((defaults.force_ratio - 0.30).abs() < 1e-9);
        assert_eq!(defaults.min_flags, 1);
        assert!((defaults.p_mtf - 0.85).abs() < 1e-9);
        assert!((defaults.p_ob - 0.50).abs() < 1e-9);
    }
}

pub fn randomize_smc_flags(gene: &mut Gene, cfg: &SmcSearchConfig, rng: &mut impl Rng) {
    gene.use_ob = rng.random_bool(cfg.p_ob);
    gene.use_fvg = rng.random_bool(cfg.p_fvg);
    gene.use_liq_sweep = rng.random_bool(cfg.p_liq);
    gene.use_premium_discount = rng.random_bool(cfg.p_premium);
    gene.use_inducement = rng.random_bool(cfg.p_inducement);
    gene.mtf_confirmation = rng.random_bool(cfg.p_mtf);
    gene.use_bos = rng.random_bool(cfg.p_bos);
    gene.use_choch = rng.random_bool(cfg.p_choch);
    gene.use_eqh = rng.random_bool(cfg.p_eqh);
    gene.use_eql = rng.random_bool(cfg.p_eql);
    gene.use_displacement = rng.random_bool(cfg.p_displacement);
}

pub fn smc_structural_flag_count(gene: &Gene) -> usize {
    let mut n = 0usize;
    if gene.use_ob {
        n += 1;
    }
    if gene.use_fvg {
        n += 1;
    }
    if gene.use_liq_sweep {
        n += 1;
    }
    if gene.use_premium_discount {
        n += 1;
    }
    if gene.use_inducement {
        n += 1;
    }
    if gene.use_bos {
        n += 1;
    }
    if gene.use_choch {
        n += 1;
    }
    if gene.use_eqh {
        n += 1;
    }
    if gene.use_eql {
        n += 1;
    }
    if gene.use_displacement {
        n += 1;
    }
    n
}

pub fn enforce_min_structural_smc_flags(
    gene: &mut Gene,
    cfg: &SmcSearchConfig,
    rng: &mut impl Rng,
) {
    let need = cfg.min_flags.min(10);
    if need == 0 {
        return;
    }
    while smc_structural_flag_count(gene) < need {
        match rng.random_range(0..10) {
            0 => gene.use_ob = true,
            1 => gene.use_fvg = true,
            2 => gene.use_liq_sweep = true,
            3 => gene.use_premium_discount = true,
            4 => gene.use_inducement = true,
            5 => gene.use_bos = true,
            6 => gene.use_choch = true,
            7 => gene.use_eqh = true,
            8 => gene.use_eql = true,
            _ => gene.use_displacement = true,
        }
    }
    if !gene.mtf_confirmation && rng.random_bool(cfg.p_mtf.max(0.5)) {
        gene.mtf_confirmation = true;
    }
}

pub fn enforce_population_smc_ratio(genes: &mut [Gene], cfg: &SmcSearchConfig) {
    if genes.is_empty() {
        return;
    }
    let target = ((genes.len() as f64) * cfg.force_ratio).ceil() as usize;
    if target == 0 {
        return;
    }
    let mut active = genes
        .iter()
        .filter(|g| smc_structural_flag_count(g) > 0)
        .count();
    if active >= target {
        return;
    }
    let mut rng = rand::rng();
    for gene in genes.iter_mut() {
        if active >= target {
            break;
        }
        if smc_structural_flag_count(gene) > 0 {
            continue;
        }
        enforce_min_structural_smc_flags(gene, cfg, &mut rng);
        active += 1;
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SmcColumns {
    ob: Option<usize>,
    fvg: Option<usize>,
    liq: Option<usize>,
    trend: Option<usize>,
    premium: Option<usize>,
    inducement: Option<usize>,
    bos: Option<usize>,
    choch: Option<usize>,
    eqh: Option<usize>,
    eql: Option<usize>,
    displacement: Option<usize>,
}

pub type SmcSignalTuple = (
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
);

fn normalize_feature_name(name: &str) -> String {
    name.to_ascii_lowercase().replace(['-', ' '], "_")
}

/// One SMC gate flag, the EXACT column names that may feed it, and the column
/// that exists in the vocabulary and plausibly carries the same meaning but
/// only under a semantic equation the operator has NOT approved.
///
/// **Why this table replaced the substring scan (2026-08-10).** The old
/// `find_feature_column` accepted `norm == alias || norm.contains(alias)` with
/// the COLUMN loop outermost. Two defects followed from that one line:
///
/// 1. Two- and three-character aliases (`ob`, `liq`, `bos`, `eqh`, `eql`,
///    `fvg`) matched as substrings anywhere in a column name. With the
///    vocabulary at 217 columns the collision surface was already populated —
///    `ob` matches `obv` and `moving_average_cross_probability`, `fvg` matches
///    `fvg_positioning_average` and `fvg_trailing_stop`, `liq` matches
///    `quant_amihud_illiquidity`, and `trend` matched 25 classic ids plus the
///    retired v2 Regime trend-strength label. The correct binding held only
///    because the SMC family is emitted FIRST in the frame and its own column
///    therefore came first in column order. That is positional luck, not a
///    binding rule, and the vocabulary just grew 27x.
/// 2. The alias PRIORITY ORDER was inert. `["smc_ob", "order_block", "ob"]`
///    did not prefer `smc_ob`: the first matching COLUMN won, whichever alias
///    it happened to match. Here the candidate list is the outer loop, so the
///    family's own name wins over a legacy spelling regardless of column order.
///
/// Every candidate below is matched by EXACT equality on the normalised full
/// column name. A future indicator named `choch_reversal_probability` or
/// `premium_zone_index` therefore cannot capture an SMC gate.
#[derive(Debug, Clone, Copy)]
struct SmcAliasSpec {
    /// Operator-facing flag name — this is the string that appears in the
    /// binding INFO line and in the unbound WARN line.
    flag: &'static str,
    /// Exact normalised column names, highest priority first.
    candidates: &'static [&'static str],
    /// A column the SMC family really does ship whose meaning is ARGUABLY this
    /// flag's, but only under an equation nobody has signed off. Never bound —
    /// named in the WARN line so the decision is visible and one edit away.
    pending_operator_approval: Option<&'static str>,
}

/// Index of each flag in [`SMC_ALIAS_SPECS`], the tally arrays, and
/// [`SmcColumns`]. Kept as consts so the three stay in lockstep.
const SMC_IDX_OB: usize = 0;
const SMC_IDX_FVG: usize = 1;
const SMC_IDX_LIQ: usize = 2;
const SMC_IDX_TREND: usize = 3;
const SMC_IDX_PREMIUM: usize = 4;
const SMC_IDX_INDUCEMENT: usize = 5;
const SMC_IDX_BOS: usize = 6;
const SMC_IDX_CHOCH: usize = 7;
const SMC_IDX_EQH: usize = 8;
const SMC_IDX_EQL: usize = 9;
const SMC_IDX_DISPLACEMENT: usize = 10;
const SMC_FLAG_COUNT: usize = 11;

/// The candidate names are taken from the SMC family's OWN emission list
/// (`neoethos-data/src/core/smc.rs`): `smc_ob`, `smc_fvg`, `smc_liq_sweep`,
/// `smc_displacement`, `smc_bos`, `smc_eqh`, `smc_eql`, `smc_inducement`,
/// `smc_trend_bias`, `smc_mss`, `smc_pd_array`.
///
/// The short second spellings (`smc_liq`, `smc_trend`, `smc_premium`,
/// `smc_choch`) do not exist in the shipped vocabulary. They are retained as
/// EXACT candidates because they are unambiguous — `smc_`-prefixed, so no
/// classic indicator can ever collide with them — and because the gate-array
/// fixtures in `genetic/search_engine.rs` build frames under those names. They
/// cost nothing and they keep a name that means exactly one thing meaning it.
///
/// Aliases DELETED outright, because they matched no column in the vocabulary
/// and existed purely as collision surface: `order_block`, `fair_value_gap`,
/// `liquidity_sweep`, `liq_sweep`, `market_trend`, `premium_discount`,
/// `change_of_character`, `break_of_structure`, `equal_highs`, `equal_lows`,
/// `impulse_displacement`, and the bare `ob` / `fvg` / `liq` / `trend` /
/// `bos` / `choch` / `eqh` / `eql` / `displacement` / `inducement` forms.
const SMC_ALIAS_SPECS: [SmcAliasSpec; SMC_FLAG_COUNT] = [
    SmcAliasSpec {
        flag: "ob",
        candidates: &["smc_ob"],
        pending_operator_approval: None,
    },
    SmcAliasSpec {
        flag: "fvg",
        candidates: &["smc_fvg"],
        pending_operator_approval: None,
    },
    SmcAliasSpec {
        flag: "liq",
        candidates: &["smc_liq_sweep", "smc_liq"],
        pending_operator_approval: None,
    },
    SmcAliasSpec {
        flag: "trend",
        candidates: &["smc_trend_bias", "smc_trend"],
        pending_operator_approval: None,
    },
    SmcAliasSpec {
        flag: "premium",
        // No `smc_premium` column exists. `smc_pd_array` is the premium/
        // discount array and is the only column carrying that meaning, but
        // binding it would change what every `use_premium_discount` gene votes
        // on, so it waits for an explicit yes.
        candidates: &["smc_premium"],
        pending_operator_approval: Some("smc_pd_array"),
    },
    SmcAliasSpec {
        flag: "inducement",
        candidates: &["smc_inducement"],
        pending_operator_approval: None,
    },
    SmcAliasSpec {
        flag: "bos",
        candidates: &["smc_bos"],
        pending_operator_approval: None,
    },
    SmcAliasSpec {
        flag: "choch",
        // No `smc_choch` column exists. `smc_mss` (market-structure shift) is
        // the same event class in most SMC literature, but "market-structure
        // shift IS change-of-character" is a semantic claim, not a lookup, and
        // it is the operator's to make.
        candidates: &["smc_choch"],
        pending_operator_approval: Some("smc_mss"),
    },
    SmcAliasSpec {
        flag: "eqh",
        candidates: &["smc_eqh"],
        pending_operator_approval: None,
    },
    SmcAliasSpec {
        flag: "eql",
        candidates: &["smc_eql"],
        pending_operator_approval: None,
    },
    SmcAliasSpec {
        flag: "displacement",
        candidates: &["smc_displacement"],
        pending_operator_approval: None,
    },
];

/// Exact-match lookup. The CANDIDATE list is the outer loop, so the priority
/// order in [`SMC_ALIAS_SPECS`] is real: the family's own name wins even when a
/// legacy spelling is emitted at a lower column index.
fn find_exact_column(names: &[String], candidates: &[&str]) -> Option<usize> {
    for cand in candidates {
        let want = normalize_feature_name(cand);
        if let Some(idx) = names
            .iter()
            .position(|raw| normalize_feature_name(raw) == want)
        {
            return Some(idx);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Binding accounting — no silent fallback.
// ---------------------------------------------------------------------------

/// Frames in which each flag bound a real SMC column.
static SMC_BOUND_FRAMES: [AtomicU64; SMC_FLAG_COUNT] =
    [const { AtomicU64::new(0) }; SMC_FLAG_COUNT];
/// Frames in which each flag had NO column and fell back to the crude
/// bar-derived approximation in [`derive_smc_arrays`].
static SMC_FALLBACK_FRAMES: [AtomicU64; SMC_FLAG_COUNT] =
    [const { AtomicU64::new(0) }; SMC_FLAG_COUNT];
/// Hash of the last binding actually logged. `build_smc_arrays` runs once per
/// candidate pool, so logging every call would drown the run; logging on CHANGE
/// means a prefilter that dropped `smc_ob` shows up as a changed binding
/// instead of as silence. `0` means "nothing logged yet".
static SMC_LAST_LOGGED_BINDING: AtomicU64 = AtomicU64::new(0);

/// Per-flag accounting of how the SMC gate was actually fed. `bound_frames`
/// counted a real SMC column; `fallback_frames` counted the crude 12/20-bar
/// re-derivation standing in for one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmcFlagTally {
    pub flag: &'static str,
    pub bound_frames: u64,
    pub fallback_frames: u64,
    /// The column that would bind this flag if the operator approved the
    /// semantic equation named in `SMC_ALIAS_SPECS`.
    pub pending_candidate: Option<&'static str>,
}

/// Snapshot of the SMC binding accounting for the process so far. Intended for
/// a run-end summary next to the other ledgers.
pub fn smc_binding_tally() -> Vec<SmcFlagTally> {
    SMC_ALIAS_SPECS
        .iter()
        .enumerate()
        .map(|(i, spec)| SmcFlagTally {
            flag: spec.flag,
            bound_frames: SMC_BOUND_FRAMES[i].load(Ordering::Relaxed),
            fallback_frames: SMC_FALLBACK_FRAMES[i].load(Ordering::Relaxed),
            pending_candidate: spec.pending_operator_approval,
        })
        .collect()
}

/// Emit the SMC binding tally. Call once at run end; every line is a count, so
/// a flag that spent the whole run on the crude approximation cannot be read as
/// a flag that was fed the real column.
pub fn log_smc_binding_tally() {
    for t in smc_binding_tally() {
        if t.bound_frames == 0 && t.fallback_frames == 0 {
            continue;
        }
        if t.fallback_frames == 0 {
            tracing::info!(
                target: "neoethos_search::smc_binding",
                flag = t.flag,
                bound_frames = t.bound_frames,
                "SMC gate fed by its real SMC column in every frame"
            );
        } else {
            tracing::warn!(
                target: "neoethos_search::smc_binding",
                flag = t.flag,
                bound_frames = t.bound_frames,
                fallback_frames = t.fallback_frames,
                pending_candidate = t.pending_candidate.unwrap_or("none"),
                "SMC gate fed by the crude 12/20-bar re-derivation in \
                 derive_smc_arrays, NOT by the SMC feature family, in \
                 fallback_frames frames"
            );
        }
    }
}

#[cfg(test)]
fn reset_smc_binding_log_dedup() {
    SMC_LAST_LOGGED_BINDING.store(0, Ordering::Relaxed);
}

/// Count the binding and, when it differs from the last one logged, say what it
/// is. The counts are unconditional — the de-duplication only silences repeated
/// LOG LINES, never the accounting.
fn record_binding(names: &[String], bound: &[Option<usize>; SMC_FLAG_COUNT]) {
    let mut hasher = DefaultHasher::new();
    for (i, spec) in SMC_ALIAS_SPECS.iter().enumerate() {
        spec.flag.hash(&mut hasher);
        match bound[i] {
            Some(col) => {
                SMC_BOUND_FRAMES[i].fetch_add(1, Ordering::Relaxed);
                names[col].hash(&mut hasher);
            }
            None => {
                SMC_FALLBACK_FRAMES[i].fetch_add(1, Ordering::Relaxed);
                "<unbound>".hash(&mut hasher);
            }
        }
    }
    // `0` is the "nothing logged yet" sentinel, so fold it onto 1.
    let sig = hasher.finish().max(1);
    if SMC_LAST_LOGGED_BINDING.swap(sig, Ordering::Relaxed) == sig {
        return;
    }

    let mut rendered: Vec<String> = Vec::with_capacity(SMC_FLAG_COUNT);
    for (i, spec) in SMC_ALIAS_SPECS.iter().enumerate() {
        match bound[i] {
            Some(col) => rendered.push(format!("{} -> {}[{}]", spec.flag, names[col], col)),
            None => rendered.push(format!("{} -> <unbound>", spec.flag)),
        }
    }
    tracing::info!(
        target: "neoethos_search::smc_binding",
        n_columns = names.len(),
        binding = %rendered.join(", "),
        "SMC gate column binding (exact match on the frame build_smc_arrays received)"
    );

    for (i, spec) in SMC_ALIAS_SPECS.iter().enumerate() {
        if bound[i].is_some() {
            continue;
        }
        tracing::warn!(
            target: "neoethos_search::smc_binding",
            flag = spec.flag,
            candidates = %spec.candidates.join(" | "),
            pending_candidate = spec.pending_operator_approval.unwrap_or("none"),
            "SMC alias UNBOUND — no column in this frame carries any of its exact \
             names, so this gate's vote comes from the crude 12/20-bar \
             re-derivation in derive_smc_arrays, NOT from the SMC feature family"
        );
    }
}

fn quantize_dir(value: f64) -> i8 {
    if value > 1e-9 {
        1
    } else if value < -1e-9 {
        -1
    } else {
        0
    }
}

fn quantize_binary(value: f64) -> i8 {
    if value > 1e-9 { 1 } else { 0 }
}

/// Resolve every SMC flag to a column index by EXACT name. Pure — no logging,
/// no counting — so tests can assert the binding rule on its own.
fn resolve_smc_columns(names: &[String]) -> [Option<usize>; SMC_FLAG_COUNT] {
    let mut out = [None; SMC_FLAG_COUNT];
    for (i, spec) in SMC_ALIAS_SPECS.iter().enumerate() {
        out[i] = find_exact_column(names, spec.candidates);
    }
    out
}

fn detect_smc_columns(names: &[String]) -> SmcColumns {
    let bound = resolve_smc_columns(names);
    record_binding(names, &bound);
    SmcColumns {
        ob: bound[SMC_IDX_OB],
        fvg: bound[SMC_IDX_FVG],
        liq: bound[SMC_IDX_LIQ],
        trend: bound[SMC_IDX_TREND],
        premium: bound[SMC_IDX_PREMIUM],
        inducement: bound[SMC_IDX_INDUCEMENT],
        bos: bound[SMC_IDX_BOS],
        choch: bound[SMC_IDX_CHOCH],
        eqh: bound[SMC_IDX_EQH],
        eql: bound[SMC_IDX_EQL],
        displacement: bound[SMC_IDX_DISPLACEMENT],
    }
}

pub fn derive_smc_arrays(ohlcv: &Ohlcv) -> SmcSignalTuple {
    let n = ohlcv.close.len();
    let mut ob = vec![0_i8; n];
    let mut fvg = vec![0_i8; n];
    let mut liq = vec![0_i8; n];
    let mut trend = vec![0_i8; n];
    let mut premium = vec![0_i8; n];
    let mut inducement = vec![0_i8; n];
    let mut bos = vec![0_i8; n];
    let mut choch = vec![0_i8; n];
    let mut eqh = vec![0_i8; n];
    let mut eql = vec![0_i8; n];
    let mut displacement = vec![0_i8; n];

    if n == 0 {
        return (
            ob,
            fvg,
            liq,
            trend,
            premium,
            inducement,
            bos,
            choch,
            eqh,
            eql,
            displacement,
        );
    }

    let lookback = 12usize;
    let eq_lookback = 20usize;
    let displacement_lookback = 20usize;

    for i in 0..n {
        if i >= lookback {
            let d = ohlcv.close[i] - ohlcv.close[i - lookback];
            trend[i] = if d > 0.0 {
                1
            } else if d < 0.0 {
                -1
            } else {
                0
            };
        } else if i > 0 {
            let d = ohlcv.close[i] - ohlcv.close[i - 1];
            trend[i] = if d > 0.0 {
                1
            } else if d < 0.0 {
                -1
            } else {
                0
            };
        }

        let mid = (ohlcv.high[i] + ohlcv.low[i]) * 0.5;
        premium[i] = if ohlcv.close[i] <= mid { 1 } else { -1 };

        if i >= 1 {
            let bull = ohlcv.close[i] > ohlcv.open[i]
                && ohlcv.close[i - 1] < ohlcv.open[i - 1]
                && ohlcv.close[i] >= ohlcv.high[i - 1];
            let bear = ohlcv.close[i] < ohlcv.open[i]
                && ohlcv.close[i - 1] > ohlcv.open[i - 1]
                && ohlcv.close[i] <= ohlcv.low[i - 1];
            ob[i] = if bull {
                1
            } else if bear {
                -1
            } else {
                0
            };

            let body = (ohlcv.close[i] - ohlcv.open[i]).abs();
            let upper = ohlcv.high[i] - ohlcv.open[i].max(ohlcv.close[i]);
            let lower = ohlcv.open[i].min(ohlcv.close[i]) - ohlcv.low[i];
            if body > 1e-12 && ((upper / body) > 2.0 || (lower / body) > 2.0) {
                inducement[i] = 1;
            }
        }

        if i >= 2 {
            if ohlcv.low[i] > ohlcv.high[i - 2] {
                fvg[i] = 1;
            } else if ohlcv.high[i] < ohlcv.low[i - 2] {
                fvg[i] = -1;
            }
        }

        if i >= 3 {
            let prev_low = ohlcv.low[(i - 3)..i]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let prev_high = ohlcv.high[(i - 3)..i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if ohlcv.low[i] < prev_low && ohlcv.close[i] > prev_low {
                liq[i] = 1;
            } else if ohlcv.high[i] > prev_high && ohlcv.close[i] < prev_high {
                liq[i] = -1;
            }
        }

        if i >= lookback {
            let prev_low = ohlcv.low[(i - lookback)..i]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let prev_high = ohlcv.high[(i - lookback)..i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if ohlcv.close[i] > prev_high {
                bos[i] = 1;
            } else if ohlcv.close[i] < prev_low {
                bos[i] = -1;
            }
        }

        if i >= 1 && trend[i] != 0 && trend[i - 1] != 0 && trend[i] != trend[i - 1] {
            choch[i] = trend[i];
        }

        if i >= eq_lookback {
            let lb = i - eq_lookback;
            let mut range_sum = 0.0;
            for j in lb..=i {
                range_sum += (ohlcv.high[j] - ohlcv.low[j]).abs();
            }
            let avg_range = range_sum / ((eq_lookback as f64) + 1.0);
            let tol = (avg_range * 0.1).max(1e-9);
            for j in lb..i {
                if (ohlcv.high[i] - ohlcv.high[j]).abs() <= tol {
                    eqh[i] = -1;
                    break;
                }
            }
            for j in lb..i {
                if (ohlcv.low[i] - ohlcv.low[j]).abs() <= tol {
                    eql[i] = 1;
                    break;
                }
            }
        }

        if i >= displacement_lookback {
            let body = (ohlcv.close[i] - ohlcv.open[i]).abs();
            let mut avg_body = 0.0;
            for j in (i - displacement_lookback)..i {
                avg_body += (ohlcv.close[j] - ohlcv.open[j]).abs();
            }
            avg_body /= displacement_lookback as f64;
            if avg_body > 1e-12 && body >= (1.8 * avg_body) {
                displacement[i] = if ohlcv.close[i] > ohlcv.open[i] {
                    1
                } else if ohlcv.close[i] < ohlcv.open[i] {
                    -1
                } else {
                    0
                };
            }
        }
    }

    (
        ob,
        fvg,
        liq,
        trend,
        premium,
        inducement,
        bos,
        choch,
        eqh,
        eql,
        displacement,
    )
}

#[cfg(test)]
mod column_binding_tests {
    use super::*;
    use ndarray::Array2;
    use neoethos_data::test_fixtures::ctrader_sample_ohlcv;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Every column name in the real vocabulary that CONTAINS an SMC alias as a
    /// substring but is not an SMC column. Under the old
    /// `norm.contains(alias)` rule each of these could capture a gate; `obv`
    /// and `fvg_positioning_average` and `quant_amihud_illiquidity` are real
    /// ids in `all_indicators.rs` today, and `choch_reversal_probability` /
    /// `premium_zone_index` are the shape of the next one to be added.
    const DECOYS: [&str; 14] = [
        "obv",
        "moving_average_cross_probability",
        "fvg_positioning_average",
        "fvg_trailing_stop",
        "quant_amihud_illiquidity",
        "adaptive_schaff_trend_cycle",
        "regime_wilder_adx_14_v3",
        "supertrend",
        "choch_reversal_probability",
        "premium_zone_index",
        "bos_confirmation_index",
        "eqh_cluster_density",
        "eql_cluster_density",
        "displacement_impulse_ratio",
    ];

    /// The SMC family's real column names, in the order `smc.rs` emits them.
    const REAL_SMC: [&str; 9] = [
        "smc_ob",
        "smc_fvg",
        "smc_liq_sweep",
        "smc_displacement",
        "smc_mss",
        "smc_bos",
        "smc_eqh",
        "smc_eql",
        "smc_inducement",
    ];

    /// THE TEST THIS FIX EXISTS FOR. A cube made only of decoys binds NOTHING.
    /// Under the old substring rule this frame captured eight of the eleven
    /// gates and fed them unrelated features in silence.
    #[test]
    fn decoy_columns_cannot_capture_any_smc_alias() {
        let bound = resolve_smc_columns(&names(&DECOYS));
        for (i, spec) in SMC_ALIAS_SPECS.iter().enumerate() {
            assert!(
                bound[i].is_none(),
                "alias `{}` was captured by decoy column `{}` — substring \
                 matching is back",
                spec.flag,
                DECOYS[bound[i].unwrap()]
            );
        }
    }

    /// The decoys come FIRST in column order. The old rule looped columns
    /// outermost and returned the first hit, so it would have answered `obv`
    /// for `ob` even with `smc_ob` present later in the frame. Position must
    /// not decide the binding.
    #[test]
    fn a_decoy_at_a_lower_index_does_not_beat_the_real_column() {
        let mut cols: Vec<&str> = DECOYS.to_vec();
        cols.extend_from_slice(&REAL_SMC);
        cols.push("smc_trend_bias");
        let all = names(&cols);
        let bound = resolve_smc_columns(&all);

        let expect = [
            (SMC_IDX_OB, "smc_ob"),
            (SMC_IDX_FVG, "smc_fvg"),
            (SMC_IDX_LIQ, "smc_liq_sweep"),
            (SMC_IDX_TREND, "smc_trend_bias"),
            (SMC_IDX_INDUCEMENT, "smc_inducement"),
            (SMC_IDX_BOS, "smc_bos"),
            (SMC_IDX_EQH, "smc_eqh"),
            (SMC_IDX_EQL, "smc_eql"),
            (SMC_IDX_DISPLACEMENT, "smc_displacement"),
        ];
        for (idx, want) in expect {
            let got = bound[idx].expect("real SMC column must bind");
            assert_eq!(
                all[got], want,
                "alias `{}` bound `{}` instead of `{}`",
                SMC_ALIAS_SPECS[idx].flag, all[got], want
            );
        }
    }

    /// The candidate list is the priority order, and it is real now. The legacy
    /// spelling sits at index 0 and the family's own name at index 1; the
    /// family's name must still win.
    #[test]
    fn candidate_priority_beats_column_order() {
        let all = names(&["smc_liq", "smc_liq_sweep", "smc_trend", "smc_trend_bias"]);
        let bound = resolve_smc_columns(&all);
        assert_eq!(all[bound[SMC_IDX_LIQ].unwrap()], "smc_liq_sweep");
        assert_eq!(all[bound[SMC_IDX_TREND].unwrap()], "smc_trend_bias");
    }

    /// A higher-timeframe copy is a DIFFERENT signal. Exact matching refuses
    /// it; the substring rule accepted it whenever the base column had been
    /// dropped by the prefilter, which is precisely the case where nobody was
    /// watching.
    #[test]
    fn timeframe_prefixed_copies_do_not_bind() {
        let all = names(&["h1_smc_ob", "h4_smc_fvg", "d1_smc_bos"]);
        let bound = resolve_smc_columns(&all);
        assert!(bound[SMC_IDX_OB].is_none());
        assert!(bound[SMC_IDX_FVG].is_none());
        assert!(bound[SMC_IDX_BOS].is_none());
    }

    /// Case and separator normalisation still applies — it just applies to an
    /// exact comparison instead of a containment test.
    #[test]
    fn binding_is_case_and_separator_insensitive_but_still_exact() {
        let all = names(&["SMC-OB", "Smc Fvg", "SMC_OBV"]);
        let bound = resolve_smc_columns(&all);
        assert_eq!(bound[SMC_IDX_OB], Some(0));
        assert_eq!(bound[SMC_IDX_FVG], Some(1));
        // `SMC_OBV` normalises to `smc_obv`, which is not `smc_ob`.
        assert_ne!(bound[SMC_IDX_OB], Some(2));
    }

    /// The two aliases with no possible binding in the shipped vocabulary,
    /// asserted against the family's real emission list so the day a
    /// `smc_choch` or `smc_premium` column appears, this test changes.
    #[test]
    fn choch_and_premium_are_unbound_in_the_real_vocabulary() {
        let all = names(&REAL_SMC);
        let bound = resolve_smc_columns(&all);
        assert!(bound[SMC_IDX_CHOCH].is_none());
        assert!(bound[SMC_IDX_PREMIUM].is_none());
        assert_eq!(
            SMC_ALIAS_SPECS[SMC_IDX_CHOCH].pending_operator_approval,
            Some("smc_mss"),
            "smc_mss is present in this frame and is NOT bound — the semantic \
             equation is the operator's to make"
        );
        assert_eq!(
            SMC_ALIAS_SPECS[SMC_IDX_PREMIUM].pending_operator_approval,
            Some("smc_pd_array")
        );
    }

    /// An unbound alias is COUNTED, not swallowed. Counts are process-global
    /// and other tests run in parallel, so this asserts a strict increase
    /// rather than an absolute value.
    #[test]
    fn unbound_aliases_are_counted_by_name() {
        let before: Vec<u64> = SMC_FALLBACK_FRAMES
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let _ = detect_smc_columns(&names(&DECOYS));
        let after: Vec<u64> = SMC_FALLBACK_FRAMES
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        for i in 0..SMC_FLAG_COUNT {
            assert!(
                after[i] > before[i],
                "fallback for `{}` was not counted",
                SMC_ALIAS_SPECS[i].flag
            );
        }
        let tally = smc_binding_tally();
        assert_eq!(tally.len(), SMC_FLAG_COUNT);
        assert!(tally.iter().all(|t| !t.flag.is_empty()));
    }

    /// The binding log de-duplicates on the binding, not on the call. A frame
    /// whose binding CHANGED — the prefilter dropped `smc_ob`, say — must
    /// re-log, because that change is exactly what has to be visible.
    #[test]
    fn a_changed_binding_is_logged_again() {
        reset_smc_binding_log_dedup();
        let with_ob = names(&["smc_ob", "smc_fvg"]);
        let without_ob = names(&["smc_fvg"]);
        let a = resolve_smc_columns(&with_ob);
        let b = resolve_smc_columns(&without_ob);
        assert!(a[SMC_IDX_OB].is_some());
        assert!(b[SMC_IDX_OB].is_none());
        record_binding(&with_ob, &a);
        let sig_a = SMC_LAST_LOGGED_BINDING.load(Ordering::Relaxed);
        record_binding(&without_ob, &b);
        let sig_b = SMC_LAST_LOGGED_BINDING.load(Ordering::Relaxed);
        assert_ne!(
            sig_a, sig_b,
            "a dropped SMC column must change the signature"
        );
        assert_ne!(sig_a, 0);
    }

    /// End to end on real EURUSD M1 bars: a frame of nothing but decoys must
    /// leave every one of the eleven arrays exactly where the bars put them.
    /// Any capture would perturb at least one of them.
    ///
    /// The baseline is NOT `derive_smc_arrays` verbatim, and this is not a
    /// concession to make a test pass. `build_smc_arrays` ends with a step
    /// that reads the arrays it is already holding rather than the frame:
    /// inducement is promoted to 1 wherever DISPLACEMENT is non-zero. With
    /// nothing bound those arrays are the derived ones, so the promotion
    /// still fires and `build != derive` for inducement — which has been true
    /// since long before the exact-binding change (verified present unchanged
    /// at `6c4e9390^`). The baseline therefore applies that same promotion
    /// explicitly. Written out here rather than borrowed from the function
    /// under test, so a leak cannot hide inside a shared helper.
    #[test]
    fn build_from_a_decoy_only_frame_equals_the_bar_derived_arrays() {
        let ohlcv = ctrader_sample_ohlcv();
        let n = ohlcv.close.len();
        let cols = names(&DECOYS);
        let mut data = Array2::<f64>::zeros((n, cols.len()));
        // Fill with a signal that would be loudly visible if it leaked into a
        // gate array: alternating +1 / -1.
        for i in 0..n {
            for j in 0..cols.len() {
                data[(i, j)] = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        let ts = ohlcv
            .timestamp
            .clone()
            .expect("cTrader fixture has canonical timestamps");
        let frame =
            neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(ts, cols, data)
                .expect("valid f64 decoy frame");

        let built = build_smc_arrays(&frame, &ohlcv).expect("SMC arrays build");
        let mut derived = derive_smc_arrays(&ohlcv);
        // The frame-independent tail step of `build_smc_arrays`, restated.
        let disp_baseline = derived.10.clone();
        for (disp, slot) in disp_baseline.iter().zip(derived.5.iter_mut()) {
            if *disp != 0 {
                *slot = 1;
            }
        }
        assert!(
            disp_baseline.iter().any(|d| *d != 0),
            "the promotion step is vacuous on these bars — the inducement \
             assertion below would then prove nothing"
        );
        assert_eq!(built.0, derived.0, "ob leaked");
        assert_eq!(built.1, derived.1, "fvg leaked");
        assert_eq!(built.2, derived.2, "liq leaked");
        assert_eq!(built.3, derived.3, "trend leaked");
        assert_eq!(built.4, derived.4, "premium leaked");
        assert_eq!(built.5, derived.5, "inducement leaked");
        assert_eq!(built.6, derived.6, "bos leaked");
        assert_eq!(built.7, derived.7, "choch leaked");
        assert_eq!(built.8, derived.8, "eqh leaked");
        assert_eq!(built.9, derived.9, "eql leaked");
        assert_eq!(built.10, derived.10, "displacement leaked");
    }

    /// The spec table, the index consts and `SmcColumns` must stay in lockstep.
    #[test]
    fn the_alias_table_is_well_formed() {
        assert_eq!(SMC_ALIAS_SPECS.len(), SMC_FLAG_COUNT);
        for spec in SMC_ALIAS_SPECS.iter() {
            assert!(
                !spec.candidates.is_empty(),
                "alias `{}` has no candidate column",
                spec.flag
            );
            for c in spec.candidates {
                assert_eq!(
                    *c,
                    normalize_feature_name(c),
                    "candidate `{c}` is not already normalised"
                );
                assert!(
                    c.starts_with("smc_"),
                    "candidate `{c}` is not smc_-prefixed, so a classic \
                     indicator could legitimately own that exact name"
                );
            }
        }
        // Flag order must match the tuple order the gate consumes.
        let order = [
            "ob",
            "fvg",
            "liq",
            "trend",
            "premium",
            "inducement",
            "bos",
            "choch",
            "eqh",
            "eql",
            "displacement",
        ];
        for (i, want) in order.iter().enumerate() {
            assert_eq!(SMC_ALIAS_SPECS[i].flag, *want);
        }
    }
}

pub fn build_smc_arrays(frame: &FeatureFrame, ohlcv: &Ohlcv) -> anyhow::Result<SmcSignalTuple> {
    let n = frame.n_samples();
    let cols = detect_smc_columns(&frame.names);
    let (
        mut ob,
        mut fvg,
        mut liq,
        mut trend,
        mut premium,
        mut inducement,
        mut bos,
        mut choch,
        mut eqh,
        mut eql,
        mut displacement,
    ) = derive_smc_arrays(ohlcv);

    let valid_value = |column: &neoethos_data::FeatureColumnF64, row: usize| {
        column.validity[row]
            .is_valid()
            .then_some(column.values[row])
    };
    let apply_dir_col = |target: &mut Vec<i8>, col_opt: Option<usize>| -> anyhow::Result<()> {
        if let Some(col) = col_opt
            && col < frame.n_features()
        {
            let column = frame.feature_column(col)?;
            for (i, slot) in target.iter_mut().enumerate().take(n) {
                *slot = valid_value(&column, i).map_or(0, quantize_dir);
            }
        }
        Ok(())
    };
    let apply_binary_col = |target: &mut Vec<i8>, col_opt: Option<usize>| -> anyhow::Result<()> {
        if let Some(col) = col_opt
            && col < frame.n_features()
        {
            let column = frame.feature_column(col)?;
            for (i, slot) in target.iter_mut().enumerate().take(n) {
                *slot = valid_value(&column, i).map_or(0, quantize_binary);
            }
        }
        Ok(())
    };
    let apply_eqh_col = |target: &mut Vec<i8>, col_opt: Option<usize>| -> anyhow::Result<()> {
        if let Some(col) = col_opt
            && col < frame.n_features()
        {
            let column = frame.feature_column(col)?;
            for (i, slot) in target.iter_mut().enumerate().take(n) {
                let Some(v) = valid_value(&column, i) else {
                    *slot = 0;
                    continue;
                };
                let q = quantize_dir(v);
                *slot = if q != 0 {
                    q
                } else if quantize_binary(v) != 0 {
                    -1
                } else {
                    0
                };
            }
        }
        Ok(())
    };
    let apply_eql_col = |target: &mut Vec<i8>, col_opt: Option<usize>| -> anyhow::Result<()> {
        if let Some(col) = col_opt
            && col < frame.n_features()
        {
            let column = frame.feature_column(col)?;
            for (i, slot) in target.iter_mut().enumerate().take(n) {
                let Some(v) = valid_value(&column, i) else {
                    *slot = 0;
                    continue;
                };
                let q = quantize_dir(v);
                *slot = if q != 0 {
                    q
                } else if quantize_binary(v) != 0 {
                    1
                } else {
                    0
                };
            }
        }
        Ok(())
    };
    // **F-040 documentation (2026-05-25)** — this closure fills zero
    // slots in `target` with the direction signal from a SECONDARY
    // column (typically BoS / CHoCH / displacement). The audit flagged
    // it as "conflating separate signals" because the source column's
    // direction is treated as the target column's direction when the
    // primary column was silent.
    //
    // The conflation is INTENTIONAL: SMC theory treats BoS / CHoCH /
    // displacement as direction-confirming signals — when an Order
    // Block hasn't been tagged in this bar but a Break-of-Structure
    // is signalling the same direction, the OB inherits that
    // direction for the gate-vote. The legacy behaviour is preserved
    // here per operator directive 2026-05-25 ("ομοιομορφία είναι
    // καλό" — uniformity of SMC voting rules across the indicators).
    // A future research-driven sweep may split these into separate
    // gate-votes; that's a Phase-C scope decision, not a bug.
    let apply_dir_fill_zeros =
        |target: &mut Vec<i8>, col_opt: Option<usize>| -> anyhow::Result<()> {
            if let Some(col) = col_opt
                && col < frame.n_features()
            {
                let column = frame.feature_column(col)?;
                for (i, slot) in target.iter_mut().enumerate().take(n) {
                    if *slot == 0 {
                        *slot = valid_value(&column, i).map_or(0, quantize_dir);
                    }
                }
            }
            Ok(())
        };
    let apply_eq_levels = |target: &mut Vec<i8>,
                           eqh_col: Option<usize>,
                           eql_col: Option<usize>|
     -> anyhow::Result<()> {
        if let Some(col) = eqh_col
            && col < frame.n_features()
        {
            let column = frame.feature_column(col)?;
            for (i, slot) in target.iter_mut().enumerate().take(n) {
                if valid_value(&column, i).is_some_and(|value| quantize_binary(value) != 0) {
                    *slot = -1;
                }
            }
        }
        if let Some(col) = eql_col
            && col < frame.n_features()
        {
            let column = frame.feature_column(col)?;
            for (i, slot) in target.iter_mut().enumerate().take(n) {
                if valid_value(&column, i).is_some_and(|value| quantize_binary(value) != 0) {
                    *slot = 1;
                }
            }
        }
        Ok(())
    };

    apply_dir_col(&mut ob, cols.ob)?;
    apply_dir_col(&mut fvg, cols.fvg)?;
    apply_dir_col(&mut liq, cols.liq)?;
    apply_dir_col(&mut trend, cols.trend)?;
    apply_dir_col(&mut premium, cols.premium)?;
    apply_binary_col(&mut inducement, cols.inducement)?;
    apply_dir_col(&mut bos, cols.bos)?;
    apply_dir_col(&mut choch, cols.choch)?;
    apply_eqh_col(&mut eqh, cols.eqh)?;
    apply_eql_col(&mut eql, cols.eql)?;
    apply_dir_col(&mut displacement, cols.displacement)?;
    apply_dir_fill_zeros(&mut ob, cols.bos)?;
    apply_dir_fill_zeros(&mut ob, cols.choch)?;
    apply_eq_levels(&mut liq, cols.eqh, cols.eql)?;
    apply_dir_fill_zeros(&mut trend, cols.bos)?;
    apply_dir_fill_zeros(&mut trend, cols.choch)?;
    apply_dir_fill_zeros(&mut trend, cols.displacement)?;

    if let Some(col) = cols.displacement
        && col < frame.n_features()
    {
        let column = frame.feature_column(col)?;
        for (i, slot) in inducement.iter_mut().enumerate().take(n) {
            if valid_value(&column, i).is_some_and(|value| quantize_dir(value) != 0) {
                *slot = 1;
            }
        }
    }
    for (disp, slot) in displacement.iter().zip(inducement.iter_mut()) {
        if *disp != 0 {
            *slot = 1;
        }
    }

    Ok((
        ob,
        fvg,
        liq,
        trend,
        premium,
        inducement,
        bos,
        choch,
        eqh,
        eql,
        displacement,
    ))
}
