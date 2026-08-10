//! Every operator-settable value must reach something that changes behaviour.
//!
//! # Why this test exists
//!
//! A census on 2026-08-04 walked all 408 fields of `Settings` and asked, for
//! each one, "what reads this?". 51 had no answer. They were not dead code in
//! the usual sense — they parsed, validated, persisted, round-tripped through
//! `GET /settings`, and several had a control in the Settings UI. They simply
//! had **no final recipient**. Setting one and watching nothing change was the
//! entire experience.
//!
//! Three shapes were found, worst last:
//!
//! * **PHANTOM** — no consumer anywhere; belongs to a feature never built.
//!   45 of them, e.g. every `perplexity_*` knob but the enable flag, all four
//!   `auto_rescore_*`, `smc_freshness_limit`, `vortex_memory_map`. Deleted.
//! * **UNWIRED** — a consumer mechanism exists but nothing connects the config
//!   to it. `challenge_phase` is the specimen: `PropFirmPhaseRiskDefaults::
//!   for_preset(preset, phase)` takes exactly that string and only tests call
//!   it. Kept, documented, and listed in [`NO_QUALIFIED_READER`] below —
//!   because wiring them would change live sizing and must be a deliberate act.
//! * **STORED** — validated, persisted, displayed, then ignored. The purest
//!   defect, because it is indistinguishable from working. `ui_locale` offered
//!   an English/Ελληνικά picker in an app with no i18n layer;
//!   `news_calendar_source` accepted any provider id while the fetcher
//!   hardcoded ForexFactory's URL. Both fixed.
//!
//! # 2026-08-09 — the guard was itself an instance of the defect it guards
//!
//! The first version of this file asked one question per field: *does the token
//! `<field>` appear anywhere in the workspace after a dot?* It had no notion of
//! **which struct the receiver was**. So any type in any crate that happened to
//! carry a same-named field satisfied the check for a `Settings` knob that
//! nothing read. Confirmed false passes at the time of the rewrite:
//!
//! | knob | what actually satisfied it |
//! |------|----------------------------|
//! | `ModelsConfig::inference_batch_size` | `AutoTuneHints::inference_batch_size` |
//! | `SystemConfig::n_jobs` | `AutoTuneHints::n_jobs` |
//! | `RiskConfig::trailing_min_lock_pips` | `BacktestSettings::trailing_min_lock_pips` |
//! | `RiskConfig::high_quality_confidence` | `BacktestSettings::high_quality_confidence` |
//! | `RiskConfig::min_risk_reward` | `StopTargetSettings::min_risk_reward` |
//!
//! It also matched inside string literals and comments, so the knob catalog's
//! own help text (`id: "risk.require_stop_loss"`) counted as a reader, and it
//! counted an assignment TARGET as a read, so the auto-tuner overwriting
//! `settings.system.n_jobs` proved that `n_jobs` was honoured.
//!
//! Worse than any single false pass: the companion test asserted the two
//! no-recipient ledgers were **capped at 3 entries each** and "only ever
//! shrink". A cap is not a standard, it is pressure — pressure to find *any*
//! matching token rather than a real recipient, and pressure to not write down
//! what you found. That is how a test ends up designed to pass.
//!
//! **Measured on the day of the rewrite**: the loose scanner reported 3
//! orphans, all three of them WRONG (`backtest_spread_pips_asian/overlap/
//! late_ny` are read by `RiskConfig::session_spread_pips()`, which it could not
//! see because it skipped `config.rs` wholesale). The strict scanner clears
//! those three and reports **10 knobs it previously passed**, every one
//! verified by hand against the code and written into [`NO_QUALIFIED_READER`]
//! below with the site that fooled the old rule. Six of the ten are worse than
//! merely unread: four are ASSIGNED by production code that then never reads
//! them back, and four more (`risk.trailing_*`) are shadowed duplicates of the
//! `models.exit_policy` keys that actually drive the search — an operator
//! tuning the trail in the section named "risk" moves nothing at all.
//!
//! # What this test enforces now
//!
//! 1. Every field reachable from `Settings` is read through a **qualified**
//!    access whose receiver resolves to the struct that declares it:
//!    * `x.models.discovery_runtime.prefilter_top_k` — a dotted chain rooted at
//!      a `Settings` section, each link a real field of the previous link's
//!      type; or
//!    * `cfg.prefilter_top_k` where `cfg` is bound **in that file** to
//!      `DiscoveryRuntimeConfig` — by a typed parameter, a typed `let`, an
//!      `impl` block (`self`), or `let cfg = &settings.models.discovery_runtime`;
//!    * or `app_runtime().server_bind`, where `app_runtime()` is a function in
//!      that file whose return type is the declaring struct;
//!    * or a `let DiscoveryRuntimeConfig { prefilter_top_k, .. }` destructure;
//!    * or — for a field read only inside `config.rs` — an accessor method on
//!      the struct itself that something outside `config.rs` calls
//!      (`settings.risk.session_spread_pips()`). An accessor with no caller is
//!      not a recipient, it is more unwired code.
//!    Strings and comments are stripped first, and an assignment target
//!    (`settings.system.n_jobs = …`) is a write, not a read.
//! 2. A read that only ever happens in `tests/`, `benches/` or `examples/` is
//!    recorded as such and does not count as a production recipient.
//! 3. There is **no cap** on the ledger. There is an explicit allow-list, one
//!    entry per knob, each naming the mechanism it fails to reach and who owns
//!    the decision to wire or delete it. A stale entry — one that has since
//!    acquired a real reader — fails the test, so the list cannot rot.
//! 4. Every field `POST /settings` accepts is actually applied to `Settings`.
//! 5. No UI control is backed by a field in the ledger — the UI may only expose
//!    knobs that reach a live consumer.
//!
//! Without this, a census that found 51 in six months finds 60 in the next.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

// ───────────────────────────── the ledger ─────────────────────────────

/// Why a knob is allowed to have no qualified production reader.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Inert {
    /// A real consumer mechanism exists; connecting it would change trading
    /// behaviour, so wiring is a deliberate, measured act — not tidying.
    Unwired,
    /// The only qualified reader lives in `tests/`, `benches/` or `examples/`.
    /// The code that reads it is real; no production run touches it.
    TestOnly,
    /// Qualified code ASSIGNS the field and nothing reads it back. The operator
    /// sets it, something overwrites it, and no consumer ever looks. This is
    /// the purest form of the defect — the value is not merely ignored, it is
    /// destroyed first.
    WrittenNeverRead,
}

/// Knobs with no qualified production reader, each with the mechanism it fails
/// to reach and **who owns the decision**.
///
/// This list has NO CAP. A cap is pressure to cheat; an allow-list is a record.
/// Entries are checked both ways: an entry whose field no longer exists fails,
/// and an entry that has since acquired a qualified production reader fails as
/// stale. Adding one is a code change in this file, reviewed like any other.
const NO_QUALIFIED_READER: &[(&str, Inert, &str)] = &[
    // ─────────────────── kept on purpose: real mechanism, unwired ───────────
    (
        "RiskConfig::challenge_phase",
        Inert::Unwired,
        "domain::prop_firm::PropFirmPhaseRiskDefaults::for_preset(preset, phase) takes this \
         string; only its own #[cfg(test)] block calls it. OWNER: operator — wiring it changes \
         live position sizing on a funded account.",
    ),
    (
        "RiskConfig::recovery_mode_enabled",
        Inert::Unwired,
        "domain::risk::RiskManager::update_recovery_state flips recovery_mode from drawdown \
         alone and consults no toggle; RiskManager has no production constructor at all. \
         OWNER: operator — honouring the toggle changes drawdown behaviour mid-trade.",
    ),
    (
        "ModelsConfig::rl_population_size",
        Inert::Unwired,
        "evolution::neat_impl::NeatTrainer::population_size is hardcoded (96 default / 24 \
         floor); this field's default of 5 would collapse it 19-fold. OWNER: operator — \
         wiring it silently would change every NEAT run.",
    ),
    // ─── 2026-08-09: found the moment the scanner started resolving receivers.
    // Every one of these PASSED the old guard on a same-named field belonging to
    // some other struct. Each was checked by hand against the code before being
    // written down; the site that fooled the old scanner is named.
    (
        "RiskConfig::challenge_mode",
        Inert::Unwired,
        "the only `challenge_mode` reads are `domain::risk::RiskManager::challenge_mode` \
         (risk.rs:506,509), a DIFFERENT struct whose value arrives as the second argument to \
         `RiskManager::new` — and every `RiskManager::new` call in the workspace is inside a \
         test module, so there is no live call site to pass the config value to. Same root \
         cause as `recovery_mode_enabled` above. OWNER: operator — deciding challenge vs \
         monthly-target gating changes when live trading stops.",
    ),
    (
        "RiskConfig::high_quality_confidence",
        Inert::Unwired,
        "the consumer is `eval.rs:736 let hq = settings.high_quality_confidence`, but that \
         `settings` is a `BacktestSettings`, whose field defaults to 0.65 at eval.rs:389 and is \
         assigned nowhere outside #[cfg(test)]. The two 0.65s agree today, which is exactly why \
         nobody noticed: changing the config moves nothing and looks like it worked. OWNER: \
         operator — it scales position size per signal confidence.",
    ),
    // ─── the four trailing shadows. 2026-08-10 UPDATE: the disagreement is now
    // NAMED IN THE LOG of every run — `resolve_and_log_duplicate_knobs`
    // (neoethos-search/src/discovery.rs) prints the winner, the loser and all
    // four value pairs once per run, WARN when they differ. That is the whole
    // change; which copy wins is unaltered.
    //
    // ⚠ DELETION IS BLOCKED, AND THE ORDER MATTERS. Live execution
    // (`live_trading.rs:1226-1272`) trails UNCONDITIONALLY with no config
    // recipient of any kind. Deleting these four before live reads
    // `models.exit_policy` converts a visibly-wrong value into an INVISIBLE
    // HARDCODE on the path that spends real money — strictly worse than the
    // defect being fixed. Wire live to `models.exit_policy` first; then delete.
    // Tracked in `docs/pending-edits-forbidden-territory.md`.
    (
        "RiskConfig::trailing_enabled",
        Inert::Unwired,
        "SHADOWED DUPLICATE. The search's exit geometry comes from `models.exit_policy` \
         (strategy_gene.rs `let exit = overrides.exit_policy`), added 2026-08-09 to replace \
         the hardcode. `risk.trailing_enabled` reaches nothing; the reads that fooled the old \
         guard are `BacktestSettings::trailing_enabled` in cubecl_eval.rs. The operator's live \
         store sets THIS copy and carries no `models.exit_policy` block at all, so his \
         hand-tuned numbers move nothing and the ExitPolicy Rust defaults are what run — \
         which every run now says out loud. OWNER: operator — delete this key AFTER live \
         trailing reads models.exit_policy, never before.",
    ),
    (
        "RiskConfig::trailing_atr_multiplier",
        Inert::Unwired,
        "SHADOWED DUPLICATE of `models.exit_policy.trailing_stop_multiplier` — see \
         `RiskConfig::trailing_enabled`. Despite the name it was never an ATR multiple: it is \
         a multiple of the position's own stop distance. The rename is part of the migration \
         and must be carried across, not dropped. OWNER: operator.",
    ),
    (
        "RiskConfig::trailing_be_trigger_r",
        Inert::Unwired,
        "SHADOWED DUPLICATE of `models.exit_policy.trailing_be_trigger_r` — see \
         `RiskConfig::trailing_enabled`. OWNER: operator.",
    ),
    (
        "RiskConfig::trailing_min_lock_pips",
        Inert::Unwired,
        "SHADOWED DUPLICATE of `models.exit_policy.trailing_min_lock_pips` — see \
         `RiskConfig::trailing_enabled`. OWNER: operator.",
    ),
    (
        "RiskConfig::monthly_profit_target_pct",
        Inert::WrittenNeverRead,
        "`server/risk.rs:174` OVERWRITES it from the chosen prop-firm preset on every \
         `POST /risk/preset`, and the only readers are `domain::risk::PropFirmRules::\
         monthly_profit_target_pct` / `RiskManager::monthly_profit_target_pct`, both built from \
         `PropFirmConstraints`, never from Settings. So the operator's number is replaced and \
         then not consulted. OWNER: operator — it is a live trading stop condition.",
    ),
    // ─── the three hardware-derived fields. SCHEDULED FOR DELETION in the
    // 2026-08-10 config wave (knob pass D4): the writer that made them
    // `WrittenNeverRead` — `AutoTuner::apply`, which had zero callers — is being
    // deleted along with them, and the honest source for all three is the
    // hardware probe (`HardwareExecutionPlan`), not a config file.
    //
    // ⚠ WHEN THE FIELD GOES, DELETE ITS ENTRY HERE IN THE SAME CHANGE. This
    // ledger is checked BOTH ways: an entry naming a field that no longer exists
    // fails as stale, exactly so that a half-landed deletion cannot pass
    // quietly. If this test fails with "ledger entry for an unknown field", the
    // fix is to remove the entry, not to re-add the field.
    (
        "SystemConfig::n_jobs",
        Inert::WrittenNeverRead,
        "the dead `AutoTuner::apply` assigned `self.settings.system.n_jobs = hints.n_jobs` and \
         every consumer read `AutoTuneHints::n_jobs` instead. One of the two false passes that \
         motivated the 2026-08-09 rewrite. The value in the shipped config (11) is the \
         fingerprint of `available_parallelism() - 1` on a 12-core box, pickled into YAML by \
         `Settings::save` and then shipped to every other machine. OWNER: operator — DELETE; \
         rayon width is a property of the box, not of a config file.",
    ),
    (
        "SystemConfig::num_gpus",
        Inert::WrittenNeverRead,
        "same dead writer; every reader is `HardwareProfile::num_gpus` (scheduler.rs, \
         cli/main.rs) which is probed from the machine, not from config. The shipped `0` is a \
         detector result frozen on a box with no card and then carried to a 3090. OWNER: \
         operator — DELETE, the probe is the honest source.",
    ),
    (
        "ModelsConfig::inference_batch_size",
        Inert::WrittenNeverRead,
        "same dead writer; the consumers use `AutoTuneHints::inference_batch_size`. The second \
         of the two false passes that motivated the rewrite. Ships 32 in both repo files — a \
         value the hardware plan would never produce. OWNER: operator — DELETE; wiring it would \
         let a config value override a hardware-derived batch size, which has NEVER-OOM \
         implications.",
    ),
];

/// The subset of [`NO_QUALIFIED_READER`] keys, as `Owner::field`.
fn ledger_keys() -> BTreeSet<&'static str> {
    NO_QUALIFIED_READER.iter().map(|(k, _, _)| *k).collect()
}

/// Bare field names in the ledger — used by the UI-side checks, which only
/// know a control's key and not the struct behind it.
fn ledger_field_names() -> BTreeSet<&'static str> {
    NO_QUALIFIED_READER
        .iter()
        .filter_map(|(k, _, _)| k.split("::").nth(1))
        .collect()
}

// ───────────────────────── the config type graph ──────────────────────────

fn workspace_root() -> PathBuf {
    // crates/neoethos-core -> crates -> <workspace root>
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("neoethos-core must live two levels below the workspace root")
        .to_path_buf()
}

fn config_rs() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("config.rs")
}

/// The struct graph declared in `config.rs`, rooted at `Settings`.
#[derive(Default)]
struct ConfigModel {
    /// struct name -> its fields, in declaration order, with the base type name.
    structs: BTreeMap<String, Vec<(String, String)>>,
    /// struct name -> set of field names it declares.
    declares: BTreeMap<String, BTreeSet<String>>,
    /// field name -> the config struct type(s) that field installs, when the
    /// field's type is itself a struct in `config.rs` (`discovery_runtime` ->
    /// `DiscoveryRuntimeConfig`).
    installs: BTreeMap<String, BTreeSet<String>>,
    /// Structs reachable from `Settings` by field traversal.
    reachable: BTreeSet<String>,
}

impl ConfigModel {
    fn declares(&self, ty: &str, field: &str) -> bool {
        self.declares.get(ty).is_some_and(|f| f.contains(field))
    }

    /// Every (owner struct, field) pair an operator can set.
    fn knobs(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for ty in &self.reachable {
            for (f, _) in self.structs.get(ty).into_iter().flatten() {
                out.push((ty.clone(), f.clone()));
            }
        }
        out
    }
}

/// The base name of a type expression: `&mut Option<foo::Bar<T>>` -> `Option`.
fn base_type(ty: &str) -> String {
    let mut t = ty.trim();
    loop {
        let stripped = t
            .strip_prefix('&')
            .or_else(|| t.strip_prefix("mut "))
            .or_else(|| t.strip_prefix("dyn "))
            .or_else(|| t.strip_prefix("impl "));
        match stripped {
            Some(rest) => t = rest.trim_start(),
            None => break,
        }
        // lifetimes: `&'a Foo`
        if let Some(rest) = t.strip_prefix('\'') {
            t = rest.trim_start_matches(|c: char| c.is_alphanumeric() || c == '_').trim_start();
        }
    }
    let head = t.split(['<', ',', ' ', ')', '>']).next().unwrap_or("");
    head.rsplit("::").next().unwrap_or("").trim().to_string()
}

/// Parse `pub struct X { pub f: T, ... }` blocks out of `config.rs`.
///
/// INDENTATION-AGNOSTIC, deliberately. This used to anchor both the `pub
/// struct` opener and the closing `}` at column 0. When `Settings` moved
/// inside the private loader module (the compiler-enforced single load path),
/// its declaration gained four spaces of indent and this parser stopped seeing
/// it — so `Settings` had zero fields, nothing was reachable from it, and all
/// three tests in this file failed with messages that pointed at the config
/// rather than at the parser. The panic at the bottom of
/// `every_settings_field_reaches_a_qualified_consumer` ("parsed only 0 config
/// fields — the struct parser is broken, not the config") exists precisely for
/// that failure mode and it fired correctly.
///
/// A struct block now closes on the first `}` at or left of the indent its own
/// declaration had, so a nested brace inside the body cannot end it early.
fn parse_config_model(src: &str) -> ConfigModel {
    let mut model = ConfigModel::default();
    let mut current: Option<(String, usize)> = None;
    for line in src.lines() {
        let trimmed_start = line.trim_start();
        let indent = line.len() - trimmed_start.len();
        if let Some(rest) = trimmed_start.strip_prefix("pub struct ") {
            let name = rest
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or_default();
            if !name.is_empty() && rest.contains('{') {
                model.structs.entry(name.to_string()).or_default();
                current = Some((name.to_string(), indent));
            }
            continue;
        }
        if trimmed_start.starts_with('}')
            && current.as_ref().is_some_and(|(_, open)| indent <= *open)
        {
            current = None;
            continue;
        }
        let Some(owner) = current.as_ref().map(|(name, _)| name.clone()) else {
            continue;
        };
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("pub ") else { continue };
        let Some((name, ty)) = rest.split_once(':') else { continue };
        let name = name.trim();
        if name.is_empty()
            || !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            continue;
        }
        let ty = base_type(ty.trim_end().trim_end_matches(','));
        model.structs.entry(owner).or_default().push((name.to_string(), ty));
    }

    for (owner, fields) in &model.structs {
        let set: BTreeSet<String> = fields.iter().map(|(f, _)| f.clone()).collect();
        model.declares.insert(owner.clone(), set);
    }
    let struct_names: BTreeSet<String> = model.structs.keys().cloned().collect();
    for fields in model.structs.values() {
        for (f, ty) in fields {
            if struct_names.contains(ty) {
                model.installs.entry(f.clone()).or_default().insert(ty.clone());
            }
        }
    }

    // Reachability from `Settings`.
    let mut stack = vec!["Settings".to_string()];
    while let Some(ty) = stack.pop() {
        if !model.reachable.insert(ty.clone()) {
            continue;
        }
        for (_, fty) in model.structs.get(&ty).into_iter().flatten() {
            if struct_names.contains(fty) {
                stack.push(fty.clone());
            }
        }
    }
    model
}

// ─────────────────────────── source scanning ───────────────────────────

/// Blank out every comment, string literal and char literal, preserving byte
/// offsets and newlines so line numbers stay true.
///
/// Without this the knob catalog's `id: "risk.require_stop_loss"` help text and
/// this file's own doc comments count as readers.
fn strip_noncode(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0usize;
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for k in from..to.min(out.len()) {
            if out[k] != b'\n' {
                out[k] = b' ';
            }
        }
    };
    while i < b.len() {
        match b[i] {
            b'/' if i + 1 < b.len() && b[i + 1] == b'/' => {
                let start = i;
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                blank(&mut out, start, i);
            }
            b'/' if i + 1 < b.len() && b[i + 1] == b'*' => {
                let start = i;
                let mut depth = 1usize;
                i += 2;
                while i < b.len() && depth > 0 {
                    if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                        depth += 1;
                        i += 2;
                    } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                blank(&mut out, start, i);
            }
            b'r' | b'b' if raw_string_hashes(b, i).is_some() => {
                let (body, hashes) = raw_string_hashes(b, i).expect("checked");
                let start = i;
                i = body;
                let mut close = None;
                while i < b.len() {
                    if b[i] == b'"' {
                        let mut k = i + 1;
                        let mut seen = 0;
                        while k < b.len() && b[k] == b'#' && seen < hashes {
                            k += 1;
                            seen += 1;
                        }
                        if seen == hashes {
                            close = Some(k);
                            break;
                        }
                    }
                    i += 1;
                }
                i = close.unwrap_or(b.len());
                blank(&mut out, start, i);
            }
            b'"' | b'`' => {
                let quote = b[i];
                let start = i;
                i += 1;
                while i < b.len() {
                    if b[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if b[i] == quote {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                blank(&mut out, start, i);
            }
            b'\'' => {
                // char literal vs lifetime. `'a'`, `'\n'` are literals;
                // `'a` (followed by non-quote) is a lifetime — leave it alone.
                let is_literal = (i + 2 < b.len() && b[i + 1] == b'\\')
                    || (i + 2 < b.len() && b[i + 2] == b'\'');
                if is_literal {
                    let start = i;
                    i += 1;
                    while i < b.len() {
                        if b[i] == b'\\' {
                            i += 2;
                            continue;
                        }
                        if b[i] == b'\'' {
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                    blank(&mut out, start, i);
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `r"`, `r#"`, `br#"` … -> (offset just past the opening quote, hash count).
fn raw_string_hashes(b: &[u8], mut i: usize) -> Option<(usize, usize)> {
    if b[i] == b'b' {
        i += 1;
        if i >= b.len() || b[i] != b'r' {
            return None;
        }
    }
    if b[i] != b'r' {
        return None;
    }
    i += 1;
    let mut hashes = 0usize;
    while i < b.len() && b[i] == b'#' {
        hashes += 1;
        i += 1;
    }
    if i < b.len() && b[i] == b'"' { Some((i + 1, hashes)) } else { None }
}

fn is_ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn ident_at(b: &[u8], start: usize) -> Option<(String, usize)> {
    if start >= b.len() || !(b[start].is_ascii_alphabetic() || b[start] == b'_') {
        return None;
    }
    let mut end = start;
    while end < b.len() && is_ident_byte(b[end]) {
        end += 1;
    }
    Some((String::from_utf8_lossy(&b[start..end]).into_owned(), end))
}

/// Walk back from `pos` (exclusive) over whitespace and return the identifier
/// ending there, with its start offset.
fn ident_ending_before(b: &[u8], pos: usize) -> Option<(String, usize)> {
    let mut j = pos;
    while j > 0 && (b[j - 1] as char).is_whitespace() {
        j -= 1;
    }
    if j == 0 || !is_ident_byte(b[j - 1]) {
        return None;
    }
    let end = j;
    while j > 0 && is_ident_byte(b[j - 1]) {
        j -= 1;
    }
    if b[j].is_ascii_digit() {
        return None; // `1.5`, `0.0` — a float, not a receiver
    }
    Some((String::from_utf8_lossy(&b[j..end]).into_owned(), j))
}

/// What a local identifier is known to be, within one file.
///
/// Keys are receiver spellings: `cfg`, `self`, and — for accessor functions —
/// `app_runtime()`, so `app_runtime().server_bind` resolves the same way a
/// typed local does.
type Bindings = BTreeMap<String, BTreeSet<String>>;

/// Bind identifiers to config struct types, from five shapes:
///   `fn f(cfg: &DiscoveryRuntimeConfig)` / `let cfg: RiskConfig`
///   `struct S { cfg: PropFirmGateConfig }` (so `self.cfg.field` resolves)
///   `impl RiskConfig { … self.field }`
///   `fn app_runtime() -> AppRuntimeConfig` (so `app_runtime().field` resolves)
///   `let cfg = &settings.models.discovery_runtime;`
fn collect_bindings(code: &str, model: &ConfigModel) -> Bindings {
    let b = code.as_bytes();
    let mut out: Bindings = BTreeMap::new();
    let bind = |name: &str, ty: &str, out: &mut Bindings| {
        out.entry(name.to_string()).or_default().insert(ty.to_string());
    };

    // (1) `ident: [&][mut ][path::]Type`
    let mut i = 0usize;
    while i < b.len() {
        if b[i] != b':' || (i + 1 < b.len() && b[i + 1] == b':') || (i > 0 && b[i - 1] == b':') {
            i += 1;
            continue;
        }
        let Some((bindee, _)) = ident_ending_before(b, i) else {
            i += 1;
            continue;
        };
        // Skip forward over `&`, `mut`, `'a`, whitespace.
        let mut j = i + 1;
        loop {
            while j < b.len() && (b[j] as char).is_whitespace() {
                j += 1;
            }
            if j < b.len() && b[j] == b'&' {
                j += 1;
                continue;
            }
            if j < b.len() && b[j] == b'\'' {
                // a lifetime: `&'a Foo`
                j += 1;
                while j < b.len() && is_ident_byte(b[j]) {
                    j += 1;
                }
                continue;
            }
            if code[j..].starts_with("mut ") {
                j += 4;
                continue;
            }
            break;
        }
        // Read the type path; keep the last segment.
        let mut last: Option<String> = None;
        let mut k = j;
        while let Some((seg, end)) = ident_at(b, k) {
            last = Some(seg);
            k = end;
            if code[k..].starts_with("::") {
                k += 2;
                continue;
            }
            break;
        }
        if let Some(ty) = last {
            // `Foo::default()` is a value, not a type annotation.
            let generic_or_end = k >= b.len() || b[k] != b':';
            if generic_or_end && model.structs.contains_key(&ty) {
                bind(&bindee, &ty, &mut out);
            }
        }
        i += 1;
    }

    // (2) `impl [<..>] [Trait for] Type {`  -> bind `self`
    let mut from = 0usize;
    while let Some(rel) = code[from..].find("impl") {
        let at = from + rel;
        from = at + 4;
        let before_ok = at == 0 || !is_ident_byte(b[at - 1]);
        if !before_ok || at + 4 >= b.len() || is_ident_byte(b[at + 4]) {
            continue;
        }
        let Some(brace) = code[at..].find('{') else { break };
        let header = &code[at + 4..at + brace];
        let header = header.split(" where ").next().unwrap_or(header);
        let target = header.rsplit(" for ").next().unwrap_or(header);
        let ty = base_type(target.trim());
        if model.structs.contains_key(&ty) {
            bind("self", &ty, &mut out);
        }
    }

    // (2b) `fn accessor(..) -> Type {`  -> bind the receiver spelling `accessor()`
    let mut from = 0usize;
    while let Some(rel) = code[from..].find("fn ") {
        let at = from + rel;
        from = at + 3;
        if at > 0 && is_ident_byte(b[at - 1]) {
            continue;
        }
        let Some((name, after)) = ident_at(b, at + 3) else { continue };
        let stop = code[after..]
            .find('{')
            .map(|e| after + e)
            .unwrap_or(code.len().min(after + 400));
        let header = &code[after..stop];
        let Some(arrow) = header.rfind("->") else { continue };
        let ty = base_type(header[arrow + 2..].split(" where ").next().unwrap_or(""));
        if model.structs.contains_key(&ty) {
            bind(&format!("{name}()"), &ty, &mut out);
        }
    }

    // (3) `let [mut ] ident = <expr ending at a sub-config field or Type::…>;`
    let mut from = 0usize;
    while let Some(rel) = code[from..].find("let ") {
        let at = from + rel;
        from = at + 4;
        if at > 0 && is_ident_byte(b[at - 1]) {
            continue;
        }
        let mut j = at + 4;
        while code[j..].starts_with("mut ") {
            j += 4;
        }
        while j < b.len() && (b[j] as char).is_whitespace() {
            j += 1;
        }
        let Some((bindee, mut k)) = ident_at(b, j) else { continue };
        while k < b.len() && (b[k] as char).is_whitespace() {
            k += 1;
        }
        // Skip an explicit type annotation — shape (1) already handled it.
        if k < b.len() && b[k] == b':' {
            continue;
        }
        if k >= b.len() || b[k] != b'=' || (k + 1 < b.len() && b[k + 1] == b'=') {
            continue;
        }
        let rhs_start = k + 1;
        let rhs_end = code[rhs_start..]
            .find(';')
            .map(|e| rhs_start + e)
            .unwrap_or(b.len().min(rhs_start + 400));
        let rhs = &code[rhs_start..rhs_end];
        if let Some(ty) = binding_type_of_expr(rhs, model, &out) {
            bind(&bindee, &ty, &mut out);
        }
    }

    out
}

/// The config type an initialiser expression evaluates to, when that is
/// syntactically evident:
///
/// * `RiskConfig::default()`, `Settings::load()?` — a type path;
/// * `&settings.models.discovery_runtime` — a chain ending at a sub-config field;
/// * `settings.as_ref().map(|s| s.models.discovery_ledger.clone())` — the same,
///   buried in adapters. The real shapes in this workspace are rarely the
///   textbook one, and a resolver that only handles the textbook one
///   under-reports, which is the failure mode being fixed.
///
/// The last qualifying segment wins, and it only qualifies when nothing
/// further is read off it — `let x = cfg.discovery_ledger.cache_dir` binds
/// nothing, because `x` is a `String`, not a config struct.
fn binding_type_of_expr(rhs: &str, model: &ConfigModel, known: &Bindings) -> Option<String> {
    let trimmed = rhs.trim_start().trim_start_matches(['&', ' ']);
    // `Type::…` at the head of the expression.
    if let Some((head, _)) = trimmed.split_once("::") {
        let ty = base_type(head);
        if model.structs.contains_key(&ty) {
            return Some(ty);
        }
    }

    const CHEAP: &[&str] = &[
        "clone",
        "to_owned",
        "cloned",
        "unwrap",
        "unwrap_or_default",
        "expect",
        "as_ref",
        "copied",
        "to_vec",
    ];

    let b = trimmed.as_bytes();
    let mut found: Option<String> = None;
    let mut i = 0usize;
    while i < b.len() {
        let Some((seg, end)) = ident_at(b, i) else {
            i += 1;
            continue;
        };
        i = end;
        // Candidate type for this segment.
        let ty = model
            .installs
            .get(&seg)
            .and_then(|s| s.iter().next().cloned())
            .or_else(|| known.get(&seg).and_then(|s| s.iter().next().cloned()));
        let Some(ty) = ty else { continue };
        // Terminal? A following `.ident` that is NOT a call means the
        // expression reads THROUGH this segment, so the binding is not it.
        let mut k = end;
        while k < b.len() && (b[k] as char).is_whitespace() {
            k += 1;
        }
        let terminal = if k < b.len() && b[k] == b'.' {
            match ident_at(b, k + 1) {
                Some((next, next_end)) => {
                    let is_call = b.get(next_end) == Some(&b'(');
                    is_call && CHEAP.contains(&next.as_str())
                }
                None => true,
            }
        } else {
            true
        };
        if terminal {
            found = Some(ty);
        }
    }
    found
}

/// Walk back from the `.` at `dot` and return the receiver chain, nearest
/// segment first. A call is spelled `name()` so it can be looked up against the
/// accessor-function bindings; an unresolvable receiver ends the chain.
fn receiver_chain(b: &[u8], dot: usize) -> Vec<String> {
    let mut chain = Vec::new();
    let mut cursor = dot;
    loop {
        let mut p = cursor;
        while p > 0 && (b[p - 1] as char).is_whitespace() {
            p -= 1;
        }
        if p == 0 {
            break;
        }
        if b[p - 1] == b')' {
            let Some(open) = match_paren_back(b, p - 1) else { break };
            let Some((name, start)) = ident_ending_before(b, open) else { break };
            chain.push(format!("{name}()"));
            p = start;
        } else if let Some((seg, start)) = ident_ending_before(b, p) {
            chain.push(seg);
            p = start;
        } else {
            break;
        }
        let mut q = p;
        while q > 0 && (b[q - 1] as char).is_whitespace() {
            q -= 1;
        }
        if q > 0 && b[q - 1] == b'.' && !(q > 1 && b[q - 2] == b'.') {
            cursor = q - 1;
        } else {
            break;
        }
    }
    chain
}

/// Is the access whose identifier ends at `k` (whitespace already skipped) the
/// left-hand side of a plain `=`? Compound assignment (`+=`) and comparison
/// (`==`, `>=`) both READ, so only a bare `=` disqualifies.
fn is_assignment_target(b: &[u8], k: usize) -> bool {
    if k >= b.len() || b[k] != b'=' {
        return false;
    }
    if b.get(k + 1) == Some(&b'=') {
        return false; // ==
    }
    if k > 0 && matches!(b[k - 1], b'=' | b'!' | b'<' | b'>' | b'+' | b'-' | b'*' | b'/' | b'%' | b'&' | b'|' | b'^') {
        return false; // !=, >=, +=, …
    }
    true
}

/// Index of the `(` matching the `)` at `close`.
fn match_paren_back(b: &[u8], close: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = close + 1;
    while i > 0 {
        i -= 1;
        match b[i] {
            b')' => depth += 1,
            b'(' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// One resolved read of a config field.
struct Read {
    owner: String,
    field: String,
}

/// Attribute a `<receiver>.<field>` occurrence to the struct(s) that declare it.
///
/// Returns empty when the receiver cannot be resolved — an unqualified match is
/// **not** evidence. That single change is the whole point of the rewrite.
fn attribute(
    field: &str,
    chain: &[String],
    bindings: &Bindings,
    model: &ConfigModel,
) -> Vec<Read> {
    let mut out = Vec::new();
    let Some(r1) = chain.first() else { return out };

    // (a) receiver bound to a config type that declares the field.
    if let Some(types) = bindings.get(r1) {
        for ty in types {
            if model.declares(ty, field) {
                out.push(Read { owner: ty.clone(), field: field.to_string() });
            }
        }
    }

    // (b) receiver is itself the field that installs the declaring struct:
    //     `<anything>.discovery_runtime.prefilter_top_k`
    if let Some(installed) = model.installs.get(r1) {
        for ty in installed {
            if !model.declares(ty, field) {
                continue;
            }
            // Check the next link up when we can resolve it: `models` must be a
            // field of whatever holds it. An unresolvable outer receiver is
            // accepted — the two-segment chain is already strong evidence.
            let consistent = match chain.get(1) {
                None => true,
                Some(r2) => {
                    let mut candidates: BTreeSet<&String> = BTreeSet::new();
                    if let Some(t) = model.installs.get(r2) {
                        candidates.extend(t);
                    }
                    if let Some(t) = bindings.get(r2) {
                        candidates.extend(t);
                    }
                    candidates.is_empty() || candidates.iter().any(|t| model.declares(t, r1))
                }
            };
            if consistent {
                out.push(Read { owner: ty.clone(), field: field.to_string() });
                // Reading `prefilter_top_k` off `discovery_runtime` is also a
                // read of `discovery_runtime` itself.
                for (parent, fields) in &model.declares {
                    if fields.contains(r1) {
                        out.push(Read { owner: parent.clone(), field: r1.clone() });
                    }
                }
            }
        }
    }
    out
}

fn collect_sources(root: &Path, out: &mut Vec<PathBuf>) {
    // `.claude` holds other worktrees of this same repo — scanning them would
    // let a stale checkout satisfy the guard for the tree under test.
    const SKIP: &[&str] = &["target", ".git", ".claude", "node_modules", "vendor", "dist"];
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if !SKIP.contains(&name.as_str()) {
                collect_sources(&path, out);
            }
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("rs") | Some("ts") | Some("tsx")
        ) {
            out.push(path);
        }
    }
}

/// Is this file test/bench/example scaffolding rather than production code?
fn is_scaffolding(path: &Path) -> bool {
    let p = path.to_string_lossy().replace('\\', "/");
    p.contains("/tests/")
        || p.contains("/benches/")
        || p.contains("/examples/")
        || p.ends_with("_tests.rs")
        || p.ends_with("_test.rs")
        || p.contains("/test_")
}

#[derive(Default)]
struct ReadIndex {
    /// (owner, field) read through a qualified access in production code.
    production: BTreeSet<(String, String)>,
    /// Same, but only in tests / benches / examples.
    scaffolding: BTreeSet<(String, String)>,
    /// `field -> file:line` for accesses that LOOK like a read but whose
    /// receiver could not be resolved. Pure diagnostics: this is what the old
    /// scanner was counting as proof.
    unqualified: BTreeMap<String, Vec<String>>,
    /// (owner, field) pairs seen as a qualified assignment TARGET. Diagnostics:
    /// a knob that is written and never read back is the purest form of the
    /// defect — the operator's value is overwritten, then ignored.
    written: BTreeSet<(String, String)>,
    /// Method names invoked as `.name(` in production / in scaffolding. Used to
    /// decide whether an accessor defined in `config.rs` has a live caller.
    called_in_production: BTreeSet<String>,
    called_in_scaffolding: BTreeSet<String>,
}

/// Credit fields read by an accessor **defined inside `config.rs`** — but only
/// when something outside `config.rs` actually calls that accessor.
///
/// `RiskConfig::session_spread_pips()` is the specimen: the three
/// `backtest_spread_pips_*` keys are read only by that method, and
/// `discovery.rs:616` calls it. Skipping `config.rs` wholesale (which both the
/// old scanner and the first cut of this one did) reports all three as orphans.
/// Crediting `config.rs` wholesale would let a private helper nobody calls
/// vouch for a knob. The caller check is the difference.
fn credit_config_rs_accessors(src: &str, model: &ConfigModel, index: &mut ReadIndex) {
    let code = strip_noncode(src);
    let mut current_impl: Option<String> = None;
    let mut current_fn: Option<String> = None;
    for line in code.lines() {
        if let Some(rest) = line.strip_prefix("impl ") {
            let head = rest.split('{').next().unwrap_or("");
            let head = head.split(" where ").next().unwrap_or(head);
            let target = head.rsplit(" for ").next().unwrap_or(head);
            let ty = base_type(target);
            current_impl = model.structs.contains_key(&ty).then_some(ty);
            current_fn = None;
            continue;
        }
        if line.starts_with('}') {
            current_impl = None;
            current_fn = None;
            continue;
        }
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("pub fn ").or_else(|| trimmed.strip_prefix("fn ")) {
            current_fn = ident_at(rest.as_bytes(), 0).map(|(n, _)| n);
        }
        let (Some(owner), Some(func)) = (current_impl.as_ref(), current_fn.as_ref()) else {
            continue;
        };
        let mut from = 0usize;
        while let Some(rel) = line[from..].find("self.") {
            let at = from + rel + 5;
            from = at;
            let Some((field, _)) = ident_at(line.as_bytes(), at) else { continue };
            if !model.declares(owner, &field) {
                continue;
            }
            let key = (owner.clone(), field);
            if index.called_in_production.contains(func) {
                index.production.insert(key);
            } else if index.called_in_scaffolding.contains(func) {
                index.scaffolding.insert(key);
            }
        }
    }
}

fn scan_workspace(model: &ConfigModel) -> ReadIndex {
    let root = workspace_root();
    let skip = std::fs::canonicalize(config_rs()).ok();
    let mut files = Vec::new();
    collect_sources(&root, &mut files);

    let all_field_names: BTreeSet<&String> =
        model.reachable.iter().flat_map(|t| model.declares.get(t)).flatten().collect();

    let mut index = ReadIndex::default();
    for file in files {
        if let (Some(skip), Ok(canon)) = (skip.as_ref(), std::fs::canonicalize(&file)) {
            if &canon == skip {
                continue;
            }
        }
        let Ok(raw) = std::fs::read_to_string(&file) else {
            continue;
        };
        let code = strip_noncode(&raw);
        let bindings = collect_bindings(&code, model);
        let b = code.as_bytes();
        let scaffolding = is_scaffolding(&file);
        let rel = file
            .strip_prefix(&root)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");

        // (d) `let SomeConfig { a, b, .. } = …` destructures.
        for ty in model.structs.keys() {
            let needle = format!("let {ty}");
            let mut from = 0usize;
            while let Some(rel_at) = code[from..].find(&needle) {
                let at = from + rel_at;
                from = at + needle.len();
                let after = code[at + needle.len()..].trim_start();
                if !after.starts_with('{') {
                    continue;
                }
                let open = at + needle.len() + code[at + needle.len()..].find('{').unwrap_or(0);
                let close = code[open..].find('}').map(|e| open + e).unwrap_or(code.len());
                let body = &code[open + 1..close];
                for tok in body.split(|c: char| !is_ident_byte(c as u8)) {
                    if model.declares(ty, tok) {
                        let key = (ty.clone(), tok.to_string());
                        if scaffolding {
                            index.scaffolding.insert(key);
                        } else {
                            index.production.insert(key);
                        }
                    }
                }
            }
        }

        let mut i = 0usize;
        while i < b.len() {
            if b[i] != b'.' {
                i += 1;
                continue;
            }
            // `..`, `.0`, `1.5` are not field reads.
            let mut j = i + 1;
            while j < b.len() && (b[j] as char).is_whitespace() {
                j += 1;
            }
            let Some((field, end)) = ident_at(b, j) else {
                i += 1;
                continue;
            };
            // A call is a method, not a field. Record it: an accessor defined
            // in config.rs only counts as a recipient if someone calls it.
            let mut k = end;
            while k < b.len() && (b[k] as char).is_whitespace() {
                k += 1;
            }
            if k < b.len() && b[k] == b'(' {
                if scaffolding {
                    index.called_in_scaffolding.insert(field.clone());
                } else {
                    index.called_in_production.insert(field.clone());
                }
                i = end;
                continue;
            }
            if !all_field_names.contains(&field) {
                i = end;
                continue;
            }
            let chain = receiver_chain(b, i);
            let hits = attribute(&field, &chain, &bindings, model);
            // An assignment TARGET is not a recipient. `settings.system.n_jobs
            // = hints.n_jobs` is the auto-tuner writing the knob; if nothing
            // ever reads it back, the operator's value is overwritten and then
            // ignored. Both of the documented false passes were exactly this.
            if is_assignment_target(b, k) {
                for r in hits {
                    index.written.insert((r.owner, r.field));
                }
                i = end;
                continue;
            }
            if hits.is_empty() {
                let line = code[..i].bytes().filter(|c| *c == b'\n').count() + 1;
                let slot = index.unqualified.entry(field.clone()).or_default();
                if slot.len() < 3 {
                    slot.push(format!("{rel}:{line}"));
                }
            } else {
                for r in hits {
                    let key = (r.owner, r.field);
                    if scaffolding {
                        index.scaffolding.insert(key);
                    } else {
                        index.production.insert(key);
                    }
                }
            }
            i = end;
        }
    }

    // Last, because it depends on the call sets the main pass collected.
    let config_src = std::fs::read_to_string(config_rs()).expect("config.rs must be readable");
    credit_config_rs_accessors(&config_src, model, &mut index);
    index
}

fn model() -> ConfigModel {
    let src = std::fs::read_to_string(config_rs()).expect("config.rs must be readable");
    parse_config_model(&src)
}

// ─────────────────────────────── the tests ───────────────────────────────

#[test]
fn every_settings_field_reaches_a_qualified_consumer() {
    let model = model();
    let knobs = model.knobs();
    assert!(
        knobs.len() > 250,
        "parsed only {} config fields — the struct parser is broken, not the config",
        knobs.len()
    );
    assert!(
        model.reachable.len() >= 20,
        "only {} config structs are reachable from `Settings` — the graph walk is broken",
        model.reachable.len()
    );
    // Nothing in config.rs that CARRIES OPERATOR KNOBS may sit outside the
    // Settings graph unnoticed.
    //
    // The filter on `pub` fields is load-bearing, not a loophole. This check
    // exists so an operator-settable struct cannot be added and quietly go
    // unchecked. A struct with no `pub` fields carries nothing an operator can
    // set — `ConfigProvenance` (which records WHICH file a Settings came from,
    // fields `source`/`path`, both private) is the case that forced this: it is
    // internal machinery of the single load path, deliberately unreachable from
    // `Settings`, and flagging it says nothing about knob coverage. If a struct
    // ever gains a `pub` field it re-enters this assertion immediately, which is
    // the property worth keeping.
    let orphan_structs: Vec<&String> = model
        .structs
        .iter()
        .filter(|(name, fields)| !model.reachable.contains(*name) && !fields.is_empty())
        .map(|(name, _)| name)
        .collect();
    assert!(
        orphan_structs.is_empty(),
        "config.rs declares struct(s) unreachable from `Settings`, so their fields are \
         unchecked by this test: {orphan_structs:?}"
    );

    let index = scan_workspace(&model);
    let ledger = ledger_keys();

    let mut no_reader: Vec<String> = Vec::new();
    let mut scaffolding_only: Vec<String> = Vec::new();
    for (owner, field) in &knobs {
        let key = (owner.clone(), field.clone());
        let label = format!("{owner}::{field}");
        if ledger.contains(label.as_str()) {
            continue;
        }
        if index.production.contains(&key) {
            continue;
        }
        let mut near = index
            .unqualified
            .get(field)
            .map(|v| format!("  looks-like-a-read (receiver unresolved): {}", v.join(", ")))
            .unwrap_or_default();
        if index.written.contains(&key) {
            near.push_str(
                "\n  NOTE: qualified code ASSIGNS this field and nothing reads it back — the \
                 operator's value is overwritten, then ignored",
            );
        }
        if index.scaffolding.contains(&key) {
            scaffolding_only.push(format!("  - {label}\n{near}"));
        } else {
            no_reader.push(format!("  - {label}\n{near}"));
        }
    }

    assert!(
        no_reader.is_empty() && scaffolding_only.is_empty(),
        "{} config field(s) have NO QUALIFIED PRODUCTION READER.\n\n\
         Nothing reads them through an access whose receiver resolves to the struct that \
         declares them. A same-named field on some other type is NOT a reader — that is the \
         hole this guard used to have.\n\n\
         No reader at all ({}):\n{}\n\n\
         Read only in tests/benches/examples ({}):\n{}\n\n\
         Pick one:\n\
         • delete the field (and its config.yaml key + any UI control), or\n\
         • wire it to the code that should honour it, or\n\
         • if a real mechanism exists but connecting it would change trading behaviour, add \
         it to NO_QUALIFIED_READER in this file WITH the mechanism named and the owner of \
         the decision.\n\
         Do not add it to the ledger to silence this — a value the operator can set that \
         changes nothing is the defect this test exists to stop.",
        no_reader.len() + scaffolding_only.len(),
        no_reader.len(),
        if no_reader.is_empty() { "  (none)".into() } else { no_reader.join("\n") },
        scaffolding_only.len(),
        if scaffolding_only.is_empty() { "  (none)".into() } else { scaffolding_only.join("\n") },
    );
}

#[test]
fn the_no_qualified_reader_ledger_is_exact() {
    // No cap. A cap is pressure to find any matching token rather than a real
    // recipient. What is enforced instead: every entry must still name a real
    // field, must still have no qualified production reader, and must say why.
    let model = model();
    let index = scan_workspace(&model);

    let mut problems: Vec<String> = Vec::new();
    for (key, kind, why) in NO_QUALIFIED_READER {
        let Some((owner, field)) = key.split_once("::") else {
            problems.push(format!("`{key}` must be spelled `OwnerStruct::field`"));
            continue;
        };
        if !model.declares(owner, field) {
            problems.push(format!(
                "`{key}` is in the ledger but `{owner}` no longer declares `{field}` — drop the \
                 stale entry"
            ));
            continue;
        }
        if !model.reachable.contains(owner) {
            problems.push(format!("`{key}`: `{owner}` is not reachable from `Settings`"));
        }
        if why.trim().len() < 40 || !why.contains("OWNER:") {
            problems.push(format!(
                "`{key}` must name the mechanism it fails to reach AND the owner of the decision \
                 (a line containing `OWNER:`)"
            ));
        }
        let pair = (owner.to_string(), field.to_string());
        if index.production.contains(&pair) {
            problems.push(format!(
                "`{key}` is listed as {kind:?} but now HAS a qualified production reader — delete \
                 the ledger entry, the knob works"
            ));
        }
        if *kind == Inert::TestOnly && !index.scaffolding.contains(&pair) {
            problems.push(format!(
                "`{key}` is listed as TestOnly but no test/bench/example reads it either — it is \
                 Unwired or dead, not test-only"
            ));
        }
        if *kind == Inert::WrittenNeverRead && !index.written.contains(&pair) {
            problems.push(format!(
                "`{key}` is listed as WrittenNeverRead but nothing assigns it through a \
                 qualified path — re-classify it as Unwired"
            ));
        }
        // `Unwired` was the one classification with no positive obligation, so
        // it was a silent catch-all: a knob that production code ASSIGNS and
        // never reads back could sit here forever, hiding the case this
        // module's own docs call the purest form of the defect — the value is
        // not merely ignored, it is destroyed first. Symmetric with the two
        // rules above, so the ledger is now closed in both directions.
        if *kind == Inert::Unwired && index.written.contains(&pair) {
            problems.push(format!(
                "`{key}` is listed as Unwired but qualified production code ASSIGNS it — \
                 re-classify it as WrittenNeverRead, which is a stronger statement: the \
                 operator's value is overwritten before anything could read it"
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "the no-qualified-reader ledger is out of date:\n{}",
        problems.join("\n")
    );
}

// ─────────────────────────── the UI side ───────────────────────────

fn settings_endpoint_src() -> String {
    let path = workspace_root()
        .join("crates")
        .join("neoethos-app")
        .join("src")
        .join("server")
        .join("settings.rs");
    std::fs::read_to_string(path).expect("server/settings.rs must be readable")
}

fn advanced_screen_src() -> String {
    let path = workspace_root()
        .join("desktop")
        .join("src")
        .join("screens")
        .join("Advanced.tsx");
    std::fs::read_to_string(path).expect("Advanced.tsx must be readable")
}

/// Field names of `struct SettingsUpdateDto` — what `POST /settings` accepts.
fn update_dto_fields(src: &str) -> BTreeSet<String> {
    let start = src
        .find("pub struct SettingsUpdateDto {")
        .expect("SettingsUpdateDto must exist");
    let body = &src[start..];
    let end = body.find("\n}").expect("SettingsUpdateDto must be closed");
    body[..end]
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("pub "))
        .filter_map(|l| l.split_once(':'))
        .map(|(n, _)| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .collect()
}

fn camel_to_snake(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[test]
fn every_ui_control_has_a_field_behind_it() {
    let ui = advanced_screen_src();
    let dto = update_dto_fields(&settings_endpoint_src());

    // `{ key: "searchPopulation", label: ... }` entries in the GROUPS catalog.
    let mut orphan_controls = Vec::new();
    for (idx, _) in ui.match_indices("key: \"") {
        let rest = &ui[idx + 6..];
        let Some(end) = rest.find('"') else { continue };
        let key = &rest[..end];
        let snake = camel_to_snake(key);
        if !dto.contains(&snake) {
            orphan_controls.push(key.to_string());
        }
    }

    assert!(
        orphan_controls.is_empty(),
        "Advanced.tsx offers control(s) that POST /settings does not accept — \
         the operator edits them, presses Save, gets \"✓ Settings saved.\", and \
         the value is dropped by serde: {orphan_controls:?}"
    );
}

#[test]
fn every_accepted_field_is_actually_applied() {
    let src = settings_endpoint_src();
    let dto = update_dto_fields(&src);
    let handler_start = src
        .find("pub async fn update_settings(")
        .expect("update_settings handler must exist");
    let handler = &src[handler_start..];

    let unapplied: Vec<&String> = dto
        .iter()
        .filter(|f| !handler.contains(&format!("payload.{f}")))
        .collect();

    assert!(
        unapplied.is_empty(),
        "POST /settings declares field(s) it never applies to Settings: \
         {unapplied:?}. Accepting a value and dropping it is worse than \
         rejecting it — the caller is told it was saved."
    );
}

/// Fields whose YAML value is a free-form map — their nested keys are data
/// (model names, timeframe names), not config field names, so the seed scan
/// must not treat them as such.
const MAP_VALUED_FIELDS: &[&str] = &[
    "hpo_trials_by_model",
    "max_epochs_by_model",
    "model_param_overrides",
    "prop_search_max_rows_by_tf",
];

#[test]
fn the_bundled_seed_config_declares_nothing_settings_would_drop() {
    // `desktop/src-tauri/resources/config.yaml` is what the installer copies
    // into the user data dir on first run. `Settings` does NOT derive
    // `deny_unknown_fields`, so a key here that no longer exists on the struct
    // is read, ignored, and never mentioned — a shipped default with no final
    // recipient, seeded onto every fresh install.
    let seed_path = workspace_root()
        .join("desktop")
        .join("src-tauri")
        .join("resources")
        .join("config.yaml");
    let seed = std::fs::read_to_string(&seed_path).expect("bundled seed config must be readable");

    let model = model();
    let known: BTreeSet<&String> =
        model.structs.values().flatten().map(|(f, _)| f).collect();

    let mut unknown = Vec::new();
    let mut skip_indent: Option<usize> = None;
    for (lineno, line) in seed.lines().enumerate() {
        let indent = line.len() - line.trim_start().len();
        if let Some(level) = skip_indent {
            if line.trim().is_empty() || indent > level {
                continue;
            }
            skip_indent = None;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') || trimmed.starts_with("- ") {
            continue;
        }
        let Some((key, _)) = trimmed.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty()
            || !key
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            continue;
        }
        if MAP_VALUED_FIELDS.contains(&key) {
            skip_indent = Some(indent);
            continue;
        }
        if !known.contains(&key.to_string()) {
            unknown.push(format!("  line {}: {key}", lineno + 1));
        }
    }

    assert!(
        unknown.is_empty(),
        "the bundled seed config.yaml declares {} key(s) that no longer exist \
         on Settings:\n{}\n\nserde ignores unknown keys, so every fresh install \
         would ship these and silently drop them. Delete them from \
         desktop/src-tauri/resources/config.yaml.",
        unknown.len(),
        unknown.join("\n")
    );
}

#[test]
fn no_ui_control_is_backed_by_a_no_recipient_field() {
    // A control may only exist for a knob that reaches a live consumer. This
    // is the check `ui_locale` failed: a Language picker, fully plumbed
    // through the DTO, backed by a field with no reader in the workspace.
    let ui = advanced_screen_src();
    let ledger = ledger_field_names();

    let mut offenders = Vec::new();
    for (idx, _) in ui.match_indices("key: \"") {
        let rest = &ui[idx + 6..];
        let Some(end) = rest.find('"') else { continue };
        let snake = camel_to_snake(&rest[..end]);
        if ledger.contains(snake.as_str()) {
            offenders.push(snake);
        }
    }

    assert!(
        offenders.is_empty(),
        "UI control(s) are backed by fields with no live consumer: {offenders:?}. \
         Wire the field or remove the control — do not ship a knob that moves \
         nothing."
    );
}

// ─────────────────────── the scanner's own tests ───────────────────────
//
// A guard that cannot be shown to catch the thing it guards is the defect it
// guards against. These pin the exact false passes the old scanner had.

#[cfg(test)]
mod scanner {
    use super::*;

    fn tiny_model() -> ConfigModel {
        parse_config_model(
            "pub struct Settings {\n    pub models: ModelsConfig,\n    pub risk: RiskConfig,\n}\n\
             pub struct ModelsConfig {\n    pub discovery_runtime: DiscoveryRuntimeConfig,\n    \
             pub inference_batch_size: usize,\n}\n\
             pub struct DiscoveryRuntimeConfig {\n    pub prefilter_top_k: usize,\n}\n\
             pub struct RiskConfig {\n    pub trailing_min_lock_pips: f64,\n}\n",
        )
    }

    fn reads(code: &str) -> BTreeSet<(String, String)> {
        let model = tiny_model();
        let stripped = strip_noncode(code);
        let bindings = collect_bindings(&stripped, &model);
        let b = stripped.as_bytes();
        let mut out = BTreeSet::new();
        let mut i = 0usize;
        while i < b.len() {
            if b[i] != b'.' {
                i += 1;
                continue;
            }
            let Some((field, end)) = ident_at(b, i + 1) else {
                i += 1;
                continue;
            };
            let chain = receiver_chain(b, i);
            for r in attribute(&field, &chain, &bindings, &model) {
                out.insert((r.owner, r.field));
            }
            i = end;
        }
        out
    }

    #[test]
    fn a_same_named_field_on_another_type_is_not_a_reader() {
        // The exact false pass: `AutoTuneHints::inference_batch_size` satisfied
        // `ModelsConfig::inference_batch_size` for a knob nothing read.
        let out = reads("fn f(h: &AutoTuneHints) { let _ = h.inference_batch_size; }");
        assert!(out.is_empty(), "unqualified read must not attribute: {out:?}");

        let out = reads("fn f(s: &BacktestSettings) { let _ = s.trailing_min_lock_pips; }");
        assert!(out.is_empty(), "unqualified read must not attribute: {out:?}");
    }

    #[test]
    fn a_dotted_chain_rooted_at_a_section_is_a_reader() {
        let out = reads("fn f(x: &X) { let _ = x.models.discovery_runtime.prefilter_top_k; }");
        assert!(
            out.contains(&(
                "DiscoveryRuntimeConfig".to_string(),
                "prefilter_top_k".to_string()
            )),
            "{out:?}"
        );
        // …and the chain also credits the container fields it went through.
        assert!(out.contains(&("ModelsConfig".to_string(), "discovery_runtime".to_string())));
    }

    #[test]
    fn a_typed_receiver_is_a_reader() {
        let out = reads("fn f(cfg: &DiscoveryRuntimeConfig) { let _ = cfg.prefilter_top_k; }");
        assert!(out.contains(&(
            "DiscoveryRuntimeConfig".to_string(),
            "prefilter_top_k".to_string()
        )));
    }

    #[test]
    fn a_let_bound_subconfig_is_a_reader() {
        // The dominant real shape: `let cfg = &settings.models.discovery_runtime;`
        let out = reads(
            "fn f(settings: &Settings) { let cfg = &settings.models.discovery_runtime; \
             let _ = cfg.prefilter_top_k; }",
        );
        assert!(out.contains(&(
            "DiscoveryRuntimeConfig".to_string(),
            "prefilter_top_k".to_string()
        )));
    }

    #[test]
    fn strings_and_comments_are_not_readers() {
        // The knob catalog ships `id: "risk.require_stop_loss"` as help text.
        let out = reads("fn f() { let id = \"risk.trailing_min_lock_pips\"; }");
        assert!(out.is_empty(), "a string literal is not a reader: {out:?}");
        let out = reads("// see x.risk.trailing_min_lock_pips\nfn f() {}");
        assert!(out.is_empty(), "a comment is not a reader: {out:?}");
        let out = reads("/// doc: x.risk.trailing_min_lock_pips\nfn f() {}");
        assert!(out.is_empty(), "a doc comment is not a reader: {out:?}");
    }

    #[test]
    fn a_method_call_is_not_a_field_read() {
        let model = tiny_model();
        assert!(model.declares("DiscoveryRuntimeConfig", "prefilter_top_k"));
        // handled in scan_workspace by the `(` lookahead; assert the primitive
        // it relies on rather than duplicating the walker.
        let code = strip_noncode("x.prefilter_top_k()");
        let b = code.as_bytes();
        let (_, end) = ident_at(b, 2).expect("ident");
        assert_eq!(b[end], b'(');
    }

    #[test]
    fn an_assignment_target_is_not_a_reader() {
        // `self.settings.system.n_jobs = hints.n_jobs` — the auto-tuner writing
        // the knob. The old guard counted this as proof the knob was honoured.
        let code = strip_noncode("s.models.inference_batch_size = hints.inference_batch_size;");
        let b = code.as_bytes();
        let dot = code.find(".inference_batch_size").expect("access");
        let (_, end) = ident_at(b, dot + 1).expect("ident");
        let mut k = end;
        while (b[k] as char).is_whitespace() {
            k += 1;
        }
        assert!(is_assignment_target(b, k), "`= x` must read as a write");

        // …but a comparison and a compound assignment both READ.
        for src in ["if s.models.inference_batch_size == 4 {}", "s.models.inference_batch_size += 1;"] {
            let code = strip_noncode(src);
            let b = code.as_bytes();
            let dot = code.find(".inference_batch_size").expect("access");
            let (_, end) = ident_at(b, dot + 1).expect("ident");
            let mut k = end;
            while (b[k] as char).is_whitespace() {
                k += 1;
            }
            assert!(!is_assignment_target(b, k), "`{src}` reads the field");
        }
    }

    #[test]
    fn a_config_rs_accessor_counts_only_when_something_calls_it() {
        let model = tiny_model();
        let src = "impl RiskConfig {\n    pub fn spread_curve(&self) -> f64 {\n        \
                   self.trailing_min_lock_pips\n    }\n}\n";

        // Nobody calls it: no credit. An accessor with no caller is not a
        // recipient, it is more unwired code.
        let mut idx = ReadIndex::default();
        credit_config_rs_accessors(src, &model, &mut idx);
        assert!(idx.production.is_empty(), "{:?}", idx.production);

        // Production calls it: credit. This is the `session_spread_pips()`
        // shape, which reports three false orphans without this rule.
        let mut idx = ReadIndex::default();
        idx.called_in_production.insert("spread_curve".to_string());
        credit_config_rs_accessors(src, &model, &mut idx);
        assert!(idx.production.contains(&(
            "RiskConfig".to_string(),
            "trailing_min_lock_pips".to_string()
        )));

        // Only a test calls it: scaffolding, not production.
        let mut idx = ReadIndex::default();
        idx.called_in_scaffolding.insert("spread_curve".to_string());
        credit_config_rs_accessors(src, &model, &mut idx);
        assert!(idx.production.is_empty());
        assert_eq!(idx.scaffolding.len(), 1);
    }

    #[test]
    fn an_accessor_function_receiver_resolves() {
        // `app_runtime().server_bind` — ten AppRuntimeConfig knobs are read
        // exactly this way and nowhere else.
        let out = reads(
            "fn cfg() -> DiscoveryRuntimeConfig { Default::default() }\n\
             fn f() { let _ = cfg().prefilter_top_k; }",
        );
        assert!(
            out.contains(&(
                "DiscoveryRuntimeConfig".to_string(),
                "prefilter_top_k".to_string()
            )),
            "{out:?}"
        );
    }

    #[test]
    fn float_literals_are_not_receivers() {
        let code = strip_noncode("let x = 1.prefilter_top_k;");
        let b = code.as_bytes();
        let dot = code.find('.').expect("dot");
        assert!(
            ident_ending_before(b, dot).is_none(),
            "a numeric receiver must not resolve"
        );
    }
}
