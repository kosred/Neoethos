//! `risk.max_portfolio_risk` is one knob with two correct answers, and which
//! one is correct is not a matter of taste.
//!
//! Under the RISKY ladder the point is to multiply a small balance, there is no
//! challenge to fail, and `risky_max_risk_per_trade` is already 0.30 — so the
//! binding constraint is the operator's tolerance for ruin and the cap is 0.34.
//!
//! Under a PROP FIRM the ceiling is arithmetic. FX pairs correlate, so the
//! honest worst case is every open position stopping out together, and then the
//! day's loss IS the total open risk. A concurrent-risk budget above the daily
//! stop can breach the daily limit in a single move — the way a challenge is
//! failed outright rather than slowly. So the cap is seeded AT the daily stop,
//! and the two limits state the same fact instead of contradicting each other.
//!
//! Carrying 0.34 into a prop-firm account is therefore not "slightly loose". It
//! is eight and a half times the number that ends the account, and it was sitting
//! in the operator's live store on 2026-08-10 because a human (me) copied a
//! value from the risky profile rather than asking which ladder was running.
//!
//! The other half of this file is the sentinel. Until 2026-08-10 the shipped
//! default was `0.0`, and `live_trading.rs` read that as "no cap" — so on a knob
//! named `max_`, the loosest possible setting and the unset state were spelled
//! the same way, on every install. A money ceiling must not be removable by
//! leaving a field alone.

use std::io::Write;

use neoethos_core::Settings;

/// Writes `body` to a temp file and loads it through the real, sealed path.
fn load(body: &str) -> Settings {
    let dir = std::env::temp_dir().join(format!(
        "neoethos-portfolio-cap-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("config.yaml");
    let mut f = std::fs::File::create(&path).expect("create temp config");
    f.write_all(body.as_bytes()).expect("write temp config");
    f.sync_all().expect("flush temp config");
    Settings::from_yaml(&path).expect("the temp config must load")
}

fn cfg(mode: &str, preset: &str, extra_risk: &str) -> String {
    format!("system:\n  trading_mode: {mode}\nrisk:\n  preset: {preset}\n{extra_risk}")
}

/// The number the operator stated on 2026-08-10: "34% cap for risky yes, for
/// prop firm no."
#[test]
fn the_risky_ladder_gets_the_risky_cap() {
    for mode in ["risky", "growth"] {
        let s = load(&cfg(mode, "none", ""));
        assert!(
            (s.risk.max_portfolio_risk - 0.34).abs() < 1e-9,
            "trading_mode `{mode}` must get the risky concurrent-risk cap 0.34, got {}",
            s.risk.max_portfolio_risk
        );
    }
    // The preset must not change this: under the risky ladder there is no
    // challenge to fail, so a firm's rulebook is not the binding constraint.
    let s = load(&cfg("risky", "ftmo", ""));
    assert!(
        (s.risk.max_portfolio_risk - 0.34).abs() < 1e-9,
        "the risky ladder's cap comes from the MODE, not the preset, got {}",
        s.risk.max_portfolio_risk
    );
}

/// The seed equals the daily stop, per preset, so one correlated move cannot
/// spend more than the day's whole budget.
#[test]
fn a_prop_firm_gets_its_own_daily_stop_as_the_cap() {
    for (preset, expected) in [("ftmo", 0.040), ("the5ers", 0.032), ("none", 0.08)] {
        let s = load(&cfg("prop_firm", preset, ""));
        assert!(
            (s.risk.max_portfolio_risk - expected).abs() < 1e-6,
            "preset `{preset}` under prop_firm must cap concurrent risk at its daily stop \
             {expected}, got {}",
            s.risk.max_portfolio_risk
        );
        assert!(
            (s.risk.max_portfolio_risk - s.risk.daily_drawdown_limit).abs() < 1e-6,
            "preset `{preset}`: the concurrent-risk cap ({}) and the daily drawdown limit ({}) \
             must state the same number — if the cap is the larger, one correlated move breaches \
             the daily limit and the challenge is over",
            s.risk.max_portfolio_risk,
            s.risk.daily_drawdown_limit
        );
    }
}

/// The exact defect found in the operator's live store on 2026-08-10.
#[test]
fn the_risky_cap_never_leaks_into_a_prop_firm_account() {
    let s = load(&cfg("prop_firm", "ftmo", ""));
    assert!(
        s.risk.max_portfolio_risk < 0.34,
        "0.34 is the RISKY cap. Under prop_firm it is 8.5x FTMO's daily stop — it does not \
         loosen the account, it ends it."
    );
}

/// A zero is the field's empty value, not a decision to remove the ceiling.
#[test]
fn a_zero_cap_is_re_seeded_rather_than_read_as_unlimited() {
    for written in ["0.0", "0", "-1.0"] {
        let s = load(&cfg(
            "prop_firm",
            "ftmo",
            &format!("  max_portfolio_risk: {written}\n"),
        ));
        assert!(
            (s.risk.max_portfolio_risk - 0.040).abs() < 1e-6,
            "`max_portfolio_risk: {written}` must be re-seeded from the preset, not read as an \
             unlimited concurrent-risk budget; got {}",
            s.risk.max_portfolio_risk
        );
    }
}

/// …and the escape hatch has to actually work, or the rule above is a lock
/// wearing the word "seed".
#[test]
fn one_point_zero_really_does_mean_no_ceiling() {
    let s = load(&cfg("prop_firm", "ftmo", "  max_portfolio_risk: 1.0\n"));
    assert!(
        (s.risk.max_portfolio_risk - 1.0).abs() < 1e-9,
        "an operator who writes 1.0 has said `the whole account may be at risk at once` in a \
         way no accident produces; his number must stand, got {}",
        s.risk.max_portfolio_risk
    );
}

/// A tighter number the operator typed is his, not the preset's.
#[test]
fn a_tighter_operator_value_survives_the_seed() {
    let s = load(&cfg("prop_firm", "ftmo", "  max_portfolio_risk: 0.02\n"));
    assert!(
        (s.risk.max_portfolio_risk - 0.02).abs() < 1e-9,
        "a preset is a seed, not a lock: 0.02 is tighter than FTMO's 0.040 and must stand, \
         got {}",
        s.risk.max_portfolio_risk
    );
}

/// The compiled defaults must not ship the old sentinel back in.
#[test]
fn the_shipped_default_carries_a_real_ceiling() {
    let d = Settings::default();
    assert!(
        d.risk.max_portfolio_risk > 0.0,
        "Settings::default() shipped max_portfolio_risk = {} — a knob named max_ at zero is \
         read as UNLIMITED concurrent risk by live_trading.rs, on every install",
        d.risk.max_portfolio_risk
    );
    assert!(
        (d.risk.max_portfolio_risk - d.risk.daily_drawdown_limit).abs() < 1e-6,
        "the default cap ({}) must equal the default daily drawdown limit ({})",
        d.risk.max_portfolio_risk,
        d.risk.daily_drawdown_limit
    );
}

/// An ABSENT key must mean the compiled default, on every field.
///
/// These four `Option<f64>` risk bands carried a field-level
/// `#[serde(default)]` until 2026-08-10, which overrode the container's and
/// made absence mean `None` instead of the struct default. The overrides-only
/// store is built on the opposite promise: a key is deleted from the file
/// precisely BECAUSE it equals the default, and must come back as that default.
///
/// The failure this catches is silent and it is money: with the field absent,
/// `prop_firm_max_risk_per_trade` fell back to `risk_per_trade` — 3% where the
/// operator had ruled 1%, on every entry.
#[test]
fn an_absent_risk_band_takes_the_compiled_default_not_none() {
    let d = Settings::default();
    let loaded = load("system:\n  trading_mode: prop_firm\nrisk:\n  preset: ftmo\n");

    assert_eq!(
        loaded.risk.prop_firm_max_risk_per_trade, d.risk.prop_firm_max_risk_per_trade,
        "an absent prop_firm_max_risk_per_trade must be the default, not None"
    );
    assert_eq!(
        loaded.risk.risky_max_risk_per_trade, d.risk.risky_max_risk_per_trade,
        "an absent risky_max_risk_per_trade must be the default, not None"
    );
    assert_eq!(
        loaded.risk.risky_min_risk_per_trade, d.risk.risky_min_risk_per_trade,
        "an absent risky_min_risk_per_trade must be the default, not None"
    );
    assert_eq!(
        loaded.risk.prop_firm_min_risk_per_trade, d.risk.prop_firm_min_risk_per_trade,
        "an absent prop_firm_min_risk_per_trade must be the default, not None"
    );

    // And the values themselves are the operator's 2026-08-10 ruling.
    assert_eq!(
        loaded.risk.prop_firm_max_risk_per_trade,
        Some(0.01),
        "1% for prop firm"
    );
    assert_eq!(
        loaded.risk.risky_max_risk_per_trade,
        Some(0.30),
        "30% for risky mode"
    );
}
