//! Formula-level uniqueness guard for the production classic-TA schema.
//!
//! These fixtures are deliberately adversarial rather than market evidence:
//! they exist only to distinguish a structural alias/ignored parameter from a
//! coincidence in one historical EURUSD frame.  No result from this file is
//! admissible as financial evidence.

use std::collections::{BTreeMap, BTreeSet};

use neoethos_data::Ohlcv;
use neoethos_data::core::all_indicators::ALL_INDICATORS;
use neoethos_data::core::hpc_ta::{
    ALT_PERIODS, IndicatorComputePolicy, compute_classic_ta_columns_with_policy,
};
use neoethos_data::core::indicator_ledger::has_finite_variation;

const SHORT_FIXTURE_BARS: usize = 2_048;
const M5_SLOTS_PER_DAY: usize = 24 * 12;
const LONG_HISTORY_DAYS: usize = 60;
// Market Structure Confluence's default shared publication warmup is 99 bars
// (basis_length = 100). Its structure state advances before that shared warmup,
// so an ordinary random walk can consume the only `prev_break_dir == 0` event
// before the named event columns become finite. Delay one first break in each
// direction beyond the warmup so the formula-level gate observes the semantic
// difference: `change = first break OR CHoCH`, while `CHoCH` requires reversal.
const MARKET_STRUCTURE_FIRST_PUBLISHED_BREAK: usize = 128;

fn adversarial_fixture(seed: u64, mode: usize) -> Ohlcv {
    // Half-causal estimator's `data_period` counts prior sessions for each
    // time-of-day slot. A 2,048-bar M5 fixture spans only about seven days, so
    // its valid 13- and 25-session sweep points can coincide simply because
    // neither store has filled yet. Keep one full 60-day fixture to exercise
    // every swept data-period (up to 50 sessions); the other shapes stay small
    // so the whole production-vocabulary gate remains focused.
    let bars = if mode == 0 {
        M5_SLOTS_PER_DAY * LONG_HISTORY_DAYS
    } else {
        SHORT_FIXTURE_BARS
    };
    let mut state = seed;
    let mut random = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut timestamp = Vec::with_capacity(bars);
    let mut open = Vec::with_capacity(bars);
    let mut high = Vec::with_capacity(bars);
    let mut low = Vec::with_capacity(bars);
    let mut close = Vec::with_capacity(bars);
    let mut volume = Vec::with_capacity(bars);
    let mut price: f64 = 1.1;
    for i in 0..bars {
        let phase = i as f64;
        let random_step = (random() - 0.5) * 0.004;
        let shaped_step = if mode < 2 && i < MARKET_STRUCTURE_FIRST_PUBLISHED_BREAK {
            0.0
        } else if mode == 0 && i == MARKET_STRUCTURE_FIRST_PUBLISHED_BREAK {
            0.012
        } else if mode == 1 && i == MARKET_STRUCTURE_FIRST_PUBLISHED_BREAK {
            -0.012
        } else {
            match mode {
                0 => random_step + (phase / 17.0).sin() * 0.0007,
                1 => {
                    let regime = if (i / 83) % 2 == 0 { 1.0 } else { -1.0 };
                    regime * 0.00045 + (phase / 9.0).sin() * 0.0012 + random_step * 0.35
                }
                _ => {
                    // Repeating gaps, dojis, long bodies and asymmetric wicks make
                    // candlestick-pattern aliases much less likely to survive by
                    // accident than on a smooth random walk.
                    let shape = match i % 12 {
                        0 => 0.006,
                        1 => -0.005,
                        2 => 0.000_001,
                        3 => -0.000_001,
                        4 => 0.003,
                        5 => -0.001,
                        6 => -0.004,
                        7 => 0.002,
                        8 => 0.0,
                        9 => 0.005,
                        10 => -0.003,
                        _ => random_step,
                    };
                    shape + (phase / 5.0).cos() * 0.0004
                }
            }
        };
        let gap = if mode < 2 && i <= MARKET_STRUCTURE_FIRST_PUBLISHED_BREAK {
            0.0
        } else if i > 0 && i % 97 == 0 {
            if (i / 97) % 2 == 0 { 0.012 } else { -0.011 }
        } else {
            0.0
        };
        let o = (price + gap).clamp(0.4, 2.5);
        let c = (o + shaped_step).clamp(0.4, 2.5);
        let upper_random = random();
        let lower_random = random();
        let (upper_wick, lower_wick) = if mode < 2 && i <= MARKET_STRUCTURE_FIRST_PUBLISHED_BREAK {
            // Fixed pre-break extremes guarantee that the centred swing
            // detector has active levels without accidentally confirming
            // either direction before the close-only witness bar.
            (0.001, 0.001)
        } else {
            (
                0.00002 + upper_random * if i % 7 == 0 { 0.007 } else { 0.0015 },
                0.00002 + lower_random * if i % 11 == 0 { 0.006 } else { 0.0013 },
            )
        };
        open.push(o);
        high.push(o.max(c) + upper_wick);
        low.push((o.min(c) - lower_wick).max(0.0001));
        close.push(c);
        volume.push(1.0 + ((i * 37) % 997) as f64 + random() * 250.0);
        timestamp.push(1_577_836_800_000 + i as i64 * 300_000);
        price = c;
    }

    Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume: Some(volume),
    }
}

fn canonical_equal(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(&left, &right)| {
            (left.is_nan() && right.is_nan())
                || (left == 0.0 && right == 0.0)
                || left.to_bits() == right.to_bits()
        })
}

fn column_identity(name: &str) -> Option<(&'static str, Option<usize>, &str)> {
    let owner = ALL_INDICATORS
        .iter()
        .copied()
        .filter(|id| name == *id || name.starts_with(&format!("{id}_")))
        .max_by_key(|id| id.len())?;
    let suffix = name
        .strip_prefix(owner)?
        .strip_prefix('_')
        .unwrap_or_default();
    if suffix.is_empty() {
        return Some((owner, None, ""));
    }
    let (first, remainder) = suffix
        .split_once('_')
        .map_or((suffix, ""), |(first, remainder)| (first, remainder));
    if let Ok(period) = first.parse::<usize>()
        && ALT_PERIODS.contains(&period)
    {
        return Some((owner, Some(period), remainder));
    }
    Some((owner, None, suffix))
}

#[test]
fn production_columns_are_not_structural_aliases_across_adversarial_fixtures() {
    let fixtures = [
        adversarial_fixture(0x2026_0818_0000_0001, 0),
        adversarial_fixture(0x2026_0818_0000_0002, 1),
        adversarial_fixture(0x2026_0818_0000_0003, 2),
    ];
    let frames = fixtures
        .iter()
        .map(|fixture| {
            compute_classic_ta_columns_with_policy(fixture, IndicatorComputePolicy::CpuOnly)
                .expect("adversarial formula fixture must build the production vocabulary")
                .into_iter()
                .collect::<BTreeMap<_, _>>()
        })
        .collect::<Vec<_>>();

    let names = frames[0].keys().cloned().collect::<Vec<_>>();
    for frame in &frames[1..] {
        assert_eq!(
            names,
            frame.keys().cloned().collect::<Vec<_>>(),
            "production schema changed between formula fixtures"
        );
    }

    let bullish_change = &frames[0]["market_structure_confluence_bullish_change"];
    let bullish_choch = &frames[0]["market_structure_confluence_bullish_choch"];
    assert_eq!(
        bullish_change[MARKET_STRUCTURE_FIRST_PUBLISHED_BREAK], 1.0,
        "the production column lost the delayed first bullish structure change"
    );
    assert_eq!(
        bullish_choch[MARKET_STRUCTURE_FIRST_PUBLISHED_BREAK], 0.0,
        "the first bullish break is not a reversal and must not be labelled CHoCH"
    );

    let bearish_change = &frames[1]["market_structure_confluence_bearish_change"];
    let bearish_choch = &frames[1]["market_structure_confluence_bearish_choch"];
    assert_eq!(
        bearish_change[MARKET_STRUCTURE_FIRST_PUBLISHED_BREAK], 1.0,
        "the production column lost the delayed first bearish structure change"
    );
    assert_eq!(
        bearish_choch[MARKET_STRUCTURE_FIRST_PUBLISHED_BREAK], 0.0,
        "the first bearish break is not a reversal and must not be labelled CHoCH"
    );

    let mut structural = BTreeSet::new();
    let mut structurally_duplicate_sweeps = BTreeSet::new();
    for (left_index, left_name) in names.iter().enumerate() {
        for right_name in &names[left_index + 1..] {
            // A pair that never has finite variation in any fixture is not
            // evidence of a formula alias: long-warmup/event outputs can both
            // be all-NaN or all-zero on a bounded fixture. Disabled/default
            // constant outputs are governed by the separately audited static
            // exclusion table. This gate asks whether two *varying formulas*
            // remain identical under three different inputs.
            if !frames.iter().any(|frame| {
                has_finite_variation(frame.get(left_name).expect("left column exists"))
                    && has_finite_variation(frame.get(right_name).expect("right column exists"))
            }) {
                continue;
            }
            if !frames.iter().all(|frame| {
                canonical_equal(
                    frame.get(left_name).expect("left column exists"),
                    frame.get(right_name).expect("right column exists"),
                )
            }) {
                continue;
            }
            structural.insert((left_name.clone(), right_name.clone()));
            let same_swept_indicator =
                match (column_identity(left_name), column_identity(right_name)) {
                    (
                        Some((left_id, left_period, left_output)),
                        Some((right_id, right_period, right_output)),
                    ) => {
                        left_id == right_id
                            && left_output == right_output
                            && left_period != right_period
                            && (left_period.is_some() || right_period.is_some())
                    }
                    _ => false,
                };
            if same_swept_indicator {
                structurally_duplicate_sweeps.insert((left_name.clone(), right_name.clone()));
            }
        }
    }

    assert!(
        structurally_duplicate_sweeps.is_empty(),
        "period sweep emitted structurally identical columns across three adversarial fixtures: \
         {structurally_duplicate_sweeps:#?}"
    );
    assert!(
        structural.is_empty(),
        "production schema still contains structural output aliases across three adversarial \
         fixtures: {structural:#?}"
    );
}
