//! The scenario as the unit of work.
//!
//! `ScenarioDescriptor` has existed in `neoethos-gpu-contracts` since the
//! contracts crate was written — ten fields describing *what evaluation to
//! run*: which gene, over which window, with which costs, under which
//! perturbation. Every one of them except `scenario_id` was written as a
//! literal zero by the host, and the native side extracted only `scenario_id`.
//! The contract was designed for this and was not being used.
//!
//! Using it changes what a launch IS. Before: one launch evaluates one gene
//! array under one settings struct, so a screen that wants the same genes under
//! three different treatments needs three launches, and a screen that wants
//! 17 400 Monte-Carlo perturbations needs 17 400 gene CLONES staged on the host.
//! After: one launch evaluates an ARRAY OF DESCRIPTORS, each naming its own
//! gene, window, costs and perturbation counter. The measured quality screen —
//! 174 candidates x (1 base + 100 Monte-Carlo + 1 sensitivity) — becomes ONE
//! descriptor array and ONE launch.
//!
//! This module is the host half of that contract: the type codes, the
//! fixed-point scales the integer fields carry, the perturbation RNG that host
//! and device must agree on bit-for-bit, and the validation that refuses a
//! descriptor the device would misread.
//!
//! It is deliberately NOT feature-gated. The descriptors are built by
//! `discovery.rs` and consumed by the CPU fallback in `eval.rs`, both of which
//! exist in every build; only the CUDA lane that runs them is gated. A contract
//! that only compiles with a card attached is a contract nobody can test.

use neoethos_gpu_contracts::device::ScenarioDescriptor;

// ── Scenario types ───────────────────────────────────────────────────────────
//
// These MUST match `kScenario*` in
// `crates/neoethos-gpu-cuda/native/prototype_b_population.cu`.
// `the_scenario_type_codes_match_the_kernel` reads them out of the `.cu` so the
// two cannot drift: a renumbering that compiles on both sides would silently
// run cost scenarios as base ones.

/// The whole series, the settings' own costs, the gene exactly as uploaded.
///
/// This is what every existing caller already asks for, and a descriptor array
/// of nothing but `Base` with zeroed fields is bit-identical to the behaviour
/// before scenarios existed. That is the parity floor for this entire change.
pub const SCENARIO_BASE: u32 = 0;

/// Monte-Carlo: the gene's thresholds, weights and stops are perturbed ON THE
/// DEVICE from `rng_counter`.
///
/// NOT the default. See [`device_monte_carlo`] — the shipped Monte-Carlo screen
/// perturbs on the host with ChaCha8 and uploads the clones, and this lane
/// cannot reproduce ChaCha8 bit-for-bit.
pub const SCENARIO_PERTURB: u32 = 1;

/// Cost sensitivity: the same gene over the same window with `spread_ticks` /
/// `commission_micros` replacing the settings' spread and commission.
pub const SCENARIO_COST: u32 = 2;

// ── Fixed-point scales ───────────────────────────────────────────────────────
//
// `spread_ticks` / `slippage_ticks` are `i32` and `commission_micros` is `i64`,
// so a cost override has to be quantised — and quantising a cost SILENTLY
// CHANGES WHAT THE SCREEN MEASURES. That is the one thing this must not do.
//
// The scales are powers of ten, and the conversion is a DIVISION by the scale
// rather than a multiplication by its reciprocal, because `1e-3` is not
// representable in binary while `1000.0` is exact: `1400 / 1000.0` is the
// correctly-rounded double nearest 1.4, which is bit-for-bit the double the
// literal `1.4` parses to. A reciprocal multiply is not guaranteed to be.
//
// Values that do NOT round-trip exactly are refused, not rounded — see
// [`spread_ticks_exact`]. A sensitivity spread of 1.234 56 pips is a real
// configuration and it must produce an error the operator can read, never a
// launch that quietly charged 1.235.

/// `spread_ticks` and `slippage_ticks` are MILLIPIPS: thousandths of a pip.
///
/// An `i32` therefore spans +/- 2 147 483 pips, which is not a constraint any
/// broker will find limiting, at a resolution 100x finer than the finest spread
/// anyone quotes.
pub const TICKS_PER_PIP: f64 = 1000.0;

/// `commission_micros` is millionths of one account-currency unit per lot.
pub const MICROS_PER_UNIT: f64 = 1_000_000.0;

/// "No override — use the settings value."
///
/// A sentinel is needed rather than zero because ZERO IS A LEGITIMATE
/// OVERRIDE: "what does this strategy look like with no spread at all" is a
/// question the sensitivity screen may reasonably ask, and a zero that means
/// "unset" makes it unaskable. Negative is impossible for a cost, so -1 is free.
pub const NO_TICK_OVERRIDE: i32 = -1;
/// As [`NO_TICK_OVERRIDE`], for `commission_micros`.
pub const NO_MICRO_OVERRIDE: i64 = -1;

/// Quantise a spread (or slippage) in pips to millipips, or `None` if the
/// quantisation would change the number.
///
/// `None` is not a failure to be worked around. It means "this cost cannot be
/// carried in a scenario descriptor without changing it", and the caller's only
/// correct response is to fall back to a separate launch with the exact f64 in
/// the settings struct — which is what the code did before scenarios existed
/// and is still correct, merely slower.
pub fn spread_ticks_exact(pips: f64) -> Option<i32> {
    if !pips.is_finite() || pips < 0.0 {
        return None;
    }
    let scaled = (pips * TICKS_PER_PIP).round();
    if scaled > i32::MAX as f64 {
        return None;
    }
    let ticks = scaled as i32;
    // The round trip is the test, and it is the SAME arithmetic the device
    // performs (`ticks / 1000.0`), so agreement here is agreement there.
    if f64::from(ticks) / TICKS_PER_PIP == pips {
        Some(ticks)
    } else {
        None
    }
}

/// Quantise a per-lot commission to micros, or `None` if it would change.
pub fn commission_micros_exact(units: f64) -> Option<i64> {
    if !units.is_finite() || units < 0.0 {
        return None;
    }
    let scaled = (units * MICROS_PER_UNIT).round();
    if scaled > i64::MAX as f64 {
        return None;
    }
    let micros = scaled as i64;
    if micros as f64 / MICROS_PER_UNIT == units {
        Some(micros)
    } else {
        None
    }
}

// ── The perturbation RNG ─────────────────────────────────────────────────────
//
// THE DETERMINISM QUESTION, ANSWERED EXPLICITLY.
//
// The shipped Monte-Carlo screen builds `mc_runs` perturbed gene clones per
// candidate on the host, with `rand_chacha::ChaCha8Rng::seed_from_u64(combo_seed
// ^ ((candidate_idx as u64) << 20) ^ run_idx)`, drawing in this exact order:
//
//   long_threshold *= 1 + U(-0.15, 0.15)
//   short_threshold *= 1 + U(-0.15, 0.15)
//   each weight    *= 1 + U(-0.20, 0.20)
//   sl_pips        *= 1 + U(-0.25, 0.25)   (only if finite and > 0)
//   tp_pips        *= 1 + U(-0.25, 0.25)   (only if finite and > 0)
//
// ChaCha8 is a STREAM cipher. A device thread that wants the 4 097th draw of a
// stream must generate the 4 096 before it, and `rand`'s `random_range` consumes
// a variable number of words per draw through rejection sampling. Reproducing
// that on a GPU is not a matter of effort — it is a different shape of
// computation, and any attempt would be a re-implementation of `rand`'s
// internals whose agreement no test in this repo could establish.
//
// So the device lane uses a COUNTER-BASED generator instead: the k-th draw of
// stream `c` is a pure function `hash(c, k)`, computable by any thread in O(1)
// with no state. It is genuinely deterministic and genuinely reproducible — it
// is simply NOT THE SAME SEQUENCE ChaCha8 produces.
//
// That difference is the reason for all of the following, and none of it is
// optional:
//
//   * the host lane stays the DEFAULT (`device_monte_carlo()` is false);
//   * the host lane is the parity REFERENCE, pinned by
//     `the_host_monte_carlo_draw_order_is_pinned` in `discovery.rs`;
//   * turning the device lane on is an explicit, deliberate act;
//   * and when it is on, the CPU fallback uses THIS generator too
//     (`perturbed_gene_arrays`), so a GPU failure does not silently swap one
//     Monte-Carlo screen for another mid-run.
//
// What the device lane buys, when an operator decides the perturbation
// DISTRIBUTION (not the exact draws) is what the screen actually measures:
// 17 400 gene clones — ~23 MB of host staging per chunk, rebuilt every chunk —
// stop existing. The gene array is 174 genes and the scenario array carries the
// counters.

/// Amplitude of the threshold perturbation: `U(-0.15, 0.15)`.
pub const MC_THRESHOLD_AMPLITUDE: f64 = 0.15;
/// Amplitude of the per-term weight perturbation: `U(-0.20, 0.20)`.
pub const MC_WEIGHT_AMPLITUDE: f64 = 0.20;
/// Amplitude of the stop/target perturbation: `U(-0.25, 0.25)`.
pub const MC_STOP_AMPLITUDE: f64 = 0.25;

/// Draw index of the long threshold.
pub const DRAW_LONG_THRESHOLD: u64 = 0;
/// Draw index of the short threshold.
pub const DRAW_SHORT_THRESHOLD: u64 = 1;
/// Draw index of the first CSR weight; term `t` uses `DRAW_WEIGHT_BASE + t`.
pub const DRAW_WEIGHT_BASE: u64 = 2;

/// Draw index of the stop perturbation for a gene with `term_count` terms.
///
/// The stops come AFTER the weights in the host's draw order, so their index
/// depends on how many terms the gene has. Getting this wrong would not fail —
/// it would perturb by a different, equally plausible number.
pub fn draw_stop(term_count: u64) -> u64 {
    DRAW_WEIGHT_BASE + term_count
}

/// Draw index of the target perturbation. See [`draw_stop`].
pub fn draw_target(term_count: u64) -> u64 {
    DRAW_WEIGHT_BASE + term_count + 1
}

/// splitmix64, the finalizer used by both lanes.
///
/// Chosen because it is four instructions of shift/multiply/xor with no state,
/// no tables and no branches, so the CUDA transcription is character-for-
/// character the same arithmetic on the same 64-bit integer type. Statistical
/// quality is not the binding requirement here — reproducibility across two
/// compilers is, and a generator with any internal state or any data-dependent
/// control flow would put that at risk for no gain.
#[inline]
fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The `draw`-th uniform of stream `counter`, in `[0, 1)`.
///
/// The conversion is exact and identical on both lanes: `h >> 11` is strictly
/// below 2^53 so it converts to `f64` without rounding, and `1 / 2^53` is
/// exactly representable so the multiply is exact too. There is no rounding
/// anywhere in this function, which is what lets a CUDA transcription be
/// bit-identical rather than merely close.
#[inline]
pub fn perturb_unit(counter: u64, draw: u64) -> f64 {
    let mixed = splitmix64(counter ^ splitmix64(draw.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    ((mixed >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
}

/// `1 + U(-amplitude, amplitude)`, the multiplier applied to a gene parameter.
///
/// Written as two separate operations on purpose. `-fmad=false` is enforced for
/// the CUDA build (`neoethos-gpu-cuda/build.rs`), so neither `2 * u - 1` nor
/// `1 + m * a` may be contracted into an FMA on the device, and the two lanes
/// therefore round at the same two places. Without that flag this function
/// would be bit-identical on paper and not in practice.
#[inline]
pub fn perturb_factor(counter: u64, draw: u64, amplitude: f64) -> f64 {
    let unit = perturb_unit(counter, draw);
    let centred = 2.0 * unit - 1.0;
    1.0 + centred * amplitude
}

/// The gene arrays a `SCENARIO_PERTURB` descriptor asks the device to evaluate,
/// materialised on the host.
///
/// This exists so the CPU fallback evaluates THE SAME PERTURBED GENE the device
/// would have. Without it a GPU failure in a device-Monte-Carlo run would fall
/// back to the unperturbed gene and report a Monte-Carlo pass count that was
/// never computed from any perturbation at all — the worst kind of wrong, since
/// every downstream number stays plausible.
///
/// Thresholds, weights and stops remain f64 in both host and device contracts.
/// No host perturbation may narrow a shared strategy parameter before launch.
#[derive(Debug, Clone, PartialEq)]
pub struct PerturbedGene {
    pub long_threshold: f64,
    pub short_threshold: f64,
    pub weights: Vec<f64>,
    pub sl_pips: f64,
    pub tp_pips: f64,
}

/// Apply the device's perturbation to one gene's parameters on the host.
///
/// `weights` is the gene's own CSR term window, not the whole population array.
pub fn perturbed_gene(
    counter: u64,
    long_threshold: f64,
    short_threshold: f64,
    weights: &[f64],
    sl_pips: f64,
    tp_pips: f64,
) -> PerturbedGene {
    let term_count = weights.len() as u64;
    let long =
        long_threshold * perturb_factor(counter, DRAW_LONG_THRESHOLD, MC_THRESHOLD_AMPLITUDE);
    let short =
        short_threshold * perturb_factor(counter, DRAW_SHORT_THRESHOLD, MC_THRESHOLD_AMPLITUDE);
    let perturbed_weights = weights
        .iter()
        .enumerate()
        .map(|(term, weight)| {
            *weight * perturb_factor(counter, DRAW_WEIGHT_BASE + term as u64, MC_WEIGHT_AMPLITUDE)
        })
        .collect();
    // The finite-and-positive guard mirrors the host screen's, and the device's:
    // a gene with no fixed stop must not acquire one by being multiplied.
    let stop = if sl_pips.is_finite() && sl_pips > 0.0 {
        sl_pips * perturb_factor(counter, draw_stop(term_count), MC_STOP_AMPLITUDE)
    } else {
        sl_pips
    };
    let target = if tp_pips.is_finite() && tp_pips > 0.0 {
        tp_pips * perturb_factor(counter, draw_target(term_count), MC_STOP_AMPLITUDE)
    } else {
        tp_pips
    };
    PerturbedGene {
        long_threshold: long,
        short_threshold: short,
        weights: perturbed_weights,
        sl_pips: stop,
        tp_pips: target,
    }
}

// ── The device Monte-Carlo switch ────────────────────────────────────────────

static DEVICE_MONTE_CARLO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Turn the on-device Monte-Carlo lane on for this process.
///
/// Explicit, once, and never from an environment variable — this repo has one
/// resolution point for behaviour-changing settings and 215 env names is the
/// defect that directive exists to stop repeating.
///
/// Returns whether the value was installed; a second call is refused rather
/// than silently ignored, because two halves of a run using two different
/// Monte-Carlo generators is exactly the failure this switch exists to make
/// impossible.
pub fn install_device_monte_carlo(enabled: bool) -> bool {
    DEVICE_MONTE_CARLO.set(enabled).is_ok()
}

/// Whether Monte-Carlo perturbation runs on the device.
///
/// FALSE by default, and the default is the shipped behaviour: ChaCha8 clones
/// built on the host. See the long note above [`MC_THRESHOLD_AMPLITUDE`] for why
/// the two lanes measure the same distribution and not the same draws.
pub fn device_monte_carlo() -> bool {
    *DEVICE_MONTE_CARLO.get_or_init(|| false)
}

// ── Descriptor construction ──────────────────────────────────────────────────

/// A full-series scenario over gene `candidate`, with the settings' own costs.
///
/// Every field the device reads is written explicitly, including the ones that
/// are zero, so that "zero" is a decision recorded here rather than a default
/// nobody chose.
pub fn base_scenario(candidate: u64, scenario_id: u64, bars: usize) -> ScenarioDescriptor {
    ScenarioDescriptor {
        base_candidate_id: candidate,
        scenario_id,
        rng_counter: 0,
        window_offset: 0,
        window_len: bars as u32,
        scenario_type: SCENARIO_BASE,
        spread_ticks: NO_TICK_OVERRIDE,
        slippage_ticks: 0,
        commission_micros: NO_MICRO_OVERRIDE,
        perturbation_offset: 0,
        perturbation_count: 0,
        reserved: 0,
    }
}

/// A full-series scenario with the cost overrides the sensitivity screen wants.
///
/// `None` for either override leaves the settings' value in place, which is how
/// a caller asks for "wider spread, same commission" without inventing a number.
pub fn cost_scenario(
    candidate: u64,
    scenario_id: u64,
    bars: usize,
    spread_ticks: Option<i32>,
    commission_micros: Option<i64>,
) -> ScenarioDescriptor {
    ScenarioDescriptor {
        scenario_type: SCENARIO_COST,
        spread_ticks: spread_ticks.unwrap_or(NO_TICK_OVERRIDE),
        commission_micros: commission_micros.unwrap_or(NO_MICRO_OVERRIDE),
        ..base_scenario(candidate, scenario_id, bars)
    }
}

/// A Monte-Carlo scenario perturbed on the device from `counter`.
pub fn perturb_scenario(
    candidate: u64,
    scenario_id: u64,
    bars: usize,
    counter: u64,
) -> ScenarioDescriptor {
    ScenarioDescriptor {
        scenario_type: SCENARIO_PERTURB,
        rng_counter: counter,
        ..base_scenario(candidate, scenario_id, bars)
    }
}

/// Refuse a descriptor array the device would misread.
///
/// Everything here is a bounds or contract violation that the kernel cannot
/// detect for itself: a `base_candidate_id` past the end of the gene array is an
/// out-of-bounds read of thresholds and CSR offsets, and a window past the end
/// of the series is an out-of-bounds read of prices. Both would produce numbers.
pub fn validate_scenarios(
    scenarios: &[ScenarioDescriptor],
    population: usize,
    bars: usize,
) -> Result<(), String> {
    if scenarios.is_empty() {
        return Err("scenario array is empty".to_string());
    }
    for (index, scenario) in scenarios.iter().enumerate() {
        if scenario.base_candidate_id as usize >= population {
            return Err(format!(
                "scenario {index} names gene {} outside the population of {population}",
                scenario.base_candidate_id
            ));
        }
        let offset = scenario.window_offset as usize;
        if offset >= bars {
            return Err(format!(
                "scenario {index} starts at bar {offset}, past the {bars}-bar series"
            ));
        }
        let len = if scenario.window_len == 0 {
            bars - offset
        } else {
            scenario.window_len as usize
        };
        if offset + len > bars {
            return Err(format!(
                "scenario {index} covers bars {offset}..{} of a {bars}-bar series",
                offset + len
            ));
        }
        match scenario.scenario_type {
            SCENARIO_BASE | SCENARIO_PERTURB | SCENARIO_COST => {}
            other => {
                return Err(format!("scenario {index} has unknown type {other}"));
            }
        }
        if scenario.spread_ticks < NO_TICK_OVERRIDE {
            return Err(format!(
                "scenario {index} has spread_ticks {} — only -1 (no override) may be negative",
                scenario.spread_ticks
            ));
        }
        if scenario.commission_micros < NO_MICRO_OVERRIDE {
            return Err(format!(
                "scenario {index} has commission_micros {} — only -1 (no override) may be negative",
                scenario.commission_micros
            ));
        }
        // `slippage_ticks` has NO sentinel — 0 already means "none" — so the
        // bound is 0 and not `NO_TICK_OVERRIDE`. It was the one cost field
        // neither this validator nor the native one checked, and it is applied
        // SIGNED at three sites in the kernel: `entry_price += direction *
        // slippage_price` and `exit_price -= direction * slippage_price` twice.
        // A negative value therefore fills every entry AND every exit better
        // than the market — free money on both sides, and a screen that reports
        // every strategy as robust. Nothing emits a non-zero slippage today;
        // `ScenarioDescriptor` is public and the whole point of the change is
        // that new callers build work lists.
        if scenario.slippage_ticks < 0 {
            return Err(format!(
                "scenario {index} has slippage_ticks {} — slippage may not be negative, that is \
                 a favourable fill on both the entry and the exit",
                scenario.slippage_ticks
            ));
        }
    }
    Ok(())
}

/// Does this scenario ask for anything the CPU mirror cannot reproduce?
///
/// The CPU fallback evaluates a scenario by materialising it into the existing
/// per-gene CPU path, which walks whatever series it is handed from bar 1. That
/// covers base, cost and device-perturbation scenarios exactly. It does NOT
/// cover a sub-window, because the adaptive stop base and the SMC rows would
/// have to be sliced with it and nothing in the tree builds a sub-window
/// descriptor yet.
///
/// So the mirror says so rather than evaluating the full series and returning a
/// number that looks like a windowed result. The device capability is real and
/// tested; the host simply has no consumer for it yet.
pub fn cpu_mirror_unsupported(scenario: &ScenarioDescriptor, bars: usize) -> Option<String> {
    let len = if scenario.window_len == 0 {
        bars.saturating_sub(scenario.window_offset as usize)
    } else {
        scenario.window_len as usize
    };
    if scenario.window_offset != 0 || len != bars {
        return Some(format!(
            "scenario {} asks for bars {}..{} and the CPU mirror only walks whole series",
            scenario.scenario_id,
            scenario.window_offset,
            scenario.window_offset as usize + len
        ));
    }
    if scenario.slippage_ticks != 0 {
        return Some(format!(
            "scenario {} asks for {} millipips of slippage and the CPU engine models none",
            scenario.scenario_id, scenario.slippage_ticks
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str =
        include_str!("../../../neoethos-gpu-cuda/native/prototype_b_population.cu");

    fn kernel_constant(name: &str) -> String {
        let line = KERNEL
            .lines()
            .find(|line| line.contains(&format!("{name} =")) && line.contains("constexpr"))
            .unwrap_or_else(|| panic!("the kernel declares {name}"));
        line.rsplit('=')
            .next()
            .expect("a constexpr has a right-hand side")
            .trim()
            .trim_end_matches(';')
            .trim_end_matches("ull")
            .trim_end_matches('u')
            .trim()
            .to_string()
    }

    /// Two languages agreeing about a number by convention is how the
    /// retry-smaller path silently stopped working. This one is checked.
    #[test]
    fn the_scenario_type_codes_match_the_kernel() {
        for (name, expected) in [
            ("kScenarioBase", SCENARIO_BASE),
            ("kScenarioPerturb", SCENARIO_PERTURB),
            ("kScenarioCost", SCENARIO_COST),
        ] {
            assert_eq!(
                kernel_constant(name),
                expected.to_string(),
                "{name} disagrees with the host constant"
            );
        }
    }

    /// The fixed-point scales decide what a cost override MEANS. A device that
    /// divides by 100 while the host multiplies by 1 000 would charge a tenth of
    /// the spread and report a strategy that survives costs it never paid.
    #[test]
    fn the_fixed_point_scales_match_the_kernel() {
        assert_eq!(kernel_constant("kTicksPerPip"), "1000.0");
        assert_eq!(kernel_constant("kMicrosPerUnit"), "1000000.0");
        assert_eq!(
            kernel_constant("kNoTickOverride"),
            NO_TICK_OVERRIDE.to_string()
        );
        assert_eq!(
            kernel_constant("kNoMicroOverride"),
            NO_MICRO_OVERRIDE.to_string()
        );
    }

    /// The whole point of refusing rather than rounding.
    #[test]
    fn a_cost_that_cannot_be_carried_exactly_is_refused_not_rounded() {
        // The values the sensitivity screen actually uses round-trip.
        assert_eq!(spread_ticks_exact(2.0), Some(2_000));
        assert_eq!(spread_ticks_exact(1.4), Some(1_400));
        assert_eq!(spread_ticks_exact(0.0), Some(0));
        assert_eq!(commission_micros_exact(7.0), Some(7_000_000));
        assert_eq!(commission_micros_exact(3.5), Some(3_500_000));

        // And the round trip is the SAME division the device performs.
        assert_eq!(
            f64::from(spread_ticks_exact(1.4).unwrap()) / TICKS_PER_PIP,
            1.4
        );
        assert_eq!(
            commission_micros_exact(3.5).unwrap() as f64 / MICROS_PER_UNIT,
            3.5
        );

        // A fourth decimal cannot be carried in millipips, so it is refused.
        // Rounding it would move the spread by 0.4 % and the screen would never
        // say so.
        assert_eq!(spread_ticks_exact(1.234_56), None);
        assert_eq!(commission_micros_exact(7.000_000_5), None);
        assert_eq!(spread_ticks_exact(f64::NAN), None);
        assert_eq!(spread_ticks_exact(-1.0), None);
    }

    /// The host constants, the contract constants and the kernel constants are
    /// three copies of the same two numbers. The kernel pair is checked above;
    /// this checks the contract pair, so a descriptor built anywhere in the
    /// workspace means the same thing.
    #[test]
    fn the_sentinels_agree_with_the_contract() {
        assert_eq!(
            NO_TICK_OVERRIDE,
            neoethos_gpu_contracts::device::NO_TICK_OVERRIDE
        );
        assert_eq!(
            NO_MICRO_OVERRIDE,
            neoethos_gpu_contracts::device::NO_MICRO_OVERRIDE
        );
    }

    /// `ScenarioDescriptor::default()` must not be a free-trading backtest.
    ///
    /// It was: `#[derive(Default)]` writes zeros, and 0 stopped meaning "use the
    /// settings' costs" the moment the sentinel became -1. Three fixture sites
    /// built descriptors with `..ScenarioDescriptor::default()`, one of them a
    /// cost fixture charging 2.0 pips and 10.0 per lot on the CPU side while the
    /// device would have charged nothing.
    #[test]
    fn the_default_descriptor_charges_the_settings_costs() {
        let default = ScenarioDescriptor::default();
        assert_eq!(default.spread_ticks, NO_TICK_OVERRIDE);
        assert_eq!(default.commission_micros, NO_MICRO_OVERRIDE);
        assert_eq!(default.slippage_ticks, 0);
        // And every builder agrees with it on the cost fields.
        let base = base_scenario(0, 0, 1_000);
        assert_eq!(base.spread_ticks, default.spread_ticks);
        assert_eq!(base.commission_micros, default.commission_micros);
        assert_eq!(base.slippage_ticks, default.slippage_ticks);
    }

    /// A negative slippage is a favourable fill on BOTH the entry and the exit.
    #[test]
    fn negative_slippage_is_refused() {
        let bars = 1_000;
        let mut favourable = base_scenario(0, 0, bars);
        favourable.slippage_ticks = -5;
        let error = validate_scenarios(&[favourable], 1, bars).unwrap_err();
        assert!(error.contains("slippage"), "{error}");

        // Zero and positive are both legitimate.
        let mut costly = base_scenario(0, 0, bars);
        costly.slippage_ticks = 5;
        assert!(validate_scenarios(&[costly], 1, bars).is_ok());
        assert!(validate_scenarios(&[base_scenario(0, 0, bars)], 1, bars).is_ok());
    }

    /// Zero must be an override, not a synonym for "unset".
    #[test]
    fn zero_cost_is_a_real_override() {
        let scenario = cost_scenario(0, 0, 1_000, spread_ticks_exact(0.0), None);
        assert_eq!(scenario.spread_ticks, 0);
        assert_ne!(scenario.spread_ticks, NO_TICK_OVERRIDE);
        assert_eq!(scenario.commission_micros, NO_MICRO_OVERRIDE);
    }

    /// The generator is a pure function of (stream, draw): the property that
    /// makes it usable from a GPU thread at all.
    #[test]
    fn the_perturbation_generator_is_indexable_and_bounded() {
        for counter in [0u64, 1, 7, 0xDEAD_BEEF_CAFE_F00D] {
            for draw in 0..64u64 {
                let unit = perturb_unit(counter, draw);
                assert!((0.0..1.0).contains(&unit), "unit {unit} out of range");
                assert_eq!(unit, perturb_unit(counter, draw), "not a pure function");
                let factor = perturb_factor(counter, draw, MC_WEIGHT_AMPLITUDE);
                assert!(
                    (1.0 - MC_WEIGHT_AMPLITUDE..=1.0 + MC_WEIGHT_AMPLITUDE).contains(&factor),
                    "factor {factor} outside the amplitude"
                );
            }
        }
        // Different streams must not collide, or 100 Monte-Carlo runs would be
        // one run counted 100 times.
        let a: Vec<f64> = (0..8).map(|d| perturb_unit(11, d)).collect();
        let b: Vec<f64> = (0..8).map(|d| perturb_unit(12, d)).collect();
        assert_ne!(a, b);
    }

    /// The draw ORDER is the contract with the host screen. Weights occupy a
    /// variable-length block, so the stop and target indices move with the gene.
    #[test]
    fn the_draw_order_puts_the_stops_after_the_weights() {
        assert_eq!(DRAW_LONG_THRESHOLD, 0);
        assert_eq!(DRAW_SHORT_THRESHOLD, 1);
        assert_eq!(DRAW_WEIGHT_BASE, 2);
        // A three-term gene draws 0,1 then 2,3,4 then 5,6.
        assert_eq!(draw_stop(3), 5);
        assert_eq!(draw_target(3), 6);
        // No index may be reused, or two parameters would move together.
        let used: Vec<u64> = (0..3)
            .map(|t| DRAW_WEIGHT_BASE + t)
            .chain([
                DRAW_LONG_THRESHOLD,
                DRAW_SHORT_THRESHOLD,
                draw_stop(3),
                draw_target(3),
            ])
            .collect();
        let mut sorted = used.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), used.len(), "a draw index is used twice");
    }

    /// A gene with no fixed stop must not acquire one by being perturbed.
    #[test]
    fn a_gene_without_fixed_stops_keeps_them_unset() {
        let unset = perturbed_gene(42, 0.5, -0.5, &[1.0, 2.0], 0.0, f64::NAN);
        assert_eq!(unset.sl_pips, 0.0);
        assert!(unset.tp_pips.is_nan());
        assert_ne!(unset.long_threshold, 0.5);

        let set = perturbed_gene(42, 0.5, -0.5, &[1.0, 2.0], 20.0, 40.0);
        assert_ne!(set.sl_pips, 20.0);
        assert!(set.sl_pips > 15.0 && set.sl_pips < 25.0);
        // And the same stream perturbs the same gene identically, which is what
        // makes the CPU fallback a mirror rather than a second experiment.
        assert_eq!(set, perturbed_gene(42, 0.5, -0.5, &[1.0, 2.0], 20.0, 40.0));
    }

    /// The parity floor: a base descriptor array must describe exactly what the
    /// engine did before scenarios existed.
    #[test]
    fn a_base_descriptor_asks_for_the_evaluation_that_already_ran() {
        let scenario = base_scenario(7, 7, 843_456);
        assert_eq!(scenario.scenario_type, SCENARIO_BASE);
        assert_eq!(scenario.window_offset, 0);
        assert_eq!(scenario.window_len, 843_456);
        assert_eq!(scenario.spread_ticks, NO_TICK_OVERRIDE);
        assert_eq!(scenario.commission_micros, NO_MICRO_OVERRIDE);
        assert_eq!(scenario.slippage_ticks, 0);
        assert_eq!(scenario.rng_counter, 0);
        assert!(cpu_mirror_unsupported(&scenario, 843_456).is_none());
    }

    #[test]
    fn a_descriptor_the_device_would_misread_is_refused() {
        let bars = 1_000;
        let good = vec![base_scenario(0, 0, bars), base_scenario(1, 1, bars)];
        assert!(validate_scenarios(&good, 2, bars).is_ok());

        // Past the end of the gene array: an out-of-bounds read of thresholds
        // and CSR offsets that would still produce a metric row.
        let bad_gene = vec![base_scenario(5, 0, bars)];
        assert!(
            validate_scenarios(&bad_gene, 2, bars)
                .unwrap_err()
                .contains("gene 5")
        );

        // Past the end of the series.
        let mut long_window = base_scenario(0, 0, bars);
        long_window.window_len = (bars + 1) as u32;
        assert!(validate_scenarios(&[long_window], 1, bars).is_err());

        let mut unknown = base_scenario(0, 0, bars);
        unknown.scenario_type = 9;
        assert!(
            validate_scenarios(&[unknown], 1, bars)
                .unwrap_err()
                .contains("type 9")
        );

        assert!(validate_scenarios(&[], 1, bars).is_err());
    }

    /// The CPU mirror must refuse what it cannot reproduce, in both directions.
    #[test]
    fn the_cpu_mirror_refuses_only_what_it_cannot_reproduce() {
        let bars = 1_000;
        // Cost and device-perturbation scenarios ARE reproducible on the CPU.
        assert!(
            cpu_mirror_unsupported(
                &cost_scenario(0, 0, bars, Some(2_000), Some(7_000_000)),
                bars
            )
            .is_none()
        );
        assert!(cpu_mirror_unsupported(&perturb_scenario(0, 0, bars, 99), bars).is_none());

        // A sub-window is not, and it says so instead of walking the whole
        // series and returning a plausible number.
        let mut window = base_scenario(0, 0, bars);
        window.window_offset = 100;
        window.window_len = 500;
        assert!(cpu_mirror_unsupported(&window, bars).is_some());

        let mut slipped = base_scenario(0, 0, bars);
        slipped.slippage_ticks = 5;
        assert!(
            cpu_mirror_unsupported(&slipped, bars)
                .unwrap()
                .contains("slippage")
        );
    }

    /// The default must be the shipped behaviour, whatever else is true.
    #[test]
    fn the_monte_carlo_default_is_the_host_lane() {
        // Note this consumes the process-wide OnceLock, which is why nothing
        // else in this module installs it.
        assert!(!device_monte_carlo());
    }
}
