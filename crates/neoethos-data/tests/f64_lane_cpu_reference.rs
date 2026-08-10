//! The parity hazard that no amount of care inside a `.cu` file can fix.
//!
//! # What this guards
//!
//! Every kernel in `vendor/vector-ta-0.2.9-patched/kernels/cuda/
//! neoethos_f64_kernels.cu` is written against a NAMED `*_scalar` CPU
//! reference, bar for bar, in that reference's exact accumulation order.
//!
//! But `hpc_ta` does not ask for the scalar kernel. It calls `compute_cpu`
//! with `Kernel::Auto`, and `neoethos-data` enables vector-ta's `nightly-avx`
//! feature, so on x86_64 several indicators resolve `Auto` to `Avx2` or
//! `Avx512` instead. `obv_with_kernel` (obv.rs:164) is the blunt case — it maps
//! `Kernel::Auto => Kernel::Avx2` unconditionally.
//!
//! A vectorised reduction that reassociates its adds produces a DIFFERENT f64
//! number from its own scalar sibling. If that happens for an indicator in the
//! device table, then a GPU-vs-CPU parity failure on a rented card would be
//! reported against the kernel when the kernel is correct and the two CPU
//! implementations are the ones that disagree. That is an expensive way to
//! learn something a laptop can prove in seconds.
//!
//! So this test runs the CPU twice — `Kernel::Scalar` and `Kernel::Auto` — over
//! the same bars and demands BIT equality, NaN mask included. It needs no GPU,
//! no CUDA toolkit and no feature flag.
//!
//! # What a failure means
//!
//! Not "loosen the tolerance". It means the named indicator must come OUT of
//! `vector_ta::indicators::dispatch::cuda_f64::F64_KERNELS` until the
//! divergence is explained, because until then there is no single CPU answer to
//! be in parity WITH.

use vector_ta::indicators::dispatch::{
    compute_cpu, IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, ParamKV, ParamValue,
};
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;

/// Every indicator the f64 device lane claims.
///
/// This list is duplicated from `cuda_f64::F64_KERNELS` on purpose: that table
/// lives behind vector-ta's `cuda` feature, and the whole value of this test is
/// that it runs on a machine with no CUDA at all. `table_matches_vector_ta`
/// below asserts the two are identical whenever the feature IS enabled, so the
/// duplication cannot rot silently.
const CLAIMED: &[&str] = &[
    // batch 1
    "sma", "ema", "rsi", "roc", "mom", "atr", "adx", "willr", "cci", "mfi",
    // batch 2 — the reachable half of hpc_ta::MULTI_PERIOD_IDS
    "tsi", "obv", //
    // 2026-08-10: `vwap` and `wilders` promoted out of WITHHELD. They were held
    // because Kernel::Scalar and Kernel::Auto disagreed by 1 ULP, so there was no
    // single CPU oracle to be in parity WITH. vector-ta fixed that at the source:
    // the divergence was never in the recurrence — all four wilders paths compute
    // the identical one-rounding `y = (x0 - y).mul_add(alpha, y)` — it was in the
    // WARM-UP SEED, summed 4-wide by scalar, 8- and 16-wide by AVX. The 4-wide
    // scalar association was chosen as the oracle with a written argument, the
    // vector paths now call it, and `vwap_row_scalar_pv` was deleted outright.
    "vwap", "wilders", //
    // batch 2 — moving averages, less `wilders`
    "wma", "smma", "dema", "tema", "zlema", "vwma", //
    // batch 2 — volatility / directional / volume
    "natr", "adxr", "efi", //
    // batch 2 — pointwise and windowed
    "medprice", "wclprice", "midpoint", "midprice", "rocp", "rocr",
];

/// The periods `hpc_ta::ALT_PERIODS` sweeps, plus 1 and 14 because several CPU
/// references carry a hand-unrolled fast path for `period == 14` and a
/// degenerate branch for `period == 1`. Those branches are exactly where a
/// scalar/AVX divergence would hide.
const PERIODS: &[i64] = &[1, 7, 14, 21, 50, 100, 200];

/// Deterministic bars with the shapes that break naive kernels: a long trend so
/// recursive indicators accumulate real rounding, flat runs where `obv`'s sign
/// term is zero, and a wide range so true-range picks a different one of its
/// three candidates from bar to bar.
fn candles(n: usize) -> Candles {
    let ts: Vec<i64> = (0..n as i64).map(|i| 1_600_000_000_000 + i * 300_000).collect();
    let (mut o, mut h, mut l, mut c, mut v) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut p = 1.1000f64;
    for i in 0..n {
        let f = i as f64;
        // Flat every 11th bar, so `c == prev_close` really occurs.
        let step = if i % 11 == 0 {
            0.0
        } else {
            (f * 0.017).sin() * 0.0009 + (f * 0.0007).cos() * 0.00035
        };
        let open = p;
        p += step;
        o.push(open);
        h.push(open.max(p) + 0.00041 + (f * 0.31).sin().abs() * 0.0006);
        l.push(open.min(p) - 0.00039 - (f * 0.23).cos().abs() * 0.0006);
        c.push(p);
        v.push(950.0 + ((i * 37) % 211) as f64);
    }
    Candles::new(ts, o, h, l, c, v)
}

/// The same walk, but with the shapes a real frame actually has and
/// [`candles`] deliberately does not: leading NaN runs of DIFFERENT length per
/// series, an interior NaN bar, and a zero close.
///
/// # Why a second fixture exists
///
/// [`candles`] is smooth and every value is finite. That is the input on which
/// a reassociating AVX reduction and its scalar sibling are MOST likely to
/// agree, because much of what separates them is how each handles a non-finite
/// operand. The two divergences the clean fixture found (`vwap` at 1 ULP,
/// `wilders` at its seed bar) are therefore the floor, not the ceiling — and 22
/// of the 27 registered indicators have genuinely distinct AVX bodies rather
/// than `*_avx2` wrappers that call `*_scalar`. Every one of those was admitted
/// to the table on the strength of a NaN-free fixture.
///
/// The gaps are placed to separate the three first-valid rules the CPU uses:
///
/// * `low` starts at index 3 and `close` at index 1, so
///   `fh.max(fl).max(fc)` = 3 while `close.position(!is_nan)` = 1;
/// * `high` carries an INTERIOR NaN at index 3, so the first index at which all
///   three are non-NaN simultaneously is 4, not 3.
///
/// Three different answers from one frame — see
/// `vector_ta::indicators::dispatch::F64FirstValidRule`.
const GAP_ALL_THREE: usize = 4;
const GAP_MAX_OF_FIRSTS: usize = 3;
const GAP_CLOSE_ONLY: usize = 1;

fn gapped_candles(n: usize) -> Candles {
    let base = candles(n);
    let ts = base.timestamp.clone();
    let mut o = base.open.clone();
    let mut h = base.high.clone();
    let mut l = base.low.clone();
    let mut c = base.close.clone();
    let mut v = base.volume.clone();

    // close starts at 1; low starts at 3; high starts at 0 but is NaN at 3.
    c[0] = f64::NAN;
    l[0] = f64::NAN;
    l[1] = f64::NAN;
    l[2] = f64::NAN;
    h[3] = f64::NAN;
    o[0] = f64::NAN;
    v[0] = f64::NAN;

    // A zero close well past every warmup. This store has already shipped
    // 14,240 bars with a zero price, so `natr`'s `close != 0.0` guard and
    // `rocr`'s zero-divisor guard are exercised rather than assumed.
    let z = n / 2;
    c[z] = 0.0;

    Candles::new(ts, o, h, l, c, v)
}

fn compute(id: &str, cd: &Candles, period: i64, kernel: Kernel) -> Option<Vec<f64>> {
    let params = [ParamKV {
        key: "period",
        value: ParamValue::Int(period),
    }];
    let out = compute_cpu(IndicatorComputeRequest {
        indicator_id: id,
        output_id: None,
        data: IndicatorDataRef::Candles {
            candles: cd,
            source: None,
        },
        params: &params,
        kernel,
    })
    .ok()?;
    match out.series {
        IndicatorSeries::F64(v) => Some(v),
        _ => None,
    }
}

/// Bit equality including the NaN mask. `a == b` is false for NaN, and the
/// warmup prefix of nearly every one of these series is NaN, so comparing raw
/// values would pass trivially on the prefix and could hide a real difference.
/// Comparing bits also catches -0.0 vs 0.0, which `==` calls equal and which
/// changes the sign of any later division.
fn first_difference(a: &[f64], b: &[f64]) -> Option<(usize, f64, f64)> {
    if a.len() != b.len() {
        return Some((usize::MAX, a.len() as f64, b.len() as f64));
    }
    for i in 0..a.len() {
        if a[i].to_bits() != b[i].to_bits() {
            return Some((i, a[i], b[i]));
        }
    }
    None
}

/// Indicators whose kernel is written and compiled but which are deliberately
/// NOT registered, because vector-ta's own scalar and AVX CPU implementations
/// disagree for them. Measured, not assumed — see
/// [`withheld_indicators_still_diverge`], which fails when a divergence
/// DISAPPEARS so the entry can be promoted rather than quietly kept withheld.
/// Empty, and that is the point.
///
/// It held `vwap` and `wilders` until 2026-08-10, and
/// `withheld_indicators_still_diverge` below is what emptied it — the test fails
/// in BOTH directions, so when vector-ta fixed the scalar/AVX seed association
/// upstream, the assertion fired and its own message spelled out the four steps.
/// vector-ta had done steps 1 and 4 (registered both in F64_KERNELS, emptied
/// WITHHELD_PENDING_CPU_SELF_CONSISTENCY); this crate had never done 2 and 3.
/// The kernels were sitting written, compiled and unreachable in between.
///
/// Leave the list and the test in place. This is the half of the record that
/// normally rots: a kernel gets disabled, the cause is fixed upstream months
/// later, and nobody notices the work is ready.
const WITHHELD: &[&str] = &[];

#[test]
fn scalar_and_auto_agree_for_every_claimed_indicator() {
    let cd = candles(3000);
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for &id in CLAIMED {
        for &period in PERIODS {
            let Some(scalar) = compute(id, &cd, period, Kernel::Scalar) else {
                continue;
            };
            let Some(auto) = compute(id, &cd, period, Kernel::Auto) else {
                failures.push(format!(
                    "{id} period={period}: Kernel::Scalar produced a series and Kernel::Auto did \
                     not — the lane cannot claim an indicator whose production path fails"
                ));
                continue;
            };
            checked += 1;
            if let Some((i, s, a)) = first_difference(&scalar, &auto) {
                failures.push(format!(
                    "{id} period={period}: Kernel::Scalar and Kernel::Auto differ at index {i} \
                     (scalar={s:?} bits=0x{:016x}, auto={a:?} bits=0x{:016x}). The f64 kernel is \
                     written against the *_scalar reference; if Auto is what production runs, \
                     there is no single CPU answer to be in parity with. Remove {id} from \
                     F64_KERNELS until this is explained.",
                    s.to_bits(),
                    a.to_bits()
                ));
            }
        }
    }

    assert!(checked > 0, "no indicator/period combination was checked");
    assert!(
        failures.is_empty(),
        "{} scalar-vs-auto divergence(s) across {checked} checked combinations:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The same scalar-vs-Auto demand, on bars that carry gaps and a zero price.
///
/// A registered indicator that agrees with itself on smooth data and disagrees
/// on a gapped frame is exactly as unusable as `vwap` and `wilders`: there is
/// still no single CPU answer for the device to be in parity with, and the
/// symptom would surface as a rented-card parity failure reported against a
/// correct kernel.
///
/// The failure text says what to do, because "loosen the tolerance" is the
/// wrong answer here and is the reflex.
#[test]
fn scalar_and_auto_agree_on_gapped_bars_too() {
    let cd = gapped_candles(3000);
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for &id in CLAIMED {
        for &period in PERIODS {
            let Some(scalar) = compute(id, &cd, period, Kernel::Scalar) else {
                continue;
            };
            let Some(auto) = compute(id, &cd, period, Kernel::Auto) else {
                failures.push(format!(
                    "{id} period={period}: Scalar produced a series on gapped bars and Auto did \
                     not"
                ));
                continue;
            };
            checked += 1;
            if let Some((i, s, a)) = first_difference(&scalar, &auto) {
                failures.push(format!(
                    "{id} period={period}: Scalar and Auto differ at index {i} on GAPPED bars \
                     (scalar={s:?} bits=0x{:016x}, auto={a:?} bits=0x{:016x}). Clean bars hid \
                     this. Move {id} to WITHHELD by name — do not widen a tolerance.",
                    s.to_bits(),
                    a.to_bits()
                ));
            }
        }
    }

    assert!(checked > 0, "no combination was checked on gapped bars");
    assert!(
        failures.is_empty(),
        "{} scalar-vs-auto divergence(s) on gapped bars across {checked} combinations:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The three first-valid rules really are three different indices on a real
/// frame shape, and each indicator really uses the one vector-ta declares.
///
/// # Why this test exists
///
/// `first_valid` sets BOTH the NaN warmup prefix AND the seed window. Feeding a
/// kernel the wrong one does not perturb the series — it SHIFTS it, and every
/// value after the seed is a different number. The device engine used to derive
/// ONE index per input SHAPE and hand it to all six high/low/close indicators;
/// three of the six do not use that rule.
///
/// Measured, not asserted from the source: each indicator is run on the clean
/// fixture (every rule gives 0) and on the gapped one, and the SHIFT between
/// the two NaN-prefix lengths is the indicator's first-valid index, because the
/// warmup offset is identical across the two runs (same period, same length).
#[test]
fn each_indicator_uses_the_first_valid_rule_its_table_row_declares() {
    let clean = candles(3000);
    let gapped = gapped_candles(3000);
    const PERIOD: i64 = 14;

    /// Length of the leading run of NaN.
    fn nan_prefix(v: &[f64]) -> usize {
        v.iter().take_while(|x| x.is_nan()).count()
    }

    /// What the gapped fixture must produce.
    enum Want {
        /// The warmup shifts by exactly this many bars, i.e. `first_valid`.
        ShiftedBy(usize),
        /// The whole column is NaN — see `natr` below.
        AllNan,
    }

    let expected: &[(&str, Want)] = &[
        // atr.rs:197-206, willr.rs:300, wclprice.rs:176 — all three non-NaN at
        // the same index. That index is 4, where high is finite again, so the
        // series is finite and the shift is directly measurable.
        ("atr", Want::ShiftedBy(GAP_ALL_THREE)),
        ("willr", Want::ShiftedBy(GAP_ALL_THREE)),
        ("wclprice", Want::ShiftedBy(GAP_ALL_THREE)),
        // adx.rs:201-219 — fh.max(fl).max(fc) = 3. `adx_scalar` seeds `prev_h`
        // from high[3] (NaN) but its accumulators are guarded by `>` tests that
        // a NaN fails, and `prev_h` is overwritten on the next bar, so the NaN
        // does not survive into `atr`. Finite series, measurable shift of 3 —
        // one bar EARLIER than atr on the same frame, which is the whole point.
        ("adx", Want::ShiftedBy(GAP_MAX_OF_FIRSTS)),
        // natr.rs:226-235 uses the SAME rule as adx and gets the same index 3,
        // but `natr_scalar` seeds with `sum_tr = high[first] - low[first]`
        // UNGUARDED. high[3] is NaN, so the seed is NaN, so `atr` is NaN, so
        // every bar of the output is NaN. This is the CPU's real answer on a
        // frame shaped like this and the kernel reproduces it exactly (same
        // seed, same index) — so it is pinned rather than papered over. It also
        // separates natr from atr more sharply than a shift would: on ONE frame
        // atr is finite from bar 4 and natr is entirely NaN.
        ("natr", Want::AllNan),
        // adxr.rs:255-258 — close alone, index 1, two bars earlier than atr.
        ("adxr", Want::ShiftedBy(GAP_CLOSE_ONLY)),
    ];

    // The three answers must actually differ, or the fixture proves nothing.
    assert_ne!(GAP_ALL_THREE, GAP_MAX_OF_FIRSTS);
    assert_ne!(GAP_MAX_OF_FIRSTS, GAP_CLOSE_ONLY);

    let mut failures: Vec<String> = Vec::new();
    for (id, want) in expected {
        let Some(a) = compute(id, &clean, PERIOD, Kernel::Scalar) else {
            failures.push(format!("{id}: no series on the clean fixture"));
            continue;
        };
        let Some(b) = compute(id, &gapped, PERIOD, Kernel::Scalar) else {
            failures.push(format!("{id}: no series on the gapped fixture"));
            continue;
        };
        match want {
            Want::AllNan => {
                if nan_prefix(&b) != b.len() {
                    failures.push(format!(
                        "{id}: expected an entirely-NaN column on the gapped fixture (its \
                         first-valid rule lands on a bar whose high is NaN and its seed is \
                         unguarded), got a NaN prefix of {} out of {} bars",
                        nan_prefix(&b),
                        b.len()
                    ));
                }
            }
            Want::ShiftedBy(want) => {
                let got = nan_prefix(&b) as i64 - nan_prefix(&a) as i64;
                if got != *want as i64 {
                    failures.push(format!(
                        "{id}: warmup shifted by {got} bars between the clean and gapped \
                         fixtures, so its first_valid is {got}; vector-ta's F64_KERNELS row \
                         declares a rule worth {want}. One of the two is wrong, and the device \
                         lane derives the index it sends to the kernel from that row."
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "first-valid rule mismatch:\n{}",
        failures.join("\n")
    );
}

/// Four of the claimed indicators ignore the swept `period` entirely, so their
/// kernels must emit identical rows for every period.
///
/// That is faithful to the CPU, not a defect: `compute_obv_batch` takes
/// `|_params|`, `compute_tsi_batch` reads `long_period`/`short_period` only,
/// and `medprice`/`wclprice` have no period parameter — so `hpc_ta` already
/// emits five byte-identical columns per frame for each of them. It is
/// surprising enough to pin here, so a future dispatcher change that STARTS
/// honouring `period` for one of them fails this test instead of silently
/// making the device lane wrong.
#[test]
fn period_invariant_indicators_really_are_period_invariant() {
    let cd = candles(2000);
    for &id in &["tsi", "obv", "medprice", "wclprice"] {
        let base = compute(id, &cd, 7, Kernel::Scalar)
            .unwrap_or_else(|| panic!("{id}: no series for period 7"));
        for &period in &[21i64, 50, 200] {
            let other = compute(id, &cd, period, Kernel::Scalar)
                .unwrap_or_else(|| panic!("{id}: no series for period {period}"));
            assert!(
                first_difference(&base, &other).is_none(),
                "{id}: period {period} differs from period 7, so it is NOT period-invariant and \
                 its kernel — which ignores periods[r] — is now wrong"
            );
        }
    }
}

/// The withheld pair must still be withheld FOR A REASON, and the reason must
/// still be true.
///
/// This is the half of the record that normally rots: a kernel gets disabled,
/// the cause is fixed upstream months later, and nobody notices that the work
/// is sitting there ready. So this test fails in BOTH directions — if a
/// withheld indicator ever appears in `CLAIMED`, and if its scalar/auto
/// divergence ever goes away.
#[test]
fn withheld_indicators_still_diverge() {
    let cd = candles(3000);
    for &id in WITHHELD {
        assert!(
            !CLAIMED.contains(&id),
            "{id} is both CLAIMED and WITHHELD — pick one"
        );
        let diverged = PERIODS.iter().any(|&period| {
            match (
                compute(id, &cd, period, Kernel::Scalar),
                compute(id, &cd, period, Kernel::Auto),
            ) {
                (Some(s), Some(a)) => first_difference(&s, &a).is_some(),
                _ => false,
            }
        });
        assert!(
            diverged,
            "{id} is withheld because Kernel::Scalar and Kernel::Auto disagreed, and they no \
             longer do. The kernel neoethos_{id}_batch_f64 is already written and compiled — put \
             its F64KernelSpec back in cuda_f64::F64_KERNELS, add it to CLAIMED here, remove it \
             from WITHHELD, and drop it from WITHHELD_PENDING_CPU_SELF_CONSISTENCY."
        );
    }
}

/// Every claimed indicator must actually be reachable the way `hpc_ta` calls
/// it: `output_id: None`, which errors for multi-output indicators and is
/// swallowed at `hpc_ta.rs:291`. An unreachable indicator emits no CPU column,
/// so a kernel for it could never be parity-checked.
#[test]
fn every_claimed_indicator_is_reachable_with_no_output_id() {
    let cd = candles(1500);
    for &id in CLAIMED {
        assert!(
            compute(id, &cd, 21, Kernel::Scalar).is_some(),
            "{id}: compute_cpu with output_id=None produced nothing. hpc_ta drops that silently, \
             so this indicator contributes no CPU column and must not be in the device table."
        );
    }
}

/// The duplicated list above must match vector-ta's real table. Only compiled
/// when the CUDA feature is on, which is exactly when the real table exists.
#[cfg(feature = "gpu-cuda")]
#[test]
fn claimed_list_matches_vector_ta_f64_kernels() {
    use vector_ta::indicators::dispatch::cuda_f64::F64_KERNELS;
    let mut real: Vec<&str> = F64_KERNELS.iter().map(|s| s.indicator_id).collect();
    let mut mine: Vec<&str> = CLAIMED.to_vec();
    real.sort_unstable();
    mine.sort_unstable();
    assert_eq!(
        mine, real,
        "the CLAIMED list in this test has drifted from cuda_f64::F64_KERNELS"
    );
}
