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

use vector_ta::indicators::damiani_volatmeter::{
    DamianiVolatmeterBatchBuilder, DamianiVolatmeterInput, DamianiVolatmeterParams,
    damiani_volatmeter_into_slice, damiani_volatmeter_with_kernel,
};
use vector_ta::indicators::dispatch::{
    IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, ParamKV, ParamValue, compute_cpu,
};
use vector_ta::indicators::moving_averages::sma::{
    SmaBatchRange, SmaInput, SmaParams, SmaStream, sma_batch_with_kernel, sma_with_kernel,
};
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;
use vector_ta::utilities::helpers::alloc_with_nan_prefix;

/// The original open-source Pine v6 implementation updates its anchor with
/// this recurrence on every bar of a bullish segment:
///
/// `_initial := _initial * (1 + (1 - exp(-exp_rate * bars_elapsed)))`
///
/// Source (published 2025-04-18, MPL-2.0):
/// https://www.tradingview.com/script/CDb3oR6A-Exponential-Trend-AlgoAlpha/
///
/// The multiplier tends to two, so a sufficiently long segment necessarily
/// exceeds the finite f64 range.  The real 200,000-row EURUSD Vortex fixture
/// reaches `+inf` at row 7,475 and never recovers because the crossover logic
/// rejects a non-finite anchor.  Matching that value on CUDA would reproduce a
/// defect, not establish mathematical truth.
#[test]
fn production_vocabulary_excludes_the_unbounded_exponential_trend_recurrence() {
    let mut anchor = 1.0f64;
    let exp_rate = 0.00003f64;
    let first_non_finite = (0usize..20_000).find(|&bars_elapsed| {
        let multiplier = 1.0 + (1.0 - (-exp_rate * bars_elapsed as f64).exp());
        anchor *= multiplier;
        !anchor.is_finite()
    });

    assert!(
        first_non_finite.is_some(),
        "the published recurrence fixture no longer demonstrates its finite-range defect; \
         re-review the source before changing the production exclusion"
    );
    assert!(
        !neoethos_data::core::all_indicators::ALL_INDICATORS.contains(&"exponential_trend"),
        "exponential_trend is still in the production feature vocabulary even though its \
         published recurrence exceeds f64 on finite input; do not clamp or treat CPU/CUDA \
         agreement on +inf as correctness"
    );
}

/// Every indicator THIS CRATE's f64 device lane can launch, plus the moving
/// averages and volatility ids kept ahead of it.
///
/// # This is NOT a mirror of `cuda_f64::F64_KERNELS`, and it never was
///
/// It used to say it was, and `claimed_list_matches_vector_ta_f64_kernels`
/// below used to assert equality. That assertion could not have passed since
/// vector-ta's f64 lane grew past its first two batches: the real table now
/// carries 338 rows against this list's ~29, so the test was a red gate waiting
/// for the first `--features gpu-cuda` build to trip it, on this machine that
/// build cannot even be attempted (no nvcc, no driver), and no card build has
/// run since. The equality was corrected to the two containments that are
/// actually true and actually load-bearing — see that test.
///
/// What this list is FOR: `cuda_f64::F64_KERNELS` lives behind vector-ta's
/// `cuda-build-native` feature, and the whole value of the tests in this file
/// is that they run on a machine with no CUDA at all. So the ids this crate
/// actually launches are named here in plain text, and the containments below
/// tie them back to the real table whenever the feature IS enabled.
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
    let ts: Vec<i64> = (0..n as i64)
        .map(|i| 1_600_000_000_000 + i * 300_000)
        .collect();
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
/// `wilders` at its seed bar — both since fixed upstream, in the CPU) are
/// therefore the floor, not the ceiling: most ids in [`CLAIMED`] have genuinely
/// distinct AVX bodies rather than `*_avx2` wrappers that call `*_scalar`, and
/// every one of them was admitted to the table on the strength of a NaN-free
/// fixture.
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
    compute_with_params(id, cd, &params, kernel)
}

fn compute_with_params(
    id: &str,
    cd: &Candles,
    params: &[ParamKV],
    kernel: Kernel,
) -> Option<Vec<f64>> {
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

/// Length of the leading run of NaN values.
fn nan_prefix(values: &[f64]) -> usize {
    values.iter().take_while(|value| value.is_nan()).count()
}

/// The SMA is the arithmetic mean of the current window. Its numerical error
/// must therefore be bounded by the window, not by the total history processed
/// before that window.
///
/// Every input below is exactly representable as `1 + ticks * 2^-52`, so the
/// exact real-number window sum is `period + sum(ticks) * 2^-52`. Keeping the
/// tick sum in `i64` gives this test an independent oracle that shares neither
/// the production rolling recurrence nor its batch prefix sums. The 250k-bar
/// run is deliberately long enough to expose both old defects: rolling error
/// accumulated with history, while subtracting two large prefix sums lost low
/// bits through cancellation.
#[test]
fn long_run_sma_stays_within_one_ulp_of_exact_window_mean() {
    const LEN: usize = 250_000;
    let ticks: Vec<i64> = (0..LEN)
        .map(|i| {
            let i = i as u64;
            ((i * 17 + (i / 97) * 31 + ((i * i) >> 7)) % 257) as i64
        })
        .collect();
    let data: Vec<f64> = ticks
        .iter()
        .map(|&tick| 1.0 + tick as f64 * f64::EPSILON)
        .collect();

    for period in [7usize, 21, 50, 100, 200] {
        let input = SmaInput::from_slice(
            &data,
            SmaParams {
                period: Some(period),
            },
        );
        let scalar = sma_with_kernel(&input, Kernel::Scalar)
            .expect("scalar SMA on exact-tick fixture")
            .values;
        let batch = sma_batch_with_kernel(
            &data,
            &SmaBatchRange {
                period: (period, period, 0),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch SMA on exact-tick fixture");
        let batch = batch
            .values_for(&SmaParams {
                period: Some(period),
            })
            .expect("one requested batch row");
        let mut stream = SmaStream::try_new(SmaParams {
            period: Some(period),
        })
        .expect("stream SMA on exact-tick fixture");

        let mut exact_tick_sum: i64 = ticks[..period].iter().sum();
        for row in 0..LEN {
            let streamed = stream.update(data[row]);
            if row + 1 < period {
                assert!(scalar[row].is_nan(), "scalar warmup at row {row}");
                assert!(batch[row].is_nan(), "batch warmup at row {row}");
                assert!(streamed.is_none(), "stream warmup at row {row}");
                continue;
            }
            if row >= period {
                exact_tick_sum += ticks[row] - ticks[row - period];
            }
            let expected = 1.0 + (exact_tick_sum as f64 / period as f64) * f64::EPSILON;
            let streamed = streamed.expect("stream value after warmup");

            for (lane, actual) in [
                ("scalar", scalar[row]),
                ("batch", batch[row]),
                ("stream", streamed),
            ] {
                let ulps = actual.to_bits().abs_diff(expected.to_bits());
                assert!(
                    ulps <= 1,
                    "SMA({period}) {lane} accumulated history-dependent error at row {row}: \
                     actual={actual:.17e} expected={expected:.17e} ({ulps} ulp)"
                );
            }
            assert_eq!(
                scalar[row].to_bits(),
                batch[row].to_bits(),
                "SMA({period}) scalar and batch use different arithmetic at row {row}"
            );
            assert_eq!(
                scalar[row].to_bits(),
                streamed.to_bits(),
                "SMA({period}) scalar and stream use different arithmetic at row {row}"
            );
        }
    }
}

/// A caller-owned output slice is storage, not an implicit input to the
/// indicator. Damiani's lag suppressor reads earlier `vol` cells, so every
/// cell it can observe must be initialized by the implementation before the
/// recurrence starts. This specifically catches the old
/// `alloc_with_nan_prefix`/`into_slice` behaviour where the first lag reads
/// whatever bytes happened to be in the destination tail.
#[test]
fn damiani_output_is_independent_of_destination_contents() {
    let cd = candles(256);
    let input = DamianiVolatmeterInput::from_slice(&cd.close, DamianiVolatmeterParams::default());

    let mut zero_vol = vec![0.0; cd.close.len()];
    let mut zero_anti = vec![0.0; cd.close.len()];
    damiani_volatmeter_into_slice(&mut zero_vol, &mut zero_anti, &input, Kernel::Scalar)
        .expect("Damiani scalar run with zero-filled destination");

    let mut poisoned_vol = vec![12_345.25; cd.close.len()];
    let mut poisoned_anti = vec![-98_765.5; cd.close.len()];
    damiani_volatmeter_into_slice(
        &mut poisoned_vol,
        &mut poisoned_anti,
        &input,
        Kernel::Scalar,
    )
    .expect("Damiani scalar run with nonzero destination");

    assert_eq!(
        first_difference(&zero_vol, &poisoned_vol),
        None,
        "Damiani vol depends on bytes supplied by the caller instead of only on market input"
    );
    assert_eq!(
        first_difference(&zero_anti, &poisoned_anti),
        None,
        "Damiani anti depends on bytes supplied by the caller instead of only on market input"
    );
}

#[test]
fn vector_ta_safe_output_allocator_initializes_every_element() {
    let values = alloc_with_nan_prefix(16, 3);
    assert_eq!(values.len(), 16);
    assert!(
        values.iter().all(|value| value.is_nan()),
        "a safe Vec<f64> must not expose an uninitialized or poison tail; got {values:?}"
    );
}

/// Independent, deliberately simple implementation of the published Damiani
/// equations for a single price slice (`high == low == close`):
///
/// * LineP = ATR(vis) / ATR(sed) + (LineP[1] - LineP[3]) / 2
/// * LineM = threshold - StdDev(vis) / StdDev(sed)
///
/// Formula sources:
/// https://www.mql5.com/en/code/21700
/// https://vectoralpha.dev/projects/ta/indicators/damiani_volatmeter/
///
/// This is intentionally O(n * window) and shares no production helper. It is
/// a mathematical oracle for a tiny fixture, not an optimized implementation.
fn published_damiani_reference(
    prices: &[f64],
    vis_atr: usize,
    vis_std: usize,
    sed_atr: usize,
    sed_std: usize,
    threshold: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = prices.len();
    let needed = *[vis_atr, vis_std, sed_atr, sed_std, 3]
        .iter()
        .max()
        .unwrap();

    let mut tr = vec![0.0; n];
    for i in 1..n {
        tr[i] = (prices[i] - prices[i - 1]).abs();
    }

    fn wilder_atr(tr: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; tr.len()];
        let mut seed = 0.0;
        for (i, &value) in tr.iter().enumerate() {
            if i < period {
                seed += value;
                if i + 1 == period {
                    out[i] = seed / period as f64;
                }
            } else {
                out[i] = ((period as f64 - 1.0) * out[i - 1] + value) / period as f64;
            }
        }
        out
    }

    fn population_std(window: &[f64]) -> f64 {
        let mean = window.iter().copied().sum::<f64>() / window.len() as f64;
        let variance = window
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / window.len() as f64;
        variance.max(0.0).sqrt()
    }

    let atr_vis = wilder_atr(&tr, vis_atr);
    let atr_sed = wilder_atr(&tr, sed_atr);
    let mut vol = vec![f64::NAN; n];
    let mut anti = vec![f64::NAN; n];

    for i in needed..n {
        let p1 = vol
            .get(i.wrapping_sub(1))
            .copied()
            .filter(|x| x.is_finite())
            .unwrap_or(0.0);
        let p3 = vol
            .get(i.wrapping_sub(3))
            .copied()
            .filter(|x| x.is_finite())
            .unwrap_or(0.0);
        let sed = if atr_sed[i] != 0.0 {
            atr_sed[i]
        } else {
            atr_sed[i] + f64::EPSILON
        };
        vol[i] = atr_vis[i] / sed + 0.5 * (p1 - p3);

        let std_vis = population_std(&prices[i + 1 - vis_std..=i]);
        let std_sed = population_std(&prices[i + 1 - sed_std..=i]);
        let std_sed_safe = if std_sed != 0.0 {
            std_sed
        } else {
            std_sed + f64::EPSILON
        };
        anti[i] = threshold - std_vis / std_sed_safe;
    }

    (vol, anti)
}

#[test]
fn damiani_matches_published_formula_and_batch_contract() {
    let prices = [
        100.0, 101.5, 100.25, 103.0, 102.0, 104.75, 106.0, 105.25, 108.5, 107.0, 109.75, 111.0,
    ];
    let params = DamianiVolatmeterParams {
        vis_atr: Some(2),
        vis_std: Some(3),
        sed_atr: Some(4),
        sed_std: Some(5),
        threshold: Some(1.4),
    };
    let input = DamianiVolatmeterInput::from_slice(&prices, params.clone());
    let actual = damiani_volatmeter_with_kernel(&input, Kernel::Scalar)
        .expect("Damiani scalar formula fixture");
    let (expected_vol, expected_anti) = published_damiani_reference(&prices, 2, 3, 4, 5, 1.4);

    for i in 0..prices.len() {
        assert_eq!(
            actual.vol[i].is_nan(),
            expected_vol[i].is_nan(),
            "vol validity differs at {i}"
        );
        assert_eq!(
            actual.anti[i].is_nan(),
            expected_anti[i].is_nan(),
            "anti validity differs at {i}"
        );
        if expected_vol[i].is_finite() {
            assert!(
                (actual.vol[i] - expected_vol[i]).abs() <= 1e-12,
                "vol formula differs at {i}: actual={} expected={}",
                actual.vol[i],
                expected_vol[i]
            );
            assert!(
                // The independent oracle uses a two-pass deviation sum while
                // production maintains rolling sum/sum_sq state. Their
                // algebra is identical but their f64 association is not.
                (actual.anti[i] - expected_anti[i]).abs() <= 1e-10,
                "anti formula differs at {i}: actual={} expected={}",
                actual.anti[i],
                expected_anti[i]
            );
        }
    }

    let batch = DamianiVolatmeterBatchBuilder::new()
        .vis_atr_range(2, 2, 0)
        .vis_std_range(3, 3, 0)
        .sed_atr_range(4, 4, 0)
        .sed_std_range(5, 5, 0)
        .threshold_range(1.4, 1.4, 0.0)
        .kernel(Kernel::ScalarBatch)
        .apply_slice(&prices)
        .expect("Damiani batch formula fixture");
    let batch_vol = batch.vol_for(&params).expect("Damiani batch vol row");
    let batch_anti = batch.anti_for(&params).expect("Damiani batch anti row");
    assert_eq!(first_difference(&actual.vol, batch_vol), None);
    assert_eq!(first_difference(&actual.anti, batch_anti), None);
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

/// Three of the claimed indicators ignore the swept `period` entirely, so their
/// kernels must emit identical rows for every period.
///
/// That is faithful to the CPU, not a defect: `compute_obv_batch` takes
/// `|_params|`, and `medprice`/`wclprice` have no period parameter — so `hpc_ta`
/// already emits five byte-identical columns per frame for each of them. It is
/// surprising enough to pin here, so a future dispatcher change that STARTS
/// honouring `period` for one of them fails this test instead of silently
/// making the device lane wrong.
#[test]
fn period_invariant_indicators_really_are_period_invariant() {
    let cd = candles(2000);
    for &id in &["obv", "medprice", "wclprice"] {
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

/// TSI has two independent named windows. NeoEthos' period sweep anchors the
/// long window and scales the short one with the default 25:13 relation; a
/// generic `period` parameter is not the production request. Pin the named
/// tuples and their structural warmup so the old five-copies-of-25/13 path
/// cannot return unnoticed.
#[test]
fn tsi_coupled_window_rows_are_distinct_and_have_named_warmups() {
    let cd = candles(2000);
    let cases = [(7i64, 4i64), (21, 11), (50, 26)];
    let mut rows = Vec::new();
    for &(long, short) in &cases {
        let params = [
            ParamKV {
                key: "long_period",
                value: ParamValue::Int(long),
            },
            ParamKV {
                key: "short_period",
                value: ParamValue::Int(short),
            },
        ];
        let row = compute_with_params("tsi", &cd, &params, Kernel::Scalar)
            .unwrap_or_else(|| panic!("TSI {long}/{short} did not produce a row"));
        assert_eq!(
            nan_prefix(&row),
            (long + short) as usize,
            "TSI {long}/{short} warmup must be first_valid + long + short"
        );
        rows.push(row);
    }
    for pair in rows.windows(2) {
        assert!(
            first_difference(&pair[0], &pair[1]).is_some(),
            "two different named TSI window pairs produced byte-identical rows"
        );
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

/// The two containments that are true, in place of the equality that was not.
///
/// # What changed and why
///
/// This test used to assert `CLAIMED == F64_KERNELS`. That is false by two
/// orders of magnitude — the real table carries 338 rows and this list ~29 —
/// and it had been false for as long as vector-ta's f64 lane has been growing
/// past its first batches. Nothing caught it because the test is
/// `gpu-cuda`-gated and no card build has run since.
///
/// The equality was never the property that mattered. Two containments are:
///
/// 1. **Every id in `GPU_SWEEP_SPECS` must be in `CLAIMED`.** `GPU_SWEEP_SPECS`
///    is what `hpc_ta` actually launches, so an id there that this file does
///    not exercise is an id whose CPU oracle has never been checked for
///    self-consistency — which is precisely the hazard the whole file exists
///    for. This direction is the one that would have caught `vwap` being
///    claimed without being measured.
/// 2. **Every id in `CLAIMED` must have a row in `F64_KERNELS`.** An id checked
///    here that vector-ta has no f64 kernel for is a test spending time on
///    something no device will ever run — and, worse, reads as coverage.
///
/// The counts of both sets are printed rather than asserted. A number in a test
/// message ages honestly; a number in an `assert_eq!` becomes a chore.
#[cfg(feature = "gpu-cuda")]
#[test]
fn claimed_list_is_consistent_with_vector_ta_f64_kernels() {
    use neoethos_data::core::gpu_indicators::GPU_SWEEP_SPECS;
    use vector_ta::indicators::dispatch::cuda_f64::F64_KERNELS;

    let real: Vec<&str> = F64_KERNELS.iter().map(|s| s.indicator_id).collect();

    let unmeasured: Vec<&str> = GPU_SWEEP_SPECS
        .iter()
        .map(|s| s.id)
        .filter(|id| !CLAIMED.contains(id))
        .collect();
    assert!(
        unmeasured.is_empty(),
        "hpc_ta launches these on the device but this file never measures their CPU oracle for \
         scalar-vs-auto self-consistency: {unmeasured:?}. Add them to CLAIMED. \
         (CLAIMED={}, GPU_SWEEP_SPECS={}, F64_KERNELS={})",
        CLAIMED.len(),
        GPU_SWEEP_SPECS.len(),
        real.len()
    );

    let phantom: Vec<&str> = CLAIMED
        .iter()
        .copied()
        .filter(|id| !real.contains(id))
        .collect();
    assert!(
        phantom.is_empty(),
        "CLAIMED names ids with no row in cuda_f64::F64_KERNELS, so no device will ever run them \
         and their presence here reads as coverage it is not: {phantom:?}. \
         (CLAIMED={}, GPU_SWEEP_SPECS={}, F64_KERNELS={})",
        CLAIMED.len(),
        GPU_SWEEP_SPECS.len(),
        real.len()
    );

    eprintln!(
        "f64 lane sets: CLAIMED={} (measured card-lessly here), GPU_SWEEP_SPECS={} (launched by \
         hpc_ta), F64_KERNELS={} (registered by vector-ta). The last is deliberately much larger: \
         most of it is base-vocabulary indicators this crate has no device lane for at all.",
        CLAIMED.len(),
        GPU_SWEEP_SPECS.len(),
        real.len()
    );
}
