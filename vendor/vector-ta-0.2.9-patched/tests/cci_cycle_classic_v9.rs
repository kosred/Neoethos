use vector_ta::indicators::cci_cycle::{
    CciCycleBatchBuilder, CciCycleError, CciCycleInput, CciCycleParams, CciCycleStream, cci_cycle,
    cci_cycle_into_slice,
};
use vector_ta::indicators::dispatch::{
    IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV, ParamValue,
    compute_cpu_batch_strict,
};
use vector_ta::utilities::enums::Kernel;

fn sample_close(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            100.0
                + x * 0.017
                + (x * 0.173).sin() * 2.4
                + (x * 0.047).cos() * 0.9
                + if i >= n / 3 { 1.25 } else { 0.0 }
                - if i >= (2 * n) / 3 { 2.1 } else { 0.0 }
        })
        .collect()
}

fn sma_seeded_average(values: &[f64], period: usize, alpha: f64) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    let mut seed_sum = 0.0;
    let mut seed_count = 0usize;
    let mut state = f64::NAN;

    for (index, &value) in values.iter().enumerate() {
        if value.is_finite() {
            if seed_count < period {
                seed_sum += value;
                seed_count += 1;
                if seed_count == period {
                    state = seed_sum / period as f64;
                }
            } else {
                state += alpha * (value - state);
            }
        }
        if seed_count == period {
            out[index] = state;
        }
    }

    out
}

fn creator_segment(close: &[f64], length: usize, factor: f64) -> Vec<f64> {
    let mut cci = vec![f64::NAN; close.len()];
    for index in length - 1..close.len() {
        let window = &close[index + 1 - length..=index];
        let mean = window.iter().sum::<f64>() / length as f64;
        let deviation =
            window.iter().map(|value| (value - mean).abs()).sum::<f64>() / length as f64;
        if deviation > 0.0 {
            cci[index] = (close[index] - mean) / (0.015 * deviation);
        }
    }

    let half = length / 2;
    let ema_short = sma_seeded_average(&cci, half, 2.0 / (half as f64 + 1.0));
    let ema_long = sma_seeded_average(&cci, length, 2.0 / (length as f64 + 1.0));
    let de = ema_short
        .iter()
        .zip(&ema_long)
        .map(|(&short, &long)| {
            if short.is_finite() && long.is_finite() {
                short + short - long
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();

    let rma_length = ((length as f64).sqrt().round() as usize).max(1);
    let ccis = sma_seeded_average(&de, rma_length, 1.0 / rma_length as f64);

    let mut f1 = vec![0.0; close.len()];
    let mut pf = vec![0.0; close.len()];
    for index in 0..close.len() {
        let start = index.saturating_sub(length - 1);
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for &value in &ccis[start..=index] {
            if value.is_finite() {
                low = low.min(value);
                high = high.max(value);
            }
        }
        let previous = if index == 0 { 0.0 } else { f1[index - 1] };
        f1[index] = if ccis[index].is_finite() && high.is_finite() && high > low {
            (ccis[index] - low) / (high - low) * 100.0
        } else {
            previous
        };
        pf[index] = if index == 0 {
            f1[index]
        } else {
            pf[index - 1] + factor * (f1[index] - pf[index - 1])
        };
    }

    let mut f2 = vec![0.0; close.len()];
    let mut pff = vec![0.0; close.len()];
    for index in 0..close.len() {
        let start = index.saturating_sub(length - 1);
        let window = &pf[start..=index];
        let low = window.iter().copied().fold(f64::INFINITY, f64::min);
        let high = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let previous = if index == 0 { 0.0 } else { f2[index - 1] };
        f2[index] = if high > low {
            (pf[index] - low) / (high - low) * 100.0
        } else {
            previous
        };
        pff[index] = if index == 0 {
            f2[index]
        } else {
            pff[index - 1] + factor * (f2[index] - pff[index - 1])
        };
    }
    pff
}

fn creator_v9_local_with_segment_reset(close: &[f64], length: usize, factor: f64) -> Vec<f64> {
    assert!(length >= 2);
    assert!(factor.is_finite());
    let mut out = vec![f64::NAN; close.len()];
    let mut start = 0usize;
    while start < close.len() {
        while start < close.len() && !close[start].is_finite() {
            start += 1;
        }
        if start == close.len() {
            break;
        }
        let mut end = start;
        while end < close.len() && close[end].is_finite() {
            end += 1;
        }
        out[start..end].copy_from_slice(&creator_segment(&close[start..end], length, factor));
        start = end;
    }
    out
}

fn assert_series_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if want.is_nan() {
            assert!(got.is_nan(), "index {index}: got {got}, expected NaN");
        } else {
            assert!(
                got.is_finite() && (got - want).abs() <= tolerance,
                "index {index}: got {got:?}, expected {want:?}, diff={:?}",
                (got - want).abs()
            );
        }
    }
}

#[test]
fn creator_formula_uses_floor_half_sma_seeds_and_exact_factors() {
    let close = sample_close(240);
    for length in [7usize, 10, 21] {
        for factor in [0.0, 0.5, 1.0] {
            let expected = creator_v9_local_with_segment_reset(&close, length, factor);
            let input = CciCycleInput::from_slice(
                &close,
                CciCycleParams {
                    length: Some(length),
                    factor: Some(factor),
                },
            );
            let actual = cci_cycle(&input).expect("creator-valid input").values;
            assert_series_close(&actual, &expected, 1e-11);
        }
    }
}

#[test]
fn startup_flat_holes_and_factor_zero_are_fail_closed() {
    let flat = vec![42.0; 96];
    let flat_actual = cci_cycle(&CciCycleInput::from_slice(
        &flat,
        CciCycleParams {
            length: Some(7),
            factor: Some(0.5),
        },
    ))
    .unwrap()
    .values;
    assert_eq!(flat_actual, vec![0.0; flat.len()]);

    let mut close = sample_close(220);
    close[110] = f64::NAN;
    let expected = creator_v9_local_with_segment_reset(&close, 7, 0.5);
    let actual = cci_cycle(&CciCycleInput::from_slice(
        &close,
        CciCycleParams {
            length: Some(7),
            factor: Some(0.5),
        },
    ))
    .unwrap()
    .values;
    assert_series_close(&actual, &expected, 1e-11);
    assert!(actual[110].is_nan());
    assert_eq!(actual[111], 0.0, "finite segment must restart at zero");

    let zero_factor = cci_cycle(&CciCycleInput::from_slice(
        &close,
        CciCycleParams {
            length: Some(7),
            factor: Some(0.0),
        },
    ))
    .unwrap()
    .values;
    for (index, (&source, &value)) in close.iter().zip(&zero_factor).enumerate() {
        if source.is_finite() {
            assert_eq!(value, 0.0, "factor=0 must freeze at index {index}");
        } else {
            assert!(value.is_nan(), "hole must remain NaN at index {index}");
        }
    }
}

#[test]
fn direct_into_batch_dispatch_and_stream_share_one_v9_contract() {
    let mut close = sample_close(192);
    close[93] = f64::NAN;
    let params = CciCycleParams {
        length: Some(7),
        factor: Some(0.5),
    };
    let expected = creator_v9_local_with_segment_reset(&close, 7, 0.5);
    let input = CciCycleInput::from_slice(&close, params.clone());

    let direct = cci_cycle(&input).unwrap().values;
    assert_series_close(&direct, &expected, 1e-11);

    let mut into = vec![f64::NAN; close.len()];
    cci_cycle_into_slice(&mut into, &input, Kernel::Scalar).unwrap();
    assert_series_close(&into, &expected, 1e-11);

    let batch = CciCycleBatchBuilder::new()
        .length_range(7, 7, 0)
        .factor_range(0.5, 0.5, 0.0)
        .kernel(Kernel::ScalarBatch)
        .apply_slice(&close)
        .unwrap();
    assert_eq!((batch.rows, batch.cols), (1, close.len()));
    assert_series_close(&batch.values, &expected, 1e-11);

    let param_values = [
        ParamKV {
            key: "length",
            value: ParamValue::Int(7),
        },
        ParamKV {
            key: "factor",
            value: ParamValue::Float(0.5),
        },
    ];
    let combos = [IndicatorParamSet {
        params: &param_values,
    }];
    let dispatched = compute_cpu_batch_strict(IndicatorBatchRequest {
        indicator_id: "cci_cycle",
        output_id: Some("value"),
        data: IndicatorDataRef::Slice { values: &close },
        combos: &combos,
        kernel: Kernel::Scalar,
    })
    .unwrap();
    assert_series_close(dispatched.values_f64.as_deref().unwrap(), &expected, 1e-11);

    let mut stream = CciCycleStream::try_new(params).unwrap();
    let streamed = close
        .iter()
        .map(|&value| stream.update(value).unwrap_or(f64::NAN))
        .collect::<Vec<_>>();
    assert_series_close(&streamed, &expected, 1e-11);
}

#[test]
fn length_one_is_not_a_classic_v9_indicator() {
    let close = sample_close(64);
    let error = cci_cycle(&CciCycleInput::from_slice(
        &close,
        CciCycleParams {
            length: Some(1),
            factor: Some(0.5),
        },
    ))
    .unwrap_err();
    assert!(matches!(
        error,
        CciCycleError::InvalidPeriod { period: 1, .. }
    ));
}
