#[path = "../src/indicators/half_causal_estimator_stable_math.rs"]
mod stable_math;

use stable_math::{NeumaierSum, StablePopulationMoments};
use std::collections::VecDeque;

const DATA_PERIOD: usize = 5;
const FILTER_LENGTH: usize = 20;
const FUTURE_LENGTH: usize = FILTER_LENGTH - 1;
const WINDOW_SIZE: usize = FILTER_LENGTH * 2 - 1;

#[derive(Clone)]
struct Bucket {
    values: Vec<f64>,
    data_period: usize,
    next: usize,
    count: usize,
    moments: StablePopulationMoments,
}

impl Bucket {
    fn new(data_period: usize) -> Self {
        Self {
            values: vec![0.0; data_period],
            data_period,
            next: 0,
            count: 0,
            moments: StablePopulationMoments::default(),
        }
    }

    fn add(&mut self, value: f64) {
        if self.data_period == 0 {
            self.count += 1;
            self.moments.add(value);
            return;
        }
        self.values[self.next] = value;
        if self.count < self.data_period {
            self.count += 1;
        }
        self.next = (self.next + 1) % self.data_period;

        let mut moments = StablePopulationMoments::default();
        let start = if self.count == self.data_period {
            self.next
        } else {
            0
        };
        for offset in 0..self.count {
            moments.add(self.values[(start + offset) % self.data_period]);
        }
        self.moments = moments;
    }

    fn value_and_confidence(&self) -> Option<(f64, f64)> {
        Some((
            self.moments.mean()?,
            self.moments.creator_inverse_cv(1.0).max(0.0),
        ))
    }
}

fn coefficients() -> [f64; WINDOW_SIZE] {
    let mut coefficients = [0.0; WINDOW_SIZE];
    let center = (WINDOW_SIZE - 1) as f64 * 0.5;
    let mut normalization = 0.0;
    for (index, coefficient) in coefficients.iter_mut().enumerate() {
        let centered = index as f64 - center;
        let ratio = centered / 20.0;
        let weight = if ratio.abs() <= 1.0 {
            0.75 * (1.0 - ratio * ratio)
        } else {
            0.0
        };
        *coefficient = weight;
        normalization += weight;
    }
    for coefficient in &mut coefficients {
        *coefficient /= normalization;
    }
    coefficients
}

fn collect_rescanned_future(
    slot: usize,
    slots_per_day: usize,
    mut lookup: impl FnMut(usize) -> Option<(f64, f64)>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut future_values = Vec::with_capacity(FUTURE_LENGTH);
    let mut future_confidence = Vec::with_capacity(FUTURE_LENGTH);
    let mut key = slot;
    while future_values.len() < FUTURE_LENGTH {
        let mut next = None;
        for offset in 1..=slots_per_day {
            let candidate = (key + offset) % slots_per_day;
            if let Some(value) = lookup(candidate) {
                next = Some((candidate, value));
                break;
            }
        }
        let (next_key, (mean, confidence)) = next?;
        key = next_key;
        future_values.insert(0, mean);
        future_confidence.insert(0, confidence);
    }
    Some((future_values, future_confidence))
}

#[derive(Clone, Default)]
struct PineFutureCache {
    values: VecDeque<f64>,
    confidence: VecDeque<f64>,
    window_key: Option<usize>,
}

impl PineFutureCache {
    fn next_valid(buckets: &[Bucket], start_key: usize) -> Option<(usize, f64, f64)> {
        for offset in 1..=buckets.len() {
            let key = (start_key + offset) % buckets.len();
            if let Some((value, confidence)) = buckets[key].value_and_confidence() {
                return Some((key, value, confidence));
            }
        }
        None
    }

    fn initialize(&mut self, buckets: &[Bucket], current_key: usize) -> Option<()> {
        self.values.clear();
        self.confidence.clear();
        let mut key = current_key;
        while self.values.len() < FUTURE_LENGTH {
            let (next_key, value, confidence) = Self::next_valid(buckets, key)?;
            key = next_key;
            self.values.push_front(value);
            self.confidence.push_front(confidence);
        }
        self.window_key = Some(key);
        Some(())
    }

    fn maintain(&mut self, buckets: &[Bucket]) -> Option<()> {
        if self.values.len() != FUTURE_LENGTH || self.confidence.len() != FUTURE_LENGTH {
            return None;
        }
        let key = self.window_key?;
        let (next_key, value, confidence) = Self::next_valid(buckets, key)?;
        let _ = self.values.pop_back();
        let _ = self.confidence.pop_back();
        self.values.push_front(value);
        self.confidence.push_front(confidence);
        self.window_key = Some(next_key);
        Some(())
    }
}

fn estimate_from_future(
    data: &[f64],
    index: usize,
    coefficients: &[f64; WINDOW_SIZE],
    future_values: &[f64],
    future_confidence: &[f64],
) -> Option<f64> {
    if future_values.len() != FUTURE_LENGTH || future_confidence.len() != FUTURE_LENGTH {
        return None;
    }

    let mut sum = NeumaierSum::default();
    for future in 0..FUTURE_LENGTH {
        sum.add_weighted(
            future_values[future],
            future_confidence[future],
            coefficients[future],
        );
    }
    for causal in 0..FILTER_LENGTH {
        let value = data[index - causal];
        if !value.is_finite() {
            return None;
        }
        let confidence = if causal == 0 {
            1.0
        } else {
            2.0 - future_confidence[FUTURE_LENGTH - causal]
        };
        sum.add_weighted(value, confidence, coefficients[FUTURE_LENGTH + causal]);
    }
    let estimate = sum.total();
    estimate.is_finite().then_some(estimate)
}

fn pine_cached_authority(
    data: &[f64],
    slots: &[usize],
    session_starts: &[bool],
    slots_per_day: usize,
    data_period: usize,
) -> Vec<f64> {
    assert_eq!(data.len(), slots.len());
    assert_eq!(data.len(), session_starts.len());
    let coefficients = coefficients();
    let mut buckets = vec![Bucket::new(data_period); slots_per_day];
    let mut output = vec![f64::NAN; data.len()];
    let mut ready = false;
    let mut future = PineFutureCache::default();

    for index in 0..data.len() {
        let slot = slots[index];
        if !ready && index > WINDOW_SIZE && session_starts[index] {
            ready = true;
        }
        if ready {
            let cache_ready = if session_starts[index] {
                future.initialize(&buckets, slot)
            } else {
                future.maintain(&buckets)
            };
            if cache_ready.is_some() && index + 1 >= FILTER_LENGTH {
                if let Some(value) = estimate_from_future(
                    data,
                    index,
                    &coefficients,
                    future.values.make_contiguous(),
                    future.confidence.make_contiguous(),
                ) {
                    output[index] = value;
                }
            }
        }
        if data[index].is_finite() {
            buckets[slot].add(data[index]);
        }
    }
    output
}

fn rescanned_authority(
    data: &[f64],
    slots: &[usize],
    session_starts: &[bool],
    slots_per_day: usize,
) -> Vec<f64> {
    let coefficients = coefficients();
    let mut buckets = vec![Bucket::new(DATA_PERIOD); slots_per_day];
    let mut output = vec![f64::NAN; data.len()];
    let mut ready = false;
    for index in 0..data.len() {
        let slot = slots[index];
        if !ready && index > WINDOW_SIZE && session_starts[index] {
            ready = true;
        }
        if ready && index + 1 >= FILTER_LENGTH {
            if let Some((future_values, future_confidence)) =
                collect_rescanned_future(slot, slots_per_day, |future_slot| {
                    buckets[future_slot].value_and_confidence()
                })
            {
                if let Some(value) = estimate_from_future(
                    data,
                    index,
                    &coefficients,
                    &future_values,
                    &future_confidence,
                ) {
                    output[index] = value;
                }
            }
        }
        if data[index].is_finite() {
            buckets[slot].add(data[index]);
        }
    }
    output
}

fn reverse_history_rescan_oracle(data: &[f64], slots_per_day: usize) -> Vec<f64> {
    let coefficients = coefficients();
    let mut output = vec![f64::NAN; data.len()];
    let mut ready = false;
    let mut previous_slot = None;

    for index in 0..data.len() {
        let slot = index % slots_per_day;
        let session_start = previous_slot
            .map(|previous| slot <= previous)
            .unwrap_or(true);
        previous_slot = Some(slot);
        if !ready && index > WINDOW_SIZE && session_start {
            ready = true;
        }
        if !ready || index + 1 < FILTER_LENGTH {
            continue;
        }

        let future = collect_rescanned_future(slot, slots_per_day, |future_slot| {
            let mut values = Vec::with_capacity(DATA_PERIOD);
            for prior in (0..index).rev() {
                if prior % slots_per_day == future_slot && data[prior].is_finite() {
                    values.push(data[prior]);
                    if values.len() == DATA_PERIOD {
                        break;
                    }
                }
            }
            values.reverse();
            let mut moments = StablePopulationMoments::default();
            for value in values {
                moments.add(value);
            }
            Some((moments.mean()?, moments.creator_inverse_cv(1.0).max(0.0)))
        });
        if let Some((future_values, future_confidence)) = future {
            if let Some(value) = estimate_from_future(
                data,
                index,
                &coefficients,
                &future_values,
                &future_confidence,
            ) {
                output[index] = value;
            }
        }
    }
    output
}

fn assert_same_bits(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        if left.is_nan() || right.is_nan() {
            assert!(left.is_nan() && right.is_nan(), "NaN mismatch at {index}");
        } else {
            assert_eq!(left.to_bits(), right.to_bits(), "bit mismatch at {index}");
        }
    }
}

fn creator_readiness_schedule(
    data: &[f64],
    slots_per_day: usize,
    filter_length: usize,
) -> Vec<bool> {
    let window_size = filter_length * 2 - 1;
    let mut ready = false;
    data.iter()
        .enumerate()
        .map(|(index, value)| {
            let session_start = index % slots_per_day == 0;
            if !ready && index > window_size && session_start {
                ready = true;
            }
            ready && value.is_finite()
        })
        .collect()
}

#[test]
fn registry_anchor_21_has_a_distinct_creator_readiness_boundary_from_base_20() {
    let slots_per_day = 40;
    let data = (0..3 * slots_per_day)
        .map(|index| 1_000.0 + index as f64)
        .collect::<Vec<_>>();
    let length_20 = creator_readiness_schedule(&data, slots_per_day, 20);
    let length_21 = creator_readiness_schedule(&data, slots_per_day, 21);

    assert!(length_20[40], "40 > 2*20-1 at the second session");
    assert!(!length_21[40], "40 is not > 2*21-1");
    assert!(length_21[40..80].iter().all(|ready| !ready));
    assert!(length_21[80], "the third session is the first L21 boundary");
}

#[test]
fn authoritative_4096_fixture_freezes_creator_v2_row_849() {
    let data = (0..4096)
        .map(|row| 900.0 + (row % 97) as f64 * 3.25 + row as f64 * 0.001)
        .collect::<Vec<_>>();
    let slots = (0..data.len()).map(|row| row % 288).collect::<Vec<_>>();
    let session_starts = slots.iter().map(|slot| *slot == 0).collect::<Vec<_>>();
    let output = pine_cached_authority(&data, &slots, &session_starts, 288, DATA_PERIOD);
    assert_eq!(output[849].to_bits(), 0x4091_ca58_b879_8573);
}

#[test]
fn cached_tod_moments_match_reverse_history_moments_through_finite_holes() {
    let mut data = (0..960)
        .map(|row| 2000.0 + (row % 53) as f64 * 0.125 + (row as f64 * 0.031).sin())
        .collect::<Vec<_>>();
    for index in [3, 47, 96, 191, 287, 384, 577, 578, 767, 811] {
        data[index] = f64::NAN;
    }

    let slots = (0..data.len()).map(|row| row % 48).collect::<Vec<_>>();
    let session_starts = slots.iter().map(|slot| *slot == 0).collect::<Vec<_>>();
    let cached = rescanned_authority(&data, &slots, &session_starts, 48);
    let oracle = reverse_history_rescan_oracle(&data, 48);
    assert_same_bits(&cached, &oracle);
}

#[test]
fn pine_cached_window_l20_slots12_wrap_is_not_a_current_slot_rescan() {
    let data = (0..144)
        .map(|row| 100.0 + (row % 17) as f64 * 0.75 + row as f64 * 0.01)
        .collect::<Vec<_>>();
    let slots = (0..data.len()).map(|row| row % 12).collect::<Vec<_>>();
    let session_starts = slots.iter().map(|slot| *slot == 0).collect::<Vec<_>>();
    let cached = pine_cached_authority(&data, &slots, &session_starts, 12, DATA_PERIOD);
    let rescanned = rescanned_authority(&data, &slots, &session_starts, 12);
    let first_difference = cached
        .iter()
        .zip(&rescanned)
        .position(|(left, right)| left.to_bits() != right.to_bits());
    assert_eq!(first_difference, Some(49));
    assert_eq!(cached[49].to_bits(), 0x405a_a4b5_cc6d_006d);
}

#[test]
fn pine_cached_window_sparse_slots48_keeps_prior_window_key() {
    let data = (0..384)
        .map(|row| 700.0 + (row % 31) as f64 * 0.125 + row as f64 * 0.002)
        .collect::<Vec<_>>();
    let sparse_day = [0, 1, 5, 9, 13, 17, 22, 26, 31, 36, 41, 45];
    let slots = (0..data.len())
        .map(|row| sparse_day[row % sparse_day.len()])
        .collect::<Vec<_>>();
    let session_starts = (0..data.len())
        .map(|row| row % sparse_day.len() == 0)
        .collect::<Vec<_>>();
    let cached = pine_cached_authority(&data, &slots, &session_starts, 48, DATA_PERIOD);
    let rescanned = rescanned_authority(&data, &slots, &session_starts, 48);
    let first_difference = cached
        .iter()
        .zip(&rescanned)
        .position(|(left, right)| left.to_bits() != right.to_bits());
    assert_eq!(first_difference, Some(49));
    assert_eq!(cached[49].to_bits(), 0x4085_ec62_d649_f32e);
}

#[test]
fn creator_data_period_zero_keeps_all_finite_slot_history_through_holes() {
    let slots_per_day = 12;
    let mut data = (0..slots_per_day * 14)
        .map(|row| {
            let slot = (row % slots_per_day) as f64;
            let day = (row / slots_per_day) as f64;
            1000.0
                + day * 5.0
                + (slot * 0.11).sin() * 30.0
                + (slot * 0.03).cos() * 12.0
                + (slot / slots_per_day as f64) * 25.0
        })
        .collect::<Vec<_>>();
    for index in [7, 28, 47, 74, 99, 121, 146] {
        data[index] = f64::NAN;
    }
    let slots = (0..data.len())
        .map(|row| row % slots_per_day)
        .collect::<Vec<_>>();
    let session_starts = slots.iter().map(|slot| *slot == 0).collect::<Vec<_>>();
    let unbounded = pine_cached_authority(&data, &slots, &session_starts, slots_per_day, 0);
    let bounded_five =
        pine_cached_authority(&data, &slots, &session_starts, slots_per_day, DATA_PERIOD);

    assert!(unbounded[167].is_finite());
    assert_ne!(unbounded[167].to_bits(), bounded_five[167].to_bits());
    assert_eq!(unbounded[167].to_bits(), 0x4090_fce6_16bd_1a7a);
}
