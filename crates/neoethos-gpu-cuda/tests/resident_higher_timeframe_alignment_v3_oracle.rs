const QNAN_BITS: u64 = 0x7ff8_0000_0000_0000;
const VALID: u8 = 0;
const STALE: u8 = 4;
const ALIGNMENT_MISSING: u8 = 9;

#[derive(Clone, Copy)]
enum Availability {
    Fixed { period_ms: i64, max_age_ms: i64 },
    NextDirectOpen,
}

fn available_at(parent_open_ms: &[i64], row: usize, rule: Availability) -> Option<i64> {
    match rule {
        Availability::Fixed { period_ms, .. } => parent_open_ms[row].checked_add(period_ms),
        Availability::NextDirectOpen => parent_open_ms.get(row + 1).copied(),
    }
}

fn linear_cpu_oracle(
    base_open_ms: &[i64],
    parent_open_ms: &[i64],
    source_values: &[f64],
    source_validity: &[u8],
    rule: Availability,
) -> (Vec<f64>, Vec<u8>) {
    let mut values = vec![f64::from_bits(QNAN_BITS); base_open_ms.len()];
    let mut validity = vec![ALIGNMENT_MISSING; base_open_ms.len()];
    let mut cursor = 0;
    let mut last = None;
    for (base_row, &base_ms) in base_open_ms.iter().enumerate() {
        while cursor < parent_open_ms.len() {
            match available_at(parent_open_ms, cursor, rule) {
                Some(available_ms) if available_ms <= base_ms => {
                    last = Some(cursor);
                    cursor += 1;
                }
                Some(_) | None => break,
            }
        }
        let Some(parent_row) = last else { continue };
        let available_ms = available_at(parent_open_ms, parent_row, rule).unwrap();
        if let Availability::Fixed { max_age_ms, .. } = rule
            && base_ms - available_ms > max_age_ms
        {
            validity[base_row] = STALE;
            continue;
        }
        validity[base_row] = source_validity[parent_row];
        if source_validity[parent_row] == VALID {
            values[base_row] = source_values[parent_row];
        }
    }
    (values, validity)
}

fn binary_native_oracle(
    base_open_ms: &[i64],
    parent_open_ms: &[i64],
    source_values: &[f64],
    source_validity: &[u8],
    rule: Availability,
) -> (Vec<f64>, Vec<u8>) {
    let mut values = vec![f64::from_bits(QNAN_BITS); base_open_ms.len()];
    let mut validity = vec![ALIGNMENT_MISSING; base_open_ms.len()];
    for (base_row, &base_ms) in base_open_ms.iter().enumerate() {
        let mut lo = 0;
        let mut hi = parent_open_ms.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match available_at(parent_open_ms, mid, rule) {
                Some(available_ms) if available_ms <= base_ms => lo = mid + 1,
                Some(_) | None => hi = mid,
            }
        }
        if lo == 0 {
            continue;
        }
        let parent_row = lo - 1;
        let available_ms = available_at(parent_open_ms, parent_row, rule).unwrap();
        if let Availability::Fixed { max_age_ms, .. } = rule
            && base_ms - available_ms > max_age_ms
        {
            validity[base_row] = STALE;
            continue;
        }
        validity[base_row] = source_validity[parent_row];
        if source_validity[parent_row] == VALID {
            values[base_row] = source_values[parent_row];
        }
    }
    (values, validity)
}

fn assert_bits_and_validity_equal(left: &(Vec<f64>, Vec<u8>), right: &(Vec<f64>, Vec<u8>)) {
    assert_eq!(left.1, right.1);
    assert_eq!(
        left.0
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        right
            .0
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn fixed_alignment_is_causal_forward_filled_and_stale_only_after_the_boundary() {
    let rule = Availability::Fixed {
        period_ms: 5,
        max_age_ms: 10,
    };
    let base = [-1, 0, 4, 5, 10, 15, 16, 20, 25, 26];
    let parent = [0, 5, 15];
    let source = [10.0, 20.0, 30.0];
    let source_validity = [0, 1, 5];
    let cpu = linear_cpu_oracle(&base, &parent, &source, &source_validity, rule);
    let native = binary_native_oracle(&base, &parent, &source, &source_validity, rule);
    assert_bits_and_validity_equal(&cpu, &native);
    assert_eq!(cpu.1, [9, 9, 9, 0, 1, 1, 1, 5, 5, 5]);
    assert_eq!(cpu.0[3].to_bits(), 10.0_f64.to_bits());
    for row in [0, 1, 2, 4, 5, 6, 7, 8, 9] {
        assert_eq!(cpu.0[row].to_bits(), QNAN_BITS);
    }

    let sparse_parent = [0];
    let sparse_source = [7.0];
    let sparse_validity = [VALID];
    let boundary_base = [5, 15, 16];
    let boundary = linear_cpu_oracle(
        &boundary_base,
        &sparse_parent,
        &sparse_source,
        &sparse_validity,
        rule,
    );
    assert_eq!(boundary.1, [VALID, VALID, STALE]);
}

#[test]
fn calendar_alignment_uses_observed_next_open_and_never_exposes_the_tail() {
    let base = [0, 22, 23, 46, 47, 48, 100];
    let parent = [0, 23, 47];
    let source = [10.0, 20.0, 30.0];
    let validity = [0, 3, 0];
    let cpu = linear_cpu_oracle(
        &base,
        &parent,
        &source,
        &validity,
        Availability::NextDirectOpen,
    );
    let native = binary_native_oracle(
        &base,
        &parent,
        &source,
        &validity,
        Availability::NextDirectOpen,
    );
    assert_bits_and_validity_equal(&cpu, &native);
    assert_eq!(cpu.1, [9, 9, 0, 0, 3, 3, 3]);
    assert!(
        !cpu.0
            .iter()
            .any(|value| value.to_bits() == 30.0_f64.to_bits())
    );

    let single_row = linear_cpu_oracle(
        &[0, 1, 100],
        &[0],
        &[99.0],
        &[VALID],
        Availability::NextDirectOpen,
    );
    let single_row_native = binary_native_oracle(
        &[0, 1, 100],
        &[0],
        &[99.0],
        &[VALID],
        Availability::NextDirectOpen,
    );
    assert_bits_and_validity_equal(&single_row, &single_row_native);
    assert_eq!(single_row.1, [ALIGNMENT_MISSING; 3]);
    assert!(
        single_row
            .0
            .iter()
            .all(|value| value.to_bits() == QNAN_BITS)
    );
}

#[test]
fn every_logical_source_validity_code_is_preserved_without_copying_invalid_payload_bits() {
    let parent = (0..10).map(|row| row * 2).collect::<Vec<_>>();
    let base = (1..=10).map(|row| row * 2).collect::<Vec<_>>();
    let source = (0..10).map(|row| row as f64 + 0.25).collect::<Vec<_>>();
    let source_validity = (0_u8..=9).collect::<Vec<_>>();
    let rule = Availability::Fixed {
        period_ms: 2,
        max_age_ms: 4,
    };
    let cpu = linear_cpu_oracle(&base, &parent, &source, &source_validity, rule);
    let native = binary_native_oracle(&base, &parent, &source, &source_validity, rule);
    assert_bits_and_validity_equal(&cpu, &native);
    assert_eq!(cpu.1, source_validity);
    assert_eq!(cpu.0[0].to_bits(), source[0].to_bits());
    assert!(cpu.0[1..].iter().all(|value| value.to_bits() == QNAN_BITS));
}

#[derive(Debug, PartialEq, Eq)]
struct ParentSegment {
    parent: usize,
    first_column: usize,
    column_count: usize,
}

fn global_batches_and_parent_segments(
    parent_widths: &[usize],
    max_batch_columns: usize,
) -> Vec<Vec<ParentSegment>> {
    let flattened = parent_widths
        .iter()
        .enumerate()
        .flat_map(|(parent, width)| std::iter::repeat_n(parent, *width))
        .collect::<Vec<_>>();
    flattened
        .chunks(max_batch_columns)
        .map(|batch| {
            let mut segments = Vec::<ParentSegment>::new();
            for (local_column, &parent) in batch.iter().enumerate() {
                if let Some(last) = segments.last_mut()
                    && last.parent == parent
                {
                    last.column_count += 1;
                } else {
                    segments.push(ParentSegment {
                        parent,
                        first_column: local_column,
                        column_count: 1,
                    });
                }
            }
            segments
        })
        .collect()
}

#[test]
fn global_variable_width_batches_preserve_cross_parent_clock_segments() {
    let batches = global_batches_and_parent_segments(&[65, 37, 2], 64);
    assert_eq!(batches.len(), 2);
    assert_eq!(
        batches[1],
        [
            ParentSegment {
                parent: 0,
                first_column: 0,
                column_count: 1,
            },
            ParentSegment {
                parent: 1,
                first_column: 1,
                column_count: 37,
            },
            ParentSegment {
                parent: 2,
                first_column: 38,
                column_count: 2,
            },
        ]
    );
}

#[test]
fn variable_width_receipt_counts_requested_device_bytes_and_actual_segment_kernels() {
    let parent_widths = [65_usize, 37, 2];
    let batches = global_batches_and_parent_segments(&parent_widths, 64);
    let columns = parent_widths.iter().sum::<usize>();
    let base_rows = 11_usize;
    let retained_feature_device_bytes = base_rows * columns * (8 + 1);
    let pointer_table_device_bytes = columns.min(64) * 4 * 8;
    let pointer_table_h2d_bytes = columns * 4 * 8;
    let native_abi_launch_count = batches.len();
    let native_kernel_launch_count = batches.iter().map(Vec::len).sum::<usize>();

    assert_eq!(columns, 104);
    assert_eq!(retained_feature_device_bytes, 10_296);
    assert_eq!(pointer_table_device_bytes, 2_048);
    assert_eq!(pointer_table_h2d_bytes, 3_328);
    assert_eq!(native_abi_launch_count, 2);
    assert_eq!(native_kernel_launch_count, 4);
}
