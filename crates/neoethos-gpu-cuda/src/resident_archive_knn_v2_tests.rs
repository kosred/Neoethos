use std::cmp::Ordering;

mod oracle {
    use std::cmp::Ordering;

    pub const W: usize = 4;
    pub const SOURCE_CURRENT: u8 = 0;
    pub const SOURCE_ARCHIVE: u8 = 1;

    const K: usize = 15;
    const MAX_EXACT_F64_INTEGER: u128 = 1_u128 << 53;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NeighborWire {
        pub source_kind: u8,
        pub source_ordinal: u32,
        pub gene_identity: u64,
        pub signature: [u64; W],
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DistanceFraction {
        pub numerator: u32,
        pub denominator: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SelectedNeighbor {
        pub source_kind: u8,
        pub source_ordinal: u32,
        pub gene_identity: u64,
        pub distance: DistanceFraction,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CheckedRational {
        numerator: u128,
        denominator: u128,
    }

    impl CheckedRational {
        pub const fn new(numerator: u128, denominator: u128) -> Self {
            Self {
                numerator,
                denominator,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct OracleResult {
        pub selected: Vec<SelectedNeighbor>,
        pub exact_sum: CheckedRational,
        pub mean: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum OracleError {
        InvalidSourceKind {
            wire: u8,
        },
        CurrentOrdinalOutOfDomain {
            ordinal: u32,
            p: u32,
        },
        ArchiveOrdinalOutOfDomain {
            ordinal: u32,
            archive_count: u32,
        },
        ZeroUnion {
            source_kind: u8,
            source_ordinal: u32,
        },
        DenominatorOutOfDomain {
            denominator: u32,
        },
        NumeratorExceedsDenominator {
            numerator: u32,
            denominator: u32,
        },
        ZeroAvailableNeighbors,
        ComparatorOverflow,
        RationalZeroDenominator,
        RationalSumOverflow,
        IntegerNotExactlyRepresentable {
            value: u128,
        },
    }

    pub fn validate_distance_fraction(
        numerator: u32,
        denominator: u32,
    ) -> Result<DistanceFraction, OracleError> {
        if !(1..=32).contains(&denominator) {
            return Err(OracleError::DenominatorOutOfDomain { denominator });
        }
        if numerator > denominator {
            return Err(OracleError::NumeratorExceedsDenominator {
                numerator,
                denominator,
            });
        }
        Ok(DistanceFraction {
            numerator,
            denominator,
        })
    }

    pub fn compare_fraction_checked(
        left: CheckedRational,
        right: CheckedRational,
    ) -> Result<Ordering, OracleError> {
        if left.denominator == 0 || right.denominator == 0 {
            return Err(OracleError::RationalZeroDenominator);
        }
        let left_cross = left
            .numerator
            .checked_mul(right.denominator)
            .ok_or(OracleError::ComparatorOverflow)?;
        let right_cross = right
            .numerator
            .checked_mul(left.denominator)
            .ok_or(OracleError::ComparatorOverflow)?;
        Ok(left_cross.cmp(&right_cross))
    }

    fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
        while right != 0 {
            let remainder = left % right;
            left = right;
            right = remainder;
        }
        left
    }

    fn reduce_rational(value: CheckedRational) -> Result<CheckedRational, OracleError> {
        if value.denominator == 0 {
            return Err(OracleError::RationalZeroDenominator);
        }
        let divisor = gcd_u128(value.numerator, value.denominator);
        Ok(CheckedRational::new(
            value.numerator / divisor,
            value.denominator / divisor,
        ))
    }

    fn add_rational_checked(
        left: CheckedRational,
        right: CheckedRational,
    ) -> Result<CheckedRational, OracleError> {
        if left.denominator == 0 || right.denominator == 0 {
            return Err(OracleError::RationalZeroDenominator);
        }
        let denominator_gcd = gcd_u128(left.denominator, right.denominator);
        let left_scale = right.denominator / denominator_gcd;
        let right_scale = left.denominator / denominator_gcd;
        let left_numerator = left
            .numerator
            .checked_mul(left_scale)
            .ok_or(OracleError::RationalSumOverflow)?;
        let right_numerator = right
            .numerator
            .checked_mul(right_scale)
            .ok_or(OracleError::RationalSumOverflow)?;
        let numerator = left_numerator
            .checked_add(right_numerator)
            .ok_or(OracleError::RationalSumOverflow)?;
        let denominator = left
            .denominator
            .checked_mul(left_scale)
            .ok_or(OracleError::RationalSumOverflow)?;
        reduce_rational(CheckedRational::new(numerator, denominator))
    }

    pub fn checked_rational_sum(
        fractions: &[CheckedRational],
    ) -> Result<CheckedRational, OracleError> {
        let mut sum = CheckedRational::new(0, 1);
        for &fraction in fractions {
            sum = add_rational_checked(sum, fraction)?;
        }
        Ok(sum)
    }

    fn distance_fraction(
        query_signature: [u64; W],
        candidate: NeighborWire,
    ) -> Result<DistanceFraction, OracleError> {
        let intersection = query_signature
            .iter()
            .zip(candidate.signature.iter())
            .map(|(&query_word, &candidate_word)| (query_word & candidate_word).count_ones())
            .sum::<u32>();
        let union = query_signature
            .iter()
            .zip(candidate.signature.iter())
            .map(|(&query_word, &candidate_word)| (query_word | candidate_word).count_ones())
            .sum::<u32>();
        if union == 0 {
            return Err(OracleError::ZeroUnion {
                source_kind: candidate.source_kind,
                source_ordinal: candidate.source_ordinal,
            });
        }
        validate_distance_fraction(union - intersection, union)
    }

    fn validate_wire_domain(
        candidate: NeighborWire,
        p: u32,
        archive_count: u32,
    ) -> Result<(), OracleError> {
        match candidate.source_kind {
            SOURCE_CURRENT if candidate.source_ordinal < p => Ok(()),
            SOURCE_CURRENT => Err(OracleError::CurrentOrdinalOutOfDomain {
                ordinal: candidate.source_ordinal,
                p,
            }),
            SOURCE_ARCHIVE if candidate.source_ordinal < archive_count => Ok(()),
            SOURCE_ARCHIVE => Err(OracleError::ArchiveOrdinalOutOfDomain {
                ordinal: candidate.source_ordinal,
                archive_count,
            }),
            wire => Err(OracleError::InvalidSourceKind { wire }),
        }
    }

    fn compare_selected(
        left: &SelectedNeighbor,
        right: &SelectedNeighbor,
    ) -> Result<Ordering, OracleError> {
        let rational_order = compare_fraction_checked(
            CheckedRational::new(
                u128::from(left.distance.numerator),
                u128::from(left.distance.denominator),
            ),
            CheckedRational::new(
                u128::from(right.distance.numerator),
                u128::from(right.distance.denominator),
            ),
        )?;
        Ok(rational_order
            .then_with(|| left.gene_identity.cmp(&right.gene_identity))
            .then_with(|| left.source_kind.cmp(&right.source_kind))
            .then_with(|| left.source_ordinal.cmp(&right.source_ordinal)))
    }

    fn sort_selected_checked(neighbors: &mut [SelectedNeighbor]) -> Result<(), OracleError> {
        let mut comparator_error = None;
        neighbors.sort_by(|left, right| {
            if comparator_error.is_some() {
                return Ordering::Equal;
            }
            match compare_selected(left, right) {
                Ok(ordering) => ordering,
                Err(error) => {
                    comparator_error = Some(error);
                    Ordering::Equal
                }
            }
        });
        match comparator_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn exact_u128_to_f64(value: u128) -> Result<f64, OracleError> {
        if value > MAX_EXACT_F64_INTEGER {
            return Err(OracleError::IntegerNotExactlyRepresentable { value });
        }
        Ok(value as f64)
    }

    fn sealed_mean(selected: &[SelectedNeighbor]) -> Result<f64, OracleError> {
        let mut sum = 0.0_f64;
        for neighbor in selected {
            let numerator = exact_u128_to_f64(u128::from(neighbor.distance.numerator))?;
            let denominator = exact_u128_to_f64(u128::from(neighbor.distance.denominator))?;
            let distance = numerator / denominator;
            sum += distance;
        }
        let q = exact_u128_to_f64(selected.len() as u128)?;
        Ok(sum / q)
    }

    pub fn select_exact_knn(
        query_current_ordinal: u32,
        query_signature: [u64; W],
        p: u32,
        archive_count: u32,
        neighbors: &[NeighborWire],
    ) -> Result<OracleResult, OracleError> {
        if query_current_ordinal >= p {
            return Err(OracleError::CurrentOrdinalOutOfDomain {
                ordinal: query_current_ordinal,
                p,
            });
        }

        let mut selected = Vec::with_capacity(neighbors.len());
        for &candidate in neighbors {
            validate_wire_domain(candidate, p, archive_count)?;
            if candidate.source_kind == SOURCE_CURRENT
                && candidate.source_ordinal == query_current_ordinal
            {
                continue;
            }
            selected.push(SelectedNeighbor {
                source_kind: candidate.source_kind,
                source_ordinal: candidate.source_ordinal,
                gene_identity: candidate.gene_identity,
                distance: distance_fraction(query_signature, candidate)?,
            });
        }
        if selected.is_empty() {
            return Err(OracleError::ZeroAvailableNeighbors);
        }

        sort_selected_checked(&mut selected)?;
        selected.truncate(K.min(selected.len()));

        let exact_fractions = selected
            .iter()
            .map(|neighbor| {
                CheckedRational::new(
                    u128::from(neighbor.distance.numerator),
                    u128::from(neighbor.distance.denominator),
                )
            })
            .collect::<Vec<_>>();
        let exact_sum = checked_rational_sum(&exact_fractions)?;
        let mean = sealed_mean(&selected)?;
        Ok(OracleResult {
            selected,
            exact_sum,
            mean,
        })
    }

    fn nonnegative_finite_ulp_distance(left: f64, right: f64) -> Option<u64> {
        if !left.is_finite()
            || !right.is_finite()
            || left.to_bits() >> 63 != 0
            || right.to_bits() >> 63 != 0
        {
            return None;
        }
        Some(left.to_bits().abs_diff(right.to_bits()))
    }

    pub fn accept_sealed_mean(expected: f64, actual: f64) -> bool {
        let Some(ulp_distance) = nonnegative_finite_ulp_distance(expected, actual) else {
            return false;
        };
        if expected.to_bits() == 0 {
            return actual.to_bits() == 0;
        }

        let absolute_error = (actual - expected).abs();
        let relative_error = absolute_error / expected;
        absolute_error <= 2.0_f64.powi(-50)
            && relative_error <= 2.0_f64.powi(-48)
            && ulp_distance <= 4
    }
}

use oracle::{
    CheckedRational, DistanceFraction, NeighborWire, OracleError, OracleResult, SOURCE_ARCHIVE,
    SOURCE_CURRENT, W, accept_sealed_mean, checked_rational_sum, compare_fraction_checked,
    select_exact_knn, validate_distance_fraction,
};

fn signature(word0: u64) -> [u64; W] {
    [word0, 0, 0, 0]
}

fn wire(
    source_kind: u8,
    source_ordinal: u32,
    gene_identity: u64,
    signature: [u64; W],
) -> NeighborWire {
    NeighborWire {
        source_kind,
        source_ordinal,
        gene_identity,
        signature,
    }
}

fn selected_keys(result: &OracleResult) -> Vec<(u64, u8, u32, u32, u32)> {
    result
        .selected
        .iter()
        .map(|neighbor| {
            (
                neighbor.gene_identity,
                neighbor.source_kind,
                neighbor.source_ordinal,
                neighbor.distance.numerator,
                neighbor.distance.denominator,
            )
        })
        .collect()
}

#[test]
fn r2_current_and_archive_neighbors_use_exact_rational_order_and_sum() {
    let query = [1_u64 << 63, 1_u64 << 17, 1_u64 << 42, 1_u64 << 5];
    let neighbors = [
        wire(
            SOURCE_CURRENT,
            0,
            10,
            [1_u64 << 63, 1_u64 << 17, 1_u64 << 42, 0],
        ),
        wire(SOURCE_CURRENT, 1, 99, query),
        wire(SOURCE_CURRENT, 2, 30, [1_u64 << 63, 0, 0, 0]),
        wire(SOURCE_ARCHIVE, 0, 20, [1_u64 << 63, 0, 1_u64 << 42, 0]),
    ];

    let result = select_exact_knn(1, query, 3, 1, &neighbors).unwrap();

    assert_eq!(
        selected_keys(&result),
        vec![
            (10, SOURCE_CURRENT, 0, 1, 4),
            (20, SOURCE_ARCHIVE, 0, 2, 4),
            (30, SOURCE_CURRENT, 2, 3, 4),
        ]
    );
    assert_eq!(result.exact_sum, CheckedRational::new(3, 2));
    assert_eq!(result.mean.to_bits(), 0.5_f64.to_bits());

    let cross_denominator_query = signature(0b0011);
    let shuffled_cross_denominator_neighbors = [
        wire(SOURCE_CURRENT, 0, 99, cross_denominator_query),
        wire(SOURCE_CURRENT, 1, 50, signature(0b0101)),
        wire(SOURCE_CURRENT, 2, 30, signature(0b0001)),
        wire(SOURCE_CURRENT, 3, 40, signature(0b1_1111)),
        wire(SOURCE_CURRENT, 4, 20, signature(0b1111)),
    ];
    let cross_denominator_result = select_exact_knn(
        0,
        cross_denominator_query,
        5,
        0,
        &shuffled_cross_denominator_neighbors,
    )
    .unwrap();

    assert_eq!(
        selected_keys(&cross_denominator_result),
        vec![
            (20, SOURCE_CURRENT, 4, 2, 4),
            (30, SOURCE_CURRENT, 2, 1, 2),
            (40, SOURCE_CURRENT, 3, 3, 5),
            (50, SOURCE_CURRENT, 1, 2, 3),
        ]
    );
    assert_eq!(
        cross_denominator_result.exact_sum,
        CheckedRational::new(34, 15)
    );
}

#[test]
fn r2_self_exclusion_is_exact_and_duplicate_zero_distance_neighbors_survive() {
    let query = signature(0b0001);
    let neighbors = [
        wire(SOURCE_CURRENT, 0, 99, query),
        wire(SOURCE_CURRENT, 1, 11, query),
        wire(SOURCE_ARCHIVE, 0, 11, query),
        wire(SOURCE_ARCHIVE, 1, 12, query),
    ];

    let result = select_exact_knn(0, query, 2, 2, &neighbors).unwrap();

    assert_eq!(
        selected_keys(&result),
        vec![
            (11, SOURCE_CURRENT, 1, 0, 1),
            (11, SOURCE_ARCHIVE, 0, 0, 1),
            (12, SOURCE_ARCHIVE, 1, 0, 1),
        ]
    );
    assert_eq!(result.exact_sum, CheckedRational::new(0, 1));
    assert_eq!(result.mean.to_bits(), 0_u64);
}

#[test]
fn r2_fewer_than_k_selects_every_available_neighbor() {
    let query = signature(0b1111);
    let neighbors = [
        wire(SOURCE_CURRENT, 0, 99, query),
        wire(SOURCE_CURRENT, 1, 10, signature(0b0111)),
        wire(SOURCE_ARCHIVE, 0, 20, signature(0b0011)),
    ];

    let result = select_exact_knn(0, query, 2, 1, &neighbors).unwrap();

    assert_eq!(
        selected_keys(&result),
        vec![(10, SOURCE_CURRENT, 1, 1, 4), (20, SOURCE_ARCHIVE, 0, 2, 4),]
    );
    assert_eq!(result.exact_sum, CheckedRational::new(3, 4));
    assert_eq!(result.mean.to_bits(), 0.375_f64.to_bits());

    let order_sensitive_query = signature(1);
    let shuffled_order_sensitive_neighbors = [
        wire(SOURCE_CURRENT, 0, 999, order_sensitive_query),
        wire(SOURCE_CURRENT, 7, 70, signature(0x3fff)),
        wire(SOURCE_CURRENT, 2, 22, signature(0x001f)),
        wire(SOURCE_CURRENT, 5, 52, signature(0x1fff)),
        wire(SOURCE_CURRENT, 1, 20, signature(0x0007)),
        wire(SOURCE_CURRENT, 4, 40, signature(0x07ff)),
        wire(SOURCE_CURRENT, 3, 21, signature(0x001f)),
        wire(SOURCE_CURRENT, 6, 51, signature(0x1fff)),
    ];
    let order_sensitive_result = select_exact_knn(
        0,
        order_sensitive_query,
        8,
        0,
        &shuffled_order_sensitive_neighbors,
    )
    .unwrap();

    assert_eq!(
        selected_keys(&order_sensitive_result),
        vec![
            (20, SOURCE_CURRENT, 1, 2, 3),
            (21, SOURCE_CURRENT, 3, 4, 5),
            (22, SOURCE_CURRENT, 2, 4, 5),
            (40, SOURCE_CURRENT, 4, 10, 11),
            (51, SOURCE_CURRENT, 6, 12, 13),
            (52, SOURCE_CURRENT, 5, 12, 13),
            (70, SOURCE_CURRENT, 7, 13, 14),
        ]
    );
    assert_eq!(
        order_sensitive_result.exact_sum,
        CheckedRational::new(178_693, 30_030)
    );
    assert_eq!(order_sensitive_result.mean.to_bits(), 0x3feb_33c3_dbd3_5979);

    let mut reverse_sum = 0.0_f64;
    for (numerator, denominator) in [
        (13_u32, 14_u32),
        (12, 13),
        (12, 13),
        (10, 11),
        (4, 5),
        (4, 5),
        (2, 3),
    ] {
        reverse_sum += f64::from(numerator) / f64::from(denominator);
    }
    let reverse_mean = reverse_sum / 7.0;
    let exact_sum_once_mean = (178_693.0_f64 / 30_030.0_f64) / 7.0_f64;
    assert_eq!(reverse_mean.to_bits(), 0x3feb_33c3_dbd3_5978);
    assert_eq!(exact_sum_once_mean.to_bits(), 0x3feb_33c3_dbd3_5978);
    assert_ne!(
        order_sensitive_result.mean.to_bits(),
        reverse_mean.to_bits()
    );
    assert_ne!(
        order_sensitive_result.mean.to_bits(),
        exact_sum_once_mean.to_bits()
    );
}

#[test]
fn r2_zero_available_neighbors_is_an_error() {
    let query = signature(0b0001);
    let neighbors = [wire(SOURCE_CURRENT, 0, 99, query)];

    assert_eq!(
        select_exact_knn(0, query, 1, 0, &neighbors),
        Err(OracleError::ZeroAvailableNeighbors)
    );
}

#[test]
fn r2_zero_union_is_an_error() {
    let query = signature(0);
    let neighbors = [
        wire(SOURCE_CURRENT, 0, 99, query),
        wire(SOURCE_CURRENT, 1, 10, signature(0)),
    ];

    assert_eq!(
        select_exact_knn(0, query, 2, 0, &neighbors),
        Err(OracleError::ZeroUnion {
            source_kind: SOURCE_CURRENT,
            source_ordinal: 1,
        })
    );
}

#[test]
fn r2_total_ties_use_gene_then_source_kind_then_source_ordinal() {
    let query = signature(0b0011);
    let tied = signature(0b0001);
    let neighbors = [
        wire(SOURCE_ARCHIVE, 0, 10, tied),
        wire(SOURCE_CURRENT, 2, 9, tied),
        wire(SOURCE_CURRENT, 0, 99, query),
        wire(SOURCE_ARCHIVE, 1, 9, tied),
        wire(SOURCE_CURRENT, 3, 10, tied),
        wire(SOURCE_CURRENT, 1, 9, tied),
    ];

    let result = select_exact_knn(0, query, 4, 2, &neighbors).unwrap();

    assert_eq!(
        selected_keys(&result),
        vec![
            (9, SOURCE_CURRENT, 1, 1, 2),
            (9, SOURCE_CURRENT, 2, 1, 2),
            (9, SOURCE_ARCHIVE, 1, 1, 2),
            (10, SOURCE_CURRENT, 3, 1, 2),
            (10, SOURCE_ARCHIVE, 0, 1, 2),
        ]
    );
    assert_eq!(result.exact_sum, CheckedRational::new(5, 2));
    assert_eq!(result.mean.to_bits(), 0.5_f64.to_bits());
}

#[test]
fn r2_k15_cutoff_prefers_current_over_archive_for_equal_distance_and_identity() {
    let query = [0xff; W];
    let neighbors = [
        wire(SOURCE_CURRENT, 0, 9999, query),
        wire(SOURCE_ARCHIVE, 0, 999, [0xff, 0xff, 0x00, 0x00]),
        wire(SOURCE_CURRENT, 7, 107, [0xff, 0xff, 0xff, 0x01]),
        wire(SOURCE_CURRENT, 1, 101, [0xff, 0xff, 0xff, 0x7f]),
        wire(SOURCE_CURRENT, 14, 114, [0xff, 0xff, 0x03, 0x00]),
        wire(SOURCE_CURRENT, 4, 104, [0xff, 0xff, 0xff, 0x0f]),
        wire(SOURCE_CURRENT, 10, 110, [0xff, 0xff, 0x3f, 0x00]),
        wire(SOURCE_CURRENT, 2, 102, [0xff, 0xff, 0xff, 0x3f]),
        wire(SOURCE_CURRENT, 12, 112, [0xff, 0xff, 0x0f, 0x00]),
        wire(SOURCE_CURRENT, 5, 105, [0xff, 0xff, 0xff, 0x07]),
        wire(SOURCE_CURRENT, 9, 109, [0xff, 0xff, 0x7f, 0x00]),
        wire(SOURCE_CURRENT, 3, 103, [0xff, 0xff, 0xff, 0x1f]),
        wire(SOURCE_CURRENT, 13, 113, [0xff, 0xff, 0x07, 0x00]),
        wire(SOURCE_CURRENT, 6, 106, [0xff, 0xff, 0xff, 0x03]),
        wire(SOURCE_CURRENT, 11, 111, [0xff, 0xff, 0x1f, 0x00]),
        wire(SOURCE_CURRENT, 8, 108, [0xff, 0xff, 0xff, 0x00]),
        wire(SOURCE_CURRENT, 15, 999, [0xff, 0xff, 0x00, 0x00]),
    ];

    let result = select_exact_knn(0, query, 16, 1, &neighbors).unwrap();

    assert_eq!(
        selected_keys(&result),
        vec![
            (101, SOURCE_CURRENT, 1, 1, 32),
            (102, SOURCE_CURRENT, 2, 2, 32),
            (103, SOURCE_CURRENT, 3, 3, 32),
            (104, SOURCE_CURRENT, 4, 4, 32),
            (105, SOURCE_CURRENT, 5, 5, 32),
            (106, SOURCE_CURRENT, 6, 6, 32),
            (107, SOURCE_CURRENT, 7, 7, 32),
            (108, SOURCE_CURRENT, 8, 8, 32),
            (109, SOURCE_CURRENT, 9, 9, 32),
            (110, SOURCE_CURRENT, 10, 10, 32),
            (111, SOURCE_CURRENT, 11, 11, 32),
            (112, SOURCE_CURRENT, 12, 12, 32),
            (113, SOURCE_CURRENT, 13, 13, 32),
            (114, SOURCE_CURRENT, 14, 14, 32),
            (999, SOURCE_CURRENT, 15, 16, 32),
        ]
    );
    assert_eq!(result.exact_sum, CheckedRational::new(121, 32));
    assert_eq!(result.mean.to_bits(), 0.2520833333333333_f64.to_bits());
}

#[test]
fn r2_rejects_out_of_domain_source_wires_and_ordinals() {
    let query = signature(0b0001);

    assert_eq!(
        select_exact_knn(2, query, 2, 0, &[]),
        Err(OracleError::CurrentOrdinalOutOfDomain { ordinal: 2, p: 2 })
    );
    assert_eq!(
        select_exact_knn(0, query, 1, 0, &[wire(2, 0, 10, query)]),
        Err(OracleError::InvalidSourceKind { wire: 2 })
    );
    assert_eq!(
        select_exact_knn(0, query, 2, 0, &[wire(SOURCE_CURRENT, 2, 10, query)],),
        Err(OracleError::CurrentOrdinalOutOfDomain { ordinal: 2, p: 2 })
    );
    assert_eq!(
        select_exact_knn(0, query, 1, 1, &[wire(SOURCE_ARCHIVE, 1, 10, query)],),
        Err(OracleError::ArchiveOrdinalOutOfDomain {
            ordinal: 1,
            archive_count: 1,
        })
    );
}

#[test]
fn r2_distance_fraction_domain_is_exactly_one_to_32_for_the_denominator() {
    assert_eq!(
        validate_distance_fraction(32, 32),
        Ok(DistanceFraction {
            numerator: 32,
            denominator: 32,
        })
    );
    assert_eq!(
        validate_distance_fraction(1, 0),
        Err(OracleError::DenominatorOutOfDomain { denominator: 0 })
    );
    assert_eq!(
        validate_distance_fraction(0, 33),
        Err(OracleError::DenominatorOutOfDomain { denominator: 33 })
    );
    assert_eq!(
        validate_distance_fraction(33, 32),
        Err(OracleError::NumeratorExceedsDenominator {
            numerator: 33,
            denominator: 32,
        })
    );

    let query = [0xff_u64; W];
    let union_33 = [0x1ff_u64, 0xff, 0xff, 0xff];
    assert_eq!(
        select_exact_knn(0, query, 2, 0, &[wire(SOURCE_CURRENT, 1, 10, union_33)],),
        Err(OracleError::DenominatorOutOfDomain { denominator: 33 })
    );
}

#[test]
fn r2_checked_fraction_comparator_accepts_the_limit_and_rejects_overflow() {
    let at_limit = CheckedRational::new(u128::MAX, 1);
    let multiplier_one = CheckedRational::new(1, 1);
    assert_eq!(
        compare_fraction_checked(at_limit, multiplier_one),
        Ok(Ordering::Greater)
    );

    let multiplier_two = CheckedRational::new(1, 2);
    assert_eq!(
        compare_fraction_checked(at_limit, multiplier_two),
        Err(OracleError::ComparatorOverflow)
    );
    assert_eq!(
        compare_fraction_checked(multiplier_two, at_limit),
        Err(OracleError::ComparatorOverflow)
    );
}

#[test]
fn r2_exact_rational_sum_reduces_and_checks_u128_arithmetic() {
    assert_eq!(
        checked_rational_sum(&[
            CheckedRational::new(1, 2),
            CheckedRational::new(1, 3),
            CheckedRational::new(1, 6),
        ]),
        Ok(CheckedRational::new(1, 1))
    );
    assert_eq!(
        checked_rational_sum(&[
            CheckedRational::new(u128::MAX, 1),
            CheckedRational::new(1, 1),
        ]),
        Err(OracleError::RationalSumOverflow)
    );
    assert_eq!(
        checked_rational_sum(&[
            CheckedRational::new(u128::MAX, 1),
            CheckedRational::new(0, 2),
        ]),
        Err(OracleError::RationalSumOverflow)
    );
    assert_eq!(
        checked_rational_sum(&[
            CheckedRational::new(1, 2),
            CheckedRational::new(u128::MAX, 1),
        ]),
        Err(OracleError::RationalSumOverflow)
    );
    assert_eq!(
        checked_rational_sum(&[CheckedRational::new(1, 2), CheckedRational::new(1, 0),]),
        Err(OracleError::RationalZeroDenominator)
    );
    assert_eq!(
        checked_rational_sum(&[
            CheckedRational::new(1, u128::MAX),
            CheckedRational::new(0, 2),
        ]),
        Err(OracleError::RationalSumOverflow)
    );
}

#[test]
fn r2_acceptance_includes_the_exact_absolute_error_boundary() {
    let expected = 1.0_f64;
    let actual = f64::from_bits(expected.to_bits() + 4);
    let absolute_error = (actual - expected).abs();

    assert_eq!(absolute_error.to_bits(), 2.0_f64.powi(-50).to_bits());
    assert!(accept_sealed_mean(expected, actual));

    let lower_expected = 8.0_f64;
    let lower_actual = f64::from_bits(lower_expected.to_bits() - 1);
    let lower_absolute_error = (lower_actual - lower_expected).abs();
    assert_eq!(lower_absolute_error.to_bits(), 2.0_f64.powi(-50).to_bits());
    assert!(accept_sealed_mean(lower_expected, lower_actual));
}

#[test]
fn r2_acceptance_includes_the_exact_relative_error_boundary() {
    let expected = f64::from_bits(1_u64 << 48);
    let actual = f64::from_bits((1_u64 << 48) + 1);
    let relative_error = (actual - expected).abs() / expected;

    assert_eq!(relative_error.to_bits(), 2.0_f64.powi(-48).to_bits());
    assert!(accept_sealed_mean(expected, actual));

    let lower_actual = f64::from_bits((1_u64 << 48) - 1);
    let lower_relative_error = (lower_actual - expected).abs() / expected;
    assert_eq!(lower_relative_error.to_bits(), 2.0_f64.powi(-48).to_bits());
    assert!(accept_sealed_mean(expected, lower_actual));
}

#[test]
fn r2_acceptance_includes_four_ulps_and_rejects_five_ulps() {
    let expected = 0.5_f64;
    let four_ulps = f64::from_bits(expected.to_bits() + 4);
    let five_ulps = f64::from_bits(expected.to_bits() + 5);
    let lower_four_ulps = f64::from_bits(expected.to_bits() - 4);
    let lower_five_ulps = f64::from_bits(expected.to_bits() - 5);

    assert!(accept_sealed_mean(expected, four_ulps));
    assert!(!accept_sealed_mean(expected, five_ulps));
    assert!(accept_sealed_mean(expected, lower_four_ulps));
    assert!(!accept_sealed_mean(expected, lower_five_ulps));
}

#[test]
fn r2_acceptance_rejects_absolute_only_and_relative_only_failures() {
    let large_expected = 8.0_f64;
    let one_large_ulp = f64::from_bits(large_expected.to_bits() + 1);
    let large_absolute_error = (one_large_ulp - large_expected).abs();
    let large_relative_error = large_absolute_error / large_expected;
    assert!(large_absolute_error > 2.0_f64.powi(-50));
    assert!(large_relative_error <= 2.0_f64.powi(-48));
    assert!(!accept_sealed_mean(large_expected, one_large_ulp));

    let two_lower_large_steps = f64::from_bits(large_expected.to_bits() - 2);
    let lower_large_absolute_error = (two_lower_large_steps - large_expected).abs();
    let lower_large_relative_error = lower_large_absolute_error / large_expected;
    assert!(lower_large_absolute_error > 2.0_f64.powi(-50));
    assert!(lower_large_relative_error <= 2.0_f64.powi(-48));
    assert_eq!(
        large_expected.to_bits() - two_lower_large_steps.to_bits(),
        2
    );
    assert!(!accept_sealed_mean(large_expected, two_lower_large_steps));

    let subnormal_expected = f64::from_bits(1_u64 << 47);
    let one_subnormal_ulp = f64::from_bits((1_u64 << 47) + 1);
    let subnormal_absolute_error = (one_subnormal_ulp - subnormal_expected).abs();
    let subnormal_relative_error = subnormal_absolute_error / subnormal_expected;
    assert!(subnormal_absolute_error <= 2.0_f64.powi(-50));
    assert!(subnormal_relative_error > 2.0_f64.powi(-48));
    assert!(!accept_sealed_mean(subnormal_expected, one_subnormal_ulp));

    let one_lower_subnormal_ulp = f64::from_bits((1_u64 << 47) - 1);
    let lower_subnormal_absolute_error = (one_lower_subnormal_ulp - subnormal_expected).abs();
    let lower_subnormal_relative_error = lower_subnormal_absolute_error / subnormal_expected;
    assert!(lower_subnormal_absolute_error <= 2.0_f64.powi(-50));
    assert!(lower_subnormal_relative_error > 2.0_f64.powi(-48));
    assert!(!accept_sealed_mean(
        subnormal_expected,
        one_lower_subnormal_ulp
    ));
}

#[test]
fn r2_acceptance_requires_exact_positive_zero_and_rejects_nonfinite_or_sign() {
    let positive_zero = 0.0_f64;
    let negative_zero = -0.0_f64;

    assert_eq!(positive_zero.to_bits(), 0_u64);
    assert!(accept_sealed_mean(positive_zero, positive_zero));
    assert!(!accept_sealed_mean(positive_zero, negative_zero));
    assert!(!accept_sealed_mean(negative_zero, positive_zero));
    assert!(!accept_sealed_mean(negative_zero, negative_zero));
    assert!(!accept_sealed_mean(positive_zero, f64::from_bits(1)));
    assert!(!accept_sealed_mean(1.0, -1.0));
    assert!(!accept_sealed_mean(-1.0, 1.0));
    assert!(!accept_sealed_mean(-1.0, -1.0));
    assert!(!accept_sealed_mean(1.0, f64::NAN));
    assert!(!accept_sealed_mean(f64::NAN, 1.0));
    assert!(!accept_sealed_mean(1.0, f64::INFINITY));
    assert!(!accept_sealed_mean(f64::INFINITY, 1.0));
}

mod archive_reference {
    use std::cmp::Ordering;

    pub const METRIC_COUNT: usize = 11;
    pub const NET_METRIC_SLOT: usize = 0;
    pub const TRADE_COUNT_METRIC_SLOT: usize = 8;

    /// Canonical full-gene bytes after normalization.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ExactGene(pub [u64; 2]);

    #[derive(Debug, Clone, PartialEq)]
    pub struct AdmissionCandidate {
        pub exact_gene: ExactGene,
        pub full_gene_hash: u64,
        pub gene_identity: u64,
        pub population_ordinal: u32,
        pub score: f64,
        pub metrics: [f64; METRIC_COUNT],
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ArchiveRecord {
        pub exact_gene: ExactGene,
        pub full_gene_hash: u64,
        pub gene_identity: u64,
        pub population_ordinal: u32,
        pub admitted_generation: u32,
        pub metric_bits: [u64; METRIC_COUNT],
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct NeighborSnapshot {
        pub generation: u32,
        pub records: Vec<ArchiveRecord>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct StageReceipt {
        pub generation: u32,
        pub committed_count_at_start: usize,
        pub target_committed_count: usize,
        pub full_gene_hash_collision_count: u64,
        pub records: Vec<ArchiveRecord>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CommitReceipt {
        pub completed_generation: u32,
        pub next_generation: u32,
        pub previous_committed_count: usize,
        pub committed_count: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ArchiveError {
        GenerationMismatch {
            current: u32,
            requested: u32,
        },
        StageAlreadyPrepared {
            generation: u32,
        },
        TransactionFaulted {
            generation: u32,
        },
        NonFiniteMetric {
            population_ordinal: u32,
            metric_slot: usize,
        },
        NoStagedTransaction {
            generation: u32,
        },
    }

    #[derive(Debug)]
    struct PendingArchive {
        generation: u32,
        target_full_gene_hash_collision_count: u64,
        records: Vec<ArchiveRecord>,
    }

    #[derive(Debug)]
    pub struct ReferenceArchive {
        capacity: usize,
        current_generation: u32,
        committed: Vec<ArchiveRecord>,
        committed_full_gene_hash_collision_count: u64,
        pending: Option<PendingArchive>,
        faulted_generation: Option<u32>,
    }

    impl ReferenceArchive {
        pub fn new(capacity: usize) -> Self {
            Self {
                capacity,
                current_generation: 0,
                committed: Vec::new(),
                committed_full_gene_hash_collision_count: 0,
                pending: None,
                faulted_generation: None,
            }
        }

        pub fn from_committed_fixture(
            capacity: usize,
            current_generation: u32,
            committed: Vec<ArchiveRecord>,
        ) -> Self {
            assert!(committed.len() <= capacity);
            Self {
                capacity,
                current_generation,
                committed,
                committed_full_gene_hash_collision_count: 0,
                pending: None,
                faulted_generation: None,
            }
        }

        pub fn current_generation(&self) -> u32 {
            self.current_generation
        }

        pub fn committed_count(&self) -> usize {
            self.committed.len()
        }

        pub fn committed_records(&self) -> &[ArchiveRecord] {
            &self.committed
        }

        pub fn committed_full_gene_hash_collision_count(&self) -> u64 {
            self.committed_full_gene_hash_collision_count
        }

        pub fn staged_count(&self) -> usize {
            self.pending
                .as_ref()
                .map_or(0, |pending| pending.records.len())
        }

        pub fn faulted_generation(&self) -> Option<u32> {
            self.faulted_generation
        }

        pub fn neighbor_snapshot_at_generation_start(
            &self,
            generation: u32,
        ) -> Result<NeighborSnapshot, ArchiveError> {
            self.require_current_generation(generation)?;
            Ok(NeighborSnapshot {
                generation,
                records: self.committed.clone(),
            })
        }

        pub fn stage_ranked_admissions(
            &mut self,
            generation: u32,
            candidates: &[AdmissionCandidate],
        ) -> Result<StageReceipt, ArchiveError> {
            self.require_current_generation(generation)?;
            if let Some(faulted_generation) = self.faulted_generation {
                return Err(ArchiveError::TransactionFaulted {
                    generation: faulted_generation,
                });
            }
            if self.pending.is_some() {
                return Err(ArchiveError::StageAlreadyPrepared { generation });
            }

            for candidate in candidates {
                if let Some(metric_slot) = candidate
                    .metrics
                    .iter()
                    .position(|metric| !metric.is_finite())
                {
                    self.faulted_generation = Some(generation);
                    return Err(ArchiveError::NonFiniteMetric {
                        population_ordinal: candidate.population_ordinal,
                        metric_slot,
                    });
                }
            }

            let mut ranked = candidates.iter().collect::<Vec<_>>();
            ranked.sort_by(|left, right| compare_rank(left, right));

            let available_slots = self.capacity.saturating_sub(self.committed.len());
            let mut consumed_slots = 0_usize;
            let mut target_full_gene_hash_collision_count =
                self.committed_full_gene_hash_collision_count;
            let mut staged: Vec<ArchiveRecord> =
                Vec::with_capacity(available_slots.min(ranked.len()));
            for candidate in ranked {
                let positive_trade_count = candidate.metrics[TRADE_COUNT_METRIC_SLOT] > 0.0;
                let positive_net = candidate.metrics[NET_METRIC_SLOT] > 0.0;
                if !positive_trade_count || !positive_net {
                    continue;
                }

                let mut already_first_seen = false;
                let mut same_hash_unequal_gene_count = 0_u64;
                for record in self.committed.iter().chain(staged.iter()) {
                    if record.full_gene_hash != candidate.full_gene_hash {
                        continue;
                    }
                    if record.exact_gene == candidate.exact_gene {
                        already_first_seen = true;
                    } else {
                        same_hash_unequal_gene_count = same_hash_unequal_gene_count
                            .checked_add(1)
                            .expect("R4 fixture same-hash bucket count must not overflow");
                    }
                }
                if already_first_seen {
                    continue;
                }
                if consumed_slots == available_slots {
                    break;
                }

                target_full_gene_hash_collision_count = target_full_gene_hash_collision_count
                    .checked_add(same_hash_unequal_gene_count)
                    .expect("R4 fixture collision counter must not overflow");

                staged.push(ArchiveRecord {
                    exact_gene: candidate.exact_gene,
                    full_gene_hash: candidate.full_gene_hash,
                    gene_identity: candidate.gene_identity,
                    population_ordinal: candidate.population_ordinal,
                    admitted_generation: generation,
                    metric_bits: candidate.metrics.map(f64::to_bits),
                });
                consumed_slots += 1;
            }

            let committed_count_at_start = self.committed.len();
            let target_committed_count = committed_count_at_start + staged.len();
            let receipt = StageReceipt {
                generation,
                committed_count_at_start,
                target_committed_count,
                full_gene_hash_collision_count: target_full_gene_hash_collision_count,
                records: staged.clone(),
            };
            self.pending = Some(PendingArchive {
                generation,
                target_full_gene_hash_collision_count,
                records: staged,
            });
            Ok(receipt)
        }

        pub fn fault_staged_transaction_before_commit(
            &mut self,
            generation: u32,
        ) -> Result<(), ArchiveError> {
            self.require_current_generation(generation)?;
            if let Some(faulted_generation) = self.faulted_generation {
                return Err(ArchiveError::TransactionFaulted {
                    generation: faulted_generation,
                });
            }
            let pending = self
                .pending
                .as_ref()
                .ok_or(ArchiveError::NoStagedTransaction { generation })?;
            if pending.generation != generation {
                return Err(ArchiveError::GenerationMismatch {
                    current: pending.generation,
                    requested: generation,
                });
            }
            self.faulted_generation = Some(generation);
            Ok(())
        }

        pub fn combined_commit(&mut self, generation: u32) -> Result<CommitReceipt, ArchiveError> {
            self.require_current_generation(generation)?;
            if let Some(faulted_generation) = self.faulted_generation {
                return Err(ArchiveError::TransactionFaulted {
                    generation: faulted_generation,
                });
            }
            let pending = self
                .pending
                .as_ref()
                .ok_or(ArchiveError::NoStagedTransaction { generation })?;
            if pending.generation != generation {
                return Err(ArchiveError::GenerationMismatch {
                    current: pending.generation,
                    requested: generation,
                });
            }

            let next_generation = generation
                .checked_add(1)
                .expect("R3 fixture generation must not overflow");
            let previous_committed_count = self.committed.len();
            let pending = self
                .pending
                .take()
                .expect("the pending archive was checked above");
            self.committed.extend(pending.records);
            self.committed_full_gene_hash_collision_count =
                pending.target_full_gene_hash_collision_count;
            self.current_generation = next_generation;
            Ok(CommitReceipt {
                completed_generation: generation,
                next_generation,
                previous_committed_count,
                committed_count: self.committed.len(),
            })
        }

        fn require_current_generation(&self, requested: u32) -> Result<(), ArchiveError> {
            if requested != self.current_generation {
                return Err(ArchiveError::GenerationMismatch {
                    current: self.current_generation,
                    requested,
                });
            }
            Ok(())
        }
    }

    pub fn compare_rank(left: &AdmissionCandidate, right: &AdmissionCandidate) -> Ordering {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.gene_identity.cmp(&right.gene_identity))
            .then_with(|| left.population_ordinal.cmp(&right.population_ordinal))
    }
}

use archive_reference::{
    AdmissionCandidate, ArchiveError, ArchiveRecord, CommitReceipt, ExactGene, METRIC_COUNT,
    NET_METRIC_SLOT, ReferenceArchive, StageReceipt, TRADE_COUNT_METRIC_SLOT,
};

fn archive_candidate(
    exact_gene: [u64; 2],
    gene_identity: u64,
    population_ordinal: u32,
    score: f64,
    net: f64,
    trade_count: f64,
) -> AdmissionCandidate {
    let mut metrics = [1.0_f64; METRIC_COUNT];
    metrics[NET_METRIC_SLOT] = net;
    metrics[TRADE_COUNT_METRIC_SLOT] = trade_count;
    AdmissionCandidate {
        exact_gene: ExactGene(exact_gene),
        full_gene_hash: exact_gene[0],
        gene_identity,
        population_ordinal,
        score,
        metrics,
    }
}

fn archive_candidate_with_metric_row(
    exact_gene: [u64; 2],
    gene_identity: u64,
    population_ordinal: u32,
    score: f64,
    metrics: [f64; METRIC_COUNT],
) -> AdmissionCandidate {
    assert!(metrics.iter().all(|metric| metric.is_finite()));
    assert!(metrics[NET_METRIC_SLOT] > 0.0);
    assert!(metrics[TRADE_COUNT_METRIC_SLOT] > 0.0);
    AdmissionCandidate {
        exact_gene: ExactGene(exact_gene),
        full_gene_hash: exact_gene[0],
        gene_identity,
        population_ordinal,
        score,
        metrics,
    }
}

fn archive_candidate_with_forced_full_gene_hash(
    exact_gene: [u64; 2],
    full_gene_hash: u64,
    gene_identity: u64,
    population_ordinal: u32,
    score: f64,
    metrics: [f64; METRIC_COUNT],
) -> AdmissionCandidate {
    assert!(metrics.iter().all(|metric| metric.is_finite()));
    assert!(metrics[NET_METRIC_SLOT] > 0.0);
    assert!(metrics[TRADE_COUNT_METRIC_SLOT] > 0.0);
    AdmissionCandidate {
        exact_gene: ExactGene(exact_gene),
        full_gene_hash,
        gene_identity,
        population_ordinal,
        score,
        metrics,
    }
}

fn archive_record_keys(records: &[ArchiveRecord]) -> Vec<(ExactGene, u64, u32, u32)> {
    records
        .iter()
        .map(|record| {
            (
                record.exact_gene,
                record.gene_identity,
                record.population_ordinal,
                record.admitted_generation,
            )
        })
        .collect()
}

fn archive_metric_bits(record: &ArchiveRecord) -> [u64; METRIC_COUNT] {
    record.metric_bits
}

fn metric_row_bits(metrics: [f64; METRIC_COUNT]) -> [u64; METRIC_COUNT] {
    metrics.map(f64::to_bits)
}

fn assert_record_metric_row(
    record: &ArchiveRecord,
    exact_gene: ExactGene,
    metrics: [f64; METRIC_COUNT],
) {
    assert_eq!(record.exact_gene, exact_gene);
    assert_eq!(archive_metric_bits(record), metric_row_bits(metrics));
}

fn assert_stage_receipt(
    receipt: &StageReceipt,
    generation: u32,
    committed_count_at_start: usize,
    target_committed_count: usize,
    expected_records: &[(ExactGene, u64, u32, u32)],
) {
    assert_eq!(receipt.generation, generation);
    assert_eq!(receipt.committed_count_at_start, committed_count_at_start);
    assert_eq!(receipt.target_committed_count, target_committed_count);
    assert_eq!(archive_record_keys(&receipt.records), expected_records);
}

#[test]
fn r3_staged_admissions_are_invisible_until_atomic_generation_commit() {
    let initial = archive_candidate([10, 100], 10, 0, 30.0, 4.0, 2.0);
    let mut archive = ReferenceArchive::new(4);
    let initial_stage = archive.stage_ranked_admissions(0, &[initial]).unwrap();
    assert_stage_receipt(&initial_stage, 0, 0, 1, &[(ExactGene([10, 100]), 10, 0, 0)]);
    assert_eq!(
        archive.combined_commit(0),
        Ok(CommitReceipt {
            completed_generation: 0,
            next_generation: 1,
            previous_committed_count: 0,
            committed_count: 1,
        })
    );

    let snapshot_at_start_of_g = archive.neighbor_snapshot_at_generation_start(1).unwrap();
    assert_eq!(snapshot_at_start_of_g.generation, 1);
    assert_eq!(
        archive_record_keys(&snapshot_at_start_of_g.records),
        vec![(ExactGene([10, 100]), 10, 0, 0)]
    );

    let candidates = [
        archive_candidate([30, 300], 30, 2, 10.0, 2.0, 3.0),
        archive_candidate([20, 200], 20, 1, 20.0, 3.0, 4.0),
    ];
    let staged = archive.stage_ranked_admissions(1, &candidates).unwrap();
    assert_stage_receipt(
        &staged,
        1,
        1,
        3,
        &[
            (ExactGene([20, 200]), 20, 1, 1),
            (ExactGene([30, 300]), 30, 2, 1),
        ],
    );
    assert_eq!(archive.staged_count(), 2);
    assert_eq!(archive.current_generation(), 1);
    assert_eq!(archive.committed_count(), 1);
    assert_eq!(
        archive_record_keys(archive.committed_records()),
        vec![(ExactGene([10, 100]), 10, 0, 0)]
    );
    assert_eq!(
        archive_record_keys(
            &archive
                .neighbor_snapshot_at_generation_start(1)
                .unwrap()
                .records
        ),
        vec![(ExactGene([10, 100]), 10, 0, 0)]
    );

    assert_eq!(
        archive.combined_commit(1),
        Ok(CommitReceipt {
            completed_generation: 1,
            next_generation: 2,
            previous_committed_count: 1,
            committed_count: 3,
        })
    );
    assert_eq!(archive.current_generation(), 2);
    assert_eq!(archive.committed_count(), 3);
    assert_eq!(archive.staged_count(), 0);
    assert_eq!(
        archive.neighbor_snapshot_at_generation_start(1),
        Err(ArchiveError::GenerationMismatch {
            current: 2,
            requested: 1,
        })
    );
    assert_eq!(
        archive_record_keys(
            &archive
                .neighbor_snapshot_at_generation_start(2)
                .unwrap()
                .records
        ),
        vec![
            (ExactGene([10, 100]), 10, 0, 0),
            (ExactGene([20, 200]), 20, 1, 1),
            (ExactGene([30, 300]), 30, 2, 1),
        ]
    );
    assert_eq!(
        archive_record_keys(&snapshot_at_start_of_g.records),
        vec![(ExactGene([10, 100]), 10, 0, 0)]
    );

    let mut faulted_archive = ReferenceArchive::new(2);
    let faulted_tail = faulted_archive
        .stage_ranked_admissions(0, &[archive_candidate([90, 900], 90, 0, 10.0, 1.0, 1.0)])
        .unwrap();
    assert_stage_receipt(&faulted_tail, 0, 0, 1, &[(ExactGene([90, 900]), 90, 0, 0)]);
    assert_eq!(faulted_archive.staged_count(), 1);
    assert_eq!(faulted_archive.committed_count(), 0);
    assert_eq!(faulted_archive.current_generation(), 0);
    assert!(faulted_archive.committed_records().is_empty());
    assert_eq!(
        faulted_archive.fault_staged_transaction_before_commit(0),
        Ok(())
    );
    assert_eq!(faulted_archive.faulted_generation(), Some(0));
    assert_eq!(
        faulted_archive.combined_commit(0),
        Err(ArchiveError::TransactionFaulted { generation: 0 })
    );
    assert_eq!(
        faulted_archive
            .stage_ranked_admissions(0, &[archive_candidate([91, 901], 91, 1, 20.0, 1.0, 1.0,)],),
        Err(ArchiveError::TransactionFaulted { generation: 0 })
    );
    assert_eq!(
        faulted_archive.combined_commit(0),
        Err(ArchiveError::TransactionFaulted { generation: 0 })
    );
    assert_eq!(faulted_archive.faulted_generation(), Some(0));
    assert_eq!(faulted_archive.staged_count(), 1);
    assert_eq!(faulted_archive.committed_count(), 0);
    assert_eq!(faulted_archive.current_generation(), 0);
    assert!(faulted_archive.committed_records().is_empty());
}

#[test]
fn r3_admission_uses_total_rank_order_independent_of_input_order() {
    let low_score = archive_candidate([40, 400], 20, 7, 10.0, 1.0, 1.0);
    let high_identity = archive_candidate([30, 300], 30, 5, 20.0, 1.0, 1.0);
    let later_ordinal = archive_candidate([20, 200], 10, 9, 20.0, 1.0, 1.0);
    let earlier_ordinal = archive_candidate([10, 100], 10, 2, 20.0, 1.0, 1.0);
    let expected = [
        (ExactGene([10, 100]), 10, 2, 0),
        (ExactGene([20, 200]), 10, 9, 0),
        (ExactGene([30, 300]), 30, 5, 0),
        (ExactGene([40, 400]), 20, 7, 0),
    ];

    let first_input = [
        low_score.clone(),
        high_identity.clone(),
        later_ordinal.clone(),
        earlier_ordinal.clone(),
    ];
    let second_input = [later_ordinal, low_score, earlier_ordinal, high_identity];
    let mut first_archive = ReferenceArchive::new(4);
    let mut second_archive = ReferenceArchive::new(4);
    let first_stage = first_archive
        .stage_ranked_admissions(0, &first_input)
        .unwrap();
    let second_stage = second_archive
        .stage_ranked_admissions(0, &second_input)
        .unwrap();

    assert_stage_receipt(&first_stage, 0, 0, 4, &expected);
    assert_stage_receipt(&second_stage, 0, 0, 4, &expected);
    first_archive.combined_commit(0).unwrap();
    second_archive.combined_commit(0).unwrap();
    assert_eq!(
        archive_record_keys(first_archive.committed_records()),
        expected
    );
    assert_eq!(
        archive_record_keys(second_archive.committed_records()),
        expected
    );
}

#[test]
fn r3_trade_count_and_net_are_distinct_strictly_positive_gates() {
    let smallest_positive = f64::from_bits(1);
    let candidates = [
        archive_candidate([10, 100], 10, 0, 80.0, 0.0, 1.0),
        archive_candidate([20, 200], 20, 1, 70.0, -0.0, 1.0),
        archive_candidate([30, 300], 30, 2, 60.0, -1.0, 1.0),
        archive_candidate([40, 400], 40, 3, 50.0, 1.0, 0.0),
        archive_candidate([50, 500], 50, 4, 40.0, 1.0, -0.0),
        archive_candidate([60, 600], 60, 5, 30.0, 1.0, -1.0),
        archive_candidate([70, 700], 70, 6, 20.0, smallest_positive, 1.0),
        archive_candidate([80, 800], 80, 7, 10.0, 1.0, smallest_positive),
    ];
    let mut archive = ReferenceArchive::new(8);

    let staged = archive.stage_ranked_admissions(0, &candidates).unwrap();

    assert_stage_receipt(
        &staged,
        0,
        0,
        2,
        &[
            (ExactGene([70, 700]), 70, 6, 0),
            (ExactGene([80, 800]), 80, 7, 0),
        ],
    );
    archive.combined_commit(0).unwrap();
    assert_eq!(
        archive_record_keys(archive.committed_records()),
        vec![
            (ExactGene([70, 700]), 70, 6, 0),
            (ExactGene([80, 800]), 80, 7, 0),
        ]
    );
}

#[test]
fn r3_nonfinite_metric_faults_the_whole_transaction_before_staging() {
    let canonical_quiet_nan = f64::from_bits(0x7ff8_0000_0000_0000);
    for (metric_slot, nonfinite, invalid_trade_count) in [
        (10_usize, canonical_quiet_nan, 2.0_f64),
        (7_usize, f64::INFINITY, 0.0_f64),
        (9_usize, f64::NEG_INFINITY, 2.0_f64),
    ] {
        let mut archive = ReferenceArchive::new(4);
        let committed_prefix = archive_candidate([10, 100], 10, 9, 40.0, 2.0, 2.0);
        archive
            .stage_ranked_admissions(0, &[committed_prefix])
            .unwrap();
        archive.combined_commit(0).unwrap();

        let eligible_before_fault = archive_candidate([20, 200], 20, 0, 30.0, 2.0, 2.0);
        let mut invalid_candidate =
            archive_candidate([30, 300], 30, 1, 20.0, 2.0, invalid_trade_count);
        invalid_candidate.metrics[metric_slot] = nonfinite;
        let eligible_after_fault = archive_candidate([40, 400], 40, 2, 10.0, 2.0, 2.0);

        assert_eq!(
            archive.stage_ranked_admissions(
                1,
                &[
                    eligible_before_fault,
                    invalid_candidate,
                    eligible_after_fault,
                ],
            ),
            Err(ArchiveError::NonFiniteMetric {
                population_ordinal: 1,
                metric_slot,
            })
        );
        assert_eq!(archive.faulted_generation(), Some(1));
        assert_eq!(archive.current_generation(), 1);
        assert_eq!(archive.committed_count(), 1);
        assert_eq!(archive.staged_count(), 0);
        assert_eq!(
            archive_record_keys(archive.committed_records()),
            vec![(ExactGene([10, 100]), 10, 9, 0)]
        );
        assert_eq!(
            archive_record_keys(
                &archive
                    .neighbor_snapshot_at_generation_start(1)
                    .unwrap()
                    .records
            ),
            vec![(ExactGene([10, 100]), 10, 9, 0)]
        );
        assert_eq!(
            archive.combined_commit(1),
            Err(ArchiveError::TransactionFaulted { generation: 1 })
        );
        assert_eq!(archive.current_generation(), 1);
        assert_eq!(archive.committed_count(), 1);
        assert_eq!(archive.staged_count(), 0);
        assert_eq!(
            archive_record_keys(archive.committed_records()),
            vec![(ExactGene([10, 100]), 10, 9, 0)]
        );
    }
}

#[test]
fn r3_rank_first_seen_duplicates_preserve_authority_without_consuming_capacity() {
    let first_ranked_metrics = [
        2.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 2.0, 19.0, 20.0,
    ];
    let later_ranked_duplicate_metrics = [
        4.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 7.0, 29.0, 30.0,
    ];
    let other_unique_metrics = [
        2.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 2.0, 39.0, 40.0,
    ];
    let one_field_different_metrics = [
        1.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 1.0, 49.0, 50.0,
    ];
    let later_ranked_duplicate =
        archive_candidate_with_metric_row([10, 100], 5, 1, 20.0, later_ranked_duplicate_metrics);
    let other_unique =
        archive_candidate_with_metric_row([20, 200], 20, 2, 10.0, other_unique_metrics);
    let one_field_different_gene =
        archive_candidate_with_metric_row([10, 101], 10, 4, 15.0, one_field_different_metrics);
    let first_ranked_gene =
        archive_candidate_with_metric_row([10, 100], 100, 9, 30.0, first_ranked_metrics);
    let mut archive = ReferenceArchive::new(4);

    let generation_zero = archive
        .stage_ranked_admissions(
            0,
            &[
                later_ranked_duplicate,
                other_unique,
                one_field_different_gene,
                first_ranked_gene,
            ],
        )
        .unwrap();
    assert_stage_receipt(
        &generation_zero,
        0,
        0,
        3,
        &[
            (ExactGene([10, 100]), 100, 9, 0),
            (ExactGene([10, 101]), 10, 4, 0),
            (ExactGene([20, 200]), 20, 2, 0),
        ],
    );
    assert_record_metric_row(
        &generation_zero.records[0],
        ExactGene([10, 100]),
        first_ranked_metrics,
    );
    assert_record_metric_row(
        &generation_zero.records[1],
        ExactGene([10, 101]),
        one_field_different_metrics,
    );
    assert_record_metric_row(
        &generation_zero.records[2],
        ExactGene([20, 200]),
        other_unique_metrics,
    );
    assert_ne!(
        archive_metric_bits(&generation_zero.records[0]),
        metric_row_bits(later_ranked_duplicate_metrics)
    );
    archive.combined_commit(0).unwrap();
    assert_record_metric_row(
        &archive.committed_records()[0],
        ExactGene([10, 100]),
        first_ranked_metrics,
    );
    assert_record_metric_row(
        &archive.committed_records()[1],
        ExactGene([10, 101]),
        one_field_different_metrics,
    );
    assert_record_metric_row(
        &archive.committed_records()[2],
        ExactGene([20, 200]),
        other_unique_metrics,
    );

    let committed_duplicate_metrics = [
        99.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 11.0, 59.0, 60.0,
    ];
    let first_new_unique_metrics = [
        2.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 2.0, 69.0, 70.0,
    ];
    let staged_duplicate_metrics = [
        7.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 8.0, 79.0, 80.0,
    ];
    let committed_duplicate =
        archive_candidate_with_metric_row([10, 100], 1, 0, 100.0, committed_duplicate_metrics);
    let first_new_unique =
        archive_candidate_with_metric_row([30, 300], 30, 5, 90.0, first_new_unique_metrics);
    let staged_duplicate =
        archive_candidate_with_metric_row([30, 300], 2, 1, 80.0, staged_duplicate_metrics);
    let generation_one = archive
        .stage_ranked_admissions(
            1,
            &[committed_duplicate, first_new_unique, staged_duplicate],
        )
        .unwrap();
    assert_stage_receipt(
        &generation_one,
        1,
        3,
        4,
        &[(ExactGene([30, 300]), 30, 5, 1)],
    );
    assert_record_metric_row(
        &generation_one.records[0],
        ExactGene([30, 300]),
        first_new_unique_metrics,
    );
    assert_ne!(
        archive_metric_bits(&generation_one.records[0]),
        metric_row_bits(staged_duplicate_metrics)
    );
    archive.combined_commit(1).unwrap();

    assert_eq!(
        archive_record_keys(archive.committed_records()),
        vec![
            (ExactGene([10, 100]), 100, 9, 0),
            (ExactGene([10, 101]), 10, 4, 0),
            (ExactGene([20, 200]), 20, 2, 0),
            (ExactGene([30, 300]), 30, 5, 1),
        ]
    );
    assert_record_metric_row(
        &archive.committed_records()[0],
        ExactGene([10, 100]),
        first_ranked_metrics,
    );
    assert_record_metric_row(
        &archive.committed_records()[1],
        ExactGene([10, 101]),
        one_field_different_metrics,
    );
    assert_record_metric_row(
        &archive.committed_records()[2],
        ExactGene([20, 200]),
        other_unique_metrics,
    );
    assert_record_metric_row(
        &archive.committed_records()[3],
        ExactGene([30, 300]),
        first_new_unique_metrics,
    );
    assert_ne!(
        archive_metric_bits(&archive.committed_records()[0]),
        metric_row_bits(committed_duplicate_metrics)
    );
    assert_ne!(
        archive_metric_bits(&archive.committed_records()[3]),
        metric_row_bits(staged_duplicate_metrics)
    );
}

#[test]
fn r3_cap_minus_one_admits_only_the_earliest_unique_and_full_cap_is_immutable() {
    let cap_prefix_metrics = [
        3.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 4.0, 109.0, 110.0,
    ];
    let cap_prefix_metric_bits = metric_row_bits(cap_prefix_metrics);
    let committed_prefix = (0_u64..49_999)
        .map(|ordinal| ArchiveRecord {
            exact_gene: ExactGene([ordinal, 1_000_000 + ordinal]),
            full_gene_hash: ordinal,
            gene_identity: 100_000 + ordinal,
            population_ordinal: (ordinal % 200) as u32,
            admitted_generation: (ordinal / 200) as u32,
            metric_bits: cap_prefix_metric_bits,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        committed_prefix.first(),
        Some(&ArchiveRecord {
            exact_gene: ExactGene([0, 1_000_000]),
            full_gene_hash: 0,
            gene_identity: 100_000,
            population_ordinal: 0,
            admitted_generation: 0,
            metric_bits: cap_prefix_metric_bits,
        })
    );
    assert_eq!(
        committed_prefix.last(),
        Some(&ArchiveRecord {
            exact_gene: ExactGene([49_998, 1_049_998]),
            full_gene_hash: 49_998,
            gene_identity: 149_998,
            population_ordinal: 198,
            admitted_generation: 249,
            metric_bits: cap_prefix_metric_bits,
        })
    );
    let mut archive =
        ReferenceArchive::from_committed_fixture(50_000, 250, committed_prefix.clone());
    assert_eq!(archive.committed_count(), 49_999);

    let later_eligible_metrics = [
        5.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 6.0, 209.0, 210.0,
    ];
    let committed_duplicate_metrics = [
        9.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 10.0, 219.0, 220.0,
    ];
    let earliest_eligible_metrics = [
        2.0, 221.0, 222.0, 223.0, 224.0, 225.0, 226.0, 227.0, 2.0, 229.0, 230.0,
    ];
    let later_eligible = archive_candidate_with_metric_row(
        [60_000, 1_060_000],
        300,
        3,
        80.0,
        later_eligible_metrics,
    );
    let committed_duplicate = archive_candidate_with_metric_row(
        [0, 1_000_000],
        1,
        199,
        100.0,
        committed_duplicate_metrics,
    );
    let earliest_eligible = archive_candidate_with_metric_row(
        [50_000, 1_050_000],
        200,
        2,
        90.0,
        earliest_eligible_metrics,
    );
    assert_ne!(
        metric_row_bits(committed_duplicate_metrics),
        cap_prefix_metric_bits
    );
    let cap_minus_one = archive
        .stage_ranked_admissions(
            250,
            &[later_eligible, committed_duplicate, earliest_eligible],
        )
        .unwrap();
    assert_stage_receipt(
        &cap_minus_one,
        250,
        49_999,
        50_000,
        &[(ExactGene([50_000, 1_050_000]), 200, 2, 250)],
    );
    assert_record_metric_row(
        &cap_minus_one.records[0],
        ExactGene([50_000, 1_050_000]),
        earliest_eligible_metrics,
    );
    assert_eq!(archive.committed_count(), 49_999);
    assert_eq!(archive.committed_records(), committed_prefix);
    assert_eq!(
        archive.combined_commit(250),
        Ok(CommitReceipt {
            completed_generation: 250,
            next_generation: 251,
            previous_committed_count: 49_999,
            committed_count: 50_000,
        })
    );
    assert_eq!(archive.committed_count(), 50_000);
    assert_eq!(&archive.committed_records()[..49_999], committed_prefix);
    assert_eq!(
        archive.committed_records().last(),
        Some(&ArchiveRecord {
            exact_gene: ExactGene([50_000, 1_050_000]),
            full_gene_hash: 50_000,
            gene_identity: 200,
            population_ordinal: 2,
            admitted_generation: 250,
            metric_bits: metric_row_bits(earliest_eligible_metrics),
        })
    );
    assert_record_metric_row(
        archive.committed_records().last().unwrap(),
        ExactGene([50_000, 1_050_000]),
        earliest_eligible_metrics,
    );
    let full_archive = archive.committed_records().to_vec();

    let full_cap_candidate_metrics = [
        7.0, 231.0, 232.0, 233.0, 234.0, 235.0, 236.0, 237.0, 8.0, 239.0, 240.0,
    ];
    let full_stage = archive
        .stage_ranked_admissions(
            251,
            &[archive_candidate_with_metric_row(
                [70_000, 1_070_000],
                400,
                5,
                100.0,
                full_cap_candidate_metrics,
            )],
        )
        .unwrap();
    assert_stage_receipt(&full_stage, 251, 50_000, 50_000, &[]);
    assert_eq!(archive.committed_count(), 50_000);
    assert_eq!(archive.committed_records(), full_archive);
    assert_eq!(
        archive.combined_commit(251),
        Ok(CommitReceipt {
            completed_generation: 251,
            next_generation: 252,
            previous_committed_count: 50_000,
            committed_count: 50_000,
        })
    );
    assert_eq!(archive.current_generation(), 252);
    assert_eq!(archive.committed_count(), 50_000);
    assert_eq!(archive.committed_records(), full_archive);
}

#[test]
fn r4_equal_hash_uses_exact_gene_and_preserves_rank_first_authority() {
    const FORCED_COLLISION_HASH: u64 = 0x0ddb_a11a_5eed_c011;
    const UNEQUAL_HASH: u64 = 0xcafe_babe_0000_0002;
    const CAPACITY_WITH_ONE_SPARE_SLOT: usize = 3;

    let first_generation_metrics = [
        2.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 3.0, 19.0, 20.0,
    ];
    let colliding_unique_metrics = [
        4.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 5.0, 29.0, 30.0,
    ];
    let later_duplicate_metrics = [
        6.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 7.0, 39.0, 40.0,
    ];
    let unequal_hash_metrics = [
        8.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 9.0, 49.0, 50.0,
    ];
    let first_generation = archive_candidate_with_forced_full_gene_hash(
        [0x1111_2222_3333_4444, 0xaaaa_bbbb_cccc_dddd],
        FORCED_COLLISION_HASH,
        100,
        4,
        30.0,
        first_generation_metrics,
    );
    let colliding_unique = archive_candidate_with_forced_full_gene_hash(
        [0x1111_2222_3333_4444, 0xaaaa_bbbb_cccc_ddde],
        FORCED_COLLISION_HASH,
        200,
        7,
        40.0,
        colliding_unique_metrics,
    );
    let later_duplicate = archive_candidate_with_forced_full_gene_hash(
        [0x1111_2222_3333_4444, 0xaaaa_bbbb_cccc_ddde],
        FORCED_COLLISION_HASH,
        1,
        0,
        20.0,
        later_duplicate_metrics,
    );
    let unequal_hash_control = archive_candidate_with_forced_full_gene_hash(
        [0x5555_6666_7777_8888, 0xeeee_ffff_0000_1111],
        UNEQUAL_HASH,
        300,
        8,
        10.0,
        unequal_hash_metrics,
    );

    assert_ne!(first_generation.exact_gene, colliding_unique.exact_gene);
    assert_eq!(
        first_generation.full_gene_hash,
        colliding_unique.full_gene_hash
    );
    assert_eq!(colliding_unique.exact_gene, later_duplicate.exact_gene);
    assert_eq!(
        colliding_unique.full_gene_hash,
        later_duplicate.full_gene_hash
    );
    assert_ne!(colliding_unique.exact_gene, unequal_hash_control.exact_gene);
    assert_ne!(
        colliding_unique.full_gene_hash,
        unequal_hash_control.full_gene_hash
    );
    assert_ne!(
        colliding_unique.gene_identity,
        later_duplicate.gene_identity
    );
    assert_ne!(
        colliding_unique.population_ordinal,
        later_duplicate.population_ordinal
    );
    assert_ne!(
        colliding_unique.score.to_bits(),
        later_duplicate.score.to_bits()
    );
    assert!(
        metric_row_bits(colliding_unique_metrics)
            .iter()
            .zip(metric_row_bits(later_duplicate_metrics))
            .all(|(rank_first, duplicate)| rank_first != &duplicate)
    );

    let generation_one_orders = [
        vec![later_duplicate.clone(), colliding_unique.clone()],
        vec![colliding_unique.clone(), later_duplicate.clone()],
    ];
    let mut committed_orders = Vec::with_capacity(generation_one_orders.len());
    for generation_one_candidates in generation_one_orders {
        let mut archive = ReferenceArchive::new(CAPACITY_WITH_ONE_SPARE_SLOT);
        let first_stage = archive
            .stage_ranked_admissions(0, std::slice::from_ref(&first_generation))
            .unwrap();
        assert_stage_receipt(
            &first_stage,
            0,
            0,
            1,
            &[(first_generation.exact_gene, 100, 4, 0)],
        );
        assert_eq!(first_stage.full_gene_hash_collision_count, 0);
        assert_eq!(archive.faulted_generation(), None);
        assert_eq!(
            archive.combined_commit(0),
            Ok(CommitReceipt {
                completed_generation: 0,
                next_generation: 1,
                previous_committed_count: 0,
                committed_count: 1,
            })
        );
        assert_eq!(archive.committed_full_gene_hash_collision_count(), 0);
        let committed_first_generation = archive.committed_records()[0].clone();
        assert_eq!(
            committed_first_generation.full_gene_hash,
            FORCED_COLLISION_HASH
        );
        assert_record_metric_row(
            &committed_first_generation,
            first_generation.exact_gene,
            first_generation_metrics,
        );
        let generation_one_snapshot = archive.neighbor_snapshot_at_generation_start(1).unwrap();
        assert_eq!(
            generation_one_snapshot.records,
            vec![committed_first_generation.clone()]
        );

        let collision_stage = archive
            .stage_ranked_admissions(1, &generation_one_candidates)
            .unwrap();
        assert_stage_receipt(
            &collision_stage,
            1,
            1,
            2,
            &[(colliding_unique.exact_gene, 200, 7, 1)],
        );
        assert_eq!(collision_stage.records.len(), 1);
        assert_eq!(archive.staged_count(), 1);
        assert_eq!(collision_stage.full_gene_hash_collision_count, 1);
        assert_eq!(archive.committed_full_gene_hash_collision_count(), 0);
        assert_eq!(archive.committed_count(), 1);
        assert_eq!(
            archive.committed_records(),
            std::slice::from_ref(&committed_first_generation)
        );
        assert_eq!(
            archive
                .neighbor_snapshot_at_generation_start(1)
                .unwrap()
                .records,
            vec![committed_first_generation.clone()]
        );
        assert_eq!(archive.faulted_generation(), None);
        assert_eq!(
            collision_stage.records[0].full_gene_hash,
            FORCED_COLLISION_HASH
        );
        assert_record_metric_row(
            &collision_stage.records[0],
            colliding_unique.exact_gene,
            colliding_unique_metrics,
        );
        assert_ne!(
            archive_metric_bits(&collision_stage.records[0]),
            metric_row_bits(later_duplicate_metrics)
        );

        assert_eq!(
            archive.combined_commit(1),
            Ok(CommitReceipt {
                completed_generation: 1,
                next_generation: 2,
                previous_committed_count: 1,
                committed_count: 2,
            })
        );
        assert_eq!(archive.current_generation(), 2);
        assert_eq!(archive.committed_count(), 2);
        assert_eq!(archive.committed_full_gene_hash_collision_count(), 1);
        assert_eq!(archive.faulted_generation(), None);
        assert_eq!(archive.committed_records()[0], committed_first_generation);
        assert_eq!(
            archive_record_keys(archive.committed_records()),
            vec![
                (first_generation.exact_gene, 100, 4, 0),
                (colliding_unique.exact_gene, 200, 7, 1),
            ]
        );
        assert_eq!(CAPACITY_WITH_ONE_SPARE_SLOT - archive.committed_count(), 1);
        assert_eq!(
            archive.committed_records()[1].full_gene_hash,
            FORCED_COLLISION_HASH
        );
        assert_record_metric_row(
            &archive.committed_records()[1],
            colliding_unique.exact_gene,
            colliding_unique_metrics,
        );
        assert_ne!(
            archive_metric_bits(&archive.committed_records()[1]),
            metric_row_bits(later_duplicate_metrics)
        );
        assert_eq!(
            archive
                .neighbor_snapshot_at_generation_start(2)
                .unwrap()
                .records,
            archive.committed_records().to_vec()
        );

        let unequal_hash_stage = archive
            .stage_ranked_admissions(2, std::slice::from_ref(&unequal_hash_control))
            .unwrap();
        assert_stage_receipt(
            &unequal_hash_stage,
            2,
            2,
            3,
            &[(unequal_hash_control.exact_gene, 300, 8, 2)],
        );
        assert_eq!(unequal_hash_stage.full_gene_hash_collision_count, 1);
        assert_eq!(archive.committed_full_gene_hash_collision_count(), 1);
        assert_eq!(archive.committed_count(), 2);
        assert_eq!(archive.staged_count(), 1);
        assert_eq!(archive.faulted_generation(), None);
        assert_eq!(
            archive
                .neighbor_snapshot_at_generation_start(2)
                .unwrap()
                .records,
            archive.committed_records().to_vec()
        );
        assert_eq!(
            archive.combined_commit(2),
            Ok(CommitReceipt {
                completed_generation: 2,
                next_generation: 3,
                previous_committed_count: 2,
                committed_count: 3,
            })
        );
        assert_eq!(archive.committed_full_gene_hash_collision_count(), 1);
        assert_eq!(archive.faulted_generation(), None);
        assert_eq!(archive.committed_count(), 3);
        assert_eq!(archive.committed_records()[2].full_gene_hash, UNEQUAL_HASH);
        assert_record_metric_row(
            &archive.committed_records()[2],
            unequal_hash_control.exact_gene,
            unequal_hash_metrics,
        );
        committed_orders.push(archive.committed_records().to_vec());
    }

    assert_eq!(committed_orders[0], committed_orders[1]);
}

#[test]
fn r4_unequal_hash_admits_without_incrementing_the_collision_counter() {
    const FIRST_HASH: u64 = 0x0123_4567_89ab_cdef;
    const UNEQUAL_HASH: u64 = 0xfedc_ba98_7654_3210;
    let first_metrics = [
        2.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 3.0, 49.0, 50.0,
    ];
    let unequal_hash_metrics = [
        4.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 5.0, 59.0, 60.0,
    ];
    let first = archive_candidate_with_forced_full_gene_hash(
        [0x1000, 0x2000],
        FIRST_HASH,
        10,
        0,
        20.0,
        first_metrics,
    );
    let unequal_hash = archive_candidate_with_forced_full_gene_hash(
        [0x3000, 0x4000],
        UNEQUAL_HASH,
        20,
        1,
        10.0,
        unequal_hash_metrics,
    );
    assert_ne!(first.exact_gene, unequal_hash.exact_gene);
    assert_ne!(first.full_gene_hash, unequal_hash.full_gene_hash);

    let mut archive = ReferenceArchive::new(3);
    let first_stage = archive
        .stage_ranked_admissions(0, std::slice::from_ref(&first))
        .unwrap();
    assert_eq!(first_stage.full_gene_hash_collision_count, 0);
    archive.combined_commit(0).unwrap();
    assert_eq!(archive.committed_full_gene_hash_collision_count(), 0);

    let unequal_hash_stage = archive
        .stage_ranked_admissions(1, std::slice::from_ref(&unequal_hash))
        .unwrap();
    assert_stage_receipt(
        &unequal_hash_stage,
        1,
        1,
        2,
        &[(unequal_hash.exact_gene, 20, 1, 1)],
    );
    assert_eq!(unequal_hash_stage.full_gene_hash_collision_count, 0);
    assert_eq!(archive.committed_full_gene_hash_collision_count(), 0);
    assert_eq!(archive.faulted_generation(), None);
    assert_eq!(
        archive.combined_commit(1),
        Ok(CommitReceipt {
            completed_generation: 1,
            next_generation: 2,
            previous_committed_count: 1,
            committed_count: 2,
        })
    );
    assert_eq!(archive.committed_full_gene_hash_collision_count(), 0);
    assert_eq!(archive.faulted_generation(), None);
    assert_eq!(archive.committed_count(), 2);
    assert_eq!(archive.committed_records()[0].full_gene_hash, FIRST_HASH);
    assert_eq!(archive.committed_records()[1].full_gene_hash, UNEQUAL_HASH);
    assert_record_metric_row(
        &archive.committed_records()[1],
        unequal_hash.exact_gene,
        unequal_hash_metrics,
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetricValueClass {
    QNaN,
    PosInf,
}

impl MetricValueClass {
    const fn ordinal(self) -> usize {
        match self {
            Self::QNaN => 0,
            Self::PosInf => 1,
        }
    }

    const fn bits(self) -> u64 {
        match self {
            Self::QNaN => 0x7ff8_0000_0000_0000,
            Self::PosInf => 0x7ff0_0000_0000_0000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MetricFaultCase {
    case_ordinal: u8,
    metric_slot: u8,
    value_class: MetricValueClass,
}

const R8_METRIC_FAULT_CASES: [MetricFaultCase; 22] = [
    MetricFaultCase {
        case_ordinal: 0,
        metric_slot: 0,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 1,
        metric_slot: 0,
        value_class: MetricValueClass::PosInf,
    },
    MetricFaultCase {
        case_ordinal: 2,
        metric_slot: 1,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 3,
        metric_slot: 1,
        value_class: MetricValueClass::PosInf,
    },
    MetricFaultCase {
        case_ordinal: 4,
        metric_slot: 2,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 5,
        metric_slot: 2,
        value_class: MetricValueClass::PosInf,
    },
    MetricFaultCase {
        case_ordinal: 6,
        metric_slot: 3,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 7,
        metric_slot: 3,
        value_class: MetricValueClass::PosInf,
    },
    MetricFaultCase {
        case_ordinal: 8,
        metric_slot: 4,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 9,
        metric_slot: 4,
        value_class: MetricValueClass::PosInf,
    },
    MetricFaultCase {
        case_ordinal: 10,
        metric_slot: 5,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 11,
        metric_slot: 5,
        value_class: MetricValueClass::PosInf,
    },
    MetricFaultCase {
        case_ordinal: 12,
        metric_slot: 6,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 13,
        metric_slot: 6,
        value_class: MetricValueClass::PosInf,
    },
    MetricFaultCase {
        case_ordinal: 14,
        metric_slot: 7,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 15,
        metric_slot: 7,
        value_class: MetricValueClass::PosInf,
    },
    MetricFaultCase {
        case_ordinal: 16,
        metric_slot: 8,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 17,
        metric_slot: 8,
        value_class: MetricValueClass::PosInf,
    },
    MetricFaultCase {
        case_ordinal: 18,
        metric_slot: 9,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 19,
        metric_slot: 9,
        value_class: MetricValueClass::PosInf,
    },
    MetricFaultCase {
        case_ordinal: 20,
        metric_slot: 10,
        value_class: MetricValueClass::QNaN,
    },
    MetricFaultCase {
        case_ordinal: 21,
        metric_slot: 10,
        value_class: MetricValueClass::PosInf,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StructuralFaultCase {
    SignatureWordCountLow {
        ordinal: u8,
        expected: u32,
        observed: u32,
    },
    SignatureWordCountHigh {
        ordinal: u8,
        expected: u32,
        observed: u32,
    },
    ZeroUnion {
        ordinal: u8,
        observed: u32,
        allowed_min: u32,
        sealed_max: u32,
    },
    ArchiveCountOverflow {
        ordinal: u8,
        sealed_capacity: u32,
        observed: u32,
    },
    BoxedReceiptTokenMismatch {
        ordinal: u8,
        observed_xor_mask: u64,
    },
    ComparatorBoundMismatch {
        ordinal: u8,
        sealed_union_max: u64,
        sealed_cross_product_max: u64,
        observed_union_max: u64,
        observed_cross_product_max: u64,
    },
}

impl StructuralFaultCase {
    const fn ordinal(self) -> usize {
        match self {
            Self::SignatureWordCountLow { ordinal, .. }
            | Self::SignatureWordCountHigh { ordinal, .. }
            | Self::ZeroUnion { ordinal, .. }
            | Self::ArchiveCountOverflow { ordinal, .. }
            | Self::BoxedReceiptTokenMismatch { ordinal, .. }
            | Self::ComparatorBoundMismatch { ordinal, .. } => ordinal as usize,
        }
    }
}

const R8_STRUCTURAL_FAULT_CASES: [StructuralFaultCase; 6] = [
    StructuralFaultCase::SignatureWordCountLow {
        ordinal: 0,
        expected: 4,
        observed: 3,
    },
    StructuralFaultCase::SignatureWordCountHigh {
        ordinal: 1,
        expected: 4,
        observed: 5,
    },
    StructuralFaultCase::ZeroUnion {
        ordinal: 2,
        observed: 0,
        allowed_min: 1,
        sealed_max: 32,
    },
    StructuralFaultCase::ArchiveCountOverflow {
        ordinal: 3,
        sealed_capacity: 50_000,
        observed: 50_001,
    },
    StructuralFaultCase::BoxedReceiptTokenMismatch {
        ordinal: 4,
        observed_xor_mask: 1_u64,
    },
    StructuralFaultCase::ComparatorBoundMismatch {
        ordinal: 5,
        sealed_union_max: 32,
        sealed_cross_product_max: 1_024,
        observed_union_max: 33,
        observed_cross_product_max: 1_024,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum R8TypedFaultV10 {
    Metric(MetricFaultCase),
    Structural(StructuralFaultCase),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct R8FirstFaultContractV10 {
    primary: R8TypedFaultV10,
    later_overwrite_probe: R8TypedFaultV10,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct R8AtomicNonPublicationContractV10 {
    commit_word_unchanged: bool,
    generation_unchanged: bool,
    current_store_unchanged: bool,
    commit_epoch_unchanged: bool,
    archive_count_unchanged: bool,
    staged_tail_unreachable: bool,
    allocation_ledger_unchanged: bool,
}

const R8_ATOMIC_NON_PUBLICATION: R8AtomicNonPublicationContractV10 =
    R8AtomicNonPublicationContractV10 {
        commit_word_unchanged: true,
        generation_unchanged: true,
        current_store_unchanged: true,
        commit_epoch_unchanged: true,
        archive_count_unchanged: true,
        staged_tail_unreachable: true,
        allocation_ledger_unchanged: true,
    };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MetricFaultContractV10 {
    cases: [MetricFaultCase; 22],
    atomic: R8AtomicNonPublicationContractV10,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StructuralFaultContractV10 {
    cases: [StructuralFaultCase; 6],
    atomic: R8AtomicNonPublicationContractV10,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CleanupStepV10 {
    ScoringArchiveArena,
    GenerationArena,
    PopulationEvaluatorSessionWorkspace,
    TrimMapCountEventArena,
    RetainedParentImportSchemaFullAdmissionOwners,
}

const R8_CLEANUP_ORDER: [CleanupStepV10; 5] = [
    CleanupStepV10::ScoringArchiveArena,
    CleanupStepV10::GenerationArena,
    CleanupStepV10::PopulationEvaluatorSessionWorkspace,
    CleanupStepV10::TrimMapCountEventArena,
    CleanupStepV10::RetainedParentImportSchemaFullAdmissionOwners,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CleanupStateContractV10 {
    EventProvedSemanticFault {
        fault: R8TypedFaultV10,
        cleanup_order: [CleanupStepV10; 5],
        native_ack_before_disarm_drop: bool,
        cleanup_calls: u8,
        second_cleanup_rejected: bool,
        reuse_rejected: bool,
    },
    NotReady {
        polls: u8,
        same_pending_owner: bool,
        same_run_token: bool,
        same_boxed_receipt_identity: bool,
        same_allocation_identities: bool,
        cleanup_delta: u8,
        poisoned: bool,
        tombstoned: bool,
    },
    UnknownCudaOutcome {
        whole_owner_poisoned: bool,
        whole_owner_retained: bool,
        detached_child_count: u8,
        cleanup_delta: u8,
        reusable: bool,
    },
    UnprovedEvent {
        plausible_terminal_bytes_ignored: bool,
        exact_event_proof: bool,
        whole_owner_poisoned: bool,
        whole_owner_retained: bool,
        cleanup_delta: u8,
        reusable: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CleanupContractV10 {
    states: [CleanupStateContractV10; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum R8FaultIdentityV10 {
    Metric(u8),
    Structural(u8),
}

const R8_FRESH_RUN_IDENTITIES: [R8FaultIdentityV10; 28] = [
    R8FaultIdentityV10::Metric(0),
    R8FaultIdentityV10::Metric(1),
    R8FaultIdentityV10::Metric(2),
    R8FaultIdentityV10::Metric(3),
    R8FaultIdentityV10::Metric(4),
    R8FaultIdentityV10::Metric(5),
    R8FaultIdentityV10::Metric(6),
    R8FaultIdentityV10::Metric(7),
    R8FaultIdentityV10::Metric(8),
    R8FaultIdentityV10::Metric(9),
    R8FaultIdentityV10::Metric(10),
    R8FaultIdentityV10::Metric(11),
    R8FaultIdentityV10::Metric(12),
    R8FaultIdentityV10::Metric(13),
    R8FaultIdentityV10::Metric(14),
    R8FaultIdentityV10::Metric(15),
    R8FaultIdentityV10::Metric(16),
    R8FaultIdentityV10::Metric(17),
    R8FaultIdentityV10::Metric(18),
    R8FaultIdentityV10::Metric(19),
    R8FaultIdentityV10::Metric(20),
    R8FaultIdentityV10::Metric(21),
    R8FaultIdentityV10::Structural(0),
    R8FaultIdentityV10::Structural(1),
    R8FaultIdentityV10::Structural(2),
    R8FaultIdentityV10::Structural(3),
    R8FaultIdentityV10::Structural(4),
    R8FaultIdentityV10::Structural(5),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreshRunIdentityContractV10 {
    run_token_distinct: bool,
    boxed_receipt_identity_distinct: bool,
    every_allocation_identity_distinct: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreshRunContractV10 {
    executed_identities: [R8FaultIdentityV10; 28],
    expected_fault_run_count: u8,
    expected_fresh_admission_count: u8,
    expected_successful_fresh_publication_count: u8,
    identity: FreshRunIdentityContractV10,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum R8ContractRequestV10 {
    MetricFaults(MetricFaultContractV10),
    StructuralFaults(StructuralFaultContractV10),
    Cleanup(CleanupContractV10),
    FreshRuns(FreshRunContractV10),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum R8ContractErrorV10 {
    R8ImplementationPending,
}

fn require_r8_required_card_v10() {
    assert!(
        std::env::var("NEOETHOS_REQUIRE_GPU").as_deref() == Ok("1"),
        "R8 required-card contract requires NEOETHOS_REQUIRE_GPU=1 and never skips"
    );
}

fn metric_fault_contract_v10() -> MetricFaultContractV10 {
    MetricFaultContractV10 {
        cases: R8_METRIC_FAULT_CASES,
        atomic: R8_ATOMIC_NON_PUBLICATION,
    }
}

fn structural_fault_contract_v10() -> StructuralFaultContractV10 {
    StructuralFaultContractV10 {
        cases: R8_STRUCTURAL_FAULT_CASES,
        atomic: R8_ATOMIC_NON_PUBLICATION,
    }
}

fn cleanup_contract_v10() -> CleanupContractV10 {
    CleanupContractV10 {
        states: [
            CleanupStateContractV10::EventProvedSemanticFault {
                fault: R8TypedFaultV10::Structural(StructuralFaultCase::ZeroUnion {
                    ordinal: 2,
                    observed: 0,
                    allowed_min: 1,
                    sealed_max: 32,
                }),
                cleanup_order: R8_CLEANUP_ORDER,
                native_ack_before_disarm_drop: true,
                cleanup_calls: 1,
                second_cleanup_rejected: true,
                reuse_rejected: true,
            },
            CleanupStateContractV10::NotReady {
                polls: 2,
                same_pending_owner: true,
                same_run_token: true,
                same_boxed_receipt_identity: true,
                same_allocation_identities: true,
                cleanup_delta: 0,
                poisoned: false,
                tombstoned: false,
            },
            CleanupStateContractV10::UnknownCudaOutcome {
                whole_owner_poisoned: true,
                whole_owner_retained: true,
                detached_child_count: 0,
                cleanup_delta: 0,
                reusable: false,
            },
            CleanupStateContractV10::UnprovedEvent {
                plausible_terminal_bytes_ignored: true,
                exact_event_proof: false,
                whole_owner_poisoned: true,
                whole_owner_retained: true,
                cleanup_delta: 0,
                reusable: false,
            },
        ],
    }
}

fn fresh_run_contract_v10() -> FreshRunContractV10 {
    FreshRunContractV10 {
        executed_identities: R8_FRESH_RUN_IDENTITIES,
        expected_fault_run_count: 28,
        expected_fresh_admission_count: 28,
        expected_successful_fresh_publication_count: 28,
        identity: FreshRunIdentityContractV10 {
            run_token_distinct: true,
            boxed_receipt_identity_distinct: true,
            every_allocation_identity_distinct: true,
        },
    }
}

fn validate_atomic_non_publication_v10(contract: R8AtomicNonPublicationContractV10) {
    assert!(contract.commit_word_unchanged);
    assert!(contract.generation_unchanged);
    assert!(contract.current_store_unchanged);
    assert!(contract.commit_epoch_unchanged);
    assert!(contract.archive_count_unchanged);
    assert!(contract.staged_tail_unreachable);
    assert!(contract.allocation_ledger_unchanged);
}

fn validate_metric_case_v10(case: MetricFaultCase) {
    assert!((case.metric_slot as usize) < 11);
    assert_eq!(
        case.case_ordinal as usize,
        2 * case.metric_slot as usize + case.value_class.ordinal()
    );
    assert_eq!(
        case.value_class.bits(),
        match case.value_class {
            MetricValueClass::QNaN => 0x7ff8_0000_0000_0000,
            MetricValueClass::PosInf => 0x7ff0_0000_0000_0000,
        }
    );
}

fn validate_structural_case_v10(case: StructuralFaultCase) {
    match case {
        StructuralFaultCase::SignatureWordCountLow {
            ordinal,
            expected,
            observed,
        } => assert_eq!((ordinal, expected, observed), (0, 4, 3)),
        StructuralFaultCase::SignatureWordCountHigh {
            ordinal,
            expected,
            observed,
        } => assert_eq!((ordinal, expected, observed), (1, 4, 5)),
        StructuralFaultCase::ZeroUnion {
            ordinal,
            observed,
            allowed_min,
            sealed_max,
        } => assert_eq!((ordinal, observed, allowed_min, sealed_max), (2, 0, 1, 32)),
        StructuralFaultCase::ArchiveCountOverflow {
            ordinal,
            sealed_capacity,
            observed,
        } => assert_eq!((ordinal, sealed_capacity, observed), (3, 50_000, 50_001)),
        StructuralFaultCase::BoxedReceiptTokenMismatch {
            ordinal,
            observed_xor_mask,
        } => assert_eq!((ordinal, observed_xor_mask), (4, 1_u64)),
        StructuralFaultCase::ComparatorBoundMismatch {
            ordinal,
            sealed_union_max,
            sealed_cross_product_max,
            observed_union_max,
            observed_cross_product_max,
        } => assert_eq!(
            (
                ordinal,
                sealed_union_max,
                sealed_cross_product_max,
                observed_union_max,
                observed_cross_product_max,
            ),
            (5, 32, 1_024, 33, 1_024)
        ),
    }
}

fn validate_typed_fault_v10(fault: R8TypedFaultV10) {
    match fault {
        R8TypedFaultV10::Metric(case) => validate_metric_case_v10(case),
        R8TypedFaultV10::Structural(case) => validate_structural_case_v10(case),
    }
}

fn first_fault_contract_v10(primary: R8TypedFaultV10) -> R8FirstFaultContractV10 {
    let later_overwrite_probe = match primary {
        R8TypedFaultV10::Structural(StructuralFaultCase::ZeroUnion { .. }) => {
            R8TypedFaultV10::Structural(StructuralFaultCase::ArchiveCountOverflow {
                ordinal: 3,
                sealed_capacity: 50_000,
                observed: 50_001,
            })
        }
        _ => R8TypedFaultV10::Structural(StructuralFaultCase::ZeroUnion {
            ordinal: 2,
            observed: 0,
            allowed_min: 1,
            sealed_max: 32,
        }),
    };
    R8FirstFaultContractV10 {
        primary,
        later_overwrite_probe,
    }
}

fn validate_first_fault_contract_v10(contract: R8FirstFaultContractV10) {
    validate_typed_fault_v10(contract.primary);
    validate_typed_fault_v10(contract.later_overwrite_probe);
    assert_ne!(contract.primary, contract.later_overwrite_probe);
    match contract.primary {
        R8TypedFaultV10::Structural(StructuralFaultCase::ZeroUnion { .. }) => assert!(matches!(
            contract.later_overwrite_probe,
            R8TypedFaultV10::Structural(StructuralFaultCase::ArchiveCountOverflow {
                observed: 50_001,
                ..
            })
        )),
        _ => assert!(matches!(
            contract.later_overwrite_probe,
            R8TypedFaultV10::Structural(StructuralFaultCase::ZeroUnion { .. })
        )),
    }
}

fn validate_metric_fault_contract_v10(contract: &MetricFaultContractV10) {
    let mut seen = [[false; 11]; 2];
    for (expected_ordinal, case) in contract.cases.into_iter().enumerate() {
        validate_metric_case_v10(case);
        assert_eq!(case.case_ordinal as usize, expected_ordinal);
        let cell = &mut seen[case.value_class.ordinal()][case.metric_slot as usize];
        assert!(!*cell, "duplicate R8 metric fault identity");
        *cell = true;
        validate_first_fault_contract_v10(first_fault_contract_v10(R8TypedFaultV10::Metric(case)));
    }
    assert!(seen.into_iter().flatten().all(|was_seen| was_seen));
    validate_atomic_non_publication_v10(contract.atomic);
}

fn validate_structural_fault_contract_v10(contract: &StructuralFaultContractV10) {
    let mut seen = [false; 6];
    for (expected_ordinal, case) in contract.cases.into_iter().enumerate() {
        validate_structural_case_v10(case);
        assert_eq!(case.ordinal(), expected_ordinal);
        let cell = &mut seen[case.ordinal()];
        assert!(!*cell, "duplicate R8 structural fault identity");
        *cell = true;
        validate_first_fault_contract_v10(first_fault_contract_v10(R8TypedFaultV10::Structural(
            case,
        )));
    }
    assert!(seen.into_iter().all(|was_seen| was_seen));
    validate_atomic_non_publication_v10(contract.atomic);
}

fn validate_cleanup_contract_v10(contract: &CleanupContractV10) {
    let mut seen = [false; 4];
    for state in contract.states {
        match state {
            CleanupStateContractV10::EventProvedSemanticFault {
                fault,
                cleanup_order,
                native_ack_before_disarm_drop,
                cleanup_calls,
                second_cleanup_rejected,
                reuse_rejected,
            } => {
                assert!(!seen[0]);
                seen[0] = true;
                validate_typed_fault_v10(fault);
                assert_eq!(cleanup_order, R8_CLEANUP_ORDER);
                assert!(native_ack_before_disarm_drop);
                assert_eq!(cleanup_calls, 1);
                assert!(second_cleanup_rejected);
                assert!(reuse_rejected);
            }
            CleanupStateContractV10::NotReady {
                polls,
                same_pending_owner,
                same_run_token,
                same_boxed_receipt_identity,
                same_allocation_identities,
                cleanup_delta,
                poisoned,
                tombstoned,
            } => {
                assert!(!seen[1]);
                seen[1] = true;
                assert_eq!(polls, 2);
                assert!(same_pending_owner);
                assert!(same_run_token);
                assert!(same_boxed_receipt_identity);
                assert!(same_allocation_identities);
                assert_eq!(cleanup_delta, 0);
                assert!(!poisoned);
                assert!(!tombstoned);
            }
            CleanupStateContractV10::UnknownCudaOutcome {
                whole_owner_poisoned,
                whole_owner_retained,
                detached_child_count,
                cleanup_delta,
                reusable,
            } => {
                assert!(!seen[2]);
                seen[2] = true;
                assert!(whole_owner_poisoned);
                assert!(whole_owner_retained);
                assert_eq!(detached_child_count, 0);
                assert_eq!(cleanup_delta, 0);
                assert!(!reusable);
            }
            CleanupStateContractV10::UnprovedEvent {
                plausible_terminal_bytes_ignored,
                exact_event_proof,
                whole_owner_poisoned,
                whole_owner_retained,
                cleanup_delta,
                reusable,
            } => {
                assert!(!seen[3]);
                seen[3] = true;
                assert!(plausible_terminal_bytes_ignored);
                assert!(!exact_event_proof);
                assert!(whole_owner_poisoned);
                assert!(whole_owner_retained);
                assert_eq!(cleanup_delta, 0);
                assert!(!reusable);
            }
        }
    }
    assert!(seen.into_iter().all(|was_seen| was_seen));
}

fn validate_fresh_run_contract_v10(contract: &FreshRunContractV10) {
    let mut executed = [0_u8; 28];
    for identity in contract.executed_identities {
        let index = match identity {
            R8FaultIdentityV10::Metric(ordinal) => {
                assert!(ordinal < 22);
                ordinal as usize
            }
            R8FaultIdentityV10::Structural(ordinal) => {
                assert!(ordinal < 6);
                22 + ordinal as usize
            }
        };
        executed[index] = executed[index]
            .checked_add(1)
            .expect("R8 executed identity counter overflow");
        assert_eq!(executed[index], 1, "duplicate R8 recovery identity");
    }
    assert!(executed.into_iter().all(|count| count == 1));
    assert_eq!(contract.expected_fault_run_count, 28);
    assert_eq!(contract.expected_fresh_admission_count, 28);
    assert_eq!(contract.expected_successful_fresh_publication_count, 28);
    assert!(contract.identity.run_token_distinct);
    assert!(contract.identity.boxed_receipt_identity_distinct);
    assert!(contract.identity.every_allocation_identity_distinct);
}

fn execute_r8_contract_v10(request: R8ContractRequestV10) -> Result<(), R8ContractErrorV10> {
    match request {
        R8ContractRequestV10::MetricFaults(contract) => {
            validate_metric_fault_contract_v10(&contract)
        }
        R8ContractRequestV10::StructuralFaults(contract) => {
            validate_structural_fault_contract_v10(&contract)
        }
        R8ContractRequestV10::Cleanup(contract) => validate_cleanup_contract_v10(&contract),
        R8ContractRequestV10::FreshRuns(contract) => validate_fresh_run_contract_v10(&contract),
    }
    Err(R8ContractErrorV10::R8ImplementationPending)
}

fn require_r8_contract_result_v10(result: Result<(), R8ContractErrorV10>) {
    match result {
        Ok(()) => {}
        Err(R8ContractErrorV10::R8ImplementationPending) => panic!("R8ImplementationPending"),
    }
}

#[test]
fn r8_all_eleven_metric_slots_reject_nan_and_infinity_atomically() {
    require_r8_required_card_v10();
    let contract = metric_fault_contract_v10();
    validate_metric_fault_contract_v10(&contract);
    require_r8_contract_result_v10(execute_r8_contract_v10(R8ContractRequestV10::MetricFaults(
        contract,
    )));
}

#[test]
fn r8_structural_fault_matrix_is_atomic() {
    require_r8_required_card_v10();
    let contract = structural_fault_contract_v10();
    validate_structural_fault_contract_v10(&contract);
    require_r8_contract_result_v10(execute_r8_contract_v10(
        R8ContractRequestV10::StructuralFaults(contract),
    ));
}

#[test]
fn r8_fault_cleanup_is_checked_once_and_owner_never_reused() {
    require_r8_required_card_v10();
    let contract = cleanup_contract_v10();
    validate_cleanup_contract_v10(&contract);
    require_r8_contract_result_v10(execute_r8_contract_v10(R8ContractRequestV10::Cleanup(
        contract,
    )));
}

#[test]
fn r8_every_recoverable_fault_allows_a_fresh_unrelated_run() {
    require_r8_required_card_v10();
    let contract = fresh_run_contract_v10();
    validate_fresh_run_contract_v10(&contract);
    require_r8_contract_result_v10(execute_r8_contract_v10(R8ContractRequestV10::FreshRuns(
        contract,
    )));
}
