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
