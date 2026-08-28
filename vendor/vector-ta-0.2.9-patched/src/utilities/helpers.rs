use crate::utilities::enums::Kernel;
use aligned_vec::AVec;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;
use std::sync::OnceLock;
use std::{mem::MaybeUninit, ptr, slice};

static BEST_SINGLE: OnceLock<Kernel> = OnceLock::new();
static BEST_BATCH: OnceLock<Kernel> = OnceLock::new();

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct X86FeatureSet {
    avx2: bool,
    fma: bool,
    avx512f: bool,
    avx512dq: bool,
    avx512vl: bool,
    avx512bw: bool,
}

#[inline(always)]
fn runtime_x86_features() -> X86FeatureSet {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        return X86FeatureSet {
            avx2: is_x86_feature_detected!("avx2"),
            fma: is_x86_feature_detected!("fma"),
            avx512f: is_x86_feature_detected!("avx512f"),
            avx512dq: is_x86_feature_detected!("avx512dq"),
            avx512vl: is_x86_feature_detected!("avx512vl"),
            avx512bw: is_x86_feature_detected!("avx512bw"),
        };
    }

    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        X86FeatureSet::default()
    }
}

#[inline(always)]
const fn kernel_is_supported_by_features(kernel: Kernel, features: X86FeatureSet) -> bool {
    match kernel {
        Kernel::Auto | Kernel::Scalar | Kernel::ScalarBatch => true,
        Kernel::Avx2 | Kernel::Avx2Batch => features.avx2 && features.fma,
        Kernel::Avx512 | Kernel::Avx512Batch => {
            features.avx2
                && features.fma
                && features.avx512f
                && features.avx512dq
                && features.avx512vl
                && features.avx512bw
        }
    }
}

#[inline(always)]
const fn best_kernel_for_x86_features(features: X86FeatureSet) -> Kernel {
    if kernel_is_supported_by_features(Kernel::Avx512, features) {
        Kernel::Avx512
    } else if kernel_is_supported_by_features(Kernel::Avx2, features) {
        Kernel::Avx2
    } else {
        Kernel::Scalar
    }
}

#[inline(always)]
const fn best_batch_kernel_for_x86_features(features: X86FeatureSet) -> Kernel {
    match best_kernel_for_x86_features(features) {
        Kernel::Avx512 => Kernel::Avx512Batch,
        Kernel::Avx2 => Kernel::Avx2Batch,
        _ => Kernel::ScalarBatch,
    }
}

/// Runtime support decision shared by the production selector and the exported
/// test-skip macro. Kept hidden from the public documentation because callers
/// should request [`Kernel::Auto`] rather than reproduce dispatch policy.
#[doc(hidden)]
#[inline(always)]
pub fn runtime_supports_kernel(kernel: Kernel) -> bool {
    kernel_is_supported_by_features(kernel, runtime_x86_features())
}

#[inline(always)]
pub fn detect_best_kernel() -> Kernel {
    *BEST_SINGLE.get_or_init(|| best_kernel_for_x86_features(runtime_x86_features()))
}

#[inline(always)]
pub fn detect_best_batch_kernel() -> Kernel {
    *BEST_BATCH.get_or_init(|| best_batch_kernel_for_x86_features(runtime_x86_features()))
}

#[cfg(target_arch = "wasm32")]
static BEST_WASM: OnceLock<Kernel> = OnceLock::new();

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub fn detect_wasm_kernel() -> Kernel {
    *BEST_WASM.get_or_init(|| {
        #[cfg(target_feature = "simd128")]
        {
            return Kernel::Scalar;
        }

        Kernel::Scalar
    })
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub fn detect_wasm_kernel() -> Kernel {
    Kernel::Scalar
}

#[macro_export]
macro_rules! skip_if_unsupported {
    ($kernel:expr, $test_name:expr) => {{
        let kernel = $kernel;
        if !$crate::utilities::helpers::runtime_supports_kernel(kernel) {
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                eprintln!(
                    "[{}] skipped {:?} – compiled without `nightly-avx`",
                    $test_name, kernel
                );
                return Ok(());
            }

            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                let need = match kernel {
                    $crate::utilities::enums::Kernel::Avx512
                    | $crate::utilities::enums::Kernel::Avx512Batch => {
                        "AVX2 + AVX-512F/DQ/VL/BW + FMA"
                    }
                    $crate::utilities::enums::Kernel::Avx2
                    | $crate::utilities::enums::Kernel::Avx2Batch => "AVX2 + FMA",
                    _ => "the requested kernel features",
                };
                eprintln!("[{}] skipped {:?} - CPU lacks {}", $test_name, kernel, need);
                return Ok(());
            }
        }
    }};
}

#[cfg(test)]
mod x86_kernel_selection_tests {
    use super::{
        X86FeatureSet, best_batch_kernel_for_x86_features, best_kernel_for_x86_features,
        kernel_is_supported_by_features,
    };
    use crate::utilities::enums::Kernel;

    const FULL: X86FeatureSet = X86FeatureSet {
        avx2: true,
        fma: true,
        avx512f: true,
        avx512dq: true,
        avx512vl: true,
        avx512bw: true,
    };
    const NONE: X86FeatureSet = X86FeatureSet {
        avx2: false,
        fma: false,
        avx512f: false,
        avx512dq: false,
        avx512vl: false,
        avx512bw: false,
    };

    #[test]
    fn complete_avx512_target_feature_union_selects_avx512() {
        assert!(kernel_is_supported_by_features(Kernel::Avx512, FULL));
        assert!(kernel_is_supported_by_features(Kernel::Avx512Batch, FULL));
        assert_eq!(best_kernel_for_x86_features(FULL), Kernel::Avx512);
        assert_eq!(
            best_batch_kernel_for_x86_features(FULL),
            Kernel::Avx512Batch
        );
    }

    #[test]
    fn each_missing_avx512_subset_feature_falls_back_to_avx2() {
        let missing_one = [
            X86FeatureSet {
                avx512f: false,
                ..FULL
            },
            X86FeatureSet {
                avx512dq: false,
                ..FULL
            },
            X86FeatureSet {
                avx512vl: false,
                ..FULL
            },
            X86FeatureSet {
                avx512bw: false,
                ..FULL
            },
        ];

        for features in missing_one {
            assert!(!kernel_is_supported_by_features(Kernel::Avx512, features));
            assert!(!kernel_is_supported_by_features(
                Kernel::Avx512Batch,
                features
            ));
            assert_eq!(best_kernel_for_x86_features(features), Kernel::Avx2);
            assert_eq!(
                best_batch_kernel_for_x86_features(features),
                Kernel::Avx2Batch
            );
        }
    }

    #[test]
    fn avx2_and_fma_are_required_by_every_vector_route() {
        for features in [
            X86FeatureSet {
                avx2: false,
                ..FULL
            },
            X86FeatureSet { fma: false, ..FULL },
        ] {
            assert!(!kernel_is_supported_by_features(Kernel::Avx2, features));
            assert!(!kernel_is_supported_by_features(
                Kernel::Avx2Batch,
                features
            ));
            assert!(!kernel_is_supported_by_features(Kernel::Avx512, features));
            assert_eq!(best_kernel_for_x86_features(features), Kernel::Scalar);
            assert_eq!(
                best_batch_kernel_for_x86_features(features),
                Kernel::ScalarBatch
            );
        }
    }

    #[test]
    fn avx2_only_and_scalar_masks_choose_the_safe_fallbacks() {
        let avx2_only = X86FeatureSet {
            avx2: true,
            fma: true,
            ..NONE
        };
        assert_eq!(best_kernel_for_x86_features(avx2_only), Kernel::Avx2);
        assert_eq!(
            best_batch_kernel_for_x86_features(avx2_only),
            Kernel::Avx2Batch
        );
        assert_eq!(best_kernel_for_x86_features(NONE), Kernel::Scalar);
        assert_eq!(
            best_batch_kernel_for_x86_features(NONE),
            Kernel::ScalarBatch
        );
    }
}
#[inline(always)]
pub fn alloc_with_nan_prefix(len: usize, warm: usize) -> Vec<f64> {
    // A `Vec<f64>` is a safe, initialized-value container. The former release
    // implementation set its length while only initializing `[0, warm)`, then
    // exposed the uninitialized tail as safe `f64` values. That is undefined
    // behaviour, and recursive indicators such as Damiani demonstrably read
    // those cells before overwriting them. Full NaN initialization is the only
    // sound contract for this return type; callers remain free to overwrite
    // the computable tail immediately.
    let _ = warm;
    vec![f64::NAN; len]
}

#[inline]
pub fn init_matrix_prefixes(buf: &mut [MaybeUninit<f64>], cols: usize, warm_prefixes: &[usize]) {
    assert!(
        cols != 0 && buf.len() % cols == 0,
        "`buf` length must be a multiple of `cols`"
    );
    let rows = buf.len() / cols;
    assert_eq!(
        rows,
        warm_prefixes.len(),
        "`warm_prefixes` length must equal number of rows"
    );

    #[cfg(debug_assertions)]
    {
        for cell in buf.iter_mut() {
            cell.write(f64::from_bits(0x22222222_22222222));
        }
    }

    buf.chunks_exact_mut(cols)
        .zip(warm_prefixes)
        .for_each(|(row, &warm)| {
            // NeoEthos patch 2026-05-26: original asserted
            //   `warm <= cols, "warm prefix exceeds row width"`
            // which abort-ed callers that wrote row-count-style warm
            // values (e.g. 14 for SMA-14) into a cols=1 row. The
            // sibling `alloc_with_nan_prefix` (line 106) already
            // clamps with `warm.min(len)`; doing the same here keeps
            // both helpers consistent and the NaN-prefix semantics
            // ("first `warm` CELLS of THIS row are NaN") intact.
            let warm = warm.min(cols);
            for cell in &mut row[..warm] {
                cell.write(f64::from_bits(0x7ff8_0000_0000_0000));
            }
        });
}

#[inline]
pub fn make_uninit_matrix(rows: usize, cols: usize) -> Vec<MaybeUninit<f64>> {
    let total = rows
        .checked_mul(cols)
        .expect("rows * cols overflowed usize");

    let mut v: Vec<MaybeUninit<f64>> = Vec::new();
    v.try_reserve_exact(total)
        .expect("OOM in make_uninit_matrix");

    #[cfg(not(debug_assertions))]
    {
        unsafe {
            v.set_len(total);
        }
    }

    #[cfg(debug_assertions)]
    {
        for _ in 0..total {
            v.push(MaybeUninit::new(f64::from_bits(0x33333333_33333333)));
        }
    }
    v
}

#[inline(always)]
pub fn alloc_uninit_f64(len: usize) -> Vec<f64> {
    #[cfg(not(debug_assertions))]
    {
        let mut v = Vec::<f64>::with_capacity(len);
        unsafe {
            v.set_len(len);
        }
        v
    }

    #[cfg(debug_assertions)]
    {
        vec![f64::from_bits(0x11111111_11111111); len]
    }
}
