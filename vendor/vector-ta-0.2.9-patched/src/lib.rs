#![cfg_attr(
    all(feature = "nightly-avx", rustc_is_nightly),
    feature(stdarch_x86_avx512)
)]
#![cfg_attr(
    all(feature = "nightly-avx", rustc_is_nightly),
    feature(avx512_target_feature)
)]
#![cfg_attr(all(feature = "nightly-avx", rustc_is_nightly), feature(portable_simd))]
#![cfg_attr(
    all(feature = "nightly-avx", rustc_is_nightly),
    feature(likely_unlikely)
)]
#![allow(warnings)]
#![allow(clippy::needless_range_loop)]

pub mod indicators;
pub mod utilities;

#[cfg(any(test, feature = "cuda-build-native"))]
pub(crate) mod native_sass;

#[cfg(test)]
mod rust_only_contract;

#[cfg(test)]
mod native_sass_contract;

#[cfg(test)]
#[path = "cuda/host_fallback_contract.rs"]
mod cuda_host_fallback_contract;

#[cfg(feature = "cuda-build-native")]
pub mod cuda;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod _rayon_test_pool {
    use ctor::ctor;
    use rayon::ThreadPoolBuilder;

    fn automatic_worker_limit(logical_threads: usize) -> usize {
        logical_threads.saturating_sub(2).max(1)
    }

    #[ctor]
    fn init_rayon_pool() {
        let logical_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let _ = ThreadPoolBuilder::new()
            .num_threads(automatic_worker_limit(logical_threads))
            .stack_size(8 * 1024 * 1024)
            .build_global();
    }

    #[test]
    fn automatic_worker_limit_reserves_one_smt_core() {
        assert_eq!(automatic_worker_limit(12), 10);
        assert_eq!(automatic_worker_limit(64), 62);
        assert_eq!(automatic_worker_limit(2), 1);
        assert_eq!(automatic_worker_limit(1), 1);
    }
}
