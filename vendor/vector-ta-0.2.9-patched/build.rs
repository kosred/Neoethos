use std::cell::RefCell;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "build_support/native_cuda_build.rs"]
mod native_cuda_build;

#[path = "src/native_sass.rs"]
mod native_sass;

use native_cuda_build::{
    ArtifactCompiler, ArtifactJob, ArtifactVerifier, KernelJob, NativeCompileOptions,
    NativePrecision, discover_native_architectures, expand_native_artifact_jobs,
    inspect_native_cubin, native_nvcc_args, order_kernel_jobs_longest_first,
    run_native_artifact_jobs, run_native_artifact_verifications, validate_unique_kernel_jobs,
};
use native_sass::{
    NativeArchPlan, NativeArtifact, native_cubin_filename, select_exact_native_cubin,
    validate_native_manifest,
};

fn main() {
    // Cargo grants this build script one implicit job slot. Connect before
    // opening any other descriptors (required by jobserver::Client::from_env
    // on Unix), then require one token for every additional NVCC worker.
    let cargo_jobserver = unsafe { jobserver::Client::from_env() };

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/cuda");

    // CPU-only builds do not compile or embed any CUDA artifact. CUDA builds
    // overwrite these sentinels after exact visible/explicit architectures
    // and the local nvcc version have been proven.
    println!("cargo:rustc-env=VECTOR_TA_CUDA_ARCHS=unknown");
    println!("cargo:rustc-env=VECTOR_TA_CUDA_ARCH_SOURCE=unknown");
    println!("cargo:rustc-env=VECTOR_TA_CUDA_NVCC_VERSION=unknown");

    if env::var("CARGO_FEATURE_CUDA_BUILD_NATIVE").is_ok() {
        compile_cuda_kernels(cargo_jobserver.as_ref());
    }

    if is_nightly() {
        println!("cargo:rustc-cfg=rustc_is_nightly");
    }
}

thread_local! {
    /// Filled synchronously by the existing 300+ compile declarations, then
    /// drained exactly once by `run_queued_kernel_jobs`.
    static KERNEL_JOBS: RefCell<Vec<KernelJob>> = const { RefCell::new(Vec::new()) };
}

fn configured_nvcc_width(job_count: usize) -> usize {
    let cargo_width = env::var("NUM_JOBS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|width| *width > 0)
        .unwrap_or(1);
    let host_width = std::thread::available_parallelism()
        .map(|width| width.get())
        .unwrap_or(1);
    cargo_width.min(host_width).min(job_count).max(1)
}

fn measured_kernel_tail_priority(rel_src: &str) -> u8 {
    match rel_src {
        // RTX 4090 / CUDA 12.8 evidence: despite being only 12 KiB, the fixed
        // 300-element local arrays make each artifact take ~16.5--16.9 s.
        // Scheduling it by source bytes started the artifact 20.34 s into a
        // 37.26 s NVCC span.  Keep this exact outlier ahead of byte heuristics.
        "kernels/cuda/market_meanness_index_kernel.cu" => 1,
        _ => 0,
    }
}

/// Drain the declared kernel jobs without creating a second parallelism
/// authority. The build-script thread owns Cargo's implicit slot; every extra
/// worker acquires/releases one token from Cargo's inherited jobserver around
/// exactly one NVCC process. Every `(source, exact architecture)` pair is an
/// independent cubin job, so a multi-GPU build uses the available compiler
/// parallelism without creating a second scheduler.
fn run_queued_kernel_jobs(cargo_jobserver: Option<&jobserver::Client>, cuda_path: &str) {
    let mut jobs = KERNEL_JOBS.with(|queue| std::mem::take(&mut *queue.borrow_mut()));
    if jobs.is_empty() {
        return;
    }
    validate_unique_kernel_jobs(&jobs)
        .unwrap_or_else(|error| panic!("vector-ta: invalid native CUDA kernel inventory: {error}"));
    order_kernel_jobs_longest_first(&mut jobs);

    let source_count = jobs.len();
    let nvcc = PathBuf::from(native_nvcc_path(cuda_path));
    let arch_plan = target_arch_plan(&nvcc);
    let stems = jobs
        .iter()
        .map(|job| job.cubin_stem.clone())
        .collect::<Vec<_>>();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let artifact_jobs = expand_native_artifact_jobs(&jobs, &arch_plan.architectures, &out_dir)
        .unwrap_or_else(|error| panic!("vector-ta: cannot expand native CUDA artifacts: {error}"));

    let width = configured_nvcc_width(artifact_jobs.len());
    eprintln!(
        "vector-ta build info: queued {} CUDA sources / {} independent NVCC artifacts; \
         configured shared width {}",
        source_count,
        artifact_jobs.len(),
        width
    );

    if cargo_jobserver.is_none() {
        println!(
            "cargo:warning=vector-ta could not inherit Cargo's jobserver; compiling NVCC jobs \
             serially rather than creating an unmanaged worker pool"
        );
    }
    let compiler = NativeNvccCompiler {
        nvcc,
        debug_line_info: env::var("CUDA_DEBUG").ok().as_deref() == Some("1"),
    };
    let report =
        match run_native_artifact_jobs(artifact_jobs.clone(), width, cargo_jobserver, &compiler) {
            Ok(report) => report,
            Err(failure) => {
                eprintln!("{}", failure.report.stable_json());
                panic!(
                    "vector-ta: native CUDA build failed closed: {}; no non-native, \
                 nearest-architecture, or CPU fallback is permitted",
                    failure.message
                );
            }
        };
    eprintln!("{}", report.stable_json());

    let verifier = NativeCubinVerifier {
        cuobjdump: PathBuf::from(native_cuobjdump_path(cuda_path)),
    };
    let verification =
        match run_native_artifact_verifications(artifact_jobs, width, cargo_jobserver, &verifier) {
            Ok(report) => report,
            Err(failure) => {
                eprintln!("{}", failure.report.stable_json());
                panic!(
                    "vector-ta: native cubin verification failed closed: {}; no unverified, PTX, \
                 nearest-architecture, or CPU fallback is permitted",
                    failure.message
                );
            }
        };
    eprintln!("{}", verification.stable_json());

    let stem_refs = stems.iter().map(String::as_str).collect::<Vec<_>>();
    validate_native_outputs_and_write_registry(&stem_refs, &arch_plan, &compiler.nvcc);
}

fn is_nightly() -> bool {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = Command::new(rustc).arg("--version").output();
    if let Ok(output) = output {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            return stdout.contains("nightly");
        }
    }
    false
}

/// Kernel sources whose entry points feed the NeoEthos f64 indicator lane.
///
/// NON-NEGOTIABLE: these are NEVER compiled with `--use_fast_math`, whatever
/// `CUDA_FAST_MATH` says. Prior generated-device-code inspection found 23
/// approximate reciprocal instructions across 17 f64 kernels, so fast math
/// measurably degrades results in this crate. Our lane opts out
/// positively (`-prec-div=true -prec-sqrt=true -fmad=false`) instead of
/// relying on an env var being spelled correctly at the call site.
///
/// `-fmad=false` forbids the COMPILER from contracting a separate multiply and
/// add. It does NOT disable an explicit `fma()` call — which is exactly what
/// the kernels use where the CPU reference uses `f64::mul_add`, so the two
/// stay bit-identical on the fused steps and unfused everywhere else.
const F64_LANE_SOURCES: &[&str] = &[
    "neoethos_f64_kernels.cu",
    "kernels/cuda/oscillators/adosc_kernel.cu",
    // ------------------------------------------------------ closer 5, round 2
    // Each of these now carries a lane-shaped f64 entry point (search the
    // file for "NEOETHOS f64 LANE"). Listing the file here opts its WHOLE
    // compilation out of `--use_fast_math`, which is the only way the opt-out
    // can be correct: the f32 and f64 entry points share one translation
    // unit, so a per-entry flag does not exist.
    "kernels/cuda/smoothed_gaussian_trend_filter_kernel.cu",
    "kernels/cuda/spearman_correlation_kernel.cu",
    "kernels/cuda/squeeze_index_kernel.cu",
    "kernels/cuda/standardized_psar_oscillator_kernel.cu",
    "kernels/cuda/statistical_trailing_stop_kernel.cu",
    "kernels/cuda/stochastic_adaptive_d_kernel.cu",
    "kernels/cuda/stochastic_connors_rsi_kernel.cu",
    "kernels/cuda/stochastic_distance_kernel.cu",
    "kernels/cuda/stochastic_money_flow_index_kernel.cu",
    "kernels/cuda/supertrend_oscillator_kernel.cu",
    "kernels/cuda/supertrend_recovery_kernel.cu",
    "kernels/cuda/trend_flow_trail_kernel.cu",
    "kernels/cuda/twiggs_money_flow_kernel.cu",
    "kernels/cuda/volatility_quality_index_kernel.cu",
    "kernels/cuda/vwap_deviation_oscillator_kernel.cu",
    "kernels/cuda/vwap_zscore_with_signals_kernel.cu",
    // ------------------------------------------------------------- shard S5
    // Each of these now carries an f64 section (search the file for
    // "f64 LANE  --  shard S5") whose entry points are what the f64 lane
    // launches. Listing the file here opts its WHOLE compilation out of
    // `--use_fast_math`, which is the only way the opt-out can be correct:
    // the f32 and f64 entry points share one translation unit, so a per-entry
    // flag does not exist. The measured reason this matters is in the doc
    // comment above: prior device-code inspection found 23 approximate f64
    // reciprocal instructions, i.e. fast math was degrading this lane.
    "kernels/cuda/atr_kernel.cu",
    "kernels/cuda/dm_kernel.cu",
    "kernels/cuda/eri_kernel.cu",
    "kernels/cuda/garman_klass_volatility_kernel.cu",
    "kernels/cuda/gopalakrishnan_range_index_kernel.cu",
    "kernels/cuda/kaufmanstop_kernel.cu",
    "kernels/cuda/keltner_kernel.cu",
    "kernels/cuda/marketefi_kernel.cu",
    "kernels/cuda/medium_ad_kernel.cu",
    "kernels/cuda/medprice_kernel.cu",
    "kernels/cuda/moving_averages/ehlers_pma_kernel.cu",
    "kernels/cuda/moving_averages/epma_kernel.cu",
    "kernels/cuda/moving_averages/mab_kernel.cu",
    "kernels/cuda/moving_averages/mwdx_kernel.cu",
    "kernels/cuda/moving_averages/sgf_kernel.cu",
    "kernels/cuda/moving_averages/sinwma_kernel.cu",
    "kernels/cuda/moving_averages/srwma_kernel.cu",
    "kernels/cuda/moving_averages/trima_kernel.cu",
    "kernels/cuda/moving_averages/vwmacd_kernel.cu",
    "kernels/cuda/moving_averages/wclprice_kernel.cu",
    "kernels/cuda/oscillators/cg_kernel.cu",
    "kernels/cuda/oscillators/coppock_kernel.cu",
    "kernels/cuda/oscillators/dpo_kernel.cu",
    "kernels/cuda/oscillators/fosc_kernel.cu",
    "kernels/cuda/oscillators/kst_kernel.cu",
    "kernels/cuda/oscillators/lrsi_kernel.cu",
    "kernels/cuda/oscillators/mom_kernel.cu",
    "kernels/cuda/oscillators/roc_kernel.cu",
    "kernels/cuda/oscillators/ultosc_kernel.cu",
    "kernels/cuda/pattern_recognition_kernel.cu",
    "kernels/cuda/pivot_kernel.cu",
    "kernels/cuda/psychological_line_kernel.cu",
    "kernels/cuda/supertrend_kernel.cu",
    "kernels/cuda/vosc_kernel.cu",
    // ------------------------------------------------------------- shard S2
    // Same contract as the S5 block above: the f64 entry point the lane
    // launches lives in the indicator's OWN file, beside the f32 entry points
    // the 180 f32 wrappers still call, so the whole translation unit opts out
    // of fast math. Search these files for "S2 f64 LANE".
    "kernels/cuda/moving_averages/sqwma_kernel.cu",
    // ------------------------------------------------------------- shard S3
    // Same contract as the blocks above: the f64 entry point the lane
    // launches lives in the indicator's OWN file, beside the f32 entry points
    // the f32 wrappers still call, so the WHOLE translation unit opts out of
    // fast math -- including the f32 kernels, which is the only way to be sure
    // no `--use_fast_math` reaches an f64 instruction in the same object.
    // Search these files for "S3 f64 LANE".
    "kernels/cuda/deviation_kernel.cu",
    "kernels/cuda/mean_ad_kernel.cu",
    "kernels/cuda/oscillators/ao_kernel.cu",
    "kernels/cuda/moving_averages/linearreg_slope_kernel.cu",
    "kernels/cuda/moving_averages/tsf_kernel.cu",
    "kernels/cuda/moving_averages/highpass_kernel.cu",
    // `is_f64_lane_source` matches with `ends_with`, so the path here must be
    // the one `compile_kernel` is actually handed. This entry used to read
    // "kernels/cuda/decycler_kernel.cu" -- a file at the kernels ROOT that no
    // `compile_kernel` call ever named. The compiled translation unit is the
    // moving_averages one (build.rs:920), and it did NOT end with the old
    // needle, so `neoethos_decycler_batch_f64` was built WITH `--use_fast_math`
    // while the entry claimed it was exempt. The root duplicate is deleted.
    "kernels/cuda/moving_averages/decycler_kernel.cu",
    "kernels/cuda/moving_averages/supersmoother_kernel.cu",
    "kernels/cuda/moving_averages/tilson_kernel.cu",
    "kernels/cuda/wad_kernel.cu",
    "kernels/cuda/sar_kernel.cu",
    "kernels/cuda/oscillators/dti_kernel.cu",
    "kernels/cuda/zscore_kernel.cu",
    "kernels/cuda/pfe_kernel.cu",
    "kernels/cuda/chande_kernel.cu",
    "kernels/cuda/di_kernel.cu",
    "kernels/cuda/oscillators/kdj_kernel.cu",
    "kernels/cuda/oscillators/aso_kernel.cu",
    "kernels/cuda/wto_kernel.cu",
    "kernels/cuda/range_filter_kernel.cu",
    "kernels/cuda/moving_averages/correlation_cycle_kernel.cu",
    "kernels/cuda/moving_averages/mama_kernel.cu",
    "kernels/cuda/moving_averages/volume_adjusted_ma_kernel.cu",
    "kernels/cuda/oscillators/reverse_rsi_kernel.cu",
    "kernels/cuda/moving_averages/ehlers_ecema_kernel.cu",
    "kernels/cuda/pvi_kernel.cu",
    "kernels/cuda/vpt_kernel.cu",
    // ------------------------------------------------------------- shard S6
    // Same contract as the S5 and S2 blocks above. Each of these files now
    // carries an f64 section -- search it for "f64 LANE  --  shard S6" -- and
    // listing it opts its WHOLE translation unit out of `--use_fast_math`,
    // because the f32 and f64 entry points share one compilation and a
    // per-entry flag does not exist.
    //
    // This DOES change the f32 entry points in these files: they stop being
    // compiled with fast math. Deliberate, and an accuracy improvement rather
    // than a regression -- the measured reason is in the doc comment above,
    // where prior device-code inspection found 23 approximate f64 reciprocal
    // instructions, i.e. the flag was degrading f64 in this crate and has
    // been degrading the f32 divides and square roots all along.
    // ------------------------------------------------------- closer 6 (C6)
    // Same contract as every block above: the f64 entry point the lane
    // launches lives in the indicator's OWN file, beside the f32 entry points
    // the f32 wrappers still call, so listing it opts the WHOLE translation
    // unit out of `--use_fast_math`. Search these files for
    // "f64 LANE  --  closer 6".
    //
    // `emd_kernel.cu`, `keltner_kernel.cu`, `lpc_kernel.cu`,
    // `fvg_trailing_stop_kernel.cu`, `macz_kernel.cu`, `msw_kernel.cu`,
    // `rvi_kernel.cu` and `yang_zhang_volatility_kernel.cu` are ALREADY in
    // this list from the S5/S6 blocks below -- those shards listed the file
    // without landing its f64 section -- so only the files this closer adds
    // for the first time appear here.
    "kernels/cuda/oscillators/stoch_kernel.cu",
    "kernels/cuda/nadaraya_watson_envelope_kernel.cu",
    "kernels/cuda/alphatrend_kernel.cu",
    "kernels/cuda/bollinger_bands_width_kernel.cu",
    "kernels/cuda/correl_hl_kernel.cu",
    "kernels/cuda/cvi_kernel.cu",
    "kernels/cuda/donchian_kernel.cu",
    "kernels/cuda/emd_kernel.cu",
    "kernels/cuda/fvg_trailing_stop_kernel.cu",
    "kernels/cuda/historical_volatility_kernel.cu",
    "kernels/cuda/lpc_kernel.cu",
    "kernels/cuda/moving_averages/buff_averages_kernel.cu",
    "kernels/cuda/moving_averages/cora_wave_kernel.cu",
    "kernels/cuda/moving_averages/fwma_kernel.cu",
    "kernels/cuda/moving_averages/hwma_kernel.cu",
    "kernels/cuda/moving_averages/jsa_kernel.cu",
    "kernels/cuda/moving_averages/macz_kernel.cu",
    "kernels/cuda/moving_averages/nma_kernel.cu",
    "kernels/cuda/moving_averages/qstick_kernel.cu",
    "kernels/cuda/moving_averages/swma_kernel.cu",
    "kernels/cuda/moving_averages/trendflex_kernel.cu",
    "kernels/cuda/moving_averages/vpwma_kernel.cu",
    "kernels/cuda/moving_averages/vwap_kernel.cu",
    "kernels/cuda/moving_averages/vwma_kernel.cu",
    "kernels/cuda/oscillators/aroonosc_kernel.cu",
    "kernels/cuda/oscillators/bop_kernel.cu",
    "kernels/cuda/oscillators/cfo_kernel.cu",
    "kernels/cuda/oscillators/dec_osc_kernel.cu",
    "kernels/cuda/oscillators/msw_kernel.cu",
    "kernels/cuda/oscillators/rocp_kernel.cu",
    "kernels/cuda/oscillators/rvi_kernel.cu",
    "kernels/cuda/oscillators/willr_kernel.cu",
    "kernels/cuda/parkinson_volatility_kernel.cu",
    "kernels/cuda/percentile_nearest_rank_kernel.cu",
    "kernels/cuda/ttm_trend_kernel.cu",
    "kernels/cuda/var_kernel.cu",
    "kernels/cuda/vertical_horizontal_filter_kernel.cu",
    "kernels/cuda/vi_kernel.cu",
    "kernels/cuda/voss_kernel.cu",
    "kernels/cuda/yang_zhang_volatility_kernel.cu",
    // --------------------------------------------------- closer 6, round 2
    // Five more files that gained a "NEOETHOS f64 LANE  --  closer 6" section
    // in this round and were NOT already opted out by an earlier shard. Each
    // one must be here or nvcc compiles its WHOLE translation unit --
    // including the new f64 entry point -- with `--use_fast_math`, which is
    // how 23 approximate f64 reciprocal instructions reached generated device
    // code in the first place.
    "kernels/cuda/oscillators/qqe_kernel.cu",
    "kernels/cuda/oscillators/srsi_kernel.cu",
    "kernels/cuda/oscillators/stc_kernel.cu",
    "kernels/cuda/net_myrsi_kernel.cu",
    "kernels/cuda/moving_averages/vlma_kernel.cu",
    // ------------------------------------------------------------- shard S1
    // Same contract as the blocks above. Each of these files now carries an
    // f64 section -- search it for "S1 f64 LANE" -- and listing it opts its
    // WHOLE translation unit out of `--use_fast_math`, because the f32 and f64
    // entry points share one compilation and nvcc has no per-entry flag.
    "kernels/cuda/kurtosis_kernel.cu",
    "kernels/cuda/nvi_kernel.cu",
    "kernels/cuda/safezonestop_kernel.cu",
    "kernels/cuda/moving_averages/alligator_kernel.cu",
    "kernels/cuda/moving_averages/alma_kernel.cu",
    "kernels/cuda/moving_averages/apo_kernel.cu",
    "kernels/cuda/moving_averages/edcf_kernel.cu",
    "kernels/cuda/moving_averages/hma_kernel.cu",
    "kernels/cuda/moving_averages/kama_kernel.cu",
    "kernels/cuda/moving_averages/linreg_kernel.cu",
    "kernels/cuda/moving_averages/pma_kernel.cu",
    "kernels/cuda/moving_averages/vidya_kernel.cu",
    "kernels/cuda/oscillators/chop_kernel.cu",
    "kernels/cuda/oscillators/emv_kernel.cu",
    "kernels/cuda/oscillators/fisher_kernel.cu",
    "kernels/cuda/oscillators/gatorosc_kernel.cu",
    "kernels/cuda/oscillators/kvo_kernel.cu",
    "kernels/cuda/oscillators/ppo_kernel.cu",
    "kernels/cuda/oscillators/stochf_kernel.cu",
    // ------------------------------------------------------------- shard S4
    // Same contract as the blocks above. Each of these carries an
    // `<id>_neo_batch_f64` entry point (search the file for "S4 f64 LANE" or
    // "NEOETHOS f64 LANE") beside the f32 entry points the f32 wrappers still
    // call, so opting the WHOLE translation unit out of `--use_fast_math` is
    // the only correct granularity — a per-entry flag does not exist.
    //
    // One of these is listed for a reason that is NOT its f64 section:
    // `damiani_volatmeter_kernel.cu:68` hardcodes
    // `const float EPS = 1.1920929e-7f`, f32 machine epsilon, which fast math
    // is entitled to fold away entirely.
    "kernels/cuda/ad_kernel.cu",
    "kernels/cuda/aroon_kernel.cu",
    "kernels/cuda/bollinger_bands_kernel.cu",
    "kernels/cuda/cksp_kernel.cu",
    "kernels/cuda/damiani_volatmeter_kernel.cu",
    "kernels/cuda/dx_kernel.cu",
    "kernels/cuda/er_kernel.cu",
    "kernels/cuda/linearreg_angle_kernel.cu",
    "kernels/cuda/mass_kernel.cu",
    "kernels/cuda/moving_averages/cwma_kernel.cu",
    "kernels/cuda/moving_averages/ehma_kernel.cu",
    "kernels/cuda/moving_averages/frama_kernel.cu",
    "kernels/cuda/moving_averages/highpass2_kernel.cu",
    "kernels/cuda/moving_averages/linearreg_intercept_kernel.cu",
    "kernels/cuda/moving_averages/smma_kernel.cu",
    "kernels/cuda/moving_averages/supersmoother_3_pole_kernel.cu",
    "kernels/cuda/natr_kernel.cu",
    "kernels/cuda/obv_kernel.cu",
    "kernels/cuda/oscillators/acosc_kernel.cu",
    "kernels/cuda/oscillators/cmo_kernel.cu",
    "kernels/cuda/oscillators/ift_rsi_kernel.cu",
    "kernels/cuda/oscillators/macd_kernel.cu",
    "kernels/cuda/oscillators/rsi_kernel.cu",
    "kernels/cuda/oscillators/ttm_squeeze_kernel.cu",
    "kernels/cuda/stddev_kernel.cu",
    "kernels/cuda/ui_kernel.cu",
    "kernels/cuda/vpci_kernel.cu",
    "kernels/cuda/wavetrend_kernel.cu",
    "kernels/cuda/dvdiqqe_kernel.cu",
    "kernels/cuda/oscillators/cci_cycle_kernel.cu",
    // ---------------------------------------------- from-scratch f64 kernels
    //
    // The nine files that shipped as one-line EMPTY stubs
    // (`extern "C" __global__ void possible_rsi_batch_f64() {}`) and now carry
    // real kernels written against the CPU reference, plus
    // `rogers_satchell_volatility_kernel.cu`, which had no `.cu` at all.
    //
    // Unlike the shard blocks above, these are f64-ONLY translation units —
    // there is no f32 entry point in them to be affected. Listing them is
    // still the only way to opt out of `--use_fast_math`, and it matters:
    // several of them run `log`, `exp`, `atan`, `acosh`, `asinh` and `sqrt`
    // inside a RECURRENCE, where an approximate reciprocal does not perturb one
    // bar, it walks forward through every later bar of the series.
    "kernels/cuda/goertzel_cycle_composite_wave_kernel.cu",
    "kernels/cuda/ichimoku_oscillator_kernel.cu",
    "kernels/cuda/ict_propulsion_block_kernel.cu",
    "kernels/cuda/insync_index_kernel.cu",
    "kernels/cuda/kase_peak_oscillator_with_divergences_kernel.cu",
    "kernels/cuda/market_structure_confluence_kernel.cu",
    "kernels/cuda/possible_rsi_kernel.cu",
    "kernels/cuda/rogers_satchell_volatility_kernel.cu",
    "kernels/cuda/smooth_theil_sen_kernel.cu",
    "kernels/cuda/vdubus_divergence_wave_pattern_generator_kernel.cu",
    // ------------------------------------------------------------- closer 3
    // Same contract as the blocks above: the f64 entry point the lane launches
    // lives in the indicator's OWN file, beside the f32 entry points the f32
    // wrappers still call, so the WHOLE translation unit opts out of fast
    // math -- a per-entry flag does not exist. Search these files for
    // "f64 LANE  --  closer C3". `marketefi_kernel.cu` and `medium_ad_kernel.cu`
    // are already listed above by shard S5 and are not repeated.
    "kernels/cuda/l1_ehlers_phasor_kernel.cu",
    "kernels/cuda/l2_ehlers_signal_to_noise_kernel.cu",
    "kernels/cuda/kairi_relative_index_kernel.cu",
    "kernels/cuda/linear_correlation_oscillator_kernel.cu",
    // ------------------------------------------------------------- closer 4
    // Same contract as the blocks above: the f64 entry point the lane launches
    // lives in the indicator's OWN file, beside the f32 entry points the f32
    // wrappers still call, so the WHOLE translation unit opts out of fast
    // math -- nvcc has no per-entry flag. Search these files for
    // "f64 LANE  --  closer 4".
    //
    // `psychological_line_kernel.cu`, `moving_averages/sinwma_kernel.cu`,
    // `moving_averages/srwma_kernel.cu` and `moving_averages/qstick_kernel.cu`
    // are already listed by shard S5/S6 above and are NOT repeated. The three
    // below were not listed by any earlier block, so without them the closer-4
    // kernels in them would be built with `--use_fast_math` whenever the env
    // var is set -- and `random_walk_index` runs `sqrt` and a reciprocal
    // INSIDE a Wilder recurrence, where an approximate reciprocal does not
    // perturb one bar, it walks forward through every later bar of the series.
    "kernels/cuda/rank_correlation_index_kernel.cu",
    "kernels/cuda/random_walk_index_kernel.cu",
    "kernels/cuda/rolling_z_score_trend_kernel.cu",
    // ------------------------------------------------------------ closer 5
    // Same contract as the blocks above: each of these carries an
    // <id>_neo_batch_f64 entry point (search the file for NEOETHOS f64 LANE)
    // beside the f32 and crate-shaped f64 entry points its own wrappers still
    // call, so opting the WHOLE translation unit out of --use_fast_math is the
    // only correct granularity -- nvcc has no per-entry flag.
    //
    // trima, vosc and ultosc are NOT repeated here: they are already listed in
    // the shard S5 block above, and listing a file twice would be a second
    // claim about the same translation unit rather than a stronger one.
    "kernels/cuda/trend_continuation_factor_kernel.cu",
    "kernels/cuda/trend_direction_force_index_kernel.cu",
    "kernels/cuda/trend_trigger_factor_kernel.cu",
    "kernels/cuda/velocity_kernel.cu",
    "kernels/cuda/velocity_acceleration_convergence_divergence_indicator_kernel.cu",
    "kernels/cuda/velocity_acceleration_indicator_kernel.cu",
    "kernels/cuda/volume_weighted_rsi_kernel.cu",
    "kernels/cuda/volume_zone_oscillator_kernel.cu",
    "kernels/cuda/momentum_ratio_oscillator_kernel.cu",
    "kernels/cuda/on_balance_volume_oscillator_kernel.cu",
    // ------------------------------------------------------------- closer 1
    // Each of these now carries an `<id>_neo_batch_f64` entry point (search
    // the file for "NEOETHOS f64 LANE") beside the entry points the
    // per-indicator wrappers still call. Listing the file here opts its WHOLE
    // compilation out of `--use_fast_math`, which is the only correct
    // granularity: the f32 and f64 entry points share one translation unit,
    // so a per-entry flag does not exist.
    //
    // `oscillators/cg_kernel.cu` is listed for a second reason as well: its
    // f32 lane hardcodes `1.1920929e-7f` at :22, f32 MACHINE EPSILON, which
    // fast math is entitled to fold away entirely. The f64 entry point uses
    // `f64::EPSILON` instead, as the CPU does (cg.rs:339).
    "kernels/cuda/absolute_strength_index_oscillator_kernel.cu",
    "kernels/cuda/accumulation_swing_index_kernel.cu",
    "kernels/cuda/adaptive_bandpass_trigger_oscillator_kernel.cu",
    "kernels/cuda/adaptive_bounds_rsi_kernel.cu",
    "kernels/cuda/adaptive_macd_kernel.cu",
    "kernels/cuda/adaptive_momentum_oscillator_kernel.cu",
    "kernels/cuda/advance_decline_line_kernel.cu",
    "kernels/cuda/andean_oscillator_kernel.cu",
    "kernels/cuda/atr_percentile_kernel.cu",
    "kernels/cuda/bull_power_vs_bear_power_kernel.cu",
    "kernels/cuda/daily_factor_kernel.cu",
    "kernels/cuda/decisionpoint_breadth_swenlin_trading_oscillator_kernel.cu",
    "kernels/cuda/didi_index_kernel.cu",
    "kernels/cuda/disparity_index_kernel.cu",
    "kernels/cuda/donchian_channel_width_kernel.cu",
    // ------------------------------------------------------- closer 2, round 2
    //
    // TWENTY-TWO FILES THAT WERE ALREADY REGISTERED IN `F64_KERNELS` AND WERE
    // STILL BEING BUILT WITH `--use_fast_math`.
    //
    // This is not a new claim about a new kernel. Every file below already
    // carries the f64 entry point its `F64Kernel` variant launches -- verified
    // by `F64Kernel::module_stem` naming this file's stem and
    // `F64Kernel::entry_point` naming a symbol that `grep -n __global__` finds
    // in it. What was missing is the ONE thing that makes such a registration
    // honest: the translation unit was never opted out of fast math, so the
    // lane was launching an f64 kernel whose divides and square roots nvcc was
    // free to lower to `rcp.approx.ftz.f64`. That is precisely the
    // contamination the doc comment at the top of this list measures (23
    // approximate f64 reciprocals across 17 files) -- these files were part of
    // the reason the number was not zero.
    //
    // Several of them run that divide INSIDE A RECURRENCE, where an
    // approximate reciprocal does not perturb one bar, it walks forward
    // through every later bar of the series: `rsx` (six cascaded EMAs),
    // `trix` (three cascaded EMAs), `ehlers_kama` / `ehlers_itrend` /
    // `ehlers_smoothed_adaptive_momentum` (IIR filters), `ewma_volatility`
    // (an EWMA of squared log returns), `emd_trend` and `impulse_macd`.
    //
    // As everywhere else in this list, the granularity is the WHOLE file
    // because nvcc has no per-entry flag and the f32 entry points these files
    // still carry share the translation unit.
    "kernels/cuda/moving_averages/pwma_kernel.cu",
    "kernels/cuda/moving_averages/nama_kernel.cu",
    "kernels/cuda/moving_averages/sama_kernel.cu",
    // --------------------------------------------------------- closer 1
    // Five more that were converted and REGISTERED but never listed, so
    // `is_f64_lane_source` answered false and nvcc built them WITH
    // `--use_fast_math`. Each carries exactly one `__global__ ..._f64`
    // (`neoethos_{gaussian,reflex,jma,maaq,tradjema}_batch_f64`), and each of
    // those five bodies is already free of f32 literals, f32-suffixed
    // functions and fast-math intrinsics -- the ONLY thing still degrading
    // them was this omission. All five are `is_sequential` IIR/adaptive
    // recurrences, which is the worst place to accept an approximate
    // reciprocal: the error is carried into every subsequent bar.
    "kernels/cuda/moving_averages/gaussian_kernel.cu",
    "kernels/cuda/moving_averages/reflex_kernel.cu",
    "kernels/cuda/moving_averages/jma_kernel.cu",
    "kernels/cuda/moving_averages/maaq_kernel.cu",
    "kernels/cuda/moving_averages/tradjema_kernel.cu",
    "kernels/cuda/moving_averages/ehlers_kama_kernel.cu",
    "kernels/cuda/moving_averages/ehlers_itrend_kernel.cu",
    "kernels/cuda/moving_averages/trix_kernel.cu",
    "kernels/cuda/oscillators/rsx_kernel.cu",
    "kernels/cuda/minmax_kernel.cu",
    "kernels/cuda/chandelier_exit_kernel.cu",
    "kernels/cuda/devstop_kernel.cu",
    "kernels/cuda/ehlers_detrending_filter_kernel.cu",
    "kernels/cuda/ehlers_simple_cycle_indicator_kernel.cu",
    "kernels/cuda/ehlers_smoothed_adaptive_momentum_kernel.cu",
    "kernels/cuda/ewma_volatility_kernel.cu",
    "kernels/cuda/fractal_dimension_index_kernel.cu",
    "kernels/cuda/impulse_macd_kernel.cu",
    "kernels/cuda/hypertrend_kernel.cu",
    "kernels/cuda/emd_trend_kernel.cu",
    "kernels/cuda/ehlers_fm_demodulator_kernel.cu",
    "kernels/cuda/forward_backward_exponential_oscillator_kernel.cu",
    "kernels/cuda/gmma_oscillator_kernel.cu",
    "kernels/cuda/evasive_supertrend_kernel.cu",
    //
    // FIVE MORE that now carry a `<id>_neo_batch_f64` entry point written by
    // this closer (search each file for "f64 LANE  --  closer 2, round 2").
    // `kaufmanstop_kernel.cu`, `oscillators/lrsi_kernel.cu`,
    // `moving_averages/mwdx_kernel.cu`, `pivot_kernel.cu` and
    // `moving_averages/sgf_kernel.cu` also received one but are already listed
    // by the shard S5 block above and are NOT repeated -- listing a file twice
    // would be a second claim about the same translation unit rather than a
    // stronger one.
    "kernels/cuda/dual_ulcer_index_kernel.cu",
    "kernels/cuda/hull_butterfly_oscillator_kernel.cu",
    "kernels/cuda/market_structure_trailing_stop_kernel.cu",
    "kernels/cuda/polynomial_regression_extrapolation_kernel.cu",
    "kernels/cuda/range_oscillator_kernel.cu",
    // ---------------------------------------------- closer 4, round 2
    // The f64 entry point this lane launches lives in the indicator's
    // OWN file, beside the entry points that file already had, so the
    // WHOLE translation unit has to opt out of fast math -- nvcc has no
    // per-entry flag. Search these files for
    // "NEOETHOS f64 LANE  --  closer 4".
    "kernels/cuda/keltner_channel_width_oscillator_kernel.cu",
    "kernels/cuda/leavitt_convolution_acceleration_kernel.cu",
    "kernels/cuda/market_meanness_index_kernel.cu",
    "kernels/cuda/monotonicity_index_kernel.cu",
    "kernels/cuda/premier_rsi_oscillator_kernel.cu",
    "kernels/cuda/pretty_good_oscillator_kernel.cu",
    "kernels/cuda/price_density_market_noise_kernel.cu",
    "kernels/cuda/projection_oscillator_kernel.cu",
    "kernels/cuda/qqe_weighted_oscillator_kernel.cu",
    "kernels/cuda/rolling_skewness_kurtosis_kernel.cu",
    // ------------------------------------------------ closer 3, round 2
    // Each of these now carries an f64 lane section (search the file for
    // "NEOETHOS f64 LANE  --  closer 3") whose `*_neo_batch_f64` entry point
    // is what the f64 lane launches. Listing the FILE here opts its WHOLE
    // compilation out of `--use_fast_math`, which is the only way the opt-out
    // can be correct: the existing entry points and the new f64 one share one
    // translation unit, so a per-entry flag does not exist. Without this the
    // generated code gains approximate f64 reciprocals -- the measured defect
    // this list exists to remove.
    "kernels/cuda/adjustable_ma_alternating_extremities_kernel.cu",
    "kernels/cuda/autocorrelation_indicator_kernel.cu",
    "kernels/cuda/historical_volatility_rank_kernel.cu",
    "kernels/cuda/historical_volatility_percentile_kernel.cu",
    "kernels/cuda/directional_imbalance_index_kernel.cu",
    "kernels/cuda/cycle_channel_oscillator_kernel.cu",
    "kernels/cuda/dynamic_momentum_index_kernel.cu",
    "kernels/cuda/ehlers_adaptive_cg_kernel.cu",
    "kernels/cuda/ehlers_adaptive_cyber_cycle_kernel.cu",
    "kernels/cuda/ehlers_data_sampling_relative_strength_indicator_kernel.cu",
    "kernels/cuda/exponential_trend_kernel.cu",
    "kernels/cuda/geometric_bias_oscillator_kernel.cu",
    "kernels/cuda/intraday_momentum_index_kernel.cu",
    "kernels/cuda/bulls_v_bears_kernel.cu",
    "kernels/cuda/candle_strength_oscillator_kernel.cu",
    "kernels/cuda/cyberpunk_value_trend_analyzer_kernel.cu",
    "kernels/cuda/fvg_positioning_average_kernel.cu",
    "kernels/cuda/hema_trend_levels_kernel.cu",
    "kernels/cuda/fibonacci_trailing_stop_kernel.cu",
    "kernels/cuda/grover_llorens_cycle_oscillator_kernel.cu",
    "kernels/cuda/demand_index_kernel.cu",
    "kernels/cuda/adaptive_schaff_trend_cycle_kernel.cu",
    "kernels/cuda/ehlers_linear_extrapolation_predictor_kernel.cu",
    "kernels/cuda/ehlers_autocorrelation_periodogram_kernel.cu",
    // ------------------------------------------------------ closer 6, round 3
    //
    // Six indicators that had NO CUDA presence at all -- no `.cu`, no wrapper,
    // no `F64_KERNELS` row -- and now carry a from-scratch f64 kernel written
    // against the CPU reference. These are f64-ONLY translation units: there is
    // no f32 entry point in any of them to be affected. Listing them is still
    // the only way to opt out of `--use_fast_math`, and it matters in every one:
    //
    //  * `n_order_ema` and `ema_deviation_corrected_t3` divide inside a
    //    recurrence, where an approximate reciprocal walks forward through
    //    every later bar rather than perturbing one;
    //  * `wave_smoother` runs `sin`/`cos` to build its weight vector;
    //  * `logarithmic_moving_average` runs `log` per weight slot and divides
    //    the window sum by the weight total on every emitted bar;
    //  * `volatility_adjusted_ma` carries three chained recurrences and calls
    //    `fma` explicitly at :411 of its CPU reference;
    //  * `elastic_volume_weighted_moving_average` divides by the rolling volume
    //    sum at every bar and feeds the quotient back in as the next `base`.
    //
    // `moving_averages/vama_kernel.cu` is NOT listed here and is NOT touched:
    // it holds the f32 entry points `vama_wrapper.rs` still loads, and it
    // belongs to shard S1.
    "kernels/cuda/moving_averages/elastic_volume_weighted_moving_average_kernel.cu",
    "kernels/cuda/moving_averages/ema_deviation_corrected_t3_kernel.cu",
    "kernels/cuda/moving_averages/logarithmic_moving_average_kernel.cu",
    "kernels/cuda/moving_averages/n_order_ema_kernel.cu",
    "kernels/cuda/moving_averages/volatility_adjusted_ma_kernel.cu",
    "kernels/cuda/moving_averages/wave_smoother_kernel.cu",
    // ------------------------------------------------------ closer 2, round 3
    //
    // Each of these now carries an `<id>_neo_batch_f64` entry point (search the
    // file for "NEOETHOS f64 LANE  --  closer 2, round 3") beside the
    // bespoke-shaped f64 entry point its own wrapper already calls. Listing the
    // FILE here opts its WHOLE compilation out of `--use_fast_math`, which is
    // the only correct granularity: nvcc has no per-entry flag.
    //
    // `possible_rsi_kernel.cu` also received one but is ALREADY listed in the
    // from-scratch block above and is NOT repeated -- listing a file twice
    // would be a second claim about the same translation unit rather than a
    // stronger one.
    //
    // Two of these matter beyond the general rule. `normalized_resonator`
    // derives its resonator gain from `tan()` and multiplies it into every
    // later bar of a 2-pole recurrence, and `regression_slope_oscillator`
    // forms every slope as the DIFFERENCE of two running sums of `log()` that
    // reach ~1e10 after 800k bars -- an approximate `log` or reciprocal does
    // not perturb one bar in either, it walks forward through the whole series.
    "kernels/cuda/neighboring_trailing_stop_kernel.cu",
    "kernels/cuda/nonlinear_regression_zero_lag_moving_average_kernel.cu",
    "kernels/cuda/normalized_resonator_kernel.cu",
    "kernels/cuda/normalized_volume_true_range_kernel.cu",
    "kernels/cuda/price_moving_average_ratio_percentile_kernel.cu",
    "kernels/cuda/range_breakout_signals_kernel.cu",
    "kernels/cuda/range_filtered_trend_signals_kernel.cu",
    "kernels/cuda/regression_slope_oscillator_kernel.cu",
    "kernels/cuda/relative_strength_index_wave_indicator_kernel.cu",
    // ------------------------------------------------ closer 3, round 3
    // Same contract as every block above: the f64 entry point the lane
    // launches lives in the indicator's OWN file, beside the f32 entry points
    // the f32 wrappers still call, so listing the file opts the WHOLE
    // translation unit out of `--use_fast_math` -- a per-entry flag does not
    // exist. Search these files for "f64 LANE  --  closer 3, round 3".
    //
    // `alphatrend_kernel.cu` and
    // `vdubus_divergence_wave_pattern_generator_kernel.cu` are ALREADY in this
    // list (from the closer-6 and stub blocks above) and are not repeated.
    //
    // TWO OF THESE FILES ARE STILL PURE f32 IN THEIR EXISTING ENTRY POINTS and
    // that is deliberate rather than unfinished: every one of the seven
    // `alphatrend_*_f32` symbols and both `avsl_*_f32` symbols is still called
    // by a live wrapper (`alphatrend_wrapper.rs:514, 559, 706, 800, 846, 1235,
    // 1340, 1465`; `avsl_wrapper.rs:264, 581, 759, 840`), so converting them in
    // place would break those callers. The rule is "add the f64 entry point
    // beside it and route our lane to the f64 one", which is what the new
    // `*_neo_batch_f64` entries do. Listing the files here DOES change the f32
    // entry points -- they stop being compiled with fast math -- and that is an
    // accuracy improvement, not a regression: `alphatrend_kernel.cu:50-52` was
    // reaching for `fabsf`/`fmaxf` under `--use_fast_math`, and
    // `avsl_kernel.cu:47` was building its NaN as an f32 bit pattern.
    "kernels/cuda/reversal_signals_kernel.cu",
    "kernels/cuda/trend_follower_kernel.cu",
    "kernels/cuda/volatility_ratio_adaptive_rsx_kernel.cu",
    "kernels/cuda/volume_energy_reservoirs_kernel.cu",
    "kernels/cuda/volume_weighted_relative_strength_index_kernel.cu",
    "kernels/cuda/volume_weighted_stochastic_rsi_kernel.cu",
    "kernels/cuda/zig_zag_channels_kernel.cu",
    "kernels/cuda/moving_averages/avsl_kernel.cu",
    // ---------------------------------------------------- closer 5, round 3
    //
    // Three files that already existed and now carry an `<id>_neo_batch_f64`
    // entry point (search each for "NEOETHOS f64 LANE"), plus two written from
    // scratch. Listing a file here opts its WHOLE compilation out of
    // `--use_fast_math`, which is the only way the opt-out can be correct: in
    // the first three the f32 and f64 entry points share one translation unit,
    // so a per-entry flag does not exist.
    //
    // `moving_averages/mab_kernel.cu`, `moving_averages/vwmacd_kernel.cu`,
    // `lpc_kernel.cu`, `moving_averages/macz_kernel.cu` and
    // `pattern_recognition_kernel.cu` are ALREADY listed above and are
    // deliberately not repeated.
    "kernels/cuda/moving_averages/rsmk_kernel.cu",
    "kernels/cuda/oscillators/squeeze_momentum_kernel.cu",
    "kernels/cuda/moving_averages/uma_kernel.cu",
    "kernels/cuda/moving_averages/corrected_moving_average_kernel.cu",
    "kernels/cuda/moving_averages/ehlers_undersampled_double_moving_average_kernel.cu",
    // ------------------------------------------------ closer 1, round 3
    // Each of these now carries a lane-shaped `*_neo_batch_f64` entry point
    // (search the file for "NEOETHOS f64 LANE  --  closer 1, round 3").
    // Listing the FILE here opts its WHOLE compilation out of
    // `--use_fast_math`, which is the only way the opt-out can be correct:
    // the existing multi-output entry point and the new lane one share one
    // translation unit, so a per-entry flag does not exist.
    // `goertzel_cycle_composite_wave_kernel.cu`, `ichimoku_oscillator_kernel.cu`
    // and `insync_index_kernel.cu` are already listed above and are not
    // repeated.
    "kernels/cuda/fibonacci_entry_bands_kernel.cu",
    "kernels/cuda/half_causal_estimator_kernel.cu",
    "kernels/cuda/linear_regression_intensity_kernel.cu",
    "kernels/cuda/macd_wave_signal_pro_kernel.cu",
    "kernels/cuda/mesa_stochastic_multi_length_kernel.cu",
    "kernels/cuda/moving_average_cross_probability_kernel.cu",
    "kernels/cuda/multi_length_stochastic_average_kernel.cu",
    // ------------------------------------------------ THE ONE BUILD, round 3
    // These seven landed in the final gap-filling round: each carries a
    // lane-shaped `*_neo_batch_f64` entry point AND a `F64_KERNELS` row, so
    // our lane routes to them — but nobody added the FILE here, which meant
    // all seven were about to be compiled WITH `--use_fast_math`.
    //
    // This is not a cosmetic omission. `bandpass` and `ott`/`otto` are IIR
    // recurrences: under fast math the divide in the recursion becomes
    // approximate f64 reciprocal instructions, and the error compounds bar
    // over bar into a value that feeds a threshold
    // comparison. Registered-but-not-opted-out is the worst of the three
    // states, because the lane USES the kernel and trusts it.
    //
    // Found by comparing, file by file, the set that defines a lane entry
    // point against this list — not by reading the round-3 reports, which
    // said these were "written and registered" and were correct as far as
    // they went.
    "kernels/cuda/bandpass_kernel.cu",
    "kernels/cuda/halftrend_kernel.cu",
    "kernels/cuda/mod_god_mode_kernel.cu",
    "kernels/cuda/moving_averages/dma_kernel.cu",
    "kernels/cuda/moving_averages/ott_kernel.cu",
    "kernels/cuda/moving_averages/otto_kernel.cu",
    "kernels/cuda/prb_kernel.cu",
];

fn is_f64_lane_source(rel_src: &str) -> bool {
    F64_LANE_SOURCES
        .iter()
        .any(|needle| rel_src.ends_with(needle))
}

fn target_arch_plan(nvcc: &Path) -> NativeArchPlan {
    let explicit_archs = env::var("CUDA_ARCHS").ok();
    let nvidia_smi =
        PathBuf::from(env::var_os("NVIDIA_SMI").unwrap_or_else(|| "nvidia-smi".into()));
    let plan = discover_native_architectures(nvcc, &nvidia_smi, explicit_archs.as_deref())
        .unwrap_or_else(|error| {
            panic!(
                "vector-ta: cannot build an exact native-SASS registry: {error}. No non-native, \
                 nearest-architecture, or CPU substitution path is permitted."
            )
        });

    let joined: Vec<String> = plan
        .architectures
        .iter()
        .map(|arch| format!("sm_{arch}"))
        .collect();
    println!("cargo:rustc-env=VECTOR_TA_CUDA_ARCHS={}", joined.join(","));
    println!(
        "cargo:rustc-env=VECTOR_TA_CUDA_ARCH_SOURCE={}",
        plan.source.as_str()
    );
    eprintln!(
        "vector-ta build info: exact native cubins for {:?} (source={})",
        plan.architectures,
        plan.source.as_str()
    );
    plan
}

/// Is `--use_fast_math` wanted for this source? OFF unless asked for by name.
///
/// ── NeoEthos patch 2026-08-09 — the default was backwards ─────────────────
/// This read `match env::var("CUDA_FAST_MATH") { Ok("0") => {} _ => on }` at
/// three sites: fast math was ON for an unset variable, i.e. for every normal
/// build. `--use_fast_math` implies `-fmad=true` (FMA contraction), `-ftz=true`
/// (denormals flushed to zero) and approximate div/rcp/sqrt — precisely the
/// three things `crates/neoethos-gpu-cuda/build.rs` spends a measured paragraph
/// forbidding with `-fmad=false`, because ONE contracted multiply-add moved a
/// stop/target boundary and diverged a candidate by 0.62 %.
///
/// These kernels are not merely nearby. `neoethos-data`'s GPU indicator lane
/// (`core::gpu_indicators`, policy `Auto` = "use the card whenever one is
/// present") computes the indicator columns from them, those columns become
/// `dataset.indicators`, and the fused Prototype B walk multiplies them by the
/// gene weights: `combined += weights[term] * indicator`. A changed indicator
/// flips `combined >= long_threshold` and therefore flips trades.
///
/// The 147 GPU parity fixtures hand the indicator matrix in directly, so they
/// never exercise this lane and would have stayed green while a real EURUSD M5
/// run computed different numbers — the exact shape of "numbers that are not
/// real in the end".
///
/// So: OFF unless `CUDA_FAST_MATH=1` is spelled out, and never at all for the
/// f64 lane, which until now declared itself NON-NEGOTIABLE in a comment that
/// no code read (`is_f64_lane_source` had no caller).
fn fast_math_requested(rel_src: &str) -> bool {
    if is_f64_lane_source(rel_src) {
        return false;
    }
    let on = env::var("CUDA_FAST_MATH").ok().as_deref() == Some("1");
    if on {
        println!(
            "cargo:warning=vector-ta: CUDA_FAST_MATH=1 — compiling {rel_src} with \
             --use_fast_math, which enables FMA contraction, flush-to-zero denormals and \
             approximate div/rcp/sqrt. Any parity claim made against this build is void."
        );
    }
    on
}

/// Refuse the old arbitrary compiler escape hatch before any output is made.
///
/// `NVCC_ARGS` used to be appended *after* the f64 precision flags. That made
/// it possible for ambient build configuration to re-enable fast math, FMA,
/// flush-to-zero, approximate division/sqrt, nested compiler threads, or even
/// a different architecture/output. NeoEthos exposes reviewed typed controls
/// for those decisions; unversioned free-form arguments cannot be part of a
/// truthful, reproducible CUDA artifact.
fn reject_free_form_nvcc_args() {
    let Some(value) = env::var_os("NVCC_ARGS") else {
        return;
    };
    let value = value.to_str().unwrap_or_else(|| {
        panic!(
            "vector-ta: NVCC_ARGS contains non-Unicode data. Refusing an uninspectable CUDA \
             compiler argument before launching nvcc."
        )
    });
    native_cuda_build::reject_free_form_nvcc_args(Some(value)).unwrap_or_else(|error| {
        panic!(
            "vector-ta: {error}. Remove NVCC_ARGS and use the reviewed CUDA_ARCHS, CUDA_DEBUG, \
             and CUDA_FILTER controls instead. Refusing before launching nvcc."
        )
    });
}

fn compile_cuda_kernels(cargo_jobserver: Option<&jobserver::Client>) {
    println!("cargo:rerun-if-changed=kernels/cuda");

    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=NVIDIA_SMI");
    println!("cargo:rerun-if-env-changed=CUDA_FILTER");
    println!("cargo:rerun-if-env-changed=CUDA_KERNEL_DIR");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUOBJDUMP");
    println!("cargo:rerun-if-env-changed=NVCC_ARGS");
    println!("cargo:rerun-if-env-changed=CUDA_DEBUG");
    println!("cargo:rerun-if-env-changed=CUDA_FAST_MATH");

    reject_free_form_nvcc_args();

    let cuda_path = find_cuda_path();

    compile_kernel(
        &cuda_path,
        "kernels/cuda/vector_ta_native_probe.cu",
        "vector_ta_native_probe",
    );

    compile_alma_kernel(&cuda_path);
    compile_cwma_kernel(&cuda_path);
    compile_epma_kernel(&cuda_path);
    compile_cora_wave_kernel(&cuda_path);
    compile_ehlers_ecema_kernel(&cuda_path);
    compile_kama_kernel(&cuda_path);
    compile_highpass_kernel(&cuda_path);
    compile_nama_kernel(&cuda_path);
    compile_wma_kernel(&cuda_path);
    compile_sinwma_kernel(&cuda_path);
    compile_tradjema_kernel(&cuda_path);
    compile_volume_adjusted_ma_kernel(&cuda_path);
    compile_supersmoother_3_pole_kernel(&cuda_path);
    compile_wto_kernel(&cuda_path);

    // The NeoEthos f64 indicator lane. Listed in `F64_LANE_SOURCES`, so this
    // one is compiled with `-prec-div=true -prec-sqrt=true -fmad=false
    // -ftz=false` and never with `--use_fast_math` — see `compile_kernel`.
    compile_kernel(
        &cuda_path,
        "kernels/cuda/neoethos_f64_kernels.cu",
        "neoethos_f64_kernels",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/buff_averages_kernel.cu",
        "buff_averages_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/dema_kernel.cu",
        "dema_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/dma_kernel.cu",
        "dma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/edcf_kernel.cu",
        "edcf_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_itrend_kernel.cu",
        "ehlers_itrend_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_kama_kernel.cu",
        "ehlers_kama_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_pma_kernel.cu",
        "ehlers_pma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/pma_kernel.cu",
        "pma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehma_kernel.cu",
        "ehma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ema_kernel.cu",
        "ema_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/apo_kernel.cu",
        "apo_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/frama_kernel.cu",
        "frama_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/fwma_kernel.cu",
        "fwma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/gaussian_kernel.cu",
        "gaussian_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/highpass2_kernel.cu",
        "highpass2_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/decycler_kernel.cu",
        "decycler_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/hma_kernel.cu",
        "hma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/hwma_kernel.cu",
        "hwma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/jma_kernel.cu",
        "jma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/jsa_kernel.cu",
        "jsa_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/linreg_kernel.cu",
        "linreg_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/linearreg_intercept_kernel.cu",
        "linearreg_intercept_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/linearreg_slope_kernel.cu",
        "linearreg_slope_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/tsf_kernel.cu",
        "tsf_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/maaq_kernel.cu",
        "maaq_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/mama_kernel.cu",
        "mama_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/mwdx_kernel.cu",
        "mwdx_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/nma_kernel.cu",
        "nma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vidya_kernel.cu",
        "vidya_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/pwma_kernel.cu",
        "pwma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/reflex_kernel.cu",
        "reflex_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sama_kernel.cu",
        "sama_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sgf_kernel.cu",
        "sgf_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sma_kernel.cu",
        "sma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/smma_kernel.cu",
        "smma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sqwma_kernel.cu",
        "sqwma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/srwma_kernel.cu",
        "srwma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/supersmoother_kernel.cu",
        "supersmoother_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/swma_kernel.cu",
        "swma_kernel",
    );
    // ---------------------------------------------------- closer 5, round 3
    // Two indicators that had NO `.cu` file at all before this round.
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/corrected_moving_average_kernel.cu",
        "corrected_moving_average_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_undersampled_double_moving_average_kernel.cu",
        "ehlers_undersampled_double_moving_average_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/tema_kernel.cu",
        "tema_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/tilson_kernel.cu",
        "tilson_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/trendflex_kernel.cu",
        "trendflex_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/trima_kernel.cu",
        "trima_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/trix_kernel.cu",
        "trix_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/uma_kernel.cu",
        "uma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vlma_kernel.cu",
        "vlma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vama_kernel.cu",
        "vama_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vpwma_kernel.cu",
        "vpwma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vwap_kernel.cu",
        "vwap_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vwma_kernel.cu",
        "vwma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vwmacd_kernel.cu",
        "vwmacd_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/avsl_kernel.cu",
        "avsl_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/wilders_kernel.cu",
        "wilders_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/zlema_kernel.cu",
        "zlema_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/alligator_kernel.cu",
        "alligator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/correlation_cycle_kernel.cu",
        "correlation_cycle_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/otto_kernel.cu",
        "otto_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/mab_kernel.cu",
        "mab_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/rsmk_kernel.cu",
        "rsmk_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/mean_ad_kernel.cu",
        "mean_ad_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/macz_kernel.cu",
        "macz_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/qstick_kernel.cu",
        "qstick_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ott_kernel.cu",
        "ott_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/wad_kernel.cu", "wad_kernel");
    compile_kernel(&cuda_path, "kernels/cuda/var_kernel.cu", "var_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/adosc_kernel.cu",
        "adosc_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ao_kernel.cu",
        "ao_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/bop_kernel.cu",
        "bop_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/coppock_kernel.cu",
        "coppock_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/gatorosc_kernel.cu",
        "gatorosc_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/macd_kernel.cu",
        "macd_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/mom_kernel.cu",
        "mom_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/roc_kernel.cu",
        "roc_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rsx_kernel.cu",
        "rsx_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/srsi_kernel.cu",
        "srsi_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/vosc_kernel.cu", "vosc_kernel");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/safezonestop_kernel.cu",
        "safezonestop_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/rocr_kernel.cu", "rocr_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/nadaraya_watson_envelope_kernel.cu",
        "nadaraya_watson_envelope_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/mfi_kernel.cu",
        "mfi_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/willr_kernel.cu",
        "willr_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/acosc_kernel.cu",
        "acosc_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/aroonosc_kernel.cu",
        "aroonosc_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cfo_kernel.cu",
        "cfo_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/fosc_kernel.cu",
        "fosc_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/dpo_kernel.cu",
        "dpo_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/lrsi_kernel.cu",
        "lrsi_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ppo_kernel.cu",
        "ppo_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/tsi_kernel.cu",
        "tsi_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/stoch_kernel.cu",
        "stoch_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/aso_kernel.cu",
        "aso_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cg_kernel.cu",
        "cg_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cmo_kernel.cu",
        "cmo_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/reverse_rsi_kernel.cu",
        "reverse_rsi_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rsi_kernel.cu",
        "rsi_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/dti_kernel.cu",
        "dti_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/emv_kernel.cu",
        "emv_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/kdj_kernel.cu",
        "kdj_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/stochf_kernel.cu",
        "stochf_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/squeeze_momentum_kernel.cu",
        "squeeze_momentum_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ttm_squeeze_kernel.cu",
        "ttm_squeeze_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cci_kernel.cu",
        "cci_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/chop_kernel.cu",
        "chop_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/dec_osc_kernel.cu",
        "dec_osc_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/fisher_kernel.cu",
        "fisher_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ift_rsi_kernel.cu",
        "ift_rsi_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ultosc_kernel.cu",
        "ultosc_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/wavetrend_kernel.cu",
        "wavetrend_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cci_cycle_kernel.cu",
        "cci_cycle_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/msw_kernel.cu",
        "msw_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/kst_kernel.cu",
        "kst_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/qqe_kernel.cu",
        "qqe_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rocp_kernel.cu",
        "rocp_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rvi_kernel.cu",
        "rvi_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/stc_kernel.cu",
        "stc_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/wclprice_kernel.cu",
        "wclprice_kernel",
    );
    compile_kernel(&cuda_path, "kernels/cuda/sar_kernel.cu", "sar_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/alphatrend_kernel.cu",
        "alphatrend_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/medprice_kernel.cu",
        "medprice_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/pattern_recognition_kernel.cu",
        "pattern_recognition_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/bandpass_kernel.cu",
        "bandpass_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/aroon_kernel.cu", "aroon_kernel");
    compile_kernel(&cuda_path, "kernels/cuda/zscore_kernel.cu", "zscore_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/yang_zhang_volatility_kernel.cu",
        "yang_zhang_volatility_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/garman_klass_volatility_kernel.cu",
        "garman_klass_volatility_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/parkinson_volatility_kernel.cu",
        "parkinson_volatility_kernel",
    );
    compile_kernel(&cuda_path, "kernels/cuda/voss_kernel.cu", "voss_kernel");
    compile_kernel(&cuda_path, "kernels/cuda/cksp_kernel.cu", "cksp_kernel");
    compile_kernel(&cuda_path, "kernels/cuda/emd_kernel.cu", "emd_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/emd_trend_kernel.cu",
        "emd_trend_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/minmax_kernel.cu", "minmax_kernel");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/bollinger_bands_width_kernel.cu",
        "bollinger_bands_width_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/deviation_kernel.cu",
        "deviation_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_filter_kernel.cu",
        "range_filter_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/kaufmanstop_kernel.cu",
        "kaufmanstop_kernel",
    );
    compile_kernel(&cuda_path, "kernels/cuda/mass_kernel.cu", "mass_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/kvo_kernel.cu",
        "kvo_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/natr_kernel.cu", "natr_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/linearreg_angle_kernel.cu",
        "linearreg_angle_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/net_myrsi_kernel.cu",
        "net_myrsi_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/percentile_nearest_rank_kernel.cu",
        "percentile_nearest_rank_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/prb_kernel.cu", "prb_kernel");

    compile_kernel(&cuda_path, "kernels/cuda/vi_kernel.cu", "vi_kernel");

    compile_kernel(&cuda_path, "kernels/cuda/vpci_kernel.cu", "vpci_kernel");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/mod_god_mode_kernel.cu",
        "mod_god_mode_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/bollinger_bands_kernel.cu",
        "bollinger_bands_kernel",
    );
    compile_kernel(&cuda_path, "kernels/cuda/ad_kernel.cu", "ad_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/devstop_kernel.cu",
        "devstop_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fvg_trailing_stop_kernel.cu",
        "fvg_trailing_stop_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/ttm_trend_kernel.cu",
        "ttm_trend_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/nvi_kernel.cu", "nvi_kernel");

    compile_kernel(&cuda_path, "kernels/cuda/pvi_kernel.cu", "pvi_kernel");

    compile_kernel(&cuda_path, "kernels/cuda/vpt_kernel.cu", "vpt_kernel");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/supertrend_kernel.cu",
        "supertrend_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/medium_ad_kernel.cu",
        "medium_ad_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/di_kernel.cu", "di_kernel");

    compile_kernel(&cuda_path, "kernels/cuda/atr_kernel.cu", "atr_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/atr_percentile_kernel.cu",
        "atr_percentile_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/chande_kernel.cu", "chande_kernel");

    compile_kernel(&cuda_path, "kernels/cuda/cvi_kernel.cu", "cvi_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/cycle_channel_oscillator_kernel.cu",
        "cycle_channel_oscillator_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/dvdiqqe_kernel.cu",
        "dvdiqqe_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/er_kernel.cu", "er_kernel");

    compile_kernel(&cuda_path, "kernels/cuda/pfe_kernel.cu", "pfe_kernel");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/keltner_kernel.cu",
        "keltner_kernel",
    );
    compile_kernel(&cuda_path, "kernels/cuda/adx_kernel.cu", "adx_kernel");
    compile_kernel(&cuda_path, "kernels/cuda/dm_kernel.cu", "dm_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/chandelier_exit_kernel.cu",
        "chandelier_exit_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/damiani_volatmeter_kernel.cu",
        "damiani_volatmeter_kernel",
    );
    compile_kernel(&cuda_path, "kernels/cuda/dx_kernel.cu", "dx_kernel");
    compile_kernel(&cuda_path, "kernels/cuda/eri_kernel.cu", "eri_kernel");

    compile_kernel(&cuda_path, "kernels/cuda/obv_kernel.cu", "obv_kernel");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/advance_decline_line_kernel.cu",
        "advance_decline_line_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/bull_power_vs_bear_power_kernel.cu",
        "bull_power_vs_bear_power_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/bulls_v_bears_kernel.cu",
        "bulls_v_bears_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/psychological_line_kernel.cu",
        "psychological_line_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/historical_volatility_kernel.cu",
        "historical_volatility_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/historical_volatility_rank_kernel.cu",
        "historical_volatility_rank_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/historical_volatility_percentile_kernel.cu",
        "historical_volatility_percentile_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/half_causal_estimator_kernel.cu",
        "half_causal_estimator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vertical_horizontal_filter_kernel.cu",
        "vertical_horizontal_filter_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/gopalakrishnan_range_index_kernel.cu",
        "gopalakrishnan_range_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/grover_llorens_cycle_oscillator_kernel.cu",
        "grover_llorens_cycle_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/dual_ulcer_index_kernel.cu",
        "dual_ulcer_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ewma_volatility_kernel.cu",
        "ewma_volatility_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/momentum_ratio_oscillator_kernel.cu",
        "momentum_ratio_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/on_balance_volume_oscillator_kernel.cu",
        "on_balance_volume_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/pretty_good_oscillator_kernel.cu",
        "pretty_good_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rolling_z_score_trend_kernel.cu",
        "rolling_z_score_trend_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rank_correlation_index_kernel.cu",
        "rank_correlation_index_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/halftrend_kernel.cu",
        "halftrend_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/pivot_kernel.cu", "pivot_kernel");

    compile_kernel(&cuda_path, "kernels/cuda/ui_kernel.cu", "ui_kernel");

    compile_kernel(&cuda_path, "kernels/cuda/stddev_kernel.cu", "stddev_kernel");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/donchian_channel_width_kernel.cu",
        "donchian_channel_width_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/donchian_kernel.cu",
        "donchian_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/adxr_kernel.cu", "adxr_kernel");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/correl_hl_kernel.cu",
        "correl_hl_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/efi_kernel.cu", "efi_kernel");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/marketefi_kernel.cu",
        "marketefi_kernel",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/kurtosis_kernel.cu",
        "kurtosis_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/velocity_kernel.cu",
        "velocity_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/velocity_acceleration_indicator_kernel.cu",
        "velocity_acceleration_indicator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/random_walk_index_kernel.cu",
        "random_walk_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/regression_slope_oscillator_kernel.cu",
        "regression_slope_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/reversal_signals_kernel.cu",
        "reversal_signals_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/relative_strength_index_wave_indicator_kernel.cu",
        "relative_strength_index_wave_indicator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/monotonicity_index_kernel.cu",
        "monotonicity_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/market_meanness_index_kernel.cu",
        "market_meanness_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/macd_wave_signal_pro_kernel.cu",
        "macd_wave_signal_pro_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/market_structure_trailing_stop_kernel.cu",
        "market_structure_trailing_stop_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_average_cross_probability_kernel.cu",
        "moving_average_cross_probability_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/multi_length_stochastic_average_kernel.cu",
        "multi_length_stochastic_average_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/absolute_strength_index_oscillator_kernel.cu",
        "absolute_strength_index_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/accumulation_swing_index_kernel.cu",
        "accumulation_swing_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/autocorrelation_indicator_kernel.cu",
        "autocorrelation_indicator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_bounds_rsi_kernel.cu",
        "adaptive_bounds_rsi_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_schaff_trend_cycle_kernel.cu",
        "adaptive_schaff_trend_cycle_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adjustable_ma_alternating_extremities_kernel.cu",
        "adjustable_ma_alternating_extremities_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_momentum_oscillator_kernel.cu",
        "adaptive_momentum_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/cyberpunk_value_trend_analyzer_kernel.cu",
        "cyberpunk_value_trend_analyzer_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/candle_strength_oscillator_kernel.cu",
        "candle_strength_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/hema_trend_levels_kernel.cu",
        "hema_trend_levels_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/daily_factor_kernel.cu",
        "daily_factor_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/demand_index_kernel.cu",
        "demand_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/decisionpoint_breadth_swenlin_trading_oscillator_kernel.cu",
        "decisionpoint_breadth_swenlin_trading_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/directional_imbalance_index_kernel.cu",
        "directional_imbalance_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_fm_demodulator_kernel.cu",
        "ehlers_fm_demodulator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_autocorrelation_periodogram_kernel.cu",
        "ehlers_autocorrelation_periodogram_kernel",
    );

    // ------------------------------------------------------ closer 6, round 3
    // The six from-scratch f64 kernels. Each is listed in `F64_LANE_SOURCES`
    // above, so each is compiled `-prec-div=true -prec-sqrt=true -fmad=false
    // -ftz=false` and never with `--use_fast_math`.
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/elastic_volume_weighted_moving_average_kernel.cu",
        "elastic_volume_weighted_moving_average_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ema_deviation_corrected_t3_kernel.cu",
        "ema_deviation_corrected_t3_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/logarithmic_moving_average_kernel.cu",
        "logarithmic_moving_average_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/n_order_ema_kernel.cu",
        "n_order_ema_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/volatility_adjusted_ma_kernel.cu",
        "volatility_adjusted_ma_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/wave_smoother_kernel.cu",
        "wave_smoother_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fractal_dimension_index_kernel.cu",
        "fractal_dimension_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fvg_positioning_average_kernel.cu",
        "fvg_positioning_average_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_energy_reservoirs_kernel.cu",
        "volume_energy_reservoirs_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fibonacci_entry_bands_kernel.cu",
        "fibonacci_entry_bands_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fibonacci_trailing_stop_kernel.cu",
        "fibonacci_trailing_stop_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/disparity_index_kernel.cu",
        "disparity_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/dynamic_momentum_index_kernel.cu",
        "dynamic_momentum_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/forward_backward_exponential_oscillator_kernel.cu",
        "forward_backward_exponential_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_simple_cycle_indicator_kernel.cu",
        "ehlers_simple_cycle_indicator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/evasive_supertrend_kernel.cu",
        "evasive_supertrend_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/l1_ehlers_phasor_kernel.cu",
        "l1_ehlers_phasor_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/l2_ehlers_signal_to_noise_kernel.cu",
        "l2_ehlers_signal_to_noise_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/leavitt_convolution_acceleration_kernel.cu",
        "leavitt_convolution_acceleration_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/linear_correlation_oscillator_kernel.cu",
        "linear_correlation_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_adaptive_cg_kernel.cu",
        "ehlers_adaptive_cg_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_detrending_filter_kernel.cu",
        "ehlers_detrending_filter_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_data_sampling_relative_strength_indicator_kernel.cu",
        "ehlers_data_sampling_relative_strength_indicator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_linear_extrapolation_predictor_kernel.cu",
        "ehlers_linear_extrapolation_predictor_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_smoothed_adaptive_momentum_kernel.cu",
        "ehlers_smoothed_adaptive_momentum_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_adaptive_cyber_cycle_kernel.cu",
        "ehlers_adaptive_cyber_cycle_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_bandpass_trigger_oscillator_kernel.cu",
        "adaptive_bandpass_trigger_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_macd_kernel.cu",
        "adaptive_macd_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/andean_oscillator_kernel.cu",
        "andean_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/exponential_trend_kernel.cu",
        "exponential_trend_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/didi_index_kernel.cu",
        "didi_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/geometric_bias_oscillator_kernel.cu",
        "geometric_bias_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/gmma_oscillator_kernel.cu",
        "gmma_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/hypertrend_kernel.cu",
        "hypertrend_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/hull_butterfly_oscillator_kernel.cu",
        "hull_butterfly_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/intraday_momentum_index_kernel.cu",
        "intraday_momentum_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/kairi_relative_index_kernel.cu",
        "kairi_relative_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/keltner_channel_width_oscillator_kernel.cu",
        "keltner_channel_width_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/impulse_macd_kernel.cu",
        "impulse_macd_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/linear_regression_intensity_kernel.cu",
        "linear_regression_intensity_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/nonlinear_regression_zero_lag_moving_average_kernel.cu",
        "nonlinear_regression_zero_lag_moving_average_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/neighboring_trailing_stop_kernel.cu",
        "neighboring_trailing_stop_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/polynomial_regression_extrapolation_kernel.cu",
        "polynomial_regression_extrapolation_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_oscillator_kernel.cu",
        "range_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rolling_skewness_kurtosis_kernel.cu",
        "rolling_skewness_kurtosis_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/price_density_market_noise_kernel.cu",
        "price_density_market_noise_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/premier_rsi_oscillator_kernel.cu",
        "premier_rsi_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/spearman_correlation_kernel.cu",
        "spearman_correlation_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/squeeze_index_kernel.cu",
        "squeeze_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/statistical_trailing_stop_kernel.cu",
        "statistical_trailing_stop_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/projection_oscillator_kernel.cu",
        "projection_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/qqe_weighted_oscillator_kernel.cu",
        "qqe_weighted_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_breakout_signals_kernel.cu",
        "range_breakout_signals_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_filtered_trend_signals_kernel.cu",
        "range_filtered_trend_signals_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_distance_kernel.cu",
        "stochastic_distance_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_adaptive_d_kernel.cu",
        "stochastic_adaptive_d_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_connors_rsi_kernel.cu",
        "stochastic_connors_rsi_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_money_flow_index_kernel.cu",
        "stochastic_money_flow_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/smoothed_gaussian_trend_filter_kernel.cu",
        "smoothed_gaussian_trend_filter_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_continuation_factor_kernel.cu",
        "trend_continuation_factor_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_follower_kernel.cu",
        "trend_follower_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_direction_force_index_kernel.cu",
        "trend_direction_force_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_flow_trail_kernel.cu",
        "trend_flow_trail_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_trigger_factor_kernel.cu",
        "trend_trigger_factor_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/twiggs_money_flow_kernel.cu",
        "twiggs_money_flow_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_zone_oscillator_kernel.cu",
        "volume_zone_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_weighted_rsi_kernel.cu",
        "volume_weighted_rsi_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_weighted_relative_strength_index_kernel.cu",
        "volume_weighted_relative_strength_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/price_moving_average_ratio_percentile_kernel.cu",
        "price_moving_average_ratio_percentile_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/mesa_stochastic_multi_length_kernel.cu",
        "mesa_stochastic_multi_length_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/normalized_volume_true_range_kernel.cu",
        "normalized_volume_true_range_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/normalized_resonator_kernel.cu",
        "normalized_resonator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volatility_quality_index_kernel.cu",
        "volatility_quality_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/goertzel_cycle_composite_wave_kernel.cu",
        "goertzel_cycle_composite_wave_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ict_propulsion_block_kernel.cu",
        "ict_propulsion_block_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ichimoku_oscillator_kernel.cu",
        "ichimoku_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/insync_index_kernel.cu",
        "insync_index_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/kase_peak_oscillator_with_divergences_kernel.cu",
        "kase_peak_oscillator_with_divergences_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/market_structure_confluence_kernel.cu",
        "market_structure_confluence_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/possible_rsi_kernel.cu",
        "possible_rsi_kernel",
    );
    // ------------------------------------------------------------- closer 4
    // `rogers_satchell_volatility_kernel.cu` was listed in `F64_LANE_SOURCES`
    // (so it opts out of fast math) but had NO `compile_kernel` call, so it
    // produced no native artifact and `load_cuda_embedded_module!` could never
    // find it. Added here rather than left for a launch to discover.
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rogers_satchell_volatility_kernel.cu",
        "rogers_satchell_volatility_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/smooth_theil_sen_kernel.cu",
        "smooth_theil_sen_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vdubus_divergence_wave_pattern_generator_kernel.cu",
        "vdubus_divergence_wave_pattern_generator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volatility_ratio_adaptive_rsx_kernel.cu",
        "volatility_ratio_adaptive_rsx_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/velocity_acceleration_convergence_divergence_indicator_kernel.cu",
        "velocity_acceleration_convergence_divergence_indicator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vwap_deviation_oscillator_kernel.cu",
        "vwap_deviation_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vwap_zscore_with_signals_kernel.cu",
        "vwap_zscore_with_signals_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_weighted_stochastic_rsi_kernel.cu",
        "volume_weighted_stochastic_rsi_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/supertrend_recovery_kernel.cu",
        "supertrend_recovery_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/supertrend_oscillator_kernel.cu",
        "supertrend_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/standardized_psar_oscillator_kernel.cu",
        "standardized_psar_oscillator_kernel",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/zig_zag_channels_kernel.cu",
        "zig_zag_channels_kernel",
    );

    compile_kernel(&cuda_path, "kernels/cuda/lpc_kernel.cu", "lpc_kernel");

    run_queued_kernel_jobs(cargo_jobserver, &cuda_path);
}

fn find_cuda_path() -> String {
    env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                use std::fs;
                let base = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA";
                if let Ok(entries) = fs::read_dir(base) {
                    let mut best: Option<(u32, u32, String)> = None;
                    for e in entries.flatten() {
                        if let Ok(name) = e.file_name().into_string() {
                            if let Some(stripped) = name.strip_prefix('v') {
                                let mut it = stripped.split('.');
                                let major = it.next().and_then(|s| s.parse::<u32>().ok());
                                let minor =
                                    it.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                                if let Some(maj) = major {
                                    let cand = (maj, minor, format!("{base}/{}", name));
                                    if let Some(cur) = &best {
                                        if cand.0 > cur.0 || (cand.0 == cur.0 && cand.1 > cur.1) {
                                            best = Some(cand);
                                        }
                                    } else {
                                        best = Some(cand);
                                    }
                                }
                            }
                        }
                    }
                    if let Some((_, _, path)) = best {
                        eprintln!("Found CUDA at: {}", path);
                        return path;
                    }
                }

                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        })
}

fn compile_alma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/alma_kernel.cu",
        "alma_kernel",
    );
}

fn compile_cwma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/cwma_kernel.cu",
        "cwma_kernel",
    );
}

fn compile_cora_wave_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/cora_wave_kernel.cu",
        "cora_wave_kernel",
    );
}

fn compile_epma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/epma_kernel.cu",
        "epma_kernel",
    );
}

fn compile_ehlers_ecema_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/ehlers_ecema_kernel.cu",
        "ehlers_ecema_kernel",
    );
}

fn compile_kama_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/kama_kernel.cu",
        "kama_kernel",
    );
}

fn compile_highpass_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/highpass_kernel.cu",
        "highpass_kernel",
    );
}

fn compile_nama_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/nama_kernel.cu",
        "nama_kernel",
    );
}

fn compile_wma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/wma_kernel.cu",
        "wma_kernel",
    );
}

fn compile_sinwma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/sinwma_kernel.cu",
        "sinwma_kernel",
    );
}

fn compile_tradjema_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/tradjema_kernel.cu",
        "tradjema_kernel",
    );
}

fn compile_volume_adjusted_ma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/volume_adjusted_ma_kernel.cu",
        "volume_adjusted_ma_kernel",
    );
}

fn compile_supersmoother_3_pole_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/supersmoother_3_pole_kernel.cu",
        "supersmoother_3_pole_kernel",
    );
}

fn compile_wto_kernel(cuda_path: &str) {
    compile_kernel(cuda_path, "kernels/cuda/wto_kernel.cu", "wto_kernel");
}

#[cfg(target_os = "windows")]
fn append_windows_nvcc_host_args(cmd: &mut std::process::Command) {
    cmd.arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
    cmd.arg("-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING");
    cmd.arg("-allow-unsupported-compiler");
    cmd.arg("-Xcompiler").arg("/Zc:preprocessor");

    if let Ok(vs_path) = find_vs_installation() {
        cmd.arg("-ccbin").arg(vs_path);
    }
}

#[cfg(not(target_os = "windows"))]
fn append_windows_nvcc_host_args(_cmd: &mut std::process::Command) {}

fn kernel_source_path(rel_src: &str) -> String {
    if let Ok(root) = env::var("CUDA_KERNEL_DIR") {
        let root = root.trim_end_matches(['/', '\\']);
        let prefix = "kernels/cuda/";
        if rel_src.starts_with(prefix) {
            format!("{}/{}", root, &rel_src[prefix.len()..])
        } else {
            rel_src.to_string()
        }
    } else {
        rel_src.to_string()
    }
}

fn cuda_filter_matches(rel_src: &str) -> bool {
    let Ok(filter) = env::var("CUDA_FILTER") else {
        return true;
    };
    filter
        .split(|c: char| c == ',' || c.is_ascii_whitespace())
        .map(str::trim)
        .any(|token| !token.is_empty() && rel_src.contains(token))
}

/// Record one source job. The 300+ existing declarations stay declarative and
/// deterministic; no child process is launched until the shared Cargo-aware
/// scheduler drains the complete queue.
fn compile_kernel(_cuda_path: &str, rel_src: &'static str, cubin_stem: &'static str) {
    let src_path = kernel_source_path(rel_src);
    println!("cargo:rerun-if-changed={}", src_path);

    if !cuda_filter_matches(rel_src) {
        eprintln!("Skipping {} due to CUDA_FILTER", rel_src);
        return;
    }

    let source_path = PathBuf::from(src_path);
    let source_bytes = std::fs::metadata(&source_path)
        .unwrap_or_else(|error| {
            panic!(
                "vector-ta: cannot stat CUDA source {} before scheduling NVCC: {error}",
                source_path.display()
            )
        })
        .len();
    KERNEL_JOBS.with(|queue| {
        queue.borrow_mut().push(KernelJob::new(
            rel_src.to_owned(),
            cubin_stem.to_owned(),
            source_path,
            source_bytes,
            measured_kernel_tail_priority(rel_src),
        ));
    });
}

fn native_nvcc_path(cuda_path: &str) -> String {
    env::var("NVCC").unwrap_or_else(|_| {
        if cfg!(target_os = "windows") {
            format!("{cuda_path}/bin/nvcc.exe")
        } else {
            format!("{cuda_path}/bin/nvcc")
        }
    })
}

fn native_cuobjdump_path(cuda_path: &str) -> String {
    env::var("CUOBJDUMP").unwrap_or_else(|_| {
        if cfg!(target_os = "windows") {
            format!("{cuda_path}/bin/cuobjdump.exe")
        } else {
            format!("{cuda_path}/bin/cuobjdump")
        }
    })
}

fn tool_version(tool: &Path) -> String {
    let output = Command::new(tool)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| panic!("vector-ta: failed to run {tool:?} --version: {error}"));
    if !output.status.success() {
        panic!(
            "vector-ta: {tool:?} --version failed; stdout={:?}; stderr={:?}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let combined = format!(
        "{} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    combined.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn validate_native_outputs_and_write_registry(
    stems: &[&str],
    arch_plan: &NativeArchPlan,
    nvcc: &Path,
) {
    use std::fmt::Write as _;

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let mut owned = Vec::with_capacity(stems.len() * arch_plan.architectures.len());
    for &stem in stems {
        for &arch in &arch_plan.architectures {
            let filename = native_cubin_filename(stem, arch);
            let bytes = std::fs::read(out_dir.join(&filename)).unwrap_or_else(|error| {
                panic!(
                    "vector-ta: native cubin manifest is missing {filename}: {error}; refusing an \
                     incomplete registry"
                )
            });
            owned.push((stem, arch, bytes));
        }
    }
    let records: Vec<NativeArtifact<'_>> = owned
        .iter()
        .map(|(stem, arch, bytes)| NativeArtifact::new(*stem, *arch, bytes.as_slice()))
        .collect();
    validate_native_manifest(&records, stems, &arch_plan.architectures).unwrap_or_else(|error| {
        panic!("vector-ta: native cubin manifest validation failed: {error}")
    });

    // Exercise the same exact-architecture selector used by the CUDA runtime
    // against every generated registry coordinate before publishing it. This
    // proves that no naming/architecture mismatch can create a registry that
    // passes the Cartesian manifest check but cannot be selected at runtime.
    for &stem in stems {
        for &arch in &arch_plan.architectures {
            let major = i32::try_from(arch / 10).expect("CUDA architecture major fits i32");
            let minor = i32::try_from(arch % 10).expect("CUDA architecture minor fits i32");
            select_exact_native_cubin(stem, major, minor, &records).unwrap_or_else(|error| {
                panic!(
                    "vector-ta: generated native cubin registry failed exact runtime-selection \
                     preflight for {stem} sm_{arch}: {error}"
                )
            });
        }
    }

    let nvcc_version = tool_version(nvcc);
    println!("cargo:rustc-env=VECTOR_TA_CUDA_NVCC_VERSION={nvcc_version}");

    let mut generated = String::new();
    writeln!(
        generated,
        "// @generated by vector-ta build.rs; do not edit."
    )
    .unwrap();
    writeln!(
        generated,
        "pub(super) const COMPILED_ARCHS: &[u32] = &{:?};",
        arch_plan.architectures
    )
    .unwrap();
    writeln!(
        generated,
        "pub(super) const COMPILED_ARCH_SOURCE: &str = {:?};",
        arch_plan.source.as_str()
    )
    .unwrap();
    writeln!(
        generated,
        "pub(super) const NVCC_VERSION: &str = {:?};",
        nvcc_version
    )
    .unwrap();
    writeln!(
        generated,
        "pub(super) const NATIVE_CUBIN_COUNT: usize = {};",
        records.len()
    )
    .unwrap();
    writeln!(
        generated,
        "pub(super) static NATIVE_CUBINS: &[crate::native_sass::NativeArtifact<'static>] = &["
    )
    .unwrap();
    for &stem in stems {
        for &arch in &arch_plan.architectures {
            let filename = native_cubin_filename(stem, arch);
            writeln!(
                generated,
                "    crate::native_sass::NativeArtifact::new({stem:?}, {arch}, \
                 include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{filename}\"))),"
            )
            .unwrap();
        }
    }
    writeln!(generated, "];").unwrap();
    std::fs::write(
        out_dir.join("vector_ta_native_cubin_registry.rs"),
        generated,
    )
    .expect("write exact native cubin registry");

    eprintln!(
        "vector-ta build info: verified and registered {} exact native cubins for {:?}; nvcc={}",
        records.len(),
        arch_plan.architectures,
        nvcc_version
    );
}

struct NativeNvccCompiler {
    nvcc: PathBuf,
    debug_line_info: bool,
}

impl ArtifactCompiler for NativeNvccCompiler {
    fn command(&self, job: &ArtifactJob) -> Result<Command, String> {
        if cfg!(target_os = "windows") && env::var("VCINSTALLDIR").is_err() {
            eprintln!(
                "Warning: VCINSTALLDIR not set. CUDA compilation may require running inside a Visual Studio Developer Command Prompt."
            );
        }

        // NON-NEGOTIABLE: the f64 lane never sees `--use_fast_math`, whatever
        // `CUDA_FAST_MATH` says. See `F64_LANE_SOURCES`.
        let precision = if is_f64_lane_source(&job.rel_src) {
            NativePrecision::StrictF64
        } else if fast_math_requested(&job.rel_src) {
            NativePrecision::FastMath
        } else {
            NativePrecision::Default
        };
        let mut command = Command::new(&self.nvcc);
        command.args(native_nvcc_args(
            job,
            NativeCompileOptions {
                precision,
                debug_line_info: self.debug_line_info,
            },
        ));
        if cfg!(target_os = "windows") {
            append_windows_nvcc_host_args(&mut command);
        }
        eprintln!("Running nvcc native-SASS command: {command:?}");
        Ok(command)
    }

    fn finish(&self, job: &ArtifactJob, output: std::process::Output) -> Result<(), String> {
        if !output.status.success() {
            return Err(format!(
                "nvcc --cubin failed for {} at exact sm_{} with {:?}; stdout={:?}; stderr={:?}",
                job.rel_src,
                job.arch,
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        println!(
            "Successfully compiled {} to {} (exact sm_{} native cubin)",
            job.source_path.display(),
            job.output_path.display(),
            job.arch
        );
        Ok(())
    }
}

struct NativeCubinVerifier {
    cuobjdump: PathBuf,
}

impl ArtifactVerifier for NativeCubinVerifier {
    fn verify(&self, job: &ArtifactJob) -> Result<(), String> {
        inspect_native_cubin(&self.cuobjdump, &job.output_path, job.arch).map_err(|error| {
            format!(
                "native cubin verification failed for {}: {error}",
                job.output_path.display()
            )
        })?;
        println!(
            "Successfully verified {} (exact sm_{} native SASS only)",
            job.output_path.display(),
            job.arch
        );
        Ok(())
    }
}
#[cfg(target_os = "windows")]

fn find_vs_installation() -> Result<String, ()> {
    let vs_paths = [
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC",
        "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2019/Professional/VC/Tools/MSVC",
        "C:/Program Files/Microsoft Visual Studio/2019/Enterprise/VC/Tools/MSVC",
    ];

    for vs_base in &vs_paths {
        if let Ok(entries) = std::fs::read_dir(vs_base) {
            if let Some(msvc_version) = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|name| name.starts_with("14."))
                .max()
            {
                let cl_path = format!("{}/{}/bin/Hostx64/x64", vs_base, msvc_version);
                if std::path::Path::new(&format!("{}/cl.exe", cl_path)).exists() {
                    eprintln!("Found cl.exe at: {}", cl_path);
                    return Ok(cl_path);
                }
            }
        }
    }

    Err(())
}
