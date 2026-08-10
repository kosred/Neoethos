use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/cuda");
    println!("cargo:rerun-if-changed=kernels/ptx");
    println!("cargo:rerun-if-changed=kernels/cubin");

    // Sentinels first. `target_archs()` re-emits both with the real values
    // when kernels are compiled from source, and a later `cargo:rustc-env`
    // overrides an earlier one — so these are what remains when the fatbin
    // path did not run (no `cuda` feature, or prebuilt staging). The runtime
    // loader treats "unknown" as "we cannot name what was compiled" rather
    // than inventing an architecture.
    println!("cargo:rustc-env=VECTOR_TA_CUDA_ARCHS=unknown");
    println!("cargo:rustc-env=VECTOR_TA_CUDA_PTX_ARCH=unknown");

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        if env::var("CARGO_FEATURE_CUDA_BUILD_PTX").is_ok() {
            compile_cuda_kernels();
        } else {
            stage_prebuilt_ptx();
        }
    }

    if is_nightly() {
        println!("cargo:rustc-cfg=rustc_is_nightly");
    }
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

// ── NeoEthos patch 2026-08-09 — ARCH-AGNOSTIC BINARY ARTIFACT ─────────────
// Upstream emitted `<stem>_sm89.cubin`: a SINGLE-architecture cubin for Ada,
// which `module_loader.rs` would only load when the running device reported
// compute capability exactly 8.9. That is the arch trap in its binary form —
// the artifact names one card in its FILENAME.
//
// The replacement is `<stem>.fatbin`: one container carrying SASS for every
// architecture in `target_archs()` PLUS embedded PTX at the highest of them,
// so the driver picks the exact SASS for the present card and JITs the PTX
// forward onto anything newer. See `compile_kernel`.
fn fatbin_name_for_ptx(ptx_name: &str) -> String {
    if let Some(stem) = ptx_name.strip_suffix(".ptx") {
        format!("{stem}.fatbin")
    } else {
        format!("{ptx_name}.fatbin")
    }
}

/// The architectures a stock build targets when the operator names none.
///
/// These are the four cards the project actually runs on — A100 (8.0),
/// RTX 3090 / A10 (8.6), RTX 4090 / L40S (8.9), H100 (9.0). A device NEWER
/// than the highest entry is served by the PTX that `compile_kernel` embeds
/// in the same fatbin, so "a card we have not compiled for" is a JIT, not a
/// failure. A device OLDER than 8.0 is not covered and is refused loudly by
/// `module_loader.rs` rather than silently mis-run.
const DEFAULT_TARGET_ARCHS: &[u32] = &[80, 86, 89, 90];

/// Kernel sources whose entry points feed the NeoEthos f64 indicator lane.
///
/// NON-NEGOTIABLE: these are NEVER compiled with `--use_fast_math`, whatever
/// `CUDA_FAST_MATH` says. The shipped PTX proves this is not a hypothetical —
/// it carries 23 `rcp.approx.ftz.f64` instructions across 17 files, i.e. fast
/// math already degrades f64 results in this crate today. Our lane opts out
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
    // comment above -- the shipped PTX already carries 23 `rcp.approx.ftz.f64`
    // instructions, i.e. fast math is degrading f64 in this crate TODAY.
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
    // where the shipped PTX is shown carrying 23 `rcp.approx.ftz.f64`
    // instructions, i.e. the flag is degrading f64 in this crate today and has
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
    // how 23 `rcp.approx.ftz.f64` instructions reached the shipped PTX in the
    // first place.
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
    // contamination the doc comment at the top of this list measures in the
    // shipped PTX (23 `rcp.approx.ftz.f64` across 17 files) -- these files were
    // part of the reason the number was not zero.
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
    // shipped PTX carries `rcp.approx.ftz.f64` -- fast math degrading f64 --
    // which is the measured defect this list exists to remove.
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
    // `rcp.approx.ftz.f64` (the shipped PTX already carries 23 of those), and
    // the error compounds bar over bar into a value that feeds a threshold
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

/// `sm_86` / `compute_86` / `8.6` / `86` → `86`. `None` when it is not an arch.
fn parse_arch(s: &str) -> Option<u32> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() < 2 {
        return None;
    }
    // `90a` / `100f` architecture-specific variants collapse to their base.
    digits[..digits.len().min(3)].parse::<u32>().ok().map(|v| {
        // "8.6" -> "86"; "100" stays 100 (Blackwell).
        v
    })
}

/// Which real-SASS architectures this nvcc can emit, e.g. `[70, 75, 80, 86,
/// 89, 90]`. Parsed from `nvcc --list-gpu-arch`, which prints one
/// `compute_XY` per line.
///
/// Computed once — `compile_kernel` runs ~330 times per build and this must
/// not spawn a process each time.
fn nvcc_supported_archs(nvcc: &str) -> &'static Vec<u32> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec<u32>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let out = Command::new(nvcc).arg("--list-gpu-arch").output();
        let mut archs: Vec<u32> = match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|l| l.trim().strip_prefix("compute_"))
                .filter_map(|d| d.trim().parse::<u32>().ok())
                .collect(),
            _ => {
                // Old nvcc without `--list-gpu-arch`. Do not guess narrow and
                // do not fail: assume the requested set is servable and let
                // nvcc reject it with its own message if not.
                println!(
                    "cargo:warning=vector-ta: `nvcc --list-gpu-arch` unavailable; cannot verify \
                     which architectures this toolkit supports. Proceeding with the requested set."
                );
                Vec::new()
            }
        };
        archs.sort_unstable();
        archs.dedup();
        archs
    })
}

/// The architectures the fatbin will carry, ascending, always non-empty.
///
/// Precedence:
///   1. `CUDA_ARCHS` — a comma/space separated LIST, all of which are built
///   2. `CUDA_ARCH`  — a single architecture (narrow, deliberate build)
///   3. [`DEFAULT_TARGET_ARCHS`] — the portable default, no card named at the
///      call site and no auto-detection of the build host
///
/// Then intersected with [`nvcc_supported_archs`] so an older or newer toolkit
/// produces a working narrower fatbin instead of failing the whole build.
fn target_archs(nvcc: &str) -> &'static Vec<u32> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec<u32>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let requested: Vec<u32> = if let Ok(list) = env::var("CUDA_ARCHS") {
            let v: Vec<u32> = list
                .split(|c: char| c == ',' || c.is_ascii_whitespace())
                .filter_map(parse_arch)
                .collect();
            if v.is_empty() {
                panic!("vector-ta: CUDA_ARCHS={list:?} contains no parseable architecture");
            }
            v
        } else if let Ok(one) = env::var("CUDA_ARCH") {
            let a = parse_arch(&one).unwrap_or_else(|| {
                panic!("vector-ta: CUDA_ARCH={one:?} does not parse as an architecture")
            });
            println!(
                "cargo:warning=vector-ta: CUDA_ARCH={one} builds a SINGLE-architecture fatbin. \
                 The resulting binary runs on sm_{a} and (via embedded PTX) newer cards only. \
                 Unset it — or use CUDA_ARCHS — for the portable default {DEFAULT_TARGET_ARCHS:?}."
            );
            vec![a]
        } else {
            DEFAULT_TARGET_ARCHS.to_vec()
        };

        let supported = nvcc_supported_archs(nvcc);
        let mut archs: Vec<u32> = if supported.is_empty() {
            requested.clone()
        } else {
            requested
                .iter()
                .copied()
                .filter(|a| supported.contains(a))
                .collect()
        };
        archs.sort_unstable();
        archs.dedup();

        if archs.is_empty() {
            panic!(
                "vector-ta: none of the requested CUDA architectures {requested:?} are supported \
                 by this nvcc (it offers {supported:?}). Set CUDA_ARCHS to a subset it can build, \
                 or install a CUDA toolkit that covers the cards you intend to run on. Refusing \
                 to silently substitute a different architecture."
            );
        }

        if archs.len() != requested.len() {
            let dropped: Vec<u32> = requested
                .iter()
                .copied()
                .filter(|a| !archs.contains(a))
                .collect();
            println!(
                "cargo:warning=vector-ta: this nvcc cannot emit SASS for {dropped:?}; the fatbin \
                 will carry {archs:?} plus forward PTX. Cards matching the dropped architectures \
                 will NOT run this build."
            );
        }

        // Recorded so the runtime loader's error can say what was compiled.
        let joined: Vec<String> = archs.iter().map(|a| format!("sm_{a}")).collect();
        println!("cargo:rustc-env=VECTOR_TA_CUDA_ARCHS={}", joined.join(","));
        println!(
            "cargo:rustc-env=VECTOR_TA_CUDA_PTX_ARCH=compute_{}",
            archs.last().expect("non-empty")
        );
        println!(
            "cargo:warning=vector-ta: building multi-arch fatbins for {archs:?} + forward PTX at \
             compute_{}",
            archs.last().expect("non-empty")
        );
        archs
    })
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

fn stage_prebuilt_ptx() {
    println!("cargo:rerun-if-env-changed=VECTOR_TA_PREBUILT_PTX_DIR");
    println!("cargo:rerun-if-env-changed=VECTOR_TA_PREBUILT_CUBIN_DIR");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    // ── NeoEthos patch 2026-08-09 — no default architecture, ever ─────────
    // Upstream defaulted to `kernels/ptx/compute_89`. Every one of the 329
    // files in that directory declares a literal `.target sm_89`, and PTX is
    // FORWARD-compatible only: none of them load on an A100 (sm_80) or an
    // RTX 3090 (sm_86). Defaulting to it means a build that names no
    // architecture silently produces an Ada-only binary.
    //
    // There is no safe default here, because a prebuilt directory IS one
    // architecture by construction. So: refuse, and name the two ways out.
    // The arch-agnostic lane is `--features cuda-build-ptx`, which compiles a
    // multi-arch fatbin (see `compile_kernel`). Staging a single-arch PTX tree
    // is still allowed, but only when the operator asks for it BY NAME.
    let ptx_dir = match env::var("VECTOR_TA_PREBUILT_PTX_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => panic!(
            "vector-ta: `cuda` is enabled without `cuda-build-ptx`, so the build wants a PREBUILT \
             PTX directory — and there is no architecture-neutral one to default to.\n\
             \n\
             Every file under `kernels/ptx/compute_89` declares `.target sm_89` (Ada / RTX 4090). \
             PTX runs FORWARD only, so those modules do not load on sm_80 (A100) or sm_86 \
             (RTX 3090); defaulting to them is how a build silently becomes 4090-only.\n\
             \n\
             Pick one:\n\
             \n\
               1. RECOMMENDED — build the arch-agnostic fatbin from source:\n\
                    cargo build -p vector-ta --features cuda-build-ptx\n\
                  This emits SASS for {DEFAULT_TARGET_ARCHS:?} plus embedded PTX for forward JIT, \
                  in ONE artifact that runs unchanged on all of them.\n\
             \n\
               2. Deliberately stage a single-architecture PTX tree, accepting that the \
                  resulting binary runs on that architecture and newer only:\n\
                    VECTOR_TA_PREBUILT_PTX_DIR=<dir> cargo build -p vector-ta --features cuda\n\
                  (the shipped Ada tree is `kernels/ptx/compute_89`)",
        ),
    };

    // The binary artifact is a multi-arch fatbin now, not an sm_89 cubin.
    // A prebuilt tree may supply one; when it does not, zero-byte placeholders
    // are written so `include_bytes!` still resolves and `module_loader.rs`
    // falls through to the staged PTX.
    let cubin_dir = match env::var("VECTOR_TA_PREBUILT_FATBIN_DIR")
        .or_else(|_| env::var("VECTOR_TA_PREBUILT_CUBIN_DIR"))
    {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => manifest_dir.join("kernels/fatbin"),
    };

    if !ptx_dir.is_dir() {
        panic!(
            "Prebuilt PTX directory not found: {}. \
Enable `--features cuda-build-ptx` to compile PTX and cubin artifacts with nvcc, or set VECTOR_TA_PREBUILT_PTX_DIR to a directory containing *.ptx files.",
            ptx_dir.display()
        );
    }

    let mut ptx_files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&ptx_dir).expect("read prebuilt PTX dir") {
        let entry = entry.expect("read prebuilt PTX dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("ptx") {
            ptx_files.push(path);
        }
    }

    let mut cubin_files: Vec<PathBuf> = Vec::new();
    if cubin_dir.is_dir() {
        for entry in std::fs::read_dir(&cubin_dir).expect("read prebuilt cubin dir") {
            let entry = entry.expect("read prebuilt cubin dir entry");
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("fatbin") {
                cubin_files.push(path);
            }
        }
    } else {
        println!(
            "cargo:warning=Prebuilt cubin directory not found: {}. Continuing with PTX-only staging.",
            cubin_dir.display()
        );
    }

    if ptx_files.is_empty() {
        panic!(
            "No prebuilt PTX files (*.ptx) found in {}. \
Enable `--features cuda-build-ptx` to compile PTX artifacts with nvcc.",
            ptx_dir.display()
        );
    }

    for src in ptx_files {
        println!("cargo:rerun-if-changed={}", src.display());
        let file_name = src
            .file_name()
            .expect("PTX file name")
            .to_string_lossy()
            .to_string();
        let dst = out_dir.join(&file_name);
        std::fs::copy(&src, &dst).unwrap_or_else(|e| {
            panic!(
                "Failed copying prebuilt PTX {} -> {}: {e}",
                src.display(),
                dst.display()
            )
        });
    }

    let mut staged_cubins = HashSet::new();
    for src in cubin_files {
        println!("cargo:rerun-if-changed={}", src.display());
        let file_name = src
            .file_name()
            .expect("cubin file name")
            .to_string_lossy()
            .to_string();
        let dst = out_dir.join(&file_name);
        std::fs::copy(&src, &dst).unwrap_or_else(|e| {
            panic!(
                "Failed copying prebuilt cubin {} -> {}: {e}",
                src.display(),
                dst.display()
            )
        });
        staged_cubins.insert(file_name);
    }

    for entry in std::fs::read_dir(&out_dir).expect("read staged PTX dir") {
        let entry = entry.expect("read staged PTX dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("ptx") {
            continue;
        }
        let ptx_name = path
            .file_name()
            .expect("PTX file name")
            .to_string_lossy()
            .to_string();
        let cubin_name = fatbin_name_for_ptx(&ptx_name);
        if staged_cubins.contains(&cubin_name) {
            continue;
        }
        std::fs::write(out_dir.join(cubin_name), []).expect("write placeholder cubin");
    }
}

fn compile_cuda_kernels() {
    println!("cargo:rerun-if-changed=kernels/cuda");

    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_FILTER");
    println!("cargo:rerun-if-env-changed=CUDA_KERNEL_DIR");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=NVCC_ARGS");
    println!("cargo:rerun-if-env-changed=CUDA_DEBUG");
    println!("cargo:rerun-if-env-changed=CUDA_FAST_MATH");
    println!("cargo:rerun-if-env-changed=VECTOR_TA_PREBUILD_PTX_DIR");
    println!("cargo:rerun-if-env-changed=VECTOR_TA_PREBUILD_CUBIN_DIR");

    let cuda_path = find_cuda_path();

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
        "neoethos_f64_kernels.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/buff_averages_kernel.cu",
        "buff_averages_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/dema_kernel.cu",
        "dema_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/dma_kernel.cu",
        "dma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/edcf_kernel.cu",
        "edcf_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_itrend_kernel.cu",
        "ehlers_itrend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_kama_kernel.cu",
        "ehlers_kama_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_pma_kernel.cu",
        "ehlers_pma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/pma_kernel.cu",
        "pma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehma_kernel.cu",
        "ehma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ema_kernel.cu",
        "ema_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/apo_kernel.cu",
        "apo_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/frama_kernel.cu",
        "frama_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/fwma_kernel.cu",
        "fwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/gaussian_kernel.cu",
        "gaussian_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/highpass2_kernel.cu",
        "highpass2_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/decycler_kernel.cu",
        "decycler_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/hma_kernel.cu",
        "hma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/hwma_kernel.cu",
        "hwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/jma_kernel.cu",
        "jma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/jsa_kernel.cu",
        "jsa_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/linreg_kernel.cu",
        "linreg_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/linearreg_intercept_kernel.cu",
        "linearreg_intercept_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/linearreg_slope_kernel.cu",
        "linearreg_slope_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/tsf_kernel.cu",
        "tsf_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/maaq_kernel.cu",
        "maaq_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/mama_kernel.cu",
        "mama_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/mwdx_kernel.cu",
        "mwdx_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/nma_kernel.cu",
        "nma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vidya_kernel.cu",
        "vidya_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/pwma_kernel.cu",
        "pwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/reflex_kernel.cu",
        "reflex_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sama_kernel.cu",
        "sama_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sgf_kernel.cu",
        "sgf_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sma_kernel.cu",
        "sma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/smma_kernel.cu",
        "smma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/sqwma_kernel.cu",
        "sqwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/srwma_kernel.cu",
        "srwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/supersmoother_kernel.cu",
        "supersmoother_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/swma_kernel.cu",
        "swma_kernel.ptx",
    );
    // ---------------------------------------------------- closer 5, round 3
    // Two indicators that had NO `.cu` file at all before this round.
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/corrected_moving_average_kernel.cu",
        "corrected_moving_average_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ehlers_undersampled_double_moving_average_kernel.cu",
        "ehlers_undersampled_double_moving_average_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/tema_kernel.cu",
        "tema_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/tilson_kernel.cu",
        "tilson_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/trendflex_kernel.cu",
        "trendflex_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/trima_kernel.cu",
        "trima_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/trix_kernel.cu",
        "trix_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/uma_kernel.cu",
        "uma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vlma_kernel.cu",
        "vlma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vama_kernel.cu",
        "vama_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vpwma_kernel.cu",
        "vpwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vwap_kernel.cu",
        "vwap_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vwma_kernel.cu",
        "vwma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vidya_kernel.cu",
        "vidya_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/vwmacd_kernel.cu",
        "vwmacd_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/avsl_kernel.cu",
        "avsl_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/wilders_kernel.cu",
        "wilders_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/zlema_kernel.cu",
        "zlema_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/alligator_kernel.cu",
        "alligator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/correlation_cycle_kernel.cu",
        "correlation_cycle_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/otto_kernel.cu",
        "otto_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/mab_kernel.cu",
        "mab_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/rsmk_kernel.cu",
        "rsmk_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/mean_ad_kernel.cu",
        "mean_ad_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/macz_kernel.cu",
        "macz_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/qstick_kernel.cu",
        "qstick_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ott_kernel.cu",
        "ott_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/wad_kernel.cu", "wad_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/var_kernel.cu", "var_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/adosc_kernel.cu",
        "adosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ao_kernel.cu",
        "ao_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/bop_kernel.cu",
        "bop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/coppock_kernel.cu",
        "coppock_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/gatorosc_kernel.cu",
        "gatorosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/macd_kernel.cu",
        "macd_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/mom_kernel.cu",
        "mom_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/roc_kernel.cu",
        "roc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rsx_kernel.cu",
        "rsx_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/srsi_kernel.cu",
        "srsi_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/vosc_kernel.cu", "vosc_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/safezonestop_kernel.cu",
        "safezonestop_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/rocr_kernel.cu", "rocr_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/nadaraya_watson_envelope_kernel.cu",
        "nadaraya_watson_envelope_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/mfi_kernel.cu",
        "mfi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/willr_kernel.cu",
        "willr_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/acosc_kernel.cu",
        "acosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/aroonosc_kernel.cu",
        "aroonosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cfo_kernel.cu",
        "cfo_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/fosc_kernel.cu",
        "fosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/dpo_kernel.cu",
        "dpo_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/lrsi_kernel.cu",
        "lrsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ppo_kernel.cu",
        "ppo_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/tsi_kernel.cu",
        "tsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/stoch_kernel.cu",
        "stoch_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/aso_kernel.cu",
        "aso_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cg_kernel.cu",
        "cg_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cmo_kernel.cu",
        "cmo_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/reverse_rsi_kernel.cu",
        "reverse_rsi_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rsi_kernel.cu",
        "rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/dti_kernel.cu",
        "dti_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/emv_kernel.cu",
        "emv_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/kdj_kernel.cu",
        "kdj_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/stochf_kernel.cu",
        "stochf_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/squeeze_momentum_kernel.cu",
        "squeeze_momentum_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ttm_squeeze_kernel.cu",
        "ttm_squeeze_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cci_kernel.cu",
        "cci_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/chop_kernel.cu",
        "chop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/dec_osc_kernel.cu",
        "dec_osc_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/fisher_kernel.cu",
        "fisher_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ift_rsi_kernel.cu",
        "ift_rsi_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/ultosc_kernel.cu",
        "ultosc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/wavetrend_kernel.cu",
        "wavetrend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/cci_cycle_kernel.cu",
        "cci_cycle_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/msw_kernel.cu",
        "msw_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/kst_kernel.cu",
        "kst_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/qqe_kernel.cu",
        "qqe_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rocp_kernel.cu",
        "rocp_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/rvi_kernel.cu",
        "rvi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/stc_kernel.cu",
        "stc_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/wclprice_kernel.cu",
        "wclprice_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/sar_kernel.cu", "sar_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/alphatrend_kernel.cu",
        "alphatrend_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/medprice_kernel.cu",
        "medprice_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/pattern_recognition_kernel.cu",
        "pattern_recognition_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/bandpass_kernel.cu",
        "bandpass_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/aroon_kernel.cu",
        "aroon_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/zscore_kernel.cu",
        "zscore_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/yang_zhang_volatility_kernel.cu",
        "yang_zhang_volatility_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/garman_klass_volatility_kernel.cu",
        "garman_klass_volatility_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/parkinson_volatility_kernel.cu",
        "parkinson_volatility_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/voss_kernel.cu", "voss_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/cksp_kernel.cu", "cksp_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/emd_kernel.cu", "emd_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/emd_trend_kernel.cu",
        "emd_trend_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/minmax_kernel.cu",
        "minmax_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/bollinger_bands_width_kernel.cu",
        "bollinger_bands_width_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/deviation_kernel.cu",
        "deviation_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_filter_kernel.cu",
        "range_filter_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/kaufmanstop_kernel.cu",
        "kaufmanstop_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/mass_kernel.cu", "mass_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/oscillators/kvo_kernel.cu",
        "kvo_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/natr_kernel.cu", "natr_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/linearreg_angle_kernel.cu",
        "linearreg_angle_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/net_myrsi_kernel.cu",
        "net_myrsi_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/percentile_nearest_rank_kernel.cu",
        "percentile_nearest_rank_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/prb_kernel.cu", "prb_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/vi_kernel.cu", "vi_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/vpci_kernel.cu", "vpci_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/mod_god_mode_kernel.cu",
        "mod_god_mode_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/bollinger_bands_kernel.cu",
        "bollinger_bands_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/ad_kernel.cu", "ad_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/devstop_kernel.cu",
        "devstop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fvg_trailing_stop_kernel.cu",
        "fvg_trailing_stop_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/ttm_trend_kernel.cu",
        "ttm_trend_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/nvi_kernel.cu", "nvi_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/pvi_kernel.cu", "pvi_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/vpt_kernel.cu", "vpt_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/supertrend_kernel.cu",
        "supertrend_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/medium_ad_kernel.cu",
        "medium_ad_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/di_kernel.cu", "di_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/atr_kernel.cu", "atr_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/atr_percentile_kernel.cu",
        "atr_percentile_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/chande_kernel.cu",
        "chande_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/cvi_kernel.cu", "cvi_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/cycle_channel_oscillator_kernel.cu",
        "cycle_channel_oscillator_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/dvdiqqe_kernel.cu",
        "dvdiqqe_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/er_kernel.cu", "er_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/pfe_kernel.cu", "pfe_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/keltner_kernel.cu",
        "keltner_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/adx_kernel.cu", "adx_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/dm_kernel.cu", "dm_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/chandelier_exit_kernel.cu",
        "chandelier_exit_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/damiani_volatmeter_kernel.cu",
        "damiani_volatmeter_kernel.ptx",
    );
    compile_kernel(&cuda_path, "kernels/cuda/dx_kernel.cu", "dx_kernel.ptx");
    compile_kernel(&cuda_path, "kernels/cuda/eri_kernel.cu", "eri_kernel.ptx");

    compile_kernel(&cuda_path, "kernels/cuda/obv_kernel.cu", "obv_kernel.ptx");
    compile_kernel(
        &cuda_path,
        "kernels/cuda/advance_decline_line_kernel.cu",
        "advance_decline_line_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/bull_power_vs_bear_power_kernel.cu",
        "bull_power_vs_bear_power_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/bulls_v_bears_kernel.cu",
        "bulls_v_bears_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/psychological_line_kernel.cu",
        "psychological_line_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/historical_volatility_kernel.cu",
        "historical_volatility_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/historical_volatility_rank_kernel.cu",
        "historical_volatility_rank_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/historical_volatility_percentile_kernel.cu",
        "historical_volatility_percentile_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/half_causal_estimator_kernel.cu",
        "half_causal_estimator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vertical_horizontal_filter_kernel.cu",
        "vertical_horizontal_filter_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/gopalakrishnan_range_index_kernel.cu",
        "gopalakrishnan_range_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/grover_llorens_cycle_oscillator_kernel.cu",
        "grover_llorens_cycle_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/dual_ulcer_index_kernel.cu",
        "dual_ulcer_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ewma_volatility_kernel.cu",
        "ewma_volatility_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/momentum_ratio_oscillator_kernel.cu",
        "momentum_ratio_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/on_balance_volume_oscillator_kernel.cu",
        "on_balance_volume_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/pretty_good_oscillator_kernel.cu",
        "pretty_good_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rolling_z_score_trend_kernel.cu",
        "rolling_z_score_trend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rank_correlation_index_kernel.cu",
        "rank_correlation_index_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/halftrend_kernel.cu",
        "halftrend_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/pivot_kernel.cu",
        "pivot_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/ui_kernel.cu", "ui_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/stddev_kernel.cu",
        "stddev_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/donchian_channel_width_kernel.cu",
        "donchian_channel_width_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/donchian_kernel.cu",
        "donchian_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/adxr_kernel.cu", "adxr_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/correl_hl_kernel.cu",
        "correl_hl_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/efi_kernel.cu", "efi_kernel.ptx");

    compile_kernel(
        &cuda_path,
        "kernels/cuda/marketefi_kernel.cu",
        "marketefi_kernel.ptx",
    );

    compile_kernel(
        &cuda_path,
        "kernels/cuda/kurtosis_kernel.cu",
        "kurtosis_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/velocity_kernel.cu",
        "velocity_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/velocity_acceleration_indicator_kernel.cu",
        "velocity_acceleration_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/random_walk_index_kernel.cu",
        "random_walk_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/regression_slope_oscillator_kernel.cu",
        "regression_slope_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/reversal_signals_kernel.cu",
        "reversal_signals_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/relative_strength_index_wave_indicator_kernel.cu",
        "relative_strength_index_wave_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/monotonicity_index_kernel.cu",
        "monotonicity_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/market_meanness_index_kernel.cu",
        "market_meanness_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/macd_wave_signal_pro_kernel.cu",
        "macd_wave_signal_pro_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/market_structure_trailing_stop_kernel.cu",
        "market_structure_trailing_stop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_average_cross_probability_kernel.cu",
        "moving_average_cross_probability_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/multi_length_stochastic_average_kernel.cu",
        "multi_length_stochastic_average_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/absolute_strength_index_oscillator_kernel.cu",
        "absolute_strength_index_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/accumulation_swing_index_kernel.cu",
        "accumulation_swing_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/autocorrelation_indicator_kernel.cu",
        "autocorrelation_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_bounds_rsi_kernel.cu",
        "adaptive_bounds_rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_schaff_trend_cycle_kernel.cu",
        "adaptive_schaff_trend_cycle_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adjustable_ma_alternating_extremities_kernel.cu",
        "adjustable_ma_alternating_extremities_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_momentum_oscillator_kernel.cu",
        "adaptive_momentum_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/cyberpunk_value_trend_analyzer_kernel.cu",
        "cyberpunk_value_trend_analyzer_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/candle_strength_oscillator_kernel.cu",
        "candle_strength_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/hema_trend_levels_kernel.cu",
        "hema_trend_levels_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/daily_factor_kernel.cu",
        "daily_factor_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/demand_index_kernel.cu",
        "demand_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/decisionpoint_breadth_swenlin_trading_oscillator_kernel.cu",
        "decisionpoint_breadth_swenlin_trading_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/directional_imbalance_index_kernel.cu",
        "directional_imbalance_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_fm_demodulator_kernel.cu",
        "ehlers_fm_demodulator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_autocorrelation_periodogram_kernel.cu",
        "ehlers_autocorrelation_periodogram_kernel.ptx",
    );

    // ------------------------------------------------------ closer 6, round 3
    // The six from-scratch f64 kernels. Each is listed in `F64_LANE_SOURCES`
    // above, so each is compiled `-prec-div=true -prec-sqrt=true -fmad=false
    // -ftz=false` and never with `--use_fast_math`.
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/elastic_volume_weighted_moving_average_kernel.cu",
        "elastic_volume_weighted_moving_average_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/ema_deviation_corrected_t3_kernel.cu",
        "ema_deviation_corrected_t3_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/logarithmic_moving_average_kernel.cu",
        "logarithmic_moving_average_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/n_order_ema_kernel.cu",
        "n_order_ema_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/volatility_adjusted_ma_kernel.cu",
        "volatility_adjusted_ma_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/moving_averages/wave_smoother_kernel.cu",
        "wave_smoother_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fractal_dimension_index_kernel.cu",
        "fractal_dimension_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fvg_positioning_average_kernel.cu",
        "fvg_positioning_average_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_energy_reservoirs_kernel.cu",
        "volume_energy_reservoirs_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fibonacci_entry_bands_kernel.cu",
        "fibonacci_entry_bands_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/fibonacci_trailing_stop_kernel.cu",
        "fibonacci_trailing_stop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/disparity_index_kernel.cu",
        "disparity_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/dynamic_momentum_index_kernel.cu",
        "dynamic_momentum_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/forward_backward_exponential_oscillator_kernel.cu",
        "forward_backward_exponential_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_simple_cycle_indicator_kernel.cu",
        "ehlers_simple_cycle_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/evasive_supertrend_kernel.cu",
        "evasive_supertrend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/l1_ehlers_phasor_kernel.cu",
        "l1_ehlers_phasor_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/l2_ehlers_signal_to_noise_kernel.cu",
        "l2_ehlers_signal_to_noise_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/leavitt_convolution_acceleration_kernel.cu",
        "leavitt_convolution_acceleration_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/linear_correlation_oscillator_kernel.cu",
        "linear_correlation_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_adaptive_cg_kernel.cu",
        "ehlers_adaptive_cg_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_detrending_filter_kernel.cu",
        "ehlers_detrending_filter_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_data_sampling_relative_strength_indicator_kernel.cu",
        "ehlers_data_sampling_relative_strength_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_linear_extrapolation_predictor_kernel.cu",
        "ehlers_linear_extrapolation_predictor_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_smoothed_adaptive_momentum_kernel.cu",
        "ehlers_smoothed_adaptive_momentum_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ehlers_adaptive_cyber_cycle_kernel.cu",
        "ehlers_adaptive_cyber_cycle_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_bandpass_trigger_oscillator_kernel.cu",
        "adaptive_bandpass_trigger_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/adaptive_macd_kernel.cu",
        "adaptive_macd_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/andean_oscillator_kernel.cu",
        "andean_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/exponential_trend_kernel.cu",
        "exponential_trend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/didi_index_kernel.cu",
        "didi_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/geometric_bias_oscillator_kernel.cu",
        "geometric_bias_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/gmma_oscillator_kernel.cu",
        "gmma_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/hypertrend_kernel.cu",
        "hypertrend_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/hull_butterfly_oscillator_kernel.cu",
        "hull_butterfly_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/intraday_momentum_index_kernel.cu",
        "intraday_momentum_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/kairi_relative_index_kernel.cu",
        "kairi_relative_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/keltner_channel_width_oscillator_kernel.cu",
        "keltner_channel_width_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/impulse_macd_kernel.cu",
        "impulse_macd_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/linear_regression_intensity_kernel.cu",
        "linear_regression_intensity_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/nonlinear_regression_zero_lag_moving_average_kernel.cu",
        "nonlinear_regression_zero_lag_moving_average_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/neighboring_trailing_stop_kernel.cu",
        "neighboring_trailing_stop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/polynomial_regression_extrapolation_kernel.cu",
        "polynomial_regression_extrapolation_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_oscillator_kernel.cu",
        "range_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rolling_skewness_kurtosis_kernel.cu",
        "rolling_skewness_kurtosis_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/price_density_market_noise_kernel.cu",
        "price_density_market_noise_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/premier_rsi_oscillator_kernel.cu",
        "premier_rsi_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/spearman_correlation_kernel.cu",
        "spearman_correlation_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/squeeze_index_kernel.cu",
        "squeeze_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/statistical_trailing_stop_kernel.cu",
        "statistical_trailing_stop_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/projection_oscillator_kernel.cu",
        "projection_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/qqe_weighted_oscillator_kernel.cu",
        "qqe_weighted_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_breakout_signals_kernel.cu",
        "range_breakout_signals_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/range_filtered_trend_signals_kernel.cu",
        "range_filtered_trend_signals_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_distance_kernel.cu",
        "stochastic_distance_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_adaptive_d_kernel.cu",
        "stochastic_adaptive_d_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_connors_rsi_kernel.cu",
        "stochastic_connors_rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/stochastic_money_flow_index_kernel.cu",
        "stochastic_money_flow_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/smoothed_gaussian_trend_filter_kernel.cu",
        "smoothed_gaussian_trend_filter_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_continuation_factor_kernel.cu",
        "trend_continuation_factor_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_follower_kernel.cu",
        "trend_follower_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_direction_force_index_kernel.cu",
        "trend_direction_force_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_flow_trail_kernel.cu",
        "trend_flow_trail_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/trend_trigger_factor_kernel.cu",
        "trend_trigger_factor_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/twiggs_money_flow_kernel.cu",
        "twiggs_money_flow_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_zone_oscillator_kernel.cu",
        "volume_zone_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_weighted_rsi_kernel.cu",
        "volume_weighted_rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_weighted_relative_strength_index_kernel.cu",
        "volume_weighted_relative_strength_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/price_moving_average_ratio_percentile_kernel.cu",
        "price_moving_average_ratio_percentile_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/mesa_stochastic_multi_length_kernel.cu",
        "mesa_stochastic_multi_length_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/normalized_volume_true_range_kernel.cu",
        "normalized_volume_true_range_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/normalized_resonator_kernel.cu",
        "normalized_resonator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volatility_quality_index_kernel.cu",
        "volatility_quality_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/goertzel_cycle_composite_wave_kernel.cu",
        "goertzel_cycle_composite_wave_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ict_propulsion_block_kernel.cu",
        "ict_propulsion_block_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/ichimoku_oscillator_kernel.cu",
        "ichimoku_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/insync_index_kernel.cu",
        "insync_index_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/kase_peak_oscillator_with_divergences_kernel.cu",
        "kase_peak_oscillator_with_divergences_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/market_structure_confluence_kernel.cu",
        "market_structure_confluence_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/possible_rsi_kernel.cu",
        "possible_rsi_kernel.ptx",
    );
    // ------------------------------------------------------------- closer 4
    // `rogers_satchell_volatility_kernel.cu` was listed in `F64_LANE_SOURCES`
    // (so it opts out of fast math) but had NO `compile_kernel` call, so it
    // produced no PTX and no fatbin and `load_cuda_embedded_module!` could
    // never find it. Added here rather than left for a launch to discover.
    compile_kernel(
        &cuda_path,
        "kernels/cuda/rogers_satchell_volatility_kernel.cu",
        "rogers_satchell_volatility_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/smooth_theil_sen_kernel.cu",
        "smooth_theil_sen_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vdubus_divergence_wave_pattern_generator_kernel.cu",
        "vdubus_divergence_wave_pattern_generator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volatility_ratio_adaptive_rsx_kernel.cu",
        "volatility_ratio_adaptive_rsx_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/velocity_acceleration_convergence_divergence_indicator_kernel.cu",
        "velocity_acceleration_convergence_divergence_indicator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vwap_deviation_oscillator_kernel.cu",
        "vwap_deviation_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/vwap_zscore_with_signals_kernel.cu",
        "vwap_zscore_with_signals_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/volume_weighted_stochastic_rsi_kernel.cu",
        "volume_weighted_stochastic_rsi_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/supertrend_recovery_kernel.cu",
        "supertrend_recovery_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/supertrend_oscillator_kernel.cu",
        "supertrend_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/standardized_psar_oscillator_kernel.cu",
        "standardized_psar_oscillator_kernel.ptx",
    );
    compile_kernel(
        &cuda_path,
        "kernels/cuda/zig_zag_channels_kernel.cu",
        "zig_zag_channels_kernel.ptx",
    );

    compile_kernel(&cuda_path, "kernels/cuda/lpc_kernel.cu", "lpc_kernel.ptx");
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
        "alma_kernel.ptx",
    );
}

fn compile_cwma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/cwma_kernel.cu",
        "cwma_kernel.ptx",
    );
}

fn compile_cora_wave_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/cora_wave_kernel.cu",
        "cora_wave_kernel.ptx",
    );
}

fn compile_epma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/epma_kernel.cu",
        "epma_kernel.ptx",
    );
}

fn compile_ehlers_ecema_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/ehlers_ecema_kernel.cu",
        "ehlers_ecema_kernel.ptx",
    );
}

fn compile_kama_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/kama_kernel.cu",
        "kama_kernel.ptx",
    );
}

fn compile_highpass_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/highpass_kernel.cu",
        "highpass_kernel.ptx",
    );
}

fn compile_nama_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/nama_kernel.cu",
        "nama_kernel.ptx",
    );
}

fn compile_wma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/wma_kernel.cu",
        "wma_kernel.ptx",
    );
}

fn compile_sinwma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/sinwma_kernel.cu",
        "sinwma_kernel.ptx",
    );
}

fn compile_tradjema_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/tradjema_kernel.cu",
        "tradjema_kernel.ptx",
    );
}

fn compile_volume_adjusted_ma_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/volume_adjusted_ma_kernel.cu",
        "volume_adjusted_ma_kernel.ptx",
    );
}

fn compile_supersmoother_3_pole_kernel(cuda_path: &str) {
    compile_kernel(
        cuda_path,
        "kernels/cuda/moving_averages/supersmoother_3_pole_kernel.cu",
        "supersmoother_3_pole_kernel.ptx",
    );
}

fn compile_wto_kernel(cuda_path: &str) {
    compile_kernel(cuda_path, "kernels/cuda/wto_kernel.cu", "wto_kernel.ptx");
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

fn compile_kernel(cuda_path: &str, rel_src: &str, ptx_name: &str) {
    use std::process::Command;

    let src_path = if let Ok(root) = env::var("CUDA_KERNEL_DIR") {
        let root = root.trim_end_matches(['/', '\\']);
        let prefix = "kernels/cuda/";
        if rel_src.starts_with(prefix) {
            format!("{}/{}", root, &rel_src[prefix.len()..])
        } else {
            rel_src.to_string()
        }
    } else {
        rel_src.to_string()
    };

    println!("cargo:rerun-if-changed={}", src_path);

    let cubin_name = fatbin_name_for_ptx(ptx_name);

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let ptx_path = out_dir.join(ptx_name);
    let cubin_path = out_dir.join(&cubin_name);

    if let Ok(filt) = env::var("CUDA_FILTER") {
        let mut any = false;
        for tok in filt.split(|c: char| c == ',' || c.is_ascii_whitespace()) {
            let t = tok.trim();
            if !t.is_empty() && rel_src.contains(t) {
                any = true;
                break;
            }
        }
        if !any {
            eprintln!("Skipping {} due to CUDA_FILTER", rel_src);

            let placeholder = ".version 7.0
.target compute_80
.address_size 64
// placeholder PTX (no kernels)
";
            std::fs::write(&ptx_path, placeholder).expect("write placeholder PTX");
            std::fs::write(&cubin_path, []).expect("write placeholder cubin");
            return;
        }
    }

    if cfg!(target_os = "windows") && env::var("VCINSTALLDIR").is_err() {
        eprintln!(
            "Warning: VCINSTALLDIR not set. CUDA compilation may require running inside a Visual Studio Developer Command Prompt."
        );
    }

    let nvcc = if let Ok(nvcc_env) = env::var("NVCC") {
        nvcc_env
    } else if cfg!(target_os = "windows") {
        format!("{}/bin/nvcc.exe", cuda_path)
    } else {
        format!("{}/bin/nvcc", cuda_path)
    };

    // ── NeoEthos patch 2026-08-09 — THE ARCH TRAP, CLOSED ─────────────────
    //
    // WHAT WAS WRONG
    //
    // Upstream compiled ONE architecture per build and defaulted it to
    // `compute_89` (Ada / RTX 4090). PTX and SASS are both FORWARD-compatible
    // only, so an sm_89 module does not run on an sm_86 (RTX 3090) or an
    // sm_80 (A100): it fails to load, per indicator, at first use, deep inside
    // a run and far from the build that caused it. An earlier patch replaced
    // the hardcoded default with auto-detection of the BUILD HOST's card —
    // which removed the silent degradation but still emitted a single-arch
    // artifact, so moving the same binary between a 3090 and an A100 still
    // required a rebuild with a different `CUDA_ARCH`.
    //
    // WHAT IT DOES NOW
    //
    // One `-fatbin` per kernel source, carrying:
    //   * `-gencode arch=compute_X,code=sm_X` — real SASS for every X in
    //     `target_archs()`, so each of those cards runs precompiled code with
    //     no JIT at all; and
    //   * `-gencode arch=compute_MAX,code=compute_MAX` — embedded PTX at the
    //     highest architecture, so a card NEWER than anything we compiled for
    //     JITs forward instead of failing.
    // plus a standalone `<stem>.ptx` at the LOWEST architecture, used by
    // `module_loader.rs` only if the fatbin itself cannot be loaded. Lowest,
    // not highest, because a fallback is worthless if it is narrower than the
    // artifact it is backing up.
    //
    // HOW THE FOUR REQUIRED CARDS ARE SATISFIED, from ONE build with NO source
    // change and NO rebuild flag change:
    //   A100  sm_80 → SASS from `-gencode arch=compute_80,code=sm_80`
    //   3090  sm_86 → SASS from `-gencode arch=compute_86,code=sm_86`
    //   4090  sm_89 → SASS from `-gencode arch=compute_89,code=sm_89`
    //   H100  sm_90 → SASS from `-gencode arch=compute_90,code=sm_90`
    //   newer       → driver JITs the embedded compute_90 PTX
    //
    // `CUDA_ARCHS` still overrides the set (accepts a comma/space list now,
    // not just a first entry), and `CUDA_ARCH` still names a single one — both
    // for operators who want a faster, narrower build. Neither is required,
    // and neither silently narrows: whatever is compiled is recorded in
    // `VECTOR_TA_CUDA_ARCHS` and quoted back by the runtime error in
    // `module_loader.rs` when a device is not covered.
    //
    // The set is INTERSECTED with what this nvcc actually supports
    // (`nvcc --list-gpu-arch`), so CUDA 11 (no sm_90) and CUDA 13 (no sm_80)
    // both produce a working narrower fatbin instead of failing the build on
    // an "unsupported gpu architecture" for one entry of the list.
    let archs = target_archs(&nvcc);
    let arch_min = *archs.first().expect("target_archs is never empty");
    let arch_max = *archs.last().expect("target_archs is never empty");
    let ptx_arch = format!("compute_{arch_min}");

    // NON-NEGOTIABLE: the f64 lane never sees `--use_fast_math`, whatever
    // `CUDA_FAST_MATH` says. See `F64_LANE_SOURCES`.
    let f64_lane = is_f64_lane_source(rel_src);
    // OFF unless `CUDA_FAST_MATH=1` is spelled out — see `fast_math_requested`.
    // This arm read `Ok("0") => {} _ => --use_fast_math`, so an UNSET variable
    // (every normal build) compiled the f32 indicator kernels with FMA
    // contraction, flush-to-zero denormals and approximate div/rcp/sqrt. Those
    // kernels produce the indicator columns the fused Prototype B walk
    // multiplies by the gene weights, and the GPU parity fixtures supply that
    // matrix directly — so they would have stayed green while a real run
    // computed different numbers.
    let fast_math = fast_math_requested(rel_src);
    let apply_precision_flags = move |cmd: &mut Command| {
        if f64_lane {
            cmd.args([
                "-prec-div=true",
                "-prec-sqrt=true",
                "-fmad=false",
                "-ftz=false",
            ]);
            return;
        }
        if fast_math {
            cmd.arg("--use_fast_math");
        }
    };

    // ---- 1. standalone PTX at the LOWEST target arch (loader fallback) ----

    let mut cmd = Command::new(&nvcc);

    cmd.args([
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-ptx",
        "-O3",
    ]);

    apply_precision_flags(&mut cmd);

    if env::var("CUDA_DEBUG").ok().as_deref() == Some("1") {
        cmd.arg("-lineinfo");
    }

    cmd.args([
        "-arch",
        ptx_arch.as_str(),
        "-o",
        ptx_path.to_str().expect("ptx path"),
        src_path.as_str(),
    ]);

    if let Ok(extra) = env::var("NVCC_ARGS") {
        for tok in extra.split_whitespace() {
            if !tok.is_empty() {
                cmd.arg(tok);
            }
        }
    }

    if cfg!(target_os = "windows") {
        append_windows_nvcc_host_args(&mut cmd);
    }

    eprintln!("Running nvcc command: {:?}", cmd);

    let output = cmd.output().expect("Failed to execute nvcc");

    if !output.status.success() {
        eprintln!("CUDA compilation failed for {rel_src}!");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        if cfg!(target_os = "windows")
            && String::from_utf8_lossy(&output.stderr).contains("Cannot find compiler 'cl.exe'")
        {
            eprintln!("\n=== CUDA Build Error: Missing Visual Studio C++ Compiler ===");
            eprintln!("nvcc requires the Microsoft Visual C++ compiler (cl.exe) to be available.");
            eprintln!(
                "Install Visual Studio Build Tools 2022 or run cargo from a Developer Command Prompt."
            );
            eprintln!("===========================================================\n");
        }

        panic!("nvcc PTX compilation failed for {rel_src} at -arch {ptx_arch}");
    }

    println!(
        "Successfully compiled {} to {} (PTX fallback, {})",
        src_path,
        ptx_path.display(),
        ptx_arch
    );

    // ---- 2. multi-arch fatbin: SASS for every target + forward PTX --------

    let mut fat_cmd = Command::new(&nvcc);
    fat_cmd.args([
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-fatbin",
        "-O3",
    ]);

    apply_precision_flags(&mut fat_cmd);

    if env::var("CUDA_DEBUG").ok().as_deref() == Some("1") {
        fat_cmd.arg("-lineinfo");
    }

    for arch in archs {
        fat_cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
    // Forward compatibility: keep PTX for the newest architecture we know
    // about inside the same container, so an unreleased card JITs rather than
    // failing to load.
    fat_cmd.arg(format!(
        "-gencode=arch=compute_{arch_max},code=compute_{arch_max}"
    ));

    fat_cmd.args([
        "-o",
        cubin_path.to_str().expect("fatbin path"),
        src_path.as_str(),
    ]);

    if let Ok(extra) = env::var("NVCC_ARGS") {
        for tok in extra.split_whitespace() {
            if !tok.is_empty() {
                fat_cmd.arg(tok);
            }
        }
    }

    if cfg!(target_os = "windows") {
        append_windows_nvcc_host_args(&mut fat_cmd);
    }

    eprintln!("Running nvcc command: {:?}", fat_cmd);

    let fat_output = fat_cmd.output().expect("Failed to execute nvcc (fatbin)");

    if !fat_output.status.success() {
        eprintln!("CUDA fatbin compilation failed for {rel_src}!");
        eprintln!("stdout: {}", String::from_utf8_lossy(&fat_output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&fat_output.stderr));
        panic!(
            "nvcc fatbin compilation failed for {rel_src} (archs {archs:?}). \
             Refusing to emit a single-architecture artifact instead: that is the arch trap this \
             build exists to close. Narrow the set explicitly with CUDA_ARCHS if this toolkit \
             cannot serve all of them."
        );
    }

    println!(
        "Successfully compiled {} to {} (fatbin, sm {:?} + compute_{} PTX)",
        src_path,
        cubin_path.display(),
        archs,
        arch_max
    );

    if let Ok(prebuild_dir) = env::var("VECTOR_TA_PREBUILD_PTX_DIR") {
        let prebuild_dir = PathBuf::from(prebuild_dir);
        std::fs::create_dir_all(&prebuild_dir).expect("create VECTOR_TA_PREBUILD_PTX_DIR");
        let dst = prebuild_dir.join(ptx_name);
        std::fs::copy(&ptx_path, &dst).unwrap_or_else(|e| {
            panic!(
                "Failed copying compiled PTX {} -> {}: {e}",
                ptx_path.display(),
                dst.display()
            )
        });
    }

    if let Ok(prebuild_dir) = env::var("VECTOR_TA_PREBUILD_CUBIN_DIR") {
        let prebuild_dir = PathBuf::from(prebuild_dir);
        std::fs::create_dir_all(&prebuild_dir).expect("create VECTOR_TA_PREBUILD_CUBIN_DIR");
        let dst = prebuild_dir.join(&cubin_name);
        std::fs::copy(&cubin_path, &dst).unwrap_or_else(|e| {
            panic!(
                "Failed copying compiled cubin {} -> {}: {e}",
                cubin_path.display(),
                dst.display()
            )
        });
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

#[cfg(not(target_os = "windows"))]
fn find_vs_installation() -> Result<String, ()> {
    Err(())
}
