//! Rust-only benchmark run preparation: snapshots, matrix, collation and the
//! preflight report.
//!
//! These commands replace the legacy Python helpers on the paid-run path. The
//! Python files remain in `scripts/gpu-bench/` as isolated legacy tooling and
//! are not invoked by the rented-GPU kit.

use anyhow::{Context, Result, bail};
use neoethos_search::gpu_native::snapshot_fixture::{
    SNAPSHOT_FIXTURE_SCHEMA_VERSION, SnapshotFixtureDto, SnapshotPopulationFixture,
    SnapshotSettingsDto,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const SMC_WIDTH: usize = 11;
const LEGACY_SHA: &str = "2be1408ee3986026fdbb2a5a74aaaf6ac67e5209";
const PASSES: [&str; 4] = [
    "clean_timing",
    "diagnostics",
    "nsight_systems",
    "nsight_compute",
];
const SNAPSHOT_TIMEFRAMES: [&str; 5] = ["H1", "M30", "M15", "M5", "M1"];

// ---------------------------------------------------------------------------
// bench-prepare
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PreparedSnapshot {
    pub snapshot: String,
    pub sha256: String,
    pub bars: usize,
    pub features: usize,
    pub population: usize,
    pub timeframe: String,
}

pub fn run_prepare(args: &[String]) -> Result<()> {
    let csv = PathBuf::from(flag(args, "--csv").context("bench-prepare requires --csv <file>")?);
    let out = PathBuf::from(flag(args, "--out").context("bench-prepare requires --out <file>")?);
    let timeframe = flag(args, "--timeframe")
        .context("bench-prepare requires --timeframe <H1|M30|M15|M5|M1>")?;
    let options = PrepareOptions {
        population: parse_usize(args, "--population", 4096)?,
        terms_per_gene: parse_usize(args, "--terms-per-gene", 4)?,
        stop_pips: parse_f64(args, "--stop-pips", 18.0)?,
        target_pips: parse_f64(args, "--target-pips", 36.0)?,
        max_hold_bars: parse_usize(args, "--max-hold-bars", 12)?,
        max_trades_per_day: parse_usize(args, "--max-trades-per-day", 20)?,
        pip_value: parse_f64(args, "--pip-value", 0.0001)?,
        spread_pips: parse_f64(args, "--spread-pips", 0.0)?,
        commission: parse_f64(args, "--commission", 0.0)?,
        pip_value_per_lot: parse_f64(args, "--pip-value-per-lot", 10.0)?,
    };
    let prepared = prepare_snapshot(&csv, &out, &timeframe, &options)?;
    println!("{}", serde_json::to_string_pretty(&prepared)?);
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrepareOptions {
    pub population: usize,
    pub terms_per_gene: usize,
    pub stop_pips: f64,
    pub target_pips: f64,
    pub max_hold_bars: usize,
    pub max_trades_per_day: usize,
    pub pip_value: f64,
    pub spread_pips: f64,
    pub commission: f64,
    pub pip_value_per_lot: f64,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            population: 4096,
            terms_per_gene: 4,
            stop_pips: 18.0,
            target_pips: 36.0,
            max_hold_bars: 12,
            max_trades_per_day: 20,
            pip_value: 0.0001,
            spread_pips: 0.0,
            commission: 0.0,
            pip_value_per_lot: 10.0,
        }
    }
}

/// Convert canonical CSV into a deterministic, validated snapshot.
///
/// The same input and options always produce byte-identical JSON, so the
/// printed SHA-256 is a stable attribution key for a paid run.
pub fn prepare_snapshot(
    csv: &Path,
    out: &Path,
    timeframe: &str,
    options: &PrepareOptions,
) -> Result<PreparedSnapshot> {
    if options.population == 0 {
        bail!("--population must be positive");
    }
    let raw = std::fs::read(csv).with_context(|| format!("read {}", csv.display()))?;
    let series = parse_canonical_csv(&raw)?;
    let bars = series.close.len();
    if bars < 64 {
        bail!("snapshot CSV must contain at least 64 bars, found {bars}");
    }

    let feature_count = series.feature_names.len();
    let mut indicators = Vec::with_capacity(feature_count * bars);
    for feature in 0..feature_count {
        for row in &series.features {
            indicators.push(row[feature] as f32);
        }
    }

    let genes = deterministic_genes(options.population, feature_count, options.terms_per_gene);
    let months = series
        .timestamps
        .iter()
        .map(|value| month_id(*value))
        .collect();
    let days = series
        .timestamps
        .iter()
        .map(|value| value.div_euclid(86_400_000))
        .collect();
    let source_hash = hex_digest(&raw);
    let source_name = csv
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| csv.display().to_string());

    let dto = SnapshotFixtureDto {
        schema_version: SNAPSHOT_FIXTURE_SCHEMA_VERSION,
        timeframe: timeframe.to_ascii_uppercase(),
        source_description: format!("{source_name} sha256={source_hash}"),
        close: series.close,
        high: series.high,
        low: series.low,
        indicators,
        feature_count,
        gene_offsets: genes.offsets,
        gene_indices: genes.indices,
        gene_weights: genes.weights,
        long_thresholds: genes.long_thresholds,
        short_thresholds: genes.short_thresholds,
        months,
        days,
        timestamps: series.timestamps,
        stop_pips: vec![options.stop_pips; options.population],
        target_pips: vec![options.target_pips; options.population],
        stop_vol_multipliers: vec![0.0; options.population],
        smc_data: vec![[0_i8; SMC_WIDTH]; bars],
        gene_smc_flags: vec![[0_i8; SMC_WIDTH]; options.population],
        smc_weights: [0.0_f32; SMC_WIDTH],
        settings: SnapshotSettingsDto {
            max_hold_bars: options.max_hold_bars,
            min_hold_bars: 0,
            max_trades_per_day: options.max_trades_per_day,
            gap_threshold_ms: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 0.0,
            trailing_be_trigger_r: 0.0,
            pip_value: options.pip_value,
            spread_pips: options.spread_pips,
            commission_per_trade: options.commission,
            pip_value_per_lot: options.pip_value_per_lot,
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            pnl_conversion_fee_rate: 0.0,
            risk_based_sizing: true,
            risk_per_trade_min: 0.005,
            risk_per_trade_max: 0.01,
            high_quality_confidence: 0.65,
            adaptive_base_pips: None,
            adaptive_rr: 2.0,
        },
    };

    // Validate through the same fixture the benchmark consumes, so an invalid
    // snapshot fails here rather than on rented hardware.
    let fixture = SnapshotPopulationFixture::from_dto(dto.clone())
        .map_err(|error| anyhow::anyhow!("snapshot validation failed: {error}"))?;

    let encoded = serde_json::to_vec(&dto).context("encode snapshot")?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, &encoded).with_context(|| format!("write {}", out.display()))?;

    Ok(PreparedSnapshot {
        snapshot: out.display().to_string(),
        sha256: hex_digest(&encoded),
        bars: fixture.bars(),
        features: fixture.features(),
        population: fixture.population(),
        timeframe: timeframe.to_ascii_uppercase(),
    })
}

struct CanonicalSeries {
    timestamps: Vec<i64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    feature_names: Vec<String>,
    features: Vec<Vec<f64>>,
}

fn parse_canonical_csv(raw: &[u8]) -> Result<CanonicalSeries> {
    let text = String::from_utf8_lossy(raw);
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines.next().context("CSV has no header")?;
    let columns: Vec<String> = header
        .split(',')
        .map(|name| name.trim().to_string())
        .collect();
    let index_of = |wanted: &str| {
        columns
            .iter()
            .position(|name| name.eq_ignore_ascii_case(wanted))
    };
    let timestamp_index = index_of("timestamp").context("CSV is missing a timestamp column")?;
    let high_index = index_of("high").context("CSV is missing a high column")?;
    let low_index = index_of("low").context("CSV is missing a low column")?;
    let close_index = index_of("close").context("CSV is missing a close column")?;
    let reserved = [timestamp_index, high_index, low_index, close_index];
    let feature_indices: Vec<usize> = (0..columns.len())
        .filter(|index| !reserved.contains(index))
        .collect();
    if feature_indices.is_empty() {
        bail!("CSV must contain at least one numeric feature column");
    }

    let mut series = CanonicalSeries {
        timestamps: Vec::new(),
        high: Vec::new(),
        low: Vec::new(),
        close: Vec::new(),
        feature_names: feature_indices
            .iter()
            .map(|index| columns[*index].clone())
            .collect(),
        features: Vec::new(),
    };
    for (offset, line) in lines.enumerate() {
        let row_number = offset + 2;
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        if cells.len() != columns.len() {
            bail!(
                "row {row_number} has {} cells, expected {}",
                cells.len(),
                columns.len()
            );
        }
        let timestamp = finite(cells[timestamp_index], "timestamp", row_number)?;
        let mut milliseconds = timestamp as i64;
        if milliseconds.abs() < 10_000_000_000 {
            milliseconds *= 1000;
        }
        let high = finite(cells[high_index], "high", row_number)?;
        let low = finite(cells[low_index], "low", row_number)?;
        if low > high {
            bail!("row {row_number}: low {low} exceeds high {high}");
        }
        if let Some(previous) = series.timestamps.last() {
            if milliseconds <= *previous {
                bail!("row {row_number}: timestamps must be strictly increasing");
            }
        }
        series.timestamps.push(milliseconds);
        series.high.push(high);
        series.low.push(low);
        series
            .close
            .push(finite(cells[close_index], "close", row_number)?);
        let mut features = Vec::with_capacity(feature_indices.len());
        for index in &feature_indices {
            features.push(finite(cells[*index], &columns[*index], row_number)?);
        }
        series.features.push(features);
    }
    Ok(series)
}

fn finite(value: &str, label: &str, row: usize) -> Result<f64> {
    let parsed: f64 = value
        .parse()
        .with_context(|| format!("row {row}: {label} is not numeric: {value:?}"))?;
    if !parsed.is_finite() {
        bail!("row {row}: {label} is not finite");
    }
    Ok(parsed)
}

struct DeterministicGenes {
    offsets: Vec<i32>,
    indices: Vec<i32>,
    weights: Vec<f32>,
    long_thresholds: Vec<f32>,
    short_thresholds: Vec<f32>,
}

fn deterministic_genes(
    population: usize,
    feature_count: usize,
    terms_per_gene: usize,
) -> DeterministicGenes {
    let terms = terms_per_gene.clamp(1, feature_count.max(1));
    let mut genes = DeterministicGenes {
        offsets: vec![0],
        indices: Vec::with_capacity(population * terms),
        weights: Vec::with_capacity(population * terms),
        long_thresholds: Vec::with_capacity(population),
        short_thresholds: Vec::with_capacity(population),
    };
    for candidate in 0..population {
        for term in 0..terms {
            genes
                .indices
                .push(((candidate + term * 3) % feature_count) as i32);
            let magnitude = 0.35 + ((candidate + term) % 5) as f32 * 0.11;
            genes.weights.push(if (candidate + term) % 2 == 0 {
                magnitude
            } else {
                -magnitude
            });
        }
        genes.offsets.push(genes.indices.len() as i32);
        let threshold = 0.20 + (candidate % 3) as f32 * 0.03;
        genes.long_thresholds.push(threshold);
        genes.short_thresholds.push(-threshold);
    }
    genes
}

/// Months since year zero, matching the canonical month bucketing.
fn month_id(timestamp_ms: i64) -> i64 {
    let days = timestamp_ms.div_euclid(86_400_000);
    let (year, month) = civil_from_days(days);
    year * 12 + month as i64 - 1
}

/// Howard Hinnant's `civil_from_days`, so the month bucket needs no date crate
/// and matches the UTC calendar exactly.
fn civil_from_days(days: i64) -> (i64, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 }.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32)
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// bench-matrix
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixJob {
    pub job_id: String,
    pub ref_name: String,
    pub git_sha: String,
    pub worktree: String,
    pub timeframe: String,
    pub prototype: String,
    pub benchmark_pass: String,
    pub fixture: String,
    pub snapshot: Option<String>,
    pub dataset_hash: String,
    pub config_hash: String,
    pub executable: bool,
    pub blocked_reason: Option<String>,
    /// Cargo feature the runner binary must have been built with. A binary
    /// without it must fail loud rather than silently measure something else.
    pub required_feature: Option<String>,
    pub output: String,
    pub command: Vec<String>,
    pub environment: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixManifest {
    pub schema_version: u32,
    pub candidate_sha: String,
    pub legacy_sha: String,
    pub total_jobs: usize,
    pub executable_jobs: usize,
    pub blocked_jobs: usize,
    pub jobs: Vec<MatrixJob>,
}

pub fn run_matrix(args: &[String]) -> Result<()> {
    let candidate_sha =
        flag(args, "--candidate-sha").context("bench-matrix requires --candidate-sha <sha>")?;
    let legacy_sha = flag(args, "--legacy-sha").unwrap_or_else(|| LEGACY_SHA.to_string());
    let out = PathBuf::from(
        flag(args, "--out").unwrap_or_else(|| "cache/gpu-bench/matrix.json".to_string()),
    );
    let request = MatrixRequest {
        candidate_sha,
        legacy_sha,
        worktrees_root: PathBuf::from(
            flag(args, "--worktrees-root")
                .unwrap_or_else(|| "cache/gpu-bench/worktrees".to_string()),
        ),
        fixture: flag(args, "--fixture").unwrap_or_else(|| "tiny".to_string()),
        snapshot_dir: flag(args, "--snapshot-dir").map(PathBuf::from),
        runs_root: PathBuf::from(
            flag(args, "--runs-root").unwrap_or_else(|| "cache/gpu-bench/runs".to_string()),
        ),
        timeframes: flag(args, "--timeframes")
            .map(|raw| {
                raw.split(',')
                    .map(|value| value.trim().to_ascii_uppercase())
                    .filter(|value| !value.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                SNAPSHOT_TIMEFRAMES
                    .iter()
                    .map(|tf| tf.to_string())
                    .collect()
            }),
        prototypes: flag(args, "--prototypes")
            .unwrap_or_else(|| "a,b,c".to_string())
            .split(',')
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .collect(),
        population: parse_usize(args, "--population", 256)?,
        batch_size: parse_usize(args, "--batch-size", 256)?,
        bars: parse_usize(args, "--bars", 4096)?,
        features: parse_usize(args, "--features", 32)?,
        warmups: parse_usize(args, "--warmups", 2)?,
        repetitions: parse_usize(args, "--repetitions", 7)?,
    };
    let manifest = build_matrix(&request)?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, serde_json::to_vec_pretty(&manifest)?)?;
    println!(
        "wrote {}: total={} executable={} blocked={}",
        out.display(),
        manifest.total_jobs,
        manifest.executable_jobs,
        manifest.blocked_jobs
    );
    for job in &manifest.jobs {
        match &job.blocked_reason {
            Some(reason) => println!("BLOCKED {}: {reason}", job.job_id),
            None => println!("{}", job.command.join(" ")),
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixRequest {
    pub candidate_sha: String,
    pub legacy_sha: String,
    pub worktrees_root: PathBuf,
    pub fixture: String,
    pub snapshot_dir: Option<PathBuf>,
    pub runs_root: PathBuf,
    pub timeframes: Vec<String>,
    pub prototypes: Vec<String>,
    pub population: usize,
    pub batch_size: usize,
    pub bars: usize,
    pub features: usize,
    pub warmups: usize,
    pub repetitions: usize,
}

/// Build the deterministic job matrix.
///
/// Job order is fixed (ref, timeframe, prototype, pass) so two runs of the same
/// request produce byte-identical manifests.
pub fn build_matrix(request: &MatrixRequest) -> Result<MatrixManifest> {
    let snapshot_fixture = match request.fixture.as_str() {
        "tiny" => false,
        "snapshot" => true,
        other => bail!("unknown --fixture `{other}`"),
    };
    for prototype in &request.prototypes {
        if !matches!(prototype.as_str(), "a" | "b" | "c") {
            bail!("unknown --prototypes entry `{prototype}`");
        }
    }
    let config_hash = stable_hash(&serde_json::json!({
        "population": request.population,
        "bars": request.bars,
        "features": request.features,
        "batch_size": request.batch_size,
        "fixture": request.fixture,
    }));

    let timeframes: Vec<(String, Option<PathBuf>)> = if snapshot_fixture {
        let directory = request
            .snapshot_dir
            .as_ref()
            .context("--fixture snapshot requires --snapshot-dir <directory>")?;
        request
            .timeframes
            .iter()
            .map(|timeframe| {
                let path = directory.join(format!("{timeframe}.json"));
                if !path.is_file() {
                    bail!("missing {timeframe} snapshot: {}", path.display());
                }
                Ok((timeframe.clone(), Some(path)))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![("TINY".to_string(), None)]
    };

    let candidate_worktree = request.worktrees_root.join("candidate");
    let legacy_worktree = request.worktrees_root.join("legacy");
    let mut jobs = Vec::new();
    for (ref_name, git_sha, worktree) in [
        ("legacy", &request.legacy_sha, &legacy_worktree),
        ("candidate", &request.candidate_sha, &candidate_worktree),
    ] {
        for (timeframe, snapshot) in &timeframes {
            let dataset_hash = match snapshot {
                Some(path) => {
                    let bytes = std::fs::read(path)
                        .with_context(|| format!("read snapshot {}", path.display()))?;
                    hex_digest(&bytes)
                }
                None => stable_hash(&serde_json::json!({"fixture": "tiny-population-v1"})),
            };
            let prototypes: Vec<String> = if ref_name == "legacy" {
                vec!["legacy".to_string()]
            } else {
                request.prototypes.clone()
            };
            for prototype in prototypes {
                for benchmark_pass in PASSES {
                    let output = request
                        .runs_root
                        .join(ref_name)
                        .join(timeframe)
                        .join(&prototype)
                        .join(format!("{benchmark_pass}.json"));
                    let mut blocked_reason = None;
                    let mut command = Vec::new();
                    let mut environment = BTreeMap::new();
                    let mut required_feature = None;

                    if ref_name == "legacy" {
                        blocked_reason = Some(
                            "historical commit predates the attributed bench adapter; a pinned \
                             legacy execution adapter is required"
                                .to_string(),
                        );
                    } else {
                        required_feature = match prototype.as_str() {
                            "b" => Some("gpu-nvidia".to_string()),
                            _ => None,
                        };
                        let binary = candidate_worktree.join("target").join("release").join(
                            if cfg!(windows) {
                                "neoethos-cli.exe"
                            } else {
                                "neoethos-cli"
                            },
                        );
                        let mut bench_args: Vec<String> = if snapshot.is_some() {
                            vec![
                                "--execute-snapshot".into(),
                                "--fixture".into(),
                                "snapshot".into(),
                                "--snapshot".into(),
                                snapshot.as_ref().expect("checked").display().to_string(),
                            ]
                        } else {
                            vec![
                                "--execute-tiny".into(),
                                "--fixture".into(),
                                "tiny".into(),
                                "--population".into(),
                                request.population.to_string(),
                                "--bars".into(),
                                request.bars.to_string(),
                                "--features".into(),
                                request.features.to_string(),
                            ]
                        };
                        bench_args.extend([
                            "--git-sha".into(),
                            git_sha.to_string(),
                            "--baseline-sha".into(),
                            request.legacy_sha.clone(),
                            "--dataset-hash".into(),
                            dataset_hash.clone(),
                            "--config-hash".into(),
                            config_hash.clone(),
                            "--timeframe".into(),
                            timeframe.clone(),
                            "--backend".into(),
                            "gpu_required".into(),
                            "--prototype".into(),
                            prototype.clone(),
                            "--pass".into(),
                            benchmark_pass.to_string(),
                            "--batch-size".into(),
                            request.batch_size.to_string(),
                            "--warmups".into(),
                            request.warmups.to_string(),
                            "--repetitions".into(),
                            request.repetitions.to_string(),
                            "--out".into(),
                            output.display().to_string(),
                        ]);
                        let (wrapped, env) = wrap_for_pass(
                            &binary.display().to_string(),
                            &bench_args,
                            benchmark_pass,
                            &output,
                        );
                        command = wrapped;
                        environment = env;
                    }

                    jobs.push(MatrixJob {
                        job_id: format!("{ref_name}/{timeframe}/{prototype}/{benchmark_pass}"),
                        ref_name: ref_name.to_string(),
                        git_sha: git_sha.to_string(),
                        worktree: worktree.display().to_string(),
                        timeframe: timeframe.clone(),
                        prototype: prototype.clone(),
                        benchmark_pass: benchmark_pass.to_string(),
                        fixture: request.fixture.clone(),
                        snapshot: snapshot.as_ref().map(|path| path.display().to_string()),
                        dataset_hash: dataset_hash.clone(),
                        config_hash: config_hash.clone(),
                        executable: blocked_reason.is_none(),
                        blocked_reason,
                        required_feature,
                        output: output.display().to_string(),
                        command,
                        environment,
                    });
                }
            }
        }
    }

    let executable_jobs = jobs.iter().filter(|job| job.executable).count();
    Ok(MatrixManifest {
        schema_version: 1,
        candidate_sha: request.candidate_sha.clone(),
        legacy_sha: request.legacy_sha.clone(),
        total_jobs: jobs.len(),
        executable_jobs,
        blocked_jobs: jobs.len() - executable_jobs,
        jobs,
    })
}

fn wrap_for_pass(
    binary: &str,
    bench_args: &[String],
    benchmark_pass: &str,
    output: &Path,
) -> (Vec<String>, BTreeMap<String, String>) {
    let mut environment = BTreeMap::new();
    let mut command = vec![binary.to_string(), "bench".to_string()];
    command.extend(bench_args.iter().cloned());
    match benchmark_pass {
        "diagnostics" => {
            environment.insert("NEOETHOS_GPU_TIMING".to_string(), "1".to_string());
        }
        "nsight_systems" => {
            let trace = output.with_extension("");
            let mut wrapped = vec![
                "nsys".to_string(),
                "profile".to_string(),
                "--force-overwrite=true".to_string(),
                "--trace=cuda,nvtx,osrt".to_string(),
                "--output".to_string(),
                trace.display().to_string(),
            ];
            wrapped.extend(command);
            command = wrapped;
        }
        "nsight_compute" => {
            let report = output.with_extension("");
            let mut wrapped = vec![
                "ncu".to_string(),
                "--target-processes".to_string(),
                "all".to_string(),
                "--set".to_string(),
                "full".to_string(),
                "--force-overwrite".to_string(),
                "--export".to_string(),
                report.display().to_string(),
            ];
            wrapped.extend(command);
            command = wrapped;
        }
        _ => {}
    }
    (command, environment)
}

fn stable_hash(value: &serde_json::Value) -> String {
    hex_digest(canonical_json(value).as_bytes())
}

/// Key-sorted, separator-free JSON so the hash does not depend on map order.
fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let entries: Vec<String> = map
                .iter()
                .collect::<std::collections::BTreeMap<_, _>>()
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).unwrap_or_default(),
                        canonical_json(value)
                    )
                })
                .collect();
            format!("{{{}}}", entries.join(","))
        }
        serde_json::Value::Array(items) => {
            let entries: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", entries.join(","))
        }
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// bench-collate
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollatedRow {
    pub report: String,
    pub git_ref: String,
    pub timeframe: String,
    pub prototype: String,
    pub benchmark_pass: String,
    pub engine_status: String,
    pub parity_matched: bool,
    pub coverage: Option<f64>,
    pub median_seconds: Option<f64>,
    pub p95_seconds: Option<f64>,
    pub candidates_per_second: Option<f64>,
    pub candidate_bars_per_second: Option<f64>,
    pub peak_vram_bytes: Option<u64>,
    pub h2d_bytes: Option<u64>,
    pub d2h_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollatedSummary {
    pub schema_version: u32,
    pub report_count: usize,
    pub parity_failures: usize,
    pub rows: Vec<CollatedRow>,
}

pub fn run_collate(args: &[String]) -> Result<()> {
    let reports = PathBuf::from(
        flag(args, "--reports").context("bench-collate requires --reports <directory>")?,
    );
    let out = PathBuf::from(
        flag(args, "--out").unwrap_or_else(|| "cache/gpu-bench/summary.json".to_string()),
    );
    let summary = collate_reports(&reports)?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, serde_json::to_vec_pretty(&summary)?)?;
    println!(
        "wrote {} with {} reports ({} parity failures)",
        out.display(),
        summary.report_count,
        summary.parity_failures
    );
    Ok(())
}

/// Collate every report under `root` in a stable path order.
///
/// A missing field stays `null`; nothing is inferred or averaged away.
pub fn collate_reports(root: &Path) -> Result<CollatedSummary> {
    let mut paths = Vec::new();
    collect_json(root, &mut paths)?;
    paths.sort();
    let mut rows = Vec::with_capacity(paths.len());
    let mut parity_failures = 0;
    for path in paths {
        let raw = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let value: serde_json::Value =
            serde_json::from_slice(&raw).with_context(|| format!("parse {}", path.display()))?;
        let identity = value.get("identity").cloned().unwrap_or_default();
        let coverage = value.get("coverage");
        let total = coverage
            .and_then(|value| value.get("total_candidates"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let supported = coverage
            .and_then(|value| value.get("supported_candidates"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let parity_matched = value
            .get("parity")
            .and_then(|value| value.get("matched"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if !parity_matched {
            parity_failures += 1;
        }
        rows.push(CollatedRow {
            report: path.display().to_string(),
            git_ref: path
                .components()
                .rev()
                .nth(3)
                .map(|component| component.as_os_str().to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_string()),
            timeframe: string_field(&identity, "timeframe"),
            prototype: string_field(&identity, "prototype"),
            benchmark_pass: string_field(&identity, "pass"),
            engine_status: string_field(&value, "engine_status"),
            parity_matched,
            coverage: (total > 0).then(|| supported as f64 / total as f64),
            median_seconds: number_field(&value, "total_wall_seconds", "median"),
            p95_seconds: number_field(&value, "total_wall_seconds", "p95"),
            candidates_per_second: number_field(&value, "throughput", "candidates_per_second"),
            candidate_bars_per_second: number_field(
                &value,
                "throughput",
                "candidate_bars_per_second",
            ),
            peak_vram_bytes: unsigned_field(&value, "throughput", "peak_vram_bytes"),
            h2d_bytes: unsigned_field(&value, "transfers", "h2d_bytes"),
            d2h_bytes: unsigned_field(&value, "transfers", "d2h_bytes"),
        });
    }
    Ok(CollatedSummary {
        schema_version: 1,
        report_count: rows.len(),
        parity_failures,
        rows,
    })
}

fn collect_json(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !root.is_dir() {
        bail!("report directory not found: {}", root.display());
    }
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_json(&path, out)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("json")
            && path.file_name().and_then(|value| value.to_str()) != Some("matrix.json")
        {
            out.push(path);
        }
    }
    Ok(())
}

fn string_field(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .map(|value| match value {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string().trim_matches('"').to_string(),
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn number_field(value: &serde_json::Value, group: &str, key: &str) -> Option<f64> {
    value
        .get(group)
        .and_then(|group| group.get(key))
        .and_then(serde_json::Value::as_f64)
}

fn unsigned_field(value: &serde_json::Value, group: &str, key: &str) -> Option<u64> {
    value
        .get(group)
        .and_then(|group| group.get(key))
        .and_then(serde_json::Value::as_u64)
}

// ---------------------------------------------------------------------------
// bench-preflight-report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreflightReport {
    pub schema_version: u32,
    pub gpu_visible: bool,
    pub gpu: String,
    pub gpu_uuid: String,
    pub driver_version: String,
    pub cuda_toolkit_version: Option<String>,
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    pub disk_free_bytes: u64,
    pub power_limit_watts: f64,
    pub max_sm_clock_mhz: u64,
    pub temperature_celsius: i64,
    pub nsight_environment_checked: bool,
    pub cupti_present: bool,
    pub compute_sanitizer_present: bool,
    pub cuda_smoke_passed: bool,
    pub direct_cuda_parity_passed: bool,
    pub compute_sanitizer_passed: bool,
}

pub fn run_preflight_report(args: &[String]) -> Result<()> {
    let out =
        PathBuf::from(flag(args, "--out").context("bench-preflight-report requires --out <file>")?);
    let report = PreflightReport {
        schema_version: 1,
        gpu_visible: true,
        gpu: flag(args, "--gpu").context("bench-preflight-report requires --gpu")?,
        gpu_uuid: flag(args, "--gpu-uuid").context("bench-preflight-report requires --gpu-uuid")?,
        driver_version: flag(args, "--driver").unwrap_or_else(|| "unresolved".to_string()),
        cuda_toolkit_version: flag(args, "--cuda-toolkit").filter(|value| !value.is_empty()),
        vram_bytes: parse_u64(args, "--vram-mib", 0)? * 1024 * 1024,
        ram_bytes: parse_u64(args, "--ram-kib", 0)? * 1024,
        disk_free_bytes: parse_u64(args, "--disk-kib", 0)? * 1024,
        power_limit_watts: parse_f64(args, "--power-limit-watts", 0.0)?,
        max_sm_clock_mhz: parse_u64(args, "--max-sm-clock-mhz", 0)?,
        temperature_celsius: parse_u64(args, "--temperature-celsius", 0)? as i64,
        nsight_environment_checked: true,
        cupti_present: true,
        compute_sanitizer_present: true,
        cuda_smoke_passed: true,
        direct_cuda_parity_passed: true,
        compute_sanitizer_passed: true,
    };
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", out.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared argument helpers
// ---------------------------------------------------------------------------

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn parse_usize(args: &[String], name: &str, default: usize) -> Result<usize> {
    flag(args, name)
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_u64(args: &[String], name: &str, default: u64) -> Result<u64> {
    flag(args, name)
        .map(|value| {
            value
                .parse::<u64>()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_f64(args: &[String], name: &str, default: f64) -> Result<f64> {
    flag(args, name)
        .map(|value| {
            value
                .parse::<f64>()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_csv() -> String {
        let mut csv = String::from("timestamp,high,low,close,f0,f1\n");
        for bar in 0..96_i64 {
            let base = 1.10 + (bar as f64 * 0.01).sin() * 0.002;
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                1_700_000_000 + bar * 60,
                base + 0.0007,
                base - 0.0007,
                base,
                (bar as f64 * 0.05).sin(),
                (bar as f64 * 0.03).cos(),
            ));
        }
        csv
    }

    fn temp_dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("neoethos-bench-prepare-{name}"));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn snapshot_preparation_is_deterministic_and_validated() {
        let directory = temp_dir("deterministic");
        let csv = directory.join("EURUSD_M1.csv");
        std::fs::write(&csv, tiny_csv()).unwrap();
        let options = PrepareOptions {
            population: 8,
            ..PrepareOptions::default()
        };

        let first = prepare_snapshot(&csv, &directory.join("a.json"), "m1", &options).unwrap();
        let second = prepare_snapshot(&csv, &directory.join("b.json"), "m1", &options).unwrap();

        assert_eq!(
            first.sha256, second.sha256,
            "the same input must hash equal"
        );
        assert_eq!(first.timeframe, "M1");
        assert_eq!(first.bars, 96);
        assert_eq!(first.features, 2);
        assert_eq!(first.population, 8);
        assert_eq!(
            std::fs::read(directory.join("a.json")).unwrap(),
            std::fs::read(directory.join("b.json")).unwrap()
        );
    }

    #[test]
    fn malformed_csv_is_rejected_with_the_offending_row() {
        let directory = temp_dir("malformed");
        let csv = directory.join("bad.csv");
        let mut text = tiny_csv();
        text.push_str("1700010000,1.1,1.2,1.1,0.0,0.0\n");
        std::fs::write(&csv, text).unwrap();
        let error = prepare_snapshot(
            &csv,
            &directory.join("out.json"),
            "M1",
            &PrepareOptions {
                population: 4,
                ..PrepareOptions::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("low"), "{error}");
    }

    #[test]
    fn a_short_series_is_refused_rather_than_padded() {
        let directory = temp_dir("short");
        let csv = directory.join("short.csv");
        std::fs::write(
            &csv,
            "timestamp,high,low,close,f0\n1700000000,1.1,1.0,1.05,0.5\n",
        )
        .unwrap();
        let error = prepare_snapshot(
            &csv,
            &directory.join("out.json"),
            "M1",
            &PrepareOptions::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("at least 64 bars"), "{error}");
    }

    #[test]
    fn month_ids_follow_the_utc_calendar() {
        // 2023-11-14T22:13:20Z and 2023-12-01T00:00:00Z are different months.
        assert_ne!(month_id(1_700_000_000_000), month_id(1_701_388_800_000));
        assert_eq!(month_id(1_700_000_000_000) + 1, month_id(1_701_388_800_000));
    }

    #[test]
    fn matrix_order_is_deterministic_and_marks_b_and_c_executable() {
        let request = MatrixRequest {
            candidate_sha: "cafebabe".into(),
            legacy_sha: LEGACY_SHA.into(),
            worktrees_root: PathBuf::from("cache/gpu-bench/worktrees"),
            fixture: "tiny".into(),
            snapshot_dir: None,
            runs_root: PathBuf::from("cache/gpu-bench/runs"),
            timeframes: vec!["TINY".into()],
            prototypes: vec!["a".into(), "b".into(), "c".into()],
            population: 64,
            batch_size: 64,
            bars: 512,
            features: 8,
            warmups: 1,
            repetitions: 3,
        };
        let first = build_matrix(&request).unwrap();
        let second = build_matrix(&request).unwrap();
        assert_eq!(first, second, "matrix generation must be deterministic");

        assert_eq!(first.total_jobs, 4 + 3 * 4);
        assert_eq!(first.executable_jobs, 12);
        assert_eq!(first.blocked_jobs, 4, "legacy stays blocked");
        let b_jobs: Vec<&MatrixJob> = first
            .jobs
            .iter()
            .filter(|job| job.prototype == "b" && job.executable)
            .collect();
        assert_eq!(b_jobs.len(), 4);
        assert!(b_jobs.iter().all(|job| {
            job.required_feature.as_deref() == Some("gpu-nvidia")
                && job.command.iter().any(|arg| arg == "--prototype")
        }));
        assert!(
            first
                .jobs
                .iter()
                .filter(|job| job.ref_name == "legacy")
                .all(|job| job.blocked_reason.is_some() && job.command.is_empty())
        );
    }

    #[test]
    fn generated_commands_never_invoke_python() {
        let request = MatrixRequest {
            candidate_sha: "cafebabe".into(),
            legacy_sha: LEGACY_SHA.into(),
            worktrees_root: PathBuf::from("cache/gpu-bench/worktrees"),
            fixture: "tiny".into(),
            snapshot_dir: None,
            runs_root: PathBuf::from("cache/gpu-bench/runs"),
            timeframes: vec!["TINY".into()],
            prototypes: vec!["a".into(), "c".into()],
            population: 8,
            batch_size: 8,
            bars: 128,
            features: 4,
            warmups: 1,
            repetitions: 1,
        };
        let manifest = build_matrix(&request).unwrap();
        for job in &manifest.jobs {
            for argument in &job.command {
                let lowered = argument.to_ascii_lowercase();
                assert!(
                    !lowered.contains("python"),
                    "job {} invokes python: {argument}",
                    job.job_id
                );
            }
        }
    }

    #[test]
    fn an_unknown_prototype_is_refused() {
        let request = MatrixRequest {
            candidate_sha: "cafebabe".into(),
            legacy_sha: LEGACY_SHA.into(),
            worktrees_root: PathBuf::from("cache/gpu-bench/worktrees"),
            fixture: "tiny".into(),
            snapshot_dir: None,
            runs_root: PathBuf::from("cache/gpu-bench/runs"),
            timeframes: vec!["TINY".into()],
            prototypes: vec!["z".into()],
            population: 8,
            batch_size: 8,
            bars: 128,
            features: 4,
            warmups: 1,
            repetitions: 1,
        };
        assert!(build_matrix(&request).is_err());
    }

    #[test]
    fn collation_preserves_missing_fields_and_counts_parity_failures() {
        let directory = temp_dir("collate");
        let nested = directory.join("candidate").join("M1").join("c");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(
            nested.join("clean_timing.json"),
            serde_json::to_vec(&serde_json::json!({
                "identity": {"timeframe": "M1", "prototype": "c", "pass": "clean_timing"},
                "engine_status": "NotBenchmarked",
                "parity": {"matched": true},
                "coverage": {"total_candidates": 10, "supported_candidates": 8},
                "total_wall_seconds": {"median": 0.5},
                "throughput": {},
                "transfers": {"h2d_bytes": 1024}
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            nested.join("diagnostics.json"),
            serde_json::to_vec(&serde_json::json!({
                "identity": {"timeframe": "M1", "prototype": "c", "pass": "diagnostics"},
                "parity": {"matched": false}
            }))
            .unwrap(),
        )
        .unwrap();

        let summary = collate_reports(&directory).unwrap();
        assert_eq!(summary.report_count, 2);
        assert_eq!(summary.parity_failures, 1);
        let clean = &summary.rows[0];
        assert_eq!(clean.coverage, Some(0.8));
        assert_eq!(clean.median_seconds, Some(0.5));
        assert_eq!(clean.candidates_per_second, None, "missing stays missing");
        assert_eq!(clean.git_ref, "candidate");
    }
}
