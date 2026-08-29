use crate::native_sass::{
    CuobjdumpReport, NativeArchInputs, NativeArchPlan, NativeSassError, VerifiedNativeCubin,
    native_cubin_filename, plan_native_architectures, verify_native_cubin,
};
use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};
use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KernelJob {
    pub(crate) rel_src: String,
    pub(crate) cubin_stem: String,
    pub(crate) source_path: PathBuf,
    pub(crate) source_bytes: u64,
    pub(crate) measured_tail_priority: u8,
}

impl KernelJob {
    pub(crate) fn new(
        rel_src: String,
        cubin_stem: String,
        source_path: PathBuf,
        source_bytes: u64,
        measured_tail_priority: u8,
    ) -> Self {
        Self {
            rel_src,
            cubin_stem,
            source_path,
            source_bytes,
            measured_tail_priority,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ArtifactJob {
    pub(crate) rel_src: String,
    pub(crate) cubin_stem: String,
    pub(crate) source_path: PathBuf,
    pub(crate) output_path: PathBuf,
    pub(crate) arch: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum NativePrecision {
    #[default]
    Default,
    FastMath,
    StrictF64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NativeCompileOptions {
    pub(crate) precision: NativePrecision,
    pub(crate) debug_line_info: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BuildSupportError {
    DuplicateSourceOutput {
        source: String,
        stem: String,
    },
    DuplicateOutputStem {
        stem: String,
    },
    DuplicateArchitecture {
        arch: u32,
    },
    EmptyArchitectureSet,
    FreeFormNvccArgs {
        value: String,
    },
    ToolSpawn {
        tool: PathBuf,
        operation: &'static str,
        message: String,
    },
    ToolFailed {
        tool: PathBuf,
        operation: &'static str,
        code: Option<i32>,
        stdout: String,
        stderr: String,
    },
    InvalidToolOutput {
        tool: PathBuf,
        operation: &'static str,
        output: String,
    },
    ArtifactRead {
        path: PathBuf,
        message: String,
    },
    NativeSass(String),
}

impl fmt::Display for BuildSupportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSourceOutput { source, stem } => write!(
                formatter,
                "duplicate CUDA source/output declaration {source} -> {stem}"
            ),
            Self::DuplicateOutputStem { stem } => {
                write!(formatter, "CUDA output stem {stem:?} is not unique")
            }
            Self::DuplicateArchitecture { arch } => {
                write!(
                    formatter,
                    "duplicate exact native CUDA architecture sm_{arch}"
                )
            }
            Self::EmptyArchitectureSet => {
                formatter.write_str("cannot expand native CUDA jobs without an architecture")
            }
            Self::FreeFormNvccArgs { value } => write!(
                formatter,
                "NVCC_ARGS is unsupported because unreviewed flags can override native-SASS, precision, output, and scheduler contracts; received {value:?}"
            ),
            Self::ToolSpawn {
                tool,
                operation,
                message,
            } => write!(
                formatter,
                "failed to run {} {operation}: {message}",
                tool.display()
            ),
            Self::ToolFailed {
                tool,
                operation,
                code,
                stdout,
                stderr,
            } => write!(
                formatter,
                "{} {operation} failed with {code:?}; stdout={stdout:?}; stderr={stderr:?}",
                tool.display()
            ),
            Self::InvalidToolOutput {
                tool,
                operation,
                output,
            } => write!(
                formatter,
                "{} {operation} returned no usable exact architectures: {output:?}",
                tool.display()
            ),
            Self::ArtifactRead { path, message } => {
                write!(formatter, "failed to read {}: {message}", path.display())
            }
            Self::NativeSass(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for BuildSupportError {}

impl From<NativeSassError> for BuildSupportError {
    fn from(error: NativeSassError) -> Self {
        Self::NativeSass(error.to_string())
    }
}

pub(crate) fn validate_unique_kernel_jobs(jobs: &[KernelJob]) -> Result<(), BuildSupportError> {
    let mut source_outputs = HashSet::with_capacity(jobs.len());
    let mut output_stems = HashSet::with_capacity(jobs.len());
    for job in jobs {
        if !source_outputs.insert((job.rel_src.as_str(), job.cubin_stem.as_str())) {
            return Err(BuildSupportError::DuplicateSourceOutput {
                source: job.rel_src.clone(),
                stem: job.cubin_stem.clone(),
            });
        }
        if !output_stems.insert(job.cubin_stem.as_str()) {
            return Err(BuildSupportError::DuplicateOutputStem {
                stem: job.cubin_stem.clone(),
            });
        }
    }
    Ok(())
}

pub(crate) fn order_kernel_jobs_longest_first(jobs: &mut [KernelJob]) {
    jobs.sort_by_cached_key(|job| {
        (
            Reverse(job.measured_tail_priority),
            Reverse(job.source_bytes),
            job.rel_src.clone(),
        )
    });
}

pub(crate) fn expand_native_artifact_jobs(
    jobs: &[KernelJob],
    architectures: &[u32],
    output_directory: &Path,
) -> Result<Vec<ArtifactJob>, BuildSupportError> {
    if architectures.is_empty() {
        return Err(BuildSupportError::EmptyArchitectureSet);
    }
    let mut unique_architectures = HashSet::with_capacity(architectures.len());
    for &arch in architectures {
        if !unique_architectures.insert(arch) {
            return Err(BuildSupportError::DuplicateArchitecture { arch });
        }
    }
    let mut artifacts = Vec::with_capacity(jobs.len() * architectures.len());
    for job in jobs {
        for &arch in architectures {
            artifacts.push(ArtifactJob {
                rel_src: job.rel_src.clone(),
                cubin_stem: job.cubin_stem.clone(),
                source_path: job.source_path.clone(),
                output_path: output_directory.join(native_cubin_filename(&job.cubin_stem, arch)),
                arch,
            });
        }
    }
    Ok(artifacts)
}

pub(crate) fn native_nvcc_args(job: &ArtifactJob, options: NativeCompileOptions) -> Vec<OsString> {
    let mut args = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "--cubin",
        "-O3",
    ]
    .into_iter()
    .map(OsString::from)
    .collect::<Vec<_>>();
    match options.precision {
        NativePrecision::Default => {}
        NativePrecision::FastMath => args.push(OsString::from("--use_fast_math")),
        NativePrecision::StrictF64 => args.extend(
            [
                "-prec-div=true",
                "-prec-sqrt=true",
                "-fmad=false",
                "-ftz=false",
            ]
            .into_iter()
            .map(OsString::from),
        ),
    }
    if options.debug_line_info {
        args.push(OsString::from("-lineinfo"));
    }
    args.push(OsString::from(format!(
        "-gencode=arch=compute_{},code=sm_{}",
        job.arch, job.arch
    )));
    args.push(OsString::from("-o"));
    args.push(job.output_path.as_os_str().to_owned());
    args.push(job.source_path.as_os_str().to_owned());
    args
}

pub(crate) fn reject_free_form_nvcc_args(value: Option<&str>) -> Result<(), BuildSupportError> {
    match value {
        Some(value) if !value.trim().is_empty() => Err(BuildSupportError::FreeFormNvccArgs {
            value: value.to_owned(),
        }),
        Some(_) | None => Ok(()),
    }
}

fn run_tool(
    tool: &Path,
    operation: &'static str,
    arguments: &[&str],
) -> Result<std::process::Output, BuildSupportError> {
    let output = Command::new(tool)
        .args(arguments)
        .output()
        .map_err(|error| BuildSupportError::ToolSpawn {
            tool: tool.to_path_buf(),
            operation,
            message: error.to_string(),
        })?;
    if !output.status.success() {
        return Err(BuildSupportError::ToolFailed {
            tool: tool.to_path_buf(),
            operation,
            code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(output)
}

fn parse_architecture(value: &str) -> Option<u32> {
    let trimmed = value.trim();
    let unprefixed = trimmed
        .strip_prefix("sm_")
        .or_else(|| trimmed.strip_prefix("compute_"))
        .unwrap_or(trimmed);
    if let Some((major, minor)) = unprefixed.split_once('.') {
        let major = major.parse::<u32>().ok()?;
        let minor = minor.parse::<u32>().ok()?;
        return (minor <= 9).then(|| major * 10 + minor);
    }
    let digits = unprefixed
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    (digits.len() >= 2).then(|| digits.parse().ok()).flatten()
}

pub(crate) fn discover_native_architectures(
    nvcc: &Path,
    nvidia_smi: &Path,
    explicit_architectures: Option<&str>,
) -> Result<NativeArchPlan, BuildSupportError> {
    let supported_output = run_tool(nvcc, "--list-gpu-arch", &["--list-gpu-arch"])?;
    let supported_text = String::from_utf8_lossy(&supported_output.stdout);
    let mut supported = supported_text
        .lines()
        .filter_map(parse_architecture)
        .collect::<Vec<_>>();
    supported.sort_unstable();
    supported.dedup();
    if supported.is_empty() {
        return Err(BuildSupportError::InvalidToolOutput {
            tool: nvcc.to_path_buf(),
            operation: "--list-gpu-arch",
            output: supported_text.into_owned(),
        });
    }

    let detected = if explicit_architectures.is_some() {
        Vec::new()
    } else {
        let detected_output = run_tool(
            nvidia_smi,
            "--query-gpu=compute_cap",
            &["--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
        )?;
        let detected_text = String::from_utf8_lossy(&detected_output.stdout);
        let mut architectures = detected_text
            .lines()
            .filter_map(parse_architecture)
            .collect::<Vec<_>>();
        architectures.sort_unstable();
        architectures.dedup();
        if architectures.is_empty() {
            return Err(BuildSupportError::InvalidToolOutput {
                tool: nvidia_smi.to_path_buf(),
                operation: "--query-gpu=compute_cap",
                output: detected_text.into_owned(),
            });
        }
        architectures
    };

    plan_native_architectures(NativeArchInputs {
        explicit_archs: explicit_architectures,
        detected_archs: &detected,
        nvcc_supported_archs: &supported,
    })
    .map_err(BuildSupportError::from)
}

pub(crate) fn inspect_native_cubin(
    cuobjdump: &Path,
    artifact_path: &Path,
    arch: u32,
) -> Result<VerifiedNativeCubin, BuildSupportError> {
    let artifact_argument = artifact_path.to_string_lossy();
    let list_ptx = run_tool(
        cuobjdump,
        "--list-ptx",
        &["--list-ptx", artifact_argument.as_ref()],
    )?;
    let dump_sass = run_tool(
        cuobjdump,
        "--dump-sass",
        &["--dump-sass", artifact_argument.as_ref()],
    )?;
    let bytes = std::fs::read(artifact_path).map_err(|error| BuildSupportError::ArtifactRead {
        path: artifact_path.to_path_buf(),
        message: error.to_string(),
    })?;
    let list_stdout = String::from_utf8_lossy(&list_ptx.stdout);
    let list_stderr = String::from_utf8_lossy(&list_ptx.stderr);
    let sass_stdout = String::from_utf8_lossy(&dump_sass.stdout);
    let sass_stderr = String::from_utf8_lossy(&dump_sass.stderr);
    verify_native_cubin(
        arch,
        artifact_path,
        &bytes,
        CuobjdumpReport {
            list_ptx_succeeded: true,
            list_ptx_stdout: &list_stdout,
            list_ptx_stderr: &list_stderr,
            dump_sass_succeeded: true,
            dump_sass_stdout: &sass_stdout,
            dump_sass_stderr: &sass_stderr,
        },
    )
    .map_err(BuildSupportError::from)
}

pub(crate) trait ArtifactCompiler: Sync {
    fn command(&self, job: &ArtifactJob) -> Result<Command, String>;
    fn finish(&self, job: &ArtifactJob, output: Output) -> Result<(), String>;
}

pub(crate) trait ArtifactVerifier: Sync {
    fn verify(&self, job: &ArtifactJob) -> Result<(), String>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SchedulerPhase {
    NvccChild,
    NativeCubinVerification,
}

impl SchedulerPhase {
    fn name(self) -> &'static str {
        match self {
            Self::NvccChild => "nvcc_child",
            Self::NativeCubinVerification => "native_cubin_verification",
        }
    }

    fn maximum_field(self) -> &'static str {
        match self {
            Self::NvccChild => "observed_max_active_nvcc",
            Self::NativeCubinVerification => "observed_max_active_verification_jobs",
        }
    }

    fn span_field(self) -> &'static str {
        match self {
            Self::NvccChild => "observed_nvcc_span_seconds",
            Self::NativeCubinVerification => "observed_verification_span_seconds",
        }
    }
}

#[derive(Clone, Debug)]
struct InvocationTelemetry {
    rel_src: String,
    output_path: PathBuf,
    arch: u32,
    start_offset: Duration,
    duration: Duration,
    succeeded: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct SchedulerReport {
    phase: SchedulerPhase,
    pub(crate) configured_width: usize,
    pub(crate) observed_max_active: usize,
    pub(crate) started: usize,
    pub(crate) completed: usize,
    pub(crate) failed: usize,
    pub(crate) active_at_end: usize,
    invocations: Vec<InvocationTelemetry>,
}

impl SchedulerReport {
    fn json_string(value: &str) -> String {
        let mut escaped = String::with_capacity(value.len() + 2);
        escaped.push('"');
        for character in value.chars() {
            match character {
                '"' => escaped.push_str("\\\""),
                '\\' => escaped.push_str("\\\\"),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                character if character.is_control() => {
                    use std::fmt::Write as _;
                    write!(escaped, "\\u{:04x}", character as u32).unwrap();
                }
                character => escaped.push(character),
            }
        }
        escaped.push('"');
        escaped
    }

    pub(crate) fn stable_json(&self) -> String {
        let mut slowest = self.invocations.clone();
        slowest.sort_by_key(|invocation| Reverse(invocation.duration));
        slowest.truncate(20);
        let invocations = slowest
            .iter()
            .map(|invocation| {
                format!(
                    "{{\"kind\":{},\"source\":{},\"output\":{},\"arch\":{},\"start_offset_seconds\":{:.6},\"duration_seconds\":{:.6},\"succeeded\":{}}}",
                    Self::json_string(self.phase.name()),
                    Self::json_string(&invocation.rel_src),
                    Self::json_string(&invocation.output_path.display().to_string()),
                    invocation.arch,
                    invocation.start_offset.as_secs_f64(),
                    invocation.duration.as_secs_f64(),
                    invocation.succeeded,
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let span = self
            .invocations
            .iter()
            .map(|invocation| invocation.start_offset + invocation.duration)
            .max()
            .unwrap_or_default();
        format!(
            "{{\"schema\":\"neoethos.vector_ta.native_cuda_scheduler.v4\",\"phase\":{},\"configured_width\":{},\"{}\":{},\"active_at_summary\":{},\"started_invocations\":{},\"completed_invocations\":{},\"failed_invocations\":{},\"{}\":{:.6},\"slowest_invocations\":[{}]}}",
            Self::json_string(self.phase.name()),
            self.configured_width,
            self.phase.maximum_field(),
            self.observed_max_active,
            self.active_at_end,
            self.started,
            self.completed,
            self.failed,
            self.phase.span_field(),
            span.as_secs_f64(),
            invocations,
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SchedulerFailure {
    pub(crate) message: String,
    pub(crate) report: SchedulerReport,
}

impl fmt::Display for SchedulerFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SchedulerFailure {}

struct TelemetryState {
    phase: SchedulerPhase,
    epoch: Instant,
    invocations: Mutex<Vec<InvocationTelemetry>>,
    active: AtomicUsize,
    maximum: AtomicUsize,
}

impl TelemetryState {
    fn new(phase: SchedulerPhase) -> Self {
        Self {
            phase,
            epoch: Instant::now(),
            invocations: Mutex::new(Vec::new()),
            active: AtomicUsize::new(0),
            maximum: AtomicUsize::new(0),
        }
    }

    fn enter(&self) -> (Instant, Duration) {
        let started = Instant::now();
        let active = self.active.fetch_add(1, Ordering::AcqRel) + 1;
        self.maximum.fetch_max(active, Ordering::AcqRel);
        (started, started.duration_since(self.epoch))
    }

    fn leave(&self, job: &ArtifactJob, started: Instant, start_offset: Duration, succeeded: bool) {
        let previous = self.active.fetch_sub(1, Ordering::AcqRel);
        assert!(
            previous > 0,
            "native CUDA scheduler telemetry active count underflowed"
        );
        self.invocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(InvocationTelemetry {
                rel_src: job.rel_src.clone(),
                output_path: job.output_path.clone(),
                arch: job.arch,
                start_offset,
                duration: started.elapsed(),
                succeeded,
            });
    }

    fn report(&self, configured_width: usize) -> SchedulerReport {
        let invocations = self
            .invocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        SchedulerReport {
            phase: self.phase,
            configured_width,
            observed_max_active: self.maximum.load(Ordering::Acquire),
            started: invocations.len(),
            completed: invocations
                .iter()
                .filter(|invocation| invocation.succeeded)
                .count(),
            failed: invocations
                .iter()
                .filter(|invocation| !invocation.succeeded)
                .count(),
            active_at_end: self.active.load(Ordering::Acquire),
            invocations,
        }
    }
}

fn pop_job(queue: &Mutex<VecDeque<ArtifactJob>>) -> Option<ArtifactJob> {
    queue
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .pop_front()
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "native CUDA compiler worker panicked with a non-string payload".to_owned()
    }
}

fn run_nvcc_child<C: ArtifactCompiler>(
    compiler: &C,
    telemetry: &TelemetryState,
    job: &ArtifactJob,
) -> Result<(), String> {
    let mut command =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compiler.command(job)))
            .unwrap_or_else(|payload| Err(panic_message(payload)))?;
    let (started, start_offset) = telemetry.enter();
    let child = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| command.output()));
    let succeeded = matches!(&child, Ok(Ok(output)) if output.status.success());
    telemetry.leave(job, started, start_offset, succeeded);
    let output = match child {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            return Err(format!(
                "failed to spawn native NVCC child for {} sm_{}: {error}",
                job.rel_src, job.arch
            ));
        }
        Err(payload) => return Err(panic_message(payload)),
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compiler.finish(job, output)
    }))
    .unwrap_or_else(|payload| Err(panic_message(payload)))
}

fn run_verification<V: ArtifactVerifier>(
    verifier: &V,
    telemetry: &TelemetryState,
    job: &ArtifactJob,
) -> Result<(), String> {
    let (started, start_offset) = telemetry.enter();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| verifier.verify(job)))
        .unwrap_or_else(|payload| Err(panic_message(payload)));
    telemetry.leave(job, started, start_offset, result.is_ok());
    result
}

fn record_failure(cancelled: &AtomicBool, first_failure: &Mutex<Option<String>>, message: String) {
    let mut failure = first_failure
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if failure.is_none() {
        *failure = Some(message);
    }
    cancelled.store(true, Ordering::Release);
}

fn run_scheduled_artifact_jobs<F>(
    jobs: Vec<ArtifactJob>,
    requested_width: usize,
    cargo_jobserver: Option<&jobserver::Client>,
    phase: SchedulerPhase,
    execute: &F,
) -> Result<SchedulerReport, SchedulerFailure>
where
    F: Fn(&ArtifactJob, &TelemetryState) -> Result<(), String> + Sync,
{
    let configured_width = requested_width.max(1).min(jobs.len().max(1));
    let telemetry = Arc::new(TelemetryState::new(phase));
    if jobs.is_empty() {
        return Ok(telemetry.report(configured_width));
    }

    if cargo_jobserver.is_none() || configured_width == 1 {
        for job in &jobs {
            if let Err(message) = execute(job, &telemetry) {
                let report = telemetry.report(1);
                return Err(SchedulerFailure { message, report });
            }
        }
        return Ok(telemetry.report(1));
    }

    let cargo_jobserver = cargo_jobserver.expect("jobserver existence checked");
    let queue = Arc::new(Mutex::new(VecDeque::from(jobs)));
    let first = pop_job(&queue).expect("the non-empty native CUDA queue lost its first job");
    let cancelled = Arc::new(AtomicBool::new(false));
    let first_failure = Arc::new(Mutex::new(None));

    std::thread::scope(|scope| {
        for _ in 1..configured_width {
            let queue = Arc::clone(&queue);
            let cancelled = Arc::clone(&cancelled);
            let first_failure = Arc::clone(&first_failure);
            let telemetry = Arc::clone(&telemetry);
            let client = cargo_jobserver.clone();
            scope.spawn(move || {
                while !cancelled.load(Ordering::Acquire) {
                    let Some(job) = pop_job(&queue) else {
                        break;
                    };
                    let permit = match client.acquire() {
                        Ok(permit) => permit,
                        Err(error) => {
                            record_failure(
                                &cancelled,
                                &first_failure,
                                format!("failed to acquire a Cargo jobserver permit: {error}"),
                            );
                            break;
                        }
                    };
                    if cancelled.load(Ordering::Acquire) {
                        drop(permit);
                        break;
                    }
                    let result = execute(&job, &telemetry);
                    drop(permit);
                    if let Err(message) = result {
                        record_failure(&cancelled, &first_failure, message);
                        break;
                    }
                }
            });
        }

        let mut next = Some(first);
        while let Some(job) = next {
            if let Err(message) = execute(&job, &telemetry) {
                record_failure(&cancelled, &first_failure, message);
                break;
            }
            if cancelled.load(Ordering::Acquire) {
                break;
            }
            next = pop_job(&queue);
        }
    });

    let report = telemetry.report(configured_width);
    let failure = first_failure
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    match failure {
        Some(message) => Err(SchedulerFailure { message, report }),
        None => Ok(report),
    }
}

pub(crate) fn run_native_artifact_jobs<C: ArtifactCompiler>(
    jobs: Vec<ArtifactJob>,
    requested_width: usize,
    cargo_jobserver: Option<&jobserver::Client>,
    compiler: &C,
) -> Result<SchedulerReport, SchedulerFailure> {
    run_scheduled_artifact_jobs(
        jobs,
        requested_width,
        cargo_jobserver,
        SchedulerPhase::NvccChild,
        &|job, telemetry| run_nvcc_child(compiler, telemetry, job),
    )
}

pub(crate) fn run_native_artifact_verifications<V: ArtifactVerifier>(
    jobs: Vec<ArtifactJob>,
    requested_width: usize,
    cargo_jobserver: Option<&jobserver::Client>,
    verifier: &V,
) -> Result<SchedulerReport, SchedulerFailure> {
    run_scheduled_artifact_jobs(
        jobs,
        requested_width,
        cargo_jobserver,
        SchedulerPhase::NativeCubinVerification,
        &|job, telemetry| run_verification(verifier, telemetry, job),
    )
}
