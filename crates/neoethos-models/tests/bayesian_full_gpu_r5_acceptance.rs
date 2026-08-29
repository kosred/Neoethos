#[allow(dead_code)]
#[path = "support/bayesian_r5.rs"]
mod bayesian_r5;

use bayesian_r5::{
    EvidenceDimensions, GitIdentity, KernelActivity, OracleParityReceipt, TimingReceipt,
    TransferActivity, TransferDirection, fixture_cases, hash_f64_matrix, hash_f64_values,
    hash_labels, public_model_oracle_receipt, validate_cuda_evidence,
};
use ndarray::Array2;
use neoethos_core::execution::BudgetedCpuExecutor;
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_models::BayesianLogitExpert;
use neoethos_models::base::ExpertModel;
use neoethos_models::statistical::common::install_statistical_runtime_from_settings;
use rusqlite::Connection;
use rusqlite::types::ValueRef;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Output, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const TRAIN_ROWS: usize = 1_000_000;
const OOS_ROWS: usize = 131_071;
const FEATURE_WIDTHS: [usize; 2] = [64, 128];
const CPU_WORKERS: usize = 7;
const MINIMUM_CPU_WORK_NS_PER_WORKER: u64 = 1_000_000;
const TIMED_SAMPLES: usize = 3;
const REQUIRED_SPEEDUP: f64 = 1.25;
const ORACLE_PARITY_TOLERANCE: f64 = 5.0e-6;
const PRIOR_PRECISION: f64 = 0.05;
const LEARNING_RATE: f64 = 2.0;
const EPOCHS: usize = 64;
const READY_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(8 * 60);
const PARENT_WALL_CEILING: Duration = Duration::from_secs(30 * 60);
const CHILD_ROLE_ENV: &str = "NEOETHOS_BAYES_R5_ACCEPTANCE_ROLE";
const CHILD_FEATURES_ENV: &str = "NEOETHOS_BAYES_R5_ACCEPTANCE_FEATURES";
const CHILD_ARTIFACT_ENV: &str = "NEOETHOS_BAYES_R5_ACCEPTANCE_ARTIFACT";
const EVIDENCE_DIR_ENV: &str = "NEOETHOS_BAYES_R5_ACCEPTANCE_EVIDENCE_DIR";
const ACCEPTANCE_SENTINEL_ENV: &str = "NEOETHOS_BAYES_R5_ACCEPTANCE_SENTINEL";
const ACCEPTANCE_SENTINEL_VALUE: &str = "ONE_BOUNDED_RTX_RUN";
const PAID_ATTEMPT_CLAIM: &str = "bayesian-r5-paid-attempt.claim";
const READY_PREFIX: &str = "NEOETHOS_R5_READY ";
const RESPONSE_PREFIX: &str = "NEOETHOS_R5_RESPONSE ";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Cpu,
    Gpu,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu7",
            Self::Gpu => "gpu:0",
        }
    }

    fn worker_width(self) -> usize {
        match self {
            Self::Cpu => CPU_WORKERS,
            Self::Gpu => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    train_rows: usize,
    features: usize,
    oos_rows: usize,
}

impl Shape {
    fn exact(features: usize) -> Self {
        assert!(FEATURE_WIDTHS.contains(&features));
        Self {
            train_rows: TRAIN_ROWS,
            features,
            oos_rows: OOS_ROWS,
        }
    }

    fn label(self) -> String {
        format!("{}x{}-oos{}", self.train_rows, self.features, self.oos_rows)
    }
}

#[derive(Debug, Clone)]
struct FixtureIdentity {
    feature_sha256: String,
    label_sha256: String,
}

struct ExactFixture {
    frame: FeatureFrame,
    labels: Vec<i32>,
    identity: FixtureIdentity,
}

fn fixture_value(row: usize, feature: usize, phase_offset: usize) -> f64 {
    let class = (row + phase_offset) % 3;
    let primary = match (feature % 4, class) {
        (0, 0) | (1, 1) | (2, 2) => 1.60,
        (0, 2) | (1, 0) | (2, 1) => -1.10,
        _ => 0.25,
    };
    let bounded_noise = (((row * (feature + 11) + phase_offset) % 257) as f64 / 256.0 - 0.5) * 0.24;
    primary + bounded_noise
}

fn fixture_label(row: usize, phase_offset: usize) -> i32 {
    let class = (row + phase_offset) % 3;
    let class = if (row + phase_offset) % 97 == 0 {
        (class + 1) % 3
    } else {
        class
    };
    [-1, 0, 1][class]
}

fn exact_fixture(rows: usize, features: usize, phase_offset: usize) -> ExactFixture {
    assert!(rows > 0 && features > 0);
    let matrix = Array2::from_shape_fn((rows, features), |(row, feature)| {
        fixture_value(row, feature, phase_offset)
    });
    let columns = (0..features)
        .map(|feature| {
            FeatureColumnF64::new(
                format!("bayes_r5_f{feature}"),
                matrix.column(feature).to_vec(),
                vec![FeatureCellValidity::Valid; rows],
            )
            .expect("construct exact finite Bayesian acceptance feature")
        })
        .collect::<Vec<_>>();
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
    .expect("construct exact Bayesian acceptance FeatureFrame");
    let labels = (0..rows)
        .map(|row| fixture_label(row, phase_offset))
        .collect::<Vec<_>>();
    assert_eq!(frame.n_samples(), rows);
    assert_eq!(frame.n_features(), features);
    let identity = FixtureIdentity {
        feature_sha256: hash_f64_matrix(&matrix),
        label_sha256: hash_labels(&labels),
    };
    ExactFixture {
        frame,
        labels,
        identity,
    }
}

fn expected_fixture_identity(rows: usize, features: usize, phase_offset: usize) -> FixtureIdentity {
    let feature_sha256 = hash_f64_values(
        rows,
        features,
        (0..rows).flat_map(|row| {
            (0..features).map(move |feature| fixture_value(row, feature, phase_offset))
        }),
    );
    let labels = (0..rows)
        .map(|row| fixture_label(row, phase_offset))
        .collect::<Vec<_>>();
    FixtureIdentity {
        feature_sha256,
        label_sha256: hash_labels(&labels),
    }
}

fn configured_model() -> BayesianLogitExpert {
    let mut model = BayesianLogitExpert::new();
    model.prior_precision = PRIOR_PRECISION;
    model.learning_rate = LEARNING_RATE;
    model.epochs = EPOCHS;
    model
}

fn assert_probabilities(probabilities: &Array2<f64>, labels: &[i32]) {
    assert_eq!(probabilities.dim(), (labels.len(), 3));
    assert!(!probabilities.is_empty());
    let mut correct = 0usize;
    let mut log_loss = 0.0_f64;
    for (row, label) in probabilities.outer_iter().zip(labels) {
        assert!(row.iter().all(|value| value.is_finite()));
        assert!((row.sum() - 1.0).abs() <= 1e-10);
        let expected = match label {
            -1 => 2,
            0 => 0,
            1 => 1,
            unexpected => panic!("illegal acceptance label {unexpected}"),
        };
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .expect("three-class row")
            .0;
        correct += usize::from(predicted == expected);
        log_loss -= row[expected].max(f64::MIN_POSITIVE).ln();
    }
    let accuracy = correct as f64 / labels.len() as f64;
    let mean_log_loss = log_loss / labels.len() as f64;
    assert!(accuracy >= 0.80, "acceptance accuracy {accuracy:.6} < 0.80");
    assert!(
        mean_log_loss <= 0.65,
        "acceptance mean log loss {mean_log_loss:.6} > 0.65"
    );
}

fn assert_exact_bits(left: &Array2<f64>, right: &Array2<f64>, context: &str) {
    assert_eq!(left.dim(), right.dim(), "{context}: shape drift");
    assert!(!left.is_empty(), "{context}: empty probability matrix");
    for (index, (lhs, rhs)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "{context}: bit drift at index {index}"
        );
    }
}

fn probability_sha256(probabilities: &Array2<f64>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(probabilities.nrows().to_le_bytes());
    hasher.update(probabilities.ncols().to_le_bytes());
    for value in probabilities {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn file_sha256(path: &Path) -> String {
    let bytes = fs::read(path)
        .unwrap_or_else(|error| panic!("read persistent receipt {}: {error}", path.display()));
    format!("{:x}", Sha256::digest(bytes))
}

fn assert_before_deadline(parent_deadline: Instant, phase: &str) {
    assert!(
        Instant::now() < parent_deadline,
        "Bayesian R5 exhausted its single {:?} parent ceiling during {phase}",
        PARENT_WALL_CEILING
    );
}

fn bounded_output(command: &mut Command, parent_deadline: Instant, context: &str) -> Output {
    assert_before_deadline(parent_deadline, context);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .unwrap_or_else(|error| panic!("launch {context}: {error}"));
    let mut stdout = child
        .stdout
        .take()
        .expect("bounded command stdout is piped");
    let mut stderr = child
        .stderr
        .take()
        .expect("bounded command stderr is piped");
    let stdout_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .read_to_end(&mut bytes)
            .expect("read bounded command stdout");
        bytes
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr
            .read_to_end(&mut bytes)
            .expect("read bounded command stderr");
        bytes
    });
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .unwrap_or_else(|error| panic!("poll {context}: {error}"))
        {
            break status;
        }
        if Instant::now() >= parent_deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            panic!(
                "{context} exhausted the single {:?} Bayesian R5 parent ceiling",
                PARENT_WALL_CEILING
            );
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    Output {
        status,
        stdout: stdout_reader.join().expect("join bounded stdout reader"),
        stderr: stderr_reader.join().expect("join bounded stderr reader"),
    }
}

#[cfg(target_os = "windows")]
fn native_thread_id() -> u64 {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThreadId() -> u32;
    }
    // SAFETY: GetCurrentThreadId takes no arguments and is valid in every
    // Windows thread. The returned ID is owned by the operating system.
    unsafe { GetCurrentThreadId() as u64 }
}

#[cfg(target_os = "windows")]
fn native_thread_cpu_time_ns() -> u64 {
    use std::ffi::c_void;

    #[repr(C)]
    struct FileTime {
        low_date_time: u32,
        high_date_time: u32,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn GetThreadTimes(
            thread: *mut c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }

    let mut creation = FileTime {
        low_date_time: 0,
        high_date_time: 0,
    };
    let mut exit = FileTime {
        low_date_time: 0,
        high_date_time: 0,
    };
    let mut kernel = FileTime {
        low_date_time: 0,
        high_date_time: 0,
    };
    let mut user = FileTime {
        low_date_time: 0,
        high_date_time: 0,
    };
    // SAFETY: GetCurrentThread returns a process-local pseudo-handle valid for
    // GetThreadTimes, and all four writable FILETIME pointers live for the call.
    let success = unsafe {
        GetThreadTimes(
            GetCurrentThread(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    };
    assert_ne!(success, 0, "GetThreadTimes failed for a CPU7 worker");
    let ticks =
        |time: &FileTime| (u64::from(time.high_date_time) << 32) | u64::from(time.low_date_time);
    ticks(&kernel)
        .checked_add(ticks(&user))
        .and_then(|ticks| ticks.checked_mul(100))
        .expect("native Windows worker CPU time fits nanoseconds")
}

#[cfg(target_os = "linux")]
fn native_thread_id() -> u64 {
    // SAFETY: SYS_gettid takes no pointer arguments and returns the calling
    // Linux thread's kernel TID.
    unsafe { libc::syscall(libc::SYS_gettid) as u64 }
}

#[cfg(target_os = "linux")]
fn native_thread_cpu_time_ns() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: time is a valid writable timespec and CLOCK_THREAD_CPUTIME_ID
    // reads only the calling worker's process CPU clock.
    let code = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
    assert_eq!(code, 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
    u64::try_from(time.tv_sec)
        .expect("worker CPU seconds are non-negative")
        .checked_mul(1_000_000_000)
        .and_then(|value| {
            value.checked_add(u64::try_from(time.tv_nsec).expect("worker nanoseconds non-negative"))
        })
        .expect("native Linux worker CPU time fits nanoseconds")
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn native_thread_id() -> u64 {
    panic!("Bayesian R5 CPU7 native identity gate supports Windows and Linux only")
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn native_thread_cpu_time_ns() -> u64 {
    panic!("Bayesian R5 CPU7 CPU-time gate supports Windows and Linux only")
}

#[derive(Clone, Debug)]
struct CpuWorkerSnapshot {
    native_tid: u64,
    name: String,
    cpu_time_ns: u64,
}

fn capture_cpu7_bound_contexts(
    broker: &CpuPermitBroker,
    accepted_lease: &CpuLease,
) -> Vec<CpuWorkerSnapshot> {
    assert_eq!(accepted_lease.width().get(), CPU_WORKERS);
    assert_eq!(BudgetedCpuExecutor::current_pool_width(), CPU_WORKERS);
    let nested_width = WorkerLimit::new(1).expect("one nested worker is legal to request");
    let identities = rayon::broadcast(|_| {
        accepted_lease.scope(|| {
            assert_eq!(BudgetedCpuExecutor::current_pool_width(), CPU_WORKERS);
            broker
                .try_acquire(CpuPermitRequest::local(nested_width))
                .expect_err("every lease-bound worker must reject fresh nested acquisition");
            CpuWorkerSnapshot {
                native_tid: native_thread_id(),
                name: std::thread::current()
                    .name()
                    .expect("budgeted worker must have a stable native name")
                    .to_string(),
                cpu_time_ns: native_thread_cpu_time_ns(),
            }
        })
    });
    assert_eq!(identities.len(), CPU_WORKERS);
    let tids = identities
        .iter()
        .map(|snapshot| snapshot.native_tid)
        .collect::<BTreeSet<_>>();
    let names = identities
        .iter()
        .map(|snapshot| snapshot.name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        tids.len(),
        CPU_WORKERS,
        "lease-bearing CPU7 call reused a native OS TID"
    );
    assert_eq!(
        names.len(),
        CPU_WORKERS,
        "lease-bearing CPU7 call reused a worker name"
    );
    assert!(
        names.iter().all(|name| name.starts_with("neoethos-cpu-")),
        "lease-bearing CPU7 call escaped BudgetedCpuExecutor: {names:?}"
    );
    identities
}

fn cpu7_work_evidence(
    before: &[CpuWorkerSnapshot],
    after: &[CpuWorkerSnapshot],
    require_meaningful_work: bool,
) -> Vec<(u64, String, u64)> {
    assert_eq!(before.len(), CPU_WORKERS);
    assert_eq!(after.len(), CPU_WORKERS);
    let before_by_tid = before
        .iter()
        .map(|snapshot| (snapshot.native_tid, snapshot))
        .collect::<BTreeMap<_, _>>();
    let mut evidence = Vec::with_capacity(CPU_WORKERS);
    for snapshot in after {
        let prior = before_by_tid
            .get(&snapshot.native_tid)
            .unwrap_or_else(|| panic!("CPU7 worker {} changed native identity", snapshot.name));
        assert_eq!(prior.name, snapshot.name, "CPU7 worker name changed");
        let work_cpu_ns = snapshot
            .cpu_time_ns
            .checked_sub(prior.cpu_time_ns)
            .expect("native worker CPU time must be monotonic");
        if require_meaningful_work {
            assert!(
                work_cpu_ns >= MINIMUM_CPU_WORK_NS_PER_WORKER,
                "public fit+predict did not use CPU7 worker {} (tid {}): {} ns < {} ns",
                snapshot.name,
                snapshot.native_tid,
                work_cpu_ns,
                MINIMUM_CPU_WORK_NS_PER_WORKER
            );
        }
        evidence.push((snapshot.native_tid, snapshot.name.clone(), work_cpu_ns));
    }
    evidence.sort_by_key(|(native_tid, _, _)| *native_tid);
    evidence
}

struct WorkloadOutcome {
    probabilities: Array2<f64>,
    cpu7_identities: Vec<(u64, String, u64)>,
}

struct WorkerState {
    role: Role,
    shape: Shape,
    train: FeatureFrame,
    train_labels: Vec<i32>,
    oos: FeatureFrame,
    oos_labels: Vec<i32>,
    train_identity: FixtureIdentity,
    oos_identity: FixtureIdentity,
    artifact_dir: PathBuf,
    broker: CpuPermitBroker,
    executor: BudgetedCpuExecutor,
}

impl WorkerState {
    fn prepare(role: Role, shape: Shape, artifact_dir: PathBuf) -> Self {
        let mut settings = neoethos_core::Settings::default();
        settings.models.statistical_device = match role {
            Role::Cpu => "cpu".to_string(),
            Role::Gpu => "gpu:0".to_string(),
        };
        install_statistical_runtime_from_settings(&settings);

        let train_fixture = exact_fixture(shape.train_rows, shape.features, 0);
        let oos_fixture = exact_fixture(shape.oos_rows, shape.features, 1_000_003);
        let width = WorkerLimit::new(role.worker_width()).expect("positive worker width");
        let broker = CpuPermitBroker::new(width);
        let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), width);
        Self {
            role,
            shape,
            train: train_fixture.frame,
            train_labels: train_fixture.labels,
            oos: oos_fixture.frame,
            oos_labels: oos_fixture.labels,
            train_identity: train_fixture.identity,
            oos_identity: oos_fixture.identity,
            artifact_dir,
            broker,
            executor,
        }
    }

    fn with_role_lease<R, Work>(
        &self,
        require_meaningful_work: bool,
        work: Work,
    ) -> (R, Vec<(u64, String, u64)>)
    where
        R: Send,
        Work: FnOnce(&CpuLease) -> R + Send,
    {
        let width = WorkerLimit::new(self.role.worker_width()).expect("positive worker width");
        let lease = self
            .broker
            .acquire(CpuPermitRequest::local(width))
            .expect("acquire exact public-model lease");
        match self.role {
            Role::Cpu => self
                .executor
                .execute_with_lease(lease.into_transfer(), |accepted_lease| {
                    let before = capture_cpu7_bound_contexts(&self.broker, accepted_lease);
                    let result = work(accepted_lease);
                    let after = capture_cpu7_bound_contexts(&self.broker, accepted_lease);
                    let identities = cpu7_work_evidence(&before, &after, require_meaningful_work);
                    (result, identities)
                })
                .expect("execute public Bayesian work with the accepted CPU7 lease"),
            Role::Gpu => {
                assert_eq!(lease.width().get(), 1);
                (work(&lease), Vec::new())
            }
        }
    }

    fn workload(&self) -> WorkloadOutcome {
        let ((probabilities, model), cpu7_identities) =
            self.with_role_lease(true, |accepted_lease| {
                let mut model = configured_model();
                ExpertModel::fit(&mut model, &self.train, &self.train_labels, accepted_lease)
                    .expect("real public Bayesian timed fit failed");
                let probabilities = ExpertModel::predict_proba(&model, &self.oos, accepted_lease)
                    .expect("real public Bayesian timed OOS prediction failed");
                assert_probabilities(&probabilities, &self.oos_labels);
                (probabilities, model)
            });
        drop(model);
        WorkloadOutcome {
            probabilities,
            cpu7_identities,
        }
    }

    fn lifecycle(&self) -> WorkloadOutcome {
        let (probabilities, cpu7_identities) = self.with_role_lease(true, |accepted_lease| {
            let mut model = configured_model();
            ExpertModel::fit(&mut model, &self.train, &self.train_labels, accepted_lease)
                .expect("real public Bayesian lifecycle fit failed");
            let first = ExpertModel::predict_proba(&model, &self.oos, accepted_lease)
                .expect("real public Bayesian lifecycle prediction failed");
            let repeated = ExpertModel::predict_proba(&model, &self.oos, accepted_lease)
                .expect("real public Bayesian lifecycle repeat prediction failed");
            assert_probabilities(&first, &self.oos_labels);
            assert_exact_bits(&first, &repeated, "same-device OOS prediction repeat");
            ExpertModel::save(&model, &self.artifact_dir)
                .expect("save genuine public Bayesian artifact");
            let mut restored = configured_model();
            ExpertModel::load(&mut restored, &self.artifact_dir)
                .expect("load genuine public Bayesian artifact");
            let after_load = ExpertModel::predict_proba(&restored, &self.oos, accepted_lease)
                .expect("restored Bayesian lifecycle prediction failed");
            assert_exact_bits(&first, &after_load, "public Bayesian save/load");
            first
        });
        WorkloadOutcome {
            probabilities,
            cpu7_identities,
        }
    }

    fn oracle_gate(&self) -> (Vec<OracleParityReceipt>, Vec<(u64, String, u64)>) {
        self.with_role_lease(false, |accepted_lease| {
            fixture_cases()
                .iter()
                .map(|case| {
                    public_model_oracle_receipt(
                        case,
                        accepted_lease,
                        &self.artifact_dir.join("oracle").join(case.name),
                        ORACLE_PARITY_TOLERANCE,
                    )
                })
                .collect()
        })
    }
}

#[cfg(target_os = "windows")]
mod cuda_profiler_range {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    use std::path::PathBuf;

    type ProfilerFn = unsafe extern "C" fn() -> i32;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    pub struct Range {
        module: *mut c_void,
        stop: ProfilerFn,
        active: bool,
    }

    impl Range {
        pub fn start() -> Result<Self, String> {
            let mut candidates = Vec::<PathBuf>::new();
            if let Some(path) = std::env::var_os("PATH") {
                for directory in std::env::split_paths(&path) {
                    if let Ok(entries) = std::fs::read_dir(directory) {
                        for entry in entries.flatten() {
                            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                            if name.starts_with("cudart64_") && name.ends_with(".dll") {
                                candidates.push(entry.path());
                            }
                        }
                    }
                }
            }
            if let Some(cuda_path) = std::env::var_os("CUDA_PATH") {
                if let Ok(entries) = std::fs::read_dir(PathBuf::from(cuda_path).join("bin")) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                        if name.starts_with("cudart64_") && name.ends_with(".dll") {
                            candidates.push(entry.path());
                        }
                    }
                }
            }
            candidates.sort();
            candidates.dedup();
            for candidate in candidates {
                let wide = candidate
                    .as_os_str()
                    .encode_wide()
                    .chain(std::iter::once(0))
                    .collect::<Vec<_>>();
                // SAFETY: wide is NUL terminated and remains alive for the call.
                let module = unsafe { LoadLibraryW(wide.as_ptr()) };
                if module.is_null() {
                    continue;
                }
                // SAFETY: symbol names are NUL terminated; successful pointers
                // have the CUDA Runtime profiler function ABI.
                let start_ptr =
                    unsafe { GetProcAddress(module, c"cudaProfilerStart".as_ptr().cast()) };
                let stop_ptr =
                    unsafe { GetProcAddress(module, c"cudaProfilerStop".as_ptr().cast()) };
                if start_ptr.is_null() || stop_ptr.is_null() {
                    // SAFETY: module came from LoadLibraryW in this function.
                    unsafe { FreeLibrary(module) };
                    continue;
                }
                // SAFETY: GetProcAddress returned the named CUDA Runtime symbols.
                let start: ProfilerFn = unsafe { std::mem::transmute(start_ptr) };
                // SAFETY: GetProcAddress returned the named CUDA Runtime symbols.
                let stop: ProfilerFn = unsafe { std::mem::transmute(stop_ptr) };
                // SAFETY: CUDA Runtime profiler API takes no parameters.
                let code = unsafe { start() };
                if code != 0 {
                    // SAFETY: module came from LoadLibraryW in this function.
                    unsafe { FreeLibrary(module) };
                    return Err(format!("cudaProfilerStart returned CUDA error {code}"));
                }
                return Ok(Self {
                    module,
                    stop,
                    active: true,
                });
            }
            Err("could not load cudart64_*.dll from PATH or CUDA_PATH/bin".to_string())
        }

        pub fn finish(mut self) -> Result<(), String> {
            self.stop_active()
        }

        fn stop_active(&mut self) -> Result<(), String> {
            if !self.active {
                return Ok(());
            }
            // SAFETY: stop is the cudaProfilerStop symbol from the loaded module.
            let code = unsafe { (self.stop)() };
            self.active = false;
            if code == 0 {
                Ok(())
            } else {
                Err(format!("cudaProfilerStop returned CUDA error {code}"))
            }
        }
    }

    impl Drop for Range {
        fn drop(&mut self) {
            let _ = self.stop_active();
            // SAFETY: module came from LoadLibraryW and is released once here.
            unsafe { FreeLibrary(self.module) };
        }
    }
}

#[cfg(target_os = "linux")]
mod cuda_profiler_range {
    use std::ffi::{CStr, CString, c_void};

    type ProfilerFn = unsafe extern "C" fn() -> i32;

    pub struct Range {
        module: *mut c_void,
        stop: ProfilerFn,
        active: bool,
    }

    impl Range {
        pub fn start() -> Result<Self, String> {
            for candidate in ["libcudart.so", "libcudart.so.13", "libcudart.so.12"] {
                let candidate = CString::new(candidate).expect("static library name");
                // SAFETY: candidate is a valid NUL-terminated library name.
                let module = unsafe { libc::dlopen(candidate.as_ptr(), libc::RTLD_NOW) };
                if module.is_null() {
                    continue;
                }
                // SAFETY: static symbol names are NUL terminated.
                let start_ptr = unsafe { libc::dlsym(module, c"cudaProfilerStart".as_ptr()) };
                let stop_ptr = unsafe { libc::dlsym(module, c"cudaProfilerStop".as_ptr()) };
                if start_ptr.is_null() || stop_ptr.is_null() {
                    // SAFETY: module came from dlopen in this function.
                    unsafe { libc::dlclose(module) };
                    continue;
                }
                // SAFETY: dlsym returned the named CUDA Runtime symbols.
                let start: ProfilerFn = unsafe { std::mem::transmute(start_ptr) };
                // SAFETY: dlsym returned the named CUDA Runtime symbols.
                let stop: ProfilerFn = unsafe { std::mem::transmute(stop_ptr) };
                // SAFETY: CUDA Runtime profiler API takes no parameters.
                let code = unsafe { start() };
                if code != 0 {
                    // SAFETY: module came from dlopen in this function.
                    unsafe { libc::dlclose(module) };
                    return Err(format!("cudaProfilerStart returned CUDA error {code}"));
                }
                return Ok(Self {
                    module,
                    stop,
                    active: true,
                });
            }
            // SAFETY: dlerror returns either null or a NUL-terminated diagnostic.
            let detail = unsafe {
                let pointer = libc::dlerror();
                if pointer.is_null() {
                    "no loader diagnostic".to_string()
                } else {
                    CStr::from_ptr(pointer).to_string_lossy().into_owned()
                }
            };
            Err(format!("could not load libcudart.so: {detail}"))
        }

        pub fn finish(mut self) -> Result<(), String> {
            self.stop_active()
        }

        fn stop_active(&mut self) -> Result<(), String> {
            if !self.active {
                return Ok(());
            }
            // SAFETY: stop is the cudaProfilerStop symbol from the loaded module.
            let code = unsafe { (self.stop)() };
            self.active = false;
            if code == 0 {
                Ok(())
            } else {
                Err(format!("cudaProfilerStop returned CUDA error {code}"))
            }
        }
    }

    impl Drop for Range {
        fn drop(&mut self) {
            let _ = self.stop_active();
            // SAFETY: module came from dlopen and is released once here.
            unsafe { libc::dlclose(self.module) };
        }
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
mod cuda_profiler_range {
    pub struct Range;

    impl Range {
        pub fn start() -> Result<Self, String> {
            Err("CUDA profiler range supports Windows and Linux only".to_string())
        }

        pub fn finish(self) -> Result<(), String> {
            unreachable!("unsupported platforms cannot start a CUDA profiler range")
        }
    }
}

fn emit(prefix: &str, payload: Value) {
    println!("{prefix}{payload}");
    std::io::stdout().flush().expect("flush R5 worker protocol");
}

fn response_payload(
    id: u64,
    operation: &str,
    state: &WorkerState,
    digest: String,
    cpu7_identities: &[(u64, String, u64)],
) -> Value {
    json!({
        "id": id,
        "operation": operation,
        "ok": true,
        "role": state.role.as_str(),
        "train_rows": state.shape.train_rows,
        "features": state.shape.features,
        "oos_rows": state.shape.oos_rows,
        "probability_sha256": digest,
        "train_feature_sha256": state.train_identity.feature_sha256,
        "train_label_sha256": state.train_identity.label_sha256,
        "oos_feature_sha256": state.oos_identity.feature_sha256,
        "oos_label_sha256": state.oos_identity.label_sha256,
        "cpu7_identities": cpu7_identities.iter().map(|(tid, name, work_cpu_ns)| {
            json!({"native_tid": tid, "name": name, "work_cpu_ns": work_cpu_ns})
        }).collect::<Vec<_>>(),
    })
}

fn run_acceptance_worker() {
    let role = match std::env::var(CHILD_ROLE_ENV).as_deref() {
        Ok("cpu7") => Role::Cpu,
        Ok("gpu:0") => Role::Gpu,
        Ok(unexpected) => panic!("unknown acceptance child role `{unexpected}`"),
        Err(error) => panic!("acceptance child role is parent-owned: {error}"),
    };
    let features = std::env::var(CHILD_FEATURES_ENV)
        .expect("parent must provide feature width")
        .parse::<usize>()
        .expect("feature width must be numeric");
    let artifact_dir = PathBuf::from(
        std::env::var_os(CHILD_ARTIFACT_ENV).expect("parent must provide artifact directory"),
    );
    let state = WorkerState::prepare(role, Shape::exact(features), artifact_dir);
    emit(
        READY_PREFIX,
        json!({
            "role": role.as_str(),
            "train_rows": state.shape.train_rows,
            "features": state.shape.features,
            "oos_rows": state.shape.oos_rows,
        }),
    );

    for line in std::io::stdin().lock().lines() {
        let line = line.expect("read parent protocol command");
        let request: Value = serde_json::from_str(&line).expect("parse parent protocol command");
        let id = request["id"].as_u64().expect("command id must be u64");
        let operation = request["operation"]
            .as_str()
            .expect("command operation must be a string");
        match operation {
            "oracle_gate" => {
                let (receipts, cpu7_identities) = state.oracle_gate();
                let encoded = serde_json::to_vec(&receipts)
                    .expect("serialize independent Bayesian oracle receipts");
                let mut payload = response_payload(
                    id,
                    operation,
                    &state,
                    format!("{:x}", Sha256::digest(&encoded)),
                    &cpu7_identities,
                );
                payload["oracle_receipts"] = serde_json::to_value(receipts)
                    .expect("encode independent Bayesian oracle receipts");
                emit(RESPONSE_PREFIX, payload);
            }
            "lifecycle" => {
                let outcome = state.lifecycle();
                emit(
                    RESPONSE_PREFIX,
                    response_payload(
                        id,
                        operation,
                        &state,
                        probability_sha256(&outcome.probabilities),
                        &outcome.cpu7_identities,
                    ),
                );
            }
            "warmup" | "sample" => {
                let outcome = state.workload();
                emit(
                    RESPONSE_PREFIX,
                    response_payload(
                        id,
                        operation,
                        &state,
                        probability_sha256(&outcome.probabilities),
                        &outcome.cpu7_identities,
                    ),
                );
            }
            "capture_lifecycle" => {
                assert_eq!(role, Role::Gpu);
                let range = cuda_profiler_range::Range::start()
                    .expect("start parent-observed CUDA lifecycle capture range");
                let outcome = state.lifecycle();
                range
                    .finish()
                    .expect("finish parent-observed CUDA lifecycle capture range");
                emit(
                    RESPONSE_PREFIX,
                    response_payload(
                        id,
                        operation,
                        &state,
                        probability_sha256(&outcome.probabilities),
                        &outcome.cpu7_identities,
                    ),
                );
            }
            "quit" => {
                emit(
                    RESPONSE_PREFIX,
                    json!({"id": id, "operation": operation, "ok": true}),
                );
                break;
            }
            unexpected => panic!("unknown parent operation `{unexpected}`"),
        }
    }
}

struct WorkerProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    reader: Option<JoinHandle<()>>,
    next_id: u64,
    shape: Shape,
    role: Role,
    profile_base: Option<PathBuf>,
    parent_deadline: Instant,
}

impl WorkerProcess {
    fn spawn(
        role: Role,
        shape: Shape,
        artifact_dir: &Path,
        profile_base: Option<&Path>,
        parent_deadline: Instant,
    ) -> Self {
        let executable = std::env::current_exe().expect("locate acceptance test executable");
        let mut command = if let Some(base) = profile_base {
            let mut command = Command::new("nsys");
            command
                .arg("profile")
                .arg("--trace=cuda")
                .arg("--capture-range=cudaProfilerApi")
                .arg("--capture-range-end=stop")
                .arg("--force-overwrite=true")
                .arg("--output")
                .arg(base)
                .arg(&executable);
            command
        } else {
            Command::new(&executable)
        };
        command
            .args([
                "--exact",
                "r5_acceptance_child_worker",
                "--ignored",
                "--nocapture",
            ])
            .env(CHILD_ROLE_ENV, role.as_str())
            .env(CHILD_FEATURES_ENV, shape.features.to_string())
            .env(CHILD_ARTIFACT_ENV, artifact_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = command.spawn().unwrap_or_else(|error| {
            panic!(
                "spawn {} acceptance worker for {}: {error}",
                role.as_str(),
                shape.label()
            )
        });
        let stdin = child.stdin.take().expect("capture worker stdin");
        let stdout = child.stdout.take().expect("capture worker stdout");
        let (sender, lines) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        let mut worker = Self {
            child,
            stdin: Some(stdin),
            lines,
            reader: Some(reader),
            next_id: 1,
            shape,
            role,
            profile_base: profile_base.map(Path::to_path_buf),
            parent_deadline,
        };
        let ready = worker.receive_prefixed(READY_PREFIX, READY_TIMEOUT);
        worker.assert_shape_role(&ready);
        worker
    }

    fn assert_shape_role(&self, payload: &Value) {
        assert_eq!(payload["role"], self.role.as_str());
        assert_eq!(payload["train_rows"], self.shape.train_rows);
        assert_eq!(payload["features"], self.shape.features);
        assert_eq!(payload["oos_rows"], self.shape.oos_rows);
    }

    fn receive_prefixed(&mut self, prefix: &str, timeout: Duration) -> Value {
        let deadline = (Instant::now() + timeout).min(self.parent_deadline);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let _ = self.child.kill();
                let _ = self.child.wait();
                panic!(
                    "{} worker reached the bounded parent deadline for {}",
                    self.role.as_str(),
                    self.shape.label()
                );
            }
            match self.lines.recv_timeout(remaining) {
                Ok(line) => {
                    if let Some(payload) = line.strip_prefix(prefix) {
                        return serde_json::from_str(payload).unwrap_or_else(|error| {
                            panic!("parse worker response `{line}`: {error}")
                        });
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    panic!(
                        "{} worker timed out after {timeout:?} for {}",
                        self.role.as_str(),
                        self.shape.label()
                    );
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let status = self.child.wait().expect("wait disconnected worker");
                    panic!(
                        "{} worker exited before protocol response for {}: {status}",
                        self.role.as_str(),
                        self.shape.label()
                    );
                }
            }
        }
    }

    fn command(&mut self, operation: &str) -> (Duration, Value) {
        let id = self.next_id;
        self.next_id += 1;
        let request = json!({"id": id, "operation": operation});
        let encoded = serde_json::to_string(&request).expect("encode parent protocol command");
        let started = Instant::now();
        let stdin = self.stdin.as_mut().expect("worker stdin remains open");
        writeln!(stdin, "{encoded}").expect("write parent protocol command");
        stdin.flush().expect("flush parent protocol command");
        let payload = self.receive_prefixed(RESPONSE_PREFIX, COMMAND_TIMEOUT);
        let elapsed = started.elapsed();
        assert_eq!(payload["id"], id);
        assert_eq!(payload["operation"], operation);
        assert_eq!(payload["ok"], true);
        if operation != "quit" {
            self.assert_shape_role(&payload);
        }
        (elapsed, payload)
    }

    fn finish(mut self) -> Option<PathBuf> {
        let _ = self.command("quit");
        drop(self.stdin.take());
        let status = self.child.wait().expect("wait acceptance worker");
        assert!(
            status.success(),
            "{} worker failed for {}: {status}",
            self.role.as_str(),
            self.shape.label()
        );
        if let Some(reader) = self.reader.take() {
            reader.join().expect("join worker stdout reader");
        }
        self.profile_base.take()
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(
        &fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display())),
    )
    .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn assert_parent_owned_gpu_artifact(artifact_dir: &Path) -> Value {
    let model = read_json(&artifact_dir.join("model.json"));
    let metadata = read_json(&artifact_dir.join("metadata.json"));
    for document in [&model, &metadata] {
        assert_eq!(document["model_name"], "bayes_logit");
        assert_eq!(document["requested_device_policy"], "gpu:0");
        assert_eq!(document["effective_device_policy"], "gpu:0");
        assert_eq!(document["runtime_backend_kind"], "native_cuda");
        assert_eq!(document["runtime_degraded_reason"], Value::Null);
    }
    let backend = model["runtime_backend"]
        .as_str()
        .expect("GPU artifact must persist runtime_backend");
    assert!(backend.contains("cuda") && backend.contains("gpu:0"));
    assert!(!backend.contains("cpu") && !backend.contains("fallback"));
    json!({
        "model_sha256": file_sha256(&artifact_dir.join("model.json")),
        "metadata_sha256": file_sha256(&artifact_dir.join("metadata.json")),
        "runtime_backend": backend,
    })
}

fn sqlite_identifier(value: &str) -> &str {
    assert!(
        value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_'),
        "unexpected nsys SQLite identifier `{value}`"
    );
    value
}

fn sqlite_table_columns(connection: &Connection, table: &str) -> BTreeSet<String> {
    let table = sqlite_identifier(table);
    let query = format!("PRAGMA table_info(\"{table}\")");
    let mut statement = connection
        .prepare(&query)
        .unwrap_or_else(|error| panic!("prepare column inventory for {table}: {error}"));
    statement
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap_or_else(|error| panic!("query column inventory for {table}: {error}"))
        .collect::<Result<BTreeSet<_>, _>>()
        .unwrap_or_else(|error| panic!("collect column inventory for {table}: {error}"))
}

fn nsys_string_ids(connection: &Connection, tables: &[String]) -> BTreeMap<i64, String> {
    let Some(table) = tables
        .iter()
        .find(|table| table.eq_ignore_ascii_case("StringIds"))
    else {
        return BTreeMap::new();
    };
    let table = sqlite_identifier(table);
    let query = format!("SELECT id, value FROM \"{table}\"");
    let mut statement = connection
        .prepare(&query)
        .expect("prepare Nsight StringIds query");
    statement
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })
        .expect("query Nsight StringIds")
        .collect::<Result<BTreeMap<_, _>, _>>()
        .expect("collect Nsight StringIds")
}

fn activity_name(value: ValueRef<'_>, strings: &BTreeMap<i64, String>) -> String {
    match value {
        ValueRef::Integer(identifier) => strings
            .get(&identifier)
            .cloned()
            .unwrap_or_else(|| format!("unresolved-string-id-{identifier}")),
        ValueRef::Text(bytes) => String::from_utf8_lossy(bytes).into_owned(),
        other => panic!("unsupported Nsight kernel-name value {other:?}"),
    }
}

fn export_and_query_independent_cuda_ledger(
    profile_base: &Path,
    stage: &str,
    shape: Shape,
    runtime_backend: &str,
    parent_deadline: Instant,
) -> Value {
    let report = profile_base.with_extension("nsys-rep");
    assert!(report.is_file(), "nsys did not create {}", report.display());
    assert!(
        fs::metadata(&report).expect("stat nsys report").len() > 0,
        "nsys report is empty"
    );
    let sqlite = profile_base.with_extension("sqlite");
    let mut export_command = Command::new("nsys");
    export_command
        .arg("export")
        .arg("--type=sqlite")
        .arg("--force-overwrite=true")
        .arg("--output")
        .arg(&sqlite)
        .arg(&report);
    let export = bounded_output(
        &mut export_command,
        parent_deadline,
        "parent-owned nsys SQLite export",
    );
    let export_receipt_path = profile_base.with_extension("nsys-export.json");
    fs::write(
        &export_receipt_path,
        serde_json::to_vec_pretty(&json!({
            "command": ["nsys", "export", "--type=sqlite", "--force-overwrite=true"],
            "status_code": export.status.code(),
            "stdout": String::from_utf8_lossy(&export.stdout),
            "stderr": String::from_utf8_lossy(&export.stderr),
        }))
        .expect("serialize nsys export receipt"),
    )
    .expect("persist nsys export receipt");
    assert!(
        export.status.success(),
        "nsys SQLite export failed for {stage}:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&export.stdout),
        String::from_utf8_lossy(&export.stderr)
    );
    assert!(sqlite.is_file(), "nsys did not create {}", sqlite.display());

    let connection = Connection::open(&sqlite).expect("open independent nsys SQLite ledger");
    let mut statement = connection
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        .expect("prepare nsys table inventory");
    let tables = statement
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query nsys table inventory")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect nsys table inventory");
    let strings = nsys_string_ids(&connection, &tables);
    let mut kernels = Vec::<KernelActivity>::new();
    let mut transfers = Vec::<TransferActivity>::new();
    let mut api_rows = 0u64;
    let mut api_tables = Vec::new();
    for table in &tables {
        sqlite_identifier(table);
        let upper = table.to_ascii_uppercase();
        let is_cupti = upper.starts_with("CUPTI_ACTIVITY_KIND_");
        let is_kernel = is_cupti && upper.contains("KERNEL");
        let is_memcpy = is_cupti && upper.contains("MEMCPY");
        let is_api = is_cupti && (upper.contains("RUNTIME") || upper.contains("DRIVER"));
        if is_kernel {
            let columns = sqlite_table_columns(&connection, table);
            for required in ["start", "end", "gridX", "gridY", "gridZ"] {
                assert!(
                    columns.contains(required),
                    "Nsight kernel table {table} lacks `{required}`"
                );
            }
            let name_column = ["demangledName", "shortName", "mangledName", "name"]
                .into_iter()
                .find(|column| columns.contains(*column))
                .unwrap_or_else(|| panic!("Nsight kernel table {table} has no name column"));
            let query = format!(
                "SELECT start, end, gridX, gridY, gridZ, \"{name_column}\" FROM \"{table}\""
            );
            let mut statement = connection
                .prepare(&query)
                .unwrap_or_else(|error| panic!("prepare Nsight kernel query {table}: {error}"));
            let rows = statement
                .query_map([], |row| {
                    let grid_x = row.get::<_, u64>(2)?;
                    let grid_y = row.get::<_, u64>(3)?;
                    let grid_z = row.get::<_, u64>(4)?;
                    Ok(KernelActivity::new(
                        activity_name(row.get_ref(5)?, &strings),
                        row.get(0)?,
                        row.get(1)?,
                        grid_x.saturating_mul(grid_y).saturating_mul(grid_z),
                    ))
                })
                .unwrap_or_else(|error| panic!("query Nsight kernel table {table}: {error}"))
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|error| panic!("collect Nsight kernel table {table}: {error}"));
            kernels.extend(rows);
        }
        if is_memcpy {
            let columns = sqlite_table_columns(&connection, table);
            for required in ["start", "end", "bytes", "copyKind"] {
                assert!(
                    columns.contains(required),
                    "Nsight memcpy table {table} lacks `{required}`"
                );
            }
            let query = format!("SELECT start, end, bytes, copyKind FROM \"{table}\"");
            let mut statement = connection
                .prepare(&query)
                .unwrap_or_else(|error| panic!("prepare Nsight memcpy query {table}: {error}"));
            let rows = statement
                .query_map([], |row| {
                    let start_ns = row.get::<_, u64>(0)?;
                    let end_ns = row.get::<_, u64>(1)?;
                    let bytes = row.get::<_, u64>(2)?;
                    let direction = match row.get::<_, i64>(3)? {
                        1 => Some(TransferDirection::HostToDevice),
                        2 => Some(TransferDirection::DeviceToHost),
                        _ => None,
                    };
                    Ok(direction
                        .map(|direction| TransferActivity::new(direction, start_ns, end_ns, bytes)))
                })
                .unwrap_or_else(|error| panic!("query Nsight memcpy table {table}: {error}"))
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|error| panic!("collect Nsight memcpy table {table}: {error}"));
            transfers.extend(rows.into_iter().flatten());
        }
        if is_api {
            let query = format!("SELECT COUNT(*) FROM \"{table}\"");
            let rows = connection
                .query_row(&query, [], |row| row.get::<_, u64>(0))
                .unwrap_or_else(|error| panic!("query nsys activity table {table}: {error}"));
            api_rows += rows;
            api_tables.push(json!({"table": table, "rows": rows}));
        }
    }
    let dimensions = EvidenceDimensions {
        train_rows: shape.train_rows,
        feature_columns: shape.features,
        oos_rows: shape.oos_rows,
        classes: 3,
    };
    let raw_kernels = kernels
        .iter()
        .map(|kernel| {
            json!({
                "name": kernel.name,
                "start_ns": kernel.start_ns,
                "end_ns": kernel.end_ns,
                "grid_blocks": kernel.grid_blocks,
            })
        })
        .collect::<Vec<_>>();
    let raw_transfers = transfers
        .iter()
        .map(|transfer| {
            json!({
                "direction": match transfer.direction {
                    TransferDirection::HostToDevice => "host_to_device",
                    TransferDirection::DeviceToHost => "device_to_host",
                },
                "start_ns": transfer.start_ns,
                "end_ns": transfer.end_ns,
                "bytes": transfer.bytes,
            })
        })
        .collect::<Vec<_>>();
    let mut semantic_errors =
        validate_cuda_evidence(runtime_backend, dimensions, &kernels, &transfers)
            .err()
            .unwrap_or_default();
    if api_rows == 0 {
        semantic_errors.push(format!(
            "independent nsys ledger saw no CUDA API activity during {stage}"
        ));
    }
    let raw_ledger = json!({
        "stage": stage,
        "report": report,
        "report_sha256": file_sha256(&report),
        "sqlite": sqlite,
        "sqlite_sha256": file_sha256(&sqlite),
        "export_receipt": export_receipt_path,
        "export_receipt_sha256": file_sha256(&export_receipt_path),
        "kernel_rows": kernels.len(),
        "api_rows": api_rows,
        "api_tables": api_tables,
        "raw_kernels": raw_kernels,
        "raw_transfers": raw_transfers,
        "semantic_errors": semantic_errors.clone(),
    });
    let raw_ledger_path = profile_base.with_extension("semantic-ledger.json");
    fs::write(
        &raw_ledger_path,
        serde_json::to_vec_pretty(&raw_ledger).expect("serialize raw semantic CUDA ledger"),
    )
    .expect("persist raw semantic CUDA ledger before verdict");
    assert!(
        semantic_errors.is_empty(),
        "semantic Nsight evidence rejected for {stage}:\n{}",
        semantic_errors.join("\n")
    );
    let validated = validate_cuda_evidence(runtime_backend, dimensions, &kernels, &transfers)
        .expect("already-established semantic CUDA evidence must validate");
    json!({
        "raw_ledger": raw_ledger,
        "raw_ledger_path": raw_ledger_path,
        "raw_ledger_sha256": file_sha256(&raw_ledger_path),
        "semantic_validation": validated,
    })
}

fn run_profiled_stage(
    run_root: &Path,
    shape: Shape,
    artifact_dir: &Path,
    stage: &str,
    parent_deadline: Instant,
    train_identity: &FixtureIdentity,
    oos_identity: &FixtureIdentity,
) -> Value {
    let profile_base = run_root.join(format!("{}-{stage}", shape.label()));
    let mut worker = WorkerProcess::spawn(
        Role::Gpu,
        shape,
        artifact_dir,
        Some(&profile_base),
        parent_deadline,
    );
    let (_, response) = worker.command(stage);
    let response = validate_role_response(Role::Gpu, &response, train_identity, oos_identity);
    let finished_base = worker
        .finish()
        .expect("profiled worker must retain parent-owned base path");
    assert_eq!(finished_base, profile_base);
    let artifact = assert_parent_owned_gpu_artifact(artifact_dir);
    let runtime_backend = artifact["runtime_backend"]
        .as_str()
        .expect("parent-owned GPU artifact must expose runtime_backend");
    let ledger = export_and_query_independent_cuda_ledger(
        &profile_base,
        stage,
        shape,
        runtime_backend,
        parent_deadline,
    );
    json!({
        "response": response,
        "parent_owned_artifact": artifact,
        "independent_cuda_ledger": ledger,
    })
}

fn validate_cpu7_response(response: &Value) -> Value {
    let identities = response["cpu7_identities"]
        .as_array()
        .expect("CPU7 response must contain identities");
    assert_eq!(identities.len(), CPU_WORKERS);
    let tids = identities
        .iter()
        .map(|identity| {
            identity["native_tid"]
                .as_u64()
                .expect("native TID must be u64")
        })
        .collect::<BTreeSet<_>>();
    let names = identities
        .iter()
        .map(|identity| {
            identity["name"]
                .as_str()
                .expect("worker name must be a string")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(tids.len(), CPU_WORKERS);
    assert_eq!(names.len(), CPU_WORKERS);
    assert!(names.iter().all(|name| name.starts_with("neoethos-cpu-")));
    let operation = response["operation"]
        .as_str()
        .expect("CPU7 response operation must be a string");
    if matches!(operation, "warmup" | "sample" | "lifecycle") {
        for identity in identities {
            let work_cpu_ns = identity["work_cpu_ns"]
                .as_u64()
                .expect("CPU7 evidence must persist native work CPU nanoseconds");
            assert!(
                work_cpu_ns >= MINIMUM_CPU_WORK_NS_PER_WORKER,
                "{operation} did not perform meaningful public-model work on {}: {work_cpu_ns} ns",
                identity["name"]
            );
        }
    }
    response.clone()
}

fn assert_fixture_hashes(
    response: &Value,
    train_identity: &FixtureIdentity,
    oos_identity: &FixtureIdentity,
) {
    assert_eq!(
        response["train_feature_sha256"], train_identity.feature_sha256,
        "worker trained on different feature bytes than the parent"
    );
    assert_eq!(
        response["train_label_sha256"], train_identity.label_sha256,
        "worker trained on different label bytes than the parent"
    );
    assert_eq!(
        response["oos_feature_sha256"], oos_identity.feature_sha256,
        "worker predicted on different OOS feature bytes than the parent"
    );
    assert_eq!(
        response["oos_label_sha256"], oos_identity.label_sha256,
        "worker used different OOS label bytes than the parent"
    );
}

fn validate_role_response(
    role: Role,
    response: &Value,
    train_identity: &FixtureIdentity,
    oos_identity: &FixtureIdentity,
) -> Value {
    assert_fixture_hashes(response, train_identity, oos_identity);
    match role {
        Role::Cpu => validate_cpu7_response(response),
        Role::Gpu => {
            assert_eq!(
                response["cpu7_identities"].as_array().map(Vec::len),
                Some(0),
                "GPU response must not manufacture CPU7 identities"
            );
            response.clone()
        }
    }
}

fn measure_role(
    role: Role,
    shape: Shape,
    artifact_dir: &Path,
    parent_deadline: Instant,
    train_identity: &FixtureIdentity,
    oos_identity: &FixtureIdentity,
    run_oracle_gate: bool,
) -> (TimingReceipt, Value) {
    let mut worker = WorkerProcess::spawn(role, shape, artifact_dir, None, parent_deadline);
    let oracle_gate = run_oracle_gate.then(|| {
        let (_, response) = worker.command("oracle_gate");
        validate_role_response(role, &response, train_identity, oos_identity)
    });
    let (warmup_duration, warmup) = worker.command("warmup");
    let warmup = validate_role_response(role, &warmup, train_identity, oos_identity);
    let mut durations = Vec::with_capacity(TIMED_SAMPLES);
    let mut samples = Vec::with_capacity(TIMED_SAMPLES);
    for _ in 0..TIMED_SAMPLES {
        let (duration, response) = worker.command("sample");
        assert!(
            !duration.is_zero(),
            "parent timing produced a zero duration"
        );
        durations.push(duration);
        samples.push(validate_role_response(
            role,
            &response,
            train_identity,
            oos_identity,
        ));
    }
    let _ = worker.finish();
    let probability_digests = samples
        .iter()
        .map(|sample| {
            sample["probability_sha256"]
                .as_str()
                .expect("timed sample must persist probability hash")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        probability_digests.len(),
        1,
        "same-role timed samples produced different probability hashes"
    );
    let timing = TimingReceipt::from_slice(warmup_duration, &durations)
        .expect("one warm-up plus exactly three timed samples");
    (
        timing.clone(),
        json!({
            "role": role.as_str(),
            "timing": timing,
            "oracle_gate": oracle_gate,
            "excluded_warmup_response": warmup,
            "timed_sample_responses": samples,
        }),
    )
}

fn median_seconds(receipt: &TimingReceipt) -> f64 {
    Duration::from_nanos(receipt.median_ns).as_secs_f64()
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("neoethos-models must be two levels below the workspace root")
        .canonicalize()
        .expect("canonicalize tested workspace root")
}

fn git_stdout(
    workspace: &Path,
    parent_deadline: Instant,
    arguments: &[&str],
    context: &str,
) -> Vec<u8> {
    let mut command = Command::new("git");
    command.current_dir(workspace).args(arguments);
    let output = bounded_output(&mut command, parent_deadline, context);
    assert!(
        output.status.success(),
        "{context} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn tested_git_identity(workspace: &Path, parent_deadline: Instant) -> GitIdentity {
    let commit = git_stdout(
        workspace,
        parent_deadline,
        &["rev-parse", "--verify", "HEAD"],
        "resolve tested implementation commit",
    );
    let tree = git_stdout(
        workspace,
        parent_deadline,
        &["rev-parse", "--verify", "HEAD^{tree}"],
        "resolve tested implementation tree",
    );
    let status = git_stdout(
        workspace,
        parent_deadline,
        &["status", "--porcelain=v1", "--untracked-files=all"],
        "verify tested implementation tree is clean",
    );
    GitIdentity::parse(
        &String::from_utf8(commit).expect("Git commit output must be UTF-8"),
        &String::from_utf8(tree).expect("Git tree output must be UTF-8"),
        &String::from_utf8(status).expect("Git status output must be UTF-8"),
    )
    .expect("tested implementation identity must be dynamic and clean")
}

struct ExclusiveAcceptanceLock(PathBuf);

impl ExclusiveAcceptanceLock {
    fn acquire(base: &Path) -> Self {
        let path = base.join("bayesian-r5-one-parent.lock");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap_or_else(|error| {
                panic!(
                    "exactly one Bayesian R5 parent may run; lock {} is unavailable: {error}",
                    path.display()
                )
            });
        writeln!(file, "pid={}", std::process::id()).expect("write exclusive parent PID");
        file.flush().expect("flush exclusive parent lock");
        Self(path)
    }
}

impl Drop for ExclusiveAcceptanceLock {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_file(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "failed to remove Bayesian R5 parent lock {}: {error}",
                self.0.display()
            );
        }
    }
}

fn claim_single_paid_attempt(base: &Path, identity: &GitIdentity) -> PathBuf {
    let path = base.join(PAID_ATTEMPT_CLAIM);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .unwrap_or_else(|error| {
            panic!(
                "the one paid Bayesian R5 attempt was already claimed at {}; manual review is required before any retry: {error}",
                path.display()
            )
        });
    writeln!(file, "schema=neoethos.bayesian-full-gpu-r5.paid-claim.v1")
        .expect("write paid-attempt claim schema");
    writeln!(file, "pid={}", std::process::id()).expect("write paid-attempt claim PID");
    writeln!(file, "implementation_commit={}", identity.commit)
        .expect("write paid-attempt implementation commit");
    writeln!(file, "implementation_tree={}", identity.tree)
        .expect("write paid-attempt implementation tree");
    file.flush().expect("flush permanent paid-attempt claim");
    path
}

fn persistent_evidence_base(workspace: &Path) -> PathBuf {
    let configured = std::env::var_os(EVIDENCE_DIR_ENV).unwrap_or_else(|| {
        panic!("{EVIDENCE_DIR_ENV} is mandatory; acceptance receipts may not be ephemeral")
    });
    let base = PathBuf::from(configured);
    fs::create_dir_all(&base).expect("create configured persistent evidence base");
    let base = base
        .canonicalize()
        .expect("canonicalize persistent evidence base");
    assert!(
        !base.starts_with(workspace),
        "persistent evidence base must be outside the tested Git workspace"
    );
    base
}

fn new_persistent_run_root(base: &Path, identity: &GitIdentity) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_nanos();
    let run_root = base.join(format!("bayesian-r5-{}-{nonce}", &identity.commit[..12]));
    fs::create_dir(&run_root).expect("create unique persistent acceptance run root");
    run_root
}

fn persist_json(path: &Path, value: &Value, context: &str) {
    fs::write(
        path,
        serde_json::to_vec_pretty(value)
            .unwrap_or_else(|error| panic!("serialize {context}: {error}")),
    )
    .unwrap_or_else(|error| panic!("persist {context} {}: {error}", path.display()));
}

fn tracked_source_closure(workspace: &Path, run_root: &Path, parent_deadline: Instant) -> Value {
    let tracked = git_stdout(
        workspace,
        parent_deadline,
        &["ls-files", "-z"],
        "inventory tracked implementation source",
    );
    let mut files = Vec::new();
    let mut total_bytes = 0u64;
    for (index, raw_path) in tracked.split(|byte| *byte == 0).enumerate() {
        if raw_path.is_empty() {
            continue;
        }
        if index % 128 == 0 {
            assert_before_deadline(parent_deadline, "tracked source closure hashing");
        }
        let relative = std::str::from_utf8(raw_path).expect("tracked Git path must be UTF-8");
        if relative.starts_with("vendor/") {
            continue;
        }
        let path = workspace.join(relative);
        let metadata = fs::metadata(&path)
            .unwrap_or_else(|error| panic!("stat tracked source {relative}: {error}"));
        assert!(
            metadata.is_file(),
            "tracked source {relative} must be a file"
        );
        total_bytes = total_bytes
            .checked_add(metadata.len())
            .expect("tracked source byte count overflow");
        files.push(json!({
            "path": relative,
            "bytes": metadata.len(),
            "sha256": file_sha256(&path),
        }));
    }
    let closure_path = run_root.join("implementation-source-closure.json");
    let closure = json!({
        "file_count": files.len(),
        "total_bytes": total_bytes,
        "files": files,
    });
    persist_json(&closure_path, &closure, "implementation source closure");
    json!({
        "path": closure_path,
        "sha256": file_sha256(&closure_path),
        "file_count": closure["file_count"],
        "total_bytes": closure["total_bytes"],
    })
}

fn verified_vendor_closure(workspace: &Path, run_root: &Path, parent_deadline: Instant) -> Value {
    let ignored = git_stdout(
        workspace,
        parent_deadline,
        &[
            "ls-files",
            "-z",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            "vendor",
        ],
        "census ignored untracked vendor inputs",
    );
    assert!(
        ignored.is_empty(),
        "ignored untracked vendor dependencies are forbidden"
    );
    let untracked = git_stdout(
        workspace,
        parent_deadline,
        &[
            "ls-files",
            "-z",
            "--others",
            "--exclude-standard",
            "--",
            "vendor",
        ],
        "census ordinary untracked vendor inputs",
    );
    assert!(
        untracked.is_empty(),
        "untracked vendor dependencies are forbidden"
    );

    let tracked = git_stdout(
        workspace,
        parent_deadline,
        &["ls-files", "-z", "--", "vendor"],
        "inventory tracked vendor inputs",
    );
    let tracked = tracked
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            std::str::from_utf8(path)
                .expect("tracked vendor path must be UTF-8")
                .to_string()
        })
        .collect::<BTreeSet<_>>();
    let ledger_path = workspace.join("audit/bayesian-full-gpu-r5-red/vendor-closure.sha256");
    let ledger_bytes = fs::read(&ledger_path).expect("read tracked vendor closure ledger");
    let ledger = std::str::from_utf8(&ledger_bytes).expect("vendor ledger must be UTF-8");
    let mut ledger_paths = BTreeSet::new();
    let mut total_bytes = 0u64;
    for (index, line) in ledger.lines().enumerate() {
        if index % 128 == 0 {
            assert_before_deadline(parent_deadline, "vendor closure verification");
        }
        let mut fields = line.splitn(3, '\t');
        let expected_hash = fields.next().expect("vendor row hash");
        let expected_bytes = fields
            .next()
            .expect("vendor row byte count")
            .parse::<u64>()
            .expect("vendor byte count must be u64");
        let relative = fields.next().expect("vendor row path");
        assert!(
            relative.starts_with("vendor/"),
            "vendor ledger path escaped vendor/: {relative}"
        );
        assert!(
            ledger_paths.insert(relative.to_string()),
            "duplicate vendor ledger path {relative}"
        );
        let path = workspace.join(relative);
        assert_eq!(
            fs::metadata(&path)
                .unwrap_or_else(|error| panic!("stat vendor input {relative}: {error}"))
                .len(),
            expected_bytes,
            "vendor byte drift for {relative}"
        );
        assert_eq!(
            file_sha256(&path),
            expected_hash,
            "vendor content drift for {relative}"
        );
        total_bytes = total_bytes
            .checked_add(expected_bytes)
            .expect("vendor byte total overflow");
    }
    assert_eq!(
        ledger_paths, tracked,
        "tracked vendor inputs must exactly equal the per-file closure ledger"
    );
    let copied_ledger = run_root.join("vendor-closure.sha256");
    fs::write(&copied_ledger, &ledger_bytes).expect("persist exact tested vendor ledger");
    json!({
        "source_path": ledger_path,
        "evidence_path": copied_ledger,
        "sha256": format!("{:x}", Sha256::digest(&ledger_bytes)),
        "file_count": ledger_paths.len(),
        "total_bytes": total_bytes,
        "ignored_untracked_count": 0,
        "ordinary_untracked_count": 0,
    })
}

fn dependency_closure(workspace: &Path, run_root: &Path, parent_deadline: Instant) -> Value {
    let mut command = Command::new("cargo");
    command.current_dir(workspace).args([
        "+nightly-2026-04-07",
        "metadata",
        "--locked",
        "--offline",
        "--format-version=1",
    ]);
    let output = bounded_output(
        &mut command,
        parent_deadline,
        "locked offline Cargo dependency closure",
    );
    assert!(
        output.status.success(),
        "cargo metadata --locked --offline failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: Value =
        serde_json::from_slice(&output.stdout).expect("parse locked offline Cargo metadata");
    let metadata_path = run_root.join("cargo-metadata-locked-offline.json");
    fs::write(&metadata_path, &output.stdout).expect("persist raw locked offline Cargo metadata");
    let command_receipt_path = run_root.join("cargo-metadata-command.json");
    persist_json(
        &command_receipt_path,
        &json!({
            "command": "cargo +nightly-2026-04-07 metadata --locked --offline --format-version=1",
            "status_code": output.status.code(),
            "stdout_sha256": format!("{:x}", Sha256::digest(&output.stdout)),
            "stderr": String::from_utf8_lossy(&output.stderr),
        }),
        "locked offline Cargo metadata command receipt",
    );
    let mut manifests = Vec::new();
    for package in metadata["packages"]
        .as_array()
        .expect("Cargo metadata packages must be an array")
    {
        let manifest = PathBuf::from(
            package["manifest_path"]
                .as_str()
                .expect("Cargo package manifest_path must be a string"),
        );
        manifests.push(json!({
            "name": package["name"],
            "version": package["version"],
            "source": package["source"],
            "manifest_path": manifest,
            "manifest_sha256": file_sha256(&manifest),
        }));
    }
    let manifests_path = run_root.join("dependency-manifest-closure.json");
    persist_json(
        &manifests_path,
        &json!({"packages": manifests}),
        "dependency manifest closure",
    );
    let cargo_lock = workspace.join("Cargo.lock");
    json!({
        "command": "cargo +nightly-2026-04-07 metadata --locked --offline --format-version=1",
        "metadata_path": metadata_path,
        "metadata_sha256": file_sha256(&metadata_path),
        "command_receipt_path": command_receipt_path,
        "command_receipt_sha256": file_sha256(&command_receipt_path),
        "dependency_manifests_path": manifests_path,
        "dependency_manifests_sha256": file_sha256(&manifests_path),
        "cargo_lock_path": cargo_lock,
        "cargo_lock_sha256": file_sha256(&cargo_lock),
    })
}

fn implementation_identity_receipt(
    workspace: &Path,
    run_root: &Path,
    git_identity: &GitIdentity,
    parent_deadline: Instant,
) -> Value {
    let executable = std::env::current_exe().expect("locate exact acceptance executable");
    let executable = executable
        .canonicalize()
        .expect("canonicalize exact acceptance executable");
    let source_closure = tracked_source_closure(workspace, run_root, parent_deadline);
    let vendor_closure = verified_vendor_closure(workspace, run_root, parent_deadline);
    let dependencies = dependency_closure(workspace, run_root, parent_deadline);
    json!({
        "implementation_commit": git_identity.commit,
        "implementation_tree": git_identity.tree,
        "source_closure": source_closure,
        "executable_path": executable,
        "executable_bytes": fs::metadata(&executable).expect("stat exact executable").len(),
        "executable_sha256": file_sha256(&executable),
        "dependencies": dependencies,
        "vendor_closure": vendor_closure,
    })
}

fn nsys_preflight(run_root: &Path, parent_deadline: Instant) -> Value {
    let mut command = Command::new("nsys");
    command.arg("--version");
    let output = bounded_output(&mut command, parent_deadline, "Nsight Systems preflight");
    let receipt = json!({
        "command": ["nsys", "--version"],
        "status_code": output.status.code(),
        "stdout": String::from_utf8_lossy(&output.stdout),
        "stderr": String::from_utf8_lossy(&output.stderr),
    });
    let path = run_root.join("nsys-preflight.json");
    persist_json(&path, &receipt, "Nsight Systems preflight receipt");
    assert!(
        output.status.success(),
        "Nsight Systems preflight failed; see {}",
        path.display()
    );
    json!({"path": path, "sha256": file_sha256(&path), "raw": receipt})
}

fn normalized_error(left: f64, right: f64) -> f64 {
    (left - right).abs() / left.abs().max(right.abs()).max(1.0)
}

fn oracle_parity_verdict(cpu_role: &Value, gpu_role: &Value) -> Result<Value, Vec<String>> {
    let decode = |role: &Value, label: &str| -> Result<Vec<OracleParityReceipt>, String> {
        let value = role
            .get("oracle_gate")
            .filter(|value| !value.is_null())
            .and_then(|gate| gate.get("oracle_receipts"))
            .cloned()
            .ok_or_else(|| format!("{label} role omitted oracle receipts"))?;
        serde_json::from_value(value)
            .map_err(|error| format!("decode {label} oracle receipts: {error}"))
    };
    let cpu = decode(cpu_role, "CPU").map_err(|error| vec![error])?;
    let gpu = decode(gpu_role, "GPU").map_err(|error| vec![error])?;
    let mut errors = Vec::new();
    if cpu.len() != 3 {
        errors.push(format!(
            "CPU oracle census expected 3 cases, saw {}",
            cpu.len()
        ));
    }
    if gpu.len() != 3 {
        errors.push(format!(
            "GPU oracle census expected 3 cases, saw {}",
            gpu.len()
        ));
    }
    let expected_cases = ["normal", "extreme-finite", "ill-conditioned"]
        .into_iter()
        .collect::<BTreeSet<_>>();
    for (label, receipts) in [("CPU", &cpu), ("GPU", &gpu)] {
        let observed_cases = receipts
            .iter()
            .map(|receipt| receipt.case_name.as_str())
            .collect::<BTreeSet<_>>();
        if observed_cases != expected_cases {
            errors.push(format!(
                "{label} oracle case census mismatch: {observed_cases:?}"
            ));
        }
        for receipt in receipts {
            if receipt.model_artifact_sha256.len() != 64 {
                errors.push(format!(
                    "{label} {} model artifact hash is not SHA-256",
                    receipt.case_name
                ));
            }
            if receipt.posterior_values.len() != receipt.oracle_posterior_values.len() {
                errors.push(format!(
                    "{label} {} public/oracle posterior lengths differ",
                    receipt.case_name
                ));
            }
            if receipt.probability_values.len() != receipt.oracle_probability_values.len()
                || receipt.probability_values.len()
                    != receipt
                        .probability_rows
                        .checked_mul(receipt.probability_columns)
                        .unwrap_or(usize::MAX)
            {
                errors.push(format!(
                    "{label} {} public/oracle probability shape differs",
                    receipt.case_name
                ));
            }
            if receipt.max_normalized_oracle_error > ORACLE_PARITY_TOLERANCE {
                errors.push(format!(
                    "{label} {} maximum oracle error {:.6e} > {:.6e}",
                    receipt.case_name, receipt.max_normalized_oracle_error, ORACLE_PARITY_TOLERANCE
                ));
            }
        }
    }
    let mut maximum_cross_device_error = 0.0_f64;
    for (index, (left, right)) in cpu.iter().zip(&gpu).enumerate() {
        if left.case_name != right.case_name {
            errors.push(format!(
                "oracle case {index} name mismatch: CPU={} GPU={}",
                left.case_name, right.case_name
            ));
        }
        for (field, cpu_hash, gpu_hash) in [
            (
                "training features",
                &left.train_feature_sha256,
                &right.train_feature_sha256,
            ),
            (
                "training labels",
                &left.train_label_sha256,
                &right.train_label_sha256,
            ),
            (
                "OOS features",
                &left.oos_feature_sha256,
                &right.oos_feature_sha256,
            ),
        ] {
            if cpu_hash != gpu_hash {
                errors.push(format!(
                    "{} {field} hash differs across devices",
                    left.case_name
                ));
            }
        }
        let cpu_backend = left.runtime_backend.to_ascii_lowercase();
        if !cpu_backend.contains("cpu")
            || cpu_backend.contains("cuda")
            || cpu_backend.contains("fallback")
        {
            errors.push(format!(
                "{} CPU oracle used invalid backend `{}`",
                left.case_name, left.runtime_backend
            ));
        }
        let gpu_backend = right.runtime_backend.to_ascii_lowercase();
        if !gpu_backend.contains("cuda")
            || gpu_backend.contains("cpu")
            || gpu_backend.contains("fallback")
        {
            errors.push(format!(
                "{} GPU oracle used invalid backend `{}`",
                right.case_name, right.runtime_backend
            ));
        }
        for (field, cpu_oracle, gpu_oracle) in [
            (
                "oracle posterior",
                &left.oracle_posterior_values,
                &right.oracle_posterior_values,
            ),
            (
                "oracle OOS probabilities",
                &left.oracle_probability_values,
                &right.oracle_probability_values,
            ),
        ] {
            if cpu_oracle.len() != gpu_oracle.len()
                || cpu_oracle
                    .iter()
                    .zip(gpu_oracle)
                    .any(|(cpu_value, gpu_value)| cpu_value.to_bits() != gpu_value.to_bits())
            {
                errors.push(format!(
                    "{} {field} was not deterministic across isolated workers",
                    left.case_name
                ));
            }
        }
        for (field, cpu_values, gpu_values) in [
            ("posterior", &left.posterior_values, &right.posterior_values),
            (
                "OOS probabilities",
                &left.probability_values,
                &right.probability_values,
            ),
        ] {
            if cpu_values.len() != gpu_values.len() {
                errors.push(format!(
                    "{} {field} length mismatch: CPU={} GPU={}",
                    left.case_name,
                    cpu_values.len(),
                    gpu_values.len()
                ));
                continue;
            }
            for (value_index, (cpu_value, gpu_value)) in
                cpu_values.iter().zip(gpu_values).enumerate()
            {
                let error = normalized_error(*cpu_value, *gpu_value);
                maximum_cross_device_error = maximum_cross_device_error.max(error);
                if error > ORACLE_PARITY_TOLERANCE {
                    errors.push(format!(
                        "{} {field} value {value_index} CPU/GPU normalized error {error:.6e} > {ORACLE_PARITY_TOLERANCE:.6e}",
                        left.case_name
                    ));
                }
            }
        }
    }
    let verdict = json!({
        "case_count": cpu.len().min(gpu.len()),
        "tolerance": ORACLE_PARITY_TOLERANCE,
        "maximum_cross_device_normalized_error": maximum_cross_device_error,
        "passed": errors.is_empty(),
        "errors": errors.clone(),
    });
    if errors.is_empty() {
        Ok(verdict)
    } else {
        Err(errors)
    }
}

#[test]
#[ignore = "requires designated RTX host, CUDA, nsys, >=1M x 128 memory, and persistent evidence path"]
fn serialized_parent_owns_all_exact_shapes_and_verdicts() {
    assert_eq!(
        std::env::var(ACCEPTANCE_SENTINEL_ENV).as_deref(),
        Ok(ACCEPTANCE_SENTINEL_VALUE),
        "single paid acceptance requires explicit one-run sentinel"
    );
    let parent_started = Instant::now();
    let parent_deadline = parent_started + PARENT_WALL_CEILING;
    let workspace = workspace_root();
    let evidence_base = persistent_evidence_base(&workspace);
    let _exclusive_parent = ExclusiveAcceptanceLock::acquire(&evidence_base);
    let git_identity = tested_git_identity(&workspace, parent_deadline);
    let paid_attempt_claim = claim_single_paid_attempt(&evidence_base, &git_identity);
    let run_root = new_persistent_run_root(&evidence_base, &git_identity);
    let implementation =
        implementation_identity_receipt(&workspace, &run_root, &git_identity, parent_deadline);
    let nsys = nsys_preflight(&run_root, parent_deadline);
    let start_receipt = json!({
        "schema": "neoethos.bayesian-full-gpu-r5.acceptance.v1",
        "implementation": implementation,
        "nsys_preflight": nsys,
        "budget": {
            "exclusive_parent": true,
            "permanent_one_attempt_claim": paid_attempt_claim,
            "permanent_one_attempt_claim_sha256": file_sha256(&paid_attempt_claim),
            "stop_on_first_failure": true,
            "wall_ceiling_seconds": PARENT_WALL_CEILING.as_secs(),
            "feature_width_order": FEATURE_WIDTHS,
            "timed_samples_per_role": TIMED_SAMPLES,
        },
        "required_commands": [
            "cargo +nightly-2026-04-07 test --locked --offline -p neoethos-models --test bayesian_full_gpu_r5_support",
            "cargo +nightly-2026-04-07 test --locked --offline -p neoethos-models --test training_summary_embargo_r5_contract",
            "cargo +nightly-2026-04-07 test --locked --offline -p neoethos-models --test bayesian_full_gpu_r5_contract",
            "cargo +nightly-2026-04-07 test --locked --offline -p neoethos-models --test bayesian_full_gpu_r5_acceptance -- --ignored --exact serialized_parent_owns_all_exact_shapes_and_verdicts --nocapture",
        ],
    });
    let start_path = run_root.join("run-start.json");
    persist_json(&start_path, &start_receipt, "acceptance start receipt");

    let mut shape_receipts = Vec::new();
    for (shape_index, features) in FEATURE_WIDTHS.into_iter().enumerate() {
        assert_before_deadline(parent_deadline, "start exact shape");
        let shape = Shape::exact(features);
        let train_identity = expected_fixture_identity(shape.train_rows, shape.features, 0);
        let oos_identity = expected_fixture_identity(shape.oos_rows, shape.features, 1_000_003);
        let gpu_artifact = run_root.join(format!("{}-gpu-artifact", shape.label()));
        let lifecycle_observation = run_profiled_stage(
            &run_root,
            shape,
            &gpu_artifact,
            "capture_lifecycle",
            parent_deadline,
            &train_identity,
            &oos_identity,
        );
        let parent_artifact = assert_parent_owned_gpu_artifact(&gpu_artifact);

        // Roles are deliberately serialized. CPU completes and exits before
        // GPU timing begins; neither role can steal compute from the other.
        let cpu_artifact = run_root.join(format!("{}-cpu-artifact", shape.label()));
        let (cpu_samples, cpu_receipt) = measure_role(
            Role::Cpu,
            shape,
            &cpu_artifact,
            parent_deadline,
            &train_identity,
            &oos_identity,
            shape_index == 0,
        );
        let cpu_raw_path = run_root.join(format!("{}-cpu-raw.json", shape.label()));
        persist_json(&cpu_raw_path, &cpu_receipt, "raw CPU timing receipt");
        let (gpu_samples, gpu_receipt) = measure_role(
            Role::Gpu,
            shape,
            &gpu_artifact,
            parent_deadline,
            &train_identity,
            &oos_identity,
            shape_index == 0,
        );
        let gpu_raw_path = run_root.join(format!("{}-gpu-raw.json", shape.label()));
        persist_json(&gpu_raw_path, &gpu_receipt, "raw GPU timing receipt");
        let cpu_median_seconds = median_seconds(&cpu_samples);
        let gpu_median_seconds = median_seconds(&gpu_samples);
        let speedup = cpu_median_seconds / gpu_median_seconds;
        let oracle_parity = (shape_index == 0).then(|| {
            oracle_parity_verdict(&cpu_receipt, &gpu_receipt)
                .unwrap_or_else(|errors| json!({"errors": errors, "passed": false}))
        });
        let oracle_passed = oracle_parity
            .as_ref()
            .is_none_or(|verdict| verdict["errors"].as_array().is_none_or(Vec::is_empty));
        let speedup_passed = speedup >= REQUIRED_SPEEDUP;

        let receipt = json!({
            "implementation_commit": git_identity.commit,
            "implementation_tree": git_identity.tree,
            "shape": {
                "train_rows": shape.train_rows,
                "features": shape.features,
                "oos_rows": shape.oos_rows,
            },
            "fixture_identity": {
                "train_feature_sha256": train_identity.feature_sha256,
                "train_label_sha256": train_identity.label_sha256,
                "oos_feature_sha256": oos_identity.feature_sha256,
                "oos_label_sha256": oos_identity.label_sha256,
            },
            "hyperparameters": {
                "prior_precision": PRIOR_PRECISION,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
            },
            "timing_contract": {
                "owner": "parent_process_wall_clock",
                "excluded_warmups_per_role": 1,
                "timed_samples_per_role": TIMED_SAMPLES,
                "work_interval": "public_fit_then_public_oos_predict",
                "cpu_median_seconds": cpu_median_seconds,
                "gpu_median_seconds": gpu_median_seconds,
                "speedup": speedup,
                "required_speedup": REQUIRED_SPEEDUP,
            },
            "cpu": cpu_receipt,
            "gpu": gpu_receipt,
            "cpu_raw_path": cpu_raw_path,
            "cpu_raw_sha256": file_sha256(&cpu_raw_path),
            "gpu_raw_path": gpu_raw_path,
            "gpu_raw_sha256": file_sha256(&gpu_raw_path),
            "oracle_parity": oracle_parity,
            "parent_artifact": parent_artifact,
            "lifecycle_observation": lifecycle_observation,
            "verdict": {
                "speedup_passed": speedup_passed,
                "oracle_passed": oracle_passed,
            },
        });
        let receipt_path = run_root.join(format!("{}-receipt.json", shape.label()));
        persist_json(
            &receipt_path,
            &receipt,
            "parent-owned shape receipt before verdict",
        );
        shape_receipts.push(json!({
            "path": receipt_path,
            "sha256": file_sha256(&receipt_path),
        }));
        assert!(
            oracle_passed,
            "{} independent CPU/GPU posterior and probability parity failed; see {}",
            shape.label(),
            receipt_path.display()
        );
        assert!(
            speedup_passed,
            "{} symmetric parent-timed speedup {speedup:.6}x < {REQUIRED_SPEEDUP:.2}x; raw timings are persisted in {}",
            shape.label(),
            receipt_path.display()
        );
    }

    let manifest = json!({
        "schema": "neoethos.bayesian-full-gpu-r5.acceptance.v1",
        "implementation_commit": git_identity.commit,
        "implementation_tree": git_identity.tree,
        "exact_executable_sha256": start_receipt["implementation"]["executable_sha256"],
        "run_start_path": start_path,
        "run_start_sha256": file_sha256(&start_path),
        "serialized_parent_test": "serialized_parent_owns_all_exact_shapes_and_verdicts",
        "elapsed_ns": u64::try_from(parent_started.elapsed().as_nanos())
            .expect("acceptance elapsed time fits u64"),
        "wall_ceiling_ns": u64::try_from(PARENT_WALL_CEILING.as_nanos())
            .expect("acceptance wall ceiling fits u64"),
        "shape_receipts": shape_receipts,
        "passed": true,
    });
    let manifest_path = run_root.join("manifest.json");
    persist_json(&manifest_path, &manifest, "final acceptance manifest");
}

#[test]
#[ignore = "private long-lived worker; serialized parent owns all inputs and verdicts"]
fn r5_acceptance_child_worker() {
    run_acceptance_worker();
}
