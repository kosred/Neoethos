#[path = "../build_support/native_cuda_build.rs"]
mod native_cuda_build;
#[path = "../src/native_sass.rs"]
mod native_sass;

use native_cuda_build::{
    ArtifactCompiler, ArtifactVerifier, KernelJob, NativeCompileOptions, NativePrecision,
    discover_native_architectures, expand_native_artifact_jobs, inspect_native_cubin,
    native_nvcc_args, order_kernel_jobs_longest_first, reject_free_form_nvcc_args,
    run_native_artifact_jobs, run_native_artifact_verifications, validate_unique_kernel_jobs,
};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const EXPECTED_KERNEL_COUNT: usize = 340;
const FAKE_CUBIN: &[u8] = b"\x7fELFneoethos-fake-native-cubin\n";
const MEASURED_TAIL_SOURCE: &str = "kernels/cuda/market_meanness_index_kernel.cu";
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory {
    path: PathBuf,
    base: PathBuf,
}

impl TestDirectory {
    fn new(label: &str) -> Self {
        let base = std::env::temp_dir();
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = base.join(format!(
            "vector-ta-native-build-{label}-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create isolated native-build test directory");
        Self { path, base }
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let file_name = self.path.file_name().and_then(|name| name.to_str());
        if self.path.starts_with(&self.base)
            && file_name.is_some_and(|name| name.starts_with("vector-ta-native-build-"))
        {
            std::fs::remove_dir_all(&self.path)
                .expect("remove isolated native-build test directory");
        }
    }
}

struct DirectoryLock(PathBuf);

impl DirectoryLock {
    fn acquire(path: PathBuf) -> Self {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    assert!(
                        Instant::now() < deadline,
                        "timed out acquiring fake-tool state lock {}",
                        path.display()
                    );
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("acquire fake-tool state lock {}: {error}", path.display()),
            }
        }
    }
}

impl Drop for DirectoryLock {
    fn drop(&mut self) {
        std::fs::remove_dir(&self.0).expect("release fake-tool state lock");
    }
}

fn read_usize(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(0)
}

fn update_fake_state(
    state: &Path,
    delta: isize,
    event: &str,
    args: &[OsString],
    code: Option<i32>,
) {
    std::fs::create_dir_all(state).expect("create fake-tool state directory");
    let _lock = DirectoryLock::acquire(state.join("lock-dir"));
    let current_path = state.join("current");
    let maximum_path = state.join("maximum");
    let current = read_usize(&current_path) as isize + delta;
    assert!(
        current >= 0,
        "fake NVCC active-process count became negative"
    );
    let current = current as usize;
    let maximum = read_usize(&maximum_path).max(current);
    std::fs::write(&current_path, current.to_string()).expect("write fake active count");
    std::fs::write(&maximum_path, maximum.to_string()).expect("write fake maximum count");

    let mut payload = json!({
        "event": event,
        "pid": std::process::id(),
        "active": current,
        "time_ns": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock precedes Unix epoch")
            .as_nanos()
            .to_string(),
        "args": args.iter().map(|arg| arg.to_string_lossy()).collect::<Vec<_>>(),
    });
    if let Some(code) = code {
        payload["returncode"] = Value::from(code);
    }
    let mut events = OpenOptions::new()
        .create(true)
        .append(true)
        .open(state.join("events.jsonl"))
        .expect("open fake-tool event stream");
    writeln!(events, "{payload}").expect("append fake-tool event");
}

struct ActiveFakeNvcc<'a> {
    state: &'a Path,
    args: &'a [OsString],
    code: i32,
}

impl Drop for ActiveFakeNvcc<'_> {
    fn drop(&mut self) {
        update_fake_state(self.state, -1, "end", self.args, Some(self.code));
    }
}

fn fake_nvcc(args: &[OsString]) -> i32 {
    let display_args = args
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if display_args == ["--list-gpu-arch"] {
        println!("compute_80\ncompute_86\ncompute_89\ncompute_90\ncompute_120");
        return 0;
    }
    if display_args == ["--version"] {
        println!("fake nvcc release 13.0");
        return 0;
    }

    let state = PathBuf::from(
        std::env::var_os("FAKE_NVCC_STATE_DIR").expect("FAKE_NVCC_STATE_DIR is required"),
    );
    update_fake_state(&state, 1, "start", args, None);
    let mut active = ActiveFakeNvcc {
        state: &state,
        args,
        code: 125,
    };
    let fail_match = std::env::var("FAKE_NVCC_FAIL_MATCH").ok();
    let must_fail = fail_match
        .as_deref()
        .is_some_and(|needle| display_args.iter().any(|arg| arg.contains(needle)));
    let delay = if must_fail { 20 } else { 160 };
    std::thread::sleep(Duration::from_millis(delay));
    if must_fail {
        active.code = 17;
        eprintln!(
            "injected fake NVCC failure for {}",
            fail_match.expect("failure match exists")
        );
        return active.code;
    }

    let output_index = display_args
        .iter()
        .position(|arg| arg == "-o")
        .map(|index| index + 1);
    let Some(output) = output_index.and_then(|index| display_args.get(index)) else {
        active.code = 18;
        eprintln!("fake NVCC invocation has no -o output: {display_args:?}");
        return active.code;
    };
    let output = Path::new(output);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("create fake cubin output directory");
    }
    std::fs::write(output, FAKE_CUBIN).expect("write fake native cubin");
    active.code = 0;
    0
}

fn fake_cuobjdump(args: &[OsString]) -> i32 {
    let args = args
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if args == ["--version"] {
        println!("fake cuobjdump release 13.0");
        return 0;
    }
    if args.len() != 2 {
        eprintln!("unexpected fake cuobjdump arguments: {args:?}");
        return 2;
    }
    match args[0].as_str() {
        "--list-ptx" => 0,
        "--dump-sass" => {
            let file_name = Path::new(&args[1])
                .file_name()
                .and_then(|name| name.to_str())
                .expect("fake cuobjdump artifact filename is UTF-8");
            let arch = file_name
                .strip_suffix(".cubin")
                .and_then(|name| name.rsplit_once("_sm"))
                .map(|(_, arch)| arch)
                .expect("recover exact architecture from fake cubin filename");
            println!("arch = sm_{arch}\ncode for sm_{arch}");
            0
        }
        operation => {
            eprintln!("unsupported fake cuobjdump operation: {operation}");
            3
        }
    }
}

fn fake_nvidia_smi(args: &[OsString]) -> i32 {
    let args = args
        .iter()
        .map(|arg| arg.to_string_lossy())
        .collect::<Vec<_>>();
    if !args
        .iter()
        .any(|arg| arg.contains("--query-gpu=compute_cap"))
    {
        eprintln!("unexpected fake nvidia-smi arguments: {args:?}");
        return 2;
    }
    println!("8.9\n8.6\n8.9");
    0
}

fn copy_self_as(directory: &Path, name: &str) -> PathBuf {
    let extension = std::env::consts::EXE_EXTENSION;
    let file_name = if extension.is_empty() {
        name.to_owned()
    } else {
        format!("{name}.{extension}")
    };
    let target = directory.join(file_name);
    std::fs::copy(
        std::env::current_exe().expect("locate harness executable"),
        &target,
    )
    .expect("copy harness as fake CUDA tool");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&target)
            .expect("stat copied fake CUDA tool")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&target, permissions).expect("mark fake CUDA tool executable");
    }
    target
}

fn extract_quoted_strings(source: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut chars = source.chars();
    while let Some(character) = chars.next() {
        if character != '"' {
            continue;
        }
        let mut value = String::new();
        let mut escaped = false;
        for character in chars.by_ref() {
            if escaped {
                value.push(character);
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                break;
            } else {
                value.push(character);
            }
        }
        values.push(value);
    }
    values
}

fn declared_kernel_pairs(source: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut remaining = source;
    while let Some(start) = remaining.find("compile_kernel(") {
        remaining = &remaining[start + "compile_kernel(".len()..];
        let end = remaining
            .find(");")
            .expect("every compile_kernel declaration is terminated");
        let strings = extract_quoted_strings(&remaining[..end]);
        if strings.len() >= 2 && strings[0].ends_with(".cu") {
            pairs.push((strings[0].clone(), strings[1].clone()));
        }
        remaining = &remaining[end + 2..];
    }
    pairs
}

fn production_kernel_jobs() -> Vec<KernelJob> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read_to_string(root.join("build.rs")).expect("read vector-ta build.rs");
    let pairs = declared_kernel_pairs(&source);
    assert_eq!(
        pairs.len(),
        EXPECTED_KERNEL_COUNT,
        "native kernel inventory count changed; review every added or removed artifact"
    );
    pairs
        .into_iter()
        .map(|(rel_src, cubin_stem)| {
            let source_path = root.join(&rel_src);
            let source_bytes = std::fs::metadata(&source_path)
                .unwrap_or_else(|error| panic!("stat {}: {error}", source_path.display()))
                .len();
            let priority = u8::from(rel_src == MEASURED_TAIL_SOURCE);
            KernelJob::new(rel_src, cubin_stem, source_path, source_bytes, priority)
        })
        .collect()
}

fn as_strings(args: &[OsString]) -> Vec<String> {
    args.iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect()
}

fn assert_native_inventory_and_command_contract() {
    let mut jobs = production_kernel_jobs();
    validate_unique_kernel_jobs(&jobs).expect("all 340 production source/stem pairs are unique");
    let pairs = jobs
        .iter()
        .map(|job| (job.rel_src.clone(), job.cubin_stem.clone()))
        .collect::<HashSet<_>>();
    let stems = jobs
        .iter()
        .map(|job| job.cubin_stem.clone())
        .collect::<HashSet<_>>();
    assert_eq!(pairs.len(), EXPECTED_KERNEL_COUNT);
    assert_eq!(stems.len(), EXPECTED_KERNEL_COUNT);

    let largest_source = jobs
        .iter()
        .max_by_key(|job| job.source_bytes)
        .expect("non-empty production kernel inventory")
        .rel_src
        .clone();
    assert_ne!(largest_source, MEASURED_TAIL_SOURCE);
    order_kernel_jobs_longest_first(&mut jobs);
    assert_eq!(jobs[0].rel_src, MEASURED_TAIL_SOURCE);

    let directory = TestDirectory::new("argv");
    let artifact = expand_native_artifact_jobs(&jobs[..1], &[120], &directory.path)
        .expect("expand one exact native artifact")
        .remove(0);
    let args = as_strings(&native_nvcc_args(
        &artifact,
        NativeCompileOptions {
            precision: NativePrecision::StrictF64,
            debug_line_info: false,
        },
    ));
    assert!(args.iter().any(|arg| arg == "--cubin"));
    assert!(
        args.iter()
            .any(|arg| arg == "-gencode=arch=compute_120,code=sm_120")
    );
    assert!(args.iter().any(|arg| arg == "-prec-div=true"));
    assert!(args.iter().any(|arg| arg == "-prec-sqrt=true"));
    assert!(args.iter().any(|arg| arg == "-fmad=false"));
    assert!(args.iter().any(|arg| arg == "-ftz=false"));
    assert!(!args.iter().any(|arg| arg.contains("code=compute_")));
    assert!(!args.iter().any(|arg| arg == "-ptx" || arg == "-fatbin"));
    let fast_math = as_strings(&native_nvcc_args(
        &artifact,
        NativeCompileOptions {
            precision: NativePrecision::FastMath,
            debug_line_info: false,
        },
    ));
    assert!(fast_math.iter().any(|arg| arg == "--use_fast_math"));
    assert!(!fast_math.iter().any(|arg| arg == "-fmad=false"));
    assert!(reject_free_form_nvcc_args(None).is_ok());
    assert!(reject_free_form_nvcc_args(Some("  ")).is_ok());
    assert!(reject_free_form_nvcc_args(Some("--use_fast_math")).is_err());

    let build = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs"))
        .expect("read build.rs source contract");
    assert!(!build.contains("rerun-if-changed=kernels/cubin"));
    let source_offset = |needle: &str| {
        build.find(needle).unwrap_or_else(|| {
            panic!("build.rs is missing required Rust contract token {needle:?}")
        })
    };
    assert!(source_offset("jobserver::Client::from_env") < source_offset("compile_cuda_kernels("));
    assert!(source_offset("reject_free_form_nvcc_args()") < source_offset("compile_alma_kernel("));
}

fn run_tool(path: &Path, args: &[&str]) -> Output {
    Command::new(path)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("run fake tool {}: {error}", path.display()))
}

fn assert_arch_discovery_and_cubin_inspection_contract() {
    let directory = TestDirectory::new("tools");
    let nvcc = copy_self_as(&directory.path, "fake-nvcc");
    let cuobjdump = copy_self_as(&directory.path, "fake-cuobjdump");
    let nvidia_smi = copy_self_as(&directory.path, "fake-nvidia-smi");

    let automatic = discover_native_architectures(&nvcc, &nvidia_smi, None)
        .expect("discover every visible supported architecture");
    assert_eq!(automatic.architectures, vec![86, 89]);
    assert_eq!(automatic.source.as_str(), "auto-detected-visible-gpus");
    let explicit = discover_native_architectures(&nvcc, &nvidia_smi, Some("sm_120, 12.0"))
        .expect("canonicalize an explicit exact architecture");
    assert_eq!(explicit.architectures, vec![120]);
    assert_eq!(explicit.source.as_str(), "explicit-cuda-archs");

    let cubin = directory.path.join("probe_sm120.cubin");
    std::fs::write(&cubin, FAKE_CUBIN).expect("write exact cubin inspection fixture");
    inspect_native_cubin(&cuobjdump, &cubin, 120)
        .expect("fake cuobjdump proves exact sm_120 SASS and no PTX");
    assert!(
        run_tool(&cuobjdump, &["--list-ptx", cubin.to_str().unwrap()])
            .status
            .success()
    );

    let artifact = native_sass::NativeArtifact::new("probe", 120, FAKE_CUBIN);
    native_sass::validate_native_manifest(&[artifact], &["probe"], &[120])
        .expect("one exact manifest coordinate is complete");
    assert_eq!(
        native_sass::select_exact_native_cubin("probe", 12, 0, &[artifact])
            .expect("runtime selector returns the same exact sm_120 cubin"),
        FAKE_CUBIN
    );
}

struct FakeCompiler {
    executable: PathBuf,
    state: PathBuf,
    fail_match: Option<String>,
    post_child_delay: Duration,
}

impl ArtifactCompiler for FakeCompiler {
    fn command(&self, job: &native_cuda_build::ArtifactJob) -> Result<Command, String> {
        let mut command = Command::new(&self.executable);
        command.args(native_nvcc_args(job, NativeCompileOptions::default()));
        command.env("FAKE_NVCC_STATE_DIR", &self.state);
        if let Some(fail_match) = &self.fail_match {
            command.env("FAKE_NVCC_FAIL_MATCH", fail_match);
        }
        Ok(command)
    }

    fn finish(&self, job: &native_cuda_build::ArtifactJob, output: Output) -> Result<(), String> {
        std::thread::sleep(self.post_child_delay);
        if output.status.success() {
            Ok(())
        } else {
            Err(format!(
                "fake NVCC failed for {} sm_{} with {:?}: {}",
                job.rel_src,
                job.arch,
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }
}

struct FakeVerifier {
    delay: Duration,
}

impl ArtifactVerifier for FakeVerifier {
    fn verify(&self, _job: &native_cuda_build::ArtifactJob) -> Result<(), String> {
        std::thread::sleep(self.delay);
        Ok(())
    }
}

fn reset_fake_state(state: &Path) {
    std::fs::create_dir_all(state).expect("create fake state directory");
    std::fs::write(state.join("current"), "0").expect("reset current count");
    std::fs::write(state.join("maximum"), "0").expect("reset maximum count");
    File::create(state.join("events.jsonl")).expect("reset fake event stream");
}

fn fake_events(state: &Path) -> Vec<Value> {
    std::fs::read_to_string(state.join("events.jsonl"))
        .expect("read fake event stream")
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).expect("parse fake event"))
        .collect()
}

fn assert_jobserver_parallel_cancel_and_telemetry_contract() {
    let directory = TestDirectory::new("scheduler");
    let fake_nvcc = copy_self_as(&directory.path, "fake-nvcc");
    let state = directory.path.join("state");
    reset_fake_state(&state);
    let mut kernels = production_kernel_jobs();
    order_kernel_jobs_longest_first(&mut kernels);

    let success_jobs = expand_native_artifact_jobs(&kernels[..12], &[89], &directory.path)
        .expect("expand successful scheduler fixture");
    let success_compiler = FakeCompiler {
        executable: fake_nvcc.clone(),
        state: state.clone(),
        fail_match: None,
        post_child_delay: Duration::ZERO,
    };
    let permits = jobserver::Client::new(3).expect("create three extra Cargo-style permits");
    let report = run_native_artifact_jobs(success_jobs, 4, Some(&permits), &success_compiler)
        .expect("all fake native cubins compile");
    assert!(report.observed_max_active > 1);
    assert!(report.observed_max_active <= 4);
    assert_eq!(report.started, 12);
    assert_eq!(report.completed, 12);
    assert_eq!(report.failed, 0);
    assert_eq!(report.active_at_end, 0);
    assert_eq!(read_usize(&state.join("current")), 0);
    assert!(read_usize(&state.join("maximum")) > 1);
    assert_eq!(
        fake_events(&state)
            .iter()
            .filter(|event| event["event"] == "start")
            .count(),
        12
    );
    let json = report.stable_json();
    assert!(json.contains("neoethos.vector_ta.native_cuda_scheduler.v4"));
    assert!(json.contains("\"phase\":\"nvcc_child\""));
    assert!(json.contains("\"kind\":\"nvcc_child\""));
    assert!(!json.contains("ptx") && !json.contains("fatbin"));

    reset_fake_state(&state);
    let split_jobs = expand_native_artifact_jobs(&kernels[..1], &[86, 89], &directory.path)
        .expect("expand independent cross-architecture jobs");
    let split_permits = jobserver::Client::new(1).expect("create one extra permit");
    let split = run_native_artifact_jobs(split_jobs, 2, Some(&split_permits), &success_compiler)
        .expect("one source compiles both architectures independently");
    assert_eq!(split.started, 2);
    assert_eq!(split.observed_max_active, 2);

    reset_fake_state(&state);
    let failure_kernels = &kernels[..4];
    let first_source = failure_kernels[0].rel_src.clone();
    let failure_jobs = expand_native_artifact_jobs(failure_kernels, &[86, 89], &directory.path)
        .expect("expand cancellation fixture");
    let failure_compiler = FakeCompiler {
        executable: fake_nvcc,
        state: state.clone(),
        fail_match: Some(first_source),
        post_child_delay: Duration::ZERO,
    };
    let failure_permits = jobserver::Client::new(3).expect("create failure-lane permits");
    let failure =
        run_native_artifact_jobs(failure_jobs, 4, Some(&failure_permits), &failure_compiler)
            .expect_err("an NVCC failure must fail the complete build lane");
    assert!(failure.report.started < 8);
    assert!((1..=2).contains(&failure.report.failed));
    assert_eq!(failure.report.active_at_end, 0);
    assert_eq!(read_usize(&state.join("current")), 0);

    reset_fake_state(&state);
    assert!(reject_free_form_nvcc_args(Some("--use_fast_math")).is_err());
    assert!(fake_events(&state).is_empty());
}

fn assert_duplicate_architecture_expansion_fails_closed() {
    let directory = TestDirectory::new("duplicate-architecture");
    let jobs = production_kernel_jobs();
    let error = expand_native_artifact_jobs(&jobs[..1], &[120, 120], &directory.path)
        .expect_err("duplicate exact architectures must fail before same-output jobs are created");
    assert_eq!(
        error.to_string(),
        "duplicate exact native CUDA architecture sm_120"
    );
}

fn assert_nvcc_telemetry_excludes_post_child_verification_work() {
    let directory = TestDirectory::new("nvcc-child-telemetry");
    let fake_nvcc = copy_self_as(&directory.path, "fake-nvcc");
    let state = directory.path.join("state");
    reset_fake_state(&state);
    let jobs = production_kernel_jobs();
    let artifact_jobs = expand_native_artifact_jobs(&jobs[..1], &[89], &directory.path)
        .expect("expand one NVCC telemetry fixture");
    let compiler = FakeCompiler {
        executable: fake_nvcc,
        state,
        fail_match: None,
        // This models native-cubin verification that runs after the NVCC child.
        // It must not inflate a metric named observed_nvcc_span_seconds.
        post_child_delay: Duration::from_millis(500),
    };
    let wall_start = Instant::now();
    let report = run_native_artifact_jobs(artifact_jobs.clone(), 1, None, &compiler)
        .expect("the fake NVCC child succeeds");
    let total_wall = wall_start.elapsed();
    let payload: Value =
        serde_json::from_str(&report.stable_json()).expect("scheduler telemetry is valid JSON");
    let nvcc_span = payload["observed_nvcc_span_seconds"]
        .as_f64()
        .expect("NVCC report has an honest child-only span");
    assert!(
        total_wall >= Duration::from_millis(600),
        "fixture did not exercise post-child work: {total_wall:?}"
    );
    assert!(
        nvcc_span < 0.4,
        "NVCC span {nvcc_span:.3}s incorrectly includes 500 ms of post-child verification"
    );

    let verification = run_native_artifact_verifications(
        artifact_jobs,
        1,
        None,
        &FakeVerifier {
            delay: Duration::from_millis(50),
        },
    )
    .expect("separate native-cubin verification succeeds");
    let verification_payload: Value = serde_json::from_str(&verification.stable_json())
        .expect("verification telemetry is valid JSON");
    assert_eq!(
        verification_payload["phase"],
        Value::from("native_cubin_verification")
    );
    assert!(verification_payload["observed_verification_span_seconds"].is_number());
    assert!(
        verification_payload
            .get("observed_max_active_nvcc")
            .is_none(),
        "verification report must not advertise verifier jobs as active NVCC children"
    );

    let build = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs"))
        .expect("read build.rs telemetry contract");
    let compiler = build
        .split("impl ArtifactCompiler for NativeNvccCompiler")
        .nth(1)
        .and_then(|tail| tail.split("struct NativeCubinVerifier").next())
        .expect("native compiler and verifier implementations are structurally separate");
    assert!(
        !compiler.contains("inspect_native_cubin"),
        "NativeNvccCompiler still performs verification inside NVCC telemetry"
    );
    assert!(build.contains("run_native_artifact_verifications"));
}

fn assert_sqwma_grid_binds_block_width_to_outputs_per_thread() {
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cuda/moving_averages/sqwma_wrapper.rs"),
    )
    .expect("read SQWMA CUDA wrapper source contract");
    let grid = source
        .split("fn grid_x_for_series")
        .nth(1)
        .and_then(|tail| tail.split("fn prepare_batch_inputs").next())
        .expect("SQWMA grid_x_for_series has a bounded source body");
    let block = grid
        .find("let bx = Self::block_x() as u64;")
        .expect("SQWMA grid sizing binds the CUDA block width");
    let outputs = grid
        .find("let opt = Self::out_per_thread() as u64;")
        .expect("SQWMA grid sizing binds outputs-per-thread before use");
    let tile = grid
        .find("let tile = bx * opt;")
        .expect("SQWMA grid sizing multiplies block width by outputs-per-thread");
    assert!(block < outputs && outputs < tile);
}

#[test]
fn duplicate_architecture_expansion_fails_closed() {
    assert_duplicate_architecture_expansion_fails_closed();
}

#[test]
fn nvcc_telemetry_excludes_post_child_verification_work() {
    assert_nvcc_telemetry_excludes_post_child_verification_work();
}

#[test]
fn sqwma_grid_binds_block_width_to_outputs_per_thread() {
    assert_sqwma_grid_binds_block_width_to_outputs_per_thread();
}

fn run_parity_contracts() {
    assert_native_inventory_and_command_contract();
    assert_arch_discovery_and_cubin_inspection_contract();
    assert_jobserver_parallel_cancel_and_telemetry_contract();
    assert_duplicate_architecture_expansion_fails_closed();
    assert_nvcc_telemetry_excludes_post_child_verification_work();
    assert_sqwma_grid_binds_block_width_to_outputs_per_thread();
    println!(
        "{}",
        json!({
            "schema": "neoethos.vector_ta.native_cuda_build_rust_parity.v1",
            "production_kernel_jobs": EXPECTED_KERNEL_COUNT,
            "native_only": true,
            "real_child_processes": true,
            "cargo_style_jobserver_permits": true,
        })
    );
}

fn main() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let executable = std::env::current_exe().expect("locate current executable");
    let stem = executable
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let code = if stem.starts_with("fake-nvcc") {
        Some(fake_nvcc(&args))
    } else if stem.starts_with("fake-cuobjdump") {
        Some(fake_cuobjdump(&args))
    } else if stem.starts_with("fake-nvidia-smi") {
        Some(fake_nvidia_smi(&args))
    } else if args.first().and_then(|arg| arg.to_str()) == Some("--check-duplicate-architecture") {
        assert_duplicate_architecture_expansion_fails_closed();
        None
    } else if args.first().and_then(|arg| arg.to_str()) == Some("--check-nvcc-telemetry") {
        assert_nvcc_telemetry_excludes_post_child_verification_work();
        None
    } else {
        run_parity_contracts();
        None
    };
    if let Some(code) = code {
        std::process::exit(code);
    }
}
