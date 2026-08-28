use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use neoethos_data::core::dataset_manifest::read_current_manifest;
use neoethos_data::core::import_limits::ImportLimits;
use neoethos_data::core::import_provenance::ImportSourceFormat;
use neoethos_data::core::import_service::{ImportRequest, import_path_to_vortex};
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
    MAX_STREAMING_VORTEX_BUFFER_BYTES,
};

mod common;

const CHILD_MARKER: &str = "NEOETHOS_IMPORT_MEMORY_CHILD";
const SOURCE_PATH: &str = "NEOETHOS_IMPORT_MEMORY_SOURCE";
const ROOT_PATH: &str = "NEOETHOS_IMPORT_MEMORY_ROOT";
const READY_PATH: &str = "NEOETHOS_IMPORT_MEMORY_READY";
const GO_PATH: &str = "NEOETHOS_IMPORT_MEMORY_GO";
const ROWS: usize = 200_000;

fn identity() -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::external(
        "bounded-memory-fixture",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity")
}

#[test]
fn bounded_memory_child() {
    if std::env::var_os(CHILD_MARKER).is_none() {
        return;
    }
    let source = PathBuf::from(std::env::var_os(SOURCE_PATH).expect("child source path"));
    let root = PathBuf::from(std::env::var_os(ROOT_PATH).expect("child root path"));
    let ready = PathBuf::from(std::env::var_os(READY_PATH).expect("child ready path"));
    let go = PathBuf::from(std::env::var_os(GO_PATH).expect("child go path"));
    fs::write(&ready, b"ready").expect("announce child readiness");
    let deadline = Instant::now() + Duration::from_secs(30);
    while !go.exists() {
        assert!(Instant::now() < deadline, "parent did not release child");
        thread::sleep(Duration::from_millis(2));
    }

    let identity = identity();
    let grant = common::import_grant();
    let result = import_path_to_vortex(ImportRequest {
        source_path: &source,
        configured_root: &root,
        identity: &identity,
        declared_format: ImportSourceFormat::Csv,
        expected_generation: None,
        limits: &ImportLimits::conservative_for_tests(),
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect("bounded child import");
    assert_eq!(result.row_count(), ROWS as u64);
}

#[test]
fn large_import_stays_within_rss_and_staging_disk_bounds() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("large.csv");
    let root = temp.path().join("canonical");
    let ready = temp.path().join("child.ready");
    let go = temp.path().join("child.go");
    write_large_csv(&source);
    let source_bytes = fs::metadata(&source).expect("source metadata").len();
    let limits = ImportLimits::conservative_for_tests();
    limits
        .check_source_bytes(source_bytes)
        .expect("fixture remains under configured source limit");

    let mut child = Command::new(std::env::current_exe().expect("current test executable"))
        .arg("--exact")
        .arg("bounded_memory_child")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD_MARKER, "1")
        .env(SOURCE_PATH, &source)
        .env(ROOT_PATH, &root)
        .env(READY_PATH, &ready)
        .env(GO_PATH, &go)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn bounded-memory child");

    let startup_deadline = Instant::now() + Duration::from_secs(30);
    while !ready.exists() {
        if let Some(status) = child.try_wait().expect("poll child startup") {
            panic!("bounded-memory child exited before readiness: {status}");
        }
        assert!(
            Instant::now() < startup_deadline,
            "bounded-memory child readiness timeout"
        );
        thread::sleep(Duration::from_millis(2));
    }
    let baseline_rss = process_rss_bytes(child.id()).expect("sample child baseline RSS");
    fs::write(&go, b"go").expect("release bounded-memory child");

    let mut peak_rss = baseline_rss;
    let mut peak_scratch = 0_u64;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll bounded-memory child") {
            break status;
        }
        if let Ok(rss) = process_rss_bytes(child.id()) {
            peak_rss = peak_rss.max(rss);
        }
        peak_scratch = peak_scratch.max(directory_file_bytes(&root));
        thread::sleep(Duration::from_millis(2));
    };
    peak_scratch = peak_scratch.max(directory_file_bytes(&root));
    assert!(status.success(), "bounded-memory child failed: {status}");

    let rss_delta = peak_rss.saturating_sub(baseline_rss);
    let rss_ceiling = u64::try_from(limits.max_arrow_batch_bytes()).expect("batch bytes")
        + MAX_STREAMING_VORTEX_BUFFER_BYTES
        + 64 * 1024 * 1024;
    assert!(
        rss_delta <= rss_ceiling,
        "import RSS delta {rss_delta} exceeded configured ceiling {rss_ceiling} (baseline={baseline_rss}, peak={peak_rss})"
    );
    let disk_ceiling = source_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(64 * 1024 * 1024))
        .expect("disk ceiling arithmetic");
    eprintln!(
        "bounded import metrics: source_bytes={source_bytes} baseline_rss={baseline_rss} peak_rss={peak_rss} rss_delta={rss_delta} rss_ceiling={rss_ceiling} peak_scratch={peak_scratch} disk_ceiling={disk_ceiling}"
    );
    assert!(
        peak_scratch <= disk_ceiling,
        "import scratch high-water {peak_scratch} exceeded {disk_ceiling}"
    );

    let manifest = read_current_manifest(&root, &identity()).expect("published manifest");
    assert_eq!(manifest.row_count(), ROWS as u64);
    let leaked_staging = root.join(".import-staging");
    if leaked_staging.exists() {
        assert_eq!(
            fs::read_dir(&leaked_staging)
                .expect("staging directory")
                .count(),
            0,
            "successful import leaked a staging file"
        );
    }
}

#[test]
fn disk_admission_reserves_staging_candidate_and_free_space_margin() {
    let limits =
        ImportLimits::conservative_for_tests().with_storage_bounds(1_000, 1_000, 2_000, 3_000);
    assert_eq!(
        limits
            .required_peak_disk_bytes(800)
            .expect("checked disk arithmetic"),
        5_800,
        "peak admission must include source staging + candidate + retained free-space margin"
    );
}

#[test]
fn candidate_limit_fails_closed_and_cleans_every_import_artifact() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("small.csv");
    let root = temp.path().join("canonical");
    fs::write(
        &source,
        concat!(
            "timestamp,open,high,low,close\n",
            "1700000040000,1.1,1.2,1.0,1.15\n",
            "1700000100000,1.15,1.25,1.05,1.2\n",
        ),
    )
    .expect("write source");
    let limits = ImportLimits::conservative_for_tests().with_storage_bounds(
        32 * 1024 * 1024,
        32 * 1024 * 1024,
        1,
        0,
    );
    let identity = identity();
    let grant = common::import_grant();
    let error = import_path_to_vortex(ImportRequest {
        source_path: &source,
        configured_root: &root,
        identity: &identity,
        declared_format: ImportSourceFormat::Csv,
        expected_generation: None,
        limits: &limits,
        auxiliary_slot: grant
            .auxiliary_slot()
            .expect("import grant owns a source-seal slot"),
    })
    .expect_err("one-byte candidate limit must fail before publication");
    assert!(format!("{error:#}").contains("CandidateBytes"), "{error:#}");
    assert!(read_current_manifest(&root, &identity).is_err());
    assert_eq!(
        directory_file_bytes(&root),
        0,
        "failed bounded import leaked staging/candidate/generation bytes"
    );
}

fn write_large_csv(path: &Path) {
    let file = File::create(path).expect("create large CSV");
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    writeln!(writer, "timestamp,open,high,low,close,volume").expect("header");
    let base = 1_700_000_040_000_i64;
    for row in 0..ROWS {
        let timestamp = base + i64::try_from(row).expect("row index") * 60_000;
        let open = 1.0 + (row % 10_000) as f64 * 0.000_000_01;
        writeln!(
            writer,
            "{timestamp},{open:.17},{:.17},{:.17},{:.17},{}",
            open + 0.000_2,
            open - 0.000_2,
            open + 0.000_1,
            row % 16_777_218,
        )
        .expect("large CSV row");
    }
    writer.flush().expect("flush large CSV");
}

fn directory_file_bytes(root: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(root) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .map(|entry| {
            let path = entry.path();
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => directory_file_bytes(&path),
                Ok(kind) if kind.is_file() => {
                    fs::metadata(path).map(|value| value.len()).unwrap_or(0)
                }
                _ => 0,
            }
        })
        .sum()
}

#[cfg(target_os = "linux")]
fn process_rss_bytes(pid: u32) -> std::io::Result<u64> {
    let status = fs::read_to_string(format!("/proc/{pid}/status"))?;
    let line = status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .ok_or_else(|| std::io::Error::other("VmRSS missing"))?;
    let kib = line
        .split_ascii_whitespace()
        .nth(1)
        .ok_or_else(|| std::io::Error::other("VmRSS value missing"))?
        .parse::<u64>()
        .map_err(std::io::Error::other)?;
    kib.checked_mul(1024)
        .ok_or_else(|| std::io::Error::other("VmRSS overflow"))
}

#[cfg(windows)]
fn process_rss_bytes(pid: u32) -> std::io::Result<u64> {
    type Handle = *mut core::ffi::c_void;
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit_handle: i32, process_id: u32) -> Handle;
        fn CloseHandle(object: Handle) -> i32;
        fn K32GetProcessMemoryInfo(
            process: Handle,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const PROCESS_VM_READ: u32 = 0x0010;
    let handle =
        unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, 0, pid) };
    if handle.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    let mut counters = ProcessMemoryCounters {
        cb: u32::try_from(std::mem::size_of::<ProcessMemoryCounters>())
            .expect("counter size fits u32"),
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    let result = unsafe {
        K32GetProcessMemoryInfo(
            handle,
            &mut counters,
            u32::try_from(std::mem::size_of::<ProcessMemoryCounters>())
                .expect("counter size fits u32"),
        )
    };
    let error = (result == 0).then(std::io::Error::last_os_error);
    unsafe {
        CloseHandle(handle);
    }
    if let Some(error) = error {
        Err(error)
    } else {
        u64::try_from(counters.working_set_size)
            .map_err(|_| std::io::Error::other("working-set size does not fit u64"))
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
fn process_rss_bytes(_pid: u32) -> std::io::Result<u64> {
    Err(std::io::Error::other(
        "RSS sampling is not implemented on this platform",
    ))
}
