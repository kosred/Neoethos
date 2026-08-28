//! Immutable private staging for mutable user-provided paths.

use crate::core::import_limits::ImportLimits;
use crate::core::source_seal::SourceSeal;
use anyhow::{Context, Result, bail};
use neoethos_core::execution_budget::AuxiliarySlotLease;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const COPY_BUFFER_BYTES: usize = 1024 * 1024;
static SNAPSHOT_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct SourceSnapshot {
    path: PathBuf,
    source_sha256: [u8; 32],
    source_size: u64,
    stable_source_identity: String,
}

impl SourceSnapshot {
    pub fn capture_path(
        source_path: &Path,
        staging_parent: &Path,
        limits: &ImportLimits,
        auxiliary_slot: &AuxiliarySlotLease,
    ) -> Result<Self> {
        Self::capture_path_observed(source_path, staging_parent, limits, auxiliary_slot, |_| {})
    }

    fn capture_path_observed(
        source_path: &Path,
        staging_parent: &Path,
        limits: &ImportLimits,
        auxiliary_slot: &AuxiliarySlotLease,
        mut copy_observer: impl FnMut(u64),
    ) -> Result<Self> {
        let source_metadata = fs::symlink_metadata(source_path)
            .with_context(|| format!("inspect import source {}", source_path.display()))?;
        if source_metadata.file_type().is_symlink() || !source_metadata.is_file() {
            bail!(
                "import source must be a regular non-symlink file: {}",
                source_path.display()
            );
        }
        limits.check_source_bytes(source_metadata.len())?;
        fs::create_dir_all(staging_parent).with_context(|| {
            format!(
                "create private import staging directory {}",
                staging_parent.display()
            )
        })?;
        check_free_disk(staging_parent, source_metadata.len(), limits)?;

        let mut source = open_source_for_seal(source_path)?;
        let mut source_seal = SourceSeal::acquire(&source, source_path, auxiliary_slot)?;
        let opened_identity = stable_file_identity(&source)?;
        let path_identity = stable_path_identity(source_path)?;
        if opened_identity != path_identity {
            bail!("import source identity changed while it was being opened");
        }

        let nonce = SNAPSHOT_NONCE.fetch_add(1, Ordering::Relaxed);
        let staging_path =
            staging_parent.join(format!("source-{}-{nonce}.stage", std::process::id()));
        let mut cleanup = SnapshotCleanup::new(staging_path.clone());
        let mut staging = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&staging_path)
            .with_context(|| format!("create private staging file {}", staging_path.display()))?;
        let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
        let mut total = 0_u64;
        let mut hasher = Sha256::new();
        copy_observer(0);
        loop {
            source_seal.check_clean()?;
            let count = source
                .read(&mut buffer)
                .with_context(|| format!("read sealed source {}", source_path.display()))?;
            if count == 0 {
                break;
            }
            total = total
                .checked_add(u64::try_from(count).context("copy count does not fit u64")?)
                .context("source snapshot byte count overflow")?;
            limits.check_source_bytes(total)?;
            limits.check_staging_bytes(total)?;
            staging
                .write_all(&buffer[..count])
                .with_context(|| format!("write staging file {}", staging_path.display()))?;
            hasher.update(&buffer[..count]);
            copy_observer(total);
            source_seal.check_clean()?;
        }
        if total != source_metadata.len() {
            bail!(
                "sealed source size changed during copy: expected {}, copied {}",
                source_metadata.len(),
                total
            );
        }
        let final_path_identity = stable_path_identity(source_path)?;
        if final_path_identity != opened_identity {
            bail!("import source path identity changed during snapshot copy");
        }
        source_seal.check_clean()?;
        staging
            .flush()
            .context("flush private import staging file")?;
        staging
            .sync_all()
            .context("fsync private import staging file")?;
        drop(staging);
        source_seal.check_clean()?;
        source_seal.release(&source)?;
        drop(source);

        let source_sha256: [u8; 32] = hasher.finalize().into();
        let verified = hash_file_bounded(&staging_path, limits)?;
        if verified != source_sha256 {
            bail!("private import staging hash mismatch after fsync/reopen");
        }
        cleanup.committed = true;
        Ok(Self {
            path: staging_path,
            source_sha256,
            source_size: total,
            stable_source_identity: opened_identity,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn source_sha256(&self) -> &[u8; 32] {
        &self.source_sha256
    }

    pub const fn source_size(&self) -> u64 {
        self.source_size
    }

    pub fn stable_source_identity(&self) -> &str {
        &self.stable_source_identity
    }
}

impl Drop for SourceSnapshot {
    fn drop(&mut self) {
        remove_snapshot_path_or_log(&self.path, "completed_snapshot");
    }
}

fn remove_snapshot_path(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("remove private import staging file {}", path.display())),
    }
}

fn remove_snapshot_path_or_log(path: &Path, cleanup_phase: &'static str) {
    if let Err(error) = remove_snapshot_path(path) {
        tracing::error!(
            target: "neoethos_data::source_snapshot",
            staging_path = %path.display(),
            cleanup_phase,
            error = %format!("{error:#}"),
            "IMPORT STAGING CLEANUP FAILED; disk bytes remain and require operator recovery"
        );
    }
}

fn hash_file_bounded(path: &Path, limits: &ImportLimits) -> Result<[u8; 32]> {
    let mut file =
        File::open(path).with_context(|| format!("reopen staging {}", path.display()))?;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut total = 0_u64;
    let mut hasher = Sha256::new();
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(count).context("hash count does not fit u64")?)
            .context("staging hash byte count overflow")?;
        limits.check_staging_bytes(total)?;
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().into())
}

fn check_free_disk(path: &Path, source_bytes: u64, limits: &ImportLimits) -> Result<()> {
    let required = limits
        .required_peak_disk_bytes(source_bytes)
        .context("calculate bounded import peak disk requirement")?;
    let available = available_disk_bytes(path)?;
    if available < required {
        bail!(
            "insufficient staging disk at {}: available {}, required {}",
            path.display(),
            available,
            required
        );
    }
    Ok(())
}

#[cfg(unix)]
fn available_disk_bytes(path: &Path) -> Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes())
        .context("staging path contains an interior NUL")?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    let result = unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) };
    if result != 0 {
        return Err(std::io::Error::last_os_error()).context("statvfs staging directory");
    }
    let stats = unsafe { stats.assume_init() };
    u64::try_from(u128::from(stats.f_bavail) * u128::from(stats.f_frsize))
        .context("available staging disk does not fit u64")
}

#[cfg(windows)]
fn available_disk_bytes(path: &Path) -> Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetDiskFreeSpaceExW(
            directory_name: *const u16,
            free_bytes_available: *mut u64,
            total_number_of_bytes: *mut u64,
            total_number_of_free_bytes: *mut u64,
        ) -> i32;
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut available = 0_u64;
    let result = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if result == 0 {
        return Err(std::io::Error::last_os_error())
            .context("GetDiskFreeSpaceExW staging directory");
    }
    Ok(available)
}

#[cfg(windows)]
fn open_source_for_seal(path: &Path) -> Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ)
        .open(path)
        .with_context(|| {
            format!(
                "open source with no write/delete sharing {}",
                path.display()
            )
        })
}

#[cfg(unix)]
fn open_source_for_seal(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .with_context(|| format!("open regular source with O_NOFOLLOW {}", path.display()))
}

#[cfg(unix)]
fn stable_file_identity(file: &File) -> Result<String> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata()?;
    Ok(format!(
        "unix-v1:dev={}:ino={}:size={}",
        metadata.dev(),
        metadata.ino(),
        metadata.size()
    ))
}

#[cfg(unix)]
fn stable_path_identity(path: &Path) -> Result<String> {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!("source path stopped naming a regular non-symlink file");
    }
    Ok(format!(
        "unix-v1:dev={}:ino={}:size={}",
        metadata.dev(),
        metadata.ino(),
        metadata.size()
    ))
}

#[cfg(windows)]
fn stable_file_identity(file: &File) -> Result<String> {
    windows_file_identity(file)
}

#[cfg(windows)]
fn stable_path_identity(path: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!("source path stopped naming a regular non-symlink file");
    }
    let file = open_source_for_seal(path)?;
    windows_file_identity(&file)
}

#[cfg(windows)]
fn windows_file_identity(file: &File) -> Result<String> {
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    struct FileTime {
        low_date_time: u32,
        high_date_time: u32,
    }
    #[repr(C)]
    struct ByHandleFileInformation {
        file_attributes: u32,
        creation_time: FileTime,
        last_access_time: FileTime,
        last_write_time: FileTime,
        volume_serial_number: u32,
        file_size_high: u32,
        file_size_low: u32,
        number_of_links: u32,
        file_index_high: u32,
        file_index_low: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFileInformationByHandle(
            file: *mut core::ffi::c_void,
            information: *mut ByHandleFileInformation,
        ) -> i32;
    }

    let mut information = std::mem::MaybeUninit::<ByHandleFileInformation>::uninit();
    let result = unsafe {
        GetFileInformationByHandle(file.as_raw_handle().cast(), information.as_mut_ptr())
    };
    if result == 0 {
        return Err(std::io::Error::last_os_error()).context("GetFileInformationByHandle source");
    }
    let information = unsafe { information.assume_init() };
    let file_index =
        (u64::from(information.file_index_high) << 32) | u64::from(information.file_index_low);
    let file_size =
        (u64::from(information.file_size_high) << 32) | u64::from(information.file_size_low);
    Ok(format!(
        "windows-v1:volume={}:file={}:size={}",
        information.volume_serial_number, file_index, file_size
    ))
}

struct SnapshotCleanup {
    path: PathBuf,
    committed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_core::execution_budget::{
        AuxiliarySlotLimit, AuxiliarySlotRequest, CompositeAdmissionAuthority,
        CompositeAdmissionGrant, CompositeAdmissionRequest, CpuPermitBroker, CpuPermitRequest,
        WorkerLimit,
    };
    use std::fs::OpenOptions;
    #[cfg(target_os = "linux")]
    use std::io::Write;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn staging_cleanup_failure_is_not_silently_discarded() {
        let temporary = tempfile::tempdir().expect("tempdir");
        let not_a_file = temporary.path().join("source-1-1.stage");
        std::fs::create_dir(&not_a_file).expect("create non-file cleanup target");
        let error = remove_snapshot_path(&not_a_file)
            .expect_err("cleanup helper must return a structured failure");
        let message = format!("{error:#}");
        assert!(message.contains("remove private import staging file"));
        assert!(message.contains("source-1-1.stage"));
    }

    fn import_grant() -> CompositeAdmissionGrant {
        let width = WorkerLimit::new(1).expect("one-worker source-snapshot test budget");
        let authority = CompositeAdmissionAuthority::new(
            CpuPermitBroker::new(width),
            AuxiliarySlotLimit::new(crate::core::source_seal_slot_limit())
                .expect("positive source-seal slot limit"),
        );
        authority
            .try_acquire(CompositeAdmissionRequest::new(
                CpuPermitRequest::local(width),
                AuxiliarySlotRequest::One,
            ))
            .expect("acquire source-snapshot test resources")
            .expect("fresh source-snapshot test authority has capacity")
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mutation_after_seal_cancels_snapshot_and_cleans_staging() {
        let _process_signal_guard = crate::core::source_seal::process_signal_test_guard();
        let temporary = tempfile::tempdir().expect("tempdir");
        let source = temporary.path().join("source.csv");
        let staging = temporary.path().join("staging");
        std::fs::write(&source, vec![b'x'; 2 * COPY_BUFFER_BYTES]).expect("write source");
        let (start_tx, start_rx) = mpsc::sync_channel(1);
        let (attempted_tx, attempted_rx) = mpsc::sync_channel(1);
        let writer_path = source.clone();
        let writer = std::thread::spawn(move || {
            start_rx.recv().expect("start mutation");
            attempted_tx.send(()).expect("announce mutation");
            let mut writer = OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&writer_path)
                .expect("writer proceeds after cancelled snapshot releases lease");
            writer.write_all(b"mutated").expect("mutate source");
        });
        let mut started = false;
        let grant = import_grant();
        let error = SourceSnapshot::capture_path_observed(
            &source,
            &staging,
            &ImportLimits::conservative_for_tests(),
            grant
                .auxiliary_slot()
                .expect("source-snapshot grant owns an auxiliary slot"),
            |copied| {
                if copied == 0 && !started {
                    started = true;
                    start_tx.send(()).expect("start writer");
                    attempted_rx
                        .recv_timeout(Duration::from_secs(2))
                        .expect("writer attempted mutation");
                }
            },
        )
        .expect_err("lease break must cancel the snapshot");
        assert!(
            format!("{error:#}").contains("broken") || format!("{error:#}").contains("changed"),
            "{error:#}"
        );
        writer.join().expect("writer thread");
        let leaked = std::fs::read_dir(&staging)
            .expect("staging directory")
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("enumerate staging");
        assert!(leaked.is_empty(), "failed snapshot leaked staging files");
    }

    #[cfg(windows)]
    #[test]
    fn windows_share_mode_rejects_mutation_while_snapshot_succeeds() {
        let temporary = tempfile::tempdir().expect("tempdir");
        let source = temporary.path().join("source.csv");
        let staging = temporary.path().join("staging");
        std::fs::write(&source, vec![b'x'; 2 * COPY_BUFFER_BYTES]).expect("write source");
        let (start_tx, start_rx) = mpsc::sync_channel(1);
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let writer_path = source.clone();
        let writer = std::thread::spawn(move || {
            start_rx.recv().expect("start mutation");
            result_tx
                .send(
                    OpenOptions::new()
                        .write(true)
                        .truncate(true)
                        .open(&writer_path),
                )
                .expect("send writer result");
        });
        let mut started = false;
        let grant = import_grant();
        let snapshot = SourceSnapshot::capture_path_observed(
            &source,
            &staging,
            &ImportLimits::conservative_for_tests(),
            grant
                .auxiliary_slot()
                .expect("source-snapshot grant owns an auxiliary slot"),
            |copied| {
                if copied == 0 && !started {
                    started = true;
                    start_tx.send(()).expect("start writer");
                    let result = result_rx
                        .recv_timeout(Duration::from_secs(2))
                        .expect("writer returned sharing result");
                    assert!(result.is_err(), "Windows mutation unexpectedly opened");
                }
            },
        )
        .expect("snapshot remains valid when Windows blocks the mutation");
        writer.join().expect("writer thread");
        assert_eq!(snapshot.source_size(), (2 * COPY_BUFFER_BYTES) as u64);
    }
}

impl SnapshotCleanup {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            committed: false,
        }
    }
}

impl Drop for SnapshotCleanup {
    fn drop(&mut self) {
        if !self.committed {
            remove_snapshot_path_or_log(&self.path, "failed_snapshot");
        }
    }
}
