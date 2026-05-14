use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use vortex_array::ArrayRef;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBufferMut;
use vortex_file::{OpenOptionsSessionExt, WriteOptionsSessionExt};
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::{RuntimeSession, RuntimeSessionExt};
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

static VORTEX_RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

static VORTEX_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let mut session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>()
        .with_handle(VORTEX_RUNTIME.handle());
    vortex_file::register_default_encodings(&mut session);
    session
});

pub fn vortex_session() -> &'static VortexSession {
    &VORTEX_SESSION
}

pub fn read_vortex_array(path: impl AsRef<Path>) -> Result<ArrayRef> {
    let path = path.as_ref();
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open vortex file {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap vortex file {}", path.display()))?;
    let vortex_file = vortex_session()
        .open_options()
        .open_buffer(mmap)
        .with_context(|| format!("failed to open vortex layout {}", path.display()))?;
    let stream = vortex_file
        .scan()
        .with_context(|| format!("failed to scan vortex file {}", path.display()))?
        .into_array_stream()
        .with_context(|| format!("failed to create vortex stream {}", path.display()))?;
    VORTEX_RUNTIME
        .block_on(stream.read_all())
        .with_context(|| format!("failed to materialize vortex file {}", path.display()))
}

pub fn write_vortex_array(path: impl AsRef<Path>, array: ArrayRef) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create vortex parent directory {}",
                parent.display()
            )
        })?;
    }

    let tmp_path = temp_path_for(path);
    let guard = TempFileGuard::new(tmp_path.clone());
    let mut buffer = ByteBufferMut::empty();
    VORTEX_RUNTIME
        .block_on(
            vortex_session()
                .write_options()
                .write(&mut buffer, array.to_array_stream()),
        )
        .with_context(|| format!("failed to encode vortex file {}", path.display()))?;

    fs::write(&tmp_path, buffer.as_slice()).with_context(|| {
        format!(
            "failed to write temporary vortex file {}",
            tmp_path.display()
        )
    })?;
    atomic_replace_file(&tmp_path, path)?;
    guard.commit();
    Ok(())
}

fn temp_path_for(target_path: &Path) -> PathBuf {
    let file_name = target_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("data.vortex");
    let pid = std::process::id();
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    target_path.with_file_name(format!("{file_name}.tmp-{pid}-{nonce}"))
}

fn atomic_replace_file(src: &Path, dst: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        atomic_replace_file_windows(src, dst)
    }
    #[cfg(not(windows))]
    {
        fs::rename(src, dst).with_context(|| {
            format!(
                "failed to rename temporary vortex file {} -> {}",
                src.display(),
                dst.display()
            )
        })
    }
}

#[cfg(windows)]
fn atomic_replace_file_windows(src: &Path, dst: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, new: *const u16, flags: u32) -> i32;
    }

    let src_wide: Vec<u16> = src.as_os_str().encode_wide().chain(Some(0)).collect();
    let dst_wide: Vec<u16> = dst.as_os_str().encode_wide().chain(Some(0)).collect();
    let ok = unsafe {
        MoveFileExW(
            src_wide.as_ptr(),
            dst_wide.as_ptr(),
            0x0000_0001 | 0x0000_0008,
        )
    };

    if ok == 0 {
        Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "failed to replace vortex file {} -> {}",
                src.display(),
                dst.display()
            )
        })
    } else {
        Ok(())
    }
}

struct TempFileGuard {
    path: PathBuf,
    committed: bool,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            committed: false,
        }
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if !self.committed {
            // Drop-cleanup: file may not exist (NotFound is fine) and we
            // can't return errors from Drop. Log other failures so a
            // permission-denied / locked-file leak surfaces.
            if let Err(err) = fs::remove_file(&self.path) {
                if err.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        target: "forex_data::vortex_io",
                        path = %self.path.display(),
                        error = %err,
                        "TempFileGuard::drop: failed to remove staged file"
                    );
                }
            }
        }
    }
}
