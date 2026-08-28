use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use vortex_array::dtype::{DType, FieldName, FieldNames};
use vortex_array::expr::{root, select};
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::{ArrayRef, ToCanonical};
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

/// Hard ceiling for memory buffered by Vortex's incremental file writer.
///
/// This is not the input-batch limit: callers must separately bound each
/// `ArrayRef`. It prevents a future Vortex strategy change from silently
/// turning an incremental import into an all-file in-memory encode.
pub const MAX_STREAMING_VORTEX_BUFFER_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VortexWriteStats {
    pub row_count: u64,
    pub file_size: u64,
    pub max_buffered_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VortexFileMetadata {
    row_count: u64,
    dtype: DType,
    footer_bytes: usize,
    max_segment_bytes: usize,
}

impl VortexFileMetadata {
    pub const fn row_count(&self) -> u64 {
        self.row_count
    }

    pub const fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub const fn footer_bytes(&self) -> usize {
        self.footer_bytes
    }

    pub const fn max_segment_bytes(&self) -> usize {
        self.max_segment_bytes
    }
}

pub fn vortex_session() -> &'static VortexSession {
    &VORTEX_SESSION
}

pub fn read_vortex_array(path: impl AsRef<Path>) -> Result<ArrayRef> {
    let path = path.as_ref();
    // Note — convert in-library panics on corrupt vortex
    // files into clean `Err` results. The on-disk format has its own
    // structural validity (mmap framing, encoding headers), but bit-rot
    // inside a column's raw payload — a flipped exponent in an f64, a
    // negative i64 length-prefix inside a variable-length string — can
    // make vortex internals panic instead of returning a structured
    // error. A panic in the read path would (pre-fix) bring down the
    // background loader thread; with the V0.4 background helper
    // (`app_services/trading/background.rs`) it now becomes a
    // `BackgroundTaskPanic` event, but that still leaves the operator
    // with a cryptic panic message. Wrap the body in `catch_unwind` so
    // corrupt data surfaces as `Err("corrupt vortex file ...")` and the
    // bootstrap pipeline can re-fetch from cTrader instead of refusing
    // to start.
    let path_owned = path.to_path_buf();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        read_vortex_array_inner(path)
    }));
    match result {
        Ok(Ok(arr)) => Ok(arr),
        Ok(Err(err)) => Err(err),
        Err(panic_payload) => {
            let message = if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else {
                "<non-string panic payload>".to_string()
            };
            tracing::error!(
                target: "neoethos_data::vortex_io",
                path = %path_owned.display(),
                panic = %message,
                "vortex parser panicked on read; treating file as corrupt"
            );
            Err(anyhow::anyhow!(
                "corrupt vortex file {}: parser panicked ({}). Delete or re-fetch.",
                path_owned.display(),
                message
            ))
        }
    }
}

/// Read only the requested fields and row interval through Vortex scan
/// pushdown. The returned array is materialized, but unrequested columns and
/// rows are never materialized by this API.
pub fn read_vortex_projection_range(
    path: impl AsRef<Path>,
    fields: &[&str],
    row_range: Range<u64>,
) -> Result<ArrayRef> {
    let path = path.as_ref();
    if fields.is_empty() {
        anyhow::bail!("Vortex projection must contain at least one field");
    }
    if row_range.start > row_range.end {
        anyhow::bail!(
            "invalid Vortex row range {}..{}",
            row_range.start,
            row_range.end
        );
    }
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open vortex file {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap vortex file {}", path.display()))?;
    let vortex_file = vortex_session()
        .open_options()
        .open_buffer(mmap)
        .with_context(|| format!("failed to open vortex layout {}", path.display()))?;
    let field_names = fields
        .iter()
        .map(|field| FieldName::from(*field))
        .collect::<FieldNames>();
    let stream = vortex_file
        .scan()
        .with_context(|| format!("failed to scan vortex file {}", path.display()))?
        .with_projection(select(field_names, root()))
        .with_row_range(row_range)
        .into_array_stream()
        .with_context(|| format!("failed to create vortex stream {}", path.display()))?;
    VORTEX_RUNTIME
        .block_on(stream.read_all())
        .with_context(|| format!("failed to materialize vortex projection {}", path.display()))
}

/// Read only the file footer and return its declared row count.
pub fn read_vortex_row_count(path: impl AsRef<Path>) -> Result<u64> {
    let path = path.as_ref();
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open vortex file {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap vortex file {}", path.display()))?;
    let vortex_file = vortex_session()
        .open_options()
        .open_buffer(mmap)
        .with_context(|| format!("failed to open vortex layout {}", path.display()))?;
    Ok(vortex_file.row_count())
}

/// Read typed footer metadata without materializing any data column.
pub fn read_vortex_file_metadata(path: impl AsRef<Path>) -> Result<VortexFileMetadata> {
    let path = path.as_ref();
    let vortex_file = VORTEX_RUNTIME
        .block_on(vortex_session().open_options().open_path(path))
        .with_context(|| format!("failed to open Vortex footer {}", path.display()))?;
    let footer_bytes = vortex_file
        .footer()
        .approx_byte_size()
        .context("Vortex reader did not report a bounded footer size")?;
    let max_segment_bytes = vortex_file
        .footer()
        .segment_map()
        .iter()
        .map(|segment| segment.length as usize)
        .max()
        .unwrap_or(0);
    Ok(VortexFileMetadata {
        row_count: vortex_file.row_count(),
        dtype: vortex_file.dtype().clone(),
        footer_bytes,
        max_segment_bytes,
    })
}

/// Materialize one bounded canonical OHLCV row interval.
pub fn read_vortex_ohlcv_projection_range(
    path: impl AsRef<Path>,
    include_volume: bool,
    row_range: Range<u64>,
) -> Result<crate::Ohlcv> {
    let fields_with_volume = ["timestamp", "open", "high", "low", "close", "volume"];
    let fields_without_volume = ["timestamp", "open", "high", "low", "close"];
    let fields: &[&str] = if include_volume {
        &fields_with_volume
    } else {
        &fields_without_volume
    };
    let array = read_vortex_projection_range(path, fields, row_range)?;
    crate::vortex_array_to_ohlcv(array)
}

/// Project one non-null i64 field over a bounded row range.
pub fn read_vortex_i64_projection_range(
    path: impl AsRef<Path>,
    field: &str,
    row_range: Range<u64>,
) -> Result<Vec<i64>> {
    let array = read_vortex_projection_range(path, &[field], row_range)?;
    let struct_array = array.to_struct();
    let projected = struct_array
        .unmasked_field_by_name(field)
        .with_context(|| format!("projected Vortex field {field} is missing"))?;
    if !projected
        .all_valid()
        .with_context(|| format!("failed to inspect projected {field} validity"))?
    {
        anyhow::bail!("projected Vortex field {field} contains null values");
    }
    Ok(projected.to_primitive().as_slice::<i64>().to_vec())
}

fn read_vortex_array_inner(path: &Path) -> Result<ArrayRef> {
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
    write_vortex_chunks(path, std::iter::once(array)).map(|_| ())
}

/// Incrementally encode bounded same-schema chunks into one Vortex file.
///
/// The old implementation first encoded the complete file into
/// `ByteBufferMut` and only then wrote it to disk, doubling peak memory for a
/// large import. This wrapper writes directly to a real staged file and keeps
/// only Vortex's bounded layout buffers resident.
pub fn write_vortex_chunks(
    path: impl AsRef<Path>,
    chunks: impl IntoIterator<Item = ArrayRef>,
) -> Result<VortexWriteStats> {
    write_vortex_chunks_fallible(path, chunks.into_iter().map(Ok::<_, anyhow::Error>))
}

/// Incrementally encode fallible chunks without collecting them first.
pub fn write_vortex_chunks_fallible(
    path: impl AsRef<Path>,
    chunks: impl IntoIterator<Item = Result<ArrayRef>>,
) -> Result<VortexWriteStats> {
    write_vortex_chunks_fallible_limited(path, chunks, u64::MAX)
}

/// Incremental Vortex writer with a hard on-write file-size ceiling. The
/// bound is enforced before bytes reach the filesystem, rather than checking
/// only after a completed candidate has already consumed the disk.
pub fn write_vortex_chunks_fallible_limited(
    path: impl AsRef<Path>,
    chunks: impl IntoIterator<Item = Result<ArrayRef>>,
    max_file_bytes: u64,
) -> Result<VortexWriteStats> {
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
    let mut chunks = chunks.into_iter();
    let first = chunks
        .next()
        .context("cannot write a Vortex file from zero chunks")??;
    let dtype = first.dtype().clone();
    let file = fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&tmp_path)
        .with_context(|| format!("failed to create staged Vortex file {}", tmp_path.display()))?;
    let mut file = BoundedFileWriter::new(file, max_file_bytes);
    let mut writer = vortex_session()
        .write_options()
        .blocking(&*VORTEX_RUNTIME)
        .writer(&mut file, dtype.clone());
    let mut max_buffered_bytes = 0_u64;
    for chunk in std::iter::once(Ok(first)).chain(chunks) {
        let chunk = chunk.context("source reader failed before producing a Vortex chunk")?;
        if chunk.dtype() != &dtype {
            anyhow::bail!(
                "Vortex chunk dtype mismatch: expected {}, got {}",
                dtype,
                chunk.dtype()
            );
        }
        writer
            .push(chunk)
            .with_context(|| format!("failed to encode Vortex chunk for {}", path.display()))?;
        max_buffered_bytes = max_buffered_bytes.max(writer.buffered_bytes());
        if max_buffered_bytes > MAX_STREAMING_VORTEX_BUFFER_BYTES {
            anyhow::bail!(
                "Vortex writer buffered {max_buffered_bytes} bytes, above the hard limit of \
                 {MAX_STREAMING_VORTEX_BUFFER_BYTES}"
            );
        }
    }
    let summary = writer
        .finish()
        .with_context(|| format!("failed to finish Vortex file {}", path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush staged Vortex file {}", tmp_path.display()))?;
    file.sync_all()
        .with_context(|| format!("failed to fsync staged Vortex file {}", tmp_path.display()))?;
    let stats = VortexWriteStats {
        row_count: summary.row_count(),
        file_size: summary.size(),
        max_buffered_bytes,
    };
    drop(file);
    atomic_replace_file(&tmp_path, path)?;
    sync_parent_directory(path)?;
    guard.commit();
    Ok(stats)
}

struct BoundedFileWriter {
    inner: fs::File,
    max_file_bytes: u64,
    position: u64,
}

impl BoundedFileWriter {
    const fn new(inner: fs::File, max_file_bytes: u64) -> Self {
        Self {
            inner,
            max_file_bytes,
            position: 0,
        }
    }

    fn sync_all(&self) -> std::io::Result<()> {
        self.inner.sync_all()
    }

    fn require_end(&self, write_bytes: usize) -> std::io::Result<()> {
        let write_bytes = u64::try_from(write_bytes).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "import limit CandidateBytes write length does not fit u64",
            )
        })?;
        let end = self.position.checked_add(write_bytes).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "import limit CandidateBytes arithmetic overflow",
            )
        })?;
        if end > self.max_file_bytes {
            return Err(std::io::Error::other(format!(
                "import limit CandidateBytes exceeded: attempted end {end}, maximum {}",
                self.max_file_bytes
            )));
        }
        Ok(())
    }
}

impl Write for BoundedFileWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.require_end(buffer.len())?;
        let written = self.inner.write(buffer)?;
        self.position = self
            .position
            .checked_add(written as u64)
            .ok_or_else(|| std::io::Error::other("candidate write position overflow"))?;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for BoundedFileWriter {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        let next = self.inner.seek(position)?;
        if next > self.max_file_bytes {
            return Err(std::io::Error::other(format!(
                "import limit CandidateBytes exceeded by seek: position {next}, maximum {}",
                self.max_file_bytes
            )));
        }
        self.position = next;
        Ok(next)
    }
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

pub(crate) fn atomic_replace_file(src: &Path, dst: &Path) -> Result<()> {
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

pub(crate) fn sync_parent_directory(path: &Path) -> Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    #[cfg(unix)]
    {
        fs::File::open(parent)
            .with_context(|| format!("failed to open directory {} for fsync", parent.display()))?
            .sync_all()
            .with_context(|| format!("failed to fsync directory {}", parent.display()))
    }
    #[cfg(windows)]
    {
        sync_directory_windows(parent)
    }
}

#[cfg(windows)]
fn sync_directory_windows(path: &Path) -> Result<()> {
    type Handle = *mut core::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileW(
            file_name: *const u16,
            desired_access: u32,
            share_mode: u32,
            security_attributes: *mut core::ffi::c_void,
            creation_disposition: u32,
            flags_and_attributes: u32,
            template_file: Handle,
        ) -> Handle;
        fn FlushFileBuffers(file: Handle) -> i32;
        fn CloseHandle(object: Handle) -> i32;
    }

    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;
    const FILE_SHARE_DELETE: u32 = 0x0000_0004;
    const OPEN_EXISTING: u32 = 3;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    let wide = windows_verbatim_wide(path)?;
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            std::ptr::null_mut(),
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS,
            std::ptr::null_mut(),
        )
    };
    if handle == (-1_isize as Handle) {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to open directory {} for fsync", path.display()));
    }
    let flush_result = unsafe { FlushFileBuffers(handle) };
    let flush_error = (flush_result == 0).then(std::io::Error::last_os_error);
    unsafe {
        CloseHandle(handle);
    }
    if let Some(error) = flush_error {
        // Windows commonly returns ERROR_ACCESS_DENIED for
        // FlushFileBuffers on a directory handle, even when opened with
        // FILE_FLAG_BACKUP_SEMANTICS. File contents are fsynced before every
        // transition and atomic_replace_file_windows uses
        // MOVEFILE_WRITE_THROUGH, which is the documented durable rename
        // primitive on this platform. Do not turn that documented directory
        // limitation into a false publication failure.
        if error.raw_os_error() == Some(5) {
            Ok(())
        } else {
            Err(error).with_context(|| format!("failed to fsync directory {}", path.display()))
        }
    } else {
        Ok(())
    }
}

#[cfg(windows)]
fn atomic_replace_file_windows(src: &Path, dst: &Path) -> Result<()> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, new: *const u16, flags: u32) -> i32;
    }

    let src_wide = windows_verbatim_wide(src)?;
    let dst_wide = windows_verbatim_wide(dst)?;
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

#[cfg(windows)]
fn windows_verbatim_wide(path: &Path) -> Result<Vec<u16>> {
    use std::os::windows::ffi::OsStrExt;

    const SEPARATOR: u16 = b'\\' as u16;
    const VERBATIM_PREFIX: [u16; 4] = [SEPARATOR, SEPARATOR, b'?' as u16, SEPARATOR];
    const UNC_PREFIX: [u16; 8] = [
        SEPARATOR,
        SEPARATOR,
        b'?' as u16,
        SEPARATOR,
        b'U' as u16,
        b'N' as u16,
        b'C' as u16,
        SEPARATOR,
    ];

    let absolute = std::path::absolute(path)
        .with_context(|| format!("failed to make Windows path absolute: {}", path.display()))?;
    let raw: Vec<u16> = absolute.as_os_str().encode_wide().collect();
    if raw.contains(&0) {
        anyhow::bail!("Windows path contains an embedded NUL: {}", path.display());
    }
    let mut wide = if raw.starts_with(&VERBATIM_PREFIX) {
        raw
    } else if raw.starts_with(&[SEPARATOR, SEPARATOR]) {
        let mut value = Vec::with_capacity(UNC_PREFIX.len() + raw.len() - 2 + 1);
        value.extend_from_slice(&UNC_PREFIX);
        value.extend_from_slice(&raw[2..]);
        value
    } else {
        let mut value = Vec::with_capacity(VERBATIM_PREFIX.len() + raw.len() + 1);
        value.extend_from_slice(&VERBATIM_PREFIX);
        value.extend_from_slice(&raw);
        value
    };
    wide.push(0);
    Ok(wide)
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
            if let Err(err) = fs::remove_file(&self.path)
                && err.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!(
                    target: "neoethos_data::vortex_io",
                    path = %self.path.display(),
                    error = %err,
                    "TempFileGuard::drop: failed to remove staged file"
                );
            }
        }
    }
}
