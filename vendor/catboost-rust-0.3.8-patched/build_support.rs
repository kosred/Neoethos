use std::ffi::OsStr;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

pub fn cargo_profile_output_dir(out_dir: &Path) -> Result<PathBuf, String> {
    if out_dir.file_name() != Some(OsStr::new("out")) {
        return Err(format!(
            "OUT_DIR must end in `out`, got `{}`",
            out_dir.display()
        ));
    }

    let package_build_dir = out_dir.parent().ok_or_else(|| {
        format!(
            "OUT_DIR has no package build directory: `{}`",
            out_dir.display()
        )
    })?;
    let cargo_build_dir = package_build_dir.parent().ok_or_else(|| {
        format!(
            "OUT_DIR has no Cargo build directory: `{}`",
            out_dir.display()
        )
    })?;
    if cargo_build_dir.file_name() != Some(OsStr::new("build")) {
        return Err(format!(
            "OUT_DIR is not in Cargo's `<profile>/build/<package>/out` layout: `{}`",
            out_dir.display()
        ));
    }

    let profile_dir = cargo_build_dir.parent().ok_or_else(|| {
        format!(
            "OUT_DIR has no Cargo profile directory: `{}`",
            out_dir.display()
        )
    })?;
    Ok(profile_dir.to_path_buf())
}

pub fn stage_selected_runtime(source: &Path, destination: &Path) {
    if !source.is_file() {
        panic!(
            "catboost-rust: selected runtime is missing: {}",
            source.display()
        );
    }
    let source_len = source
        .metadata()
        .unwrap_or_else(|error| {
            panic!(
                "catboost-rust: failed to inspect selected runtime {}: {error}",
                source.display()
            )
        })
        .len();
    if source_len == 0 {
        panic!(
            "catboost-rust: selected runtime is empty: {}",
            source.display()
        );
    }
    // A running Windows application may hold its DLL open without write
    // sharing. Keep it in place only after comparing every byte with the
    // selected build runtime; existence, length or filename alone is not proof.
    let already_staged =
        same_runtime_contents(source, destination, source_len).unwrap_or_else(|error| {
            panic!(
                "catboost-rust: failed to compare selected runtime {} with {}: {error}",
                source.display(),
                destination.display()
            )
        });
    if already_staged {
        eprintln!(
            "INFO catboost-rust: verified identical selected runtime {} -> {} ({source_len} bytes); no replacement needed",
            source.display(),
            destination.display()
        );
        return;
    }
    let copied_bytes = fs::copy(source, destination).unwrap_or_else(|error| {
        panic!(
            "catboost-rust: failed to stage selected runtime {} as {}: {error}",
            source.display(),
            destination.display()
        )
    });
    if copied_bytes != source_len {
        panic!(
            "catboost-rust: staged {copied_bytes} of {source_len} bytes from {} to {}",
            source.display(),
            destination.display()
        );
    }
    eprintln!(
        "INFO catboost-rust: staged selected runtime {} -> {} ({source_len} bytes)",
        source.display(),
        destination.display()
    );
}

fn same_runtime_contents(source: &Path, destination: &Path, source_len: u64) -> io::Result<bool> {
    let mut destination_file = match fs::File::open(destination) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    let destination_metadata = destination_file.metadata()?;
    if !destination_metadata.is_file() || destination_metadata.len() != source_len {
        return Ok(false);
    }
    let mut source_file = fs::File::open(source)?;
    let mut source_chunk = [0_u8; 64 * 1024];
    let mut destination_chunk = [0_u8; 64 * 1024];
    let mut remaining = source_len;
    while remaining != 0 {
        let count = remaining.min(source_chunk.len() as u64) as usize;
        source_file.read_exact(&mut source_chunk[..count])?;
        destination_file.read_exact(&mut destination_chunk[..count])?;
        if source_chunk[..count] != destination_chunk[..count] {
            return Ok(false);
        }
        remaining -= count as u64;
    }
    // Reject a file that grew during the comparison as well as any prefix-only
    // match. Both files must end at the validated selected-runtime length.
    let mut extra = [0_u8; 1];
    Ok(source_file.read(&mut extra)? == 0 && destination_file.read(&mut extra)? == 0)
}
