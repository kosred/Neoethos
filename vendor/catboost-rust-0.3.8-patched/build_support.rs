use std::ffi::OsStr;
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
