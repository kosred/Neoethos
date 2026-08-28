use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use neoethos_execution_budget::{X8664V3PreflightErrorV1, require_current_x86_64_v3_v1};

pub const PRIVATE_V3_PAYLOAD_MARKER_V1: &str = ".x86-64-v3";
pub const PUBLIC_LAUNCHER_FAILURE_EXIT_CODE_V1: i32 = 78;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LauncherErrorCodeV1 {
    CpuPreflightRefused,
    CurrentExecutableUnavailable,
    InvalidLauncherPath,
    PrivatePayloadMissing,
    PrivatePayloadLaunchFailed,
    PrivatePayloadTerminatedWithoutExitCode,
}

#[derive(Debug)]
pub struct LauncherErrorV1 {
    kind: LauncherErrorKindV1,
}

#[derive(Debug)]
enum LauncherErrorKindV1 {
    CpuPreflight(X8664V3PreflightErrorV1),
    CurrentExecutable(io::ErrorKind),
    InvalidLauncherPath,
    PrivatePayloadMissing(OsString),
    PrivatePayloadLaunch(io::ErrorKind),
    PrivatePayloadTerminatedWithoutExitCode,
}

impl LauncherErrorV1 {
    pub fn from_cpu_preflight(error: X8664V3PreflightErrorV1) -> Self {
        Self {
            kind: LauncherErrorKindV1::CpuPreflight(error),
        }
    }

    pub const fn code(&self) -> LauncherErrorCodeV1 {
        match &self.kind {
            LauncherErrorKindV1::CpuPreflight(_) => LauncherErrorCodeV1::CpuPreflightRefused,
            LauncherErrorKindV1::CurrentExecutable(_) => {
                LauncherErrorCodeV1::CurrentExecutableUnavailable
            }
            LauncherErrorKindV1::InvalidLauncherPath => LauncherErrorCodeV1::InvalidLauncherPath,
            LauncherErrorKindV1::PrivatePayloadMissing(_) => {
                LauncherErrorCodeV1::PrivatePayloadMissing
            }
            LauncherErrorKindV1::PrivatePayloadLaunch(_) => {
                LauncherErrorCodeV1::PrivatePayloadLaunchFailed
            }
            LauncherErrorKindV1::PrivatePayloadTerminatedWithoutExitCode => {
                LauncherErrorCodeV1::PrivatePayloadTerminatedWithoutExitCode
            }
        }
    }

    pub const fn exit_code(&self) -> i32 {
        PUBLIC_LAUNCHER_FAILURE_EXIT_CODE_V1
    }
}

impl fmt::Display for LauncherErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            LauncherErrorKindV1::CpuPreflight(error) => error.fmt(formatter),
            LauncherErrorKindV1::CurrentExecutable(error_kind) => write!(
                formatter,
                "NEOETHOS_LAUNCHER_V1 status=refused stage=current_executable error={error_kind:?}"
            ),
            LauncherErrorKindV1::InvalidLauncherPath => formatter.write_str(
                "NEOETHOS_LAUNCHER_V1 status=refused stage=payload_path \
                 error=launcher_has_no_file_name",
            ),
            LauncherErrorKindV1::PrivatePayloadMissing(file_name) => write!(
                formatter,
                "NEOETHOS_LAUNCHER_V1 status=refused stage=payload_presence \
                 error=private_payload_missing file={}",
                file_name.to_string_lossy()
            ),
            LauncherErrorKindV1::PrivatePayloadLaunch(error_kind) => write!(
                formatter,
                "NEOETHOS_LAUNCHER_V1 status=refused stage=payload_launch error={error_kind:?}"
            ),
            LauncherErrorKindV1::PrivatePayloadTerminatedWithoutExitCode => formatter.write_str(
                "NEOETHOS_LAUNCHER_V1 status=refused stage=payload_exit \
                 error=terminated_without_exit_code",
            ),
        }
    }
}

impl Error for LauncherErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            LauncherErrorKindV1::CpuPreflight(error) => Some(error),
            _ => None,
        }
    }
}

pub fn private_v3_payload_path_v1(public_launcher_path: &Path) -> Result<PathBuf, LauncherErrorV1> {
    let file_stem = public_launcher_path
        .file_stem()
        .filter(|file_stem| !file_stem.is_empty())
        .ok_or(LauncherErrorV1 {
            kind: LauncherErrorKindV1::InvalidLauncherPath,
        })?;
    let mut payload_file_name = file_stem.to_os_string();
    payload_file_name.push(PRIVATE_V3_PAYLOAD_MARKER_V1);
    if let Some(extension) = public_launcher_path.extension() {
        payload_file_name.push(".");
        payload_file_name.push(extension);
    }
    Ok(public_launcher_path.with_file_name(payload_file_name))
}

pub fn run_public_launcher_v1() -> Result<i32, LauncherErrorV1> {
    require_current_x86_64_v3_v1().map_err(LauncherErrorV1::from_cpu_preflight)?;

    let public_launcher_path = std::env::current_exe().map_err(|error| LauncherErrorV1 {
        kind: LauncherErrorKindV1::CurrentExecutable(error.kind()),
    })?;
    let payload_path = private_v3_payload_path_v1(&public_launcher_path)?;
    if !payload_path.is_file() {
        return Err(LauncherErrorV1 {
            kind: LauncherErrorKindV1::PrivatePayloadMissing(
                payload_path
                    .file_name()
                    .map_or_else(OsString::new, std::ffi::OsStr::to_os_string),
            ),
        });
    }

    let status = Command::new(&payload_path)
        .args(std::env::args_os().skip(1))
        .status()
        .map_err(|error| LauncherErrorV1 {
            kind: LauncherErrorKindV1::PrivatePayloadLaunch(error.kind()),
        })?;
    status.code().ok_or(LauncherErrorV1 {
        kind: LauncherErrorKindV1::PrivatePayloadTerminatedWithoutExitCode,
    })
}
