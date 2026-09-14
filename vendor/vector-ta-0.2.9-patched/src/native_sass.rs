use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArchRequestSource {
    ExplicitList,
    DetectedVisibleDevices,
}

impl ArchRequestSource {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitList => "explicit-cuda-archs",
            Self::DetectedVisibleDevices => "auto-detected-visible-gpus",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeArchPlan {
    pub(crate) source: ArchRequestSource,
    pub(crate) architectures: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct NativeArtifact<'a> {
    pub(crate) stem: &'a str,
    pub(crate) arch: u32,
    pub(crate) bytes: &'a [u8],
}

impl<'a> NativeArtifact<'a> {
    pub(crate) const fn new(stem: &'a str, arch: u32, bytes: &'a [u8]) -> Self {
        Self { stem, arch, bytes }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CuobjdumpReport<'a> {
    pub(crate) list_ptx_succeeded: bool,
    pub(crate) list_ptx_stdout: &'a str,
    pub(crate) list_ptx_stderr: &'a str,
    pub(crate) dump_sass_succeeded: bool,
    pub(crate) dump_sass_stdout: &'a str,
    pub(crate) dump_sass_stderr: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct VerifiedNativeCubin {
    pub(crate) arch: u32,
    pub(crate) byte_len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeSassError {
    DuplicateArtifact {
        stem: String,
        arch: u32,
    },
    MissingArtifact {
        stem: String,
        arch: u32,
    },
    UnexpectedArtifact {
        stem: String,
        arch: u32,
    },
    EmptyArtifact {
        stem: String,
        arch: u32,
    },
    NotElfCubin {
        arch: u32,
    },
    CuobjdumpInspectionFailed {
        operation: &'static str,
        arch: u32,
        stdout: String,
        stderr: String,
    },
    EmbeddedPtx {
        arch: u32,
        listing: String,
    },
    WrongSassArchitecture {
        expected: u32,
        found: Vec<u32>,
    },
    InvalidDeviceCapability {
        major: i32,
        minor: i32,
    },
    MissingKernel {
        stem: String,
    },
    MissingExactArchitecture {
        stem: String,
        requested: u32,
        available: Vec<u32>,
    },
}

impl fmt::Display for NativeSassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateArtifact { stem, arch } => {
                write!(f, "duplicate native cubin for {stem} sm_{arch}")
            }
            Self::MissingArtifact { stem, arch } => {
                write!(f, "missing native cubin for {stem} sm_{arch}")
            }
            Self::UnexpectedArtifact { stem, arch } => {
                write!(f, "unexpected native cubin for {stem} sm_{arch}")
            }
            Self::EmptyArtifact { stem, arch } => {
                write!(f, "native cubin for {stem} sm_{arch} is empty")
            }
            Self::NotElfCubin { arch } => {
                write!(f, "native sm_{arch} artifact is not an ELF cubin")
            }
            Self::CuobjdumpInspectionFailed {
                operation,
                arch,
                stdout,
                stderr,
            } => write!(
                f,
                "cuobjdump {operation} failed for sm_{arch}; stdout={stdout:?}; stderr={stderr:?}"
            ),
            Self::EmbeddedPtx { arch, listing } => write!(
                f,
                "native sm_{arch} cubin contains a PTX payload: {listing:?}"
            ),
            Self::WrongSassArchitecture { expected, found } => write!(
                f,
                "native cubin expected only sm_{expected} SASS but cuobjdump reported {found:?}"
            ),
            Self::InvalidDeviceCapability { major, minor } => {
                write!(f, "invalid device compute capability {major}.{minor}")
            }
            Self::MissingKernel { stem } => {
                write!(f, "native cubin registry has no kernel named {stem:?}")
            }
            Self::MissingExactArchitecture {
                stem,
                requested,
                available,
            } => write!(
                f,
                "native cubin registry has no exact sm_{requested} artifact for {stem:?}; available exact architectures: {available:?}"
            ),
        }
    }
}

impl std::error::Error for NativeSassError {}

pub(crate) fn native_cubin_filename(stem: &str, arch: u32) -> String {
    format!("{stem}_sm{arch}.cubin")
}

pub(crate) fn validate_native_manifest(
    records: &[NativeArtifact<'_>],
    expected_stems: &[&str],
    expected_architectures: &[u32],
) -> Result<(), NativeSassError> {
    let expected_stems: BTreeSet<&str> = expected_stems.iter().copied().collect();
    let expected_architectures: BTreeSet<u32> = expected_architectures.iter().copied().collect();
    let mut seen = BTreeSet::new();

    for record in records {
        if !expected_stems.contains(record.stem) || !expected_architectures.contains(&record.arch) {
            return Err(NativeSassError::UnexpectedArtifact {
                stem: record.stem.to_owned(),
                arch: record.arch,
            });
        }
        if record.bytes.is_empty() {
            return Err(NativeSassError::EmptyArtifact {
                stem: record.stem.to_owned(),
                arch: record.arch,
            });
        }
        if !seen.insert((record.stem, record.arch)) {
            return Err(NativeSassError::DuplicateArtifact {
                stem: record.stem.to_owned(),
                arch: record.arch,
            });
        }
    }

    for stem in expected_stems {
        for &arch in &expected_architectures {
            if !seen.contains(&(stem, arch)) {
                return Err(NativeSassError::MissingArtifact {
                    stem: stem.to_owned(),
                    arch,
                });
            }
        }
    }
    Ok(())
}

fn reported_sass_architectures(output: &str) -> Vec<u32> {
    let mut found = BTreeSet::new();
    let mut remaining = output;
    while let Some(offset) = remaining.find("sm_") {
        let suffix = &remaining[offset + 3..];
        let digits: String = suffix
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect();
        if let Ok(arch) = digits.parse::<u32>() {
            found.insert(arch);
        }
        remaining = suffix;
    }
    found.into_iter().collect()
}

pub(crate) fn verify_native_cubin(
    expected_arch: u32,
    artifact_path: &Path,
    artifact: &[u8],
    report: CuobjdumpReport<'_>,
) -> Result<VerifiedNativeCubin, NativeSassError> {
    if !artifact.starts_with(b"\x7fELF") {
        return Err(NativeSassError::NotElfCubin {
            arch: expected_arch,
        });
    }
    if !report.list_ptx_succeeded {
        return Err(NativeSassError::CuobjdumpInspectionFailed {
            operation: "--list-ptx",
            arch: expected_arch,
            stdout: report.list_ptx_stdout.to_owned(),
            stderr: report.list_ptx_stderr.to_owned(),
        });
    }
    if !list_ptx_proves_absence(
        artifact_path,
        report.list_ptx_stdout,
        report.list_ptx_stderr,
    ) {
        let ptx_listing = match (
            report.list_ptx_stdout.is_empty(),
            report.list_ptx_stderr.is_empty(),
        ) {
            (false, false) => format!(
                "stdout={:?}; stderr={:?}",
                report.list_ptx_stdout, report.list_ptx_stderr
            ),
            (false, true) => report.list_ptx_stdout.to_owned(),
            (true, false) => report.list_ptx_stderr.to_owned(),
            (true, true) => String::new(),
        };
        return Err(NativeSassError::EmbeddedPtx {
            arch: expected_arch,
            listing: ptx_listing,
        });
    }
    if !report.dump_sass_succeeded {
        return Err(NativeSassError::CuobjdumpInspectionFailed {
            operation: "--dump-sass",
            arch: expected_arch,
            stdout: report.dump_sass_stdout.to_owned(),
            stderr: report.dump_sass_stderr.to_owned(),
        });
    }
    let found = reported_sass_architectures(report.dump_sass_stdout);
    if found.as_slice() != [expected_arch] {
        return Err(NativeSassError::WrongSassArchitecture {
            expected: expected_arch,
            found,
        });
    }

    Ok(VerifiedNativeCubin {
        arch: expected_arch,
        byte_len: artifact.len(),
    })
}

fn list_ptx_proves_absence(artifact_path: &Path, stdout: &str, stderr: &str) -> bool {
    // CUDA versions emit this informational line on either stream. Accept
    // exactly one complete, path-bound diagnostic, never mixed output or PTX.
    let output = match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => return true,
        (false, true) => stdout,
        (true, false) => stderr,
        (false, false) => return false,
    };

    let diagnostic = if let Some(line) = output.strip_suffix("\r\n") {
        line
    } else if let Some(line) = output.strip_suffix('\n') {
        line
    } else {
        output
    };
    if diagnostic.contains('\r') || diagnostic.contains('\n') {
        return false;
    }

    diagnostic
        == format!(
            "cuobjdump info    : No PTX file found to extract from '{}'. You may try with -all option.",
            artifact_path.display()
        )
}

pub(crate) fn select_exact_native_cubin<'a>(
    stem: &str,
    device_major: i32,
    device_minor: i32,
    records: &'a [NativeArtifact<'a>],
) -> Result<&'a [u8], NativeSassError> {
    if device_major <= 0 || !(0..=9).contains(&device_minor) {
        return Err(NativeSassError::InvalidDeviceCapability {
            major: device_major,
            minor: device_minor,
        });
    }
    let requested = (device_major as u32)
        .checked_mul(10)
        .and_then(|major| major.checked_add(device_minor as u32))
        .ok_or(NativeSassError::InvalidDeviceCapability {
            major: device_major,
            minor: device_minor,
        })?;

    let stem_records: Vec<&NativeArtifact<'_>> = records
        .iter()
        .filter(|record| record.stem == stem)
        .collect();
    if stem_records.is_empty() {
        return Err(NativeSassError::MissingKernel {
            stem: stem.to_owned(),
        });
    }
    let available: Vec<u32> = stem_records
        .iter()
        .map(|record| record.arch)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let exact: Vec<&NativeArtifact<'_>> = stem_records
        .into_iter()
        .filter(|record| record.arch == requested)
        .collect();
    match exact.as_slice() {
        [record] if !record.bytes.is_empty() => Ok(record.bytes),
        [record] => Err(NativeSassError::EmptyArtifact {
            stem: record.stem.to_owned(),
            arch: record.arch,
        }),
        [] => Err(NativeSassError::MissingExactArchitecture {
            stem: stem.to_owned(),
            requested,
            available,
        }),
        _ => Err(NativeSassError::DuplicateArtifact {
            stem: stem.to_owned(),
            arch: requested,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ELF: &[u8] = b"\x7fELFsynthetic-parser-fixture";

    fn report<'a>(stdout: &'a str, stderr: &'a str) -> CuobjdumpReport<'a> {
        CuobjdumpReport {
            list_ptx_succeeded: true,
            list_ptx_stdout: stdout,
            list_ptx_stderr: stderr,
            dump_sass_succeeded: true,
            dump_sass_stdout: "code for sm_86\n",
            dump_sass_stderr: "",
        }
    }

    fn no_ptx_message(path: &Path) -> String {
        format!(
            "cuobjdump info    : No PTX file found to extract from '{}'. You may try with -all option.",
            path.display()
        )
    }

    #[test]
    fn no_ptx_listing_accepts_empty_successful_inspection() {
        assert_eq!(
            verify_native_cubin(86, Path::new("native.cubin"), ELF, report("", "")),
            Ok(VerifiedNativeCubin {
                arch: 86,
                byte_len: ELF.len()
            })
        );
    }

    #[test]
    fn no_ptx_listing_accepts_exact_info_on_either_stream() {
        for path in [
            Path::new(r"C:\CUDA artifact directory\exact_sm86.cubin"),
            Path::new("/tmp/CUDA artifact directory/exact_sm86.cubin"),
        ] {
            for ending in ["", "\n", "\r\n"] {
                let message = no_ptx_message(path) + ending;
                for inspection in [report(&message, ""), report("", &message)] {
                    assert!(verify_native_cubin(86, path, ELF, inspection).is_ok());
                }
            }
        }
    }

    #[test]
    fn no_ptx_listing_rejects_payload_unknown_and_mixed_streams() {
        let path = Path::new("native.cubin");
        let message = no_ptx_message(path);
        for payload in [
            "PTX file    1: native.sm_86.ptx\n",
            ".version 8.0\n.target sm_86\n",
            "unknown tool output",
            " ",
        ] {
            let mixed = format!("{message}\n{payload}");
            for inspection in [
                report(payload, ""),
                report("", payload),
                report(&mixed, ""),
                report("", &mixed),
                report(&message, payload),
                report(payload, &message),
            ] {
                assert!(matches!(
                    verify_native_cubin(86, path, ELF, inspection),
                    Err(NativeSassError::EmbeddedPtx { .. })
                ));
            }
        }
        assert!(matches!(
            verify_native_cubin(86, path, ELF, report(&message, &message)),
            Err(NativeSassError::EmbeddedPtx { .. })
        ));
    }

    #[test]
    fn no_ptx_listing_rejects_rebound_malformed_and_injected_paths() {
        let path = Path::new("native.cubin");
        let message = no_ptx_message(path);
        for malformed in [
            no_ptx_message(Path::new("different.cubin")),
            message.replace("info    :", "info :"),
            message.replace("'native.cubin'", "native.cubin"),
            format!("{message} extra"),
            format!("\n{message}"),
            format!("{message}\n\n"),
            format!("{message}\r"),
            format!("{message}\n{message}"),
        ] {
            for inspection in [report(&malformed, ""), report("", &malformed)] {
                assert!(matches!(
                    verify_native_cubin(86, path, ELF, inspection),
                    Err(NativeSassError::EmbeddedPtx { .. })
                ));
            }
        }
        for injected in [Path::new("native\n.cubin"), Path::new("native\r.cubin")] {
            let message = no_ptx_message(injected);
            assert!(matches!(
                verify_native_cubin(86, injected, ELF, report(&message, "")),
                Err(NativeSassError::EmbeddedPtx { .. })
            ));
        }
    }

    #[test]
    fn no_ptx_listing_does_not_override_tool_or_sass_failure() {
        let path = Path::new("native.cubin");
        let message = no_ptx_message(path);
        let mut inspection = report(&message, "");
        inspection.list_ptx_succeeded = false;
        assert!(matches!(
            verify_native_cubin(86, path, ELF, inspection),
            Err(NativeSassError::CuobjdumpInspectionFailed {
                operation: "--list-ptx",
                ..
            })
        ));
        inspection = report(&message, "");
        inspection.dump_sass_succeeded = false;
        assert!(matches!(
            verify_native_cubin(86, path, ELF, inspection),
            Err(NativeSassError::CuobjdumpInspectionFailed {
                operation: "--dump-sass",
                ..
            })
        ));
        inspection = report(&message, "");
        inspection.dump_sass_stdout = "code for sm_89\n";
        assert!(matches!(
            verify_native_cubin(86, path, ELF, inspection),
            Err(NativeSassError::WrongSassArchitecture { .. })
        ));
        assert!(matches!(
            verify_native_cubin(86, path, b".version 8.0", report(&message, "")),
            Err(NativeSassError::NotElfCubin { .. })
        ));
    }
}
