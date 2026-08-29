use std::{collections::BTreeSet, env, process::Command};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExactCudaArchitectures {
    pub(crate) numeric: String,
    pub(crate) native_only: String,
    pub(crate) resolution_mode: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExactCudaArchitectureInputs<'a> {
    pub(crate) build_mode: Option<&'a str>,
    pub(crate) explicit_architectures: Option<&'a str>,
    pub(crate) legacy_cuda_archs: Option<&'a str>,
    pub(crate) legacy_cudaarchs: Option<&'a str>,
    pub(crate) legacy_cmake_cuda_architectures: Option<&'a str>,
    pub(crate) visible_compute_capabilities: Option<&'a str>,
    pub(crate) nvcc_real_architectures: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchitectureRequest<'a> {
    HostAuto,
    CrossReleaseExplicit(&'a str),
}

fn parse_architecture(value: &str) -> Result<u32, String> {
    let value = value.trim();
    let unprefixed = value
        .strip_prefix("sm_")
        .or_else(|| value.strip_prefix("compute_"))
        .unwrap_or(value);

    let architecture = if let Some((major, minor)) = unprefixed.split_once('.') {
        let major = major
            .parse::<u32>()
            .map_err(|_| format!("invalid CUDA architecture {value:?}"))?;
        let minor = minor
            .parse::<u32>()
            .map_err(|_| format!("invalid CUDA architecture {value:?}"))?;
        if minor > 9 {
            return Err(format!("invalid CUDA architecture {value:?}"));
        }
        major
            .checked_mul(10)
            .and_then(|base| base.checked_add(minor))
            .ok_or_else(|| format!("invalid CUDA architecture {value:?}"))?
    } else {
        unprefixed
            .parse::<u32>()
            .map_err(|_| format!("invalid CUDA architecture {value:?}"))?
    };

    if architecture < 10 {
        return Err(format!("invalid CUDA architecture {value:?}"));
    }
    Ok(architecture)
}

pub(crate) fn parse_exact_cuda_architectures(
    value: &str,
) -> Result<ExactCudaArchitectures, String> {
    let architectures = parse_exact_cuda_architecture_set(value)?;
    Ok(render_architectures(architectures, "parsed_explicit"))
}

fn parse_exact_cuda_architecture_set(value: &str) -> Result<BTreeSet<u32>, String> {
    let tokens = value
        .split(|character: char| {
            character == ',' || character == ';' || character.is_ascii_whitespace()
        })
        .filter(|token| !token.is_empty());
    let mut architectures = BTreeSet::new();
    for token in tokens {
        let architecture = parse_architecture(token)?;
        if !architectures.insert(architecture) {
            return Err(format!(
                "duplicate CUDA architecture sm_{architecture} in the exact architecture set"
            ));
        }
    }
    if architectures.is_empty() {
        return Err("the exact CUDA architecture set must not be empty".to_owned());
    }
    Ok(architectures)
}

fn render_architectures(
    architectures: BTreeSet<u32>,
    resolution_mode: &'static str,
) -> ExactCudaArchitectures {
    let numeric = architectures
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(";");
    let native_only = architectures
        .iter()
        .map(|architecture| format!("{architecture}-real"))
        .collect::<Vec<_>>()
        .join(";");
    ExactCudaArchitectures {
        numeric,
        native_only,
        resolution_mode,
    }
}

fn validate_legacy_architecture_inputs(
    inputs: ExactCudaArchitectureInputs<'_>,
    request: ArchitectureRequest<'_>,
) -> Result<(), String> {
    for (name, value) in [
        ("CUDAARCHS", inputs.legacy_cudaarchs),
        (
            "CMAKE_CUDA_ARCHITECTURES",
            inputs.legacy_cmake_cuda_architectures,
        ),
    ] {
        if value.is_some() {
            return Err(format!(
                "{name} is a rejected ambient CUDA architecture authority; use host_auto or the \
                 typed NEOETHOS_CUDA_BUILD_MODE=cross_release_explicit contract"
            ));
        }
    }

    if let Some(cuda_archs) = inputs.legacy_cuda_archs {
        let ArchitectureRequest::CrossReleaseExplicit(typed_architectures) = request else {
            return Err(
                "CUDA_ARCHS is a rejected ambient CUDA architecture authority in host_auto; use visible-device discovery"
                    .to_owned(),
            );
        };
        let typed = parse_exact_cuda_architectures(typed_architectures).map_err(|error| {
            format!("invalid typed NEOETHOS_CUDA_ARCHS architecture set: {error}")
        })?;
        let compatibility = parse_exact_cuda_architectures(cuda_archs)
            .map_err(|error| format!("invalid CUDA_ARCHS compatibility assertion: {error}"))?;
        if compatibility.numeric != typed.numeric {
            return Err(format!(
                "CUDA_ARCHS compatibility assertion `{}` does not exactly match typed NEOETHOS_CUDA_ARCHS `{}`",
                compatibility.numeric, typed.numeric
            ));
        }
    }
    Ok(())
}

fn architecture_request_from_inputs<'a>(
    build_mode: Option<&str>,
    explicit_architectures: Option<&'a str>,
) -> Result<ArchitectureRequest<'a>, String> {
    match (build_mode.map(str::trim), explicit_architectures) {
        (None | Some("host_auto"), None) => Ok(ArchitectureRequest::HostAuto),
        (Some("cross_release_explicit"), Some(architectures))
            if !architectures.trim().is_empty() =>
        {
            Ok(ArchitectureRequest::CrossReleaseExplicit(architectures))
        }
        (None, Some(_)) => Err("NEOETHOS_CUDA_ARCHS is accepted only with \
             NEOETHOS_CUDA_BUILD_MODE=cross_release_explicit"
            .to_owned()),
        (Some("host_auto"), Some(_)) => Err(
            "host_auto detects every visible GPU; it cannot accept NEOETHOS_CUDA_ARCHS".to_owned(),
        ),
        (Some("cross_release_explicit"), None | Some("")) => Err(
            "cross_release_explicit requires a non-empty NEOETHOS_CUDA_ARCHS numeric set"
                .to_owned(),
        ),
        (Some(other), _) => Err(format!(
            "unsupported NEOETHOS_CUDA_BUILD_MODE `{other}`; expected host_auto or \
             cross_release_explicit"
        )),
    }
}

fn parse_visible_compute_capabilities(output: &str) -> Result<BTreeSet<u32>, String> {
    let mut architectures = BTreeSet::new();
    for (line_index, raw_line) in output.lines().enumerate() {
        let capability = raw_line.trim();
        if capability.is_empty() {
            continue;
        }
        let mut parts = capability.split('.');
        let major_text = parts.next().unwrap_or_default();
        let minor_text = parts.next().unwrap_or_default();
        if parts.next().is_some()
            || major_text.is_empty()
            || minor_text.len() != 1
            || !major_text.bytes().all(|byte| byte.is_ascii_digit())
            || !minor_text.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(format!(
                "nvidia-smi returned malformed compute capability `{capability}` on line {}",
                line_index + 1
            ));
        }
        let major = major_text
            .parse::<u32>()
            .map_err(|error| format!("invalid compute-capability major `{major_text}`: {error}"))?;
        let minor = minor_text
            .parse::<u32>()
            .map_err(|error| format!("invalid compute-capability minor `{minor_text}`: {error}"))?;
        if major == 0 {
            return Err(format!(
                "nvidia-smi returned invalid compute capability `{capability}`"
            ));
        }
        let architecture = major
            .checked_mul(10)
            .and_then(|base| base.checked_add(minor))
            .ok_or_else(|| format!("compute capability `{capability}` overflows the build ABI"))?;
        architectures.insert(architecture);
    }
    if architectures.is_empty() {
        return Err("CUDA host_auto found no visible NVIDIA GPU compute capabilities".to_owned());
    }
    Ok(architectures)
}

fn is_valid_cuda_target_suffix(suffix: &str) -> bool {
    let digit_count = suffix
        .bytes()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    digit_count != 0 && matches!(&suffix[digit_count..], "" | "a" | "f")
}

fn parse_nvcc_real_targets(output: &str) -> Result<BTreeSet<String>, String> {
    let mut targets = BTreeSet::new();
    for raw_line in output.lines() {
        let target = raw_line.trim();
        if target.is_empty() {
            continue;
        }
        let suffix = target.strip_prefix("sm_").ok_or_else(|| {
            format!(
                "nvcc --list-gpu-code returned malformed target `{target}`; expected `sm_<digits>`"
            )
        })?;
        if !is_valid_cuda_target_suffix(suffix) {
            return Err(format!(
                "nvcc --list-gpu-code returned malformed target `{target}`"
            ));
        }
        targets.insert(target.to_owned());
    }
    if targets.is_empty() {
        return Err("nvcc --list-gpu-code returned no real CUDA targets".to_owned());
    }
    Ok(targets)
}

fn validate_nvcc_real_targets(
    architectures: &ExactCudaArchitectures,
    output: &str,
) -> Result<(), String> {
    let supported = parse_nvcc_real_targets(output)?;
    let missing = architectures
        .numeric
        .split(';')
        .map(|architecture| format!("sm_{architecture}"))
        .filter(|target| !supported.contains(target))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "the selected nvcc does not support required real CUDA targets: {}",
            missing.join(", ")
        ))
    }
}

pub(crate) fn resolve_exact_cuda_architectures_from_inputs(
    inputs: ExactCudaArchitectureInputs<'_>,
) -> Result<ExactCudaArchitectures, String> {
    let request =
        architecture_request_from_inputs(inputs.build_mode, inputs.explicit_architectures)?;
    validate_legacy_architecture_inputs(inputs, request)?;
    let architectures = match request {
        ArchitectureRequest::HostAuto => render_architectures(
            parse_visible_compute_capabilities(inputs.visible_compute_capabilities.ok_or_else(
                || "host_auto requires complete nvidia-smi compute-capability output".to_owned(),
            )?)?,
            "host_auto",
        ),
        ArchitectureRequest::CrossReleaseExplicit(explicit_architectures) => {
            let mut architectures = parse_exact_cuda_architectures(explicit_architectures)?;
            architectures.resolution_mode = "cross_release_explicit";
            architectures
        }
    };
    validate_nvcc_real_targets(&architectures, inputs.nvcc_real_architectures)?;
    Ok(architectures)
}

fn read_optional_env(name: &str) -> Result<Option<String>, String> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => Err(format!(
            "{name} contains non-Unicode data and cannot be audited"
        )),
    }
}

fn run_tool(program: &str, arguments: &[&str], operation: &str) -> Result<String, String> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| format!("failed to {operation} with `{program}`: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "failed to {operation} with `{program}` (status {:?}): stdout={:?}; stderr={:?}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| {
        format!("{program} returned non-UTF-8 output while trying to {operation}: {error}")
    })
}

pub(crate) fn resolve_exact_cuda_architectures() -> Result<ExactCudaArchitectures, String> {
    for name in [
        "NEOETHOS_CUDA_BUILD_MODE",
        "NEOETHOS_CUDA_ARCHS",
        "CUDA_ARCHS",
        "CUDAARCHS",
        "CMAKE_CUDA_ARCHITECTURES",
        "NVIDIA_SMI",
        "CUDACXX",
        "PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    let build_mode = read_optional_env("NEOETHOS_CUDA_BUILD_MODE")?;
    let explicit_architectures = read_optional_env("NEOETHOS_CUDA_ARCHS")?;
    let legacy_cuda_archs = read_optional_env("CUDA_ARCHS")?;
    let legacy_cudaarchs = read_optional_env("CUDAARCHS")?;
    let legacy_cmake_cuda_architectures = read_optional_env("CMAKE_CUDA_ARCHITECTURES")?;
    let early_inputs = ExactCudaArchitectureInputs {
        build_mode: build_mode.as_deref(),
        explicit_architectures: explicit_architectures.as_deref(),
        legacy_cuda_archs: legacy_cuda_archs.as_deref(),
        legacy_cudaarchs: legacy_cudaarchs.as_deref(),
        legacy_cmake_cuda_architectures: legacy_cmake_cuda_architectures.as_deref(),
        visible_compute_capabilities: None,
        nvcc_real_architectures: "",
    };
    let request = architecture_request_from_inputs(
        early_inputs.build_mode,
        early_inputs.explicit_architectures,
    )?;
    validate_legacy_architecture_inputs(early_inputs, request)?;

    let nvidia_smi = read_optional_env("NVIDIA_SMI")?.unwrap_or_else(|| "nvidia-smi".to_owned());
    let visible_compute_capabilities = match request {
        ArchitectureRequest::HostAuto => Some(run_tool(
            &nvidia_smi,
            &["--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            "detect visible NVIDIA compute capabilities",
        )?),
        ArchitectureRequest::CrossReleaseExplicit(_) => None,
    };
    let nvcc = read_optional_env("CUDACXX")?.unwrap_or_else(|| "nvcc".to_owned());
    let nvcc_real_architectures = run_tool(
        &nvcc,
        &["--list-gpu-code"],
        "list nvcc real CUDA architectures",
    )?;

    let architectures =
        resolve_exact_cuda_architectures_from_inputs(ExactCudaArchitectureInputs {
            build_mode: build_mode.as_deref(),
            explicit_architectures: explicit_architectures.as_deref(),
            legacy_cuda_archs: legacy_cuda_archs.as_deref(),
            legacy_cudaarchs: legacy_cudaarchs.as_deref(),
            legacy_cmake_cuda_architectures: legacy_cmake_cuda_architectures.as_deref(),
            visible_compute_capabilities: visible_compute_capabilities.as_deref(),
            nvcc_real_architectures: &nvcc_real_architectures,
        })?;
    eprintln!(
        "CUDA build architecture authority: mode={}, exact_sass=sm_{}",
        architectures.resolution_mode,
        architectures.numeric.replace(';', ",sm_")
    );
    Ok(architectures)
}
