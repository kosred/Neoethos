use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const DEVICE_SOURCES: [&str; 16] = [
    "native/smoke.cu",
    "native/prototype_b.cu",
    "native/prototype_b_population.cu",
    "native/resident_feature_store_v3.cu",
    "native/resident_higher_timeframe_alignment_v3.cu",
    "native/resident_footprint_v2.cu",
    "native/resident_quant_v3.cu",
    "native/resident_regime_v3.cu",
    "native/resident_scoring_novelty_v1.cu",
    "native/resident_generation_v1.cu",
    "native/resident_archive_knn_v2.cu",
    "native/resident_session_v2.cu",
    "native/resident_robust_normalization_v2.cu",
    "native/resident_smc_v3.cu",
    "native/resident_classic_ta_v3.cu",
    "native/resident_trim_prefilter_v1.cu",
];

const RESIDENT_SEARCH_SLICE2_PRIVATE_HEADERS: [&str; 3] = [
    "native/resident_generation_v2_internal.cuh",
    "native/resident_scoring_novelty_v2_internal.cuh",
    "native/resident_archive_knn_v2_abi.cuh",
];

const PRECISION_FLAGS: [&str; 4] = [
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
];

const REJECTED_ENVIRONMENT: [&str; 7] = [
    "NEOETHOS_CUDA_ARCH",
    "CUDA_ARCH",
    "CUDA_ARCHS",
    "NVCC_ARGS",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "CUDAFLAGS",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolutionMode {
    HostAuto,
    CrossReleaseExplicit,
}

impl ResolutionMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::HostAuto => "host_auto",
            Self::CrossReleaseExplicit => "cross_release_explicit",
        }
    }
}

/// The exact set of real GPU images emitted by this build.
///
/// Architectures are numeric compute-capability tokens (`86`, `89`, `120`),
/// sorted and deduplicated. The plan deliberately has no PTX target: a build
/// for a visible GPU must contain native SASS for that GPU or fail before any
/// CUDA translation unit is compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaArchitecturePlan {
    mode: ResolutionMode,
    architectures: Vec<u16>,
}

impl CudaArchitecturePlan {
    pub fn host_auto_from_tool_output(
        nvidia_smi_compute_capabilities: &str,
        nvcc_virtual_architectures: &str,
        nvcc_real_architectures: &str,
    ) -> Result<Self, String> {
        let architectures = parse_host_compute_capabilities(nvidia_smi_compute_capabilities)?;
        Self::validated(
            ResolutionMode::HostAuto,
            architectures,
            nvcc_virtual_architectures,
            nvcc_real_architectures,
        )
    }

    pub fn cross_release_from_tool_output(
        explicit_architectures: &str,
        nvcc_virtual_architectures: &str,
        nvcc_real_architectures: &str,
    ) -> Result<Self, String> {
        let architectures = parse_cross_release_architectures(explicit_architectures)?;
        Self::validated(
            ResolutionMode::CrossReleaseExplicit,
            architectures,
            nvcc_virtual_architectures,
            nvcc_real_architectures,
        )
    }

    fn validated(
        mode: ResolutionMode,
        architectures: Vec<u16>,
        nvcc_virtual_architectures: &str,
        nvcc_real_architectures: &str,
    ) -> Result<Self, String> {
        validate_nvcc_support(
            &architectures,
            nvcc_virtual_architectures,
            nvcc_real_architectures,
        )?;
        Ok(Self {
            mode,
            architectures,
        })
    }

    pub fn resolution_mode(&self) -> &'static str {
        self.mode.as_str()
    }

    pub fn architectures(&self) -> &[u16] {
        &self.architectures
    }

    pub fn gencode_args(&self) -> Vec<String> {
        self.architectures
            .iter()
            .map(|architecture| {
                format!("--generate-code=arch=compute_{architecture},code=sm_{architecture}")
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactMetadata {
    pub logical_name: String,
    pub sha256: String,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactCodeImages {
    sass_targets: Vec<String>,
    ptx_targets: Vec<String>,
}

impl ArtifactCodeImages {
    pub fn sass_targets(&self) -> &[String] {
        &self.sass_targets
    }

    pub fn ptx_targets(&self) -> &[String] {
        &self.ptx_targets
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativePlatform {
    Unix,
    Windows,
}

impl NativePlatform {
    fn from_target_os(target_os: &str) -> Result<Self, String> {
        match target_os {
            "windows" => Ok(Self::Windows),
            "linux" | "macos" | "freebsd" => Ok(Self::Unix),
            other => Err(format!(
                "the native CUDA builder does not support target OS `{other}`"
            )),
        }
    }

    pub fn object_extension(self) -> &'static str {
        match self {
            Self::Unix => "o",
            Self::Windows => "obj",
        }
    }

    pub fn device_archive_name(self) -> &'static str {
        match self {
            Self::Unix => "libneoethos_gpu_cuda_native.a",
            Self::Windows => "neoethos_gpu_cuda_native.lib",
        }
    }

    fn cuobjdump_program(self) -> &'static str {
        match self {
            Self::Unix => "cuobjdump",
            Self::Windows => "cuobjdump.exe",
        }
    }
}

#[derive(Debug)]
struct ResolvedCudaBuild {
    nvcc: String,
    nvcc_version: String,
    cuobjdump: String,
    cuobjdump_version: String,
    platform: NativePlatform,
    plan: CudaArchitecturePlan,
}

fn parse_host_compute_capabilities(output: &str) -> Result<Vec<u16>, String> {
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
            .parse::<u16>()
            .map_err(|error| format!("invalid compute-capability major `{major_text}`: {error}"))?;
        let minor = minor_text
            .parse::<u16>()
            .map_err(|error| format!("invalid compute-capability minor `{minor_text}`: {error}"))?;
        if major == 0 {
            return Err(format!(
                "nvidia-smi returned invalid compute capability `{capability}`"
            ));
        }
        let architecture = major
            .checked_mul(10)
            .and_then(|value| value.checked_add(minor))
            .ok_or_else(|| format!("compute capability `{capability}` overflows the build ABI"))?;
        architectures.insert(architecture);
    }
    if architectures.is_empty() {
        return Err(
            "CUDA host-auto build found no visible NVIDIA GPU compute capabilities".to_string(),
        );
    }
    Ok(architectures.into_iter().collect())
}

fn parse_cross_release_architectures(raw: &str) -> Result<Vec<u16>, String> {
    if raw.trim().is_empty() {
        return Err(
            "cross-release CUDA architectures must be a semicolon-separated numeric set"
                .to_string(),
        );
    }
    let mut architectures = BTreeSet::new();
    for token in raw.split(';') {
        let token = token.trim();
        if token.is_empty() || !token.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "cross-release CUDA architectures must be a semicolon-separated numeric set; \
                 rejected `{raw}`"
            ));
        }
        let architecture = token.parse::<u16>().map_err(|error| {
            format!("invalid cross-release CUDA architecture `{token}`: {error}")
        })?;
        if architecture == 0 {
            return Err("cross-release CUDA architecture 0 is invalid".to_string());
        }
        architectures.insert(architecture);
    }
    Ok(architectures.into_iter().collect())
}

fn parse_nvcc_targets(
    output: &str,
    prefix: &str,
    option: &str,
) -> Result<BTreeSet<String>, String> {
    let mut targets = BTreeSet::new();
    for raw_line in output.lines() {
        let target = raw_line.trim();
        if target.is_empty() {
            continue;
        }
        let suffix = target.strip_prefix(prefix).ok_or_else(|| {
            format!(
                "nvcc {option} returned malformed target `{target}`; expected `{prefix}<digits>`"
            )
        })?;
        if !is_valid_cuda_target_suffix(suffix) {
            return Err(format!(
                "nvcc {option} returned malformed target `{target}`"
            ));
        }
        targets.insert(target.to_string());
    }
    Ok(targets)
}

fn is_valid_cuda_target_suffix(suffix: &str) -> bool {
    let digit_count = suffix
        .bytes()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digit_count == 0 {
        return false;
    }
    matches!(&suffix[digit_count..], "" | "a" | "f")
}

fn validate_nvcc_support(
    architectures: &[u16],
    nvcc_virtual_architectures: &str,
    nvcc_real_architectures: &str,
) -> Result<(), String> {
    let virtual_targets =
        parse_nvcc_targets(nvcc_virtual_architectures, "compute_", "--list-gpu-arch")?;
    let real_targets = parse_nvcc_targets(nvcc_real_architectures, "sm_", "--list-gpu-code")?;
    let mut missing = Vec::new();
    for architecture in architectures {
        let compute = format!("compute_{architecture}");
        let sm = format!("sm_{architecture}");
        if !virtual_targets.contains(&compute) {
            missing.push(compute);
        }
        if !real_targets.contains(&sm) {
            missing.push(sm);
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "the selected nvcc does not support required CUDA targets: {}",
            missing.join(", ")
        ))
    }
}

/// Reject every environment path that could mutate architecture, output kind,
/// or floating-point semantics outside this reviewed builder.
pub fn validate_external_environment<K: AsRef<str>, V: AsRef<str>>(
    entries: &[(K, V)],
) -> Result<(), String> {
    for (name, value) in entries {
        let name = name.as_ref();
        let value = value.as_ref();
        if REJECTED_ENVIRONMENT.contains(&name) && !value.trim().is_empty() {
            return Err(format!(
                "environment variable `{name}` is forbidden for the native CUDA builder; \
                 architecture and precision are controlled by the typed build contract"
            ));
        }
        if name == "CUDA_FAST_MATH"
            && !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            )
        {
            return Err(format!(
                "environment variable `CUDA_FAST_MATH={value}` violates the exact-precision \
                 CUDA build contract"
            ));
        }
    }
    Ok(())
}

fn validate_process_environment() -> Result<(), String> {
    let mut entries = Vec::new();
    for name in REJECTED_ENVIRONMENT
        .iter()
        .copied()
        .chain(std::iter::once("CUDA_FAST_MATH"))
    {
        if let Some(value) = read_optional_env(name)? {
            entries.push((name.to_string(), value));
        }
    }
    validate_external_environment(&entries)
}

fn read_optional_env(name: &str) -> Result<Option<String>, String> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => {
            Err(format!("environment variable `{name}` is not valid UTF-8"))
        }
    }
}

pub fn build_nvcc_argv(
    plan: &CudaArchitecturePlan,
    source: &str,
    output: &str,
    debug: bool,
    platform: NativePlatform,
) -> Vec<String> {
    let mut arguments = vec![
        "-c".to_string(),
        source.to_string(),
        "-o".to_string(),
        output.to_string(),
        "-std=c++17".to_string(),
    ];
    arguments.extend(plan.gencode_args());
    arguments.extend(PRECISION_FLAGS.iter().map(|flag| (*flag).to_string()));
    if platform == NativePlatform::Unix {
        arguments.push("-Xcompiler=-fPIC".to_string());
    }
    arguments.extend(["-I".to_string(), "native".to_string()]);
    arguments.push(if debug { "-lineinfo" } else { "-O3" }.to_string());
    arguments
}

pub fn build_nvcc_archive_argv(
    plan: &CudaArchitecturePlan,
    output: &str,
    objects: &[PathBuf],
) -> Vec<String> {
    let mut arguments = vec!["--lib".to_string(), "-o".to_string(), output.to_string()];
    arguments.extend(plan.gencode_args());
    arguments.extend(
        objects
            .iter()
            .map(|object| object.to_string_lossy().into_owned()),
    );
    arguments
}

pub fn inspect_artifact_images(
    plan: &CudaArchitecturePlan,
    cuobjdump_elf_list: &str,
    cuobjdump_ptx_list: &str,
) -> Result<ArtifactCodeImages, String> {
    let sass_targets = extract_cuda_targets(cuobjdump_elf_list, "sm_");
    if sass_targets.is_empty() {
        return Err("cuobjdump found no SASS images in the native CUDA archive".to_string());
    }
    let expected_sass = plan
        .architectures()
        .iter()
        .map(|architecture| format!("sm_{architecture}"))
        .collect::<BTreeSet<_>>();
    if sass_targets != expected_sass {
        return Err(format!(
            "emitted CUDA archive SASS targets do not match requested SASS targets: \
             requested={expected_sass:?} inspected={sass_targets:?}"
        ));
    }

    let mut ptx_targets = extract_cuda_targets(cuobjdump_ptx_list, "compute_");
    ptx_targets.extend(extract_cuda_targets(cuobjdump_ptx_list, "sm_"));
    if !cuobjdump_ptx_list.trim().is_empty()
        && ptx_targets.is_empty()
        && !is_canonical_no_ptx_diagnostic(cuobjdump_ptx_list)
    {
        return Err(format!(
            "cuobjdump returned an unrecognized non-empty PTX listing: `{}`",
            normalize_for_error(cuobjdump_ptx_list)
        ));
    }
    if !ptx_targets.is_empty() {
        return Err(format!(
            "native CUDA archive unexpectedly embeds PTX targets: {}",
            ptx_targets.iter().cloned().collect::<Vec<_>>().join(", ")
        ));
    }

    Ok(ArtifactCodeImages {
        sass_targets: plan
            .architectures()
            .iter()
            .map(|architecture| format!("sm_{architecture}"))
            .collect(),
        ptx_targets: Vec::new(),
    })
}

fn is_canonical_no_ptx_diagnostic(output: &str) -> bool {
    let mut diagnostic_archive = None;
    let mut members = Vec::new();

    for line in output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if let Some(archive) = canonical_no_ptx_archive(line) {
            if diagnostic_archive.replace(archive).is_some() {
                return false;
            }
        } else if let Some(member) = canonical_archive_member(line) {
            members.push(member);
        } else {
            return false;
        }
    }

    let Some(diagnostic_archive) = diagnostic_archive else {
        return false;
    };
    let mut member_objects = BTreeSet::new();
    members
        .into_iter()
        .all(|(archive, object)| archive == diagnostic_archive && member_objects.insert(object))
}

fn canonical_no_ptx_archive(line: &str) -> Option<&str> {
    let remainder = line.strip_prefix("cuobjdump info")?.trim_start();
    let remainder = remainder.strip_prefix(": No PTX file found to extract from '")?;
    let (archive, suffix) = remainder.rsplit_once('\'')?;
    (!archive.is_empty()
        && archive.trim() == archive
        && !archive.contains('\'')
        && matches!(suffix, "." | ". You may try with -all option."))
    .then_some(archive)
}

fn canonical_archive_member(line: &str) -> Option<(&str, &str)> {
    let member = line.strip_prefix("member ")?.strip_suffix(':')?;
    let (archive, object) = member.rsplit_once(':')?;
    (!archive.is_empty()
        && archive.trim() == archive
        && !archive.chars().any(char::is_control)
        && is_safe_archive_member_basename(object))
    .then_some((archive, object))
}

fn is_safe_archive_member_basename(object: &str) -> bool {
    !object.is_empty()
        && !object.contains("..")
        && object
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn extract_cuda_targets(output: &str, prefix: &str) -> BTreeSet<String> {
    output
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .filter_map(|token| {
            let suffix = token.strip_prefix(prefix)?;
            is_valid_cuda_target_suffix(suffix).then(|| token.to_string())
        })
        .collect()
}

pub fn render_manifest_v1(
    plan: &CudaArchitecturePlan,
    nvcc_version: &str,
    cuobjdump_version: &str,
    debug: bool,
    artifact: &ArtifactMetadata,
    images: &ArtifactCodeImages,
) -> String {
    let architectures = plan
        .architectures()
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let gencode = render_json_string_array(&plan.gencode_args());
    let sass_targets = render_json_string_array(images.sass_targets());
    let ptx_targets = render_json_string_array(images.ptx_targets());
    let precision_flags = render_json_string_array(
        &PRECISION_FLAGS
            .iter()
            .map(|flag| (*flag).to_string())
            .collect::<Vec<_>>(),
    );
    format!(
        "{{\"schema\":\"neoethos.cuda-native-build.v1\",\"resolution_mode\":\"{}\",\
         \"architectures\":[{architectures}],\"gencode\":{gencode},\
         \"sass_targets\":{sass_targets},\"ptx_targets\":{ptx_targets},\
         \"precision_flags\":{precision_flags},\"optimization\":\"{}\",\
         \"nvcc_version\":\"{}\",\"cuobjdump_version\":\"{}\",\
         \"artifact\":{{\"logical_name\":\"{}\",\
         \"sha256\":\"{}\",\"byte_len\":{}}}}}",
        plan.resolution_mode(),
        if debug { "-lineinfo" } else { "-O3" },
        json_escape(nvcc_version),
        json_escape(cuobjdump_version),
        json_escape(&artifact.logical_name),
        json_escape(&artifact.sha256),
        artifact.byte_len
    )
}

fn render_json_string_array(values: &[String]) -> String {
    let entries = values
        .iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{entries}]")
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::new();
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                write!(&mut escaped, "\\u{:04x}", character as u32)
                    .expect("writing to a String cannot fail");
            }
            character => escaped.push(character),
        }
    }
    escaped
}

fn main() {
    let cuda_feature = env::var_os("CARGO_FEATURE_CUDA").is_some();
    emit_rerun_contract(cuda_feature);
    // CpuOnly/default builds deliberately perform no NVIDIA or nvcc probe.
    // CUDA builds resolve and validate the complete architecture plan before
    // even the host ABI compiler is started.
    let cuda_build = if cuda_feature {
        Some(resolve_cuda_build().unwrap_or_else(|error| panic!("{error}")))
    } else {
        None
    };

    // Host C++ and CUDA are separate units on purpose. A single `cc::Build`
    // with `.cuda(true)` drives every file through nvcc — including the plain
    // `.cpp` — and hands it gcc-shaped flags (`-ffunction-sections`, `-Wall`,
    // `-gdwarf-4`, ...) that nvcc rejects outright.
    let mut host = cc::Build::new();
    host.cpp(true).std("c++17").include("native");
    host.file("native/layout_asserts.cpp");
    if !cuda_feature {
        host.file("native/stub.cpp");
    }
    host.compile("neoethos_gpu_cuda_abi");

    if let Some(cuda_build) = cuda_build {
        compile_device_objects(&cuda_build);
    }
}

fn emit_rerun_contract(cuda_feature: bool) {
    println!("cargo:rerun-if-changed=native/neoethos_gpu_cuda.h");
    println!("cargo:rerun-if-changed=native/resident_higher_timeframe_alignment_v3_abi.cuh");
    println!("cargo:rerun-if-changed=native/resident_quant_v3_abi.cuh");
    println!("cargo:rerun-if-changed=native/resident_trim_prefilter_v1_abi.cuh");
    println!("cargo:rerun-if-changed=native/resident_session_v2_abi.cuh");
    println!("cargo:rerun-if-changed=native/layout_asserts.cpp");
    println!("cargo:rerun-if-changed=native/stub.cpp");
    for source in DEVICE_SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
    for header in RESIDENT_SEARCH_SLICE2_PRIVATE_HEADERS {
        println!("cargo:rerun-if-changed={header}");
    }
    for name in [
        "CUDACXX",
        "CUDAOBJDUMP",
        "NVIDIA_SMI",
        "NEOETHOS_CUDA_BUILD_MODE",
        "NEOETHOS_CUDA_ARCHS",
        "NEOETHOS_CUDA_ARCH",
        "CUDA_ARCH",
        "CUDA_ARCHS",
        "NVCC_ARGS",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "CUDAFLAGS",
        "CUDA_FAST_MATH",
        "DEBUG",
        "CUDA_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    if cuda_feature {
        for name in ["PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"] {
            println!("cargo:rerun-if-env-changed={name}");
        }
        let linux_inventory = Path::new("/proc/driver/nvidia/gpus");
        if linux_inventory.exists() {
            println!(
                "cargo:rerun-if-changed={}",
                linux_inventory.to_string_lossy()
            );
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchitectureRequest {
    HostAuto,
    CrossRelease(String),
}

pub fn architecture_request_from_env(
    build_mode: Option<&str>,
    explicit_architectures: Option<&str>,
) -> Result<ArchitectureRequest, String> {
    match (build_mode, explicit_architectures) {
        (None | Some("host_auto"), None) => Ok(ArchitectureRequest::HostAuto),
        (Some("cross_release_explicit"), Some(architectures))
            if !architectures.trim().is_empty() =>
        {
            Ok(ArchitectureRequest::CrossRelease(architectures.to_string()))
        }
        (None, Some(_)) => Err("NEOETHOS_CUDA_ARCHS is accepted only with \
                 NEOETHOS_CUDA_BUILD_MODE=cross_release_explicit"
            .to_string()),
        (Some("host_auto"), Some(_)) => Err(
            "host_auto detects every visible GPU; it cannot accept NEOETHOS_CUDA_ARCHS".to_string(),
        ),
        (Some("cross_release_explicit"), None | Some("")) => Err(
            "cross_release_explicit requires a non-empty NEOETHOS_CUDA_ARCHS numeric set"
                .to_string(),
        ),
        (Some(other), _) => Err(format!(
            "unsupported NEOETHOS_CUDA_BUILD_MODE `{other}`; expected host_auto or \
                 cross_release_explicit"
        )),
    }
}

fn resolve_cuda_build() -> Result<ResolvedCudaBuild, String> {
    validate_process_environment()?;

    let target_os = env::var("CARGO_CFG_TARGET_OS")
        .map_err(|error| format!("Cargo did not provide CARGO_CFG_TARGET_OS: {error}"))?;
    let platform = NativePlatform::from_target_os(&target_os)?;

    let build_mode = read_optional_env("NEOETHOS_CUDA_BUILD_MODE")?;
    let explicit_architectures = read_optional_env("NEOETHOS_CUDA_ARCHS")?;
    let requested =
        architecture_request_from_env(build_mode.as_deref(), explicit_architectures.as_deref())?;

    let host_capabilities = match &requested {
        ArchitectureRequest::HostAuto => {
            let nvidia_smi =
                read_optional_env("NVIDIA_SMI")?.unwrap_or_else(|| "nvidia-smi".to_string());
            Some(run_tool(
                &nvidia_smi,
                &["--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
                "detect visible NVIDIA compute capabilities",
            )?)
        }
        ArchitectureRequest::CrossRelease(_) => None,
    };

    let nvcc = read_optional_env("CUDACXX")?.unwrap_or_else(|| "nvcc".to_string());
    let nvcc_version = normalize_tool_output(
        &run_tool(&nvcc, &["--version"], "read the CUDA compiler version")?,
        "nvcc --version",
    )?;
    let virtual_architectures = run_tool(
        &nvcc,
        &["--list-gpu-arch"],
        "list nvcc virtual architectures",
    )?;
    let real_architectures = run_tool(&nvcc, &["--list-gpu-code"], "list nvcc real architectures")?;
    let cuobjdump = read_optional_env("CUDAOBJDUMP")?
        .unwrap_or_else(|| sibling_cuda_tool(&nvcc, platform.cuobjdump_program()));
    let cuobjdump_version = normalize_tool_output(
        &run_tool(
            &cuobjdump,
            &["--version"],
            "read the CUDA binary inspector version",
        )?,
        "cuobjdump --version",
    )?;

    let plan = match requested {
        ArchitectureRequest::HostAuto => CudaArchitecturePlan::host_auto_from_tool_output(
            host_capabilities
                .as_deref()
                .expect("host-auto always captures nvidia-smi output"),
            &virtual_architectures,
            &real_architectures,
        )?,
        ArchitectureRequest::CrossRelease(architectures) => {
            CudaArchitecturePlan::cross_release_from_tool_output(
                &architectures,
                &virtual_architectures,
                &real_architectures,
            )?
        }
    };

    Ok(ResolvedCudaBuild {
        nvcc,
        nvcc_version,
        cuobjdump,
        cuobjdump_version,
        platform,
        plan,
    })
}

fn sibling_cuda_tool(nvcc: &str, program_name: &str) -> String {
    let path = Path::new(nvcc);
    match path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        Some(parent) => parent.join(program_name).to_string_lossy().into_owned(),
        None => program_name.to_string(),
    }
}

fn run_tool(tool: &str, arguments: &[&str], purpose: &str) -> Result<String, String> {
    run_tool_clean(tool, arguments, purpose)
}

fn run_tool_clean(tool: &str, arguments: &[&str], purpose: &str) -> Result<String, String> {
    let (stdout, stderr) = run_tool_with_diagnostics(tool, arguments, purpose)?;
    if !stderr.trim().is_empty() {
        return Err(format!(
            "`{tool}` emitted an unexpected diagnostic while trying to {purpose}: `{}`",
            normalize_for_error(&stderr)
        ));
    }
    Ok(stdout)
}

fn run_tool_with_diagnostics(
    tool: &str,
    arguments: &[&str],
    purpose: &str,
) -> Result<(String, String), String> {
    let output = Command::new(tool)
        .args(arguments)
        .output()
        .map_err(|error| format!("failed to run `{tool}` to {purpose}: {error}"))?;
    let stdout = String::from_utf8(output.stdout).map_err(|error| {
        format!("`{tool}` emitted non-UTF-8 stdout while trying to {purpose}: {error}")
    })?;
    let stderr = String::from_utf8(output.stderr).map_err(|error| {
        format!("`{tool}` emitted non-UTF-8 stderr while trying to {purpose}: {error}")
    })?;
    if !output.status.success() {
        return Err(format!(
            "`{tool}` failed to {purpose} (status {}): stdout=`{}` stderr=`{}`",
            output.status,
            normalize_for_error(&stdout),
            normalize_for_error(&stderr)
        ));
    }
    Ok((stdout, stderr))
}

fn normalize_tool_output(output: &str, command: &str) -> Result<String, String> {
    let normalized = output.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        Err(format!("{command} returned empty output"))
    } else {
        Ok(normalized)
    }
}

fn normalize_for_error(output: &str) -> String {
    output.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Compile the device translation units by driving nvcc directly.
///
/// cc-rs always passes `--device-c` in CUDA mode, which produces relocatable
/// device code that then *requires* a separate `nvcc -dlink` step. Cargo links
/// the resulting archive with the host linker instead, so the kernels are never
/// registered and every launch fails at runtime. Whole-program compilation has
/// no such requirement, so each device unit is compiled here explicitly.
fn compile_device_objects(cuda_build: &ResolvedCudaBuild) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let debug = env::var("DEBUG")
        .map(|value| value == "true")
        .unwrap_or(false);

    let mut objects = Vec::new();
    for source in DEVICE_SOURCES {
        let stem = Path::new(source)
            .file_stem()
            .expect("device sources have file names")
            .to_string_lossy()
            .into_owned();
        let object = out_dir.join(format!("{stem}.{}", cuda_build.platform.object_extension()));
        let object_text = object.to_str().unwrap_or_else(|| {
            panic!(
                "CUDA OUT_DIR object path is not valid UTF-8: {}",
                object.display()
            )
        });
        let arguments = build_nvcc_argv(
            &cuda_build.plan,
            source,
            object_text,
            debug,
            cuda_build.platform,
        );
        let status = Command::new(&cuda_build.nvcc)
            .args(&arguments)
            .status()
            .unwrap_or_else(|error| {
                panic!(
                    "failed to run {} on {source} with reviewed argv {:?}: {error}",
                    cuda_build.nvcc, arguments
                )
            });
        if !status.success() {
            panic!(
                "nvcc failed to compile {source} for {:?} (status {status})",
                cuda_build.plan.architectures()
            );
        }
        objects.push(object);
    }

    let archive = out_dir.join(cuda_build.platform.device_archive_name());
    let _ = std::fs::remove_file(&archive);
    let archive_text = archive.to_str().unwrap_or_else(|| {
        panic!(
            "CUDA OUT_DIR archive path is not valid UTF-8: {}",
            archive.display()
        )
    });
    let archive_arguments = build_nvcc_archive_argv(&cuda_build.plan, archive_text, &objects);
    let status = Command::new(&cuda_build.nvcc)
        .args(&archive_arguments)
        .status()
        .unwrap_or_else(|error| {
            panic!(
                "failed to run {} for the CUDA static archive with reviewed argv {:?}: {error}",
                cuda_build.nvcc, archive_arguments
            )
        });
    if !status.success() {
        panic!(
            "nvcc failed to build the device archive for {:?} (status {status})",
            cuda_build.plan.architectures()
        );
    }

    let elf_list = run_tool_clean(
        &cuda_build.cuobjdump,
        &["--list-elf", archive_text],
        "inspect emitted CUDA SASS images",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let (ptx_stdout, ptx_stderr) = run_tool_with_diagnostics(
        &cuda_build.cuobjdump,
        &["--list-ptx", archive_text],
        "inspect emitted CUDA PTX images",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let ptx_list = match (ptx_stdout.trim().is_empty(), ptx_stderr.trim().is_empty()) {
        (false, false) => format!("{ptx_stdout}\n{ptx_stderr}"),
        (false, true) => ptx_stdout,
        (true, false) => ptx_stderr,
        (true, true) => String::new(),
    };
    let images = inspect_artifact_images(&cuda_build.plan, &elf_list, &ptx_list)
        .unwrap_or_else(|error| panic!("{error}"));
    let artifact = artifact_metadata(&archive).unwrap_or_else(|error| panic!("{error}"));
    let manifest = render_manifest_v1(
        &cuda_build.plan,
        &cuda_build.nvcc_version,
        &cuda_build.cuobjdump_version,
        debug,
        &artifact,
        &images,
    );
    let manifest_path = out_dir.join("neoethos_cuda_build_manifest_v1.json");
    std::fs::write(&manifest_path, manifest.as_bytes()).unwrap_or_else(|error| {
        panic!(
            "failed to write CUDA build manifest {}: {error}",
            manifest_path.display()
        )
    });
    embed_manifest(&out_dir, &manifest);

    println!("cargo:rustc-env=NEOETHOS_CUDA_BUILD_MANIFEST_V1={manifest}");
    println!(
        "cargo:rustc-env=NEOETHOS_CUDA_BUILD_ARCHITECTURES={}",
        cuda_build
            .plan
            .architectures()
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join(";")
    );
    println!(
        "cargo:rustc-env=NEOETHOS_CUDA_BUILD_RESOLUTION_MODE={}",
        cuda_build.plan.resolution_mode()
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // `+whole-archive` is load-bearing. nvcc registers each fatbin from a
    // constructor in `.init_array`; nothing in Rust references that constructor
    // by symbol, so a normal archive link drops the registration.
    println!("cargo:rustc-link-lib=static:+whole-archive=neoethos_gpu_cuda_native");
    println!("cargo:rustc-link-lib=static:+whole-archive=neoethos_cuda_build_manifest");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if cuda_build.platform == NativePlatform::Unix {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    if let Ok(path) = env::var("CUDA_PATH") {
        let cuda_library = match cuda_build.platform {
            NativePlatform::Unix => PathBuf::from(path).join("lib64"),
            NativePlatform::Windows => PathBuf::from(path).join("lib").join("x64"),
        };
        println!("cargo:rustc-link-search=native={}", cuda_library.display());
    }
    if cuda_build.platform == NativePlatform::Unix {
        for candidate in ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"] {
            if Path::new(candidate).is_dir() {
                println!("cargo:rustc-link-search=native={candidate}");
            }
        }
    }
}

fn artifact_metadata(path: &Path) -> Result<ArtifactMetadata, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("failed to open CUDA artifact {}: {error}", path.display()))?;
    let byte_len = file
        .metadata()
        .map_err(|error| format!("failed to stat CUDA artifact {}: {error}", path.display()))?
        .len();
    if byte_len == 0 {
        return Err(format!("CUDA artifact {} is empty", path.display()));
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash CUDA artifact {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(ArtifactMetadata {
        logical_name: path
            .file_name()
            .expect("CUDA artifact has a file name")
            .to_string_lossy()
            .into_owned(),
        sha256: format!("{:x}", hasher.finalize()),
        byte_len,
    })
}

fn embed_manifest(out_dir: &Path, manifest: &str) {
    let generated_source = out_dir.join("neoethos_cuda_build_manifest_v1.cpp");
    let bytes = manifest
        .as_bytes()
        .iter()
        .chain(std::iter::once(&0_u8))
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let source = format!(
        "#include <cstddef>\n\
         extern \"C\" {{\n\
         #if defined(_MSC_VER)\n\
         #pragma comment(linker, \"/include:neoethos_cuda_build_manifest_v1\")\n\
         __declspec(dllexport)\n\
         #else\n\
         __attribute__((used, retain, visibility(\"default\")))\n\
         #endif\n\
         extern const unsigned char neoethos_cuda_build_manifest_v1[] = {{{bytes}}};\n\
         extern const std::size_t neoethos_cuda_build_manifest_v1_len = {};\n\
         }}\n",
        manifest.len()
    );
    std::fs::write(&generated_source, source).unwrap_or_else(|error| {
        panic!(
            "failed to write embedded CUDA manifest source {}: {error}",
            generated_source.display()
        )
    });
    let mut manifest_build = cc::Build::new();
    manifest_build
        .cpp(true)
        .std("c++17")
        .cargo_metadata(false)
        .file(&generated_source)
        .compile("neoethos_cuda_build_manifest");
}
