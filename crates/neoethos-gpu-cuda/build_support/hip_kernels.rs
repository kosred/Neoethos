//! One production compiler path for the existing CUDA source inventory on HIP.
//! Translation is generated; there is no separately maintained math source tree.
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const SESSION_RENAME: &str =
    "-Dneoethos_resident_session_f64_v2=neoethos_hip_resident_session_f64_v2";

pub struct KernelArchive {
    pub native_directory: PathBuf,
    pub artifact_sha256: String,
    pub sources_json: String,
}

pub fn embed_manifest_sha256(root: &Path, output: &Path, manifest: &str) -> Result<(), String> {
    let digest = Sha256::digest(manifest.as_bytes());
    let bytes = digest
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let source = output.join("hip_native_manifest_v1.cpp");
    // Separate from the device archive: the manifest binds that archive, so
    // inserting its own digest into it would create a circular artifact hash.
    fs::write(&source, format!(
        "extern \"C\" const unsigned char* neoethos_hip_native_build_manifest_sha256_v1() {{\nstatic const unsigned char value[32] = {{{bytes}}};\nreturn value;\n}}\n"
    )).map_err(|e| e.to_string())?;
    let object = output.join("hip_native_manifest_v1.o");
    let args = super::compile_arguments(&root.join("include"), &source, &object);
    super::run(
        &root.join("llvm/bin/clang++"),
        &args,
        output,
        "native-manifest-compile",
    )?;
    let archive = output.join("libneoethos_hip_native_manifest_v1.a");
    match fs::remove_file(&archive) {
        Ok(()) => (),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        Err(e) => return Err(e.to_string()),
    }
    super::run(
        &root.join("llvm/bin/llvm-ar"),
        &[
            "rcs".into(),
            archive.to_string_lossy().into_owned(),
            object.to_string_lossy().into_owned(),
        ],
        output,
        "native-manifest-archive",
    )?;
    println!("cargo:rustc-link-lib=static=neoethos_hip_native_manifest_v1");
    Ok(())
}

fn project_sources(sources: &[&str]) -> Result<BTreeMap<PathBuf, String>, String> {
    let crate_root = Path::new(".").canonicalize().map_err(|e| e.to_string())?;
    let mut pending = sources.iter().map(PathBuf::from).collect::<Vec<_>>();
    let mut records = BTreeMap::new();
    while let Some(source) = pending.pop() {
        let absolute = source
            .canonicalize()
            .map_err(|e| format!("{}: {e}", source.display()))?;
        let relative = absolute
            .strip_prefix(&crate_root)
            .map_err(|_| format!("HIP project include escapes crate: {}", absolute.display()))?
            .to_path_buf();
        if records.contains_key(&relative) {
            continue;
        }
        let content = fs::read_to_string(&absolute).map_err(|e| e.to_string())?;
        for line in content.lines().map(str::trim) {
            let Some(directive) = line.strip_prefix('#').map(str::trim_start) else {
                continue;
            };
            let Some(include) = directive.strip_prefix("include").map(str::trim_start) else {
                continue;
            };
            let Some(quoted) = include.strip_prefix('"') else {
                continue;
            };
            let (name, _) = quoted
                .split_once('"')
                .ok_or("unterminated project include")?;
            let candidate = absolute.parent().ok_or("source parent missing")?.join(name);
            let candidate = if candidate.is_file() {
                candidate
            } else {
                crate_root.join("native").join(name)
            };
            pending.push(candidate);
        }
        records.insert(relative, super::super::artifact_metadata(&absolute)?.sha256);
    }
    Ok(records)
}

fn error_diagnostics(output: &Path, label: &str) -> Result<(), String> {
    for channel in ["stdout", "stderr"] {
        let log = fs::read_to_string(output.join(format!("hip-runtime-{label}.{channel}.log")))
            .map_err(|e| e.to_string())?;
        if log.lines().any(has_error_diagnostic) {
            return Err(format!(
                "HIPIFY {label} emitted errors despite its exit status; logs retained"
            ));
        }
    }
    Ok(())
}

fn has_error_diagnostic(line: &str) -> bool {
    let line = line.trim_start();
    let line = line.strip_prefix("[HIPIFY]").map_or(line, str::trim_start);
    line.contains(": error:")
        || line.contains(": fatal error:")
        || line.starts_with("error:")
        || line.starts_with("fatal error:")
}

fn translation_stamp(
    input: &str,
    destination: &Path,
    output: &Path,
    label: &str,
) -> Result<String, String> {
    let mut stamp = format!("neoethos.hipify-perl-cache.v1\n{input}\n");
    // Device/source artifacts must be nonempty; successful log streams may be empty.
    stamp.push_str(&super::super::artifact_metadata(destination)?.sha256);
    stamp.push('\n');
    for path in [
        output.join(format!("hip-runtime-{label}.stdout.log")),
        output.join(format!("hip-runtime-{label}.stderr.log")),
    ] {
        stamp.push_str(&format!(
            "{:x}",
            Sha256::digest(fs::read(&path).map_err(|e| e.to_string())?)
        ));
        stamp.push('\n');
    }
    Ok(stamp)
}

fn translate_one(
    translator: &Path,
    perl: &Path,
    tool_identity: &str,
    source: &Path,
    destination: &Path,
    output: &Path,
    label: &str,
) -> Result<(), String> {
    // Reuse only an exact completed translation, including both full log hashes.
    // Failed/partial runs never write a receipt. Compiler/image verification is
    // NOT cached here and no receipt is evidence of device execution.
    let arguments = vec![
        translator.to_string_lossy().into_owned(),
        source.to_string_lossy().into_owned(),
        "-o".into(),
        destination.to_string_lossy().into_owned(),
        "--print-stats".into(),
    ];
    let mut input = Sha256::new();
    input.update(tool_identity.as_bytes());
    input.update(super::super::artifact_metadata(source)?.sha256.as_bytes());
    for argument in &arguments {
        input.update((argument.len() as u64).to_le_bytes());
        input.update(argument.as_bytes());
    }
    let input = format!("{:x}", input.finalize());
    let receipt = output.join(format!("hip-runtime-{label}.translation-cache"));
    let current = translation_stamp(&input, destination, output, label);
    if let (Ok(saved), Ok(current)) = (fs::read_to_string(&receipt), current)
        && saved == current
    {
        eprintln!("HIPIFY cache: exact source/tool/output/logs reused for {label}");
        for channel in ["stdout", "stderr"] {
            eprint!(
                "{}",
                fs::read_to_string(output.join(format!("hip-runtime-{label}.{channel}.log")))
                    .map_err(|e| e.to_string())?
            );
        }
        error_diagnostics(output, label)?;
        return Ok(());
    }
    // HIPIFY Perl performs many whole-source regex passes; population needs a
    // separate, bounded deadline. Use the pinned system interpreter explicitly.
    super::run_with_timeout(
        perl,
        &arguments,
        output,
        label,
        std::time::Duration::from_secs(900),
    )?;
    error_diagnostics(output, label)?;
    if fs::metadata(destination).map_err(|e| e.to_string())?.len() == 0 {
        return Err(format!(
            "HIPIFY produced empty source: {}",
            source.display()
        ));
    }
    fs::write(
        receipt,
        translation_stamp(&input, destination, output, label)?,
    )
    .map_err(|e| e.to_string())
}

fn compile_one(
    root: &Path,
    generated: &Path,
    output: &Path,
    target: &str,
    source: &str,
    fixtures: bool,
) -> Result<PathBuf, String> {
    let stem = Path::new(source)
        .file_stem()
        .and_then(|v| v.to_str())
        .ok_or("invalid kernel name")?;
    let object = output.join(format!("hip_{stem}.o"));
    let mut arguments = vec![
        "--no-default-config".into(),
        "--target=x86_64-unknown-linux-gnu".into(),
        "-x".into(),
        "hip".into(),
        format!("--rocm-path={}", root.display()),
        format!("--hip-path={}", root.display()),
        format!("--offload-arch={target}"),
        "-std=c++17".into(),
        "-O3".into(),
        "-fPIC".into(),
        "-Wall".into(),
        "-Wextra".into(),
        "-Werror".into(),
        "-fno-fast-math".into(),
        "-ffp-contract=off".into(),
        "-fdenormal-fp-math=ieee".into(),
        "-Xclang".into(),
        "-fdenormal-fp-math-f32=ieee".into(),
        "-fno-gpu-flush-denormals-to-zero".into(),
        "-fhip-fp32-correctly-rounded-divide-sqrt".into(),
        "-mno-unsafe-fp-atomics".into(),
        "-D__HIP_PLATFORM_AMD__=1".into(),
        SESSION_RENAME.into(),
        "-I".into(),
        root.join("include").to_string_lossy().into_owned(),
        "-I".into(),
        generated.join("native").to_string_lossy().into_owned(),
        "-c".into(),
        generated.join(source).to_string_lossy().into_owned(),
        "-o".into(),
        object.to_string_lossy().into_owned(),
    ];
    if fixtures {
        arguments.push("-DNEOETHOS_CUDA_DEVICE_FIXTURES_V2=1".into());
    }
    super::run(
        &root.join("llvm/bin/clang++"),
        &arguments,
        output,
        &format!("{stem}-compile"),
    )?;
    let images = super::run(
        &root.join("llvm/bin/llvm-objdump"),
        &["--offloading".into(), object.to_string_lossy().into_owned()],
        output,
        &format!("{stem}-images"),
    )?;
    let extracted = images
        .lines()
        .filter_map(|line| line.strip_prefix("Extracting offload bundle: "))
        .collect::<Vec<_>>();
    let device = format!("{}.0.hipv4-amdgcn-amd-amdhsa--{target}", object.display());
    let host = format!("{}.0.host-x86_64-unknown-linux-gnu-", object.display());
    if extracted != [host.as_str(), device.as_str()] {
        return Err(format!(
            "HIP {source} offload bundle set mismatch: {extracted:?}"
        ));
    }
    let elf = super::run(
        &root.join("llvm/bin/llvm-readelf"),
        &["--file-header".into(), device],
        output,
        &format!("{stem}-device-elf"),
    )?;
    let flags = elf
        .lines()
        .find_map(|line| line.trim().strip_prefix("Flags:"))
        .ok_or("device ELF flags missing")?;
    if !elf.contains("EM_AMDGPU")
        || !elf.contains("AMDGPU - HSA")
        || !flags.split(',').any(|flag| flag.trim() == target)
    {
        return Err(format!(
            "HIP {source} is not an exact {target} AMD HSA image"
        ));
    }
    Ok(object)
}

pub fn compile(
    root: &Path,
    output: &Path,
    target: &str,
    sources: &[&str],
    fixtures: bool,
) -> Result<KernelArchive, String> {
    let translator = root
        .join("bin/hipify-perl")
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let perl = Path::new("/usr/bin/perl")
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let tool_identity = format!(
        "{}:{}",
        super::super::artifact_metadata(&translator)?.sha256,
        super::super::artifact_metadata(&perl)?.sha256
    );
    for tool in [
        translator.clone(),
        perl.clone(),
        root.join("llvm/bin/clang++"),
        root.join("llvm/bin/llvm-ar"),
        root.join("llvm/bin/llvm-objdump"),
        root.join("llvm/bin/llvm-readelf"),
    ] {
        if !tool.is_file() {
            return Err(format!("required HIP tool missing: {}", tool.display()));
        }
        println!("cargo:rerun-if-changed={}", tool.display());
    }
    for directory in [root.join("include"), root.join("amdgcn/bitcode")] {
        println!("cargo:rerun-if-changed={}", directory.display());
    }
    let resource = super::run(
        &root.join("llvm/bin/clang++"),
        &["--no-default-config".into(), "-print-resource-dir".into()],
        output,
        "kernel-resource-dir",
    )?;
    println!(
        "cargo:rerun-if-changed={}",
        Path::new(resource.trim()).join("include").display()
    );
    let records = project_sources(sources)?;
    let generated = output.join("hip-kernels");
    fs::create_dir_all(&generated).map_err(|e| e.to_string())?;
    for relative in records.keys() {
        println!("cargo:rerun-if-changed={}", relative.display());
        let source = relative.canonicalize().map_err(|e| e.to_string())?;
        let destination = generated.join(relative);
        fs::create_dir_all(
            destination
                .parent()
                .ok_or("generated source parent missing")?,
        )
        .map_err(|e| e.to_string())?;
        let label = format!(
            "{}-hipify",
            relative.to_string_lossy().replace(['/', '\\'], "_")
        );
        translate_one(
            &translator,
            &perl,
            &tool_identity,
            &source,
            &destination,
            output,
            &label,
        )?;
    }
    if tool_identity
        != format!(
            "{}:{}",
            super::super::artifact_metadata(&translator)?.sha256,
            super::super::artifact_metadata(&perl)?.sha256
        )
    {
        return Err("HIPIFY script/interpreter changed during translation".into());
    }
    // Bound native children to Cargo's requested parallelism (at most two on
    // this local preparation path). Every started child is joined on failure.
    let jobs = std::env::var("NUM_JOBS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 2);
    let mut objects = Vec::new();
    for chunk in sources.chunks(jobs) {
        let outcomes = std::thread::scope(|scope| {
            let workers = chunk
                .iter()
                .map(|source| {
                    let generated = &generated;
                    scope.spawn(move || {
                        compile_one(root, generated, output, target, source, fixtures)
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .map_err(|_| "HIP compiler worker panicked".to_owned())
                        .and_then(|v| v)
                })
                .collect::<Vec<_>>()
        });
        for outcome in outcomes {
            objects.push(outcome?);
        }
    }
    let archive = output.join("libneoethos_hip_kernels_v1.a");
    // ar rcs preserves unspecified OLD members. This exact generated artifact
    // must start empty, including after shrinking the TU set in the same OUT_DIR.
    match fs::remove_file(&archive) {
        Ok(()) => (),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(format!("cannot replace generated HIP archive: {error}")),
    }
    let mut arguments = vec!["rcs".into(), archive.to_string_lossy().into_owned()];
    arguments.extend(
        objects
            .iter()
            .map(|object| object.to_string_lossy().into_owned()),
    );
    super::run(
        &root.join("llvm/bin/llvm-ar"),
        &arguments,
        output,
        "kernel-archive",
    )?;
    let members = super::run(
        &root.join("llvm/bin/llvm-ar"),
        &["t".into(), archive.to_string_lossy().into_owned()],
        output,
        "kernel-archive-members",
    )?;
    let expected = objects
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if members.lines().collect::<Vec<_>>()
        != expected.iter().map(String::as_str).collect::<Vec<_>>()
    {
        return Err("HIP archive members do not match the exact current TU set".into());
    }
    let mut pins = Vec::new();
    // Re-discover the complete quoted-include graph as well as every byte hash.
    if project_sources(sources)? != records {
        return Err("HIP source closure changed during build".into());
    }
    for (path, hash) in &records {
        pins.push(format!(
            "{{\"path\":\"{}\",\"sha256\":\"{hash}\"}}",
            super::super::json_escape(&path.to_string_lossy().replace('\\', "/"))
        ));
    }
    println!("cargo:rustc-link-lib=static:+whole-archive=neoethos_hip_kernels_v1");
    Ok(KernelArchive {
        native_directory: generated.join("native"),
        artifact_sha256: super::super::artifact_metadata(&archive)?.sha256,
        sources_json: format!("[{}]", pins.join(",")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "linux")]
    #[test]
    fn only_an_exact_completed_translation_skips_the_child() {
        let output = std::env::temp_dir().join(format!(
            "neoethos-hip-cache-run-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&output).unwrap();
        let source = output.join("input.cu");
        let translated = output.join("output.cu");
        let translator = output.join("fake-translator.pl");
        fs::write(
            &translator,
            r#"use strict; use warnings;
open(my $input, '<', $ARGV[0]) or die $!;
open(my $output, '>', $ARGV[2]) or die $!;
while (<$input>) { print $output $_; }
close($output) or die $!;
open(my $count, '>>', $ARGV[2] . '.count') or die $!;
print $count "run\n";
close($count) or die $!;
print "[HIPIFY] info: fixture only\n";
"#,
        )
        .unwrap();
        fs::write(&source, "first").unwrap();
        let run = |identity| {
            translate_one(
                &translator,
                Path::new("/usr/bin/perl"),
                identity,
                &source,
                &translated,
                &output,
                "cache-test",
            )
            .unwrap()
        };
        run("script/interpreter1");
        run("script/interpreter1");
        assert_eq!(
            fs::read_to_string(output.join("output.cu.count")).unwrap(),
            "run\n"
        );
        fs::write(&source, "second").unwrap();
        run("script/interpreter1");
        run("script/interpreter2");
        fs::write(output.join("hip-runtime-cache-test.stderr.log"), "altered").unwrap();
        run("script/interpreter2");
        fs::write(&translated, "altered").unwrap();
        run("script/interpreter2");
        assert_eq!(
            fs::read_to_string(output.join("output.cu.count")).unwrap(),
            "run\n".repeat(5)
        );
        assert_eq!(fs::read_to_string(&translated).unwrap(), "second");
        for entry in fs::read_dir(&output).unwrap() {
            fs::remove_file(entry.unwrap().path()).unwrap();
        }
        fs::remove_dir(output).unwrap();
    }
    #[test]
    fn translation_receipt_requires_exact_nonempty_output_and_both_logs() {
        let output = std::env::temp_dir().join(format!(
            "neoethos-hip-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&output).unwrap();
        let destination = output.join("translated.cu");
        let stdout = output.join("hip-runtime-test.stdout.log");
        let stderr = output.join("hip-runtime-test.stderr.log");
        fs::write(&destination, "source").unwrap();
        fs::write(&stdout, "").unwrap();
        fs::write(&stderr, "warning\n").unwrap();
        let initial = translation_stamp("inputs", &destination, &output, "test").unwrap();
        assert_ne!(
            initial,
            translation_stamp("other tool/source", &destination, &output, "test").unwrap()
        );
        fs::write(&stderr, "different warning\n").unwrap();
        assert_ne!(
            initial,
            translation_stamp("inputs", &destination, &output, "test").unwrap()
        );
        fs::write(&stderr, "warning\n").unwrap();
        assert_eq!(
            initial,
            translation_stamp("inputs", &destination, &output, "test").unwrap()
        );
        fs::write(&destination, "different source").unwrap();
        assert_ne!(
            initial,
            translation_stamp("inputs", &destination, &output, "test").unwrap()
        );
        fs::write(&destination, "").unwrap();
        assert!(translation_stamp("inputs", &destination, &output, "test").is_err());
        for file in [destination, stdout, stderr] {
            fs::remove_file(file).unwrap();
        }
        fs::remove_dir(output).unwrap();
    }
    #[test]
    fn translator_error_prefixes_are_not_success_or_warnings() {
        for line in [
            "error: unsupported",
            "  fatal error: missing",
            "file.cu:3: error: bad",
            "[HIPIFY] error: cannot open file",
            " [HIPIFY] fatal error: malformed",
        ] {
            assert!(has_error_diagnostic(line), "{line}");
        }
        for line in [
            "[HIPIFY] info: converted",
            "file.cu:3: warning: unsupported CUDA-only name",
            "[HIPIFY] info: CONVERTED refs by names:",
            "  control_error => control_error: 1",
        ] {
            assert!(!has_error_diagnostic(line), "{line}");
        }
    }
}
