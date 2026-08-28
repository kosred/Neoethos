use std::path::{Path, PathBuf};

fn rust_sources(root: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).expect("read vector-ta source directory") {
        let path = entry.expect("read vector-ta source entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs")
            && path.file_name().and_then(|value| value.to_str()) != Some("rust_only_contract.rs")
        {
            out.push(path);
        }
    }
}

#[test]
fn vector_ta_distribution_is_rust_only_and_has_no_unsigned_pattern_abi() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for manifest_name in ["Cargo.toml", "Cargo.toml.orig"] {
        let manifest = std::fs::read_to_string(root.join(manifest_name))
            .unwrap_or_else(|error| panic!("read {manifest_name}: {error}"));
        for forbidden in [
            concat!("cd", "ylib"),
            concat!("py", "thon ="),
            concat!("wa", "sm ="),
            concat!("py", "o3"),
            concat!("num", "py"),
            concat!("wasm-", "bindgen"),
            concat!("serde-wasm-", "bindgen"),
            concat!("js-", "sys"),
            concat!("wasm", "bind"),
            concat!("wasm-", "pack"),
            concat!("profile.", "wasm"),
        ] {
            assert!(
                !manifest.contains(forbidden),
                "Rust-only {manifest_name} still contains `{forbidden}`"
            );
        }
    }
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo.toml");
    let manifest = manifest.replace("\r\n", "\n");
    assert!(
        manifest.contains("crate-type = [\n    \"rlib\",\n]"),
        "Rust-only vector-ta must publish only an rlib"
    );

    for removed in [
        root.join("src/bindings/python.rs"),
        root.join("src/bindings/wasm.rs"),
        root.join("src/utilities/dlpack_cuda.rs"),
    ] {
        assert!(
            !removed.exists(),
            "dead language binding remains: {}",
            removed.display()
        );
    }

    let mut sources = Vec::new();
    rust_sources(&root.join("src"), &mut sources);
    rust_sources(&root.join("benches"), &mut sources);
    let forbidden_source_markers = [
        concat!("feature = \"py", "thon\""),
        concat!("feature = \"wa", "sm\""),
        concat!("py", "o3::"),
        concat!("num", "py::"),
        concat!("wasm_", "bindgen"),
        concat!("serde_wasm_", "bindgen"),
        concat!("js_", "sys"),
        concat!("extern \"C\" ", "fn"),
        concat!("pub mod bind", "ings"),
        concat!("IndicatorCuda", "BitmaskRequest"),
        concat!("IndicatorCudaDevice", "BitmaskRequest"),
        concat!("PatternRecognitionCuda", "BitmaskOutput"),
        concat!("DevicePattern", "BitmaskU64"),
        concat!("pattern_recognition_cuda_", "bitmask"),
        concat!("compute_native_matrix_", "bitmask_u64"),
        concat!("pack_matrix_u8_", "device_into"),
        concat!("pack_matrix_u8_", "host"),
        concat!("feature = \"cu", "da\""),
        concat!("CARGO_FEATURE_CU", "DA\""),
    ];
    for source_path in sources {
        let source = std::fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", source_path.display()));
        for forbidden in forbidden_source_markers {
            assert!(
                !source.contains(forbidden),
                "Rust-only source {} still contains `{forbidden}`",
                source_path.display()
            );
        }
    }

    let packed_kernel = concat!("pattern_pack_u8_", "to_u64_kernel");
    for relative in ["kernels/cuda/pattern_recognition_kernel.cu"] {
        let source = std::fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            !source.contains(packed_kernel),
            "superseded unsigned pattern kernel remains in {relative}"
        );
    }
}

#[test]
fn vector_ta_cuda_distribution_is_native_sass_only() {
    fn record_forbidden(violations: &mut Vec<String>, surface: &str, source: &str, marker: &str) {
        if source.contains(marker) {
            violations.push(format!("{surface} still contains `{marker}`"));
        }
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();

    for manifest_name in ["Cargo.toml", "Cargo.toml.orig"] {
        let manifest = std::fs::read_to_string(root.join(manifest_name))
            .unwrap_or_else(|error| panic!("read {manifest_name}: {error}"));
        for marker in [
            concat!("cuda-build-", "ptx"),
            concat!("/kernels/", "ptx/**"),
            concat!("\ncu", "da = ["),
            concat!("cuda-build-native = [\"cu", "da\"]"),
            concat!("required-features = [\"cu", "da\"]"),
        ] {
            record_forbidden(&mut violations, manifest_name, &manifest, marker);
        }
        if !manifest.contains(concat!("cuda-build-", "native")) {
            violations.push(format!(
                "{manifest_name} does not expose the reviewed `cuda-build-native` feature"
            ));
        }
    }

    for removed in [
        root.join(concat!("kernels/", "ptx")),
        root.join("kernels/fatbin"),
    ] {
        if removed.exists() {
            violations.push(format!(
                "superseded CUDA artifact tree remains: {}",
                removed.display()
            ));
        }
    }

    let read = |relative: &str| {
        std::fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let build = read("build.rs");
    let build_support = read("build_support/native_cuda_build.rs");
    let build_surfaces = format!("{build}\n{build_support}");
    for marker in [
        concat!("CudaArtifactKind::", "Ptx"),
        concat!("CudaArtifactKind::", "Fatbin"),
        concat!("stage_prebuilt_", "ptx"),
        concat!("VECTOR_TA_PREBUILT_", "PTX_DIR"),
        concat!("VECTOR_TA_PREBUILD_", "PTX_DIR"),
        concat!("VECTOR_TA_PREBUILT_", "FATBIN_DIR"),
        concat!("VECTOR_TA_CUDA_", "PTX_ARCH"),
        concat!("code=", "compute_"),
        concat!("\"-", "ptx\""),
        concat!("\"-", "fatbin\""),
        concat!("placeholder ", "PTX"),
        concat!("ptx", "_name"),
        concat!("kernel", "_stem"),
        concat!("CARGO_FEATURE_CU", "DA\""),
        concat!("env::var(\"CUDA_", "ARCH\")"),
    ] {
        record_forbidden(
            &mut violations,
            "build.rs + build_support/native_cuda_build.rs",
            &build_surfaces,
            marker,
        );
    }
    if !build_surfaces.contains(concat!("\"--", "cubin\""))
        && !build_surfaces.contains(concat!("\"-", "cubin\""))
    {
        violations.push(
            "native CUDA build surfaces have no reviewed nvcc cubin-output argument for native SASS"
                .to_owned(),
        );
    }

    let readme = read("README.md");
    for marker in [
        concat!("cuda-build-", "ptx"),
        concat!("VECTOR_TA_PREBUILT_", "PTX_DIR"),
        concat!("prebuilt ", "PTX"),
    ] {
        record_forbidden(&mut violations, "README.md", &readme, marker);
    }

    let mut cuda_sources = Vec::new();
    rust_sources(&root.join("src/cuda"), &mut cuda_sources);
    let mut cubin_loader_calls = 0usize;
    for source_path in cuda_sources {
        let source = std::fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", source_path.display()));
        let surface = source_path
            .strip_prefix(root)
            .unwrap_or(&source_path)
            .display()
            .to_string();
        for marker in [
            concat!("Module::from_", "ptx"),
            concat!("Module::from_", "fatbin"),
            concat!("ModuleJit", "Option"),
            concat!("Opt", "Level"),
            concat!("COMPILED_", "PTX_ARCH"),
            concat!("VECTOR_TA_CUDA_FORCE_", "PTX"),
            concat!("VECTOR_TA_CUDA_FORCE_", "FATBIN"),
            concat!("PROBE_", "PTXS"),
            concat!("load_", "ptx_module"),
            concat!(".pt", "x\""),
            concat!(".fat", "bin\""),
        ] {
            record_forbidden(&mut violations, &surface, &source, marker);
        }
        cubin_loader_calls += source.matches(concat!("Module::from_", "cubin")).count();
    }
    if cubin_loader_calls != 1 {
        violations.push(format!(
            "native CUDA runtime must have exactly one central Module::from_cubin call; found \
             {cubin_loader_calls}"
        ));
    }

    assert!(
        violations.is_empty(),
        "vector-ta CUDA is not native-SASS-only:\n  - {}",
        violations.join("\n  - ")
    );
}

#[test]
fn rust_native_cuda_build_parity_surfaces_are_registered() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let support = root.join("build_support/native_cuda_build.rs");
    let harness = root.join("tests/native_cuda_build_harness.rs");

    assert!(
        support.is_file(),
        "the Rust-native CUDA build scheduler support is not implemented: {}",
        support.display()
    );
    assert!(
        harness.is_file(),
        "the Rust-native CUDA fake-tool integration harness is not implemented: {}",
        harness.display()
    );

    for manifest_name in ["Cargo.toml", "Cargo.toml.orig"] {
        let manifest = std::fs::read_to_string(root.join(manifest_name))
            .unwrap_or_else(|error| panic!("read {manifest_name}: {error}"));
        for required in [
            "/build_support/**/*.rs",
            "/tests/native_cuda_build_harness.rs",
            "name = \"native_cuda_build_harness\"",
            "path = \"tests/native_cuda_build_harness.rs\"",
            "harness = false",
        ] {
            assert!(
                manifest.contains(required),
                "Rust parity manifest {manifest_name} is missing `{required}`"
            );
        }
    }
}

#[test]
fn superseded_python_cuda_build_tools_are_absent() {
    let vector_ta = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository = vector_ta
        .parent()
        .and_then(Path::parent)
        .expect("vendored vector-ta has a repository parent");
    let support = vector_ta.join("build_support/native_cuda_build.rs");
    let harness = vector_ta.join("tests/native_cuda_build_harness.rs");
    assert!(support.is_file(), "Rust build support must precede cleanup");
    assert!(
        harness.is_file(),
        "Rust integration harness must precede cleanup"
    );

    for removed in [
        repository.join("scripts/gpu-bench/test_vector_ta_nvcc_build_contract.py"),
        repository.join("scripts/gpu-bench/test_vector_ta_nvcc_scheduler_integration.py"),
        repository.join("scripts/gpu-bench/nvcc_timeline.py"),
    ] {
        assert!(
            !removed.exists(),
            "superseded Python CUDA build tool remains after Rust parity: {}",
            removed.display()
        );
    }

    let active_surfaces = [
        repository.join("scripts/gpu-bench/README.md"),
        repository.join("scripts/gpu-bench/run_rented.sh"),
        vector_ta.join("Cargo.toml"),
        vector_ta.join("Cargo.toml.orig"),
        vector_ta.join("README.md"),
        vector_ta.join("build.rs"),
        support.clone(),
        harness.clone(),
    ];
    for surface in active_surfaces {
        let source = std::fs::read_to_string(&surface)
            .unwrap_or_else(|error| panic!("read active surface {}: {error}", surface.display()));
        for forbidden in [
            "test_vector_ta_nvcc_build_contract.py",
            "test_vector_ta_nvcc_scheduler_integration.py",
            "nvcc_timeline.py",
            concat!("NEOETHOS_REAL_", "NVCC"),
            concat!("NEOETHOS_NVCC_", "TRACE_DIR"),
        ] {
            assert!(
                !source.contains(forbidden),
                "active surface {} still references superseded Python tool marker `{forbidden}`",
                surface.display()
            );
        }
    }

    for replacement in [support, harness] {
        let source = std::fs::read_to_string(&replacement)
            .unwrap_or_else(|error| panic!("read {}: {error}", replacement.display()));
        for forbidden in [
            concat!("py", "thon"),
            concat!("NEOETHOS_REAL_", "NVCC"),
            concat!("NEOETHOS_NVCC_", "TRACE_DIR"),
        ] {
            assert!(
                !source
                    .to_ascii_lowercase()
                    .contains(&forbidden.to_ascii_lowercase()),
                "Rust replacement {} retains superseded marker `{forbidden}`",
                replacement.display()
            );
        }
    }
}
