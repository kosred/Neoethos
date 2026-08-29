use std::fs;
use std::path::{Path, PathBuf};

const RETIRED_FILES: [&str; 3] = [
    "prototype_c_gpu.rs",
    "signal_trace_gpu.rs",
    "trade_trace_gpu.rs",
];

const RETIRED_SYMBOLS: [&str; 5] = [
    "try_prototype_c_gpu_first_hit",
    "gpu_signal_trace",
    "cpu_signal_trace",
    "gpu_trade_trace",
    "cpu_trade_trace",
];

fn collect_rust_files(root: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).expect("read source directory") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

#[test]
fn superseded_gpu_diagnostic_kernels_cannot_return() {
    let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let gpu_native = source_root.join("gpu_native");

    for retired in RETIRED_FILES {
        assert!(
            !gpu_native.join(retired).exists(),
            "superseded self-mirroring GPU diagnostic still exists: {retired}"
        );
    }

    let module_source =
        fs::read_to_string(gpu_native.join("mod.rs")).expect("read GPU-native module registry");
    for retired in [
        "pub mod prototype_c_gpu;",
        "pub mod signal_trace_gpu;",
        "mod trade_trace_gpu;",
    ] {
        assert!(
            !module_source.contains(retired),
            "GPU-native registry still exposes superseded module `{retired}`"
        );
    }
    assert!(
        module_source.contains("pub mod prototype_c_engine;"),
        "the resident Prototype C engine must remain the sole Prototype C device authority"
    );

    let mut rust_files = Vec::new();
    collect_rust_files(&source_root, &mut rust_files);
    for path in rust_files {
        let source = fs::read_to_string(&path).expect("read Rust source");
        for retired in RETIRED_SYMBOLS {
            assert!(
                !source.contains(retired),
                "superseded GPU diagnostic symbol `{retired}` remains in {}",
                path.display()
            );
        }
    }
}
