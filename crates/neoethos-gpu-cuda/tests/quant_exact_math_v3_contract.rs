use std::fs;
use std::path::{Path, PathBuf};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("neoethos-gpu-cuda lives under crates/")
        .to_path_buf()
}

fn read(path: &str) -> String {
    fs::read_to_string(repository_root().join(path))
        .unwrap_or_else(|error| panic!("read {path}: {error}"))
}

#[test]
fn quant_v3_pins_one_sun_openlibm_cpu_authority_and_immutable_receipt() {
    let source = read("crates/neoethos-data/src/core/quant_exact_math_v3.rs");
    for required in [
        "QUANT_OPENLIBM_COMMIT_V3",
        "82e90aef0657289192efe77be89791c07dea0775",
        "QUANT_OPENLIBM_E_LOG_SOURCE_SHA256_V3",
        "8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD",
        "quant_log_positive_f64_v3",
        "Sun fdlibm/OpenLibm e_log",
        "wide_domain_matches_high_precision_checkpoints",
        "positive finite binary64 domain",
    ] {
        assert!(
            source.contains(required),
            "CPU authority omitted `{required}`"
        );
    }
    assert!(!source.contains(".ln()"));
}

#[test]
fn quant_v3_cuda_mirror_uses_only_explicit_rn_edges_and_no_native_log() {
    let source = read("crates/neoethos-gpu-cuda/native/resident_quant_v3.cu");
    for required in [
        "quant_log_positive_f64_v3",
        "__dadd_rn",
        "__dsub_rn",
        "__dmul_rn",
        "__ddiv_rn",
        "82e90aef0657289192efe77be89791c07dea0775",
        "8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD",
    ] {
        assert!(
            source.contains(required),
            "CUDA authority omitted `{required}`"
        );
    }
    for forbidden in [" log(", "::log(", " log2(", " log10(", "__log"] {
        assert!(
            !source.contains(forbidden),
            "CUDA used native `{forbidden}`"
        );
    }
}
