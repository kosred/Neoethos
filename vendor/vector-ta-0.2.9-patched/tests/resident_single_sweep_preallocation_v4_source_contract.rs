use std::fs;
use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("vendor/vector-ta-0.2.9-patched"))
}

fn wrapper() -> String {
    fs::read_to_string(manifest_dir().join(Path::new("src/cuda/neoethos_f64_wrapper.rs")))
        .expect("read VectorTA f64 wrapper")
}

fn cuda_module() -> String {
    fs::read_to_string(manifest_dir().join(Path::new("src/cuda/mod.rs")))
        .expect("read VectorTA CUDA module exports")
}

fn cargo_manifest() -> String {
    fs::read_to_string(manifest_dir().join(Path::new("Cargo.toml")))
        .expect("read VectorTA Cargo manifest")
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn braced_body_after<'a>(source: &'a str, marker: &str) -> &'a str {
    let start = source.find(marker).expect("function marker");
    let open = source[start..].find('{').expect("function body") + start;
    let mut depth = 0_usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open + 1..open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated function body")
}

#[test]
fn single_sweep_plan_is_move_only_checked_and_bound_before_device_allocation() {
    let source = wrapper();
    let compact = compact(&source);
    for required in [
        "pub struct F64ResidentSingleSweepAllocationPlanV4",
        "pub fn preflight_resident_single_sweep_allocation_v4(",
        "pub fn sweep_resident_preplanned_v4(",
        "expected_preallocation_plan != *preallocation_plan",
        "F64ResidentPreallocationPlanMismatchV4",
        "output_bytes",
        "parameter_i32_bytes",
        "parameter_f64_bytes",
        "scratch_f64_bytes",
        "scratch_i32_bytes",
        "chunk_rows",
    ] {
        assert!(
            source.contains(required),
            "missing exact plan token {required:?}"
        );
    }
    assert!(!source.contains("impl Clone for F64ResidentSingleSweepAllocationPlanV4"));
    assert!(!source.contains("impl Copy for F64ResidentSingleSweepAllocationPlanV4"));
    let declaration = compact
        .split_once("pubstructF64ResidentSingleSweepAllocationPlanV4")
        .expect("single-sweep plan declaration")
        .0
        .rsplit_once("#[derive(")
        .map(|(_, derive)| derive)
        .unwrap_or_default();
    assert!(!declaration.contains("Clone") && !declaration.contains("Copy"));

    let bind = source
        .find("expected_preallocation_plan != *preallocation_plan")
        .expect("preallocation equality guard");
    let first_allocation = source
        .find("DeviceBuffer::<f64>::uninitialized_async(output_elements")
        .expect("resident output allocation");
    assert!(
        bind < first_allocation,
        "plan equality must precede output allocation"
    );
}

#[test]
fn single_sweep_plan_accounts_every_primary_owner_family_without_a_live_probe() {
    let source = wrapper();
    let plan = braced_body_after(
        &source,
        "pub fn preflight_resident_single_sweep_allocation_v4(",
    );
    for required in [
        "std::mem::size_of::<f64>()",
        "std::mem::size_of::<i32>()",
        "F64Kernel::Cci",
        "exact_coefficient_stride_v3",
        "F64Kernel::HalfCausalEstimator",
        "HCE_V2_SCRATCH_F64_ELEMS",
        "HCE_V2_SCRATCH_I32_ELEMS",
    ] {
        assert!(plan.contains(required), "planner omitted {required:?}");
    }
    assert!(!plan.contains("mem_get_info"));
    assert!(!plan.contains("DeviceBuffer"));
    assert!(!plan.contains("from_slice"));
}

#[test]
fn single_sweep_plan_is_exported_through_the_reviewed_cuda_surface() {
    let module = cuda_module();
    let export = module
        .split_once("pub use neoethos_f64_wrapper::{")
        .and_then(|(_, tail)| tail.split_once("};"))
        .map(|(export, _)| export)
        .expect("explicit NeoEthos f64 wrapper export list");
    for required in [
        "F64ResidentSingleSweepAllocationPlanV4",
        "preflight_resident_single_sweep_allocation_v4",
    ] {
        assert!(
            export.contains(required),
            "CUDA surface omitted {required:?}"
        );
    }
    let manifest = cargo_manifest();
    let registration = "[[test]]\nname = \"resident_single_sweep_preallocation_v4_source_contract\"\npath = \"tests/resident_single_sweep_preallocation_v4_source_contract.rs\"";
    assert_eq!(
        manifest.matches(registration).count(),
        1,
        "autotests=false requires one explicit allocation-v4 source-contract registration"
    );
}
