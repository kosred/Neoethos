from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)

cube_path = Path("crates/neoethos-search/src/cubecl_eval.rs")
cube = cube_path.read_text(encoding="utf-8")
cube = replace_once(
    cube,
    "    use std::sync::atomic::{AtomicU64, Ordering};\n",
    "    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};\n",
    "telemetry AtomicBool import",
)
cube = replace_once(
    cube,
    "    static GPU_CALLS: AtomicU64 = AtomicU64::new(0);\n",
    "    static ENABLED: AtomicBool = AtomicBool::new(false);\n    static GPU_CALLS: AtomicU64 = AtomicU64::new(0);\n",
    "telemetry enabled flag",
)
cube = replace_once(
    cube,
    "    pub(super) fn reset() {\n        for counter in [\n",
    "    pub(super) fn reset() {\n        ENABLED.store(true, Ordering::Relaxed);\n        clear();\n    }\n\n    pub(super) fn disable() {\n        ENABLED.store(false, Ordering::Relaxed);\n        clear();\n    }\n\n    fn clear() {\n        for counter in [\n",
    "telemetry reset/disable split",
)
for signature, label in [
    ("    pub(super) fn record_call() {\n", "record_call guard"),
    ("    pub(super) fn record_streamed_dataset_upload(bytes: usize) {\n", "streamed upload guard"),
    ("    pub(super) fn record_gene_upload(bytes: usize) {\n", "gene upload guard"),
    ("    pub(super) fn record_resident_hit() {\n", "resident hit guard"),
    ("    pub(super) fn record_resident_miss(bytes: usize) {\n", "resident miss guard"),
    ("    pub(super) fn record_readback(use_fused: bool, compact_bytes: usize, dense_bytes: usize) {\n", "readback guard"),
]:
    cube = replace_once(
        cube,
        signature,
        signature + "        if !ENABLED.load(Ordering::Relaxed) {\n            return;\n        }\n",
        label,
    )
cube = replace_once(
    cube,
    "pub(crate) fn reset_cubecl_transfer_telemetry() {\n    transfer_telemetry::reset();\n}\n\n",
    "pub(crate) fn reset_cubecl_transfer_telemetry() {\n    transfer_telemetry::reset();\n}\n\npub(crate) fn disable_cubecl_transfer_telemetry() {\n    transfer_telemetry::disable();\n}\n\n",
    "telemetry disable export",
)
cube_path.write_text(cube, encoding="utf-8")

proto_path = Path("crates/neoethos-search/src/gpu_native/prototype_a.rs")
proto = proto_path.read_text(encoding="utf-8")
proto = replace_once(
    proto,
    "pub fn reset_prototype_a_telemetry() {\n",
    "pub fn disable_prototype_a_telemetry() {\n    #[cfg(feature = \"gpu\")]\n    crate::cubecl_eval::disable_cubecl_transfer_telemetry();\n}\n\npub fn reset_prototype_a_telemetry() {\n",
    "Prototype A telemetry disable API",
)
proto_path.write_text(proto, encoding="utf-8")
print("made Prototype A transfer telemetry opt-in for diagnostic passes")
