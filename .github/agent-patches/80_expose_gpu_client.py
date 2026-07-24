from pathlib import Path

path = Path("crates/neoethos-search/src/cubecl_eval.rs")
text = path.read_text(encoding="utf-8")
needle = "fn create_gpu_client(device_override: Option<usize>)"
count = text.count(needle)
if count != 2:
    raise RuntimeError(f"expected two backend client factories, found {count}")
text = text.replace(needle, "pub(crate) fn create_gpu_client(device_override: Option<usize>)")
path.write_text(text, encoding="utf-8")
print("exposed compile-time-selected CubeCL client factory to Stage-1 prototype modules")
