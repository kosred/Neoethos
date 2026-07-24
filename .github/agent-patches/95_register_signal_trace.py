from pathlib import Path

path = Path("crates/neoethos-search/src/gpu_native/mod.rs")
text = path.read_text(encoding="utf-8")
needle = "pub mod semantics;\n"
insert = "#[cfg(any(feature = \"gpu-cuda\", feature = \"gpu-vulkan\"))]\npub mod signal_trace_gpu;\npub mod semantics;\n"
if text.count(needle) != 1:
    raise RuntimeError(f"signal trace module anchor: expected one match, found {text.count(needle)}")
path.write_text(text.replace(needle, insert, 1), encoding="utf-8")
print("registered separate GPU signal-trace specialization")
