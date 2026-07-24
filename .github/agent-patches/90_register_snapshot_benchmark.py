from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

replace_once(
    Path("crates/neoethos-search/src/gpu_native/mod.rs"),
    "pub mod ranking;\n",
    "pub mod ranking;\npub mod snapshot_fixture;\n",
    "snapshot fixture module",
)
replace_once(
    Path("crates/neoethos-cli/src/main.rs"),
    "mod gpu_bench;\n",
    "mod gpu_bench;\nmod gpu_bench_snapshot;\n",
    "snapshot bench CLI module",
)
replace_once(
    Path("crates/neoethos-cli/src/gpu_bench.rs"),
    "pub fn run(args: &[String]) -> Result<()> {\n",
    "pub fn run(args: &[String]) -> Result<()> {\n    if args.iter().any(|arg| arg == \"--execute-snapshot\") {\n        return crate::gpu_bench_snapshot::run(args);\n    }\n",
    "snapshot bench dispatch",
)
print("registered versioned real-data snapshot fixture and executable CLI path")
