from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "crates/neoethos-cli/src/main.rs"
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    text = text.replace(old, new, 1)


replace_once(
    "mod tui;\n",
    "mod gpu_bench;\nmod tui;\n",
    "gpu bench module declaration",
)
replace_once(
    '        "batch-discover" => cmd_batch_discover(&args[2..]),\n',
    '        "batch-discover" => cmd_batch_discover(&args[2..]),\n        "bench" => gpu_bench::run(&args[2..]),\n',
    "bench command dispatch",
)
replace_once(
    '    println!("  resample --symbol EURUSD --base M1 --target H1 --root data");\n',
    '    println!("  resample --symbol EURUSD --base M1 --target H1 --root data");\n    println!("  bench --dry-run --fixture tiny --prototype a --backend cuda --out cache/gpu-bench/plan.json");\n',
    "bench help line",
)
path.write_text(text, encoding="utf-8")
(root / ".github/agent-patches/commit-message.txt").write_text(
    "feat(cli): add GPU benchmark plan and report command\n",
    encoding="utf-8",
)
