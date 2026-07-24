from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)

root = Path("Cargo.toml")
text = root.read_text(encoding="utf-8")
member = '  "crates/neoethos-gpu-contracts",\n'
if text.count(member) != 2:
    raise RuntimeError(
        f"CUDA workspace member: expected contracts in members and default-members, found {text.count(member)}"
    )
text = text.replace(
    member,
    member + '  "crates/neoethos-gpu-cuda",\n',
    1,
)
root.write_text(text, encoding="utf-8")

for script in (
    Path("scripts/gpu-bench/preflight.sh"),
    Path("scripts/gpu-bench/prepare_worktrees.sh"),
    Path("scripts/gpu-bench/run_matrix.py"),
    Path("scripts/gpu-bench/collate.py"),
):
    if not script.is_file():
        raise RuntimeError(f"missing rented-GPU script: {script}")
    script.chmod(0o755)

workflow = Path(".github/workflows/agent-stage1.yml")
text = workflow.read_text(encoding="utf-8")
text = replace_once(
    text,
    """      - name: Test search crate
""",
    """      - name: Test CUDA ABI scaffold
        shell: bash
        run: |
          set -o pipefail
          cargo test -p neoethos-gpu-cuda 2>&1 | tee /tmp/stage1-diagnostics/cuda-scaffold-test.log

      - name: Lint rented-GPU run kit
        shell: bash
        run: |
          set -o pipefail
          bash -n scripts/gpu-bench/preflight.sh scripts/gpu-bench/prepare_worktrees.sh
          python3 -m py_compile scripts/gpu-bench/run_matrix.py scripts/gpu-bench/collate.py

      - name: Test search crate
""",
    "CUDA and run-kit CI steps",
)
workflow.write_text(text, encoding="utf-8")
print("registered CUDA ABI scaffold and fail-fast rented-GPU run kit")
