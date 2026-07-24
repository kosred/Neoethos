from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)

root = Path("Cargo.toml")
text = root.read_text(encoding="utf-8")
text = replace_once(
    text,
    '  "crates/neoethos-gpu-contracts",\n',
    '  "crates/neoethos-gpu-contracts",\n  "crates/neoethos-gpu-cuda",\n',
    "CUDA workspace member",
)
root.write_text(text, encoding="utf-8")

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
