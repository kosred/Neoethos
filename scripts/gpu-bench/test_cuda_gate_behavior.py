#!/usr/bin/env python3
"""Behavior tests for non-vacuous CUDA and memcheck output gates."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts" / "gpu-bench" / "run_cuda_test_gate.sh"
SANITIZER_GATE = ROOT / "scripts" / "gpu-bench" / "run_compute_sanitizer_gate.sh"
MEMCHECK_VALIDATION = (
    ROOT / "scripts" / "gpu-bench" / "run_cuda_memcheck_validation.sh"
)
HARDWARE_GATE = ROOT / "scripts" / "gpu-bench" / "check_cuda_hardware.sh"
FAKE_SANITIZER = r'''#!/usr/bin/env bash
set -euo pipefail
sanitizer_log=""
while (( $# )); do
  case "$1" in
    --log-file) sanitizer_log="$2"; shift 2 ;;
    --tool|--leak-check|--target-processes|--require-cuda-init|--error-exitcode) shift 2 ;;
    *) break ;;
  esac
done
printf '%s\n' "${FAKE_SANITIZER_SUMMARY:?}" > "$sanitizer_log"
"$@"
exit "${FAKE_SANITIZER_EXIT:-0}"
'''
FAKE_MEMCHECK_CARGO = r'''#!/usr/bin/env bash
set -euo pipefail
[[ "${NEOETHOS_RUN_CUDA_SEARCH_TESTS:-}" == "1" ]]
[[ "${NEOETHOS_REQUIRE_GPU:-}" == "1" ]]
args=" $* "
case "$args" in
  *" -p neoethos-gpu-cuda "*" tests::real_cuda_smoke_executes_f64_first_hit_without_narrowing "*) passed=1 ;;
  *" --features gpu-b-native "*" eval::trailing_parity_tests:: "*) passed=3 ;;
  *" --features gpu-cuda "*" eval::gpu_cpu_parity_tests:: "*) passed=7 ;;
  *" --features gpu-cuda "*" gpu_native::prototype_a::tests::direct_prototype_a_engine_is_resident_and_matches_cpu_fixture "*) passed=1 ;;
  *" --features gpu-cuda "*" gpu_native::prototype_c_engine::device_tests:: "*) passed=7 ;;
  *) printf 'unexpected fake cargo command: %s\n' "$args" >&2; exit 78 ;;
esac
printf 'running %s tests\n' "$passed"
printf 'test result: ok. %s passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n' "$passed"
'''
FAKE_NVIDIA_SMI = r'''#!/usr/bin/env bash
set -euo pipefail
printf 'search=%s,require=%s\n' \
  "${NEOETHOS_RUN_CUDA_SEARCH_TESTS:-unset}" \
  "${NEOETHOS_REQUIRE_GPU:-unset}"
trap 'exit 0' TERM INT
while :; do sleep 0.1; done
'''


def bash_path() -> str | None:
    if os.name == "nt":
        git_bash = pathlib.Path(r"C:\Program Files\Git\bin\bash.exe")
        if git_bash.exists():
            return str(git_bash)
    return shutil.which("bash")


def cargo_result(passed: int, ignored: int = 0, extra: str = "") -> str:
    selected = passed + ignored
    noun = "test" if selected == 1 else "tests"
    return (
        f"running {selected} {noun}\n"
        f"{extra}"
        f"test result: ok. {passed} passed; 0 failed; {ignored} ignored; "
        "0 measured; 0 filtered out; finished in 0.01s\n"
    )


class CudaGateBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bash = bash_path()
        if cls.bash is None:
            raise unittest.SkipTest("bash is unavailable")

    def run_gate(
        self, expected_passed: int, expected_ignored: int, output: str
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="neoethos-cuda-gate-") as raw:
            log = pathlib.Path(raw) / "gate.log"
            return subprocess.run(
                [
                    self.bash,
                    str(GATE),
                    str(log),
                    str(expected_passed),
                    str(expected_ignored),
                    self.bash,
                    "-c",
                    "printf '%s' \"$1\"",
                    "bash",
                    output,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

    def test_accepts_only_the_exact_passing_and_ignored_counts(self) -> None:
        completed = self.run_gate(67, 2, cargo_result(67, 2))
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("passed=67 ignored=2 selected=69", completed.stdout)

    def test_zero_tests_cannot_green_a_required_filter(self) -> None:
        completed = self.run_gate(3, 0, cargo_result(0))
        self.assertEqual(completed.returncode, 90, completed.stdout)
        self.assertIn("test-count mismatch", completed.stdout)

    def test_skip_fallback_and_substitution_text_are_each_fatal(self) -> None:
        for diagnostic in (
            "SKIPPED real CUDA test\n",
            "CPU fallback used\n",
            "substituted another engine\n",
        ):
            with self.subTest(diagnostic=diagnostic):
                completed = self.run_gate(1, 0, cargo_result(1, extra=diagnostic))
                self.assertEqual(completed.returncode, 87, completed.stdout)

    def test_missing_compute_sanitizer_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="neoethos-empty-path-") as raw:
            env = os.environ.copy()
            env["PATH"] = raw
            completed = subprocess.run(
                [
                    self.bash,
                    str(SANITIZER_GATE),
                    str(pathlib.Path(raw) / "tests.log"),
                    str(pathlib.Path(raw) / "memcheck.log"),
                    "3",
                    "0",
                    "true",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        self.assertEqual(completed.returncode, 69, completed.stdout)
        self.assertIn("compute-sanitizer is required", completed.stdout)

    def run_fake_sanitizer(
        self, summary: str, sanitizer_exit: int = 0
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="neoethos-fake-sanitizer-") as raw:
            tmp = pathlib.Path(raw)
            fake = tmp / "compute-sanitizer"
            fake.write_bytes(FAKE_SANITIZER.encode("utf-8"))
            fake.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = str(tmp) + os.pathsep + env.get("PATH", "")
            env["FAKE_SANITIZER_SUMMARY"] = summary
            env["FAKE_SANITIZER_EXIT"] = str(sanitizer_exit)
            return subprocess.run(
                [
                    self.bash,
                    str(SANITIZER_GATE),
                    str(tmp / "tests.log"),
                    str(tmp / "memcheck.log"),
                    "3",
                    "0",
                    self.bash,
                    "-c",
                    "printf '%s' \"$1\"",
                    "bash",
                    cargo_result(3),
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

    def test_memcheck_requires_explicit_zero_errors_and_zero_leaked_bytes(self) -> None:
        clean_summary = (
            "========= ERROR SUMMARY: 0 errors\n"
            "========= LEAK SUMMARY: 0 bytes leaked"
        )
        clean = self.run_fake_sanitizer(clean_summary)
        self.assertEqual(clean.returncode, 0, clean.stdout)

        nonzero_command = self.run_fake_sanitizer(clean_summary, sanitizer_exit=86)
        self.assertEqual(nonzero_command.returncode, 86, nonzero_command.stdout)
        self.assertIn("exited 86 despite clean summaries", nonzero_command.stdout)

        errored_and_leaked = self.run_fake_sanitizer(
            "========= ERROR SUMMARY: 16 errors\n"
            "========= LEAK SUMMARY: 839,910,208 bytes leaked",
            sanitizer_exit=86,
        )
        self.assertEqual(errored_and_leaked.returncode, 92, errored_and_leaked.stdout)
        self.assertIn("did not report ERROR SUMMARY: 0 errors", errored_and_leaked.stdout)

        leaked_only = self.run_fake_sanitizer(
            "========= ERROR SUMMARY: 0 errors\n"
            "========= LEAK SUMMARY: 1 byte leaked",
            sanitizer_exit=86,
        )
        self.assertEqual(leaked_only.returncode, 93, leaked_only.stdout)
        self.assertIn("did not report LEAK SUMMARY: 0 bytes leaked", leaked_only.stdout)

    def test_memcheck_inventory_runs_every_exact_binary_without_export_scope_loss(self) -> None:
        with tempfile.TemporaryDirectory(prefix="neoethos-cuda-memcheck-") as raw:
            tmp = pathlib.Path(raw)
            for name, source in (
                ("compute-sanitizer", FAKE_SANITIZER),
                ("cargo", FAKE_MEMCHECK_CARGO),
                ("nvidia-smi", FAKE_NVIDIA_SMI),
            ):
                executable = tmp / name
                executable.write_bytes(source.encode("utf-8"))
                executable.chmod(0o755)
            log_dir = tmp / "logs"
            env = os.environ.copy()
            env["PATH"] = str(tmp) + os.pathsep + env.get("PATH", "")
            env["FAKE_SANITIZER_SUMMARY"] = (
                "========= ERROR SUMMARY: 0 errors\n"
                "========= LEAK SUMMARY: 0 bytes leaked"
            )
            env["NEOETHOS_GPU_SANITIZER_LOG_DIR"] = str(log_dir)
            env["NEOETHOS_GPU_TELEMETRY_LOG"] = str(log_dir / "telemetry.csv")
            completed = subprocess.run(
                [self.bash, str(MEMCHECK_VALIDATION), "all"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=10,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            expected = {
                "native-abi-f64": 1,
                "native-b": 3,
                "cubecl-population": 7,
                "prototype-a-direct": 1,
                "prototype-c-device": 7,
            }
            for label, passed in expected.items():
                with self.subTest(label=label):
                    test_log = (log_dir / f"{label}-tests.log").read_text(
                        encoding="utf-8"
                    )
                    self.assertIn(f"{passed} passed", test_log)
            telemetry = (log_dir / "telemetry.csv").read_text(encoding="utf-8")
            self.assertIn("search=1,require=1", telemetry)

    def run_hardware_gate(
        self, name: str, compute_capability: str, vram_mib: int, *, override: bool = False
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in (
            "NEOETHOS_EXPECT_GPU_SUBSTRING",
            "NEOETHOS_MIN_COMPUTE_CAPABILITY",
            "NEOETHOS_MIN_VRAM_MIB",
            "NEOETHOS_ALLOW_OTHER_GPU",
        ):
            env.pop(key, None)
        if override:
            env["NEOETHOS_ALLOW_OTHER_GPU"] = "1"
        return subprocess.run(
            [self.bash, str(HARDWARE_GATE), name, compute_capability, str(vram_mib)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_default_hardware_policy_accepts_real_3090_and_a6000_shapes(self) -> None:
        for name, vram_mib in (
            ("NVIDIA GeForce RTX 3090", 24_576),
            ("NVIDIA RTX A6000", 49_140),
        ):
            with self.subTest(name=name):
                completed = self.run_hardware_gate(name, "8.6", vram_mib)
                self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_default_hardware_policy_rejects_unknown_lower_cc_and_lower_vram(self) -> None:
        cases = (
            ("NVIDIA RTX 4090", "8.9", 24_576, 21),
            ("NVIDIA GeForce RTX 3090", "8.5", 24_576, 28),
            ("NVIDIA GeForce RTX 3090", "8.6", 23_999, 22),
        )
        for name, cc, vram, expected in cases:
            with self.subTest(name=name, cc=cc, vram=vram):
                completed = self.run_hardware_gate(name, cc, vram)
                self.assertEqual(completed.returncode, expected, completed.stdout)

    def test_explicit_hardware_override_is_loud_and_allows_an_unknown_lower_card(self) -> None:
        completed = self.run_hardware_gate("Unknown NVIDIA", "7.5", 12_000, override=True)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("WARNING", completed.stdout)


if __name__ == "__main__":
    unittest.main()
