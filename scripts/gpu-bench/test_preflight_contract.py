#!/usr/bin/env python3
"""Static contract tests for the fail-fast rented-GPU preflight."""

from __future__ import annotations

import pathlib
import unittest


PREFLIGHT = pathlib.Path(__file__).with_name("preflight.sh")
VALIDATION = pathlib.Path(__file__).with_name("run_cuda_validation.sh")
GATE = pathlib.Path(__file__).with_name("run_cuda_test_gate.sh")
SANITIZER_GATE = pathlib.Path(__file__).with_name("run_compute_sanitizer_gate.sh")
MEMCHECK_VALIDATION = pathlib.Path(__file__).with_name("run_cuda_memcheck_validation.sh")
HARDWARE_GATE = pathlib.Path(__file__).with_name("check_cuda_hardware.sh")
ROOT = pathlib.Path(__file__).resolve().parents[2]
SEARCH_EVAL = ROOT / "crates" / "neoethos-search" / "src" / "eval.rs"
PROTOTYPE_C_TESTS = (
    ROOT
    / "crates"
    / "neoethos-search"
    / "src"
    / "gpu_native"
    / "prototype_c_engine"
    / "device_tests.rs"
)
GPU_INDICATORS = ROOT / "crates" / "neoethos-data" / "src" / "core" / "gpu_indicators.rs"


class PreflightContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PREFLIGHT.read_text(encoding="utf-8")
        cls.validation = VALIDATION.read_text(encoding="utf-8") if VALIDATION.exists() else ""
        cls.gate = GATE.read_text(encoding="utf-8") if GATE.exists() else ""
        cls.sanitizer_gate = (
            SANITIZER_GATE.read_text(encoding="utf-8") if SANITIZER_GATE.exists() else ""
        )
        cls.memcheck_validation = (
            MEMCHECK_VALIDATION.read_text(encoding="utf-8")
            if MEMCHECK_VALIDATION.exists()
            else ""
        )
        cls.hardware_gate = (
            HARDWARE_GATE.read_text(encoding="utf-8") if HARDWARE_GATE.exists() else ""
        )
        cls.search_eval = SEARCH_EVAL.read_text(encoding="utf-8")
        cls.prototype_c_tests = PROTOTYPE_C_TESTS.read_text(encoding="utf-8")
        cls.gpu_indicators = GPU_INDICATORS.read_text(encoding="utf-8")

    def test_runs_the_current_non_vacuous_cuda_validation_inventory(self) -> None:
        self.assertIn('run_cuda_validation.sh" all', self.source)
        required_filters = {
            "real_cuda_smoke_executes_f64_first_hit_without_narrowing": "1 0",
            "eval::trailing_parity_tests::": "3 0",
            "eval::gpu_cpu_parity_tests::": "7 0",
            "eval::cubecl_trailing_parity_tests::gpu_cubecl_trailing_stop_matches_cpu": "1 0",
            "cubecl_eval::fused_parity_tests::fused_path_is_byte_identical_to_windowed_path": "1 0",
            "gpu_native::prototype_a::tests::direct_prototype_a_engine_is_resident_and_matches_cpu_fixture": "1 0",
            "gpu_native::prototype_c_engine::device_tests::": "7 0",
            "core::gpu_indicators::tests::": "67 2",
            "core::hpc_ta::tests::gpu_cpu_indicator_sweep_parity": "1 0",
        }
        for test_filter, counts in required_filters.items():
            with self.subTest(test_filter=test_filter):
                self.assertIn(test_filter, self.validation)
                self.assertRegex(
                    self.validation,
                    rf'run_gate\s+[^\n]*\s{counts}\s+[^\n]*\n(?:[^\n]*\n){{0,12}}[^\n]*{test_filter}',
                )

        for retired_filter in (
            "real_cuda_smoke_is_explicitly_gpu_gated",
            "gpu_event_first_hit_matches_reference_when_adapter_is_available",
            "direct_gpu_trace_matches_cpu_when_an_adapter_is_available",
            "direct_trade_trace_levels_four_through_nine_match_cpu",
        ):
            with self.subTest(retired_filter=retired_filter):
                self.assertNotIn(retired_filter, self.source + self.validation)

    def test_compute_sanitizer_is_an_executed_fail_loud_gate(self) -> None:
        for required in (
            "command -v compute-sanitizer",
            "--tool memcheck",
            "--leak-check full",
            "--target-processes all",
            "--error-exitcode",
            "ERROR SUMMARY: 0 errors",
            "LEAK SUMMARY: 0 bytes leaked",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.sanitizer_gate)
        self.assertIn('run_cuda_memcheck_validation.sh" all', self.source)
        for label, count, test_filter in (
            (
                "native-abi-f64",
                "1 0",
                "tests::real_cuda_smoke_executes_f64_first_hit_without_narrowing",
            ),
            ("native-b", "3 0", "eval::trailing_parity_tests::"),
            ("cubecl-population", "7 0", "eval::gpu_cpu_parity_tests::"),
            (
                "prototype-a-direct",
                "1 0",
                "gpu_native::prototype_a::tests::direct_prototype_a_engine_is_resident_and_matches_cpu_fixture",
            ),
            (
                "prototype-c-device",
                "7 0",
                "gpu_native::prototype_c_engine::device_tests::",
            ),
        ):
            with self.subTest(label=label):
                self.assertIn(label, self.memcheck_validation)
                self.assertIn(test_filter, self.memcheck_validation)
                self.assertRegex(
                    self.memcheck_validation,
                    rf'run_sanitizer\s+{label}[^\n]*\s{count}\s+[^\n]*\n(?:[^\n]*\n){{0,16}}[^\n]*{test_filter}',
                )

    def test_cuda_test_exports_survive_background_gpu_telemetry(self) -> None:
        search_export = "export NEOETHOS_RUN_CUDA_SEARCH_TESTS=1"
        require_export = "export NEOETHOS_REQUIRE_GPU=1"
        telemetry = "nvidia-smi"
        self.assertIn(search_export, self.memcheck_validation)
        self.assertIn(require_export, self.memcheck_validation)
        self.assertIn(telemetry, self.memcheck_validation)
        self.assertLess(self.memcheck_validation.index(search_export), self.memcheck_validation.index(telemetry))
        self.assertLess(self.memcheck_validation.index(require_export), self.memcheck_validation.index(telemetry))
        self.assertNotRegex(
            self.memcheck_validation,
            r"NEOETHOS_(?:RUN_CUDA_SEARCH_TESTS|REQUIRE_GPU)=1[^\n]*nvidia-smi[^\n]*&",
        )

    def test_report_is_written_by_the_rust_cli_without_python(self) -> None:
        # The payload moved out of an inline Python heredoc into
        # `neoethos-cli bench-preflight-report`, which now owns the schema and
        # the `compute_sanitizer_passed` field. The paid-run path stays
        # Python-free.
        self.assertIn("bench-preflight-report", self.source)
        self.assertIn("--gpu-uuid", self.source)
        for forbidden in ("python3", "python "):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)

    def test_successful_cargo_skip_output_is_rejected(self) -> None:
        for required in (
            "expected_passed",
            "expected_ignored",
            "observed_passed",
            "observed_ignored",
            "skipped",
            "fallback",
            "substitut",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.gate.lower())

    def test_native_builder_environment_aliases_are_cleared(self) -> None:
        for name in (
            "CUDA_ARCHS",
            "CUDA_ARCH",
            "CMAKE_CUDA_ARCHITECTURES",
            "NVCC_ARGS",
            "NVCC_PREPEND_FLAGS",
            "NVCC_APPEND_FLAGS",
            "CUDAFLAGS",
        ):
            with self.subTest(name=name):
                self.assertIn(f"-u {name}", self.validation)

    def test_pinned_counts_match_the_current_filtered_source_modules(self) -> None:
        cubecl = self.search_eval.split("mod gpu_cpu_parity_tests", 1)[1].split(
            "mod cubecl_trailing_parity_tests", 1
        )[0]
        native_b = self.search_eval.split("mod trailing_parity_tests", 1)[1].split(
            "mod gap_threshold_tests", 1
        )[0]
        data = self.gpu_indicators.split("mod tests", 1)[1]

        self.assertEqual(cubecl.count("#[test]"), 7)
        self.assertEqual(native_b.count("#[test]"), 3)
        self.assertEqual(self.prototype_c_tests.count("#[test]"), 7)
        self.assertEqual(data.count("#[test]"), 69)
        self.assertEqual(data.count("#[ignore"), 2)

    def test_rtx_3090_and_a6000_are_supported_without_lowering_the_hardware_floor(self) -> None:
        for supported_name in ("RTX 3090", "RTX A6000"):
            with self.subTest(supported_name=supported_name):
                self.assertIn(supported_name, self.hardware_gate)
        self.assertIn("--query-gpu=compute_cap", self.hardware_gate)
        self.assertIn('MIN_COMPUTE_CAPABILITY="${NEOETHOS_MIN_COMPUTE_CAPABILITY:-86}"', self.hardware_gate)
        self.assertIn('MIN_VRAM_MIB="${NEOETHOS_MIN_VRAM_MIB:-24000}"', self.hardware_gate)
        self.assertIn("NEOETHOS_ALLOW_OTHER_GPU", self.hardware_gate)
        self.assertIn('check_cuda_hardware.sh"', self.source)
        self.assertNotIn('EXPECTED_GPU="${NEOETHOS_EXPECT_GPU_SUBSTRING:-RTX A6000}"', self.source)


if __name__ == "__main__":
    unittest.main()
