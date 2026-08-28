#!/usr/bin/env python3
"""Static contracts for the staged paid-NVIDIA bootstrap."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "scripts" / "gpu-bench" / "remote_bootstrap.sh"
RENTED = ROOT / "scripts" / "gpu-bench" / "run_rented.sh"


class RemoteBootstrapContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
        cls.rented = RENTED.read_text(encoding="utf-8")

    def test_stages_call_the_strict_named_validation_groups(self) -> None:
        self.assertIn('check_cuda_hardware.sh"', self.bootstrap)
        self.assertRegex(
            self.bootstrap,
            r"for tool in [^\n]*compute-sanitizer",
        )
        self.assertIn('run_cuda_validation.sh" native', self.bootstrap)
        self.assertIn('run_cuda_validation.sh" cubecl', self.bootstrap)
        self.assertIn('run_cuda_validation.sh" data', self.bootstrap)
        self.assertNotIn("--features gpu-b-native prototype_b", self.bootstrap)
        self.assertNotIn("--features gpu-cuda gpu_native::", " ".join(self.bootstrap.split()))

    def test_stage2_memchecks_native_abi_and_three_native_b_parity_tests(self) -> None:
        self.assertIn('run_cuda_memcheck_validation.sh" native', self.bootstrap)
        self.assertIn("stage2-native-memcheck", self.bootstrap)
        self.assertIn("stage2-native-telemetry.csv", self.bootstrap)

    def test_stage3_memchecks_cubecl_population_direct_a_and_full_c_group(self) -> None:
        self.assertIn('run_cuda_memcheck_validation.sh" cubecl', self.bootstrap)
        normal = self.bootstrap.index('run_cuda_validation.sh" cubecl')
        memcheck = self.bootstrap.index('run_cuda_memcheck_validation.sh" cubecl')
        self.assertLess(normal, memcheck)
        self.assertIn("stage3-search-memcheck", self.bootstrap)
        self.assertIn("stage3-search-telemetry.csv", self.bootstrap)

    def test_cli_matrix_is_not_presented_as_prototype_a_proof(self) -> None:
        self.assertNotIn("stage 3: attributed matrix", self.bootstrap)
        self.assertNotIn("--prototypes a", self.bootstrap)

        self.assertIn('NEOETHOS_BENCH_PROTOTYPES:-b,c', self.rented)
        self.assertIn("Prototype A CLI benchmarking is disabled", self.rented)


if __name__ == "__main__":
    unittest.main()
