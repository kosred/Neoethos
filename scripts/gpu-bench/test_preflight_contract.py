#!/usr/bin/env python3
"""Static contract tests for the fail-fast rented-GPU preflight."""

from __future__ import annotations

import pathlib
import unittest


PREFLIGHT = pathlib.Path(__file__).with_name("preflight.sh")


class PreflightContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PREFLIGHT.read_text(encoding="utf-8")

    def test_runs_every_required_direct_cuda_probe(self) -> None:
        for test_name in (
            "real_cuda_smoke_is_explicitly_gpu_gated",
            "gpu_event_first_hit_matches_reference_when_adapter_is_available",
            "direct_gpu_trace_matches_cpu_when_an_adapter_is_available",
            "direct_trade_trace_levels_four_through_nine_match_cpu",
        ):
            with self.subTest(test_name=test_name):
                self.assertIn(test_name, self.source)

    def test_compute_sanitizer_is_an_executed_fail_loud_gate(self) -> None:
        for required in (
            "compute-sanitizer",
            "--tool memcheck",
            "--target-processes all",
            "--error-exitcode",
            "compute_sanitizer_passed",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.source)

    def test_successful_cargo_skip_output_is_rejected(self) -> None:
        self.assertIn("run_required_cuda_probe", self.source)
        self.assertRegex(self.source, r"grep .+skipped")


if __name__ == "__main__":
    unittest.main()
