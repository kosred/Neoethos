#!/usr/bin/env python3
"""Contract tests for the read-only Stage 1 GitHub Actions workflow."""

from __future__ import annotations

import pathlib
import unittest


WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "agent-stage1.yml"
)
CUDA_WORKFLOW = WORKFLOW.with_name("ci.yml")


class Stage1WorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = WORKFLOW.read_text(encoding="utf-8")
        cls.compact = " ".join(cls.source.replace("\\\n", " ").split())
        cls.cuda_source = CUDA_WORKFLOW.read_text(encoding="utf-8")

    def test_direct_probe_step_runs_the_complete_gpu_native_suite(self) -> None:
        self.assertIn(
            "cargo test -p neoethos-search --features gpu-vulkan "
            "gpu_native:: -- --nocapture",
            self.compact,
        )

    def test_workflow_does_not_hide_backend_failures_by_log_substring(self) -> None:
        self.assertNotIn("requested_backends: Backends", self.source)
        self.assertNotIn("run_probe()", self.source)

    def test_cuda_ci_uses_the_strict_complete_validation_runner(self) -> None:
        self.assertIn("scripts/gpu-bench/check_cuda_hardware.sh", self.cuda_source)
        self.assertIn(
            "scripts/gpu-bench/run_cuda_memcheck_validation.sh all", self.cuda_source
        )
        hardware = self.cuda_source.index("scripts/gpu-bench/check_cuda_hardware.sh")
        validation = self.cuda_source.index("scripts/gpu-bench/run_cuda_validation.sh all")
        memcheck = self.cuda_source.index(
            "scripts/gpu-bench/run_cuda_memcheck_validation.sh all"
        )
        self.assertLess(hardware, validation)
        self.assertLess(validation, memcheck)
        self.assertIn("scripts/gpu-bench/run_cuda_validation.sh all", self.cuda_source)
        self.assertNotIn(
            "cargo test -p neoethos-search --release --features gpu-cuda gpu_",
            self.cuda_source,
        )

    def test_cuda_ci_sets_the_explicit_search_device_gate(self) -> None:
        self.assertIn("NEOETHOS_RUN_CUDA_SEARCH_TESTS", self.cuda_source)


if __name__ == "__main__":
    unittest.main()
