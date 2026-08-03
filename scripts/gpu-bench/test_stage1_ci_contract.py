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


class Stage1WorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = WORKFLOW.read_text(encoding="utf-8")
        cls.compact = " ".join(cls.source.replace("\\\n", " ").split())

    def test_direct_probe_step_runs_the_complete_gpu_native_suite(self) -> None:
        self.assertIn(
            "cargo test -p neoethos-search --features gpu-vulkan "
            "gpu_native:: -- --nocapture",
            self.compact,
        )

    def test_workflow_does_not_hide_backend_failures_by_log_substring(self) -> None:
        self.assertNotIn("requested_backends: Backends", self.source)
        self.assertNotIn("run_probe()", self.source)


if __name__ == "__main__":
    unittest.main()
