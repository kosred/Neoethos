#!/usr/bin/env python3
"""Source contracts for VectorTA's Cargo-coordinated NVCC build lane."""

from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_RS = ROOT / "vendor" / "vector-ta-0.2.9-patched" / "build.rs"
MANIFESTS = [
    ROOT / "vendor" / "vector-ta-0.2.9-patched" / "Cargo.toml",
    ROOT / "vendor" / "vector-ta-0.2.9-patched" / "Cargo.toml.orig",
]


class VectorTaNvccBuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = BUILD_RS.read_text(encoding="utf-8")

    def source_offset(self, needle: str) -> int:
        self.assertTrue(
            needle in self.source,
            f"{BUILD_RS.relative_to(ROOT)} is missing required contract token {needle!r}",
        )
        return self.source.index(needle)

    def test_missing_generated_directory_is_not_a_cargo_watch_input(self) -> None:
        self.assertNotIn('rerun-if-changed=kernels/cubin', self.source)

    def test_build_connects_to_cargos_jobserver_before_starting_work(self) -> None:
        connect = self.source_offset("jobserver::Client::from_env")
        compile_start = self.source_offset("compile_cuda_kernels(")
        self.assertLess(connect, compile_start)
        for manifest in MANIFESTS:
            text = manifest.read_text(encoding="utf-8")
            self.assertIn("[build-dependencies]", text, manifest)
            self.assertRegex(text, r'(?m)^jobserver\s*=')

    def test_kernel_calls_enqueue_then_one_scheduler_drains_the_queue(self) -> None:
        self.source_offset("struct KernelJob")
        self.source_offset("fn run_queued_kernel_jobs")

        enqueue_start = self.source_offset("fn compile_kernel(")
        worker_start = self.source_offset("fn compile_kernel_now(")
        enqueue_body = self.source[enqueue_start:worker_start]
        self.assertNotIn(".output()", enqueue_body)
        self.assertIn("KERNEL_JOBS", enqueue_body)

    def test_parallel_workers_are_permit_bound_and_fail_closed(self) -> None:
        scheduler_start = self.source_offset("fn run_queued_kernel_jobs")
        scheduler = self.source[scheduler_start:]
        self.assertIn("acquire()", scheduler)
        self.assertIn("AtomicBool", scheduler)
        self.assertIn("resume_unwind", scheduler)

    def test_free_form_nvcc_args_fail_closed_before_any_kernel_launch(self) -> None:
        self.source_offset("fn reject_free_form_nvcc_args")
        compile_start = self.source_offset("fn compile_cuda_kernels")
        first_declaration = self.source_offset("compile_alma_kernel(")
        preflight = self.source[compile_start:first_declaration]
        self.assertIn("reject_free_form_nvcc_args()", preflight)

        worker_start = self.source_offset("fn compile_kernel_now(")
        worker = self.source[worker_start:]
        self.assertNotIn('env::var("NVCC_ARGS")', worker)

    def test_every_declared_kernel_has_one_unique_output_pair(self) -> None:
        declarations = re.findall(
            r'compile_kernel\(\s*(?:&)?cuda_path,\s*"([^"]+)",\s*"([^"]+)"',
            self.source,
        )
        self.assertGreater(len(declarations), 300)
        duplicates = sorted({pair for pair in declarations if declarations.count(pair) > 1})
        self.assertEqual(duplicates, [], f"duplicate CUDA source/output declarations: {duplicates}")
        outputs = [output for _, output in declarations]
        duplicate_outputs = sorted({output for output in outputs if outputs.count(output) > 1})
        self.assertEqual(duplicate_outputs, [], f"colliding CUDA output names: {duplicate_outputs}")
        self.source_offset("fn validate_unique_kernel_jobs")


if __name__ == "__main__":
    unittest.main()
