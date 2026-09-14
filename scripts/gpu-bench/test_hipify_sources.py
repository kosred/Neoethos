#!/usr/bin/env python3
"""Fake-tool orchestration tests. These do NOT test HIP/CUDA or GPU numerics."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("hipify_sources.py")
SPEC = importlib.util.spec_from_file_location("hipify_sources", SCRIPT)
HIPIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HIPIFY)


class HipifyStagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="neoethos_fake_hipify_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "repo with spaces"
        self.native = self.repo / HIPIFY.GPU_BUILD.parent / "native"
        self.native.mkdir(parents=True)
        self.build = self.repo / HIPIFY.GPU_BUILD
        self.build.write_text('const DEVICE_SOURCES: [&str; 2] = [\n "native/a.cu", "native/b.cu",\n];\n')
        (self.native / "a.cu").write_text('#include "types.cuh"\n// cuda test a\n')
        (self.native / "b.cu").write_text('// cuda test b\n')
        (self.native / "types.cuh").write_text('#pragma once\n// cuda header\n')
        self.vector_build = self.repo / HIPIFY.VECTOR_BUILD
        self.vector_build.parent.mkdir(parents=True)
        self.vector_build.write_text('compile_kernel(&cuda_path, "kernels/cuda/f64.cu", "neoethos_f64_kernels");\n')
        self.vector = self.vector_build.parent / "kernels/cuda/f64.cu"
        self.vector.parent.mkdir(parents=True)
        self.vector.write_text('// cuda f64\n')
        self.cuda = self.root / "cuda toolkit"
        (self.cuda / "include").mkdir(parents=True)
        (self.cuda / "include/cuda.h").write_text('// fake header\n')
        (self.cuda / "include/cuda_runtime.h").write_text('// fake runtime header\n')
        self.fake = self.root / "fake hipify.py"
        self.fake.write_text(
            'import pathlib,sys,time\n'
            'a=sys.argv[1:]\n'
            'if a == ["--version"]:\n'
            ' print("FAKE HIPIFY TOOL FOR ORCHESTRATION TEST ONLY");sys.exit(0)\n'
            'print("INFO beginning fake translation")\n'
            'print("WARNING preserved in full",file=sys.stderr)\n'
            'print("X"*90000)\n'
            'text=pathlib.Path(a[0]).read_text()\n'
            'if "SLOW" in text: time.sleep(2)\n'
            'if "FAIL" in text:\n'
            ' print("ERROR explicit fake tool failure",file=sys.stderr);sys.exit(7)\n'
            'if "-o" in a and "EMPTY" not in text:\n'
            ' pathlib.Path(a[a.index("-o")+1]).write_text(text.replace("cuda", "hip"))\n',
            encoding="utf-8",
        )
        self.real_run = HIPIFY.run_command
        self.command_patch = patch.object(HIPIFY, "run_command", self.fake_command)
        self.command_patch.start()
        self.addCleanup(self.command_patch.stop)
        self.environment = patch.dict(os.environ, {key: "" for key in HIPIFY.HIDDEN_INCLUDE_ENV})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def fake_command(self, argv, cwd, log_base, timeout):
        # Execute a genuine child through Python on both Windows and Linux;
        # this injection is test-only and is NOT a production CLI option.
        return self.real_run([sys.executable, str(self.fake), *argv[1:]], cwd, log_base, timeout)

    def args(self, mode="translate", name="stage"):
        return HIPIFY.parser().parse_args([
            mode, "--repo", str(self.repo), "--output", str(self.root / name),
            "--hipify-clang", sys.executable, "--cuda-path", str(self.cuda),
        ])

    def test_real_builder_list_is_discovered_without_a_second_tu_list(self):
        repo = SCRIPT.resolve().parents[2]
        tus, headers, builders = HIPIFY.discover(repo)
        declared = HIPIFY.rust_string_array((repo / HIPIFY.GPU_BUILD).read_text(), "DEVICE_SOURCES")
        self.assertEqual([str(p.relative_to(repo / HIPIFY.GPU_BUILD.parent)).replace("\\", "/") for p in tus[:-1]], declared)
        self.assertEqual(tus[-1].name, "neoethos_f64_kernels.cu")
        self.assertTrue(headers)
        self.assertEqual(len(builders), 2)

    def test_translate_preserves_sources_full_logs_and_header_geometry(self):
        before = {p: HIPIFY.digest(p) for p in self.repo.rglob("*") if p.is_file()}
        manifest_path, manifest = HIPIFY.stage(self.args())
        self.assertTrue(manifest["success"])
        self.assertEqual(len(manifest["results"]), 4)
        self.assertEqual(manifest["native_translation_units"], 2)
        self.assertEqual(manifest["target"], "gfx942")
        self.assertEqual(manifest["cuda_parse_arch"], "sm_86")
        for result in manifest["results"]:
            self.assertGreater(Path(result["stdout"]["path"]).stat().st_size, 90000)
            self.assertIn("WARNING preserved in full", Path(result["stderr"]["path"]).read_text())
            self.assertIn("--cuda-gpu-arch=sm_86", result["argv"])
            self.assertLess(result["argv"].index("--cuda-gpu-arch=sm_86"), result["argv"].index("--"))
            self.assertLess(result["argv"].index("--default-preprocessor"), result["argv"].index("--"))
            self.assertNotIn("--amap", result["argv"])
            self.assertNotIn("--cuda-host-only", result["argv"])
            self.assertNotIn("-fsyntax-only", result["argv"])
            self.assertNotIn("--inplace", result["argv"])
            self.assertNotIn("--offload-arch=gfx942", result["argv"])
            self.assertIn("hip", Path(result["output"]["path"]).read_text())
        self.assertEqual(before, {p: HIPIFY.digest(p) for p in before})
        self.assertTrue(HIPIFY.validate(manifest_path, HIPIFY.digest(manifest_path))["success"])

    def test_preprocessor_selection_and_explicit_defines_are_recorded(self):
        args = self.args()
        args.define = ["NEOETHOS_CUDA_DEVICE_FIXTURES_V2=1"]
        _, manifest = HIPIFY.stage(args)
        self.assertEqual(manifest["defines"], args.define)
        self.assertEqual(manifest["preprocessor_mode"], "compiler-defined-active-branches")
        self.assertIn("regenerate", manifest["inactive_branch_policy"])
        for result in manifest["results"]:
            self.assertIn("-DNEOETHOS_CUDA_DEVICE_FIXTURES_V2=1", result["argv"])
            self.assertIn("--default-preprocessor", result["argv"])

    def test_diagnostics_do_not_confuse_error_api_statistics_with_errors(self):
        log = self.root / "diagnostics.log"
        log.write_text("error: missing architecture\n"
                       "warning: CUDA version partially supported\n"
                       "  error: 2\n"
                       "/tmp/example.cu:42:8: fatal error: absent header\n"
                       "[HIPIFY] error: Hipifying failed\n")
        self.assertEqual(HIPIFY.diagnostic_counts([log]), {"error_lines": 3, "warning_lines": 1})

    def test_zero_exit_does_not_claim_error_diagnostics_absent(self):
        self.fake.write_text(self.fake.read_text() + '\nprint("error: diagnostic despite zero exit",file=sys.stderr)\n')
        path, manifest = HIPIFY.stage(self.args())
        self.assertTrue(manifest["success"])
        self.assertFalse(manifest["error_diagnostics_absent"])
        self.assertEqual(manifest["diagnostics"]["error_lines"], 4)
        self.assertIn("host-side AST", manifest["proof_scope"])
        validation = HIPIFY.validate(path)
        self.assertTrue(validation["success"])
        self.assertFalse(validation["error_diagnostics_absent"])

    def test_analyze_produces_no_translated_sources(self):
        path, manifest = HIPIFY.stage(self.args("analyze"))
        self.assertTrue(manifest["success"])
        self.assertEqual(manifest["outputs"], [])
        self.assertTrue(all("--no-output" in r["argv"] for r in manifest["results"]))
        self.assertTrue(HIPIFY.validate(path)["success"])

    def test_failed_translation_keeps_all_files_logs_and_failed_manifest(self):
        (self.native / "a.cu").write_text("FAIL\n")
        path, manifest = HIPIFY.stage(self.args())
        self.assertFalse(manifest["success"])
        self.assertEqual(len(manifest["results"]), 4)
        failed = [r for r in manifest["results"] if r["exit_code"] == 7]
        self.assertEqual(len(failed), 1)
        self.assertIn("ERROR explicit fake tool failure", Path(failed[0]["stderr"]["path"]).read_text())
        self.assertFalse(HIPIFY.validate(path)["success"])

    def test_missing_output_despite_zero_exit_is_failure(self):
        (self.native / "a.cu").write_text("EMPTY\n")
        _, manifest = HIPIFY.stage(self.args())
        self.assertFalse(manifest["success"])
        self.assertTrue(any("nonempty" in (r["error"] or "") for r in manifest["results"]))

    def test_child_filesystem_failure_does_not_discard_other_results(self):
        def failing_command(argv, cwd, log_base, timeout):
            if argv[1].endswith("a.cu"):
                raise PermissionError("explicit test-only filesystem failure")
            return self.fake_command(argv, cwd, log_base, timeout)

        with patch.object(HIPIFY, "run_command", failing_command):
            path, manifest = HIPIFY.stage(self.args())
        self.assertFalse(manifest["success"])
        self.assertEqual(len(manifest["results"]), 4)
        self.assertEqual(sum(HIPIFY.succeeded(r) for r in manifest["results"]), 3)
        self.assertFalse(HIPIFY.validate(path)["success"])

    def test_timeout_retains_partial_logs_and_is_not_success(self):
        (self.native / "a.cu").write_text("SLOW\n")
        args = self.args()
        args.timeout_seconds = 0.5
        _, manifest = HIPIFY.stage(args)
        self.assertFalse(manifest["success"])
        self.assertTrue(any(r["timed_out"] for r in manifest["results"]))

    def test_output_inside_repo_existing_or_broad_root_is_refused(self):
        for output in (self.repo / "output", self.root, self.root.parent, self.cuda / "output"):
            args = self.args()
            args.output = output
            with self.assertRaises(HIPIFY.StageError):
                HIPIFY.stage(args)

    def test_list_drift_and_traversal_are_refused(self):
        for text in (
            'const DEVICE_SOURCES: [&str; 2] = ["native/a.cu"];',
            'const DEVICE_SOURCES: [&str; 2] = ["native/a.cu", "native/a.cu"];',
            'const DEVICE_SOURCES: [&str; 1] = ["native/../../escape.cu"];',
            'const DEVICE_SOURCES: [&str; 1] = [concat!("native/", "a.cu")];',
        ):
            self.build.write_text(text)
            with self.assertRaises(HIPIFY.StageError):
                HIPIFY.discover(self.repo)

    def test_validator_detects_log_source_output_and_external_header_drift(self):
        path, manifest = HIPIFY.stage(self.args())
        targets = [Path(manifest["results"][0]["stdout"]["path"]), self.native / "a.cu",
                   Path(manifest["outputs"][0]["path"]), self.cuda / "include/cuda.h"]
        for target in targets:
            original = target.read_bytes()
            target.write_bytes(original + b"changed")
            self.assertFalse(HIPIFY.validate(path)["success"])
            target.write_bytes(original)
        (self.cuda / "include/added.hpp").write_text("new header")
        self.assertFalse(HIPIFY.validate(path)["success"])
        with self.assertRaises(HIPIFY.StageError):
            HIPIFY.validate(path, "0" * 64)

    def test_validator_detects_new_project_header_membership(self):
        path, _ = HIPIFY.stage(self.args())
        (self.native / "new_header.cuh").write_text("// newly added input\n")
        result = HIPIFY.validate(path)
        self.assertFalse(result["success"])
        self.assertIn("project source membership changed", result["drift"])

    def test_implicit_include_environment_is_refused(self):
        with patch.dict(os.environ, {"CPATH": str(self.root)}):
            with self.assertRaises(HIPIFY.StageError):
                HIPIFY.stage(self.args())


if __name__ == "__main__":
    unittest.main()
