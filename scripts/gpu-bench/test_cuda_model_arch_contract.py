#!/usr/bin/env python3
"""Contract tests for CUDA architecture propagation into model builds."""

from __future__ import annotations

import os
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
XGBOOST_BUILD = ROOT / "vendor" / "xgboost_lib-sys" / "build.rs"
LIGHTGBM_BUILD = ROOT / "vendor" / "lightgbm3-sys" / "build.rs"
LIGHTGBM_CMAKE = (
    ROOT / "vendor" / "lightgbm3-sys" / "lightgbm" / "CMakeLists.txt"
)
VECTOR_TA_BUILD = ROOT / "vendor" / "vector-ta-0.2.9-patched" / "build.rs"
VECTOR_TA_CARGO = ROOT / "vendor" / "vector-ta-0.2.9-patched" / "Cargo.toml"
NATIVE_CUDA_BUILD = ROOT / "crates" / "neoethos-gpu-cuda" / "build.rs"
BUILD_HOST_PROBE = ROOT / "scripts" / "build" / "resolve_host.rs"
GPU_RUNNER = ROOT / "audit-logs" / "vast-47806434" / "run-gpu-auto-host-and-tsi.sh"
PARALLEL_GPU_RUNNER = (
    ROOT
    / "audit-logs"
    / "vast-47806434"
    / "run-vector-ta-parallel-builder.sh"
)
CARGO_CONFIG = ROOT / ".cargo" / "config.toml"
RUST_TOOLCHAIN = ROOT / "rust-toolchain.toml"
BUILD_HOST_SH = ROOT / "scripts" / "build-host.sh"
BUILD_HOST_PS1 = ROOT / "scripts" / "build-host.ps1"


class CudaModelArchitectureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.xgboost_build = XGBOOST_BUILD.read_text(encoding="utf-8")
        cls.lightgbm_build = LIGHTGBM_BUILD.read_text(encoding="utf-8")
        cls.lightgbm_cmake = LIGHTGBM_CMAKE.read_text(encoding="utf-8")
        cls.vector_ta_build = VECTOR_TA_BUILD.read_text(encoding="utf-8")
        cls.vector_ta_cargo = VECTOR_TA_CARGO.read_text(encoding="utf-8")
        cls.native_cuda_build = NATIVE_CUDA_BUILD.read_text(encoding="utf-8")
        cls.build_host_probe = BUILD_HOST_PROBE.read_text(encoding="utf-8")
        cls.gpu_runner = GPU_RUNNER.read_text(encoding="utf-8")
        cls.parallel_gpu_runner = PARALLEL_GPU_RUNNER.read_text(encoding="utf-8")
        cls.cargo_config = CARGO_CONFIG.read_text(encoding="utf-8")
        cls.rust_toolchain = RUST_TOOLCHAIN.read_text(encoding="utf-8")
        cls.build_host_sh = BUILD_HOST_SH.read_text(encoding="utf-8")
        cls.build_host_ps1 = BUILD_HOST_PS1.read_text(encoding="utf-8")

    def test_both_build_scripts_track_the_project_architecture_setting(self) -> None:
        marker = "cargo:rerun-if-env-changed=NEOETHOS_CUDA_ARCHS"
        for source in [
            self.xgboost_build,
            self.lightgbm_build,
            self.vector_ta_build,
            self.native_cuda_build,
        ]:
            self.assertIn(marker, source)

    def test_xgboost_receives_the_official_cmake_architecture_setting(self) -> None:
        self.assertIn(
            '.define("CMAKE_CUDA_ARCHITECTURES", cuda_architectures)',
            self.xgboost_build,
        )

    def test_lightgbm_receives_and_preserves_the_requested_architectures(self) -> None:
        self.assertIn(
            '.define("NEOETHOS_CUDA_ARCHITECTURES", cuda_architectures)',
            self.lightgbm_build,
        )
        self.assertIn(
            "if(NOT DEFINED NEOETHOS_CUDA_ARCHITECTURES)", self.lightgbm_cmake
        )
        self.assertIn("message(FATAL_ERROR", self.lightgbm_cmake)
        self.assertNotIn(
            'set(CUDA_ARCHS "60" "61" "62" "70" "75")',
            self.lightgbm_cmake,
        )

    def test_cuda_model_builds_do_not_keep_the_old_implicit_architecture_path(self) -> None:
        required_call = "cuda_build_arch::required_cuda_arch_numbers()"
        self.assertIn(required_call, self.xgboost_build)
        self.assertIn(required_call, self.lightgbm_build)
        self.assertNotIn("requested_cuda_architectures()", self.xgboost_build)
        self.assertNotIn("requested_cuda_architectures()", self.lightgbm_build)

    def test_vector_ta_and_native_cuda_use_the_same_validated_numeric_set(self) -> None:
        required_call = "cuda_build_arch::required_cuda_arch_numbers()"
        self.assertIn(required_call, self.vector_ta_build)
        self.assertIn(required_call, self.native_cuda_build)

    def test_vector_ta_cuda_compilation_is_unique_and_cargo_jobserver_bounded(self) -> None:
        ptx_outputs = re.findall(
            r'"([A-Za-z0-9_./-]+[.]ptx)"', self.vector_ta_build
        )
        duplicates = sorted(
            name for name in set(ptx_outputs) if ptx_outputs.count(name) > 1
        )
        self.assertEqual([], duplicates, f"duplicate PTX outputs: {duplicates}")
        self.assertIn("jobserver::Client::from_env()", self.vector_ta_build)
        self.assertIn("finish_kernel_compilations", self.vector_ta_build)
        self.assertIn("[build-dependencies.jobserver]", self.vector_ta_cargo)
        self.assertNotIn("std::thread::available_parallelism", self.vector_ta_build)

    def test_vector_ta_rejects_external_codegen_overrides(self) -> None:
        self.assertIn("external NVCC_ARGS is unsupported", self.vector_ta_build)
        self.assertIn("precision-changing NVCC_ARGS", self.vector_ta_build)
        self.assertEqual(2, self.vector_ta_build.count('"--threads=1"'))
        self.assertNotIn(".args(&extra_args);", self.vector_ta_build)
        self.assertNotIn(
            'if let Ok(extra) = env::var("NVCC_ARGS")', self.vector_ta_build
        )
        for option in [
            "--use_fast_math",
            "fmad",
            "ftz",
            "prec-div",
            "prec-sqrt",
        ]:
            self.assertIn(option, self.vector_ta_build)

    def test_old_cuda_architecture_environment_paths_are_deleted(self) -> None:
        for source in [self.vector_ta_build, self.native_cuda_build]:
            self.assertNotIn("cargo:rerun-if-env-changed=CUDA_ARCH", source)
            self.assertNotIn("cargo:rerun-if-env-changed=CUDA_ARCHS", source)
            self.assertNotIn("cargo:rerun-if-env-changed=NEOETHOS_CUDA_ARCH\"", source)

    def test_old_cuda_architecture_names_are_absent_from_active_repository_surfaces(self) -> None:
        old_names = ["CUDA_" + "ARCHS", "CUDA_" + "ARCH", "NEOETHOS_CUDA_" + "ARCH"]
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, old_names)) + r")\b")
        ignored_directory_names = {".git", "audit-logs", "target"}
        ignored_files = {
            pathlib.Path(__file__).resolve(),
        }
        checked_suffixes = {
            ".cmake",
            ".md",
            ".ps1",
            ".py",
            ".rs",
            ".sh",
            ".toml",
            ".yaml",
            ".yml",
        }
        violations: list[str] = []
        for directory, child_directories, file_names in os.walk(ROOT):
            current = pathlib.Path(directory)
            child_directories[:] = [
                name for name in child_directories if name not in ignored_directory_names
            ]
            if current == ROOT / "docs":
                child_directories[:] = [name for name in child_directories if name != "superpowers"]
            for file_name in file_names:
                path = current / file_name
                if path.resolve() in ignored_files:
                    continue
                if path.name != "CMakeLists.txt" and path.suffix.lower() not in checked_suffixes:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                for line_number, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        violations.append(
                            f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}"
                        )

        self.assertEqual([], violations, "old CUDA architecture paths returned:\n" + "\n".join(violations))

    def test_vps_build_resolves_cpu_and_gpu_inputs_from_the_current_host(self) -> None:
        self.assertIn("std::thread::available_parallelism()", self.build_host_probe)
        self.assertIn(
            '"--query-gpu=uuid,pci.bus_id,name,compute_cap"', self.build_host_probe
        )
        self.assertIn("pci_bus_id", self.build_host_probe)
        self.assertIn("cuda_architectures=", self.build_host_probe)
        self.assertIn("scripts/build/resolve_host.rs", self.gpu_runner)
        self.assertIn("-D warnings", self.gpu_runner)
        self.assertIn("export NEOETHOS_CUDA_ARCHS=$cuda_architectures", self.gpu_runner)
        self.assertNotIn("sm89-compile", self.gpu_runner)
        self.assertNotIn("FIXED_STAGE_DONE", self.gpu_runner)
        self.assertNotRegex(
            self.gpu_runner,
            r"(?:export\s+)?NEOETHOS_CUDA_ARCHS\s*=\s*[0-9]",
        )

    def test_host_build_has_an_explicit_cpu_only_plan_without_stale_cuda_state(self) -> None:
        self.assertIn('"cpu_only"', self.build_host_probe)
        self.assertIn('println!("accelerator_mode={accelerator_mode}")', self.build_host_probe)
        self.assertIn("ErrorKind::NotFound", self.build_host_probe)
        self.assertIn("accelerator_mode", self.build_host_sh)
        self.assertIn("unset NEOETHOS_CUDA_ARCHS", self.build_host_sh)
        self.assertIn("accelerator_mode", self.build_host_ps1)
        self.assertIn("Remove-Item Env:NEOETHOS_CUDA_ARCHS", self.build_host_ps1)

    def test_repository_build_budget_is_adaptive_and_not_multiplied(self) -> None:
        self.assertRegex(self.cargo_config, r"(?m)^jobs\s*=\s*-2\s*$")
        combined = self.cargo_config + "\n" + self.rust_toolchain
        for stale in ["-Zthreads", '"-Z"', "threads=8", "x86-64-v3", "target-cpu=native"]:
            self.assertNotIn(stale, combined)

    def test_windows_and_linux_build_entrypoints_forward_one_host_plan(self) -> None:
        for wrapper in [self.build_host_sh, self.build_host_ps1]:
            self.assertIn("scripts/build/resolve_host.rs", wrapper.replace("\\", "/"))
            self.assertIn("CARGO_BUILD_JOBS", wrapper)
            self.assertIn("NEOETHOS_CUDA_ARCHS", wrapper)
            self.assertIn("cargo", wrapper)
            self.assertNotRegex(
                wrapper,
                r"(?:NEOETHOS_CUDA_ARCHS|CARGO_BUILD_JOBS)\s*=\s*[0-9]",
            )

    def test_parallel_runner_accounts_for_cgroup_v1_and_v2(self) -> None:
        self.assertIn("detect_cpu_accounting", self.parallel_gpu_runner)
        self.assertIn("/sys/fs/cgroup/cpu.stat", self.parallel_gpu_runner)
        self.assertIn(
            "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage",
            self.parallel_gpu_runner,
        )
        self.assertIn("usage_usec", self.parallel_gpu_runner)
        self.assertIn("throttled_usec", self.parallel_gpu_runner)
        self.assertIn("throttled_time", self.parallel_gpu_runner)
        self.assertIn("read_cpu_usage_usec", self.parallel_gpu_runner)
        self.assertIn("read_throttled_usec", self.parallel_gpu_runner)
        for compiler_worker in [
            "rust-lld",
            "ld[.]lld",
            "cc1",
            "cc1plus",
            "clang",
            "clang[+][+]",
        ]:
            self.assertIn(compiler_worker, self.parallel_gpu_runner)


if __name__ == "__main__":
    unittest.main()
