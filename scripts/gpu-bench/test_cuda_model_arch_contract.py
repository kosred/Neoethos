#!/usr/bin/env python3
"""Contract tests for CUDA architecture propagation into model builds."""

from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
XGBOOST_BUILD = ROOT / "vendor" / "xgboost_lib-sys" / "build.rs"
LIGHTGBM_BUILD = ROOT / "vendor" / "lightgbm3-sys" / "build.rs"
LIGHTGBM_CMAKE = (
    ROOT / "vendor" / "lightgbm3-sys" / "lightgbm" / "CMakeLists.txt"
)
CUDA_ARCH_HELPER = ROOT / "vendor" / "cuda_build_arch.rs"
VECTOR_TA_BUILD = ROOT / "vendor" / "vector-ta-0.2.9-patched" / "build.rs"
VECTOR_TA_CARGO = ROOT / "vendor" / "vector-ta-0.2.9-patched" / "Cargo.toml"
VECTOR_TA_README = ROOT / "vendor" / "vector-ta-0.2.9-patched" / "README.md"
VECTOR_TA_NATIVE_SASS = (
    ROOT / "vendor" / "vector-ta-0.2.9-patched" / "src" / "native_sass.rs"
)
NATIVE_CUDA_BUILD = ROOT / "crates" / "neoethos-gpu-cuda" / "build.rs"
BUILD_HOST_PROBE = ROOT / "scripts" / "build" / "resolve_host.rs"
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
        cls.cuda_arch_helper = CUDA_ARCH_HELPER.read_text(encoding="utf-8")
        cls.vector_ta_build = VECTOR_TA_BUILD.read_text(encoding="utf-8")
        cls.vector_ta_cargo = VECTOR_TA_CARGO.read_text(encoding="utf-8")
        cls.vector_ta_readme = VECTOR_TA_README.read_text(encoding="utf-8")
        cls.vector_ta_native_sass = VECTOR_TA_NATIVE_SASS.read_text(encoding="utf-8")
        cls.native_cuda_build = NATIVE_CUDA_BUILD.read_text(encoding="utf-8")
        cls.build_host_probe = BUILD_HOST_PROBE.read_text(encoding="utf-8")
        cls.cargo_config = CARGO_CONFIG.read_text(encoding="utf-8")
        cls.rust_toolchain = RUST_TOOLCHAIN.read_text(encoding="utf-8")
        cls.build_host_sh = BUILD_HOST_SH.read_text(encoding="utf-8")
        cls.build_host_ps1 = BUILD_HOST_PS1.read_text(encoding="utf-8")

    def test_both_build_scripts_track_the_project_architecture_setting(self) -> None:
        for name in ["NEOETHOS_CUDA_BUILD_MODE", "NEOETHOS_CUDA_ARCHS"]:
            self.assertIn(f'"{name}"', self.cuda_arch_helper)
            self.assertIn(f'"{name}"', self.native_cuda_build)
        self.assertIn("cargo:rerun-if-env-changed={name}", self.cuda_arch_helper)
        self.assertIn("cargo:rerun-if-env-changed={name}", self.native_cuda_build)
        for source in [self.xgboost_build, self.lightgbm_build, self.vector_ta_build]:
            self.assertIn("cuda_build_arch.rs", source)
            self.assertIn("resolve_exact_cuda_architectures", source)

    def test_xgboost_receives_the_official_cmake_architecture_setting(self) -> None:
        self.assertIn(
            '.define("CMAKE_CUDA_ARCHITECTURES", &architectures.native_only)',
            self.xgboost_build,
        )

    def test_lightgbm_receives_and_preserves_the_requested_architectures(self) -> None:
        self.assertIn(
            '.define(\n                "NEOETHOS_EXACT_CUDA_ARCHITECTURES",\n                &architectures.native_only,',
            self.lightgbm_build,
        )
        self.assertIn(
            "if(NOT DEFINED NEOETHOS_EXACT_CUDA_ARCHITECTURES OR ",
            self.lightgbm_cmake,
        )
        self.assertIn("message(FATAL_ERROR", self.lightgbm_cmake)
        self.assertNotIn(
            'set(CUDA_ARCHS "60" "61" "62" "70" "75")',
            self.lightgbm_cmake,
        )
        self.assertNotIn("-virtual", self.lightgbm_cmake)

    def test_cuda_model_builds_do_not_keep_the_old_implicit_architecture_path(self) -> None:
        required_call = "cuda_build_arch::resolve_exact_cuda_architectures()"
        self.assertIn(required_call, self.xgboost_build)
        self.assertIn(required_call, self.lightgbm_build)
        self.assertNotIn("requested_cuda_architectures()", self.xgboost_build)
        self.assertNotIn("requested_cuda_architectures()", self.lightgbm_build)

    def test_vector_ta_and_native_cuda_use_the_same_validated_numeric_set(self) -> None:
        self.assertIn(
            "cuda_build_arch::resolve_exact_cuda_architectures()",
            self.vector_ta_build,
        )
        self.assertIn("let resolved =", self.vector_ta_build)
        self.assertIn(".numeric", self.vector_ta_build)
        self.assertIn("architecture_request_from_env", self.native_cuda_build)
        for source in [self.vector_ta_build, self.native_cuda_build]:
            self.assertIn("NEOETHOS_CUDA_BUILD_MODE", source)
            self.assertIn("NEOETHOS_CUDA_ARCHS", source)

    def test_vector_ta_cuda_compilation_is_unique_and_cargo_jobserver_bounded(self) -> None:
        ptx_outputs = re.findall(
            r'"([A-Za-z0-9_./-]+[.]ptx)"', self.vector_ta_build
        )
        duplicates = sorted(
            name for name in set(ptx_outputs) if ptx_outputs.count(name) > 1
        )
        self.assertEqual([], duplicates, f"duplicate PTX outputs: {duplicates}")
        self.assertIn("jobserver::Client::from_env()", self.vector_ta_build)
        self.assertIn("run_queued_kernel_jobs", self.vector_ta_build)
        self.assertIn("run_native_artifact_jobs", self.vector_ta_build)
        self.assertRegex(self.vector_ta_cargo, r"(?m)^jobserver\s*=")
        self.assertIn(
            "cargo_width.min(host_width).min(job_count).max(1)",
            self.vector_ta_build,
        )

    def test_vector_ta_rejects_external_codegen_overrides(self) -> None:
        self.assertIn("reject_free_form_nvcc_args", self.vector_ta_build)
        self.assertIn("Remove NVCC_ARGS", self.vector_ta_build)
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
        self.assertNotIn('env::var("CUDA_ARCHS").ok()', self.vector_ta_build)
        self.assertNotIn('env::var("NVCC").unwrap_or_else', self.vector_ta_build)
        self.assertIn("NVCC is a rejected compiler-path authority", self.vector_ta_build)
        self.assertIn("CUDA_ARCHS is a rejected ambient", self.cuda_arch_helper)
        self.assertIn("does not exactly match typed NEOETHOS_CUDA_ARCHS", self.cuda_arch_helper)
        self.assertNotIn("`CUDA_ARCHS=", self.vector_ta_readme)
        self.assertNotIn("set CUDA_ARCHS or build", self.vector_ta_native_sass)

    def test_build_resolves_cpu_and_gpu_inputs_from_the_current_host(self) -> None:
        self.assertIn("std::thread::available_parallelism()", self.build_host_probe)
        self.assertIn(
            '"--query-gpu=uuid,pci.bus_id,name,compute_cap"', self.build_host_probe
        )
        self.assertIn("pci_bus_id", self.build_host_probe)
        self.assertIn("cuda_architectures=", self.build_host_probe)

    def test_host_build_has_an_explicit_cpu_only_plan_without_stale_cuda_state(self) -> None:
        self.assertIn('"cpu_only"', self.build_host_probe)
        self.assertIn('println!("accelerator_mode={accelerator_mode}")', self.build_host_probe)
        self.assertIn("ErrorKind::NotFound", self.build_host_probe)
        self.assertIn("accelerator_mode", self.build_host_sh)
        self.assertIn("unset NEOETHOS_CUDA_BUILD_MODE", self.build_host_sh)
        self.assertIn("unset NEOETHOS_CUDA_ARCHS", self.build_host_sh)
        self.assertIn("accelerator_mode", self.build_host_ps1)
        self.assertIn("Remove-Item Env:NEOETHOS_CUDA_BUILD_MODE", self.build_host_ps1)
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
            self.assertIn("NEOETHOS_CUDA_BUILD_MODE", wrapper)
            self.assertIn("NEOETHOS_CUDA_ARCHS", wrapper)
            self.assertIn("cargo", wrapper)
            self.assertNotRegex(
                wrapper,
                r"(?:NEOETHOS_CUDA_ARCHS|CARGO_BUILD_JOBS)\s*=\s*[0-9]",
            )

if __name__ == "__main__":
    unittest.main()
