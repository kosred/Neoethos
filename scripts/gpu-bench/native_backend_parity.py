#!/usr/bin/env python3
"""Build once, run separately, compare CPU/CUDA/HIP native parity evidence.

This is a small correctness fixture tool, not the Rust paid benchmark runner.
Neither successful compilation nor a CPU run is device parity. Comparison
requires successful real-device receipts from BOTH CUDA and HIP.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import hipify_sources as audit

NATIVE = Path("crates/neoethos-gpu-cuda/native")
TESTS = Path("crates/neoethos-gpu-cuda/tests")
FIXTURE = TESTS / "fixtures/exact_log_backend_vectors_v1.csv"
SCHEMA = "neoethos.native-backend-parity.v1"
# Intentional fixture coverage, not limits on application search population.
EXPECTED_CASES = {"first-hit": 212, "exact-log": 48}
SUITES = {
    "first-hit": ("neoethos.first-hit-backend-parity.v1", [
        TESTS / "first_hit_backend_parity_v1.cpp", NATIVE / "prototype_b.cu",
        NATIVE / "neoethos_gpu_cuda.h"]),
    "exact-log": ("neoethos.exact-log-backend-parity.v1", [
        TESTS / "exact_log_backend_parity_v1.cu", TESTS / "exact_log_cpu_parity_v1.rs",
        FIXTURE, NATIVE / "resident_exact_log_v3.cuh",
        Path("crates/neoethos-data/src/core/quant_exact_math_v3.rs")]),
}
HIP_FLAGS = ["-x", "hip", "--offload-arch=gfx942", "-std=c++17", "-O3",
             "-fno-fast-math", "-ffp-contract=off", "-fdenormal-fp-math=ieee",
             "-Xclang", "-fdenormal-fp-math-f32=ieee",
             "-fno-gpu-flush-denormals-to-zero",
             "-fhip-fp32-correctly-rounded-divide-sqrt", "-mno-unsafe-fp-atomics"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise audit.StageError(message)


def word(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value) is not None


def read_receipt(path: Path, expected: str, phase: str) -> dict:
    require(audit.digest(path) == expected.lower(), "receipt SHA256 mismatch")
    result = json.loads(path.read_text(encoding="utf-8"))
    require(result.get("schema") == SCHEMA and result.get("phase") == phase,
            "wrong receipt kind")
    return result


def immutable_output(path: Path, repo: Path, protected: list[Path]) -> Path:
    output = audit.checked_output(path, repo, protected)
    output.mkdir()
    return output


def source_pins(repo: Path, suite: str) -> list[dict]:
    paths = [*SUITES[suite][1], Path("scripts/gpu-bench/native_backend_parity.py"),
             Path("scripts/gpu-bench/hipify_sources.py"), Path("rust-toolchain.toml")]
    return [dict(audit.file_record(audit.source_path(repo, p)), relative=p.as_posix())
            for p in paths]


def changed(pins: list[dict]) -> list[str]:
    return audit.check_records([{k: r[k] for k in ("path", "bytes", "sha256")}
                                for r in pins])


def identity(pins: list[dict]) -> dict:
    return {r["relative"]: r["sha256"] for r in pins}


def build(args: argparse.Namespace) -> tuple[Path, dict]:
    repo = args.repo.resolve(strict=True)
    compiler = args.compiler.absolute()
    audit.regular_file(compiler)
    for name in (*audit.HIDDEN_INCLUDE_ENV, "HIPCC_COMPILE_FLAGS_APPEND",
                 "HIPCC_LINK_FLAGS_APPEND", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        require(not os.environ.get(name), f"unset implicit compiler flags: {name}")
    pins = source_pins(repo, args.suite)
    tool = audit.file_record(compiler)
    hip_stage = None
    include = repo / NATIVE
    if args.backend == "hip":
        require(compiler.name == "hipcc", "HIP fixture requires the AMD hipcc driver")
        rocm = compiler.parent.parent
        require((rocm / "include/hip/hip_runtime.h").is_file(), "incomplete explicit ROCm root")
        for key, value in (("HIP_PLATFORM", "amd"), ("ROCM_PATH", str(rocm)), ("HIP_PATH", str(rocm))):
            require(not os.environ.get(key) or os.environ[key] == value, f"conflicting HIP environment: {key}")
            os.environ[key] = value
        require(args.hip_manifest is not None and args.hip_manifest_sha256 is not None,
                "HIP requires a pinned official HIPIFY manifest")
        hip_stage = audit.validate(args.hip_manifest, args.hip_manifest_sha256)
        require(hip_stage["success"], "official HIPIFY stage integrity failed")
        manifest = json.loads(args.hip_manifest.read_text())
        require(manifest["mode"] == "translate" and not manifest["defines"],
                "expected default production translation, not another configuration")
        require(Path(manifest["repo"]).resolve() == repo, "HIPIFY repository mismatch")
        include = Path(manifest["output"]) / "gfx942/hip" / NATIVE
        # HIPIFY can emit initialization diagnostics despite translated output.
        # Keep its complete validation in the receipt; never call it clean.
    output = immutable_output(args.output, repo, [compiler, include])
    binary = output / "parity"
    harness = repo / SUITES[args.suite][1][0]
    if args.backend == "cpu" and args.suite == "exact-log":
        command = [str(compiler), "+nightly-2026-04-07", "--edition=2021", "-O",
                   str(repo / TESTS / "exact_log_cpu_parity_v1.rs"), "-o", str(binary)]
        version_command = [str(compiler), "+nightly-2026-04-07", "--version", "--verbose"]
    else:
        version_command = [str(compiler), "--version"]
        if args.backend == "cpu":
            flags = ["-std=c++17", "-O3", "-fno-fast-math", "-ffp-contract=off"]
        elif args.backend == "cuda":
            flags = ["-x", "cu", "-std=c++17", "-O3", "--fmad=false", "--ftz=false",
                     "--prec-div=true", "--prec-sqrt=true"]
            for arch in ("86", "89", "120"):
                flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
        else:
            flags = HIP_FLAGS
        command = [str(compiler), *flags, f"-DNEOETHOS_PARITY_{args.backend.upper()}=1",
                   "-I", str(include), str(harness)]
        if args.backend != "cpu" and args.suite == "first-hit":
            command.append(str(include / "prototype_b.cu"))
        command += ["-o", str(binary)]
    version = audit.run_command(version_command, repo, output / "compiler-version", 60)
    result = audit.run_command(command, repo, output / "build", 180) if audit.succeeded(version) else None
    drift = changed(pins) + audit.check_records([tool])
    post_hip = audit.validate(args.hip_manifest, args.hip_manifest_sha256) if hip_stage else None
    success = bool(result and audit.succeeded(result) and binary.is_file()
                   and binary.stat().st_size and not drift and (not post_hip or post_hip["success"]))
    receipt = dict(schema=SCHEMA, phase="build", suite=args.suite, backend=args.backend,
                   repo=str(repo), sources=pins, compiler=tool, version=version, result=result,
                   hip_stage=hip_stage, post_hip_stage=post_hip, drift=drift,
                   binary=audit.file_record(binary) if binary.is_file() else None,
                   success=success, device_executed=False, application_integrated=False,
                   proof_scope="compile and link this named fixture only, not parity")
    path = output / "receipt.json"
    audit.write_manifest(path, receipt)
    return path, receipt


def parse_results(content: str, suite: str, backend: str) -> list[dict]:
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    require(len(rows) >= 3 and all(isinstance(r, dict) for r in rows), "missing result records")
    meta, summary = rows[0], rows[-1]
    require(meta.get("type") == "metadata" and meta.get("schema") == SUITES[suite][0]
            and meta.get("backend") == backend, "backend/schema metadata mismatch")
    if suite == "first-hit":
        fingerprint = meta.get("input_identity")
        require(isinstance(fingerprint, dict)
                and fingerprint.get("algorithm") == "fnv1a64_le_v1_noncryptographic"
                and word(fingerprint.get("value")),
                "missing canonical first-hit input fingerprint")
        require(meta.get("fixtures") == 106 and meta.get("scope") == "first_hit_discrete_decisions_only",
                "wrong first-hit fixture coverage/scope")
        require(meta.get("role") == ("independent_reference" if backend == "cpu" else "production_kernel"),
                "wrong first-hit execution role")
        if backend != "cpu":
            device = meta.get("device")
            require(isinstance(device, dict) and bool(device.get("name"))
                    and type(device.get("ordinal")) is int and device["ordinal"] >= 0
                    and type(device.get("count")) is int and device["count"] > device["ordinal"]
                    and bool(device.get("architecture")) and device.get("warp_size") in (32, 64),
                    "missing actual first-hit device identity")
    else:
        require(meta.get("role") == ("production_cpu" if backend == "cpu" else "production_device"),
                "wrong production-math execution role")
        if backend != "cpu":
            require(bool(meta.get("device_name")) and bool(meta.get("architecture"))
                    and type(meta.get("device_ordinal")) is int and meta["device_ordinal"] >= 0
                    and meta.get("warp_size") in (32, 64), "missing actual math device identity")
    require(summary.get("type") == "summary" and type(summary.get("failures")) is int
            and summary["failures"] == 0,
            "missing or failed summary")
    require(summary.get("device_executed") is (backend != "cpu"),
            "device execution required; CPU is never a GPU substitute")
    cases = rows[1:-1]
    require(type(summary.get("cases")) is int and summary["cases"] == len(cases)
            and len(cases) == EXPECTED_CASES[suite], "zero, partial or inconsistent case count")
    if "cases" in meta:
        require(meta["cases"] == len(cases), "metadata case count mismatch")
    require(all(r.get("type") in ("case", "log", "primitive", "first_hit")
                and isinstance(r.get("id"), str) and r["id"] for r in cases),
            "unexpected or anonymous case record")
    require(len({r["id"] for r in cases}) == len(cases), "duplicate case IDs")
    for row in cases:
        if suite == "exact-log":
            require(row.get("type") == "case" and row.get("passed") is True
                    and type(row.get("accepted")) is bool
                    and all(word(row.get(k)) for k in ("input_bits", "b_bits", "c_bits", "output_bits"))
                    and row.get("operation") in ("log", "add", "sub", "mul", "div", "unfused", "fused")
                    and (row.get("accuracy_ulp") is None or type(row["accuracy_ulp"]) is int
                         and 0 <= row["accuracy_ulp"] <= 1), "failed/malformed mathematical checkpoint")
        else:
            expected, inputs = row.get("expected"), row.get("inputs")
            require(row.get("type") == "first_hit" and isinstance(expected, dict)
                    and set(expected) == {"exit_bar", "exit_reason"}
                    and all(type(v) is int for v in expected.values())
                    and row.get("result") == expected,
                    "failed independent first-hit expectation")
            require(isinstance(inputs, dict)
                    and all(type(inputs.get(k)) is int for k in ("rows", "entry_bar", "last_bar", "direction", "precedence"))
                    and 0 <= inputs["entry_bar"] < inputs["last_bar"] < inputs["rows"]
                    and inputs["direction"] in (-1, 1) and inputs["precedence"] in (0, 1)
                    and all(word(inputs.get(k)) for k in ("stop_f64_bits", "target_f64_bits")),
                    "missing/malformed first-hit inputs")
    return cases


def run(args: argparse.Namespace) -> tuple[Path, dict]:
    build_receipt = read_receipt(args.build_receipt, args.build_sha256, "build")
    require(build_receipt["success"], "refusing failed build")
    repo = Path(build_receipt["repo"])
    records = [*build_receipt["sources"], build_receipt["binary"]]
    require(not changed(records), "source/binary drift since build")
    output = immutable_output(args.output, repo, [Path(build_receipt["binary"]["path"])])
    command = [build_receipt["binary"]["path"]]
    if build_receipt["suite"] == "exact-log":
        command.append(str(repo / FIXTURE))
    elif build_receipt["backend"] != "cpu":
        command += ["--device", "0"]
    # Default ordinal zero is explicit in both GPU harnesses. This fixture tool
    # has no automatic device selection, rentals, data import or trade actions.
    result = audit.run_command(command, repo, output / "execute", 60)
    cases, metadata, error = [], None, None
    try:
        require(audit.succeeded(result), "executable failed; inspect complete stdout/stderr")
        content = Path(result["stdout"]["path"]).read_text(encoding="utf-8")
        cases = parse_results(content, build_receipt["suite"], build_receipt["backend"])
        metadata = json.loads(next(line for line in content.splitlines() if line.strip()))
    except (audit.StageError, ValueError, OSError) as exc:
        error = str(exc)
    drift = changed(records)
    receipt = dict(schema=SCHEMA, phase="run", suite=build_receipt["suite"],
                   backend=build_receipt["backend"], source_identity=identity(build_receipt["sources"]),
                   build_receipt=audit.file_record(args.build_receipt), binary=build_receipt["binary"],
                   result=result, error=error, metadata=metadata, cases=cases, drift=drift,
                   success=not error and not drift, application_integrated=False,
                   proof_scope="named fixture execution only; requires cross-backend comparison")
    path = output / "receipt.json"
    audit.write_manifest(path, receipt)
    return path, receipt


def compare_receipts(receipts: list[dict]) -> int:
    require(len(receipts) == 3 and {r.get("backend") for r in receipts} == {"cpu", "cuda", "hip"},
            "exactly one CPU, CUDA and HIP run required")
    first = next(r for r in receipts if r["backend"] == "cpu")
    require(all(r.get("success") is True and r.get("phase") == "run" and r.get("schema") == SCHEMA
                for r in receipts), "failed/unrecognized run is not parity")
    require(all(r["suite"] == first["suite"] and r["source_identity"] == first["source_identity"]
                for r in receipts), "different fixture or production sources")
    require(len(first["cases"]) == EXPECTED_CASES[first["suite"]], "partial parity result")
    for other in receipts:
        require(other["metadata"].get("input_identity") == first["metadata"].get("input_identity"),
                "different runtime input fingerprints")
        require(other["cases"] == first["cases"], f"bit/discrete mismatch: {other['backend']}")
    return len(first["cases"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    item = sub.add_parser("build")
    item.add_argument("--repo", type=Path, required=True)
    item.add_argument("--suite", choices=SUITES, required=True)
    item.add_argument("--backend", choices=("cpu", "cuda", "hip"), required=True)
    item.add_argument("--compiler", type=Path, required=True)
    item.add_argument("--hip-manifest", type=Path)
    item.add_argument("--hip-manifest-sha256")
    item.add_argument("--output", type=Path, required=True)
    item = sub.add_parser("run")
    item.add_argument("--build-receipt", type=Path, required=True)
    item.add_argument("--build-sha256", required=True)
    item.add_argument("--output", type=Path, required=True)
    item = sub.add_parser("compare")
    item.add_argument("--run-receipt", type=Path, action="append", required=True)
    item.add_argument("--run-sha256", action="append", required=True)
    args = parser.parse_args()
    try:
        if args.mode == "compare":
            require(len(args.run_receipt) == len(args.run_sha256) == 3, "three receipts and hashes required")
            receipts = [read_receipt(p, h, "run") for p, h in zip(args.run_receipt, args.run_sha256)]
            for r in receipts:
                require(not audit.check_records([r["result"]["stdout"], r["result"]["stderr"]]),
                        "execution log drift")
                content = Path(r["result"]["stdout"]["path"]).read_text(encoding="utf-8")
                actual = parse_results(content, r["suite"], r["backend"])
                require(json.loads(next(line for line in content.splitlines() if line.strip())) == r["metadata"],
                        "metadata differs from original execution")
                require(actual == r["cases"] and audit.succeeded(r["result"]), "invalid execution evidence")
            count = compare_receipts(receipts)
            print(json.dumps({"suite": receipts[0]["suite"], "three_backend_parity": True,
                              "cases": count, "application_integrated": False}))
            return 0
        path, result = build(args) if args.mode == "build" else run(args)
        print(json.dumps({"receipt": str(path), "sha256": audit.digest(path),
                          "success": result["success"], "phase": result["phase"]}))
        return 0 if result["success"] else 1
    except (audit.StageError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Native backend parity refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
