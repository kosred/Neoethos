#!/usr/bin/env python3
"""Stage and audit official hipify-clang translation; never build or run a GPU.

The native TU list comes from DEVICE_SOURCES in the real Rust builder. This is
translation tooling, not a replacement production build or numeric proof.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time

SCHEMA = "neoethos.hipify-source-stage.v1"
GPU_BUILD = Path("crates/neoethos-gpu-cuda/build.rs")
VECTOR_BUILD = Path("vendor/vector-ta-0.2.9-patched/build.rs")
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".cuh", ".inc", ".inl", ".def"}
HIDDEN_INCLUDE_ENV = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "CUDAHOSTCXX")


class StageError(Exception):
    """An explicit refusal, never a successful translation."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def beneath(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def regular_file(path: Path) -> Path:
    result = path.resolve(strict=True)
    if not result.is_file():
        raise StageError(f"not a regular file: {path}")
    return result


def rust_string_array(text: str, name: str) -> list[str]:
    pattern = rf"\bconst\s+{re.escape(name)}\s*:\s*\[&str;\s*(\d+)\s*\]\s*=\s*\[(.*?)\];"
    matches = list(re.finditer(pattern, text, re.S))
    if len(matches) != 1:
        raise StageError(f"expected exactly one literal {name} array")
    body = matches[0].group(2)
    # Deliberately accepts only the builder's literal list grammar. A source
    # refactor to generated entries must be reviewed, not partially parsed.
    tokens = re.findall(r'//[^\n]*|"[^"\\\r\n]*"|\s+|,|.', body)
    values = []
    for token in tokens:
        if token.startswith('//') or token.isspace() or token == ',':
            continue
        if token.startswith('"') and token.endswith('"'):
            values.append(token[1:-1])
        else:
            raise StageError(f"unsupported {name} list token: {token!r}")
    if len(values) != int(matches[0].group(1)) or not values or len(set(values)) != len(values):
        raise StageError(f"{name} count/uniqueness mismatch")
    return values


def source_path(repo: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise StageError(f"unsafe source path: {relative}")
    result = regular_file(repo / relative)
    if not beneath(result, repo):
        raise StageError(f"source escapes repository: {relative}")
    return result


def discover(repo: Path) -> tuple[list[Path], list[Path], list[Path]]:
    build = source_path(repo, GPU_BUILD)
    vector_build = source_path(repo, VECTOR_BUILD)
    native_root = build.parent / "native"
    native = []
    for entry in rust_string_array(build.read_text(encoding="utf-8"), "DEVICE_SOURCES"):
        relative = PurePosixPath(entry)
        if relative.parts[0] != "native" or relative.suffix != ".cu":
            raise StageError(f"unexpected native translation unit: {entry}")
        native.append(source_path(repo, GPU_BUILD.parent / Path(relative)))
    # Bind the extra TU to the real compile_kernel call, not a second TU list.
    vector_text = vector_build.read_text(encoding="utf-8")
    calls = re.findall(
        r'\bcompile_kernel\s*\(\s*&cuda_path\s*,\s*"([^"\n]+)"\s*,\s*"neoethos_f64_kernels"\s*,?\s*\)',
        vector_text,
    )
    if len(calls) != 1:
        raise StageError("expected one actual vector-ta neoethos_f64_kernels producer")
    vector = source_path(repo, VECTOR_BUILD.parent / Path(PurePosixPath(calls[0])))
    if vector.suffix != ".cu":
        raise StageError("vector-ta f64 producer is not a CUDA TU")
    tus = native + [vector]
    headers = set()
    for candidate in native_root.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in HEADER_SUFFIXES:
            headers.add(source_path(repo, candidate.relative_to(repo)))
    # Follow quoted project includes to preserve their relative directory
    # geometry. System CUDA/C++ headers are read from explicitly recorded roots.
    pending = list(tus) + list(headers)
    visited = set()
    while pending:
        item = pending.pop()
        if item in visited:
            continue
        visited.add(item)
        for name in re.findall(r'^\s*#\s*include\s*"([^"\r\n]+)"', item.read_text(encoding="utf-8"), re.M):
            candidates = [item.parent / name, native_root / name]
            found = next((p for p in candidates if p.is_file()), None)
            if found is None:
                raise StageError(f"unresolved quoted project include {name!r} in {item}")
            included = found.resolve(strict=True)
            if not beneath(included, repo):
                raise StageError(f"project include escapes repository: {included}")
            if included.suffix.lower() not in HEADER_SUFFIXES:
                raise StageError(f"unexpected quoted include kind: {included}")
            headers.add(included)
            pending.append(included)
    return tus, sorted(headers), [build, vector_build]


def checked_output(raw: Path, repo: Path, protected: list[Path]) -> Path:
    output = raw.resolve()
    if output.exists() or not output.parent.is_dir():
        raise StageError("output must be a NEW directory with an existing parent")
    if beneath(output, repo) or beneath(repo, output) or output == Path.home().resolve():
        raise StageError("output must be outside the repository and broad roots")
    for item in protected:
        if beneath(output, item) or beneath(item, output):
            raise StageError(f"output overlaps a tool/include source: {item}")
    return output


def include_records(roots: list[Path]) -> list[dict]:
    records = {}
    for root in roots:
        for item in root.rglob("*"):
            if item.is_file() and item.suffix.lower() in HEADER_SUFFIXES:
                resolved = item.resolve(strict=True)
                if not beneath(resolved, root):
                    raise StageError(f"include symlink escapes recorded root: {item}")
                records[str(item)] = file_record(item)
    return [records[key] for key in sorted(records)]


def run_command(argv: list[str], cwd: Path, log_base: Path, timeout: float) -> dict:
    log_base.parent.mkdir(parents=True, exist_ok=True)
    stdout = log_base.with_suffix(".stdout.log")
    stderr = log_base.with_suffix(".stderr.log")
    started = time.time()
    error = None
    timed_out = False
    code = None
    with stdout.open("xb") as out, stderr.open("xb") as err:
        try:
            # No shell, no in-memory log truncation, and only this owned child
            # is terminated if its finite timeout expires.
            result = subprocess.run(argv, cwd=cwd, stdout=out, stderr=err, timeout=timeout, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            error = str(exc)
        except OSError as exc:
            error = str(exc)
    return {"argv": argv, "cwd": str(cwd), "started_unix": started,
            "seconds": time.time() - started, "exit_code": code,
            "timed_out": timed_out, "error": error,
            "stdout": file_record(stdout), "stderr": file_record(stderr)}


def diagnostic_counts(logs: list[Path]) -> dict:
    # Recognize Clang/HIPIFY diagnostic lines, not indented statistics such as
    # "  error: 2" (the count of converted CUDA error API references).
    pattern = re.compile(r"^(?:\[HIPIFY\]\s*|.*:\d+(?::\d+)?:\s*)?(fatal error|error|warning):")
    counts = {"error_lines": 0, "warning_lines": 0}
    for path in logs:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = pattern.match(line)
                if match:
                    counts["warning_lines" if match[1] == "warning" else "error_lines"] += 1
    return counts


def succeeded(result: dict) -> bool:
    return result["exit_code"] == 0 and not result["error"] and not result["timed_out"]


def check_records(records: list[dict]) -> list[str]:
    drift = []
    for record in records:
        try:
            if file_record(Path(record["path"])) != record:
                drift.append(record["path"])
        except OSError:
            drift.append(record["path"])
    return drift


def source_membership_drift(manifest: dict) -> list[str]:
    repo = Path(manifest["repo"])
    try:
        tus, headers, builders = discover(repo)
        current = sorted(str(p) for p in set(tus + headers + builders))
        recorded = sorted(r["path"] for r in manifest["sources"])
        if (current != recorded
                or [str(p.relative_to(repo)) for p in tus] != manifest["translation_units"]
                or [str(p.relative_to(repo)) for p in headers] != manifest["headers"]):
            return ["project source membership changed"]
    except (OSError, StageError) as exc:
        return [f"project source discovery changed: {exc}"]
    return []


def write_manifest(path: Path, manifest: dict) -> None:
    # The directory is exclusively owned by this invocation. No resume/cache
    # overwrite: publish once after all children finish, including failures.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def stage(args: argparse.Namespace) -> tuple[Path, dict]:
    repo = args.repo.resolve(strict=True)
    tool = regular_file(args.hipify_clang)
    cuda = args.cuda_path.resolve(strict=True)
    cuda_include = cuda / "include"
    if not (cuda_include / "cuda_runtime.h").is_file() or not (cuda_include / "cuda.h").is_file():
        raise StageError("explicit CUDA path lacks include/cuda_runtime.h or cuda.h")
    extras = [p.resolve(strict=True) for p in args.include]
    resource = args.clang_resource_directory.resolve(strict=True) if args.clang_resource_directory else None
    roots = [cuda_include, *extras, *([resource / "include"] if resource else [])]
    if any(not root.is_dir() for root in roots):
        raise StageError("all explicit include/resource directories must exist")
    if args.jobs not in (1, 2) or not 0 < args.timeout_seconds <= 3600:
        raise StageError("jobs must be 1 or 2; timeout must be in (0, 3600]")
    for define in args.define:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*(?:=[^\r\n\x00]*)?", define):
            raise StageError(f"invalid preprocessor definition: {define!r}")
    for key in HIDDEN_INCLUDE_ENV:
        if os.environ.get(key):
            raise StageError(f"unset implicit parser input {key}; pass explicit include/tool arguments")
    tus, headers, builders = discover(repo)
    output = checked_output(args.output, repo, [tool, cuda, *roots])
    originals = [file_record(p) for p in sorted(set(tus + headers + builders))]
    external = include_records(roots)
    output.mkdir()
    partition = output / args.target
    inputs = partition / "input"
    hip = partition / "hip"
    staged = []
    for record in originals:
        original = Path(record["path"])
        destination = inputs / original.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        copied = file_record(destination)
        if copied["sha256"] != record["sha256"] or copied["bytes"] != record["bytes"]:
            raise StageError(f"source changed while staging: {original}; partial output retained at {output}")
        staged.append(copied)
    manifest = {"schema": SCHEMA, "mode": args.mode, "target": args.target,
                "cuda_parse_arch": "sm_86", "repo": str(repo), "output": str(output),
                "preprocessor_mode": "compiler-defined-active-branches",
                "inactive_branch_policy": "preserved untranslated; regenerate for changed explicit defines, including fixture-enabled builds",
                "tool": file_record(tool), "cuda_path": str(cuda), "include_roots": [str(p) for p in roots],
                "clang_resource_directory": str(resource) if resource else None,
                "defines": args.define, "jobs": args.jobs, "timeout_seconds": args.timeout_seconds,
                "native_translation_units": len(tus) - 1, "vector_translation_units": 1,
                "translation_units": [str(p.relative_to(repo)) for p in tus],
                "headers": [str(p.relative_to(repo)) for p in headers],
                "sources": originals, "staged_inputs": staged, "external_headers": external,
                "results": [], "outputs": [], "drift": [], "success": False,
                "proof_scope": "HIPIFY upstream host-side AST analysis/translation for the recorded compiler and defines only; inactive branches are preserved untranslated and require regeneration for another configuration. Success denotes tool exit/output/integrity, not clean diagnostics. No device-code validation, HIP compile, GPU execution, numeric parity or application integration"}
    version = run_command([str(tool), "--version"], partition, partition / "logs" / "tool-version", args.timeout_seconds)
    manifest["tool_version"] = version
    if succeeded(version):
        def process(item: Path) -> dict:
            relative = item.relative_to(repo)
            source = inputs / relative
            destination = hip / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            # This is an official HIPIFY option, before its Clang separator.
            # HIPIFY 7.2.3 still emits separate initialization-driver errors;
            # moving this option does not fix or suppress those diagnostics.
            argv = [str(tool), str(source), f"--cuda-path={cuda}",
                    "--cuda-gpu-arch=sm_86", "--default-preprocessor", "--print-stats"]
            if resource:
                argv.append(f"--clang-resource-directory={resource}")
            if args.mode == "analyze":
                argv.append("--no-output")
            else:
                argv.extend(["-o", str(destination)])
            argv.extend(["--", "-x", "cuda", "-std=c++17",
                         "-I", str(inputs / GPU_BUILD.parent / "native"), "-I", str(source.parent)])
            for include in extras:
                argv.extend(["-I", str(include)])
            argv.extend(f"-D{value}" for value in args.define)
            # Filename plus original suffix avoids TU/header stem collisions.
            log_base = partition / "logs" / relative.parent / (relative.name + ".hipify")
            result = run_command(argv, partition, log_base, args.timeout_seconds)
            result["source"] = str(relative)
            if args.mode == "translate":
                if destination.is_file() and destination.stat().st_size:
                    result["output"] = file_record(destination)
                else:
                    result["error"] = result["error"] or "translator did not produce a nonempty output file"
            return result
        # Fixed original order in manifest, bounded workers, all failures kept.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            pending = [(item, pool.submit(process, item)) for item in headers + tus]
            for item, future in pending:
                try:
                    manifest["results"].append(future.result())
                except OSError as exc:
                    # A per-file filesystem failure must not discard already
                    # completed child evidence or prevent the other children.
                    relative = item.relative_to(repo)
                    log_base = partition / "logs" / relative.parent / (relative.name + ".hipify")
                    failure = {"source": str(relative), "exit_code": None,
                               "timed_out": False, "error": str(exc)}
                    for stream in ("stdout", "stderr"):
                        log = log_base.with_suffix(f".{stream}.log")
                        failure[stream] = file_record(log) if log.is_file() else None
                    manifest["results"].append(failure)
        manifest["outputs"] = [r["output"] for r in manifest["results"] if "output" in r]
    manifest["drift"] = check_records(originals + staged + external + [manifest["tool"]])
    manifest["drift"].extend(source_membership_drift(manifest))
    # Detect added/deleted external headers as well as changed content.
    if include_records(roots) != external:
        manifest["drift"].append("external include tree membership/content changed")
    manifest["success"] = (succeeded(version) and len(manifest["results"]) == len(headers + tus)
                           and all(succeeded(r) for r in manifest["results"]) and not manifest["drift"])
    manifest["diagnostics"] = diagnostic_counts([
        Path(result[key]["path"]) for result in [version, *manifest["results"]]
        for key in ("stdout", "stderr") if result.get(key)])
    manifest["error_diagnostics_absent"] = manifest["diagnostics"]["error_lines"] == 0
    path = output / "manifest.json"
    write_manifest(path, manifest)
    return path, manifest


def validate(path: Path, expected: str | None = None) -> dict:
    path = regular_file(path)
    if expected and digest(path).lower() != expected.lower():
        raise StageError("manifest SHA256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("target") != "gfx942":
        raise StageError("unsupported manifest schema/target")
    records = [manifest["tool"], *manifest["sources"], *manifest["staged_inputs"],
               *manifest["external_headers"], *manifest["outputs"]]
    for result in [manifest["tool_version"], *manifest["results"]]:
        records.extend(result[key] for key in ("stdout", "stderr") if result.get(key))
    drift = check_records(records)
    drift.extend(source_membership_drift(manifest))
    if include_records([Path(p) for p in manifest["include_roots"]]) != manifest["external_headers"]:
        drift.append("external include tree membership/content changed")
    expected_sources = manifest["headers"] + manifest["translation_units"]
    complete = ([r["source"] for r in manifest["results"]] == expected_sources
                and succeeded(manifest["tool_version"])
                and all(succeeded(r) for r in manifest["results"]))
    if manifest["mode"] == "translate":
        complete = complete and len(manifest["outputs"]) == len(expected_sources)
    diagnostics = diagnostic_counts([
        Path(result[key]["path"]) for result in [manifest["tool_version"], *manifest["results"]]
        for key in ("stdout", "stderr") if result.get(key) and Path(result[key]["path"]).is_file()])
    if "diagnostics" in manifest and diagnostics != manifest["diagnostics"]:
        drift.append("diagnostic counts changed")
    return {"manifest": str(path), "sha256": digest(path), "checked_files": len(records),
            "drift": drift, "success": bool(manifest["success"] and complete and not drift),
            "diagnostics": diagnostics, "error_diagnostics_absent": diagnostics["error_lines"] == 0,
            "proof_scope": manifest["proof_scope"]}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="mode", required=True)
    for mode in ("analyze", "translate"):
        item = sub.add_parser(mode)
        item.add_argument("--repo", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--hipify-clang", type=Path, required=True)
        item.add_argument("--cuda-path", type=Path, required=True)
        item.add_argument("--target", choices=["gfx942"], default="gfx942")
        item.add_argument("--clang-resource-directory", type=Path)
        item.add_argument("--include", type=Path, action="append", default=[])
        item.add_argument("--define", action="append", default=[])
        item.add_argument("--jobs", type=int, choices=[1, 2], default=2)
        item.add_argument("--timeout-seconds", type=float, default=300)
    item = sub.add_parser("validate")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--expected-sha256")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.mode == "validate":
            result = validate(args.manifest, args.expected_sha256)
        else:
            path, manifest = stage(args)
            result = {"manifest": str(path), "sha256": digest(path), "success": manifest["success"],
                      "files": len(manifest["results"]), "drift": manifest["drift"],
                      "diagnostics": manifest["diagnostics"],
                      "error_diagnostics_absent": manifest["error_diagnostics_absent"],
                      "proof_scope": manifest["proof_scope"]}
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1
    except (StageError, OSError, ValueError, KeyError) as exc:
        print(f"HIPIFY staging refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
