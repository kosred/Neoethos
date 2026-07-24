#!/usr/bin/env python3
"""Generate or execute an attributed rented-NVIDIA benchmark matrix.

Only workloads with a real execution adapter are marked executable. Missing
Prototype B/C, historical-legacy, or snapshot adapters remain blocked in the
manifest instead of producing fake benchmark JSON.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
from typing import Any

LEGACY_SHA = "2be1408ee3986026fdbb2a5a74aaaf6ac67e5209"
PASSES = ("clean_timing", "diagnostics", "nsight_systems", "nsight_compute")
SNAPSHOT_TIMEFRAMES = ("H1", "M30", "M15", "M5", "M1")


@dataclasses.dataclass(frozen=True)
class Job:
    job_id: str
    ref_name: str
    git_sha: str
    worktree: str
    timeframe: str
    prototype: str
    benchmark_pass: str
    fixture: str
    executable: bool
    blocked_reason: str | None
    output: str
    command: list[str]
    environment: dict[str, str]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def checked_head(path: pathlib.Path, expected: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"missing pinned worktree: {path}")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    if actual != expected:
        raise SystemExit(f"worktree {path} is at {actual}, expected {expected}")
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=path).returncode != 0
    if dirty:
        raise SystemExit(f"worktree is dirty: {path}")


def candidate_binary(worktree: pathlib.Path) -> pathlib.Path:
    name = "neoethos-cli.exe" if os.name == "nt" else "neoethos-cli"
    return worktree / "target" / "release" / name


def wrapped_command(
    binary: pathlib.Path,
    bench_args: list[str],
    benchmark_pass: str,
    output: pathlib.Path,
) -> tuple[list[str], dict[str, str]]:
    environment: dict[str, str] = {}
    command = [str(binary), "bench", *bench_args]
    if benchmark_pass == "diagnostics":
        environment["NEOETHOS_GPU_TIMING"] = "1"
    elif benchmark_pass == "nsight_systems":
        trace = output.with_suffix("")
        command = [
            "nsys", "profile", "--force-overwrite=true", "--trace=cuda,nvtx,osrt",
            "--output", str(trace), *command,
        ]
    elif benchmark_pass == "nsight_compute":
        report = output.with_suffix("")
        command = [
            "ncu", "--target-processes", "all", "--set", "full",
            "--force-overwrite", "--export", str(report), *command,
        ]
    return command, environment


def build_jobs(args: argparse.Namespace) -> list[Job]:
    root = args.worktrees_root.resolve()
    candidate = root / "candidate"
    legacy = root / "legacy"
    checked_head(candidate, args.candidate_sha)
    checked_head(legacy, args.legacy_sha)

    if args.fixture == "snapshot":
        if args.dataset is None or not args.dataset.is_file():
            raise SystemExit("--fixture snapshot requires --dataset <file>")
        if args.config is None or not args.config.is_file():
            raise SystemExit("--fixture snapshot requires --config <file>")
        dataset_hash = sha256(args.dataset)
        config_hash = sha256(args.config)
        timeframes = tuple(x.strip() for x in args.timeframes.split(",") if x.strip())
    else:
        dataset_hash = stable_hash({"fixture": "tiny-population-v1"})
        config_hash = stable_hash(
            {
                "population": args.population,
                "bars": args.bars,
                "features": args.features,
                "batch_size": args.batch_size,
            }
        )
        timeframes = ("TINY",)

    prototypes = tuple(x.strip().lower() for x in args.prototypes.split(",") if x.strip())
    jobs: list[Job] = []
    for ref_name, git_sha, worktree in (
        ("legacy", args.legacy_sha, legacy),
        ("candidate", args.candidate_sha, candidate),
    ):
        for timeframe in timeframes:
            for prototype in (("legacy",) if ref_name == "legacy" else prototypes):
                for benchmark_pass in PASSES:
                    output = (
                        args.out
                        / ref_name
                        / timeframe
                        / prototype
                        / f"{benchmark_pass}.json"
                    ).resolve()
                    executable = False
                    blocked: str | None = None
                    command: list[str] = []
                    environment: dict[str, str] = {}

                    if ref_name == "legacy":
                        blocked = (
                            "historical commit predates the attributed bench adapter; "
                            "a legacy execution adapter is required before comparison"
                        )
                    elif args.fixture == "snapshot":
                        blocked = (
                            "snapshot loader/executor is not implemented in Stage 1; "
                            "manifest only"
                        )
                    elif prototype != "a":
                        blocked = f"Prototype {prototype.upper()} has no executable kernel yet"
                    else:
                        binary = candidate_binary(candidate)
                        bench_args = [
                            "--execute-tiny",
                            "--git-sha", git_sha,
                            "--baseline-sha", args.legacy_sha,
                            "--dataset-hash", dataset_hash,
                            "--config-hash", config_hash,
                            "--timeframe", timeframe,
                            "--backend", "gpu_required",
                            "--prototype", prototype,
                            "--fixture", "tiny",
                            "--pass", benchmark_pass,
                            "--population", str(args.population),
                            "--batch-size", str(args.batch_size),
                            "--bars", str(args.bars),
                            "--features", str(args.features),
                            "--warmups", str(args.warmups),
                            "--repetitions", str(args.repetitions),
                            "--out", str(output),
                        ]
                        command, environment = wrapped_command(
                            binary, bench_args, benchmark_pass, output
                        )
                        executable = True

                    job_id = "/".join(
                        (ref_name, timeframe, prototype, benchmark_pass)
                    )
                    jobs.append(
                        Job(
                            job_id=job_id,
                            ref_name=ref_name,
                            git_sha=git_sha,
                            worktree=str(worktree),
                            timeframe=timeframe,
                            prototype=prototype,
                            benchmark_pass=benchmark_pass,
                            fixture=args.fixture,
                            executable=executable,
                            blocked_reason=blocked,
                            output=str(output),
                            command=command,
                            environment=environment,
                        )
                    )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--legacy-sha", default=LEGACY_SHA)
    parser.add_argument(
        "--worktrees-root",
        type=pathlib.Path,
        default=pathlib.Path("cache/gpu-bench/worktrees"),
    )
    parser.add_argument("--fixture", choices=("tiny", "snapshot"), default="tiny")
    parser.add_argument("--dataset", type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("cache/gpu-bench/runs"))
    parser.add_argument("--timeframes", default=",".join(SNAPSHOT_TIMEFRAMES))
    parser.add_argument("--prototypes", default="a,b,c")
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bars", type=int, default=4096)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-blocked", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "matrix.json"
    manifest.write_text(
        json.dumps([dataclasses.asdict(job) for job in jobs], indent=2) + "\n",
        encoding="utf-8",
    )
    executable = [job for job in jobs if job.executable]
    blocked = [job for job in jobs if not job.executable]
    print(
        f"wrote {manifest}: total={len(jobs)} executable={len(executable)} "
        f"blocked={len(blocked)}"
    )
    for job in jobs:
        if job.executable:
            print(shlex.join(job.command))
        else:
            print(f"BLOCKED {job.job_id}: {job.blocked_reason}")

    if not args.execute:
        return 0
    if blocked and not args.skip_blocked:
        raise SystemExit(
            "matrix contains blocked jobs; use --skip-blocked to execute only supported jobs"
        )
    for job in executable:
        output = pathlib.Path(job.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        binary = candidate_binary(pathlib.Path(job.worktree))
        if not binary.is_file():
            raise SystemExit(
                f"missing release binary {binary}; build the pinned candidate worktree first"
            )
        env = os.environ.copy()
        env.update(job.environment)
        subprocess.run(job.command, cwd=job.worktree, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
