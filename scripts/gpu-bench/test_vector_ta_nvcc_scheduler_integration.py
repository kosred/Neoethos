#!/usr/bin/env python3
"""Linux integration proof for VectorTA's Cargo-jobserver NVCC scheduler.

The fake compiler creates the requested artifacts while recording real process
overlap. This exercises Cargo's inherited jobserver instead of merely checking
for scheduler-shaped source text.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
VECTOR_TA = ROOT / "vendor" / "vector-ta-0.2.9-patched"
BUILD_RS = VECTOR_TA / "build.rs"
NIGHTLY = "nightly-2026-04-07"
WIDTH = 4

FAKE_NVCC = r'''#!/usr/bin/env python3
import fcntl
import json
import os
import pathlib
import sys
import time

args = sys.argv[1:]
if args == ["--list-gpu-arch"]:
    print("compute_89")
    raise SystemExit(0)

state = pathlib.Path(os.environ["FAKE_NVCC_STATE_DIR"])
lock_path = state / "lock"
current_path = state / "current"
maximum_path = state / "maximum"
events_path = state / "events.jsonl"

def locked_update(delta, event):
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = int(current_path.read_text(encoding="utf-8") or "0")
        current += delta
        current_path.write_text(str(current), encoding="utf-8")
        maximum = int(maximum_path.read_text(encoding="utf-8") or "0")
        maximum_path.write_text(str(max(maximum, current)), encoding="utf-8")
        with events_path.open("a", encoding="utf-8") as events:
            events.write(json.dumps({
                "event": event,
                "pid": os.getpid(),
                "active": current,
                "args": args,
                "time_ns": time.monotonic_ns(),
            }) + "\n")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

locked_update(1, "start")
try:
    time.sleep(float(os.environ.get("FAKE_NVCC_DELAY_SECONDS", "0.20")))
    fail_match = os.environ.get("FAKE_NVCC_FAIL_MATCH")
    if fail_match and any(fail_match in arg for arg in args):
        print(f"injected fake NVCC failure for {fail_match}", file=sys.stderr)
        raise SystemExit(17)
    try:
        output_index = args.index("-o") + 1
        output = pathlib.Path(args[output_index])
    except (ValueError, IndexError):
        print(f"fake NVCC invocation has no -o output: {args!r}", file=sys.stderr)
        raise SystemExit(18)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"neoethos-fake-nvcc-artifact\n")
finally:
    locked_update(-1, "end")
'''


def selected_sources(count: int) -> list[str]:
    source = BUILD_RS.read_text(encoding="utf-8")
    candidates = re.findall(r'"(kernels/cuda/[^"\n]+\.cu)"', source)
    unique = list(dict.fromkeys(candidates))
    if len(unique) < count:
        raise AssertionError(f"only found {len(unique)} CUDA source declarations")
    return unique[:count]


def reset_state(state: pathlib.Path) -> None:
    state.mkdir(parents=True, exist_ok=True)
    for name in ("current", "maximum"):
        (state / name).write_text("0", encoding="utf-8")
    (state / "events.jsonl").write_text("", encoding="utf-8")


def read_events(state: pathlib.Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (state / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


class VectorTaNvccSchedulerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        if sys.platform != "linux":
            self.skipTest("the process-overlap harness currently uses Linux fcntl")
        if shutil.which("cargo") is None:
            self.skipTest("cargo is unavailable")

    def cargo_check(
        self,
        project: pathlib.Path,
        target: pathlib.Path,
        fake_nvcc: pathlib.Path,
        state: pathlib.Path,
        sources: list[str],
        *,
        fail_match: str | None = None,
        nvcc_args: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "CARGO_BUILD_JOBS": str(WIDTH),
                "CARGO_TARGET_DIR": str(target),
                "CUDA_ARCHS": "89",
                "CUDA_FILTER": ",".join(sources),
                "CUDA_PATH": "/usr/local/cuda",
                "FAKE_NVCC_DELAY_SECONDS": "0.20",
                "FAKE_NVCC_STATE_DIR": str(state),
                "NVCC": str(fake_nvcc),
            }
        )
        if fail_match is not None:
            env["FAKE_NVCC_FAIL_MATCH"] = fail_match
        if nvcc_args is not None:
            env["NVCC_ARGS"] = nvcc_args
        return subprocess.run(
            [
                "cargo",
                f"+{NIGHTLY}",
                "check",
                "--offline",
                f"-j{WIDTH}",
            ],
            cwd=project,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )

    def test_parallel_width_and_failure_cancellation_use_cargos_budget(self) -> None:
        sources = selected_sources(12)
        with tempfile.TemporaryDirectory(prefix="neoethos-fake-nvcc-") as tmp_raw:
            tmp = pathlib.Path(tmp_raw)
            project = tmp / "probe"
            state = tmp / "state"
            target = tmp / "target"
            project.mkdir()
            (project / "src").mkdir()
            (project / "Cargo.toml").write_text(
                textwrap.dedent(
                    f'''\
                    [workspace]

                    [package]
                    name = "vector-ta-nvcc-scheduler-probe"
                    version = "0.0.0"
                    edition = "2021"

                    [dependencies]
                    vector-ta = {{ path = {json.dumps(VECTOR_TA.as_posix())}, features = ["cuda-build-ptx"] }}
                    '''
                ),
                encoding="utf-8",
            )
            (project / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
            fake_nvcc = tmp / "fake-nvcc"
            fake_nvcc.write_text(FAKE_NVCC, encoding="utf-8")
            fake_nvcc.chmod(0o755)

            reset_state(state)
            success = self.cargo_check(project, target, fake_nvcc, state, sources)
            self.assertEqual(
                success.returncode,
                0,
                "successful scheduler probe failed:\n" + "\n".join(success.stdout.splitlines()[-120:]),
            )
            maximum = int((state / "maximum").read_text(encoding="utf-8"))
            starts = [event for event in read_events(state) if event["event"] == "start"]
            self.assertGreater(maximum, 1, "NVCC remained serial")
            self.assertLessEqual(maximum, WIDTH, "NVCC exceeded Cargo's configured job budget")
            self.assertEqual(len(starts), len(sources) * 2)

            failure_sources = sources[:8]
            reset_state(state)
            failure = self.cargo_check(
                project,
                target,
                fake_nvcc,
                state,
                failure_sources,
                fail_match=failure_sources[0],
            )
            self.assertNotEqual(failure.returncode, 0, "injected NVCC failure was ignored")
            failed_starts = [
                event for event in read_events(state) if event["event"] == "start"
            ]
            self.assertLess(
                len(failed_starts),
                len(failure_sources) * 2,
                "scheduler started the entire queue after the first injected failure",
            )
            self.assertEqual((state / "current").read_text(encoding="utf-8"), "0")

            reset_state(state)
            hostile = self.cargo_check(
                project,
                target,
                fake_nvcc,
                state,
                sources[:4],
                nvcc_args="--use_fast_math",
            )
            self.assertNotEqual(hostile.returncode, 0, "unsafe free-form NVCC flags were accepted")
            self.assertIn("NVCC_ARGS is unsupported", hostile.stdout)
            self.assertEqual(
                read_events(state),
                [],
                "the build launched NVCC before rejecting unsafe free-form arguments",
            )
            print(
                json.dumps(
                    {
                        "schema": "neoethos.vector_ta.fake_nvcc_scheduler.v1",
                        "configured_width": WIDTH,
                        "observed_max_active_nvcc": maximum,
                        "successful_source_jobs": len(sources),
                        "successful_nvcc_invocations": len(starts),
                        "failure_source_jobs": len(failure_sources),
                        "nvcc_invocations_before_cancellation": len(failed_starts),
                        "hostile_nvcc_args_rejected_before_launch": True,
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
