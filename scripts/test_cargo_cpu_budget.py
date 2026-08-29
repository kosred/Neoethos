#!/usr/bin/env python3
"""Prove Cargo inherits one portable effective-threads-minus-two budget."""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import tempfile
import textwrap
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".cargo" / "config.toml"


class CargoCpuBudgetContractTests(unittest.TestCase):
    def test_repo_config_reserves_two_threads_without_nested_rustc_pool(self) -> None:
        config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            config.get("build", {}).get("jobs"),
            -2,
            "Cargo must derive its process-wide jobserver width as effective logical threads - 2",
        )

        for target, target_config in config.get("target", {}).items():
            rustflags = target_config.get("rustflags", [])
            flattened = " ".join(str(flag) for flag in rustflags)
            self.assertNotRegex(
                flattened,
                r"(?:^|\s)-Z\s+threads(?:=|\s)|(?:^|\s)-Zthreads(?:=|\s)",
                f"{target} creates an unmanaged nested rustc frontend pool",
            )

        with tempfile.TemporaryDirectory(prefix="neoethos-cargo-budget-") as tmp_raw:
            probe = pathlib.Path(tmp_raw)
            (probe / "src").mkdir()
            (probe / "Cargo.toml").write_text(
                textwrap.dedent(
                    """\
                    [workspace]

                    [package]
                    name = "neoethos-cargo-budget-probe"
                    version = "0.0.0"
                    edition = "2024"
                    build = "build.rs"
                    """
                ),
                encoding="utf-8",
            )
            (probe / "src" / "lib.rs").write_text("pub fn probe() {}\n", encoding="utf-8")
            (probe / "build.rs").write_text(
                textwrap.dedent(
                    """\
                    fn main() {
                        let available = std::thread::available_parallelism()
                            .map(|value| value.get())
                            .unwrap_or(1);
                        let jobs = std::env::var("NUM_JOBS").unwrap_or_default();
                        println!(
                            "cargo:warning=NEOETHOS_CARGO_BUDGET available={} jobs={}",
                            available, jobs
                        );
                    }
                    """
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.pop("CARGO_BUILD_JOBS", None)
            completed = subprocess.run(
                [
                    "cargo",
                    "check",
                    "--offline",
                    "--manifest-path",
                    str(probe / "Cargo.toml"),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            match = re.search(
                r"NEOETHOS_CARGO_BUDGET available=(\d+) jobs=(\d+)",
                completed.stdout,
            )
            self.assertIsNotNone(match, completed.stdout)
            available, jobs = (int(value) for value in match.groups())
            expected = max(1, available - 2)
            self.assertEqual(jobs, expected)
            print(
                json.dumps(
                    {
                        "schema": "neoethos.cargo.cpu_budget.v1",
                        "available_parallelism": available,
                        "cargo_jobs": jobs,
                        "reserved_threads": available - jobs,
                        "nested_rustc_threads_flag": False,
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
