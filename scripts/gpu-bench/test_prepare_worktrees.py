#!/usr/bin/env python3
"""Integration test for pinned benchmark worktree preparation."""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
import unittest


SOURCE_SCRIPT = pathlib.Path(__file__).with_name("prepare_worktrees.sh")


def run(*args: str, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


class PrepareWorktreesTests(unittest.TestCase):
    def test_prepares_clean_candidate_and_legacy_at_exact_shas(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = pathlib.Path(raw_tmp)
            repo = tmp / "repo"
            repo.mkdir()
            run("git", "init", "-b", "master", cwd=repo)
            run("git", "config", "user.name", "GPU Bench Test", cwd=repo)
            run("git", "config", "user.email", "gpu-bench@example.invalid", cwd=repo)

            tracked = repo / "tracked.txt"
            tracked.write_text("legacy\n", encoding="utf-8")
            run("git", "add", "tracked.txt", cwd=repo)
            run("git", "commit", "-m", "legacy", cwd=repo)
            legacy_sha = run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()

            tracked.write_text("candidate\n", encoding="utf-8")
            run("git", "commit", "-am", "candidate", cwd=repo)
            candidate_sha = run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()

            normalized_script = tmp / "prepare_worktrees.sh"
            normalized_script.write_text(
                SOURCE_SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n"),
                encoding="utf-8",
            )
            root = repo / "cache" / "gpu-bench" / "worktrees"
            completed = run(
                "bash",
                str(normalized_script),
                str(root),
                candidate_sha,
                legacy_sha,
                cwd=repo,
            )
            self.assertIn("Prepared pinned worktrees", completed.stdout)

            manifest = json.loads((root / "worktrees.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["candidate"]["sha"], candidate_sha)
            self.assertEqual(manifest["legacy"]["sha"], legacy_sha)
            for name, expected in (
                ("candidate", candidate_sha),
                ("legacy", legacy_sha),
            ):
                worktree = root / name
                actual = run("git", "rev-parse", "HEAD", cwd=worktree).stdout.strip()
                self.assertEqual(actual, expected)
                run("git", "diff", "--quiet", cwd=worktree)


if __name__ == "__main__":
    unittest.main()
