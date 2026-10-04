#!/usr/bin/env python3
"""Run independent mapping tools with bounded waits and explicit failure evidence."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(label, command, output, timeout):
    record = {"command": command, "timeout_seconds": timeout}
    try:
        with (output / (label + ".stdout")).open("w") as stdout, (output / (label + ".stderr")).open("w") as stderr:
            # A process group ensures a timeout also stops tool subprocesses.
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
                record.update(status="completed" if code == 0 else "failed", exit_code=code)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                record.update(status="timeout", exit_code=None)
    except OSError as error:
        record.update(status="unavailable", detail=str(error))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("map_repo.py")
    checks = {}
    if args.semantic:
        checks["rust_scip"] = run("rust-scip", ["rustup", "run", "nightly-2026-04-07",
                                                  "rust-analyzer", "scip", "."], output, 1200)
        if checks["rust_scip"]["status"] == "completed" and Path("index.scip").is_file():
            checks["scip_json"] = run("scip-json", ["scip", "print", "--json", "index.scip"], output, 120)
            Path("index.scip").replace(output / "index.scip")
        elif checks["rust_scip"]["status"] == "completed":
            checks["rust_scip"]["status"] = "missing_index"
        command = [sys.executable, str(script), "--output", str(output)]
        if checks.get("scip_json", {}).get("status") == "completed":
            command += ["--scip-json", str(output / "scip-json.stdout")]
    else:
        files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
        (output / "project-files.txt").write_text("\n".join(p for p in files if p and not p.startswith("vendor/")) + "\n")
        checks["cloc"] = run("cloc", ["cloc", "--list-file=" + str(output / "project-files.txt"),
                                      "--json", "--by-file", "--skip-uniqueness", "--timeout=0",
                                      "--out=" + str(output / "loc.json")], output, 180)
        project_paths = [p for p in files if p and not p.startswith("vendor/") and not Path(p).is_symlink()]
        checks["jscpd"] = run("jscpd", ["jscpd", "--no-gitignore", "--min-lines", "12",
                                        "--min-tokens", "80", "--max-size", "16mb", "--mode", "weak",
                                        "--ignore-identifiers", "--fail-on-empty", "--reporters", "json,html",
                                        "--output", str(output / "duplicates"), *project_paths], output, 300)
        checks["cargo_metadata"] = run("cargo-metadata", ["cargo", "+stable", "metadata", "--no-deps",
                                                           "--format-version", "1", "--frozen"], output, 120)
        checks["rust_scip"] = {"status": "not_requested", "profile": "repo-nightly-default-features"}
        command = [sys.executable, str(script), "--output", str(output)]
    checks["graph"] = run("graph", command, output, 300)
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    message = "# Repository map\n\n"
    for name, record in checks.items():
        message += f"- **{name}**: `{record['status']}`\n"
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text())
        message += f"\nCommit `{summary['commit']}`: {summary['files']} files, {summary['nodes']} nodes, {summary['edges']} edges.\n"
    message += "\nDownload the map artifact and open `index.html`; the query database and duplication details have separate artifacts. `checks.json` and `coverage.json` record gaps. Syntax candidates are not resolved calls or permission to delete code.\n"
    (output / "README.md").write_text(message)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as file:
            file.write(message)
    print(message)
    # Tool failures remain visible rather than turning an incomplete map into a green full-coverage claim.
    return int(any(c["status"] not in {"completed", "not_requested"} for c in checks.values()))


if __name__ == "__main__":
    raise SystemExit(main())
