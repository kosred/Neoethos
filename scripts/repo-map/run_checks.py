#!/usr/bin/env python3
"""Run independent mapping tools with bounded waits and explicit failure evidence."""
import argparse
import json
import os
import shutil
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
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    import signal
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                record.update(status="timeout", exit_code=None)
    except OSError as error:
        record.update(status="unavailable", detail=str(error))
    return record


def duplication_check(output, baseline_ref=None):
    # npm's Windows .cmd shim cannot be launched directly by subprocess.
    native = Path(os.environ.get("LOCALAPPDATA", ".")) / "neoethos/tools/code-intelligence/node_modules/jscpd-windows-x64-msvc/bin/jscpd.exe"
    executable = str(native) if os.name == "nt" and native.is_file() else shutil.which("jscpd") or "jscpd"
    command = [executable, "--config", ".jscpd.json", "--fail-on-empty",
               "--reporters", "json,html", "--output", str(output / "duplicates")]
    if baseline_ref:
        command += ["--baseline-from-ref", baseline_ref, "--fail-on-new-clones", "0"]
    report = output / "duplicates/jscpd-report.json"
    report.unlink(missing_ok=True)  # Never attach statistics from an earlier run to a failure.
    record = run("jscpd", command, output, 300)
    if record["status"] in {"completed", "failed"} and report.is_file():
        stats = json.loads(report.read_text(encoding="utf-8"))["statistics"]["total"]
        record["statistics"] = {key: stats.get(key) for key in
                                ("sources", "clones", "duplicatedLines", "percentage", "newClones")}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--duplicates-only", action="store_true", help="Check working-tree duplication without rebuilding the map")
    parser.add_argument("--baseline-ref", help="Fail on new clones compared with this Git commit/ref")
    args = parser.parse_args()
    if args.semantic and (args.duplicates_only or args.baseline_ref):
        parser.error("semantic indexing and duplication checking are separate modes")
    if args.duplicates_only and not args.baseline_ref:
        parser.error("--duplicates-only requires --baseline-ref")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.duplicates_only:
        record = duplication_check(output, args.baseline_ref)
        (output / "checks.json").write_text(json.dumps({"jscpd": record}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
        return int(record["status"] != "completed")
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
            command += ["--scip-source-commit", subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()]
    else:
        files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
        (output / "project-files.txt").write_text("\n".join(p for p in files if p and not p.startswith("vendor/")) + "\n")
        checks["cloc"] = run("cloc", ["cloc", "--list-file=" + str(output / "project-files.txt"),
                                      "--json", "--by-file", "--skip-uniqueness", "--timeout=0",
                                      "--out=" + str(output / "loc.json")], output, 180)
        checks["jscpd"] = duplication_check(output, args.baseline_ref)
        checks["cargo_metadata"] = run("cargo-metadata", ["cargo", "+stable", "metadata", "--no-deps",
                                                           "--format-version", "1", "--frozen"], output, 120)
        checks["rust_scip"] = {"status": "not_requested", "profile": "repo-nightly-default-features"}
        command = [sys.executable, str(script), "--output", str(output)]
    checks["graph"] = run("graph", command, output, 300)
    if args.semantic and checks["graph"]["status"] == "completed":
        checks["database_parts"] = run("database-parts", [sys.executable, str(script.with_name("pack_graph.py")),
                                           str(output / "graph.sqlite"), str(output / "parts")], output, 180)
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    message = "# Repository map\n\n"
    for name, record in checks.items():
        message += f"- **{name}**: `{record['status']}`\n"
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text())
        message += f"\nCommit `{summary['commit']}`: {summary['files']} files, {summary['nodes']} nodes, {summary['edges']} edges.\n"
    if (output / "architecture.json").exists():
        architecture = json.loads((output / "architecture.json").read_text())
        rust = architecture["rust_semantic_coverage"]
        message += f"\nRust documents indexed: {rust['indexed_files']} / {rust['tracked_project_files']}. See `architecture.md` and `architecture.json` for component sizes, dependencies, duplication and gaps.\n"
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
