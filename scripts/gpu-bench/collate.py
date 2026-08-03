#!/usr/bin/env python3
"""Collate NeoEthos benchmark JSON without inventing missing measurements."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

FIELDS = (
    "ref", "timeframe", "prototype", "pass", "engine_status", "parity_matched",
    "coverage", "median_seconds", "p95_seconds", "candidates_per_second",
    "candidate_bars_per_second", "peak_vram_bytes", "h2d_bytes", "d2h_bytes",
)


def row(path: pathlib.Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    identity = data["identity"]
    coverage = data["coverage"]
    total = coverage.get("total_candidates", 0)
    supported = coverage.get("supported_candidates", 0)
    return {
        "ref": path.parts[-4] if len(path.parts) >= 4 else "unknown",
        "timeframe": identity.get("timeframe"),
        "prototype": identity.get("prototype"),
        "pass": identity.get("pass"),
        "engine_status": data.get("engine_status"),
        "parity_matched": data.get("parity", {}).get("matched"),
        "coverage": supported / total if total else None,
        "median_seconds": data.get("total_wall_seconds", {}).get("median"),
        "p95_seconds": data.get("total_wall_seconds", {}).get("p95"),
        "candidates_per_second": data.get("throughput", {}).get("candidates_per_second"),
        "candidate_bars_per_second": data.get("throughput", {}).get("candidate_bars_per_second"),
        "peak_vram_bytes": data.get("throughput", {}).get("peak_vram_bytes"),
        "h2d_bytes": data.get("transfers", {}).get("h2d_bytes"),
        "d2h_bytes": data.get("transfers", {}).get("d2h_bytes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("cache/gpu-bench/summary.csv"))
    args = parser.parse_args()
    paths = sorted(p for p in args.root.rglob("*.json") if p.name != "matrix.json")
    rows = [row(path) for path in paths]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} with {len(rows)} reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
