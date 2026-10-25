#!/usr/bin/env python3
"""Read-only analysis of NeoEthos selection-stage funnel artifacts.

Only *_funnel.json files are opened. No final out-of-sample evaluation.
"""
import argparse
import collections
import json
from pathlib import Path


def summarize(profile, source):
    if not isinstance(profile, dict) or not isinstance(profile.get("stages"), list):
        raise ValueError("not a NeoEthos funnel profile")
    window = profile.get("stage1_evaluation_window") or {}
    n, a, b = (window.get(k) for k in ("selection_rows", "start_row", "end_row_exclusive"))
    pct = (100 * (b-a)/n) if all(type(x) is int for x in (n, a, b)) and 0 <= a < b <= n else None
    stages = {s.get("name"): s for s in profile["stages"]}
    reasons = collections.Counter()
    for stage in profile["stages"]:
        for pair in stage.get("top_reasons") or []:
            if isinstance(pair, list) and len(pair) == 2 and type(pair[1]) is int:
                reasons[str(stage.get("name")) + ":" + str(pair[0])] += pair[1]
    census = profile.get("candidate_census") or {}
    wf = profile.get("walkforward_selection_cohort") or {}
    trials = wf.get("trials") or []
    return dict(
        source=str(source), mode=profile.get("mode") or "unknown",
        symbol=profile.get("symbol") or "unknown",
        timeframe=profile.get("timeframe") or "unknown",
        stage1_pct=pct,
        ga=census.get("ga_returned_candidates"),
        quality=(stages.get("passed_quality") or {}).get("count_out"),
        wf_tested=census.get("walkforward_tested"),
        wf_passed=census.get("walkforward_passed"),
        wf_fold_positive=sum(t.get("positive_folds", 0) for t in trials),
        wf_fold_total=sum(t.get("reported_splits", 0) for t in trials),
        bottleneck=profile.get("bottleneck_stage") or "unknown",
        reasons=reasons.most_common(12),
    )


def report(rows):
    lines = [
        "# NeoEthos pre-OOS selection diagnostics", "",
        "This report reads only funnel files. It does NOT execute or inspect final OOS.",
        "Risky 60% refers to profitable walk-forward folds, not trade win rate.", "",
    ]
    if not rows:
        lines += [
            "NO RUN EVIDENCE FOUND. CI may test code, but cannot decide which Stage 1",
            "percentage performs better without genuine comparable run funnels.", "",
        ]
        return "\n".join(lines)
    lines += [
        "| Mode | Symbol / TF | Stage 1 | GA candidates | Quality | WF passed/tested | Bottleneck |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for x in rows:
        f = lambda n: "n/a" if n is None else str(n)
        lines.append(
            "| {mode} | {symbol}/{timeframe} | {pct} | {ga} | {quality} | {passed}/{tested} | {bottleneck} |".format(
                mode=x["mode"], symbol=x["symbol"], timeframe=x["timeframe"],
                pct="n/a" if x["stage1_pct"] is None else f'{x["stage1_pct"]:.1f}%',
                ga=f(x["ga"]), quality=f(x["quality"]),
                passed=f(x["wf_passed"]), tested=f(x["wf_tested"]),
                bottleneck=x["bottleneck"])
        )
    lines += ["", "## Recorded rejection reasons", ""]
    for x in rows:
        lines += [f'{x["mode"]} {x["symbol"]}/{x["timeframe"]} ({x["source"]}):']
        lines += [f'- {name}: {count}' for name, count in x["reasons"]]
        lines += [""]
    lines += [
        "Do not select Stage 1 percentage using final OOS.",
        "A good result requires stronger internal walk-forward evidence and positive net expectancy after costs.",
        "A 45–50% scalp trade win rate is not an automatic reject.", "",
    ]
    return "\n".join(lines)


def self_test():
    def fixture(mode, p):
        return {
            "mode": mode, "symbol": "EURUSD", "timeframe": "M5",
            "stage1_evaluation_window": {"selection_rows": 1000, "start_row": 0, "end_row_exclusive": p},
            "stages": [{"name": "passed_quality", "count_out": 4, "top_reasons": [["net_expectancy", 12]]}],
            "candidate_census": {"ga_returned_candidates": 80, "walkforward_tested": 4, "walkforward_passed": 2},
            "walkforward_selection_cohort": {"trials": [{"positive_folds": 3, "reported_splits": 5}]},
            "final_oos": {"sentinel": "DO_NOT_READ_THIS"},
        }
    rows = [summarize(fixture(mode, pct), "synthetic_test") for mode in ("Risky", "PropFirm") for pct in (250, 500, 1000)]
    assert [round(row["stage1_pct"]) for row in rows[:3]] == [25, 50, 100]
    output = report(rows)
    assert "DO_NOT_READ_THIS" not in output
    assert "net_expectancy" in output
    assert "NO RUN EVIDENCE FOUND" in report([])
    print("PASS: pre-OOS 25/50/100, both modes, named reasons, no final OOS read")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="pre-oos-artifacts")
    p.add_argument("--output", default="pre-oos-report.md")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    root = Path(args.input_dir)
    paths = sorted(root.rglob("*_funnel.json")) if root.is_dir() else []
    rows = [summarize(json.loads(path.read_text(encoding="utf-8")), path) for path in paths]
    Path(args.output).write_text(report(rows), encoding="utf-8")
    print(f"Analyzed {len(rows)} real funnel artifact(s); report: {args.output}")


if __name__ == "__main__":
    main()
