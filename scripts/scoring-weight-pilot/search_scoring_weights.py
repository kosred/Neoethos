#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Bayesian-style scoring-weight pilot over existing exit-ladder artifacts.

Searches weights w1..w5 in [0,1] (normalized to sum 1) maximizing
    S = w1*monthly_win_rate
      + w2*median_month_pnl_pct*100
      + w3*avg_win_over_avg_loss
      - w4*max(0, -(worst_month_pnl_pct*100) - 2.0)
      - w5*max_drawdown
over the recorded exit-policy variants of two neoethos exit-ladder artifacts,
under hard constraints worst_month_pnl_pct >= -0.03, monthly_win_rate >= 0.55
and net_profit > 0.

OFFLINE SCORING-KNOB SEARCH ONLY, NOT A GATE CHANGE: no Rust code is modified,
no cargo is run, no new build/backtest is produced. The artifacts are
ResearchOnly counterfactual replays (not promotion-eligible).

scikit-optimize is not installed in .git/dukascopy-venv, so the optimizer is a
Gaussian-process-free random + local search in the pure stdlib (deterministic
seed). Because S is linear in w and each per-dataset selection is an argmax
over a finite variant set, the joint objective is piecewise-linear convex, so
its maximum over the weight simplex sits at a vertex; the search evaluates all
five pure-knob vertices explicitly and verifies this.

Outputs next to this file: report.json (full results) and README.md.
"""

import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_REPORT = os.path.join(BASE, "report.json")
OUT_README = os.path.join(BASE, "README.md")

DATASETS = {
    "eurusd_m15_weekly": "C:/Users/konst/AppData/Local/neoethos/experiments/eurusd-m15-weekly-20261021-v7-exit-ladder3/exit-policy.json",
    "eurusd_m5_risky2": "C:/Users/konst/AppData/Local/neoethos/experiments/eurusd-m5-risky2-ladder3b/exit-policy.json",
}

STRICT_GATES = {"worst": -0.03, "mwr": 0.55, "net_min": 0.0}

KNOB_NAMES = [
    "w1_monthly_win_rate",
    "w2_median_month_pnl_pct",
    "w3_avg_win_over_avg_loss",
    "w4_worst_month_penalty",
    "w5_max_drawdown",
]

RANDOM_SAMPLES = 3000
REFINE_STARTS = 40
REFINE_ITERS = 60
SEED = 42
EPS = 1e-12


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows():
    rows = {}
    meta = {}
    for ds, path in DATASETS.items():
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        meta[ds] = {
            "path": path,
            "sha256": sha256_file(path),
            "schema": j.get("schema"),
            "artifact_class": j.get("artifact_class"),
            "promotion_eligibility": j.get("promotion_eligibility"),
            "authorization_issued": j.get("authorization_issued"),
            "warning": j.get("warning"),
        }
        pool = []
        ev = j["exit_policy_variants"]
        for c in ev["candidates"]:
            sid = c["candidate"]["strategy_id"]
            for v in c["variants"]:
                names = v["metrics"]["names"]
                values = v["metrics"]["values"]
                m = dict(zip(names, values))
                mon = v["ledger"]["monthly"]
                wr = float(m["win_rate"])
                pf = float(m["profit_factor"])
                aw_al = (pf * (1.0 - wr) / wr) if wr > EPS else float("inf")
                pool.append({
                    "dataset": ds,
                    "strategy_id": sid,
                    "label": v["label"],
                    "net_profit": float(m["net_profit"]),
                    "sharpe": float(m["sharpe"]),
                    "max_drawdown": float(m["max_drawdown"]),
                    "win_rate": wr,
                    "profit_factor": pf,
                    "expectancy": float(m["expectancy"]),
                    "monthly_win_rate": float(mon["monthly_win_rate"]),
                    "worst_month_pnl_pct": float(mon["worst_month_pnl_pct"]),
                    "median_month_pnl_pct": float(mon["median_month_pnl_pct"]),
                    "months_observed": int(mon["months_observed"]),
                    "positive_months": int(mon["positive_months"]),
                    "negative_months": int(mon["negative_months"]),
                    "aw_al": aw_al,
                })
        rows[ds] = pool
    return rows, meta


def feasible(row, gates):
    if row["monthly_win_rate"] < gates["mwr"]:
        return False
    if row["net_profit"] <= gates["net_min"]:
        return False
    if gates.get("worst") is not None and row["worst_month_pnl_pct"] < gates["worst"]:
        return False
    return True


def score(w, row):
    mon_pen = max(0.0, -(row["worst_month_pnl_pct"] * 100.0) - 2.0)
    return (w[0] * row["monthly_win_rate"]
            + w[1] * row["median_month_pnl_pct"] * 100.0
            + w[2] * row["aw_al"]
            - w[3] * mon_pen
            - w[4] * row["max_drawdown"])


def normalize(raw):
    s = float(sum(raw))
    if s <= 0.0:
        return None
    return [x / s for x in raw]


def select_best(w, pool, gates):
    best = None
    best_score = None
    tied = []
    for row in pool:
        if not feasible(row, gates):
            continue
        sc = score(w, row)
        if best_score is None or sc > best_score + EPS:
            best_score = sc
            best = row
            tied = [row["label"]]
        elif abs(sc - best_score) <= EPS:
            tied.append(row["label"])
    if best is None:
        return None
    return {"score": best_score, "row": best, "tied_labels": sorted(set(tied))}


def objective(w, rows, gates_by_ds):
    total = 0.0
    selections = {}
    infeasible = {}
    for ds, pool in rows.items():
        sel = select_best(w, pool, gates_by_ds[ds])
        if sel is None:
            infeasible[ds] = True
            selections[ds] = None
        else:
            total += sel["score"]
            selections[ds] = sel
    return total, selections, infeasible


def entropy(w):
    e = 0.0
    for x in w:
        if x > EPS:
            e -= x * math.log(x)
    return e


def selection_report(sel, reason="no variant passes this scenario's gates"):
    if sel is None:
        return {"selected": None, "reason": reason}
    r = sel["row"]
    return {
        "selected": True,
        "strategy_id": r["strategy_id"],
        "label": r["label"],
        "score": round(sel["score"], 9),
        "monthly_win_rate": r["monthly_win_rate"],
        "worst_month_pnl_pct": r["worst_month_pnl_pct"],
        "median_month_pnl_pct": r["median_month_pnl_pct"],
        "aw_al": round(r["aw_al"], 6) if r["aw_al"] != float("inf") else None,
        "net_profit": round(r["net_profit"], 4),
        "max_drawdown": r["max_drawdown"],
        "sharpe": r["sharpe"],
        "score_inputs": {
            "monthly_win_rate": r["monthly_win_rate"],
            "median_month_pnl_pct_x100": r["median_month_pnl_pct"] * 100.0,
            "avg_win_over_avg_loss": r["aw_al"],
            "worst_month_penalty_gt_2pct": max(0.0, -(r["worst_month_pnl_pct"] * 100.0) - 2.0),
            "max_drawdown": r["max_drawdown"],
        },
        "tied_labels": sel["tied_labels"],
    }


def run_search(rows, gates_by_ds, seed=SEED,
               n_random=RANDOM_SAMPLES, n_starts=REFINE_STARTS, n_iters=REFINE_ITERS):
    rng = random.Random(seed)
    vertices = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    seeds_raw = [list(v) for v in vertices] + [[0.2] * 5]
    seeds_raw += [[rng.random() for _ in range(5)] for _ in range(n_random)]

    evaluated = []

    def evaluate(raw):
        norm = normalize(raw)
        if norm is None:
            return None
        obj, sels, inf = objective(norm, rows, gates_by_ds)
        return (tuple(raw), tuple(norm), obj, sels, inf)

    for raw in seeds_raw:
        r = evaluate(raw)
        if r is not None:
            evaluated.append(r)

    starts = sorted(evaluated, key=lambda r: -r[2])[:n_starts]
    for base in starts:
        cur_raw = list(base[0])
        cur = base
        for _ in range(n_iters):
            prop = list(cur_raw)
            mode = rng.random()
            if mode < 0.45:
                j = rng.randrange(5)
                prop[j] = min(1.0, max(0.0, prop[j] * rng.uniform(0.6, 1.5)))
            elif mode < 0.7:
                k = rng.randrange(5)
                t = rng.uniform(0.0, 0.6)
                for j in range(5):
                    prop[j] = (1.0 - t) * prop[j] + t * (1.0 if j == k else 0.0)
            else:
                for j in range(5):
                    prop[j] = min(1.0, max(0.0, prop[j] + rng.uniform(-0.2, 0.2)))
            r = evaluate(prop)
            if r is not None:
                evaluated.append(r)
                if r[2] > cur[2] + EPS:
                    cur_raw = list(prop)
                    cur = r
        evaluated.append(cur)

    best_of = {}
    for r in evaluated:
        key = tuple(round(x, 5) for x in r[0])
        if key not in best_of or r[2] > best_of[key][2]:
            best_of[key] = r
    uniq = list(best_of.values())
    uniq.sort(key=lambda r: (-r[2], -entropy(r[1])))

    vertex_recs = [r for r in uniq if set(r[0]) <= {0.0, 1.0}]
    best_vertex = max(vertex_recs, key=lambda r: r[2]) if vertex_recs else None
    return uniq, best_vertex


def group_top(uniq, rows, gates_by_ds, top_n=5, max_groups=10):
    groups = {}
    for raw, norm, obj, sels, inf in uniq:
        key = tuple(
            (sels[ds]["row"]["strategy_id"], sels[ds]["row"]["label"])
            if sels[ds] is not None else None
            for ds in rows
        )
        g = groups.setdefault(key, {"objective": obj, "entries": []})
        if obj > g["objective"]:
            g["objective"] = obj
        g["entries"].append((raw, norm, obj))
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["objective"])

    def make_row(raw, norm, obj, entry_kind):
        sel_out = {}
        for ds in rows:
            sel = select_best(norm, rows[ds], gates_by_ds[ds])
            sel_out[ds] = selection_report(sel)
        return {
            "rank": None,
            "entry_kind": entry_kind,
            "weights_raw": [round(x, 6) for x in raw],
            "weights_normalized": [round(x, 6) for x in norm],
            "objective": round(obj, 9),
            "selections": sel_out,
        }

    groups_all = []
    for rank, (key, g) in enumerate(ranked[:max_groups], 1):
        best_entry = max(g["entries"], key=lambda e: (e[2], entropy(e[1])))
        row = make_row(best_entry[0], best_entry[1], best_entry[2],
                       "distinct selection regime")
        row["rank"] = rank
        groups_all.append(row)

    out = [dict(r) for r in groups_all[:top_n]]
    if len(out) < top_n:
        covered = {k for k, _ in ranked}
        used_vecs = [r["weights_normalized"] for r in out]
        for raw, norm, obj, sels, inf in uniq:
            if len(out) >= top_n:
                break
            key = tuple(
                (sels[ds]["row"]["strategy_id"], sels[ds]["row"]["label"])
                if sels[ds] is not None else None
                for ds in rows
            )
            if key not in covered:
                continue
            if not all(max(abs(a - b) for a, b in zip(norm, u)) > 0.05
                       for u in used_vecs):
                continue
            src_rank = next((i + 1 for i, (k, _) in enumerate(ranked) if k == key), None)
            row = make_row(
                raw, norm, obj,
                "additional near-optimal vector (same selected variants as rank " + str(src_rank) + ")")
            row["rank"] = len(out) + 1
            out.append(row)
            used_vecs.append(norm)
    return out, groups_all


def vertex_sensitivity(rows, gates_by_ds):
    out = []
    for k, name in enumerate(KNOB_NAMES):
        w = [0.0] * 5
        w[k] = 1.0
        obj, sels, inf = objective(w, rows, gates_by_ds)
        out.append({
            "knob": name,
            "weights_normalized": w,
            "objective": round(obj, 9),
            "selections": {ds: selection_report(sels[ds]) for ds in rows},
        })
    return out


def data_summary(rows):
    summ = {}
    for ds, pool in rows.items():
        by_cand = {}
        for r in pool:
            by_cand.setdefault(r["strategy_id"], []).append(r)
        cands = []
        for sid, pl in by_cand.items():
            pass_mwr_net = [r for r in pl
                            if r["monthly_win_rate"] >= STRICT_GATES["mwr"]
                            and r["net_profit"] > STRICT_GATES["net_min"]]
            cands.append({
                "strategy_id": sid,
                "variants": len(pl),
                "strict_feasible": sum(1 for r in pl if feasible(r, STRICT_GATES)),
                "pass_mwr_and_net": len(pass_mwr_net),
                "best_worst_month_among_mwr_net_passers": max(
                    (r["worst_month_pnl_pct"] for r in pass_mwr_net), default=None),
                "max_monthly_win_rate": max(r["monthly_win_rate"] for r in pl),
            })
        summ[ds] = {"candidates": cands}
    return summ


SCENARIOS = [
    {
        "id": "strict_gates",
        "description": (
            "Primary run: exactly the requested hard gates on both datasets "
            "(worst_month_pnl_pct >= -0.03 AND monthly_win_rate >= 0.55 AND "
            "net_profit > 0). eurusd_m5_risky2 has zero variants passing these "
            "gates, so it contributes nothing to the objective and the ranking "
            "is driven by the m15 feasible pool."
        ),
        "gates": {ds: dict(STRICT_GATES) for ds in DATASETS},
    },
    {
        "id": "m5_worst_gate_disabled_sensitivity",
        "description": (
            "Documented sensitivity run (NOT a gate change): m15 keeps the "
            "strict gates; m5 keeps monthly_win_rate >= 0.55 and "
            "net_profit > 0 but its worst-month hard gate is disabled because "
            "-3% is unreachable there (best achievable worst month among m5 "
            "mwr/net-passing variants is about -26.9%). The w4 term in S still "
            "penalizes worst-month depth."
        ),
        "gates": {
            "eurusd_m15_weekly": dict(STRICT_GATES),
            "eurusd_m5_risky2": {"worst": None, "mwr": 0.55, "net_min": 0.0},
        },
    },
]


def main():
    rows, meta = load_rows()
    report = {
        "title": "scoring-weight-pilot: Bayesian-style scoring-weight search over existing exit-ladder artifacts",
        "nature": "OFFLINE SCORING-KNOB SEARCH, NOT A GATE CHANGE. No Rust code modified, no cargo run, no new builds.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/scoring-weight-pilot/search_scoring_weights.py",
        "score_formula": (
            "S = w1*monthly_win_rate + w2*median_month_pnl_pct*100 "
            "+ w3*avg_win_over_avg_loss "
            "- w4*max(0, -(worst_month_pnl_pct*100) - 2.0) - w5*max_drawdown; "
            "weights w1..w5 in [0,1], normalized to sum 1."
        ),
        "avg_win_over_avg_loss_derivation": (
            "The artifacts store no avg-win/avg-loss field; per variant it is "
            "derived as profit_factor * (1 - win_rate) / win_rate."
        ),
        "optimizer": {
            "method": "Gaussian-process-free random + local search (pure stdlib)",
            "skopt_available_in_dukascopy_venv": False,
            "note": "scikit-optimize is not installed in .git/dukascopy-venv, so the sanctioned fallback random+local search is used.",
            "random_samples": RANDOM_SAMPLES,
            "refined_starts": REFINE_STARTS,
            "refine_iterations_per_start": REFINE_ITERS,
            "seed": SEED,
            "theory_note": "S is linear in w and per-dataset selection is argmax over a finite variant set, so the joint objective is piecewise-linear convex and its simplex maximum sits at a pure-knob vertex; all five vertices are evaluated explicitly.",
        },
        "datasets": meta,
        "data_summary": data_summary(rows),
        "scenarios": [],
        "headline_findings": [],
    }

    scenario_results = []
    for sc in SCENARIOS:
        gates_by_ds = sc["gates"]
        feasible_counts = {
            ds: sum(1 for r in rows[ds] if feasible(r, gates_by_ds[ds]))
            for ds in DATASETS
        }
        uniq, best_vertex = run_search(rows, gates_by_ds)
        top, groups_all = group_top(uniq, rows, gates_by_ds, top_n=5, max_groups=10)
        for row in top + groups_all:
            for ds, s in row["selections"].items():
                if s["selected"] and s["worst_month_pnl_pct"] < -0.3:
                    s["caution"] = (
                        "blowup-profile replay: worst month "
                        + str(round(s["worst_month_pnl_pct"] * 100, 2))
                        + "% and max drawdown "
                        + str(round(s["max_drawdown"] * 100, 2))
                        + "%; selectable only because this scenario disables the "
                        "worst-month hard gate. The strict -3% gate is what "
                        "excludes such profiles."
                    )
        vertex = vertex_sensitivity(rows, gates_by_ds)
        entry = {
            "id": sc["id"],
            "description": sc["description"],
            "gates": gates_by_ds,
            "feasible_variant_counts": feasible_counts,
            "objective_semantics": (
                "sum over datasets of S of the best feasible variant; a dataset "
                "with no feasible variant contributes 0 and is flagged."
            ),
            "search_stats": {
                "unique_weight_vectors_evaluated": len(uniq),
                "best_objective_found": round(uniq[0][2], 9) if uniq else None,
                "best_pure_knob_vertex_objective": round(best_vertex[2], 9) if best_vertex else None,
                "vertex_optimality_check": (
                    (best_vertex is not None)
                    and (uniq[0][2] <= best_vertex[2] + 1e-6)
                ),
            },
            "distinct_selection_regimes_found": len(groups_all),
            "top_weight_vectors": top,
            "top_weight_vectors_extended": groups_all,
            "single_knob_sensitivity": vertex,
        }
        scenario_results.append(entry)
        report["scenarios"].append(entry)

    m5s = report["data_summary"]["eurusd_m5_risky2"]["candidates"]
    m15s = report["data_summary"]["eurusd_m15_weekly"]["candidates"]
    m5_best_worst = [c["best_worst_month_among_mwr_net_passers"] for c in m5s]
    m5_best_worst = [x for x in m5_best_worst if x is not None]
    m15_best_worst = [c["best_worst_month_among_mwr_net_passers"] for c in m15s]
    m15_best_worst = [x for x in m15_best_worst if x is not None]

    strict = scenario_results[0]
    strict_top = strict["top_weight_vectors"][0]
    m15_sel = strict_top["selections"]["eurusd_m15_weekly"]
    relaxed = scenario_results[1]
    relaxed_top = relaxed["top_weight_vectors"][0]
    m15r = relaxed_top["selections"]["eurusd_m15_weekly"]
    m5r = relaxed_top["selections"]["eurusd_m5_risky2"]

    def fmt_w(row):
        return "(" + ", ".join(str(x) for x in row["weights_normalized"]) + ")"

    findings = [
        "STRICT GATES: eurusd_m5_risky2 has ZERO feasible variants under the requested hard gates. Among variants passing monthly_win_rate>=0.55 and net_profit>0, the best worst month is "
        + (", ".join(str(round(x * 100, 2)) + "%" for x in sorted(m5_best_worst, reverse=True)) if m5_best_worst else "none")
        + ". No scoring-weight vector can make m5 satisfy a -3% worst-month gate; the gate itself is the binding limit, not the weights.",
        "STRICT GATES: the m15 feasible pool is "
        + str(strict["feasible_variant_counts"]["eurusd_m15_weekly"])
        + " variants, all from candidate gene_782139_19 (baseline candidate gene_953632_18 near-misses with best worst month "
        + (str(round(m15_best_worst[0] * 100, 2)) + "%" if m15_best_worst else "n/a") + ").",
        "Strict optimum: w=" + fmt_w(strict_top)
        + " selects m15 " + m15_sel["label"] + " (strategy " + m15_sel["strategy_id"]
        + "; S=" + str(round(m15_sel["score"], 4))
        + "; mwr=" + str(m15_sel["monthly_win_rate"])
        + ", worst=" + str(round(m15_sel["worst_month_pnl_pct"] * 100, 2)) + "%"
        + ", aw/al=" + str(m15_sel["aw_al"])
        + ", net=" + str(m15_sel["net_profit"]) + ").",
        "Sensitivity (m5 worst-month gate disabled, w4 penalty still active): joint optimum w="
        + fmt_w(relaxed_top)
        + " selects m15 " + m15r["label"] + " (aw/al=" + str(m15r["aw_al"])
        + ", net=" + str(m15r["net_profit"]) + ") AND m5 " + m5r["label"]
        + " / " + m5r["strategy_id"] + " (aw/al=" + str(m5r["aw_al"])
        + ", mwr=" + str(m5r["monthly_win_rate"])
        + ", worst=" + str(round(m5r["worst_month_pnl_pct"] * 100, 2)) + "%"
        + ", net=" + str(m5r["net_profit"]) + ").",
        "Scale note: median_month_pnl_pct*100 and aw/al dominate S magnitude; max_drawdown is stored as a fraction (0..1), so w5 needs ~100x relative weight to matter. The optimizer therefore concentrates weight on aw/al (and median-month pnl in the relaxed run).",
    ]
    m5sen = relaxed_top["selections"]["eurusd_m5_risky2"]
    if m5sen["selected"] and m5sen["worst_month_pnl_pct"] < -0.3:
        findings.append(
            "Sensitivity caution: with the m5 worst-month gate disabled, the formula selects a blowup-profile replay (m5 " + m5sen["label"]
            + ": worst month " + str(round(m5sen["worst_month_pnl_pct"] * 100, 2))
            + "%, max drawdown " + str(round(m5sen["max_drawdown"] * 100, 2))
            + "%) - evidence that the -3% worst-month gate, not the scoring weights, is what keeps such profiles out."
        )
    findings.append(
        "Offline only: both artifacts are ResearchOnly counterfactual replays (authorization_issued=false); this pilot does not change any gate, policy, cost or sizing."
    )
    report["headline_findings"] = findings

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    readme = build_readme(report)
    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write(readme)

    def brief(rows5):
        out = []
        for t in rows5:
            out.append({
                "rank": t["rank"],
                "kind": t["entry_kind"],
                "w": t["weights_normalized"],
                "obj": t["objective"],
                "sel": {
                    ds: (s["label"] + " / " + s["strategy_id"]
                         if s["selected"] else None)
                    for ds, s in t["selections"].items()
                },
            })
        return out

    summary = {
        "strict_gates": {
            "feasible_counts": strict["feasible_variant_counts"],
            "regimes_found": strict["distinct_selection_regimes_found"],
            "top5": brief(strict["top_weight_vectors"]),
            "best": strict_top["selections"],
        },
        "sensitivity": {
            "feasible_counts": relaxed["feasible_variant_counts"],
            "regimes_found": relaxed["distinct_selection_regimes_found"],
            "top5": brief(relaxed["top_weight_vectors"]),
            "best": relaxed_top["selections"],
        },
        "headline_findings": findings,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return report


def build_readme(report):
    strict = report["scenarios"][0]
    relaxed = report["scenarios"][1]
    st = strict["top_weight_vectors"][0]
    rt = relaxed["top_weight_vectors"][0]
    m15s = st["selections"]["eurusd_m15_weekly"]
    m15r = rt["selections"]["eurusd_m15_weekly"]
    m5r = rt["selections"]["eurusd_m5_risky2"]

    def wline(entry):
        return "w = (" + ", ".join(str(x) for x in entry["weights_normalized"]) + ")"

    def selline(s):
        if s["selected"] is None:
            return "- " + s["reason"]
        return ("- " + s["label"] + " (" + s["strategy_id"] + "): S="
                + str(round(s["score"], 4)) + ", monthly_win_rate="
                + str(s["monthly_win_rate"]) + ", worst_month="
                + str(round(s["worst_month_pnl_pct"] * 100, 2)) + "%, aw/al="
                + str(s["aw_al"]) + ", net_profit=" + str(s["net_profit"]))

    lines = []
    lines.append("# scoring-weight-pilot")
    lines.append("")
    lines.append("Offline Bayesian-style scoring-weight search over existing exit-ladder artifacts.")
    lines.append("**This is a scoring-knob search, NOT a gate change.** No Rust code was modified,")
    lines.append("no cargo was run, no new backtest/build was produced, and nothing in this pilot")
    lines.append("changes production policy, gates, costs or sizing. Both input artifacts are")
    lines.append("ResearchOnly saved-candidate counterfactual replays (authorization_issued: false,")
    lines.append("promotion-ineligible); their own warnings state variant settings are diagnostic only.")
    lines.append("")
    lines.append("## Inputs")
    for ds, m in report["datasets"].items():
        lines.append("- " + m["path"])
    lines.append("")
    lines.append("Variant pool: exit_policy_variants.candidates[*].variants[*] (label + metrics +")
    lines.append("ledger.monthly per variant; two strategy candidates per dataset, 366 exit-policy")
    lines.append("variants each).")
    lines.append("")
    lines.append("## Score")
    lines.append("")
    lines.append("    S = w1*monthly_win_rate + w2*median_month_pnl_pct*100 + w3*avg_win_over_avg_loss")
    lines.append("      - w4*max(0, -(worst_month_pnl_pct*100) - 2.0) - w5*max_drawdown")
    lines.append("")
    lines.append("weights w1..w5 in [0,1], normalized to sum 1. The artifacts store no avg-win/avg-loss")
    lines.append("field, so avg_win_over_avg_loss is derived per variant as")
    lines.append("profit_factor * (1 - win_rate) / win_rate.")
    lines.append("")
    lines.append("## Hard constraints (primary run)")
    lines.append("")
    lines.append("    worst_month_pnl_pct >= -0.03")
    lines.append("    monthly_win_rate  >=  0.55")
    lines.append("    net_profit        >   0")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("scikit-optimize is not installed in .git/dukascopy-venv, so the search uses a")
    lines.append("Gaussian-process-free random + local search (pure Python stdlib, deterministic seed):")
    lines.append("3000 normalized random weight vectors plus the five pure-knob vertices, refined by")
    lines.append("local hill-climbing from the 40 best starts. Because S is linear in w and selection")
    lines.append("is an argmax over a finite variant set, the joint objective is piecewise-linear")
    lines.append("convex, so its maximum over the weight simplex sits at a vertex; the search")
    lines.append("evaluates all five vertices and verifies this.")
    lines.append("")
    lines.append("## Key results (full detail in report.json)")
    lines.append("")
    lines.append("Strict gates, top vector " + wline(st) + ":")
    lines.append("- eurusd_m15_weekly: " + selline(m15s))
    lines.append("- eurusd_m5_risky2: " + selline(st["selections"]["eurusd_m5_risky2"]))
    lines.append("")
    lines.append("Sensitivity (m5 worst-month gate disabled, w4 penalty still active), top vector "
                + wline(rt) + ":")
    lines.append("- eurusd_m15_weekly: " + selline(m15r))
    lines.append("- eurusd_m5_risky2: " + selline(m5r))
    lines.append("")
    lines.append("Headline findings:")
    for h in report["headline_findings"]:
        lines.append("- " + h)
    lines.append("")
    lines.append("## Re-run")
    lines.append("")
    lines.append("    .git/dukascopy-venv/Scripts/python.exe scripts/scoring-weight-pilot/search_scoring_weights.py")
    lines.append("")
    lines.append("No extra dependencies. Outputs: report.json and this README next to the script.")
    lines.append("")
    lines.append("## Provenance & ledger")
    lines.append("")
    lines.append("Results were appended to issue neo-671 in the Beads ledger as offline pilot")
    lines.append("evidence. No gate was changed.")
    lines.append("")
    return chr(10).join(lines)


if __name__ == "__main__":
    main()
