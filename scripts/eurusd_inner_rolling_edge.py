#!/usr/bin/env python3
"""Independent inner chronological-fitness experiment, first 80% EURUSD only.

NOT a NeoEthos GA. NO final out-of-sample bar is read. Reuses signal families
from research-only edge lab; parameters and selection rules are frozen here.
Choose candidates using five chronological blocks inside the EARLY 75% of
selection. Only then inspect the late 25% internal forward, which was
already viewed in other experiments and therefore is NOT pristine.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import eurusd_dynamic_edge_lab as lab

MODE_PROFILES=(
    ("Risky",100.0,0.01,0,30.0,25.0),
    ("Risky",100.0,0.03,0,30.0,25.0),
    ("PropFirm",10000.0,0.005,8,30.0,8.0),
    ("PropFirm",10000.0,0.01,8,30.0,8.0),
)
STRICT_SPREAD=0.4
EXTRA_FEE=0.2


def evaluate(sig, arrays, start, stop, capital, risk, cap, rr, hold, lev=30.0, fee=EXTRA_FEE):
    return lab.metrics(
        lab.simulate(sig,*arrays,start,stop,capital,risk,cap,
                     rr,hold,STRICT_SPREAD,fee,lev),capital
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",default="EURUSD.tar")
    ap.add_argument("--out",default="eurusd-rolling-research-results")
    args=ap.parse_args()
    out=Path(args.out)
    out.mkdir(exist_ok=True,parents=True)
    data,total,selected=lab.load(args.archive)
    learn=int(selected*.75)
    cut=np.linspace(0,learn,6,dtype=np.int64)
    features=lab.features(data)
    arrays=tuple(np.ascontiguousarray(data[k].astype(np.float64))
                 for k in ("open","high","low","close"))+(
                 np.ascontiguousarray(data["timestamp"].astype(np.int64)),
                 np.ascontiguousarray(features["atr"].astype(np.float64)))
    family_counts=defaultdict(int)
    pass_counts=defaultdict(int)
    best={}
    all_five_pass=[]
    candidates_seen=0
    for name,family,sig,rr,hold in lab.generate(features):
        candidates_seen+=1
        family_counts[family]+=1
        s=np.ascontiguousarray(sig,dtype=np.int8)
        for mode,capital,risk,cap,lev,dd_ceiling in MODE_PROFILES:
            runs=[evaluate(s,arrays,int(a),int(b),capital,risk,cap,rr,hold,lev)
                  for a,b in zip(cut[:-1],cut[1:])]
            positives=sum(m["net_usd"]>0 for m in runs)
            used=sum(m["entries"]>=10 for m in runs)
            maxdd=max(m["max_drawdown_pct"] for m in runs)
            stopped=any(m["risk_stop_triggered"]!=0 for m in runs)
            total_entries=sum(m["entries"] for m in runs)
            sumret=sum(m["return_pct"] for m in runs)
            eligible=(positives>=4 and used>=4 and total_entries>=60
                      and maxdd<dd_ceiling and not stopped and sumret>0)
            key=(mode,risk,family)
            if eligible:
                pass_counts[(mode,risk,family)]+=1
            # Stable ranking: best robust candidates first, then combined
            # fold evidence; never use later internal-forward values here.
            rank=(int(eligible),positives,used,round(sumret-0.7*maxdd,4),
                  round(-maxdd,4),total_entries)
            if key not in best or rank>best[key]["rank"]:
                best[key]={
                    "mode":mode,"capital":capital,"risk":risk,
                    "family":family,"rule":name,"rr":rr,"hold":hold,
                    "signal":s.copy(),"cap":cap,
                    "qualified":bool(eligible),"rank":rank,
                    "train_folds":runs,
                    "positive_train_folds":positives,
                    "sum_train_return_pct":round(sumret,3)
                }
            if eligible:
                all_five_pass.append({
                    "mode":mode,"risk":risk,"family":family,"rule":name,
                    "positive_train_folds":positives,
                    "sum_train_return_pct":round(sumret,3),
                    "max_training_drawdown_pct":maxdd
                })
    rows=[]
    for key,p in sorted(best.items()):
        s=p["signal"];risk=p["risk"];capital=p["capital"];cap=p["cap"]
        # Existing internal-forward has already been used in exploratory
        # development. Treat this only as exploratory resampling, not fresh OOS.
        late=evaluate(s,arrays,learn,selected,capital,risk,cap,p["rr"],p["hold"])
        worse=evaluate(s,arrays,learn,selected,capital,risk,cap,p["rr"],p["hold"],fee=.4)
        late_cut=np.linspace(learn,selected,5,dtype=np.int64)
        quarters=[evaluate(s,arrays,int(a),int(b),capital,risk,cap,p["rr"],p["hold"])
                  for a,b in zip(late_cut[:-1],late_cut[1:])]
        rows.append({
            "mode":p["mode"],"capital":capital,"risk":risk,
            "family":p["family"],"rule":p["rule"],
            "qualified_on_inner_learning_folds":p["qualified"],
            "eligible_candidates_in_family":pass_counts.get((p["mode"],risk,p["family"]),0),
            "positive_learning_folds":p["positive_train_folds"],
            "sum_learning_return_pct":p["sum_train_return_pct"],
            "learning_folds":p["train_folds"],
            "exploratory_late_internal_forward":late,
            "late_internal_forward_cost_stress":worse,
            "positive_late_quarters":sum(x["net_usd"]>0 for x in quarters),
            "late_quarters":quarters,
        })
    qualified=sum(x["qualified_on_inner_learning_folds"] for x in rows)
    late_positive=sum(x["qualified_on_inner_learning_folds"] and
                      x["exploratory_late_internal_forward"]["net_usd"]>0 for x in rows)
    payload={
        "experiment":"inner-five-chronological-learning-folds-with-dynamic-lots",
        "full_rows":total,"selection_only_rows":selected,
        "final_oos_rows_not_read":total-selected,
        "learning_rows":learn,"learning_fold_boundaries":cut.tolist(),
        "candidate_rules":candidates_seen,
        "families":dict(family_counts),
        "cost_model":"0.4 pip assumed full spread +0.2 pip extra fee; stress .4 pip extra",
        "reference_leverage":30,
        "profiles":[{"mode":m,"capital":c,"risk_fraction":r}
                    for m,c,r,cap,lev,dd in MODE_PROFILES],
        "qualifying_candidates_by_family":[
            {"mode":m,"risk":r,"family":f,"eligible":n}
            for (m,r,f),n in sorted(pass_counts.items())
        ],
        "total_eligible_candidate_profile_pairs":len(all_five_pass),
        "eligible_candidates":all_five_pass,
        "qualified_family_profile_representatives":qualified,
        "late_positive_of_qualified_representatives":late_positive,
        "results":rows,
        "caveats":[
            "NOT NeoEthos genetic search; OHLCV without real broker Bid/Ask",
            "Late internal-forward has already been observed in previous rounds and is reused for diagnosis; never call it independent OOS",
            "Training fold selection creates selection bias: 4/5 positive does not prove forward edge",
            "Stops, fees, margin-call logic and live eligibility are simplified",
            "No entry on final reserved 20% of actual archive",
        ]
    }
    (out/"results.json").write_text(json.dumps(payload,indent=2)+"\n")
    lines=[
        "# EURUSD inner chronological folds: seek repeatable net edge", "",
        "Independent diagnostic, NOT NeoEthos GA, no final OOS read.",
        f"Real M5 bars: {total:,}; selection {selected:,}; early learning {learn:,}.",
        f"Scored {candidates_seen} rules across five early chronological folds",
        f"and {len(MODE_PROFILES)} mode-risk profiles: {candidates_seen*5*len(MODE_PROFILES):,} model evaluations.",
        f"Strict 0.4 pip spread and 0.2 pip extra roundtrip fee; dynamic lots; leverage 30:1.",
        "Eligibility requires 4/5 positive learning folds, at least four",
        "activity-bearing folds, drawdown below profile ceiling and no risk stop.",
        f"Eligible candidate/profile combinations: {len(all_five_pass)}.",
        f"Eligible family/profile representatives: {qualified}.",
        f"Of eligible representatives, late INTERNAL forward positive: {late_positive}.",
        "", "| Mode | Risk/trade | Family | Eligible rules | Learning positive folds | Late internal return | Late quarters |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for x in rows:
        y=x["exploratory_late_internal_forward"]
        lines.append(
            f'| {x["mode"]} | {x["risk"]:.2%} | {x["family"]} | '
            f'{x["eligible_candidates_in_family"]} | '
            f'{x["positive_learning_folds"]}/5 | {y["return_pct"]:+.2f}% | '
            f'{x["positive_late_quarters"]}/4 |'
        )
    lines += [
        "", "A large positive later result for a non-eligible training candidate",
        "is a regime-shift hypothesis, not evidence of robust selected edge.",
        "Repeated observation of this internal forward contaminates it as a",
        "future untouched holdout. Do not tune strategy on the final reserved 20%.",
        "Inspect JSON for every candidate that cleared the inner stability gate.",
    ]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
