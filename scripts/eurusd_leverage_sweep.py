#!/usr/bin/env python3
"""Selection-only leverage, risk and spread ablation using 6 strategy families.

NEVER reads final 20% of EURUSD bars (see edge_lab.load).
This is deliberately NOT the full NeoEthos GA, cTrader broker model or a
live-trading authorization. Leverages 50..1000 are hypothetical mechanical
margin ceilings, not claims of German/EU retail eligibility.

A family's signal is chosen on EARLY training data only with exactly one
reference financial contract (30:1, Risky 100 USD / 10%, PropFirm 10000 USD /
1%, 0.4 pip spread +0.2 pip extra fee); scenario sensitivity is then
assessed on later INTERNAL selection-window slices. Repeated scenario
inspection makes that internal window exploratory, NOT pristine OOS.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import eurusd_dynamic_edge_lab as lab

LEVERAGES = (20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
COSTS = ((0.0, 0.0), (0.2, 0.0), (0.4, 0.2), (0.4, 0.4))
PORTFOLIOS = (
    ("Risky", 20.0, (0.03, 0.10, 0.20), 0),
    ("Risky", 100.0, (0.01, 0.03, 0.10), 0),
    ("Risky", 1000.0, (0.01, 0.03, 0.10), 0),
    ("PropFirm", 10000.0, (0.0025, 0.005, 0.01), 8),
)


def self_test():
    n = 160
    t = np.arange(n,dtype=np.int64)*300000
    o = np.full(n,1.18,dtype=np.float64)
    h = np.full(n,1.181,dtype=np.float64)
    l = np.full(n,1.179,dtype=np.float64)
    c = np.full(n,1.18,dtype=np.float64)
    atr = np.full(n,10.,dtype=np.float64)
    s = np.zeros(n,dtype=np.int8)
    s[2]=1
    # Risk 2 USD, 15 pip stop, min lot .01, EURUSD $1180 min notional.
    low = lab.simulate(s,o,h,l,c,t,atr,0,n,20.,.1,0,1.5,12,.2,.2,30.)
    high = lab.simulate(s,o,h,l,c,t,atr,0,n,20.,.1,0,1.5,12,.2,.2,100.)
    assert low[2] == 0 and low[6] >= 1, ("30:1 should not admit $20 min lot",low)
    assert high[2] == 1 and high[12] >= .01, ("100:1 should admit $20 0.01 lot",high)
    # When risk budget rather than margin binds, leverage must not change PnL.
    a = lab.simulate(s,o,h,l,c,t,atr,0,n,1000.,.01,0,1.5,12,.2,.2,30.)
    b = lab.simulate(s,o,h,l,c,t,atr,0,n,1000.,.01,0,1.5,12,.2,.2,500.)
    assert abs(a[0]-b[0])<1e-9 and a[2] == b[2] == 1
    print("PASS: 30x rejects $20 minimum lot, 100x permits, no-edge-change when cap nonbinding")


def choose_families(f, data, learn):
    ts = np.ascontiguousarray(data["timestamp"].astype(np.int64))
    op = np.ascontiguousarray(data["open"].astype(np.float64))
    hi = np.ascontiguousarray(data["high"].astype(np.float64))
    lo = np.ascontiguousarray(data["low"].astype(np.float64))
    cl = np.ascontiguousarray(data["close"].astype(np.float64))
    atr = np.ascontiguousarray(f["atr"].astype(np.float64))
    arrays=(op,hi,lo,cl,ts,atr)
    train_end=int(learn*.25)
    best={}
    counts=defaultdict(int)
    for name,family,s,rr,hold in lab.generate(f):
        counts[family]+=1
        sig=np.ascontiguousarray(s,dtype=np.int8)
        for mode,capital,risk,cap in (
            ("Risky",100.,.10,0),("PropFirm",10000.,.01,8)
        ):
            raw=lab.simulate(sig,*arrays,0,train_end,capital,risk,cap,rr,hold,.4,.2,30.)
            m=lab.metrics(raw,capital)
            rank=lab.score(mode,m,train_end)
            key=(mode,family)
            fallback=m["return_pct"]-m["max_drawdown_pct"]*.02
            criterion=(1 if rank>-1e17 else 0,rank if rank>-1e17 else fallback)
            if key not in best or criterion>best[key]["criterion"]:
                best[key]={
                    "name":name,"family":family,"rr":rr,"hold":hold,
                    "sig":sig.copy(),"mode":mode,
                    "reference_train":m,"qualified_training":bool(rank>-1e17),
                    "criterion":criterion,
                }
    return arrays,best,dict(counts),train_end


def execute(arrays,p,offset,end,capital,risk,cap,spread,fee,lev):
    return lab.metrics(
        lab.simulate(p["sig"],*arrays,offset,end,capital,risk,cap,
                     p["rr"],p["hold"],spread,fee,lev),
        capital
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",default="EURUSD.tar")
    ap.add_argument("--out",default="eurusd-leverage-sweep-results")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    self_test()
    if args.self_test:
        return
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    data,total,allowed=lab.load(args.archive)
    learn=int(allowed*.75)
    f=lab.features(data)
    arrays,picks,counts,train_end=choose_families(f,data,learn)
    boundary=(learn,allowed)
    scenario=[]
    for (mode,capital,risk_grid,daily_cap) in PORTFOLIOS:
        for family in sorted(counts):
            cand=picks[(mode,family)]
            for risk in risk_grid:
                for lev in LEVERAGES:
                    for spread,fee in COSTS:
                        m=execute(arrays,cand,*boundary,capital,risk,daily_cap,spread,fee,lev)
                        scenario.append({
                            "mode":mode,"starting_usd":capital,"risk_fraction":risk,
                            "leverage_margin_cap":int(lev),"family":family,
                            "rule":cand["name"],"training_qualified_at_reference":cand["qualified_training"],
                            "spread_pips":spread,"extra_fee_pips":fee,
                            **m,
                        })
    # Four forward subperiods at only the strictest cost and a few predeclared
    # leverage/risk/capital profiles: report stability, not chance winner-picking.
    folds=[]
    cuts=np.linspace(learn,allowed,5).astype(int)
    for mode,capital,risk_grid,cap in PORTFOLIOS:
        risk = .10 if mode=="Risky" and capital in (20.,100.) else (.03 if mode=="Risky" else .01)
        for family in sorted(counts):
            cand=picks[(mode,family)]
            for lev in (30.,100.,500.):
                ms=[execute(arrays,cand,int(a),int(b),capital,risk,cap,.4,.2,lev)
                    for a,b in zip(cuts[:-1],cuts[1:])]
                folds.append({
                    "mode":mode,"capital":capital,"risk_fraction":risk,
                    "leverage":int(lev),"family":family,
                    "positive_quarters":sum(m["net_usd"]>0 for m in ms),
                    "quarters":ms,
                })
    chosen=[
        {"mode":x["mode"],"family":x["family"],"rule":x["name"],
         "qualified_training_at_reference":x["qualified_training"],
         "training":x["reference_train"]}
        for x in picks.values()
    ]
    output={
        "experiment":"hypothetical-margin-cap-SENSITIVITY-not-edge-certification",
        "source_data":"SHA-pinned EURUSD M5 OHLCV from neoethos-data snapshot-2026-08-09",
        "total_data_rows":total,
        "selection_only_rows":allowed,
        "never_scored_final_oos_rows":total-allowed,
        "family_search_end_row":train_end,
        "internal_forward_start_row":learn,
        "candidate_counts_by_family":counts,
        "families_selected_from_first_quarter_of_learning_pool_only":chosen,
        "leverage_sweep":list(LEVERAGES),
        "costs":list(COSTS),
        "total_scenarios":len(scenario),
        "scenario_grid":scenario,
        "forward_quarter_grid":folds,
        "limitations":[
            "This Python script is not the NeoEthos GA or a broker quote/tick replay",
            "Leverage is a simulated notional margin ceiling, not real available broker leverage",
            "Fills use OHLC bars and stop-first conservative interpretation",
            "No broker-authoritative Bid/Ask, commissions, swap, financing or slippage",
            "The internal validation was already observed in previous experiments; exploratory only",
            "No final 20% holdout bars touched",
            "No real money, deployment or regulatory entitlement",
        ],
    }
    (out/"results.json").write_text(json.dumps(output,indent=2)+"\n")
    # Anchored comparisons use SAME selected strategy and same assumptions
    # except leverage: causal impact of margin ceiling alone.
    def find(mode,capital,risk,lev,family,spread=.4,fee=.2):
        return next(x for x in scenario if x["mode"]==mode and
                    x["starting_usd"]==capital and x["risk_fraction"]==risk and
                    x["leverage_margin_cap"]==lev and x["family"]==family and
                    x["spread_pips"]==spread and x["extra_fee_pips"]==fee)
    lines=[
        "# EURUSD pre-OOS: leverage vs actual dynamic lot size", "",
        "**Research-only. Not a NeoEthos GA result or live broker return.**",
        f"Selection rows: {allowed:,}; final 20% untouched: {total-allowed:,}.",
        f"Trained {len(chosen)} family-mode references from earliest {train_end:,} rows.",
        f"Executed {len(scenario):,} explicit sensitivity scenarios over",
        f"{len(counts)} signal families and {len(folds)} four-quarter stability series.",
        "Risk-budget dynamic lots. Leverage controls only the maximum allowed",
        "notional exposure, not the gross per-pip return of a fixed position.",
        "Strict comparison cost: 0.4 pip spread plus 0.2 pip extra roundtrip fee.",
        "NO broker Bid/Ask, real commission, financing or actual margin call.",
        "", "## Matched contract experiments on the preselected families", "",
        "| Mode | Starting $ | Risk/trade | Family | Return 30x | Return 100x | Return 500x | Trades 30x/100x/500x |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for mode,capital,risk,cap in (
        ("Risky",20.,.10,0),("Risky",100.,.10,0),
        ("Risky",1000.,.03,0),("PropFirm",10000.,.01,8)
    ):
        for family in sorted(counts):
            r=[find(mode,capital,risk,lev,family) for lev in (30,100,500)]
            lines.append(
                f"| {mode} | {capital:.0f} | {risk:.2%} | {family} | "
                f"{r[0]['return_pct']:+.2f}% | {r[1]['return_pct']:+.2f}% | "
                f"{r[2]['return_pct']:+.2f}% | "
                f"{r[0]['entries']}/{r[1]['entries']}/{r[2]['entries']} |"
            )
    lines += [
        "", "## Strict-cost internal-forward stability (predeclared 4 subperiods)", "",
        "| Mode | Capital $ | Family | Positive quarters at 30x / 100x / 500x |",
        "|---|---:|---|---|",
    ]
    for mode,capital,risk,cap in (
        ("Risky",20.,.10,0),("Risky",100.,.10,0),
        ("Risky",1000.,.03,0),("PropFirm",10000.,.01,8)
    ):
        for family in sorted(counts):
            frows=[x for x in folds if x["mode"]==mode and
                   x["capital"]==capital and x["family"]==family]
            frows.sort(key=lambda x:x["leverage"])
            lines.append(
                f"| {mode} | {capital:.0f} | {family} | "+
                " / ".join(f"{x['positive_quarters']}/4" for x in frows)+" |"
            )
    lines += [
        "", "## Research cautions", "",
        "- A change in PnL with leverage can reflect more lots or fewer broker-minimum blocks, not improved per-unit edge.",
        "- Greater lot sizes compound losses too; drawdown and stop/ruin constraints still matter.",
        "- European retail major-FX CFD leverage is ordinarily limited to 30:1; 100-1000x columns are hypothetical.",
        "- Only TRAINING was used to choose the family's reference rule. Internal-forward is reused for sensitivity and not a virgin holdout.",
        "- Nothing evaluated, visited, or optimized on the reserved last 20% of market observations.",
        "- Download the JSON for all lots, size blocks, exposure caps, profitability and risk-stops.",
    ]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
