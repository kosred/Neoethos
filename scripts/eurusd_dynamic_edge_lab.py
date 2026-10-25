#!/usr/bin/env python3
"""Selection-only exploratory edge audit. NOT NeoEthos GA; NO final OOS read.

Pinned EURUSD M5 OHLCV. Discover within earliest 75% of the first 80% of
archive bars; 25/50/100% early-window arms, identical 25%-of-selection
internal validation for all arms. Signal is calculated at CLOSE of bar t,
position opens at OPEN of t+1. SL/TP intrabar; SL wins simultaneous touches.
Dynamic lot sizing from current USD equity, true stop distance, EURUSD
pip-value, 0.01 lot broker grid, 30x notional cap, 100 lot maximum.
Costs = full-roundtrip spread plus explicit extra roundtrip commission-pip
scenario. Broker spread curves, tick fills and actual commission unavailable.
No signals/trades/indicators computed on the held-out final 20%.
"""
import argparse
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import vortex as vx
from numba import njit

ARCHIVE_SHA = "042aa66fde29e697bd30d93d85c4762076c06e292097e42680ed0067b06ca061"
M5_SHA = "4b12214461f8abe09c93f656192fb6c4b71d5c9ca2ea03e37403c52ad5f3bcd0"
PIP = 10000.0
PIP_VALUE = 10.0  # USD per EURUSD pip / 1 standard lot, USD research account.
RR_PROFILES = ((1.5, 12), (2.5, 36))
MODES = (("Risky", 100.0, 0.10, 0), ("PropFirm", 10000.0, 0.01, 8))
ARM_FRACTIONS = (0.25, 0.5, 1.0)


def load(archive):
    h = hashlib.sha256()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == ARCHIVE_SHA
    with tarfile.open(archive, "r") as tar:
        matches = [x for x in tar if x.isfile() and x.name == "symbol=EURUSD/timeframe=M5/data.vortex"]
        assert len(matches) == 1
        assert 0 < matches[0].size < 100_000_000
        dest = Path("/tmp/edge-lab-EURUSD-M5.vortex")
        with tar.extractfile(matches[0]) as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == M5_SHA
    data = vx.open(str(dest)).to_arrow()
    if hasattr(data, "read_all"):
        data = data.read_all()
    total = data.num_rows
    end = int(0.8 * total)
    # Hard bound: no values from terminal 20% enter any numpy/feature array.
    series = {}
    for name in ("timestamp", "open", "high", "low", "close"):
        series[name] = np.asarray(
            data.column(name).slice(0, end).to_numpy(zero_copy_only=False))
    del data
    return series, total, end


@njit(cache=True)
def simulate(sig, op, hi, lo, cl, timestamp, atr_pips, start, stop,
             initial_usd, risk_fraction, max_entries_day, rr, hold_bars,
             spread_pips, extra_fee_pips, max_notional_multiple=30.0):
    equity = initial_usd
    peak = equity
    max_dd = 0.0
    entries = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    blocked_min = 0
    bound_notional = 0
    bound_risk = 0
    total_lots = 0.0
    total_cost_usd = 0.0
    pos = 0
    price_in = 0.0
    stop_px = 0.0
    target_px = 0.0
    lot = 0.0
    elapsed = 0
    prev_day = -1
    entries_today = 0
    day_start_equity = equity
    active_day = 0
    breach = 0
    max_lot = 0.0
    min_lot = 100.0
    gap_exits = 0
    for i in range(max(start + 1, 1), min(stop, len(op))):
        day = timestamp[i] // 86400000
        if day != prev_day:
            prev_day = day
            entries_today = 0
            day_start_equity = equity
        # Contract conservative: invalid broker OHLC bar cannot trade.
        valid = (np.isfinite(op[i]) and op[i] > 0.0 and
                 np.isfinite(hi[i]) and hi[i] >= op[i] and
                 np.isfinite(lo[i]) and lo[i] <= op[i] and
                 np.isfinite(cl[i]) and lo[i] <= cl[i] <= hi[i] and
                 hi[i] >= lo[i])
        contiguous = timestamp[i] - timestamp[i - 1] == 300000
        if pos != 0:
            elapsed += 1
            # Stop-first conservative resolution of same-bar TP and SL.
            exit_px = 0.0
            if not valid or not contiguous:
                if np.isfinite(op[i]) and op[i] > 0:
                    exit_px = op[i]
                else:
                    exit_px = price_in
                gap_exits += 1
            elif pos > 0:
                if op[i] <= stop_px:
                    exit_px = op[i]
                elif lo[i] <= stop_px:
                    exit_px = stop_px
                elif op[i] >= target_px:
                    exit_px = op[i]
                elif hi[i] >= target_px:
                    exit_px = target_px
            else:
                if op[i] >= stop_px:
                    exit_px = op[i]
                elif hi[i] >= stop_px:
                    exit_px = stop_px
                elif op[i] <= target_px:
                    exit_px = op[i]
                elif lo[i] <= target_px:
                    exit_px = target_px
            if exit_px == 0 and elapsed >= hold_bars:
                exit_px = cl[i]
            if exit_px != 0:
                gross = (exit_px - price_in) * pos * PIP * PIP_VALUE * lot
                fee = (spread_pips + extra_fee_pips) * PIP_VALUE * lot
                pnl = gross - fee
                equity += pnl
                total_cost_usd += fee
                if pnl > 0:
                    wins += 1
                    gross_win += pnl
                else:
                    gross_loss -= pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / max(peak, 1e-9)
                if dd > max_dd:
                    max_dd = dd
                pos = 0
                if equity <= initial_usd * 0.05:
                    breach = 1
                    break
                if max_entries_day > 0 and (
                    equity <= initial_usd * 0.90 or
                    equity <= day_start_equity - 0.05 * initial_usd
                ):
                    breach = 2
                    break
                # No immediate re-entry on the same closing bar.
                continue
        if pos != 0:
            continue
        if not valid or not contiguous:
            continue
        if max_entries_day > 0 and entries_today >= max_entries_day:
            continue
        # at OPEN of i use only signal and ATR from completed bar i-1.
        direction = int(sig[i-1])
        if direction == 0:
            continue
        atr = atr_pips[i-1]
        if not np.isfinite(atr) or atr <= 0:
            continue
        sl = min(35.0, max(4.0, 1.5 * atr))
        if sl <= 0:
            continue
        raw = equity * risk_fraction / (sl * PIP_VALUE)
        notional_max = (equity * max_notional_multiple) / (op[i] * 100000.0)
        if raw > notional_max:
            bound_notional += 1
        else:
            bound_risk += 1
        capped = min(raw, notional_max, 100.0)
        lots = math.floor((capped + 1e-11) / 0.01) * 0.01
        if lots < 0.01:
            blocked_min += 1
            continue
        pos = direction
        lot = lots
        total_lots += lot
        if lot > max_lot:
            max_lot = lot
        if lot < min_lot:
            min_lot = lot
        price_in = op[i]
        stop_px = price_in - direction * sl / PIP
        target_px = price_in + direction * sl * rr / PIP
        elapsed = 0
        entries += 1
        entries_today += 1
    # End-of-period positions intentionally unscored: do not look beyond boundary.
    if entries > 0 and pos != 0:
        gross = (cl[stop-1] - price_in) * pos * PIP * PIP_VALUE * lot
        fee = (spread_pips + extra_fee_pips) * PIP_VALUE * lot
        pnl = gross - fee
        equity += pnl
        total_cost_usd += fee
        if pnl > 0:
            wins += 1
            gross_win += pnl
        else:
            gross_loss -= pnl
        if equity > peak:
            peak = equity
        max_dd = max(max_dd, (peak-equity)/max(peak,1e-9))
    return (equity, max_dd, entries, wins, gross_win, gross_loss,
            blocked_min, bound_notional, bound_risk, total_lots,
            total_cost_usd, breach, max_lot, min_lot if entries else 0.0, gap_exits)


def metrics(vals, initial):
    (equity, dd, n, w, gp, gl, blocked, notional, risk_bound,
     lots, costs, breach, lotmax, lotmin, gaps) = vals
    return {
        "initial_usd": initial, "final_usd": round(float(equity), 2),
        "net_usd": round(float(equity-initial), 2),
        "return_pct": round(float((equity/initial-1)*100), 3),
        "max_drawdown_pct": round(float(dd*100), 3),
        "entries": int(n), "wins": int(w),
        "win_rate_pct": round(100*int(w)/int(n), 3) if n else None,
        "profit_factor": round(float(gp/gl), 3) if gl > 0 else None,
        "mean_lots": round(float(lots)/int(n), 4) if n else 0,
        "min_lots": round(float(lotmin), 3), "max_lots": round(float(lotmax), 3),
        "blocked_below_min_lot": int(blocked),
        "notional_cap_binding_proposals": int(notional),
        "risk_budget_binding_proposals": int(risk_bound),
        "total_cost_usd": round(float(costs), 2),
        "gap_exits": int(gaps),
        "risk_stop_triggered": int(breach),
    }


def features(data):
    o, h, l, c = [data[key].astype(float) for key in ("open", "high", "low", "close")]
    bad = ~(np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
            & (o > 0) & (h >= np.maximum(o,c)) & (l <= np.minimum(o,c)))
    o[bad] = np.nan
    h[bad] = np.nan
    l[bad] = np.nan
    c[bad] = np.nan
    cs = pd.Series(c)
    os = pd.Series(o)
    hs = pd.Series(h)
    ls = pd.Series(l)
    tr = pd.concat([hs-ls, abs(hs-cs.shift(1)), abs(ls-cs.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(14,min_periods=14).mean().to_numpy() * PIP
    ema = {n:cs.ewm(span=n,adjust=False,min_periods=n).mean().to_numpy()
           for n in (12,24,48,96,192)}
    z = {}
    for n in (24,72,144):
        mean = cs.rolling(n,min_periods=n).mean()
        sd = cs.rolling(n,min_periods=n).std()
        z[n] = ((cs-mean)/sd).to_numpy()
    chan = {n:(hs.shift(1).rolling(n,min_periods=n).max().to_numpy(),
               ls.shift(1).rolling(n,min_periods=n).min().to_numpy())
            for n in (24,48,96)}
    ret = {n:(cs-cs.shift(n)).to_numpy()*PIP for n in (3,6,12)}
    delta=cs.diff()
    for_rsi={}
    up=delta.clip(lower=0); down=-delta.clip(upper=0)
    for n in (7,14):
        au=up.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
        ad=down.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
        rs=au/(ad.replace(0,np.nan))
        for_rsi[n]=(100-100/(1+rs)).to_numpy()
    hours=(data["timestamp"]//3600000 % 24).astype(np.int16)
    return {"close":c, "open":o, "high":h, "low":l, "atr":atr,
            "ema":ema, "z":z, "channel":chan, "ret":ret,
            "rsi":for_rsi, "hours":hours, "bad_rows":int(bad.sum())}


def generate(f):
    c, o, a = f["close"], f["open"], f["atr"]
    families={}
    for n in (24,72,144):
        for threshold in (1.0,1.8,2.4):
            z=f["z"][n]
            families[f"reversion/z{n}/{threshold}"] = np.where(
                z < -threshold,1,np.where(z>threshold,-1,0))
    for fast, slow in ((12,48),(24,96),(48,192)):
        sign=np.where(f["ema"][fast]>f["ema"][slow],1,-1)
        for r in (3,6,12):
            for threshold in (1.0,3.0):
                momentum=f["ret"][r]
                families[f"trend/ema{fast}-{slow}/r{r}t{threshold}"]=np.where(
                    (momentum > threshold)&(sign>0),1,
                    np.where((momentum < -threshold)&(sign<0),-1,0))
    for n in (24,48,96):
        hi,lo=f["channel"][n]
        for k in (0.0,0.3,0.8):
            families[f"breakout/donchian{n}/atr{k}"]=np.where(
                c>hi+k*a/PIP,1,np.where(c<lo-k*a/PIP,-1,0))
    for n in (7,14):
        for threshold in (20,30,40):
            r=f["rsi"][n]
            families[f"rsi-reversal/{n}/t{threshold}"]=np.where(
                r<threshold,1,np.where(r>100-threshold,-1,0))
    for n in (24,72):
        z=f["z"][n]
        sign=np.where(f["ema"][48]>f["ema"][192],1,-1)
        for k in (0.7,1.2,1.8):
            families[f"trend-pullback/z{n}/t{k}"]=np.where(
                (sign>0)&(z < -k),1,np.where((sign<0)&(z>k),-1,0))
    rng=(f["high"]-f["low"])*PIP
    for k in (1.5,2.0,3.0):
        families[f"vol-expansion/bar-atr{k}"]=np.where(
            rng>k*a,np.where(c>o,1,np.where(c<o,-1,0)),0)
    for key, raw in families.items():
        for session, eligible in (
            ("all",np.ones(len(c),dtype=bool)),
            ("london", (f["hours"]>=7)&(f["hours"]<16)),
            ("overlap", (f["hours"]>=12)&(f["hours"]<17)),
        ):
            s=np.where(eligible,raw,0)
            s=np.where(np.isfinite(c)&np.isfinite(a)&(a>0),s,0).astype(np.int8)
            for rr, hold in RR_PROFILES:
                yield f"{key}/{session}/RR{rr}/hold{hold}",key.split("/")[0],s,rr,hold


def score(mode, m, train_len):
    if m["entries"] < max(40, train_len//12000):
        return -1e18
    if m["risk_stop_triggered"]:
        return -1e18
    if m["return_pct"] <= 0:
        return -1e18
    if mode == "PropFirm":
        if m["max_drawdown_pct"] >= 8:
            return -1e18
        return m["return_pct"]-1.3*m["max_drawdown_pct"]
    if m["max_drawdown_pct"] >= 70:
        return -1e18
    return math.log(max(m["final_usd"],1e-6)/m["initial_usd"])-0.005*m["max_drawdown_pct"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive", default="EURUSD.tar")
    p.add_argument("--out", default="eurusd-edge-lab-results")
    p.add_argument("--max-candidates",type=int,default=0)
    args=p.parse_args()
    path=Path(args.out);path.mkdir(parents=True,exist_ok=True)
    data,total,allowed=load(args.archive)
    n=len(data["timestamp"])
    assert allowed==n and n>100000
    learning=int(allowed*0.75)
    f=features(data)
    arrays={k:np.ascontiguousarray(data[k].astype(float)) for k in ("open","high","low","close")}
    ts=np.ascontiguousarray(data["timestamp"].astype(np.int64))
    atr=np.ascontiguousarray(f["atr"].astype(np.float64))
    count=0
    family_counts={}
    chosen={}
    best_unqualified={}
    for name,family,s,rr,hold in generate(f):
        count+=1;family_counts[family]=family_counts.get(family,0)+1
        sig=np.ascontiguousarray(s)
        for mode,capital,risk,cap in MODES:
            for arm in ARM_FRACTIONS:
                stop=int(learning*arm)
                m=metrics(simulate(sig, arrays["open"],arrays["high"],arrays["low"],arrays["close"],
                                   ts,atr,0,stop,capital,risk,cap,rr,hold,.4,.2),capital)
                rank=score(mode,m,stop)
                key=(mode,arm)
                if key not in chosen or rank>chosen[key]["rank"]:
                    chosen[key]=dict(rank=rank,name=name,family=family,rr=rr,hold=hold,
                                     sig=sig.copy(),train=m,mode=mode,capital=capital,risk=risk,cap=cap)
                if key not in best_unqualified or m["return_pct"]>best_unqualified[key]["return_pct"]:
                    best_unqualified[key]={"name":name,"return_pct":m["return_pct"]}
        if args.max_candidates and count>=args.max_candidates:
            break
    rows=[]
    for (mode,arm),selected in sorted(chosen.items()):
        # If none clears eligibility, report no discovery, never promote a loser.
        accepted=selected["rank"]> -1e17
        s=selected["sig"]
        rr=selected["rr"];hold=selected["hold"]
        capital=selected["capital"];risk=selected["risk"];cap=selected["cap"]
        def run(start,stop,spread,fee,initial=None):
            ini=capital if initial is None else initial
            return metrics(simulate(s,arrays["open"],arrays["high"],arrays["low"],arrays["close"],
                                    ts,atr,start,stop,ini,risk,cap,rr,hold,spread,fee),ini)
        costs={"spread_0.0_fee_0.0":run(learning,allowed,0,0),
               "spread_0.2_fee_0.0":run(learning,allowed,.2,0),
               "spread_0.4_fee_0.0":run(learning,allowed,.4,0),
               "spread_0.4_fee_0.2":run(learning,allowed,.4,.2),
               "spread_0.4_fee_0.4":run(learning,allowed,.4,.4)}
        cut=np.linspace(learning,allowed,5).astype(int)
        folds=[run(int(st),int(en),.4,.2) for st,en in zip(cut[:-1],cut[1:])]
        zero_capital=run(learning,allowed,.4,.2,20.) if mode=="Risky" else None
        rows.append({
            "mode":mode,"training_fraction":arm,
            "admitted_in_training":accepted,
            "selected_rule":selected["name"],"family":selected["family"],
            "train_at_0.4_spread_plus_0.2_fee":selected["train"],
            "internal_forward_spread_fee_sensitivity":costs,
            "internal_forward_quarters_at_0.4_spread_plus_0.2_fee":folds,
            "positive_quarters":sum(1 for x in folds if x["net_usd"]>0),
            "risky_20usd_viability":zero_capital,
            "validation_is_selection_only":True,
        })
    result={
        "name":"EURUSD M5 independent exploratory dynamic-lot edge lab",
        "limitations":"NOT NeoEthos GA, no cTrader bid/ask or broker commission confirmation, no final OOS",
        "data_full_bars":total,"allowed_selection_bars":allowed,
        "untouched_final_bars":total-allowed,"training_pool_bars":learning,
        "internal_forward_bars":allowed-learning,
        "bad_OHLC_rows_selection":f["bad_rows"],"candidate_count":count,
        "family_counts":family_counts,"methodology":"chronological candidate selection on early train arms; fixed later internal forward",
        "assumed_contract":{"deposit_currency":"USD","EURUSD_pip_value_usd_per_lot":10,
                            "min_lot":0.01,"lot_step":0.01,"max_lots":100,
                            "max_notional_multiple":30},
        "sizing":"dynamic each entry equity * risk_fraction / (SL pips * $10), minimum of broker/grid/notional",
        "modes":{"Risky":{"capital":100,"risk_fraction":0.10,"daily_trade_cap":None},
                 "PropFirm":{"capital":10000,"risk_fraction":0.01,"daily_trade_cap":8}},
        "results":rows,
    }
    (path/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Dynamic-lot EURUSD M5 edge-discovery diagnostic","",
           "**Selection only — no final OOS. NOT the NeoEthos genetic engine.**",
           f"Real M5 bars: {total:,}; allowed {allowed:,}; learning {learning:,}; internal forward {allowed-learning:,}.",
           f"Tested: {count} candidates across {len(family_counts)} distinct signal families.",
           "Spread stress 0.0/0.2/0.4 plus extra fee 0.0/0.2/0.4 pip per round trip.",
           "Position size changes with CURRENT equity, actual stop distance, and broker 0.01 lot grid.",
           "Live spread timing, commission and slippage not available; not deployable evidence.","",
           "| Mode | Early window | Selected family | Qualified in training? | Train % | Internal forward % (0.4+0.2) | Forward folds +/4 | Forward min/avg/max lot |",
           "|---|---:|---|---|---:|---:|---:|---|"]
    for x in rows:
        base=x["internal_forward_spread_fee_sensitivity"]["spread_0.4_fee_0.2"]
        ok="yes" if x["admitted_in_training"] else "**no**"
        lines.append(f'| {x["mode"]} | {x["training_fraction"]:.0%} | {x["family"]} | {ok} | '
                     f'{x["train_at_0.4_spread_plus_0.2_fee"]["return_pct"]:+.2f}% | '
                     f'{base["return_pct"]:+.2f}% | {x["positive_quarters"]}/4 | '
                     f'{base["min_lots"]:.2f}/{base["mean_lots"]:.2f}/{base["max_lots"]:.2f} |')
    lines.extend(["","Important: positive quarter counts are fixed time segments, not independent OOS.",
                  "A training survivor with negative internal forward fails research-level validation.",
                  "The 20 USD Risky lot feasibility probe is in detailed JSON.",
                  "Nothing in this report authorizes real money or claims production strategy viability.",""])
    (path/"report.md").write_text("\n".join(lines))
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
