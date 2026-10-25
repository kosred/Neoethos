#!/usr/bin/env python3
"""Independent EURUSD M5 25/50/100 training-window experiment.

THIS IS NOT THE NEOETHOS GA. First 80% of real OHLCV is allowed,
last 20% never used. Last quarter of allowed 80% is fixed internal
validation. Explore the first 25/50/100% of the earlier learning pool.
"""
import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import vortex as vx

ARCHIVE_SHA = "042aa66fde29e697bd30d93d85c4762076c06e292097e42680ed0067b06ca061"
M5_SHA = "4b12214461f8abe09c93f656192fb6c4b71d5c9ca2ea03e37403c52ad5f3bcd0"
SPREADS = (0.0, 0.2, 0.4)
MODES = ("Risky", "PropFirm")


def load(archive_path):
    digest = hashlib.sha256()
    with open(archive_path, "rb") as inp:
        for part in iter(lambda: inp.read(4 << 20), b""):
            digest.update(part)
    assert digest.hexdigest() == ARCHIVE_SHA, "archive checksum mismatch"
    with tarfile.open(archive_path, "r") as tar:
        matches = [m for m in tar if m.isfile() and m.name == "symbol=EURUSD/timeframe=M5/data.vortex"]
        assert len(matches) == 1
        m = matches[0]
        assert m.size < 100_000_000
        dest = Path("/tmp/neoethos-research-m5.vortex")
        with tar.extractfile(m) as src, dest.open("wb") as out:
            import shutil
            shutil.copyfileobj(src, out)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == M5_SHA
    arr = vx.open(str(dest)).to_arrow()
    if hasattr(arr, "read_all"):
        arr = arr.read_all()
    full = arr.num_rows
    allowed = int(full * 0.8)
    data = {
        name: np.asarray(arr.column(name).slice(0, allowed).to_numpy(zero_copy_only=False))
        for name in ("timestamp", "open", "close", "high", "low")
    }
    # Never retrieve or score the final 20% of bars.
    del arr
    return data, full, allowed


def zscore(close, width):
    valid = np.isfinite(close) & (close > 0)
    x = np.where(valid, close, 0.0).astype(float)
    cs = np.r_[0.0, np.cumsum(x)]
    cs2 = np.r_[0.0, np.cumsum(x*x)]
    counts = np.r_[0, np.cumsum(valid.astype(np.int64))]
    n = counts[width:] - counts[:-width]
    mu = (cs[width:] - cs[:-width]) / width
    variance = (cs2[width:] - cs2[:-width]) / width - mu*mu
    z = np.full(len(close), np.nan)
    good = (n == width) & (variance > 1e-12)
    v = np.full(len(mu), np.nan)
    v[good] = (close[width-1:][good] - mu[good]) / np.sqrt(variance[good])
    z[width-1:] = v
    return z


def candidates(closes):
    for width in (12, 24, 48, 96, 192):
        z = zscore(closes, width)
        for threshold in (0.65, 1.1, 1.55):
            sig = np.where(z > threshold, 1, np.where(z < -threshold, -1, 0)).astype(np.int8)
            yield f"momentum-z{width}-t{threshold}", sig
            yield f"reversion-z{width}-t{threshold}", -sig


def measure(position, gross, turnover, spread, start, end):
    pnl = position[start:end] * gross[start:end] - spread * turnover[start:end]
    trades = int(np.sum((position[start:end] != 0) & (turnover[start:end] >= 0.5)))
    equity = np.r_[0., np.cumsum(pnl)]
    dd = float(np.max(np.maximum.accumulate(equity) - equity))
    net = float(np.sum(pnl))
    return {
        "net_pips": round(net, 3),
        "trades": trades,
        "pips_per_entry": round(net/trades, 4) if trades else None,
        "max_dd_pips": round(dd, 3),
        "positive": net > 0
    }


def objective(mode, m, n):
    if m["trades"] < max(30, n//8000):
        return float("-inf")
    p = m["net_pips"]*1000/n
    if mode == "Risky":
        return p - 0.025*m["max_dd_pips"]/max(1.0, math.sqrt(n/1000))
    return p/(1 + m["max_dd_pips"]/100)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--archive", default="EURUSD.tar")
    a.add_argument("--out", default="stage1-baseline-results")
    args = a.parse_args()
    path = Path(args.out)
    path.mkdir(exist_ok=True, parents=True)
    d, total, allowed = load(args.archive)
    learn = int(0.75 * allowed)
    o, c, t = d["open"], d["close"], d["timestamp"]
    assert len(o) > 40000 and np.all(np.diff(t) > 0)
    good = (np.isfinite(o[1:-1]) & np.isfinite(o[2:]) &
            (o[1:-1] > 0) & (o[2:] > 0) &
            np.isfinite(c[:-2]) & (c[:-2] > 0) &
            ((t[1:-1]-t[:-2]) == 300000) &
            ((t[2:]-t[1:-1]) == 300000))
    gross = np.where(good, (o[2:]-o[1:-1])*10000, 0.0)
    candidates_seen = 0
    picked = {(mode, fraction): (float("-inf"), None, None, None)
              for mode in MODES for fraction in (0.25, 0.5, 1.0)}
    for label, s in candidates(c):
        candidates_seen += 1
        position = np.where(good, s[:-2], 0).astype(np.int8)
        prior = np.r_[0, position[:-1]]
        turnover = abs(position.astype(np.int16)-prior.astype(np.int16)) * 0.5
        for fraction in (0.25, 0.5, 1.0):
            end = int(learn*fraction)
            metrics = measure(position, gross, turnover, 0.2, 0, end)
            for mode in MODES:
                score = objective(mode, metrics, end)
                if score > picked[(mode, fraction)][0]:
                    picked[(mode, fraction)] = score, label, position, turnover

    rows = []
    for (mode, fraction), (score, name, pos, turn) in sorted(picked.items()):
        if pos is None:
            rows.append(dict(mode=mode, fraction=fraction, error="no activity-qualified candidate"))
            continue
        stops = np.linspace(learn, len(gross), 5).astype(int)
        quarters = [measure(pos, gross, turn, 0.2, int(x), int(y))
                    for x, y in zip(stops[:-1], stops[1:])]
        rows.append(dict(
            mode=mode, fraction=fraction, selected_rule=name, selection_score=round(score, 5),
            stage1_train=measure(pos, gross, turn, 0.2, 0, int(learn*fraction)),
            internal_forward={str(s): measure(pos, gross, turn, s, learn, len(gross))
                              for s in SPREADS},
            internal_forward_quarters=quarters,
            positive_forward_quarters=sum(x["positive"] for x in quarters),
        ))
    payload = dict(
        experiment="simple-rules-independent-baseline-NOT-NEOETHOS-GA",
        full_rows=total, selection_prefix_rows=allowed,
        untouched_final_suffix_rows=total-allowed,
        learning_pool_rows=learn,
        internal_forward_rows=len(gross)-learn,
        invalid_or_discontinuous_intervals=int((~good).sum()),
        candidates=candidates_seen,
        assumed_spreads_pips=SPREADS,
        selection_spread_pips=0.2,
        commission_and_slippage="unknown, NOT deducted",
        strict_limit="no final OOS, no trading or deployment",
        results=rows
    )
    (path/"results.json").write_text(json.dumps(payload, indent=2)+"\n")
    report = [
        "# EURUSD M5 pre-OOS window experiment", "",
        "Independent deterministic baseline, NOT the NeoEthos genetic search.",
        "Final reserved 20% was never read for scoring. Internal forward is",
        "the last 25% of the allowed 80%; Stage 1 uses 25/50/100% of its earlier learning pool.",
        f"Bars: {total:,} total; {allowed:,} allowed; {learn:,} learning; {len(gross)-learn:,} internal forward.",
        f"Rules: {candidates_seen}. Invalid/gapped intervals: {int((~good).sum()):,}.",
        "Round-trip spreads: 0.0 / 0.2 / 0.4 pips. NO commission or slippage.",
        "Signals are observed on the closing bar, trades execute at the next bar open.",
        "The Risky / PropFirm selections are simple objectives, NOT full risk-mode gates.", "",
        "| Mode proxy | Stage1 | Selected rule | Training pips (0.2) | Internal forward pips (0.0 / 0.2 / 0.4) | Positive quarters (0.2) |",
        "|---|---:|---|---:|---|---:|"
    ]
    for row in rows:
        if "error" in row:
            report.append(f'| {row["mode"]} | {row["fraction"]:.0%} | {row["error"]} | - | - | - |')
            continue
        f = row["internal_forward"]
        report.append(
            f'| {row["mode"]} | {row["fraction"]:.0%} | {row["selected_rule"]} | '
            f'{row["stage1_train"]["net_pips"]:+.1f} | '
            f'{f["0.0"]["net_pips"]:+.1f} / {f["0.2"]["net_pips"]:+.1f} / {f["0.4"]["net_pips"]:+.1f} | '
            f'{row["positive_forward_quarters"]}/4 |')
    report.extend(["", "This diagnostic is not broker-realistic PnL or evidence of deployable edge.",
                   "For drawdown, trades and each forward quarter see results.json.", ""])
    (path/"report.md").write_text("\n".join(report))
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
