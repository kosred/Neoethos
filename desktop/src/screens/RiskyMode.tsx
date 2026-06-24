import { useEffect, useState } from "react";
import { riskyScenarios, type RiskyParams, type RiskyScenario } from "../api";

const pctOf = (v: number) => `${(v * 100).toFixed(0)}%`;
const days = (n: number) => (n >= 365 ? `${(n / 365).toFixed(1)}y` : n >= 30 ? `${(n / 30).toFixed(1)}mo` : `${Math.round(n)}d`);

export default function RiskyMode() {
  const [d, setD] = useState<RiskyScenario | null>(null);
  const [p, setP] = useState<RiskyParams>({});
  const [seeded, setSeeded] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    riskyScenarios({})
      .then((r) => {
        setD(r);
        setP({ startingUsd: r.startingUsd, targetUsd: r.targetUsd, riskFraction: r.riskFraction, winRate: r.winRate, rewardToRisk: r.rewardToRisk, tradesPerDay: r.tradesPerDay });
        setSeeded(true);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (!seeded || !p.startingUsd) return;
    const id = setTimeout(() => riskyScenarios(p).then(setD).catch((e) => setErr(String(e))), 250);
    return () => clearTimeout(id);
  }, [p, seeded]);

  const set = (k: keyof RiskyParams, v: number) => setP((prev) => ({ ...prev, [k]: v }));
  const mult = d ? d.targetUsd / d.startingUsd : 0;
  const ruinHigh = d ? d.ruinProbability > 0.25 : false;

  return (
    <div className="screen">
      <h1>Risky Mode <span className="badge live">AGGRESSIVE</span></h1>
      <p className="sub">Account-multiplication mode — small bankroll → big target, fractional-Kelly sizing</p>

      <div className="banner warn">
        High-risk growth strategy. Per-trade risk runs {d ? `${pctOf(d.riskFractionMin)}–${pctOf(d.riskFractionMax)}` : "30–50%"} of
        bankroll — a drawdown can wipe the account. Kill-switch tiers (daily/weekly/monthly) cap losses, but ruin is a real outcome.
      </div>
      {err && <div className="banner warn">{err}</div>}

      <h2>Target</h2>
      <div className="ticket">
        <div className="ticket-row">
          <label>Start<input type="number" value={p.startingUsd ?? ""} onChange={(e) => set("startingUsd", Number(e.target.value))} style={{ width: 100 }} /></label>
          <label>Target<input type="number" value={p.targetUsd ?? ""} onChange={(e) => set("targetUsd", Number(e.target.value))} style={{ width: 120 }} /></label>
          <span className="muted" style={{ alignSelf: "flex-end", paddingBottom: 6 }}>= <b style={{ color: "#e5e7eb" }}>{mult.toFixed(0)}×</b></span>
        </div>
        <div className="ticket-row" style={{ marginTop: 8 }}>
          <span style={{ fontSize: 11, color: "#6b7280", alignSelf: "center" }}>Aggression (risk/trade):</span>
          {d && [d.riskFractionMin, (d.riskFractionMin + d.riskFractionMax) / 2, d.riskFractionMax].map((rf, i) => (
            <button key={i} className={Math.abs((p.riskFraction ?? 0) - rf) < 0.001 ? "primary" : ""} onClick={() => set("riskFraction", rf)}>
              {pctOf(rf)}
            </button>
          ))}
        </div>
        <div className="ticket-row" style={{ marginTop: 8 }}>
          <label>Win rate %<input type="number" value={p.winRate != null ? Math.round(p.winRate * 100) : ""} onChange={(e) => set("winRate", Number(e.target.value) / 100)} style={{ width: 80 }} /></label>
          <label>Reward:Risk<input type="number" step="0.1" value={p.rewardToRisk ?? ""} onChange={(e) => set("rewardToRisk", Number(e.target.value))} style={{ width: 80 }} /></label>
          <label>Trades/day<input type="number" value={p.tradesPerDay ?? ""} onChange={(e) => set("tradesPerDay", Number(e.target.value))} style={{ width: 80 }} /></label>
        </div>
      </div>

      <h2>Time to target</h2>
      {d ? (
        <div className="cards">
          <div className="card"><div className="card-label">BEST CASE</div><div className="card-value">{days(d.bestCaseDays)}</div></div>
          <div className="card accent"><div className="card-label">EXPECTED</div><div className="card-value">{days(d.expectedDays)}</div></div>
          <div className="card"><div className="card-label">CONSERVATIVE</div><div className="card-value">{days(d.conservativeDays)}</div></div>
          <div className="card"><div className="card-label">RUIN PROBABILITY</div><div className="card-value" style={{ color: ruinHigh ? "#ef4444" : "#fca5a5" }}>{(d.ruinProbability * 100).toFixed(1)}%</div></div>
        </div>
      ) : (
        <p className="muted">Computing…</p>
      )}

      <p className="muted small" style={{ marginTop: 12 }}>
        Discover risky strategies in <b>Discovery</b> (risky mode), then run them in <b>Autopilot</b>. The engine's
        RiskyModeManager enforces staged sizing + kill-switch limits live.
      </p>
    </div>
  );
}
