import { riskyScenarios, settings, strategyList, riskInfo } from "../api";
import { usePoll } from "../hooks";

const days = (n: number | null) =>
  n == null || Number.isNaN(n) ? "never" : n >= 365 ? `${(n / 365).toFixed(1)}y` : n >= 30 ? `${(n / 30).toFixed(1)}mo` : `${Math.round(n)}d`;
const eur = (v: number) => `€${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

export default function RiskyMode() {
  const { data: cfg } = usePoll(settings, 0);
  const { data: risk } = usePoll(riskInfo, 0);
  const { data: list } = usePoll(strategyList, 0);

  const start = cfg?.riskyStartBalance ?? 100;
  const target = cfg?.riskyTargetBalance ?? 100000;
  const horizon = cfg?.riskyHorizonDays ?? 180;
  // Engine-decided projection for the CONFIG goal (sourced from config, NOT user input).
  const { data: proj } = usePoll(() => riskyScenarios({ startingUsd: start, targetUsd: target }), 0, [start, target]);

  const mult = target / start;
  const riskyStrats = (list?.strategies ?? []).filter((s) => s.mode === "risky");
  const propActive = risk?.propFirmRulesEnabled;

  return (
    <div className="screen">
      <h1>Risky Mode <span className="badge live">AUTOMATIC · AGGRESSIVE</span></h1>
      <p className="sub">The bot hunts for strategies to reach the goal — sizing, win-rate, R:R and costs are all engine-decided</p>

      <div className="banner warn">
        Fully automatic — you don't set any parameters. The engine searches for strategies that can hit the goal,
        sizes them itself, and computes real costs per-lot from the broker. High risk: ruin is a real outcome.
      </div>

      <h2>The goal (from config)</h2>
      <div className="cards">
        <div className="card"><div className="card-label">START</div><div className="card-value">{eur(start)}</div></div>
        <div className="card accent"><div className="card-label">TARGET</div><div className="card-value">{eur(target)}</div></div>
        <div className="card"><div className="card-label">MULTIPLIER</div><div className="card-value">{mult.toFixed(0)}×</div></div>
        <div className="card"><div className="card-label">HORIZON</div><div className="card-value">{days(horizon)}</div></div>
      </div>

      <h2>Engine estimate <span className="muted small">(not user-tunable)</span></h2>
      {proj ? (
        <div className="cards">
          <div className="card"><div className="card-label">EXPECTED</div><div className="card-value">{days(proj.expectedDays)}</div></div>
          <div className="card"><div className="card-label">BEST CASE</div><div className="card-value">{days(proj.bestCaseDays)}</div></div>
          <div className="card"><div className="card-label">RISK / TRADE</div><div className="card-value">{(proj.riskFraction * 100).toFixed(0)}%</div></div>
          <div className="card"><div className="card-label">RUIN PROB.</div><div className="card-value" style={{ color: "#ef4444" }}>{(proj.ruinProbability * 100).toFixed(1)}%</div></div>
        </div>
      ) : (
        <p className="muted">Computing…</p>
      )}
      <p className="muted small">Costs (spread/commission/swap) are taken per-lot from the broker and folded into every estimate automatically.</p>

      <h2>Strategies found for this dream</h2>
      {riskyStrats.length === 0 ? (
        <p className="muted">No risky-mode strategies discovered yet — run Discovery in risky mode. Validated ones appear in Strategy Report.</p>
      ) : (
        <table className="tbl">
          <thead><tr><th>Symbol</th><th>TF</th><th>Win%</th><th>Trades</th><th>CAGR%</th><th>OOS</th><th>Flags</th></tr></thead>
          <tbody>
            {riskyStrats.slice(0, 12).map((s) => (
              <tr key={s.base}>
                <td><b>{s.symbol}</b></td>
                <td>{s.timeframe}</td>
                <td>{s.winRate != null ? (s.winRate * 100).toFixed(1) : "—"}</td>
                <td>{s.trades}</td>
                <td>{Math.abs(s.cagrPct) > 1000 ? "🚩" : s.cagrPct.toFixed(1)}</td>
                <td><span className={`badge ${s.walkforwardPassed ? "demo" : "live"}`}>{s.walkforwardPassed ? "✓" : "✗"}</span></td>
                <td>{s.flags.length > 0 ? <span className="sell small">🚩 {s.flags.length}</span> : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h2>Mode exclusivity</h2>
      <div className="banner info">
        Risky and Prop-firm are two separate configs — only ONE runs at a time. The other is dormant until you pause
        or the account is blown. Currently active rules: <b>{propActive ? "Prop-firm" : "Risky"}</b>.
      </div>
    </div>
  );
}
