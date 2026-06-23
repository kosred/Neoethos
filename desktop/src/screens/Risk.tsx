import { useState } from "react";
import { riskInfo, setRiskPreset } from "../api";
import { usePoll } from "../hooks";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

export default function Risk() {
  const { data, error, reload } = usePoll(riskInfo, 0);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const apply = async (preset: string) => {
    setBusy(true);
    setMsg(`Applying ${preset}…`);
    try {
      await setRiskPreset(preset);
      setMsg(`✓ Preset set to ${preset}.`);
      await reload();
    } catch (e) {
      setMsg(`Failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Risk</h1>
      <p className="sub">Position sizing limits, drawdown guards, and prop-firm presets</p>

      {error && <div className="banner warn">{error}</div>}

      {data && (
        <>
          <div className="cards">
            <div className="card"><div className="card-label">RISK / TRADE</div><div className="card-value">{pct(data.riskPerTrade)}</div></div>
            <div className="card"><div className="card-label">DAILY DD LIMIT</div><div className="card-value">{pct(data.dailyDrawdownLimit)}</div></div>
            <div className="card"><div className="card-label">TOTAL DD LIMIT</div><div className="card-value">{pct(data.totalDrawdownLimit)}</div></div>
            <div className="card"><div className="card-label">MAX LOT</div><div className="card-value">{data.maxLotSize}</div></div>
          </div>

          <div className="settings-grid" style={{ marginTop: 14 }}>
            <div className="kv"><span>Risk/trade range</span><b>{pct(data.minRiskPerTrade)} – {pct(data.maxRiskPerTrade)}</b></div>
            <div className="kv"><span>Require stop-loss</span><b className={data.requireStopLoss ? "buy" : "sell"}>{data.requireStopLoss ? "yes" : "no"}</b></div>
            <div className="kv"><span>Prop-firm rules</span><b className={data.propFirmRulesEnabled ? "buy" : ""}>{data.propFirmRulesEnabled ? "enabled" : "off"}</b></div>
            <div className="kv"><span>Risky cooldown</span><b>{data.riskyModeCooldownRemainingSecs != null ? `${data.riskyModeCooldownRemainingSecs}s` : "—"}</b></div>
          </div>

          <h2>Active preset: {data.presetDisplayName || data.preset}</h2>
          <table className="tbl">
            <thead>
              <tr><th>Preset</th><th>Daily loss</th><th>Max DD</th><th>Profit target</th><th>Min days</th><th></th></tr>
            </thead>
            <tbody>
              {data.availablePresets.map((p) => {
                const active = p.id === data.preset;
                return (
                  <tr key={p.id}>
                    <td>{p.displayName || p.id}</td>
                    <td>{(p.maxDailyLossPct ?? 0).toFixed(1)}%</td>
                    <td>{(p.maxOverallDrawdownPct ?? 0).toFixed(1)}%</td>
                    <td>{(p.challengeProfitTargetPct ?? 0).toFixed(1)}%</td>
                    <td>{p.minTradingDays ?? 0}</td>
                    <td>
                      {active ? (
                        <span className="buy small">● Active</span>
                      ) : (
                        <button disabled={busy} onClick={() => apply(p.id)}>Use</button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {msg && <div className="banner info">{msg}</div>}
        </>
      )}
    </div>
  );
}
