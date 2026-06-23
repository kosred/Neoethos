import { useState } from "react";
import { riskInfo, setRiskPreset, type PresetSummary } from "../api";
import { usePoll } from "../hooks";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const pp = <T,>(...vals: (T | undefined)[]) => vals.find((v) => v !== undefined);

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

  const presetName = (p: PresetSummary) => pp(p.displayName, p.display_name) ?? p.id;

  return (
    <div className="screen">
      <h1>Risk</h1>
      <p className="sub">Position sizing limits, drawdown guards, and prop-firm presets</p>

      {error && <div className="banner warn">{error}</div>}

      {data && (
        <>
          <div className="cards">
            <div className="card"><div className="card-label">RISK / TRADE</div><div className="card-value">{pct(data.risk_per_trade)}</div></div>
            <div className="card"><div className="card-label">DAILY DD LIMIT</div><div className="card-value">{pct(data.daily_drawdown_limit)}</div></div>
            <div className="card"><div className="card-label">TOTAL DD LIMIT</div><div className="card-value">{pct(data.total_drawdown_limit)}</div></div>
            <div className="card"><div className="card-label">MAX LOT</div><div className="card-value">{data.max_lot_size}</div></div>
          </div>

          <div className="settings-grid" style={{ marginTop: 14 }}>
            <div className="kv"><span>Risk/trade range</span><b>{pct(data.min_risk_per_trade)} – {pct(data.max_risk_per_trade)}</b></div>
            <div className="kv"><span>Require stop-loss</span><b className={data.require_stop_loss ? "buy" : "sell"}>{data.require_stop_loss ? "yes" : "no"}</b></div>
            <div className="kv"><span>Prop-firm rules</span><b className={data.prop_firm_rules_enabled ? "buy" : ""}>{data.prop_firm_rules_enabled ? "enabled" : "off"}</b></div>
            <div className="kv"><span>Risky cooldown</span><b>{data.risky_mode_cooldown_remaining_secs != null ? `${data.risky_mode_cooldown_remaining_secs}s` : "—"}</b></div>
          </div>

          <h2>Active preset: {data.preset_display_name || data.preset}</h2>
          <table className="tbl">
            <thead>
              <tr><th>Preset</th><th>Daily loss</th><th>Max DD</th><th>Profit target</th><th>Min days</th><th></th></tr>
            </thead>
            <tbody>
              {data.available_presets.map((p) => {
                const active = p.id === data.preset;
                return (
                  <tr key={p.id}>
                    <td>{presetName(p)}</td>
                    <td>{(pp(p.maxDailyLossPct, p.max_daily_loss_pct) ?? 0).toFixed(1)}%</td>
                    <td>{(pp(p.maxOverallDrawdownPct, p.max_overall_drawdown_pct) ?? 0).toFixed(1)}%</td>
                    <td>{(pp(p.challengeProfitTargetPct, p.challenge_profit_target_pct) ?? 0).toFixed(1)}%</td>
                    <td>{pp(p.minTradingDays, p.min_trading_days) ?? 0}</td>
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
