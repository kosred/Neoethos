import { useState } from "react";
import { riskInfo, setRiskPreset, settings } from "../api";
import { usePoll } from "../hooks";
import { HelpPanel, HelpStep } from "../components/Help";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

export default function Risk() {
  const { data, error, reload } = usePoll(riskInfo, 0);
  // `system.trading_mode` is what decides which rule set governs sizing. This
  // screen used to render `risk.prop_firm_rules` instead, a field with one
  // write, one display read and ZERO decisions — it could announce "Prop-firm"
  // while the account was being sized under risky rules.
  const { data: cfg } = usePoll(settings, 0);
  const riskyLive = cfg?.tradingMode === "risky";
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

      <HelpPanel id="risk">
        <p>These are the guardrails every <b>automated</b> trade must respect — how much to risk per trade and when to stop after losses.</p>
        <HelpStep n={1}>Pick a <b>preset</b> (e.g. an FTMO-style prop-firm profile) to load a sensible, tested set of limits in one click.</HelpStep>
        <HelpStep n={2}>Review the values: <b>risk per trade</b>, <b>daily</b> and <b>total drawdown</b> caps, and <b>max lot size</b>. If a daily/total loss limit is hit, the engine stops trading to protect the account.</HelpStep>
        {/* Was: "Manual orders in Positions are not gated by these". False
            since 2026-08-09 — orders.rs:116-137 refuses a manual order with a
            400 when require_stop_loss is on, and its refusal text points at a
            Settings control that does not exist anywhere in this app. */}
        <p className="muted small">These apply to autopilot / risky mode. <b>Require stop-loss is the exception</b> — with it on, a manual order from Positions without a stop is refused too.</p>
      </HelpPanel>

      {error && <div className="banner warn">{error}</div>}

      {/* 💰 In risky mode the card below is not what the bot risks: live sizing
          ignores risk.risk_per_trade (live_trading.rs:1664-1680) and uses the
          engine ladder, capped by max_portfolio_risk. Say so above the number,
          not in a footnote. */}
      {data && riskyLive && (
        <div className="banner warn">
          <b>Trading mode is RISKY — the RISK / TRADE figure below is not in force.</b> Risky live
          sizing discards <code>risk.risk_per_trade</code> and uses the engine's own 30–50% ladder,
          capped by <b>max portfolio risk</b>. The band actually applied is on the{" "}
          <b>Risky Mode</b> screen. These drawdown caps, the max lot and the preset below still
          apply.
        </div>
      )}

      {data && (
        <>
          <div className="cards">
            <div className="card">
              <div className="card-label">RISK / TRADE{riskyLive ? " (NOT IN FORCE)" : ""}</div>
              <div className="card-value" style={riskyLive ? { opacity: 0.45 } : undefined}>{pct(data.riskPerTrade)}</div>
            </div>
            <div className="card"><div className="card-label">DAILY DD LIMIT</div><div className="card-value">{pct(data.dailyDrawdownLimit)}</div></div>
            <div className="card"><div className="card-label">TOTAL DD LIMIT</div><div className="card-value">{pct(data.totalDrawdownLimit)}</div></div>
            <div className="card"><div className="card-label">MAX LOT</div><div className="card-value">{data.maxLotSize}</div></div>
          </div>

          <div className="settings-grid" style={{ marginTop: 14 }}>
            <div className="kv"><span>Risk/trade range</span><b>{pct(data.minRiskPerTrade)} – {pct(data.maxRiskPerTrade)}</b></div>
            <div className="kv"><span>Require stop-loss <span className="muted small">(autopilot + manual)</span></span><b className={data.requireStopLoss ? "buy" : "sell"}>{data.requireStopLoss ? "yes" : "no"}</b></div>
            {/* Derived from system.trading_mode — the field that actually
                decides. The old row read risk.prop_firm_rules, which no engine
                reads: every discovery call passes a hardcoded
                PropFirmRiskRules::default() regardless. */}
            <div className="kv"><span>Rules in force</span><b className={riskyLive ? "sell" : "buy"}>{cfg ? (riskyLive ? "Risky" : "Prop-firm") : "—"}</b></div>
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
      <p className="muted small" style={{ marginTop: 12 }}>
        Aggressive account-multiplication lives in its own <b>Risky Mode</b> screen.
      </p>
    </div>
  );
}
