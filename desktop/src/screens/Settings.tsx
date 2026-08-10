import { useEffect, useState } from "react";
import {
  brokerStatus,
  brokerAccounts,
  reauthBroker,
  selectAccount,
  settings as getSettings,
  updateSettings,
  setRiskPreset,
  riskInfo,
  brokerCredentials,
  saveBrokerCredentials,
  type BrokerStatus,
  type AccountInfo,
  type BrokerCredentials,
} from "../api";
import { HelpPanel } from "../components/Help";

export default function Settings() {
  const [status, setStatus] = useState<BrokerStatus | null>(null);
  const [accounts, setAccounts] = useState<AccountInfo[]>([]);
  const [cfg, setCfg] = useState<any>(null);
  const [presets, setPresets] = useState<{ id: string; displayName: string }[]>([]);
  const [risk, setRisk] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [modeBusy, setModeBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const refresh = async () => {
    try {
      setStatus(await brokerStatus());
    } catch (e) {
      setMsg(String(e));
    }
    try {
      setCfg(await getSettings());
    } catch {
      /* settings optional */
    }
    try {
      // Use the REAL prop-firm presets the backend's /risk/preset accepts
      // (ftmo/myforexfunds/…), not /settings/presets (conservative/balanced/…)
      // which setRiskPreset rejects with `unknown preset`.
      setPresets((await riskInfo()).availablePresets);
    } catch {
      /* presets optional */
    }
    try {
      setRisk(await riskInfo());
    } catch {
      /* risk optional */
    }
  };

  const setCompute = async (mode: "auto" | "cpu" | "gpu") => {
    setMsg(`Compute mode → ${mode}…`);
    try {
      await updateSettings({ computeMode: mode });
      setCfg(await getSettings());
      setMsg(`✓ Compute mode = ${mode}.`);
    } catch (e) {
      setMsg(`Compute switch failed: ${e}`);
    }
  };

  const applyPreset = async (id: string) => {
    setMsg(`Applying risk preset ${id}…`);
    try {
      await setRiskPreset(id);
      setRisk(await riskInfo());
      setMsg(`✓ Risk preset = ${id}.`);
    } catch (e) {
      setMsg(`Preset failed: ${e}`);
    }
  };

  const setNews = async (patch: Record<string, unknown>) => {
    setMsg("Saving news settings…");
    try {
      await updateSettings(patch as any);
      setCfg(await getSettings());
      setMsg("✓ News settings saved to config.yaml.");
    } catch (e) {
      setMsg(`News save failed: ${e}`);
    }
  };

  const setLoop = async (patch: Record<string, unknown>) => {
    setMsg("Saving autopilot-loop settings…");
    try {
      await updateSettings(patch as any);
      setCfg(await getSettings());
      setMsg("✓ Saved to config.yaml.");
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const setMode = async (mode: "risky" | "prop_firm") => {
    setModeBusy(true);
    setMsg(`Switching discovery mode to ${mode}…`);
    try {
      await updateSettings({ tradingMode: mode });
      setCfg(await getSettings());
      setMsg(
        `✓ Discovery mode = ${mode}. ${
          mode === "risky"
            ? "Aggressive account-multiplication search (goal-compounding, drawdown-agnostic)."
            : "FTMO-style robust search (prop-firm window-pass gates)."
        } Applies to the next discovery run.`,
      );
    } catch (e) {
      setMsg(`Mode switch failed: ${e}`);
    } finally {
      setModeBusy(false);
    }
  };

  // Editable "search tuning" state, seeded from config whenever it (re)loads.
  const [tune, setTune] = useState<any>({});
  const [tuneBusy, setTuneBusy] = useState(false);
  useEffect(() => {
    if (!cfg) return;
    setTune({
      riskyStartBalance: cfg.riskyStartBalance,
      riskyTargetBalance: cfg.riskyTargetBalance,
      riskyHorizonDays: cfg.riskyHorizonDays,
      prefilterTopK: cfg.prefilterTopK,
      convergencePatience: cfg.convergencePatience,
      stagnationPatience: cfg.stagnationPatience,
      noveltyWeight: cfg.noveltyWeight,
      disableSmcGate: cfg.disableSmcGate,
    });
  }, [cfg]);

  const num = (v: any) => (v === "" || v == null ? undefined : Number(v));
  const saveTuning = async () => {
    setTuneBusy(true);
    setMsg("Saving search tuning to config.yaml…");
    try {
      await updateSettings({
        riskyStartBalance: num(tune.riskyStartBalance),
        riskyTargetBalance: num(tune.riskyTargetBalance),
        riskyHorizonDays: num(tune.riskyHorizonDays),
        prefilterTopK: num(tune.prefilterTopK),
        convergencePatience: num(tune.convergencePatience),
        stagnationPatience: num(tune.stagnationPatience),
        noveltyWeight: num(tune.noveltyWeight),
        disableSmcGate: !!tune.disableSmcGate,
      });
      setCfg(await getSettings());
      setMsg("✓ Saved to config.yaml — applies to the next Discovery run.");
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    } finally {
      setTuneBusy(false);
    }
  };
  const setT = (k: string, v: any) => setTune((t: any) => ({ ...t, [k]: v }));

  // ── cTrader API credentials (audit #119) ────────────────────────────────
  // The Dashboard banner has told the operator for months to "go to Settings
  // and add cTrader credentials" while no such form existed anywhere in
  // `desktop/src`. Credentials are compiled into the binary by
  // `neoethos-app/build.rs`; a revoked client_id therefore locked him out of
  // his own broker until someone rebuilt and reinstalled the app. The backend
  // endpoints (`GET`/`POST /broker/credentials`) already existed and are
  // secret-safe: the GET returns a MASK and a boolean, never the secret, and
  // an empty secret on POST means "keep the saved one".
  //
  // Nothing here logs, stores or echoes the typed secret. It goes straight to
  // the backend, which writes it to `broker_credentials.toml` under the app
  // data dir — the same store the OAuth flow already uses.
  const [creds, setCreds] = useState<BrokerCredentials | null>(null);
  const [credForm, setCredForm] = useState({ clientId: "", clientSecret: "", accountId: "", environment: "Demo", redirectUri: "" });
  const [credsBusy, setCredsBusy] = useState(false);
  const [showCreds, setShowCreds] = useState(false);
  const loadCreds = async () => {
    try {
      const c = await brokerCredentials();
      setCreds(c);
      // Pre-fill everything EXCEPT the secret (the server never sends it).
      setCredForm({
        clientId: c.clientId ?? "",
        clientSecret: "",
        accountId: c.accountId ?? "",
        environment: c.environment || "Demo",
        redirectUri: c.redirectUri ?? "",
      });
    } catch (e) {
      setMsg(`Could not read broker credentials: ${e}`);
    }
  };
  const saveCreds = async () => {
    setCredsBusy(true);
    setMsg("Saving cTrader credentials…");
    try {
      const r = await saveBrokerCredentials(credForm);
      // Drop the typed secret from component state the moment it is saved.
      setCredForm((f) => ({ ...f, clientSecret: "" }));
      await loadCreds();
      await refresh();
      setMsg(`✓ ${r?.message ?? "Credentials saved."}`);
    } catch (e) {
      setMsg(`Saving credentials failed: ${e}`);
    } finally {
      setCredsBusy(false);
    }
  };

  const doReauth = async () => {
    setBusy(true);
    setMsg("Opening browser for cTrader OAuth… approve in the browser, then return here.");
    try {
      const r = await reauthBroker();
      setMsg(
        `✓ ${r.message} (token ${r.accessTokenLen} chars, refresh ${r.refreshTokenPresent ? "saved" : "missing"}). ` +
          `From now on the session auto-refreshes — no re-auth needed again.`,
      );
      await refresh();
    } catch (e) {
      setMsg(`Re-auth failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const loadAccounts = async () => {
    setBusy(true);
    try {
      setAccounts(await brokerAccounts());
      setMsg("");
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  };

  const useAccount = async (a: AccountInfo) => {
    setBusy(true);
    setMsg(`Switching to ${a.label}…`);
    try {
      const s = await selectAccount(a.accountId, a.isLive === true, a.label);
      setStatus(s);
      await loadAccounts();
      setMsg(
        `✓ Active account: ${a.label} — environment set to ${a.isLive ? "Live" : "Demo"}. ` +
          `Balance/positions refresh on the Dashboard.`,
      );
    } catch (e) {
      setMsg(`Switch failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Settings</h1>
      <p className="sub">Discovery mode &amp; broker connection</p>

      <HelpPanel id="settings-mode">
        <p>The <b>discovery mode</b> decides what kind of strategies the search hunts for. Pick it here; it applies to the next Discovery / Autopilot run.</p>
        <p><b>Prop-firm</b> = robust, FTMO-style strategies that must pass strict per-window rules (low drawdown, daily-loss limits). <b>Risky</b> = aggressive account-multiplication — it ranks by how fast it compounds toward your goal at half-Kelly and is drawdown-agnostic. Risky's internals are engine-decided; you only choose the mode + the goal.</p>
      </HelpPanel>

      <h2>Discovery mode</h2>
      <div className="ticket">
        <div className="seg" style={{ maxWidth: 360 }}>
          <button
            className={cfg?.tradingMode === "prop_firm" ? "on" : ""}
            disabled={modeBusy}
            onClick={() => setMode("prop_firm")}
          >
            🛡 Prop-firm (robust)
          </button>
          <button
            className={cfg?.tradingMode === "risky" ? "on buy" : ""}
            disabled={modeBusy}
            onClick={() => setMode("risky")}
          >
            🚀 Risky (multiply)
          </button>
        </div>
        {cfg && (
          <p className="muted small" style={{ marginTop: 10 }}>
            Active: <b>{cfg.tradingMode ?? "?"}</b>
            {cfg.tradingMode === "risky" && cfg.riskyStartBalance != null && (
              <> · goal €{Math.round(cfg.riskyStartBalance).toLocaleString()} → €{Math.round(cfg.riskyTargetBalance).toLocaleString()} in {cfg.riskyHorizonDays} days</>
            )}
          </p>
        )}
      </div>

      <h2>Risky goal</h2>
      <p className="muted small">What the Risky search compounds toward. Only used in Risky mode; sizing/win-rate are engine-decided.</p>
      <div className="ticket">
        <div className="ticket-row">
          <label>Start (€)<input type="number" min="1" step="50" value={tune.riskyStartBalance ?? ""} onChange={(e) => setT("riskyStartBalance", e.target.value)} /></label>
          <label>Target (€)<input type="number" min="1" step="1000" value={tune.riskyTargetBalance ?? ""} onChange={(e) => setT("riskyTargetBalance", e.target.value)} /></label>
          <label>Horizon (days)<input type="number" min="1" step="30" value={tune.riskyHorizonDays ?? ""} onChange={(e) => setT("riskyHorizonDays", e.target.value)} /></label>
        </div>
      </div>

      <h2>Search tuning <span className="muted small">(anti-stagnation — change these if Discovery stalls / finds few strategies)</span></h2>
      <div className="ticket">
        <div className="ticket-row" style={{ flexWrap: "wrap", gap: 16 }}>
          <label style={{ minWidth: 150 }}>Indicator pool
            <input type="number" min="10" step="10" value={tune.prefilterTopK ?? ""} onChange={(e) => setT("prefilterTopK", e.target.value)} />
            <span className="muted small">how many indicators the GA may use. Higher = more diverse strategies. <b>Raise if it stalls.</b> Auto-capped at the number of available indicators + SMC — a value above that just means "use them all".</span>
          </label>
          <label style={{ minWidth: 150 }}>Explore patience
            <input type="number" min="10" step="50" value={tune.convergencePatience ?? ""} onChange={(e) => setT("convergencePatience", e.target.value)} />
            <span className="muted small">flat generations before the GA gives up. Raise to search much longer.</span>
          </label>
          <label style={{ minWidth: 150 }}>Diversity kick
            <input type="number" min="1" step="1" value={tune.stagnationPatience ?? ""} onChange={(e) => setT("stagnationPatience", e.target.value)} />
            <span className="muted small">flat generations before heavier mutation + fresh genes kick in.</span>
          </label>
          <label style={{ minWidth: 150 }}>Novelty reward
            <input type="number" min="0" max="1" step="0.05" value={tune.noveltyWeight ?? ""} onChange={(e) => setT("noveltyWeight", e.target.value)} />
            <span className="muted small">0 = off. 0.1–0.3 rewards DIFFERENT genes → more regimes.</span>
          </label>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 8, minWidth: 200 }}>
            <input type="checkbox" checked={!!tune.disableSmcGate} onChange={(e) => setT("disableSmcGate", e.target.checked)} />
            Disable SMC gate
          </label>
        </div>
        <div className="btn-row">
          <button className="primary" disabled={tuneBusy || !cfg} onClick={saveTuning}>{tuneBusy ? "Saving…" : "Save tuning"}</button>
          <span className="muted small">Writes to config.yaml · applies to the next Discovery run.</span>
        </div>
      </div>

      <h2>Compute</h2>
      <p className="muted small">Which hardware discovery/training use. <b>auto</b> picks the best device and fits any card; <b>cpu</b> forces the CPU lane (safest); <b>gpu</b> forces GPU.</p>
      <div className="ticket">
        <div className="seg" style={{ maxWidth: 360 }}>
          {(["auto", "cpu", "gpu"] as const).map((m) => (
            <button key={m} className={cfg?.computeMode === m ? "on" : ""} onClick={() => setCompute(m)}>{m.toUpperCase()}</button>
          ))}
        </div>
        {cfg && <p className="muted small" style={{ marginTop: 8 }}>Active: <b>{cfg.computeMode ?? "?"}</b></p>}
      </div>

      <h2>Risk &amp; sizing</h2>
      <p className="muted small">Position-sizing limits + drawdown guards for AUTOMATED trading (Autopilot/Risky). Pick a preset — the daily/total drawdown caps below update to that firm's rules. <b>Risk %/trade</b> is your own choice: change it in <b>Advanced</b> or the <b>Discovery</b> pre-flight.</p>
      <div className="ticket">
        {presets.length > 0 && (
          <label>Preset
            <select value={risk?.preset ?? ""} onChange={(e) => applyPreset(e.target.value)} style={{ width: 240 }}>
              {!presets.some((p) => p.id === risk?.preset) && <option value="">{risk?.preset ?? "(current)"}</option>}
              {presets.map((p) => <option key={p.id} value={p.id}>{p.displayName}</option>)}
            </select>
          </label>
        )}
        {risk && (
          <div className="cards" style={{ marginTop: 12, gridTemplateColumns: "repeat(4,1fr)" }}>
            <div className="card"><div className="card-label">RISK / TRADE</div><div className="card-value">{((risk.riskPerTrade ?? 0) * 100).toFixed(2)}%</div></div>
            <div className="card"><div className="card-label">DAILY DD CAP</div><div className="card-value">{((risk.dailyDrawdownLimit ?? 0) * 100).toFixed(1)}%</div></div>
            <div className="card"><div className="card-label">TOTAL DD CAP</div><div className="card-value">{((risk.totalDrawdownLimit ?? 0) * 100).toFixed(1)}%</div></div>
            <div className="card"><div className="card-label">MAX LOT</div><div className="card-value">{risk.maxLotSize ?? "—"}</div></div>
          </div>
        )}
        <p className="muted small" style={{ marginTop: 8 }}>Manual orders (Positions) are not gated by these — they apply to automated trading.</p>
      </div>

      <h2>Autopilot loop</h2>
      <p className="muted small">What happens automatically when auto-cull permanently retires a losing strategy.</p>
      <div className="ticket">
        <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={cfg?.autoRediscoverOnCull ?? true}
            onChange={(e) => setLoop({ autoRediscoverOnCull: e.target.checked })}
          />
          Auto-rediscover after a cull — when a strategy is retired (blacklisted forever), automatically start a fresh Discovery on the same symbol + timeframe to refill the gap. Runs when the Discovery engine is idle.
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 8, marginTop: 8 }}>
          <input
            type="checkbox"
            checked={!!cfg?.liveMlGate}
            onChange={(e) => setLoop({ liveMlGate: e.target.checked })}
          />
          <span>
            <b>Live ML gate</b> — the trained model ensemble scales each live entry's risk
            (agreement × regime × anomaly). Strategies still pick the direction; the models can
            only <b>shrink</b> size or skip a bar on a hard regime/anomaly collapse — never flip
            a trade, never create one. Needs trained models for the engine's symbol + timeframe;
            if none load, trading continues gene-only (logged). Takes effect on the next engine start.
          </span>
        </label>
      </div>

      <h2>News gate</h2>
      <p className="muted small">How automated trading behaves around high-impact news events.</p>
      <div className="ticket">
        <div className="ticket-row" style={{ flexWrap: "wrap", gap: 18 }}>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
            <input type="checkbox" checked={!!cfg?.newsCalendarEnabled} onChange={(e) => setNews({ newsCalendarEnabled: e.target.checked })} />
            Economic calendar enabled
          </label>
          <label>Behaviour
            <select value={cfg?.newsTradingMode ?? "block_on_news"} onChange={(e) => setNews({ newsTradingMode: e.target.value })} style={{ width: 220 }}>
              <option value="block_on_news">Block on news (pause trading)</option>
              <option value="allow_always">Allow always (ignore news)</option>
              <option value="warn_only">Warn only</option>
            </select>
          </label>
        </div>
        {cfg?.newsCalendarSource && <p className="muted small" style={{ marginTop: 8 }}>Calendar source: <code>{cfg.newsCalendarSource}</code></p>}
      </div>

      <h2>Data location</h2>
      <div className="ticket">
        <p className="muted small">
          Downloaded bars, trained models, cache and the journal all live under <code>{cfg?.dataDir ?? "—"}</code>.
          Browse/open folders in <b>Files &amp; Storage</b>; download history + refresh broker costs in <b>Data</b>.
        </p>
      </div>

      <h2>Broker connection</h2>
      <div className="settings-grid">
        <div className="kv">
          <span>Configured</span>
          <b className={status?.configured ? "buy" : "sell"}>{status?.configured ? "yes" : "no"}</b>
        </div>
        <div className="kv">
          <span>Token stored</span>
          <b className={status?.hasToken ? "buy" : "sell"}>{status?.hasToken ? "yes" : "no"}</b>
        </div>
        <div className="kv">
          <span>Environment</span>
          <b>{status?.environment ?? "—"}</b>
        </div>
        <div className="kv">
          <span>Account</span>
          <b>{status?.accountId ?? "—"}</b>
        </div>
      </div>

      <h2>
        cTrader API credentials
        <button
          className="link"
          style={{ marginLeft: 10 }}
          onClick={() => {
            const next = !showCreds;
            setShowCreds(next);
            if (next && !creds) loadCreds();
          }}
        >
          {showCreds ? "hide" : "show"}
        </button>
      </h2>
      <p className="muted small">
        The app ships with a built-in cTrader Open API application. You only need this if your
        broker revokes it, or you want to use your own — otherwise leave it alone. Get the values
        from <code>connect.spotware.com</code> → your application. The secret is stored locally and
        is <b>never</b> shown back to you or sent anywhere but your own machine.
      </p>
      {showCreds && (
        <div className="ticket">
          {creds && (
            <p className="muted small" style={{ marginTop: 0 }}>
              Saved secret: <b>{creds.clientSecretConfigured ? creds.clientSecretMask : "none"}</b>
              {" · "}leave the field blank to keep it.
            </p>
          )}
          <div className="ticket-row" style={{ flexWrap: "wrap", gap: 14 }}>
            <label style={{ minWidth: 260 }}>
              Client ID
              <input
                type="text"
                autoComplete="off"
                spellCheck={false}
                value={credForm.clientId}
                onChange={(e) => setCredForm((f) => ({ ...f, clientId: e.target.value }))}
                style={{ width: 260 }}
              />
            </label>
            <label style={{ minWidth: 260 }}>
              Client secret
              <input
                type="password"
                autoComplete="off"
                spellCheck={false}
                placeholder={creds?.clientSecretConfigured ? "(unchanged)" : ""}
                value={credForm.clientSecret}
                onChange={(e) => setCredForm((f) => ({ ...f, clientSecret: e.target.value }))}
                style={{ width: 260 }}
              />
            </label>
            <label style={{ minWidth: 150 }}>
              Environment
              <select
                value={credForm.environment}
                onChange={(e) => setCredForm((f) => ({ ...f, environment: e.target.value }))}
                style={{ width: 150 }}
              >
                <option value="Demo">Demo (safe)</option>
                <option value="Live">Live (real money)</option>
              </select>
            </label>
            <label style={{ minWidth: 180 }}>
              Account id (optional)
              <input
                type="text"
                autoComplete="off"
                spellCheck={false}
                value={credForm.accountId}
                onChange={(e) => setCredForm((f) => ({ ...f, accountId: e.target.value }))}
                style={{ width: 180 }}
              />
            </label>
            <label style={{ minWidth: 260 }}>
              Redirect URI (leave blank for the default)
              <input
                type="text"
                autoComplete="off"
                spellCheck={false}
                value={credForm.redirectUri}
                onChange={(e) => setCredForm((f) => ({ ...f, redirectUri: e.target.value }))}
                style={{ width: 260 }}
              />
            </label>
          </div>
          <div className="btn-row">
            <button className="primary" disabled={credsBusy} onClick={saveCreds}>
              {credsBusy ? "Saving…" : "Save credentials"}
            </button>
            <span className="muted small">
              After saving, press <b>Authenticate cTrader</b> below once — the saved credentials are
              what the OAuth flow uses.
            </span>
          </div>
        </div>
      )}

      <div className="banner info">
        Authentication is <b>automatic</b>. You only authenticate <b>once</b> — after that the access
        token is silently refreshed via the stored refresh token on every launch and before it
        expires. You should never have to re-authenticate unless the broker revokes access.
      </div>

      <div className="btn-row">
        <button className="primary" onClick={doReauth} disabled={busy}>
          {busy ? "Working…" : status?.hasToken ? "Re-authenticate (only if revoked)" : "Authenticate cTrader (one time)"}
        </button>
        <button onClick={loadAccounts} disabled={busy}>
          List accounts
        </button>
      </div>

      {msg && <div className="banner info">{msg}</div>}

      {accounts.length > 0 && (
        <table className="tbl">
          <thead>
            <tr>
              <th>Type</th>
              <th>Account</th>
              <th>ID</th>
              <th>Login</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {accounts.map((a) => (
              <tr key={a.accountId}>
                <td>
                  <span className={`badge ${a.isLive ? "live" : "demo"}`}>
                    {a.isLive === null ? "?" : a.isLive ? "LIVE" : "DEMO"}
                  </span>
                </td>
                <td>{a.brokerTitle}{a.accountName ? ` · ${a.accountName}` : ""}</td>
                <td>{a.accountId}</td>
                <td>{a.login ?? "—"}</td>
                <td>
                  {a.enabled ? (
                    <span className="buy small">● Active</span>
                  ) : (
                    <button disabled={busy} onClick={() => useAccount(a)}>
                      Use
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
