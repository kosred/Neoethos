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
  type RiskInfo,
  type SettingsUpdate,
  type SettingsView,
} from "../api";

export default function Settings() {
  const [status, setStatus] = useState<BrokerStatus | null>(null);
  const [accounts, setAccounts] = useState<AccountInfo[]>([]);
  const [cfg, setCfg] = useState<SettingsView | null>(null);
  const [presets, setPresets] = useState<{ id: string; displayName: string }[]>([]);
  const [risk, setRisk] = useState<RiskInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadErrors, setLoadErrors] = useState<string[]>([]);

  const refresh = async () => {
    setLoading(true);
    const [brokerResult, configResult, riskResult] = await Promise.allSettled([
      brokerStatus(), getSettings(), riskInfo(),
    ]);
    const failures: string[] = [];
    if (brokerResult.status === "fulfilled") setStatus(brokerResult.value);
    else { setStatus(null); failures.push(`Broker status: ${brokerResult.reason}`); }
    if (configResult.status === "fulfilled") setCfg(configResult.value);
    else { setCfg(null); failures.push(`Settings: ${configResult.reason}`); }
    if (riskResult.status === "fulfilled") {
      setRisk(riskResult.value);
      setPresets(riskResult.value.availablePresets);
    } else {
      setRisk(null);
      setPresets([]);
      failures.push(`Risk configuration: ${riskResult.reason}`);
    }
    setLoadErrors(failures);
    setLoading(false);
  };

  const savePatch = async (patch: SettingsUpdate, label: string) => {
    if (busy || loading || !cfg) return;
    setBusy(true);
    setMsg(`Saving ${label}…`);
    try {
      await updateSettings(patch);
      setMsg(`✓ ${label} saved.`);
      await refresh();
    } catch (e) {
      setMsg(`${label} save failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };
  const setCompute = (mode: "auto" | "cpu" | "gpu") => savePatch({ computeMode: mode }, `Training compute mode ${mode}`);
  const setNews = (patch: SettingsUpdate) => savePatch(patch, "News settings");
  const setLoop = (patch: SettingsUpdate) => savePatch(patch, "Autopilot-loop settings");

  const applyPreset = async (id: string) => {
    if (busy || loading || !risk) return;
    setBusy(true);
    setMsg(`Applying risk preset ${id}…`);
    try {
      await setRiskPreset(id);
      setRisk(await riskInfo());
      setMsg(`✓ Risk preset = ${id}.`);
    } catch (e) {
      setMsg(`Preset failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, []);

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
  const [credsLoading, setCredsLoading] = useState(false);
  const [showCreds, setShowCreds] = useState(false);
  const loadCreds = async () => {
    setCredsLoading(true);
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
    } finally {
      setCredsLoading(false);
    }
  };
  const saveCreds = async () => {
    if (credsBusy || credsLoading || !creds) return;
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
          `The app can attempt token refresh; broker access must still be verified.`,
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

  const activateAccount = async (a: AccountInfo) => {
    setBusy(true);
    setMsg(`Switching to ${a.label}…`);
    try {
      const s = await selectAccount(a.accountId, a.isLive === true, a.label);
      setStatus(s);
      await loadAccounts();
      setMsg(
        `✓ Active account: ${a.label} — environment set to ${a.isLive ? "Live" : "Demo"}. ` +
          `Balance and positions refresh in Trading → Market & positions.`,
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
      <p className="sub">Broker connection, training hardware and live safeguards</p>
      <div className="btn-row"><button disabled={busy || loading || credsBusy} onClick={() => void refresh()}>{loading ? "Loading settings…" : "Refresh settings"}</button></div>
      {loadErrors.map((error) => <div className="banner warn" role="alert" key={error}>{error}</div>)}
      {msg && <div className="banner info" role="status">{msg}</div>}

      {/* This control writes system.enable_gpu_preference, which gates
          TRAINING. The discovery SEARCH device is models.prop_search_device,
          and backend.rs:126-130 lets it REPLACE the global whenever it is
          non-empty — both shipped config files set it. So the old copy
          ("forces the CPU lane") and the old "Active:" line were both false on
          any box with that key set: press CPU and the search still ran on the
          card while this screen reported cpu.
          The refuters established these are two axes and must not be merged
          (cpu training + gpu search is the A6000 configuration on record), so
          the fix is to stop this control from claiming the other axis. */}
      <h2>Compute <span className="muted small">(training device)</span></h2>
      <p className="muted small">
        Saved training-device preference. Availability, memory admission and the selected model's
        implemented backend determine whether a job can run; this setting is not proof of GPU execution.
      </p>
      <div className="ticket">
        <div className="seg" style={{ maxWidth: 360 }}>
          {(["auto", "cpu", "gpu"] as const).map((m) => (
            <button key={m} disabled={busy || loading || !cfg} className={cfg?.computeMode === m ? "on" : ""} onClick={() => setCompute(m)}>{m.toUpperCase()}</button>
          ))}
        </div>
        {cfg && <p className="muted small" style={{ marginTop: 8 }}>Saved: <b>{cfg.computeMode ?? "?"}</b> <span className="muted">(training)</span></p>}
        <p className="muted small">
          ⚠ This does <b>not</b> set the discovery-search device. The search reads{" "}
          <code>models.prop_search_device</code>, which overrides this value whenever it is set —
          choosing CPU here can still give you a CUDA search. Set the search device beside the
          search budget in <b>Research → Strategy search</b>; the run receipt must report what was
          actually admitted.
        </p>
      </div>

      <h2>Live risk &amp; sizing</h2>
      <p className="muted small">Position-sizing limits and account drawdown guards for automated trading. Search-time risk, objectives and validation windows now live in <b>Research → Strategy search</b>.</p>
      <div className="ticket">
        {presets.length > 0 && (
          <label>Preset
            <select disabled={busy || loading || !risk} value={risk?.preset ?? ""} onChange={(e) => applyPreset(e.target.value)} style={{ width: 240 }}>
              {!presets.some((p) => p.id === risk?.preset) && <option value="">{risk?.preset ?? "(current)"}</option>}
              {presets.map((p) => <option key={p.id} value={p.id}>{p.displayName}</option>)}
            </select>
          </label>
        )}
        {/* Sizing calibration belongs to selection/fitting, not the reserved
            final test. The fixed preset fraction is not the Risky estimate. */}
        {risk && cfg?.tradingMode === "risky" && (
          <div className="banner warn" style={{ marginTop: 12 }}>
            <b>Trading mode is RISKY — the “risk / trade” number below is not what the bot risks.</b>{" "}
            Risky live sizing uses the selected portfolio's selection-validation calibration,
            subject to configured risk ceilings and broker lot limits. This is not the independent
            final OOS test. The half-Kelly estimate is a sizing heuristic, not a guarantee of
            portfolio safety or profit. The fixed
            <code> risk.risk_per_trade </code> preset below is not that estimate. Search-time sizing
            is configured in <b>Research → Strategy search</b>.
          </div>
        )}
        {risk && (
          <div className="cards" style={{ marginTop: 12, gridTemplateColumns: "repeat(4,1fr)" }}>
            <div className="card">
              <div className="card-label">RISK / TRADE{cfg?.tradingMode === "risky" ? " (NOT IN FORCE)" : ""}</div>
              <div className="card-value" style={cfg?.tradingMode === "risky" ? { opacity: 0.45 } : undefined}>
                {risk.riskPerTrade == null ? "—" : `${(risk.riskPerTrade * 100).toFixed(2)}%`}
              </div>
            </div>
            <div className="card"><div className="card-label">DAILY DD CAP</div><div className="card-value">{risk.dailyDrawdownLimit == null ? "—" : `${(risk.dailyDrawdownLimit * 100).toFixed(1)}%`}</div></div>
            <div className="card"><div className="card-label">TOTAL DD CAP</div><div className="card-value">{risk.totalDrawdownLimit == null ? "—" : `${(risk.totalDrawdownLimit * 100).toFixed(1)}%`}</div></div>
            <div className="card"><div className="card-label">MAX LOT</div><div className="card-value">{risk.maxLotSize ?? "—"}</div></div>
          </div>
        )}
        {/* This used to read "Manual orders (Positions) are not gated by these".
            False since 2026-08-09: orders.rs:116-137 refuses a manual order
            with 400 when require_stop_loss is on. A promise that manual is
            unconstrained, on a screen the operator checks before placing one
            by hand, is a promise that produces a rejected order at the worst
            possible moment. */}
        <p className="muted small" style={{ marginTop: 8 }}>
          These preset sizing and drawdown limits apply to <b>automated</b> trading.
          Manual orders require a stop-loss when <b>Require stop-loss</b> is ON. Current setting:{" "}
          {risk ? <b className={risk.requireStopLoss ? "sell" : ""}>{risk.requireStopLoss ? "ON" : "off"}</b> : "(unknown)"}
          . When OFF, new manual orders still require SL or TP unless submitted with the explicit
          risky override. Broker order constraints still apply. Set <code>risk.require_stop_loss</code>
          in <b>Settings → Advanced → raw config.yaml</b>.
        </p>
      </div>

      <h2>Autopilot loop</h2>
      <p className="muted small">What happens automatically when auto-cull permanently retires a losing strategy.</p>
      <div className="ticket">
        <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            disabled={busy || loading || !cfg}
            checked={cfg?.autoRediscoverOnCull ?? false}
            onChange={(e) => setLoop({ autoRediscoverOnCull: e.target.checked })}
          />
          Auto-rediscover after a cull — when a strategy is retired (blacklisted forever), automatically start a fresh Discovery on the same symbol + timeframe to refill the gap. Runs when the Discovery engine is idle.
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 8, marginTop: 8 }}>
          <input
            type="checkbox"
            disabled={busy || loading || !cfg}
            checked={!!cfg?.liveMlGate}
            onChange={(e) => setLoop({ liveMlGate: e.target.checked })}
          />
          <span>
            <b>Live ML gate</b> — the trained model ensemble scales each live entry's risk
            (agreement × regime × anomaly). Strategies still pick the direction; the models can
            only <b>shrink</b> size or skip a bar on a hard regime/anomaly collapse — never flip
            a trade, never create one. When enabled, it requires the selected portfolio's exact
            trained candidate and matching saved combined research with the current inference settings.
            Missing or mismatched models refuse engine startup; there is no automatic gene-only fallback.
            When disabled, entries use strategy genes without model gating. All other trading checks
            still apply. Takes effect on the next engine start.
          </span>
        </label>
      </div>

      <h2>News gate</h2>
      <p className="muted small">How automated trading behaves around high-impact news events.</p>
      <div className="ticket">
        <div className="ticket-row" style={{ flexWrap: "wrap", gap: 18 }}>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
            <input type="checkbox" disabled={busy || loading || !cfg} checked={!!cfg?.newsCalendarEnabled} onChange={(e) => setNews({ newsCalendarEnabled: e.target.checked })} />
            Economic calendar enabled
          </label>
          <label>Behaviour
            <select
              disabled={busy || loading || !cfg}
              value={cfg?.newsTradingMode ?? "block_on_news"}
              onChange={(event) => setNews({
                newsTradingMode: event.target.value as SettingsUpdate["newsTradingMode"],
              })}
              style={{ width: 220 }}
            >
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
          Configured historical-data directory: <code>{cfg?.dataDir ?? "—"}</code>.
          Models, cache and logs can use separate directories. Check the resolved locations in
          <b> Settings → Storage</b>; download history and refresh broker costs in <b>Data</b>.
        </p>
      </div>

      <h2>Broker connection</h2>
      <div className="settings-grid">
        <div className="kv">
          <span>Configured</span>
          <b className={status ? status.configured ? "buy" : "sell" : "muted"}>{status ? status.configured ? "yes" : "no" : "unknown"}</b>
        </div>
        <div className="kv">
          <span>Token stored</span>
          <b className={status ? status.hasToken ? "buy" : "sell" : "muted"}>{status ? status.hasToken ? "yes" : "no" : "unknown"}</b>
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
        is not returned by the read API. It is used to authenticate with cTrader.
      </p>
      {showCreds && (
        <div className="ticket">
          {creds && (
            <p className="muted small" style={{ marginTop: 0 }}>
              Saved secret: <b>{creds.clientSecretConfigured ? creds.clientSecretMask : "none"}</b>
              {" · "}leave the field blank to keep it.
            </p>
          )}
          <fieldset className="ticket-row" disabled={credsBusy || credsLoading || !creds} style={{ flexWrap: "wrap", gap: 14, border: 0, padding: 0, margin: 0 }}>
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
          </fieldset>
          <div className="btn-row">
            <button className="primary" disabled={credsBusy || credsLoading || !creds} onClick={saveCreds}>
              {credsBusy ? "Saving…" : "Save credentials"}
            </button>
            {!creds && <button disabled={credsLoading} onClick={() => void loadCreds()}>{credsLoading ? "Loading credentials…" : "Retry loading credentials"}</button>}
            <span className="muted small">
              After saving, press <b>Authenticate cTrader</b> below once — the saved credentials are
              what the OAuth flow uses.
            </span>
          </div>
        </div>
      )}

      <div className="banner info">
        The app can refresh access using a stored refresh token. Saved credentials do not prove an
        active broker session; verify current market and account timestamps. Re-authentication may
        be necessary when refresh fails or access is revoked.
      </div>

      <div className="btn-row">
        <button className="primary" onClick={doReauth} disabled={busy}>
          {busy ? "Working…" : status?.hasToken ? "Re-authenticate (only if revoked)" : "Authenticate cTrader (one time)"}
        </button>
        <button onClick={loadAccounts} disabled={busy}>
          List accounts
        </button>
      </div>

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
                    <button disabled={busy} onClick={() => activateAccount(a)}>
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
