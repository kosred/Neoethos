import { useEffect, useState } from "react";
import {
  brokerStatus,
  brokerAccounts,
  reauthBroker,
  type BrokerStatus,
  type AccountInfo,
} from "../api";

export default function Settings() {
  const [status, setStatus] = useState<BrokerStatus | null>(null);
  const [accounts, setAccounts] = useState<AccountInfo[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const refresh = async () => {
    try {
      setStatus(await brokerStatus());
    } catch (e) {
      setMsg(String(e));
    }
  };

  useEffect(() => {
    refresh();
  }, []);

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

  return (
    <div className="screen">
      <h1>Settings</h1>
      <p className="sub">Broker connection</p>

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
              <th>Account ID</th>
              <th>Broker</th>
              <th>Name</th>
              <th>Type</th>
            </tr>
          </thead>
          <tbody>
            {accounts.map((a) => (
              <tr key={a.accountId}>
                <td>{a.accountId}</td>
                <td>{a.brokerTitle}</td>
                <td>{a.accountName}</td>
                <td>{a.isLive === null ? "—" : a.isLive ? "LIVE" : "DEMO"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
