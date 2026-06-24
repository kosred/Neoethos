import { useState } from "react";
import { pendingActions, confirmAction, rejectAction } from "../api";
import { usePoll } from "../hooks";

export default function Actions() {
  const { data, error, reload } = usePoll(pendingActions, 3000);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const items: any[] = Array.isArray(data) ? data : (data?.actions ?? data?.pending ?? []);

  const decide = async (id: string, ok: boolean) => {
    setBusy(true);
    setMsg(ok ? `Confirming ${id}…` : `Rejecting ${id}…`);
    try {
      await (ok ? confirmAction(id) : rejectAction(id));
      setMsg(`✓ ${ok ? "confirmed" : "rejected"} ${id}`);
      await reload();
    } catch (e) {
      setMsg(`Failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Actions {items.length > 0 && <span className="badge live">{items.length} pending</span>}</h1>
      <p className="sub">Trade-management actions awaiting your approval</p>
      {error && <div className="banner warn">{error}</div>}
      {msg && <div className="banner info">{msg}</div>}

      {items.length === 0 ? (
        <p className="muted">No pending actions — the engine has nothing waiting for approval.</p>
      ) : (
        <div className="news-list">
          {items.map((a, i) => {
            const id = String(a.id ?? a.actionId ?? i);
            return (
              <div className="news-item" key={id}>
                <div className="news-title">{a.kind ?? a.type ?? a.action ?? "Action"} — {a.symbol ?? ""}</div>
                <div className="muted small" style={{ whiteSpace: "pre-wrap" }}>
                  {a.summary ?? a.description ?? JSON.stringify(a)}
                </div>
                <div className="btn-row">
                  <button className="primary" disabled={busy} onClick={() => decide(id, true)}>Confirm</button>
                  <button className="danger" disabled={busy} onClick={() => decide(id, false)}>Reject</button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
