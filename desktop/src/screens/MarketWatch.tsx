import { useEffect, useMemo, useState } from "react";
import { getWatchlist, setWatchlist, serverSymbols, type BrokerSymbol } from "../api";
import { usePoll, useSpotStream } from "../hooks";

export default function MarketWatch() {
  const { data: universe } = usePoll(serverSymbols, 0);
  const { ticks, connected } = useSpotStream();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [filter, setFilter] = useState("");

  useEffect(() => {
    getWatchlist()
      .then((w) => {
        const syms: string[] = Array.isArray(w) ? w : (w?.symbols ?? []);
        setSelected(new Set(syms.map((s) => s.toUpperCase())));
      })
      .catch((e) => setMsg(String(e)));
  }, []);

  const groups = useMemo(() => {
    const all = universe?.symbols ?? [];
    const f = filter.trim().toUpperCase();
    const g: Record<string, BrokerSymbol[]> = {};
    for (const s of all) {
      if (f && !s.symbolName.toUpperCase().includes(f)) continue;
      const k = s.assetClass || "Other";
      (g[k] ??= []).push(s);
    }
    for (const k of Object.keys(g)) g[k].sort((a, b) => a.symbolName.localeCompare(b.symbolName));
    return g;
  }, [universe, filter]);

  const toggle = (name: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  const addGroup = (syms: BrokerSymbol[]) =>
    setSelected((prev) => new Set([...prev, ...syms.map((s) => s.symbolName)]));

  const save = async () => {
    setBusy(true);
    setMsg(`Subscribing ${selected.size} symbols…`);
    try {
      await setWatchlist([...selected]);
      setMsg(`✓ Subscribed ${selected.size} symbols — live ticks within ~5s.`);
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const total = universe?.symbols.length ?? 0;
  const liveRows = Object.values(ticks).sort((a, b) => a.symbolName.localeCompare(b.symbolName));

  return (
    <div className="screen">
      <h1>Market Watch <span className={`stream-pill ${connected ? "on" : ""}`}>{connected ? "● LIVE" : "○ offline"}</span></h1>
      <p className="sub">Subscribe any of the broker's {total} symbols (forex · metals · indices) — drives the live stream</p>

      <div className="btn-row">
        <input placeholder="filter…" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ width: 160 }} />
        <span className="muted small">{selected.size} selected</span>
        <button className="primary" disabled={busy} onClick={save}>Subscribe selected</button>
        <button disabled={busy} onClick={() => setSelected(new Set())}>Clear</button>
      </div>
      {msg && <div className="banner info">{msg}</div>}

      {Object.keys(groups).sort().map((cls) => (
        <details key={cls} className="knob-section" open={total <= 40}>
          <summary>
            {cls} ({groups[cls].length})
            <button
              className="small"
              style={{ marginLeft: 10, padding: "1px 8px" }}
              onClick={(e) => { e.preventDefault(); addGroup(groups[cls]); }}
            >+ all</button>
          </summary>
          <div className="sym-grid">
            {groups[cls].map((s) => {
              const t = ticks[s.symbolName];
              const on = selected.has(s.symbolName);
              return (
                <label key={s.symbolId} className={`sym-cell ${on ? "on" : ""}`} title={s.description ?? ""}>
                  <input type="checkbox" checked={on} onChange={() => toggle(s.symbolName)} />
                  <b>{s.symbolName}</b>
                  <span className="muted small">{t ? t.midPrice?.toFixed(5) : ""}</span>
                </label>
              );
            })}
          </div>
        </details>
      ))}

      <h2>Live prices ({liveRows.length} streaming)</h2>
      {liveRows.length === 0 ? (
        <p className="muted">No live ticks yet — select symbols above and Subscribe.</p>
      ) : (
        <table className="tbl">
          <thead><tr><th>Symbol</th><th>Bid</th><th>Ask</th><th>Mid</th><th>Age</th></tr></thead>
          <tbody>
            {liveRows.map((r) => (
              <tr key={r.symbolId}>
                <td><b>{r.symbolName}</b></td>
                <td>{r.bid ?? "—"}</td>
                <td>{r.ask ?? "—"}</td>
                <td>{r.midPrice ?? "—"}</td>
                <td className="muted small">{r.freshnessSeconds?.toFixed(1)}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
