import { useEffect, useState } from "react";
import { appInfo, brokerStatus, type AppInfo, type BrokerStatus } from "./api";
import Dashboard from "./screens/Dashboard";
import Markets from "./screens/Markets";
import Positions from "./screens/Positions";
import Settings from "./screens/Settings";
import "./App.css";

type View = "dashboard" | "markets" | "positions" | "settings";

const NAV: { id: View; label: string; icon: string }[] = [
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "markets", label: "Markets", icon: "📈" },
  { id: "positions", label: "Positions", icon: "≡" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

export default function App() {
  const [view, setView] = useState<View>("dashboard");
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [status, setStatus] = useState<BrokerStatus | null>(null);

  useEffect(() => {
    appInfo().then(setInfo).catch(() => {});
    const tick = () => brokerStatus().then(setStatus).catch(() => {});
    tick();
    const id = setInterval(tick, 5000);
    return () => clearInterval(id);
  }, []);

  const brokerLabel = !status
    ? "…"
    : !status.configured
      ? "not configured"
      : status.hasToken
        ? `${status.environment} · ready`
        : `${status.environment} · needs auth`;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="logo">
          NeoEthos <span className="pill">TAURI</span>
        </div>
        <nav>
          {NAV.map((n) => (
            <button
              key={n.id}
              className={`nav-item${view === n.id ? " active" : ""}`}
              onClick={() => setView(n.id)}
            >
              <span className="nav-icon">{n.icon}</span>
              {n.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className={`dot ${status?.hasToken ? "ok" : "off"}`} />
          cTrader: {brokerLabel}
        </div>
      </aside>

      <div className="main">
        <div className="content">
          {view === "dashboard" && <Dashboard />}
          {view === "markets" && <Markets />}
          {view === "positions" && <Positions />}
          {view === "settings" && <Settings />}
        </div>
        <footer className="statusbar">
          <span>cTrader · {brokerLabel}</span>
          <span className="spacer" />
          <span className="muted">{info?.data_root ?? ""}</span>
          <span className="ver">v{info?.version ?? "…"}</span>
        </footer>
      </div>
    </div>
  );
}
