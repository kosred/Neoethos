import { useEffect, useState, type ReactNode } from "react";
import { appInfo, brokerStatus, type AppInfo, type BrokerStatus } from "./api";
import Dashboard from "./screens/Dashboard";
import Markets from "./screens/Markets";
import Positions from "./screens/Positions";
import Discovery from "./screens/Discovery";
import Training from "./screens/Training";
import StrategyLab from "./screens/StrategyLab";
import Autonomous from "./screens/Autonomous";
import Risk from "./screens/Risk";
import Intelligence from "./screens/Intelligence";
import Journal from "./screens/Journal";
import MarketWatch from "./screens/MarketWatch";
import News from "./screens/News";
import Data from "./screens/Data";
import Hardware from "./screens/Hardware";
import AiDesk from "./screens/AiDesk";
import Settings from "./screens/Settings";
import "./App.css";

type View =
  | "dashboard" | "markets" | "marketwatch" | "positions" | "discovery" | "training"
  | "strategylab" | "autonomous" | "risk" | "intelligence" | "journal" | "news"
  | "data" | "hardware" | "aidesk" | "settings";

type NavEntry = { id: View; label: string; icon: string } | { divider: string };

const NAV: NavEntry[] = [
  { divider: "Trading" },
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "markets", label: "Markets", icon: "📈" },
  { id: "marketwatch", label: "Market Watch", icon: "👁" },
  { id: "positions", label: "Positions", icon: "≡" },
  { id: "autonomous", label: "Autonomous", icon: "🤖" },
  { divider: "Research" },
  { id: "discovery", label: "Discovery", icon: "🧬" },
  { id: "training", label: "Training", icon: "🎓" },
  { id: "strategylab", label: "Strategy Lab", icon: "⚗" },
  { id: "intelligence", label: "Intelligence", icon: "🧠" },
  { divider: "Insight" },
  { id: "journal", label: "Journal", icon: "📒" },
  { id: "news", label: "News", icon: "📰" },
  { id: "aidesk", label: "AI Desk", icon: "💬" },
  { id: "risk", label: "Risk", icon: "🛡" },
  { divider: "System" },
  { id: "data", label: "Data", icon: "🗄" },
  { id: "hardware", label: "Hardware", icon: "🖥" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

const SCREENS: Record<View, ReactNode> = {
  dashboard: <Dashboard />,
  markets: <Markets />,
  marketwatch: <MarketWatch />,
  positions: <Positions />,
  autonomous: <Autonomous />,
  discovery: <Discovery />,
  training: <Training />,
  strategylab: <StrategyLab />,
  intelligence: <Intelligence />,
  journal: <Journal />,
  news: <News />,
  aidesk: <AiDesk />,
  risk: <Risk />,
  data: <Data />,
  hardware: <Hardware />,
  settings: <Settings />,
};

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
          {NAV.map((n, i) =>
            "divider" in n ? (
              <div className="nav-divider" key={`d${i}`}>{n.divider}</div>
            ) : (
              <button
                key={n.id}
                className={`nav-item${view === n.id ? " active" : ""}`}
                onClick={() => setView(n.id)}
              >
                <span className="nav-icon">{n.icon}</span>
                {n.label}
              </button>
            ),
          )}
        </nav>
        <div className="sidebar-foot">
          <div className={`dot ${status?.hasToken ? "ok" : "off"}`} />
          cTrader: {brokerLabel}
        </div>
      </aside>

      <div className="main">
        <div className="content">{SCREENS[view]}</div>
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
