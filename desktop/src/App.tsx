import { useEffect, useState, type ReactNode } from "react";
import { appInfo, brokerStatus, type AppInfo, type BrokerStatus } from "./api";
import Cockpit from "./screens/Cockpit";
import Dashboard from "./screens/Dashboard";
import Markets from "./screens/Markets";
import MarketWatch from "./screens/MarketWatch";
import Positions from "./screens/Positions";
import Account from "./screens/Account";
import Actions from "./screens/Actions";
import Autopilot from "./screens/Autopilot";
import RiskyMode from "./screens/RiskyMode";
import Risk from "./screens/Risk";
import Discovery from "./screens/Discovery";
import Training from "./screens/Training";
import StrategyLab from "./screens/StrategyLab";
import Intelligence from "./screens/Intelligence";
import Files from "./screens/Files";
import Data from "./screens/Data";
import Journal from "./screens/Journal";
import News from "./screens/News";
import AiDesk from "./screens/AiDesk";
import Hardware from "./screens/Hardware";
import Advanced from "./screens/Advanced";
import Settings from "./screens/Settings";
import "./App.css";

type View =
  | "cockpit" | "dashboard" | "markets" | "marketwatch" | "positions" | "account" | "actions"
  | "autopilot" | "riskymode" | "risk" | "discovery" | "training" | "strategylab" | "intelligence"
  | "files" | "data" | "journal" | "news" | "aidesk" | "hardware" | "advanced" | "settings";

type NavEntry = { id: View; label: string; icon: string } | { divider: string };

const NAV: NavEntry[] = [
  { divider: "Trade" },
  { id: "cockpit", label: "Trade", icon: "🎯" },
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "markets", label: "Markets", icon: "📈" },
  { id: "marketwatch", label: "Market Watch", icon: "👁" },
  { id: "positions", label: "Positions", icon: "≡" },
  { id: "account", label: "Account", icon: "💳" },
  { id: "actions", label: "Actions", icon: "✓" },
  { divider: "Autopilot" },
  { id: "autopilot", label: "Autopilot", icon: "🤖" },
  { id: "riskymode", label: "Risky Mode", icon: "🚀" },
  { id: "risk", label: "Risk", icon: "🛡" },
  { divider: "Research" },
  { id: "discovery", label: "Discovery", icon: "🧬" },
  { id: "training", label: "Training", icon: "🎓" },
  { id: "strategylab", label: "Strategy Lab", icon: "⚗" },
  { id: "intelligence", label: "Intelligence", icon: "🧠" },
  { divider: "Data & Files" },
  { id: "files", label: "Files & Storage", icon: "🗂" },
  { id: "data", label: "Data", icon: "🗄" },
  { divider: "Desk" },
  { id: "journal", label: "Journal", icon: "📒" },
  { id: "news", label: "News", icon: "📰" },
  { id: "aidesk", label: "AI Desk", icon: "💬" },
  { divider: "System" },
  { id: "hardware", label: "Hardware", icon: "🖥" },
  { id: "advanced", label: "Advanced", icon: "🔧" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

const SCREENS: Record<View, ReactNode> = {
  cockpit: <Cockpit />,
  dashboard: <Dashboard />,
  markets: <Markets />,
  marketwatch: <MarketWatch />,
  positions: <Positions />,
  account: <Account />,
  actions: <Actions />,
  autopilot: <Autopilot />,
  riskymode: <RiskyMode />,
  risk: <Risk />,
  discovery: <Discovery />,
  training: <Training />,
  strategylab: <StrategyLab />,
  intelligence: <Intelligence />,
  files: <Files />,
  data: <Data />,
  journal: <Journal />,
  news: <News />,
  aidesk: <AiDesk />,
  hardware: <Hardware />,
  advanced: <Advanced />,
  settings: <Settings />,
};

export default function App() {
  const [view, setView] = useState<View>("cockpit");
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
