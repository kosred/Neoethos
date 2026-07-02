import { useEffect, useState, type ReactNode } from "react";
import { appInfo, brokerStatus, type AppInfo, type BrokerStatus } from "./api";
import Cockpit from "./screens/Cockpit";
import Dashboard from "./screens/Dashboard";
import Markets from "./screens/Markets";
import MarketWatch from "./screens/MarketWatch";
import Account from "./screens/Account";
import Actions from "./screens/Actions";
import Autopilot from "./screens/Autopilot";
import RiskyMode from "./screens/RiskyMode";
import Risk from "./screens/Risk";
import Discovery from "./screens/Discovery";
import Training from "./screens/Training";
import StrategyLab from "./screens/StrategyLab";
import StrategyReport from "./screens/StrategyReport";
import Intelligence from "./screens/Intelligence";
import Files from "./screens/Files";
import Data from "./screens/Data";
import News from "./screens/News";
import AiDesk from "./screens/AiDesk";
import Supervisor from "./screens/Supervisor";
import Hardware from "./screens/Hardware";
import Advanced from "./screens/Advanced";
import Settings from "./screens/Settings";
import Help from "./screens/Help";
import "./App.css";

type View =
  | "help"
  | "cockpit" | "dashboard" | "markets" | "marketwatch" | "account" | "actions"
  | "autopilot" | "riskymode" | "risk" | "discovery" | "training" | "strategylab" | "strategyreport" | "intelligence"
  | "files" | "data" | "news" | "aidesk" | "supervisor" | "hardware" | "advanced" | "settings";

type NavEntry = { id: View; label: string; icon: string } | { divider: string };

const NAV: NavEntry[] = [
  { id: "help", label: "Help & Guide", icon: "❓" },
  { divider: "Trade" },
  { id: "cockpit", label: "Trade", icon: "🎯" },
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "markets", label: "Markets", icon: "📈" },
  { id: "marketwatch", label: "Market Watch", icon: "👁" },
  { id: "account", label: "Account & Journal", icon: "💳" },
  { id: "actions", label: "Actions", icon: "✓" },
  { divider: "Autopilot" },
  { id: "autopilot", label: "Autopilot", icon: "🤖" },
  { id: "riskymode", label: "Risky Mode", icon: "🚀" },
  { id: "risk", label: "Risk", icon: "🛡" },
  { divider: "Research" },
  { id: "discovery", label: "Discovery", icon: "🧬" },
  { id: "training", label: "Training", icon: "🎓" },
  { id: "strategylab", label: "Strategy Lab", icon: "⚗" },
  { id: "strategyreport", label: "Strategy Report", icon: "📅" },
  { id: "intelligence", label: "Intelligence", icon: "🧠" },
  { divider: "Data & Files" },
  { id: "files", label: "Files & Storage", icon: "🗂" },
  { id: "data", label: "Data", icon: "🗄" },
  { divider: "Desk" },
  { id: "news", label: "News", icon: "📰" },
  { id: "aidesk", label: "AI Desk", icon: "💬" },
  { id: "supervisor", label: "Supervisor", icon: "🧭" },
  { divider: "System" },
  { id: "hardware", label: "Hardware", icon: "🖥" },
  { id: "advanced", label: "Advanced", icon: "🔧" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

const SCREENS: Record<View, ReactNode> = {
  help: <Help />,
  cockpit: <Cockpit />,
  dashboard: <Dashboard />,
  markets: <Markets />,
  marketwatch: <MarketWatch />,
  account: <Account />,
  actions: <Actions />,
  autopilot: <Autopilot />,
  riskymode: <RiskyMode />,
  risk: <Risk />,
  discovery: <Discovery />,
  training: <Training />,
  strategylab: <StrategyLab />,
  strategyreport: <StrategyReport />,
  intelligence: <Intelligence />,
  files: <Files />,
  data: <Data />,
  news: <News />,
  aidesk: <AiDesk />,
  supervisor: <Supervisor />,
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
