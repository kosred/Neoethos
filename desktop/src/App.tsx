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
import Hardware from "./screens/Hardware";
import Configuration from "./screens/Configuration";
import Help from "./screens/Help";
import "./App.css";

type View =
  | "help"
  | "cockpit" | "dashboard" | "markets" | "marketwatch" | "account" | "actions"
  | "autopilot" | "riskymode" | "risk" | "discovery" | "training" | "strategylab" | "strategyreport" | "intelligence"
  | "files" | "data" | "news" | "aidesk" | "hardware" | "settings";

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
  // AI Desk = the ONE LLM surface: unified chat (Assistant ↔ Supervisor
  // modes) + the supervisor control panel (2026-07-11 consolidation).
  { id: "aidesk", label: "AI Desk", icon: "💬" },
  { divider: "System" },
  { id: "hardware", label: "Hardware", icon: "🖥" },
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
  hardware: <Hardware />,
  settings: <Configuration />,
};

export default function App() {
  const [view, setView] = useState<View>("cockpit");
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [status, setStatus] = useState<BrokerStatus | null>(null);

  // A focused <input type="number"> changes its VALUE when the mouse wheel
  // passes over it. Scrolling a settings page therefore silently rewrites the
  // knobs under the cursor — the operator found risk-per-trade, drawdown caps
  // and population sitting at NEGATIVE numbers after nothing but scrolling.
  // On a trading system that is not a cosmetic bug. Blurring on wheel keeps
  // the page scrolling normally while making the value untouchable.
  useEffect(() => {
    const onWheel = (e: WheelEvent) => {
      const el = e.target as HTMLElement | null;
      if (
        el instanceof HTMLInputElement &&
        el.type === "number" &&
        document.activeElement === el
      ) {
        el.blur();
      }
    };
    document.addEventListener("wheel", onWheel, { passive: true });
    return () => document.removeEventListener("wheel", onWheel);
  }, []);

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
