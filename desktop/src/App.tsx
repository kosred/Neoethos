import {
  Suspense,
  lazy,
  memo,
  useCallback,
  useMemo,
  useEffect,
  useState,
  type ComponentType,
  type LazyExoticComponent,
} from "react";
import { apiGet, appInfo, brokerStatus } from "./api";
import { brokerConnectionView, type BrokerConnectionObservation } from "./runtimeStatus";
import { usePoll } from "./hooks";
import { brokerUiAccess } from "./brokerUi";
import { BrokerUiContext } from "./brokerUiContext";
import { ScreenBoundary } from "./components/ScreenBoundary";
import "./App.css";

const AutomationWorkspace = lazy(() => import("./screens/AutomationWorkspace"));
const Data = lazy(() => import("./screens/Data"));
const ResearchWorkspace = lazy(() => import("./screens/ResearchWorkspace"));
const SettingsWorkspace = lazy(() => import("./screens/SettingsWorkspace"));
const TradingWorkspace = lazy(() => import("./screens/TradingWorkspace"));

type View = "trading" | "research" | "automation" | "data" | "settings";

type NavEntry = Readonly<{
  id: View;
  label: string;
  eyebrow: string;
  component: ComponentType | LazyExoticComponent<ComponentType>;
}>;

const NAV: readonly NavEntry[] = [
  { id: "trading", label: "Trading", eyebrow: "01", component: TradingWorkspace },
  { id: "research", label: "Research", eyebrow: "02", component: ResearchWorkspace },
  { id: "automation", label: "Automation", eyebrow: "03", component: AutomationWorkspace },
  { id: "data", label: "Data", eyebrow: "04", component: Data },
  { id: "settings", label: "Settings", eyebrow: "05", component: SettingsWorkspace },
];

const brokerObservation = () => apiGet<BrokerConnectionObservation>("/broker/status");

// The connection clock must not re-render the chart/workspace every second.
// NAV entries have stable identity; workspace-owned polls still update normally.
const WorkspacePane = memo(function WorkspacePane({ active }: { active: NavEntry }) {
  const ActiveWorkspace = active.component;
  return (
    <ScreenBoundary key={active.id} label={active.label}>
      <Suspense fallback={<div className="workspace-loading">Loading workspace…</div>}>
        <ActiveWorkspace />
      </Suspense>
    </ScreenBoundary>
  );
});

export default function App() {
  const [view, setView] = useState<View>("trading");
  const [nowMs, setNowMs] = useState(() => Date.now());
  const { data: info, error: infoError } = usePoll(appInfo);
  const { data: status, error: statusError } = usePoll(brokerStatus, 5000);
  const { data: connection, error: connectionError } = usePoll(async () => {
    const observation = await brokerObservation();
    // A new response may arrive after the last clock tick. Compare it with
    // actual current time, not a one-second-old value that looks "future".
    setNowMs(Date.now());
    return observation;
  }, 5000);

  useEffect(() => {
    const interval = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(interval);
  }, []);

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

  const connectionView = brokerConnectionView(status, connection, nowMs, statusError, connectionError);
  const brokerLabel = connectionView.label;
  const active = NAV.find((entry) => entry.id === view) ?? NAV[0];
  const openBrokerSetup = useCallback(() => setView("settings"), []);
  const brokerUi = useMemo(() => ({
    access: brokerUiAccess(status, statusError),
    openSetup: openBrokerSetup,
  }), [status, statusError, openBrokerSetup]);

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="logo">
          <span className="logo-mark">NE</span>
          <span>
            NeoEthos
            <small>Research &amp; execution</small>
          </span>
        </div>
        <nav aria-label="Primary">
          {NAV.map((entry) => (
            <button
              key={entry.id}
              className={`nav-item${view === entry.id ? " active" : ""}`}
              aria-current={view === entry.id ? "page" : undefined}
              onClick={() => setView(entry.id)}
            >
              <span className="nav-index">{entry.eyebrow}</span>
              <span>{entry.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="connection-row">
            <div className={`dot ${connectionView.connected ? "ok" : "off"}`} aria-hidden="true" />
            <span>cTrader</span>
          </div>
          <strong title={connectionView.detail}>{brokerLabel}</strong>
        </div>
      </aside>

      <div className="main">
        <div className="content">
          <BrokerUiContext.Provider value={brokerUi}>
            <WorkspacePane active={active} />
          </BrokerUiContext.Provider>
        </div>
        <footer className="statusbar">
          <span>cTrader · {brokerLabel}</span>
          <span className="spacer" />
          <span className="muted" title={infoError}>{infoError ? "Application info unavailable" : info?.data_root ?? ""}</span>
          <span className="ver">v{info?.version ?? "…"}</span>
        </footer>
      </div>
    </div>
  );
}
