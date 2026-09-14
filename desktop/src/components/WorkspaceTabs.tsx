import {
  Suspense,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type ComponentType,
  type LazyExoticComponent,
} from "react";
import { ScreenBoundary } from "./ScreenBoundary";

export type WorkspaceScreen = ComponentType | LazyExoticComponent<ComponentType>;

export type WorkspaceTab = Readonly<{
  id: string;
  label: string;
  description: string;
  component: WorkspaceScreen;
}>;

export function WorkspaceTabs({
  label,
  tabs,
  initial,
}: {
  label: string;
  tabs: readonly WorkspaceTab[];
  initial: string;
}) {
  const baseId = useId();
  const [activeId, setActiveId] = useState(initial);
  const tabRefs = useRef(new Map<string, HTMLButtonElement>());
  const active = tabs.find((tab) => tab.id === activeId) ?? tabs[0];

  if (!active) return null;

  const ActiveScreen = active.component;
  const tabId = (id: string) => `${baseId}-${id}-tab`;
  const panelId = `${baseId}-panel`;
  const moveFocus = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const target = event.key === "ArrowRight"
      ? (index + 1) % tabs.length
      : event.key === "ArrowLeft"
        ? (index - 1 + tabs.length) % tabs.length
        : event.key === "Home"
          ? 0
          : event.key === "End"
            ? tabs.length - 1
            : null;
    if (target === null) return;
    event.preventDefault();
    tabRefs.current.get(tabs[target].id)?.focus();
  };

  return (
    <div className="workspace">
      <div className="workspace-tabs" role="tablist" aria-label={label}>
        {tabs.map((tab, index) => (
          <button
            key={tab.id}
            ref={(node) => {
              if (node) tabRefs.current.set(tab.id, node);
              else tabRefs.current.delete(tab.id);
            }}
            id={tabId(tab.id)}
            type="button"
            role="tab"
            tabIndex={tab.id === active.id ? 0 : -1}
            aria-selected={tab.id === active.id}
            aria-controls={panelId}
            className={`workspace-tab${tab.id === active.id ? " active" : ""}`}
            title={tab.description}
            onClick={() => setActiveId(tab.id)}
            onKeyDown={(event) => moveFocus(event, index)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div
        id={panelId}
        role="tabpanel"
        aria-labelledby={tabId(active.id)}
        tabIndex={0}
        className="workspace-panel"
      >
        <ScreenBoundary key={active.id} label={active.label}>
          <Suspense fallback={<div className="workspace-loading">Loading section…</div>}>
            <ActiveScreen />
          </Suspense>
        </ScreenBoundary>
      </div>
    </div>
  );
}
