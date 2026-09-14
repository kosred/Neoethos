import { lazy } from "react";
import { WorkspaceTabs, type WorkspaceTab } from "../components/WorkspaceTabs";

const Advanced = lazy(() => import("./Advanced"));
const Files = lazy(() => import("./Files"));
const Hardware = lazy(() => import("./Hardware"));
const Settings = lazy(() => import("./Settings"));

const TABS: readonly WorkspaceTab[] = [
  {
    id: "general",
    label: "General",
    description: "Broker connection, account, training and live safeguards.",
    component: Settings,
  },
  {
    id: "advanced",
    label: "Advanced",
    description: "Full configuration, diagnostics and distributed compute controls.",
    component: Advanced,
  },
  {
    id: "hardware",
    label: "Hardware",
    description: "Detected compute devices and truthful runtime capabilities.",
    component: Hardware,
  },
  {
    id: "storage",
    label: "Storage",
    description: "Canonical data, model, cache and journal locations.",
    component: Files,
  },
];

export default function SettingsWorkspace() {
  return <WorkspaceTabs label="Settings sections" tabs={TABS} initial="general" />;
}
