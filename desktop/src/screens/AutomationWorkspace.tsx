import { lazy } from "react";
import { WorkspaceTabs, type WorkspaceTab } from "../components/WorkspaceTabs";

const AiDesk = lazy(() => import("./AiDesk"));
const Autopilot = lazy(() => import("./Autopilot"));

const TABS: readonly WorkspaceTab[] = [
  {
    id: "engines",
    label: "Engines",
    description: "Replay, inspect and supervise deployed strategy engines.",
    component: Autopilot,
  },
  {
    id: "supervisor",
    label: "Supervisor",
    description: "AI assistant, guarded supervisor controls and decision history.",
    component: AiDesk,
  },
];

export default function AutomationWorkspace() {
  return <WorkspaceTabs label="Automation sections" tabs={TABS} initial="engines" />;
}
