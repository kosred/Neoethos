import { lazy } from "react";
import { WorkspaceTabs, type WorkspaceTab } from "../components/WorkspaceTabs";

const Discovery = lazy(() => import("./Discovery"));
const Intelligence = lazy(() => import("./Intelligence"));
const StrategyLab = lazy(() => import("./StrategyLab"));
const StrategyReport = lazy(() => import("./StrategyReport"));
const Training = lazy(() => import("./Training"));

const TABS: readonly WorkspaceTab[] = [
  {
    id: "search",
    label: "Search",
    description: "Run receipt-bound indicator and SMC strategy research.",
    component: Discovery,
  },
  {
    id: "results",
    label: "Results",
    description: "Inspect discovered strategies, evidence and historical performance.",
    component: StrategyReport,
  },
  {
    id: "validation",
    label: "Validation",
    description: "Review promotion eligibility and validation evidence.",
    component: StrategyLab,
  },
  {
    id: "models",
    label: "Final evaluation",
    description: "Evaluate locked strategies alone or with trained candidate models and inspect saved results.",
    component: Training,
  },
  {
    id: "inventory",
    label: "Inventory",
    description: "Inspect stored model artifacts and published training handoff targets.",
    component: Intelligence,
  },
];

export default function ResearchWorkspace() {
  return <WorkspaceTabs label="Research sections" tabs={TABS} initial="search" />;
}
