import { lazy } from "react";
import { WorkspaceTabs, type WorkspaceTab } from "../components/WorkspaceTabs";

const Account = lazy(() => import("./Account"));
const Actions = lazy(() => import("./Actions"));
const Cockpit = lazy(() => import("./Cockpit"));
const News = lazy(() => import("./News"));

const TABS: readonly WorkspaceTab[] = [
  {
    id: "market",
    label: "Market & positions",
    description: "Live chart, manual order ticket, positions and broker-confirmed protection.",
    component: Cockpit,
  },
  {
    id: "performance",
    label: "Performance",
    description: "Account, journal, execution quality and profit-capture analysis.",
    component: Account,
  },
  {
    id: "orders",
    label: "Orders & approvals",
    description: "Conditional orders and human approval requests.",
    component: Actions,
  },
  {
    id: "news",
    label: "News",
    description: "Market headlines and the current news-gate context.",
    component: News,
  },
];

export default function TradingWorkspace() {
  return <WorkspaceTabs label="Trading sections" tabs={TABS} initial="market" />;
}
