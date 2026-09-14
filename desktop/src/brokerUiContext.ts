import { createContext, useContext } from "react";
import { brokerUiAccess, type BrokerUiAccess } from "./brokerUi";

export const BrokerUiContext = createContext<{
  access: BrokerUiAccess;
  openSetup: () => void;
}>({ access: brokerUiAccess(null), openSetup: () => {} });

export const useBrokerUi = () => useContext(BrokerUiContext);
