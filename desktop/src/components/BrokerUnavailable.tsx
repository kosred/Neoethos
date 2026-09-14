import { useBrokerUi } from "../brokerUiContext";

export default function BrokerUnavailable() {
  const { access, openSetup } = useBrokerUi();
  if (access.requestsEnabled) return null;
  return (
    <div className="banner info" role={access.phase === "unavailable" ? "alert" : "status"}>
      <strong>{access.title}</strong>
      <p>{access.detail}</p>
      {access.setupAvailable && <button type="button" onClick={openSetup}>Open Broker Setup</button>}
    </div>
  );
}
