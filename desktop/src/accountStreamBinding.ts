import type { AccountStreamSnap } from "./api";
import type { BrokerAccountScope } from "./brokerUi";

export type AccountStreamView = Readonly<{
  scope: BrokerAccountScope | null;
  snap: AccountStreamSnap | null;
  connected: boolean;
  error: string;
}>;

/** Keyed React remounting alone cannot identify a late old-account payload on a new SSE connection. */
export function accountStreamViewForScope(view: AccountStreamView, scope: BrokerAccountScope | null): AccountStreamView {
  return scope !== null && view.scope !== null && view.scope.accountId === scope.accountId && view.scope.environment === scope.environment
    ? view
    : { scope, snap: null, connected: false, error: "" };
}

/** One account subscription's bound identity and lifetime; no polling, auth or execution. */
export function createAccountStreamBinding(expected: BrokerAccountScope, publish: (view: AccountStreamView) => void) {
  const scope = { ...expected };
  let active = true;
  let snap: AccountStreamSnap | null = null;
  let connected = false;
  let identityError = "";
  let transportError = "";
  const emit = () => publish({ scope, snap, connected, error: identityError || transportError });
  return {
    receive(incoming: AccountStreamSnap | null) {
      if (!active) return;
      if (!incoming || incoming.sourceAccountId !== scope.accountId || incoming.sourceEnvironment !== scope.environment) {
        snap = null;
        identityError = "Account snapshot identity is missing or differs from the selected account/environment. Balance and positions are unknown.";
      } else {
        snap = incoming;
        identityError = "";
      }
      emit();
    },
    status(value: boolean, error = value ? "" : "Account stream disconnected; awaiting reconnection.") {
      if (!active) return;
      connected = value;
      transportError = error;
      // A transport reconnection is not proof of the account carried by the next payload.
      emit();
    },
    stop() {
      active = false;
    },
  };
}
