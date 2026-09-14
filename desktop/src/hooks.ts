import { useCallback, useEffect, useRef, useState } from "react";
import { createPollLifecycle } from "./pollLifecycle";
import { accountStreamViewForScope, createAccountStreamBinding, type AccountStreamView } from "./accountStreamBinding";
import type { BrokerAccountScope } from "./brokerUi";
import {
  streamSpots,
  streamAccount,
  type Tick,
} from "./api";

/** Live tick stream → a map keyed by symbol name, plus a connected flag. */
export function useSpotStream(enabled = true) {
  const [ticks, setTicks] = useState<Record<string, Tick>>({});
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let close = () => {};
    streamSpots(
      (t) => alive && setTicks((m) => ({ ...m, [t.symbolName]: t })),
      (c) => { if (alive) { setConnected(c); setError(c ? "" : "Quote stream disconnected; awaiting reconnection."); } },
    ).then((c) => {
      if (alive) close = c;
      else c();
    }).catch((reason) => { if (alive) { setConnected(false); setError(String(reason)); } });
    return () => {
      alive = false;
      close();
    };
  }, [enabled]);
  const visibleTicks: Record<string, Tick> = enabled ? ticks : {};
  return { ticks: visibleTicks, connected: enabled && connected, error: enabled ? error : "" };
}

/** Live account snapshots are displayed only for the explicit currently selected account/environment. */
export function useAccountStream(scope: BrokerAccountScope | null) {
  const [view, setView] = useState<AccountStreamView>({
    scope: null, snap: null, connected: false, error: "",
  });
  const accountId = scope?.accountId ?? null;
  const environment = scope?.environment ?? null;
  useEffect(() => {
    if (accountId === null || environment === null) return;
    const binding = createAccountStreamBinding({ accountId, environment }, setView);
    let alive = true;
    let close = () => {};
    streamAccount(binding.receive, binding.status).then((c) => {
      if (alive) close = c;
      else c();
    }).catch((reason) => binding.status(false, String(reason)));
    return () => {
      alive = false;
      binding.stop();
      close();
    };
  }, [accountId, environment]);
  // Hide old-scope values during render, before effect cleanup/setup runs.
  return accountStreamViewForScope(view, scope);
}

/**
 * Fetch once on mount (and re-fetch every `intervalMs` if > 0). Returns the
 * latest data, an error string, a loading flag, and a manual `reload`.
 * `dependencyKey` identifies the data (e.g. symbol). Old-key data is hidden
 * immediately. Automatic polls never overlap; a manual reload supersedes
 * older requests, so late responses cannot roll the display back.
 * Disabled polls issue no automatic or manual request, and expose no old data.
 */
export function usePoll<T>(
  fetcher: () => Promise<T>,
  intervalMs = 0,
  dependencyKey?: unknown,
  enabled = true,
) {
  const [snapshot, setSnapshot] = useState<{ key: unknown; data: T | null; error: string; loading: boolean }>({
    key: dependencyKey, data: null, error: "", loading: true,
  });
  const requests = useRef({ latest: 0, pending: 0 });
  const fetcherRef = useRef(fetcher);
  const keyRef = useRef(dependencyKey);
  const enabledRef = useRef(enabled);
  const lifecycleRef = useRef(createPollLifecycle());

  useEffect(() => {
    fetcherRef.current = fetcher;
    keyRef.current = dependencyKey;
    enabledRef.current = enabled;
  }, [fetcher, dependencyKey, enabled]);

  const reload = useCallback(() => lifecycleRef.current.run(() => {
    if (!enabledRef.current) return Promise.resolve();
    const group = requests.current;
    const requestId = ++group.latest;
    group.pending += 1;
    const key = keyRef.current;
    const fetch = fetcherRef.current;
    const isCurrent = () => lifecycleRef.current.isActive() && enabledRef.current && group === requests.current && requestId === group.latest;
    return Promise.resolve().then(async () => {
      if (!isCurrent()) return;
      setSnapshot((previous) => ({
        key,
        data: Object.is(previous.key, key) ? previous.data : null,
        error: Object.is(previous.key, key) ? previous.error : "",
        loading: true,
      }));
      const data = await fetch();
      if (isCurrent()) setSnapshot({ key, data, error: "", loading: false });
    })
      .catch((e) => {
        if (isCurrent()) setSnapshot((previous) => ({
          key, data: Object.is(previous.key, key) ? previous.data : null, error: String(e), loading: false,
        }));
      })
      .finally(() => {
        group.pending -= 1;
      });
  }), []);

  useEffect(() => {
    const lifecycle = lifecycleRef.current;
    lifecycle.start();
    requests.current = { latest: 0, pending: 0 };
    if (enabled) void reload();
    let id: ReturnType<typeof setInterval> | undefined;
    if (enabled && intervalMs > 0) id = setInterval(() => {
      if (requests.current.pending === 0) void reload();
    }, intervalMs);
    return () => {
      lifecycle.stop();
      requests.current = { latest: 0, pending: 0 };
      if (id) clearInterval(id);
    };
  }, [reload, intervalMs, dependencyKey, enabled]);

  if (!enabled) return { data: null, error: "", loading: false, reload };
  return Object.is(snapshot.key, dependencyKey)
    ? { data: snapshot.data, error: snapshot.error, loading: snapshot.loading, reload }
    : { data: null, error: "", loading: true, reload };
}
