import { useCallback, useEffect, useRef, useState } from "react";
import {
  streamSpots,
  streamAccount,
  type Tick,
  type AccountStreamSnap,
} from "./api";

/** Live tick stream → a map keyed by symbol name, plus a connected flag. */
export function useSpotStream() {
  const [ticks, setTicks] = useState<Record<string, Tick>>({});
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let alive = true;
    let close = () => {};
    streamSpots(
      (t) => alive && setTicks((m) => ({ ...m, [t.symbolName]: t })),
      (c) => alive && setConnected(c),
    ).then((c) => {
      if (alive) close = c;
      else c();
    });
    return () => {
      alive = false;
      close();
    };
  }, []);
  return { ticks, connected };
}

/** Live account snapshot stream (balance/equity/positions), plus connected flag. */
export function useAccountStream() {
  const [snap, setSnap] = useState<AccountStreamSnap | null>(null);
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let alive = true;
    let close = () => {};
    streamAccount(
      (s) => alive && setSnap(s),
      (c) => alive && setConnected(c),
    ).then((c) => {
      if (alive) close = c;
      else c();
    });
    return () => {
      alive = false;
      close();
    };
  }, []);
  return { snap, connected };
}

/**
 * Fetch once on mount (and re-fetch every `intervalMs` if > 0). Returns the
 * latest data, an error string, a loading flag, and a manual `reload`.
 * `deps` re-creates the fetcher when they change (e.g. a selected symbol).
 */
export function usePoll<T>(
  fetcher: () => Promise<T>,
  intervalMs = 0,
  deps: unknown[] = [],
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const alive = useRef(true);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const reload = useCallback(() => {
    return fetcher()
      .then((d) => {
        if (!alive.current) return;
        setData(d);
        setError("");
      })
      .catch((e) => {
        if (alive.current) setError(String(e));
      })
      .finally(() => {
        if (alive.current) setLoading(false);
      });
  }, deps);

  useEffect(() => {
    alive.current = true;
    reload();
    let id: ReturnType<typeof setInterval> | undefined;
    if (intervalMs > 0) id = setInterval(reload, intervalMs);
    return () => {
      alive.current = false;
      if (id) clearInterval(id);
    };
  }, [reload, intervalMs]);

  return { data, error, loading, reload };
}
