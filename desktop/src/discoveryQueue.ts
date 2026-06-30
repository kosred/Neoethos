// Discovery multi-pair queue — a tiny module-level store so a queued sweep
// survives navigating between screens (the app is single-process, so the
// window stays open the whole run). The Discovery screen drives it on each
// `/engines/status` poll tick via `drive()`; the backend still runs ONE
// discovery at a time, the queue just feeds it the next (symbol, TF).
import { discoveryStart, discoveryStop, type StartJob } from "./api";

export type QStatus = "pending" | "running" | "done" | "failed";
export type QItem = {
  id: string;
  symbol: string; // "" = resolve from config
  tf: string; // "" = resolve from config
  status: QStatus;
  note?: string;
};

type State = {
  items: QItem[];
  active: boolean;
  knobs: Partial<StartJob>; // population/generations/… applied to every item
};

let state: State = { items: [], active: false, knobs: {} };
const subs = new Set<() => void>();
// Phase of the currently-running item, tracked outside React so the poll
// driver can tell "I just issued start, waiting for the backend to confirm
// running" (`starting`) apart from "backend is running it" (`running`).
let phase: "idle" | "starting" | "running" = "idle";
let issuing = false; // a discoveryStart() call is in flight

function emit() {
  for (const f of subs) f();
}
function set(p: Partial<State>) {
  state = { ...state, ...p };
  emit();
}

export function subscribe(f: () => void): () => void {
  subs.add(f);
  return () => {
    subs.delete(f);
  };
}
export function getSnapshot(): State {
  return state;
}

const labelOf = (symbol: string, tf: string) =>
  `${symbol || "(config)"} · ${tf || "(config)"}`;

/** Replace the queue with a fresh set of (symbol, TF) items + shared knobs. */
export function setQueue(
  pairs: { symbol: string; tf: string }[],
  knobs: Partial<StartJob>,
): void {
  const items: QItem[] = pairs.map((p, i) => ({
    id: `${p.symbol}_${p.tf}_${i}`,
    symbol: p.symbol,
    tf: p.tf,
    status: "pending",
  }));
  set({ items, knobs, active: false });
  phase = "idle";
}

export function startQueue(): void {
  if (state.items.some((i) => i.status === "pending")) set({ active: true });
}

export async function stopQueue(): Promise<void> {
  set({
    active: false,
    items: state.items.map((i) =>
      i.status === "pending"
        ? { ...i, status: "failed", note: "cancelled" }
        : i,
    ),
  });
  phase = "idle";
  try {
    await discoveryStop();
  } catch {
    /* best-effort */
  }
}

export function clearQueue(): void {
  set({ items: [], active: false });
  phase = "idle";
}

export const labelFor = labelOf;

/** Called every poll tick from the Discovery screen with the live backend
 *  discovery state. Advances the queue: confirms a start, detects completion,
 *  and kicks off the next pending item. Idempotent + guarded so repeated
 *  ticks never double-start. */
export async function drive(
  backendRunning: boolean,
  summary: string,
): Promise<void> {
  if (!state.active) return;
  const curIdx = state.items.findIndex((i) => i.status === "running");

  if (phase === "starting") {
    // Waiting for the backend to acknowledge the start we issued.
    if (backendRunning) phase = "running";
    return;
  }

  if (phase === "running") {
    if (!backendRunning && curIdx >= 0) {
      set({
        items: state.items.map((it, idx) =>
          idx === curIdx
            ? { ...it, status: "done", note: summary || "completed" }
            : it,
        ),
      });
      phase = "idle";
    }
    return;
  }

  // phase === "idle": start the next pending item if the engine is free.
  if (backendRunning || issuing) return;
  const nextIdx = state.items.findIndex((i) => i.status === "pending");
  if (nextIdx < 0) {
    set({ active: false });
    return;
  }
  const it = state.items[nextIdx];
  issuing = true;
  set({
    items: state.items.map((x, idx) =>
      idx === nextIdx ? { ...x, status: "running" } : x,
    ),
  });
  phase = "starting";
  const body: StartJob = {
    symbol: it.symbol || undefined,
    base_tf: it.tf || undefined,
    ...state.knobs,
  };
  try {
    await discoveryStart(body);
  } catch (e) {
    set({
      items: state.items.map((x, idx) =>
        idx === nextIdx ? { ...x, status: "failed", note: String(e) } : x,
      ),
    });
    phase = "idle";
  } finally {
    issuing = false;
  }
}
