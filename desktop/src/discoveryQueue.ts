// Discovery multi-pair queue — a tiny module-level store so a queued sweep
// survives navigating between screens (the app is single-process, so the
// window stays open the whole run). The Discovery screen drives it on each
// `/engines/status` poll tick via `drive()`; the backend still runs ONE
// discovery at a time, and every item retains the exact inventory receipt.
import { discoveryStart, discoveryStop } from "./api";
import {
  dataOperationErrorText,
  discoveryStartBody,
  type DatasetInventoryEntry,
  type DiscoveryKnobs,
} from "./apiContracts";
import {
  queueTerminalOutcome,
  type EngineRunState,
  type QueueTerminalOutcome,
} from "./discoveryQueueState";

export { queueTerminalOutcome } from "./discoveryQueueState";

export type QStatus = "pending" | "running" | "done" | "failed";
export type QItem = DatasetInventoryEntry & {
  id: string;
  status: QStatus;
  note?: string;
};

type State = {
  items: QItem[];
  active: boolean;
  knobs: DiscoveryKnobs; // population/generations/… applied to every item
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

const labelOf = (symbol: string, timeframe: string, generation?: string) =>
  `${symbol} · ${timeframe}${generation ? ` · ${generation}` : ""}`;

/** Replace the queue with authoritative inventory entries + shared knobs. */
export function setQueue(
  selectedDatasets: readonly DatasetInventoryEntry[],
  knobs: DiscoveryKnobs,
): void {
  const items: QItem[] = selectedDatasets.map((entry, i) => ({
    ...entry,
    id: `${entry.datasetIdentity}_${entry.generation}_${i}`,
    datasetIdentity: entry.datasetIdentity,
    generation: entry.generation,
    status: "pending",
  }));
  set({ items, knobs, active: false });
  phase = "idle";
}

export function startQueue(): void {
  if (state.items.some((i) => i.status === "pending")) set({ active: true });
}

export async function stopQueue(): Promise<void> {
  // Mark the in-flight item too — after this the driver stops ticking, so a
  // "running" row left behind would show as running forever.
  set({
    active: false,
    items: state.items.map((i) =>
      i.status === "pending" || i.status === "running"
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

function finishCurrent(index: number, outcome: QueueTerminalOutcome): void {
  set({
    items: state.items.map((item, itemIndex) =>
      itemIndex === index ? { ...item, ...outcome } : item,
    ),
  });
  phase = "idle";
}

/** Called every poll tick from the Discovery screen with the live backend
 *  discovery state. Advances the queue: confirms a start, detects completion,
 *  and kicks off the next pending item. Idempotent + guarded so repeated
 *  ticks never double-start. */
export async function drive(
  backendState: EngineRunState,
  summary: string,
): Promise<void> {
  if (!state.active) return;
  const curIdx = state.items.findIndex((i) => i.status === "running");
  const terminal = queueTerminalOutcome(backendState, summary);

  if (phase === "starting") {
    // Waiting for the backend to acknowledge the start we issued.
    if (backendState === "Running") {
      phase = "running";
    } else if (terminal && curIdx >= 0) {
      // A short job may reach a terminal state between two poll ticks, so the
      // queue must not require an observed Running sample first.
      finishCurrent(curIdx, terminal);
    }
    return;
  }

  if (phase === "running") {
    if (terminal && curIdx >= 0) {
      finishCurrent(curIdx, terminal);
    } else if (backendState === "Idle" && curIdx >= 0) {
      finishCurrent(curIdx, {
        status: "failed",
        note: summary || "backend lost the terminal discovery outcome",
      });
    }
    return;
  }

  // phase === "idle": start the next pending item if the engine is free.
  if (backendState === "Running" || issuing) return;
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
  const body = discoveryStartBody(it, state.knobs);
  try {
    await discoveryStart(body);
  } catch (e) {
    set({
      items: state.items.map((x, idx) =>
        idx === nextIdx ? { ...x, status: "failed", note: dataOperationErrorText(e) } : x,
      ),
    });
    phase = "idle";
  } finally {
    issuing = false;
  }
}
