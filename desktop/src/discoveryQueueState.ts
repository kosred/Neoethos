/** Exact wire states emitted by `/engines/status`. */
export type EngineRunState =
  | "Idle"
  | "Running"
  | "Succeeded"
  | "Failed"
  | "Cancelled";

export type QueueTerminalOutcome = Readonly<{
  status: "done" | "failed";
  note: string;
}>;

/** Preserve the backend's terminal truth instead of collapsing it to !running. */
export function queueTerminalOutcome(
  state: EngineRunState,
  summary: string,
): QueueTerminalOutcome | null {
  switch (state) {
    case "Succeeded":
      return { status: "done", note: summary || "completed" };
    case "Failed":
      return { status: "failed", note: summary || "failed" };
    case "Cancelled":
      return { status: "failed", note: summary || "cancelled" };
    case "Idle":
    case "Running":
      return null;
  }
}
