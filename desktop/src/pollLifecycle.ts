/** Per-hook lifetime: detached async continuations must not dispatch another request. */
export function createPollLifecycle() {
  let active = false;
  return {
    start() {
      active = true;
    },
    stop() {
      active = false;
    },
    isActive() {
      return active;
    },
    run(request: () => Promise<void>): Promise<void> {
      return active ? request() : Promise.resolve();
    },
  };
}
