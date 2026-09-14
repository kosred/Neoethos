import assert from "node:assert/strict";
import test from "node:test";
import { createPollLifecycle } from "../src/pollLifecycle.ts";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

test("a delayed mutation callback cannot dispatch a saved reload after that hook's lifetime stops", async () => {
  const oldView = createPollLifecycle();
  let fetches = 0;
  let publications = 0;
  const savedReload = () => oldView.run(async () => {
    fetches++;
    if (oldView.isActive()) publications++;
  });
  await savedReload();
  assert.equal(fetches, 0, "nothing dispatches before effect setup");
  oldView.start();
  await savedReload();
  assert.equal(fetches, 1);
  assert.equal(publications, 1);

  const mutation = deferred();
  const continuation = mutation.promise.then(savedReload);
  oldView.stop();
  mutation.resolve();
  await continuation;
  assert.equal(fetches, 1, "an awaited old order must not refresh the now-detached account view");
  assert.equal(publications, 1);

  const newView = createPollLifecycle();
  newView.start();
  await newView.run(async () => { fetches++; publications++; });
  await savedReload();
  assert.equal(fetches, 2, "new view can poll; the old callback remains detached");
  assert.equal(publications, 2);
});

test("a completion already in flight cannot publish into a stopped lifetime", async () => {
  const view = createPollLifecycle();
  const response = deferred();
  let publications = 0;
  view.start();
  const request = view.run(async () => {
    await response.promise;
    if (view.isActive()) publications++;
  });
  view.stop();
  response.resolve();
  await request;
  assert.equal(publications, 0);
});

test("effect reactivation still allows requests after an interval or dependency reset", async () => {
  const view = createPollLifecycle();
  let calls = 0;
  view.start();
  await view.run(async () => { calls++; });
  view.stop();
  await view.run(async () => { calls++; });
  view.start();
  await view.run(async () => { calls++; });
  assert.equal(calls, 2);
});
