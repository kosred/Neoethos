# Federation — share compute, no server needed

*SETI@home for strategy discovery.* The rich buy server farms; a group of
friends with ordinary PCs can out-search one big machine — because discovery
scales with **coverage** (more symbols × timeframes × seeds), each combo is an
independent job, and a discovered strategy is cheap to verify. So nobody has to
be trusted: results are re-checked locally before they can ever touch money.

This guide covers **Federation Phase 0**, which works **today** in v0.5.2 over
HTTP. (The peer-to-peer mesh over the open internet is Phase 1 — see
`mesh/README.md`.)

---

## The two roles

- **Coordinator** — one person publishes a *work plan* (a list of
  `symbol timeframe` combos) and receives everyone's results. Any NeoEthos
  instance can be the coordinator; the app already runs an HTTP server.
- **Worker** — anyone points their machine at the coordinator's URL. Their
  machine fetches a combo, runs its own Discovery on it, and sends the result
  back. You can be *both* at once.

Everything lives in **Advanced → Federation**.

---

## Setup — coordinator (10 minutes)

1. **Expose your app's HTTP port to your group.** The port is ephemeral; the
   easiest safe option is [Tailscale](https://tailscale.com/) (`tailscale
   serve`) which gives your peers a private URL with zero port-forwarding.
   Alternatives: a LAN IP for people on your network, or a port forward /
   ngrok tunnel if you know what you're doing.
2. Open **Advanced → Federation → Coordinator**.
3. Type your work plan, **one combo per line**:
   ```
   EURUSD M15
   GBPUSD M15
   USDJPY H1
   XAUUSD M5
   ```
4. Set a **shared token** (any secret word). Only workers who send it can
   fetch jobs or submit — this keeps strangers out.
5. Click **Publish work plan**. The panel shows the queue, active leases, and
   results as they arrive.

Results land in `cache/federation_inbox/` and appear in your normal strategy
list — **and still pass every local gate** (Strategy Lab, tail risk, the
blacklist, the demo forward-test gate) before you would ever trade them.

## Setup — worker (2 minutes)

1. Open **Advanced → Federation → Worker**.
2. Paste the coordinator's **URL** and the **shared token** they gave you.
3. (Optional) give your machine a name.
4. Click **Start worker**. That's it — your machine now pulls a combo, runs
   Discovery locally, submits the result, and repeats. **Stop worker** halts
   it after the current step.

---

## What's safe, and what's shared

- **Shared:** the software, the historical bars (public), and the discovered
  **strategy artifacts**. That's it.
- **Never shared:** your broker keys, your account, your trades, your money.
  The federation protocol has no concept of any of them.
- **Trust:** you don't have to trust workers. Every submitted strategy is
  re-verified by *your* engine against *your* gates before it means anything.
  A retired/blacklisted strategy stays dead even if someone re-submits it.
- **The hard line:** you may share *strategies*. Do **not** try to pool or
  redistribute *profits* between members — that crosses into regulated
  collective-investment territory. Federation is a research commons, not a
  fund. (See `docs/p2p-mesh-design-2026-07-03.md` §4.)

## Why this beats one big machine

A heavy timeframe stagnates early — deeper search on one combo gives little.
The real gains come from **breadth**: 30 pairs × 7 timeframes × 2 modes ×
several seeds is hundreds of independent jobs. Ten friends with 6-core PCs
cover that breadth far faster than one expensive box ever could — and each
keeps full control of what they run and trade.

## Troubleshooting

- *Worker says "coordinator unreachable"* — check the URL and that the
  coordinator's app is running and exposed (Tailscale/port). Token must match.
- *Worker fetches nothing* — the queue is empty; the coordinator should
  publish a plan (or all combos are leased/done).
- *A result didn't show up as tradeable* — it did arrive (see the coordinator
  panel), but it must still pass your local gates. Check Strategy Lab.

## Roadmap

Phase 0 (this) works within a trusted group that can reach one coordinator.
**Phase 1** removes the coordinator entirely: a true peer-to-peer mesh (iroh:
QUIC + relays + hole-punching) where nodes find each other over the open
internet with no server at all. It is built as an **isolated sidecar** so it
can never destabilise the trading engine — see `mesh/README.md`.
