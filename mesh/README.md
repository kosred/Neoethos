# NeoEthos Mesh — the P2P sidecar

> **The rich buy server farms. We share.**
> This is the path that lets NeoEthos users on the open internet pool their
> compute for strategy discovery — with no central server, and without anyone
> needing a public IP.

## Why it's a separate program (read this first)

The NeoEthos trading engine sits on a delicately pinned dependency stack
(GPU/cubecl/burn, specific rustls/reqwest versions). `iroh` — the P2P library
this uses — drags in a large, fast-moving tree (QUIC/quinn, its own rustls
generation). Linking that into the engine would risk exactly the disaster the
project must never allow: **one dependency conflict setting the whole thing
back months.**

So the mesh is a **completely isolated binary**:

- its **own Cargo workspace + `Cargo.lock`** (this directory) — it shares
  *nothing* with the main workspace's dependency resolution;
- it is listed under `exclude` in the root `Cargo.toml`, so no root `cargo`
  command ever builds it into the engine;
- it talks to the running NeoEthos app **only over the localhost HTTP API**
  (`/federation/*`, the endpoints shipped in v0.5.2).

**A bug in here can crash this process and nothing else.** The trading engine
never even links against iroh. That is the whole design.

## Status: Phase A (identity + connectivity) — DONE

`neoethos-mesh` today:

1. **Identity** — loads or creates a stable ed25519 key at
   `<data-dir>/identity.key`; its public key is this node's permanent mesh
   address.
2. **Connectivity** — binds an iroh endpoint (QUIC + TLS 1.3) and comes online
   via the default relay network, so the node is reachable by other NeoEthos
   nodes anywhere, through NAT.
3. **Bridge check** — verifies it can reach the local app's
   `GET /federation/status`.

```bash
# Build (from this directory — its own isolated workspace):
cargo build --release          # → target/release/neoethos-mesh

# Run alongside a running NeoEthos app:
./target/release/neoethos-mesh --app-url http://127.0.0.1:<APP_PORT>
```

The app's HTTP port is ephemeral; the Federation panel (Advanced → Federation)
is where the HTTP coordinator/worker already lives today.

**Distributed discovery already works today** over HTTP (Advanced →
Federation) for a group that can reach one coordinator (Tailscale / port
forward). This sidecar is the road to doing the same P2P over the open
internet, serverless.

## Roadmap — Phases B–F (the spec, so anyone can continue)

Each phase is independent and testable. Build them in order; keep every
result flowing through the app's existing local gates (Strategy Lab, tail
risk, blacklist, demo gate) — **the mesh changes transport, never trust.**

- **Phase B — gossip announce/discover.** Use `iroh-gossip`. Topic
  `neoethos/announce`: each node broadcasts `{node_id, capabilities
  (cpu_cores, ram_gb, work_types, supported_symbols), rep, proto_ver}` every
  ~5 min. Maintain a local peer table. Topic `neoethos/work/discovery`:
  `need`/`offer` adverts for `(symbol, base_tf, seed)` combos.
- **Phase C — the work protocol over QUIC streams (not gossip).** Signed
  `claim → accept → result` messages (ed25519). Lease TTL (12 h) with
  re-queue on expiry — mirror `app_services/federation.rs` semantics exactly,
  since Phase 0 already proved them over HTTP.
- **Phase D — capability matching.** Route GPU work only to GPU nodes,
  discovery to CPU nodes; pick peers by capability + (later) reputation.
- **Phase E — artifacts via `iroh-blobs`.** Portfolios/trades transferred as
  BLAKE3-verified blobs; the requester downloads, then runs the SAME
  deterministic verification as `federation::submit` before accepting.
- **Phase F — trust & reputation.** Start with an explicit **allow-list of
  node IDs** (small honest communities need nothing more). Add local
  reputation scoring only if the allow-list stops scaling. Never build a
  gameable global-consensus reputation before it's actually needed.

### The bridge contract (mesh ⇄ app)

The mesh never re-implements discovery. It calls the app it runs beside:

| Mesh needs to… | Calls the local app |
|---|---|
| know what work to offer / results received | `GET /federation/status` |
| publish the operator's work plan | `POST /federation/jobs` |
| lease a combo to run locally | `GET /federation/job` |
| run discovery for a claimed combo | `POST /engines/discovery/start` |
| hand a peer's result to the local gates | `POST /federation/submit` |

A remote peer's `claim/accept/result` is translated into these local HTTP
calls. That keeps ALL trading-critical logic in the audited engine and this
sidecar as pure transport.

## Hard lines (from PRINCIPLES.md and the design record)

- **Strategies may be shared. Pooled profits may NOT be built into the
  protocol** — that is collective-investment / regulated territory (see
  `docs/p2p-mesh-design-2026-07-03.md` §4). The protocol carries strategies
  and reputation only, never money.
- Every imported result passes every local gate before any real money.
- Retired (blacklisted) strategies stay dead even if a peer re-submits them.

## License

AGPL-3.0-or-later, like the rest of NeoEthos. `iroh` and its dependencies keep
their own licenses.
