# NeoEthos Mesh — the fully-automatic P2P sidecar

> **The rich buy server farms. We share.**
> NeoEthos users pool their compute for strategy discovery over the open
> internet — **no server, no port-forwarding, no Tailscale, no human in the
> loop.** A node just starts and joins the swarm.

## Why it's a separate program (read this first)

The NeoEthos trading engine sits on a delicately pinned dependency stack
(GPU/cubecl/burn, specific rustls/reqwest versions). `iroh` — the P2P library
this uses — brings its own large, fast-moving tree (QUIC/quinn, its own rustls
generation). Linking that into the engine would risk the one thing the project
must never allow: **one dependency conflict setting the whole thing back
months.**

So the mesh is a **completely isolated binary**:

- its **own Cargo workspace + `Cargo.lock`** (this directory) — it shares
  *nothing* with the main workspace's dependency resolution;
- it is listed under `exclude` in the root `Cargo.toml`, so no root `cargo`
  command ever builds it into the engine;
- it talks to the running NeoEthos app **only over the localhost HTTP API**
  (`/federation/*`, shipped in v0.5.2).

**A bug in here can crash this process and nothing else.** The trading engine
never even links against iroh. That is the whole design.

Built on **iroh 1.0** (edition 2024) — the current stable line.

## How the automatic mesh works (no human in the loop)

1. **Identity** — a stable ed25519 key at `<data-dir>/identity.key`; its public
   key is this node's permanent mesh address.
2. **Automatic connectivity** — iroh's default relay network does NAT
   hole-punching and address discovery (`discovery_n0`). The node becomes
   reachable from anywhere with **zero network configuration** — no ports to
   open, no VPN.
3. **Automatic peer discovery** — every NeoEthos node subscribes to the SAME
   fixed gossip *rendezvous topic* and periodically announces itself
   (`{node_id, cpu_cores, work_types, app_online}`). Nodes learn about each
   other with no manual setup.
4. **Work bridge** — remote work requests are translated into calls to the
   local app's `/federation/*` API, so all trading-critical logic stays in the
   audited engine and every imported result still passes the local gates
   (Strategy Lab, tail risk, blacklist, demo gate) before any real money.

### The one unavoidable P2P detail: bootstrap

Every serverless P2P network (Bitcoin, BitTorrent, IPFS) needs *some* first
contact to join the swarm. NeoEthos uses **bootstrap seed nodes**: pass a
comma-separated list of node ids in `NEOETHOS_MESH_SEEDS`. This is invisible to
users — the app ships with community seed ids, or a group shares one. A node
with no seeds still works; it simply waits to be found. (When you run a stable
node with a public identity, share its id so others can seed off it.)

## Build & run

```bash
# From THIS directory (its own isolated workspace):
cargo build --release            # → target/release/neoethos-mesh

# Run alongside a running NeoEthos app — that's all:
./target/release/neoethos-mesh --app-url http://127.0.0.1:<APP_PORT>

# Optionally join faster via known seeds:
NEOETHOS_MESH_SEEDS=<id1>,<id2> ./target/release/neoethos-mesh
```

The app's HTTP port is ephemeral; the Federation panel (Advanced → Federation)
shows the local coordinator/worker that already runs today.

## Work distribution — BUILT (discovery, verified)

The mesh now distributes work, not just discovers peers:

- Every node **announces** `{cpu_cores, work_types, app_online, has_work}` on
  the rendezvous topic (has_work = local `/federation/status` has queued jobs).
- A node whose engine is **idle** and that sees a peer with `has_work` acts as
  a **worker**: it opens a QUIC bi-stream (ALPN `neoethos/mesh/0`) to that
  coordinator, `GetJob`s a combo, runs it on its OWN local engine
  (`/engines/discovery/start`), collects the produced portfolio
  (`/portfolios/list` + reads the file), and `Submit`s it back.
- The **coordinator**'s protocol handler proxies those two RPCs to its own
  local `/federation/job` and `/federation/submit` — so the result lands in
  the inbox and **passes every local gate** before it can mean anything. The
  mesh is pure transport; trust logic never leaves the audited app.

**Verified:** builds clean on the repo nightly; a single node comes online via
relay in ~120 ms; **two independent nodes discover each other** over gossip
(node B bootstrapped off node A's id via `NEOETHOS_MESH_SEEDS`, mutual announce
exchange confirmed). Full work exchange wants two machines each running a
NeoEthos app — the honest remaining test.

## Roadmap — the remaining bricks

- **Training model-weight return** — a training job runs on the worker today
  (real compute offload), but returning the ~50 MB model to the coordinator
  needs **`iroh-blobs`** (BLAKE3-verified) + a model sink. Until then, training
  distribution offloads the compute but the weights stay on the worker.
- **Capability matching** (Phase D) — route GPU work to GPU nodes; the
  `cpu_cores`/`gpu` fields are already announced.
- **Re-announce on NeighborUp** — instant discovery instead of ≤20 s.
- **Trust** — an allow-list of node ids first; reputation only if it stops
  scaling.

## Hard lines (from PRINCIPLES.md and the design record)

- **Strategies may be shared. Pooled profits may NOT be built into the
  protocol** — that is regulated collective-investment territory
  (`docs/p2p-mesh-design-2026-07-03.md` §4). The protocol carries strategies
  and reputation only, never money.
- Every imported result passes every local gate before any real money.
- Retired (blacklisted) strategies stay dead even if a peer re-submits them.

## Testing note

Automatic connectivity + discovery compile and run; real cross-internet NAT
traversal needs 2+ machines on different networks to exercise fully. Start two
instances (different `--data-dir`) and watch them discover each other in the
logs.

## License

AGPL-3.0-or-later, like the rest of NeoEthos. `iroh` and its dependencies keep
their own licenses.
