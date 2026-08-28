# P2P mesh (iroh) — design decision record (2026-07-03)

**Status: Phase A + B BUILT & VERIFIED (commit 8ea8ed4c).** The isolated
`mesh/` sidecar runs on **iroh 1.0.1** (edition 2024): a node starts, gets a
stable ed25519 identity, comes online via the n0 relay network in ~120 ms
(no config), and joins a gossip rendezvous swarm — verified live. Remaining:
the work protocol (claim/accept/result) + capability matching + artifact
transfer. This document records the architecture, the operator-protective
amendments, and (below) the review of the 2026-07-03 alternative-design
dialogue.

## Verdict on the technology comparison

**iroh is the right choice** for a future NeoEthos mesh — the comparison
(libp2p = overkill/complex, Mainline DHT = bare discovery with months of
missing plumbing, iroh = QUIC + TLS1.3 + relay/hole-punch + gossip + BLAKE3
blobs out of the box) matches the state of the Rust ecosystem. The proposed
wire protocol (signed claim/accept/result, lease TTL, deterministic
verification, blob-hash artifacts) is sound and mirrors what Phase 0 already
does over HTTP.

## The four amendments (operator-protective)

### 1. SIDECAR, never in-app — the non-negotiable one
The trading engine sits on a delicately pinned dependency stack (GPU/cubecl/
burn workarounds, pinned rustls/reqwest). iroh drags a large tree (quinn,
its own rustls generation, tokio features). One version conflict inside the
workspace = exactly the "one small mistake, months back" scenario.

Therefore the mesh is a **separate binary (`neoethos-mesh`) with its own
Cargo workspace + lockfile**, talking to the app ONLY through the existing
localhost HTTP API (`/federation/*`, `/engines/*`, `/portfolios/*`). The
mesh translates gossip ⇄ HTTP. Consequences:
- a mesh bug can crash the mesh, never the trader;
- the app needs ZERO new dependencies — Phase 0 endpoints ARE the contract;
- the mesh can be shipped, updated and killed independently.

### 2. Trust-list before reputation
Global reputation needs consensus and is gameable; local reputation tables
diverge. A community of 2–50 people runs fine on an **explicit allowlist of
node IDs** (the shared-token model Phase 0 already has, upgraded to ed25519
identities). Reputation scoring (the doc's Phase E) moves to the END of the
roadmap, if ever. Small honest communities > premature trustless design.

### 3. Verification stays exactly where Phase 0 put it
Submitted artifacts land in `federation_inbox` and pass the FULL local
defence stack (Strategy Lab gates, tail risk, blacklist, demo forward-test
gate) before any money. The mesh only changes TRANSPORT, never trust.

### 4. ⚠ Regulatory line — strategies yes, pooled profits no
Sharing compute and strategy artifacts (AGPL, each person trades their OWN
account) is a research community. **Pooling or distributing PROFITS across
members is a different legal animal** (collective-investment / AIFMD/MiFID
territory in the EU) and must NOT be built into the protocol. If the
community ever wants that, it needs legal counsel first — not code. The
protocol therefore carries strategies and reputation only, never money.

## Go / no-go criteria for starting Phase A

Start `neoethos-mesh` Phase A (identity + endpoint + connectivity, nothing
else) only when ALL hold:
1. ≥ 3 real users have exchanged work through HTTP Phase 0 (Tailscale) and
   the social loop worked (jobs sized right, results verified, inbox sane);
2. v0.5.2 fresh-start discovery has produced OOS-validated strategies and
   the operator's live account survived a month on them;
3. one volunteer besides the operator commits to running a bootstrap node.

Build order then follows the proposed phases with the amendments:
A (identity/connectivity, simulation-mode tests in-process first) →
B (gossip announce) → C (claim/accept/result over QUIC) → D (capability
matching) → F (iroh-blobs artifacts) → … → E/reputation last, only if the
trust-list stops scaling.

## What Phase 0 already proves meanwhile

The HTTP federation is the protocol prototype: same job/lease/submit/verify
semantics, same inbox, same gates. Every lesson it teaches (job sizing, data
availability, verification load, lease TTLs) transfers 1:1 to the mesh.

---

# Review of the alternative-design dialogue (2026-07-03)

A second model proposed a full architecture (libp2p vs iroh vs Mainline DHT,
wire formats, security, failure modes). Honest verdict: **it largely CONFIRMS
the path we already took, with four corrections.** Capturing what's worth
keeping, and flagging what's wrong, so nothing good is lost and nothing wrong
is built.

## Where it is right (and matches what we built)

- **iroh over libp2p / Mainline** — correct. (Its own comparison table lands on
  iroh, contradicting its opening "libp2p solves everything" framing. The
  libp2p section is moot.) We chose iroh and it runs.
- **Control plane (P2P) + data plane (HTTP)** — correct, and exactly what the
  sidecar does: gossip/QUIC for discovery + coordination, the app's
  `/federation/*` HTTP for the actual work. Keep this split.
- **Signed `claim → accept → result`, lease TTL + re-queue, deterministic
  re-verification, BLAKE3 artifact hashes** — sound; already the mesh roadmap.
- **Process discipline** (design doc → in-process simulation test → minimal
  first step → 2-then-3-instance integration test) — correct, and it is our
  doctrine. Note we are AHEAD: identity + connectivity + gossip discovery are
  built and verified, not just planned.

## The four corrections (important)

1. **"We need a volunteer with a public IP / our own relays" — OVERSTATED.**
   iroh's free **n0 relay network** already does NAT hole-punching for us; a
   node with no public IP comes online in ~120 ms (verified). Running our own
   relay is a *future scaling/independence* optimization, **not** a requirement
   to launch. Do not gate the community on someone owning a public IP.
2. **Reputation is premature and is the WRONG primary defence.** The proposal
   puts `rep_score` in every announce and a reputation gossip topic. But
   *self-reported* reputation is meaningless, and global reputation needs
   consensus and is gameable. Our real Sybil/garbage defence is that
   **verification is cheap and deterministic** (re-run one backtest) **plus an
   allow-list of node ids**. Reputation stays last-if-ever (amendment #2).
   The announce carries *capabilities*, not a self-graded score.
3. **Don't gossip raw addresses.** The proposed announce ships `direct_addrs`
   + `relay_url`. Unnecessary — in iroh you connect by **EndpointId** and the
   endpoint's discovery finds the path. Announce = identity + capabilities.
4. **Don't over-engineer v1.** One announce/rendezvous topic is enough to
   start (we have it). Per-work-type topics + a reputation topic are a
   later optimization if message volume ever demands, not a v1 requirement.

## Worth adopting — the security checklist (corrected)

Primary defence is **cheap deterministic re-verification + allow-list**, with
these as the concrete checks:

| Attack | Our defence |
|---|---|
| Sybil (fake identities) | Allow-list of node ids; new ids get nothing until admitted. Verification is cheap, so fakes gain nothing. |
| Garbage / poisoned result | Requester re-runs the SAME deterministic validation as `federation::submit`; junk is dropped. |
| Man-in-the-middle | QUIC + TLS 1.3 (iroh, built-in). |
| Replay | Unique `job_id` + timestamp + nonce in signed messages. |
| Free-riding (claim, never deliver) | Lease TTL → re-queue; drop the peer. |
| Stolen result | Results signed by the worker key; requester publishes the verdict. |
| Data pollution (wrong bars) | `data_hash` (BLAKE3) checked before the run. |
| Eclipse / isolation | Multiple peers + seed redundancy; never depend on one. |
| Pooled-profit scam | **Not possible — the protocol carries no money** (amendment #4). |

## Worth adopting — failure modes

| Failure | Behaviour | Recovery |
|---|---|---|
| Worker dies mid-job | Lease expires | Job re-offered to another worker |
| Requester dies | Worker's submit fails, retries, then drops the job | Worker takes new work |
| Relay dies | iroh auto-reconnects to another relay | Automatic |
| Network partition | Gossip continues locally; syncs on rejoin | Automatic |
| Duplicate claim | First accept wins; others get lease-rejected | Automatic |
| Slow worker | Exceeds lease → re-assigned | Automatic |

## Worth adopting — the work-distribution matrix

Which engine work splits, by what key, and its hardware bias — the planning
map for Phases C–G:

| Work | Split key | HW bias | Artifact size |
|---|---|---|---|
| GA discovery | (symbol, tf, seed) | CPU cores | ~100 KB / portfolio |
| Model training | (model type, symbol) | GPU (CUDA) | ~50 MB / model |
| Backtest / scoring | (strategy, period) | RAM + CPU | ~1 KB / result |
| Challenge / tail-risk MC | (portfolio, sizing) | CPU | ~1 KB / result |
| Evaluation | (portfolio, params) | RAM + CPU | ~10 KB / report |
| **Live ensemble inference** | **does NOT distribute** | needs the operator's broker session | — |

The last row is the important one the dialogue got right: anything that touches
the live broker account stays local. The mesh distributes *research*, never
*execution*.

## Net

The dialogue is a good design review that validates our architecture and
supplies three genuinely useful reference tables (security, failures, work
matrix — captured above, corrected). It does **not** change the plan or
warrant a rewrite: keep the isolated sidecar, keep verification + allow-list
as the trust model, use n0 relays now, and build the work protocol next per
`mesh/README.md`. Reputation and our-own-relays are optional later, not
launch blockers.
