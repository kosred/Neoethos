# P2P mesh (iroh) — design decision record (2026-07-03)

**Status: APPROVED DIRECTION, DEFERRED BUILD.** Phase 0 (HTTP federation,
commit a731b985) must earn real-world mileage first. This document records
the evaluated architecture + the amendments that protect the working system,
so the build can start later without re-deriving anything.

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
