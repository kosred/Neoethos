# NeoEthos Pool Protocol — Masterless Distributed Computing

> **1000 machines × 6 cores / 8GB RAM = 6000 cores / 8TB RAM as one hypercomputer.**
>
> Every node works on the SAME population. Every evaluation is distributed. Every
> node independently converges to the SAME next generation — no coordinator, no
> parameter server, no single point of failure.

---

## Research Foundation

This design is validated by published peer-reviewed work:

| Paper | Key Result | Relevance |
|---|---|---|
| **EvAg** (Laredo et al., GPEM 2010) | Gossip-structured P2P EA; each individual is self-scheduled; scales better than canonical GAs even sequentially | Population structure via gossip, no master |
| **GoDE** (Biazzini & Montresor, ICPADS 2010) | Gossip-based Differential Evolution; epidemic communication; churn-resistant | Decentralized optimization over P2P |
| **P2P EA with autonomous selection** (Wickramasinghe et al., GECCO 2007) | Individuals make selection decisions locally using gossip-based population estimates | Fully decentralized selection |
| **Moshpit SGD** (Rybkin et al., NeurIPS 2021) | Decentralized all-reduce; nodes self-organize into dynamic groups; no parameter server | ML training without coordinator |
| **Tiered Gossip Learning** (Sharma et al., ICLR 2026) | Two-layer push-gossip-pull; fully coordinator-free; matches FL accuracy | Scalable P2P ML training |
| **P2PL** (arXiv 2023) | Device-to-device deep learning; max norm sync + P2P consensus; no edge server | Decentralized DL for heterogeneous devices |
| **QuinkGL** (2026, open-source) | Production P2P FL: gossip, Byzantine tolerance, NAT traversal | Reference implementation patterns |
| **Resilience to churn** (Laredo et al., 2008) | P2P EA maintains convergence even when nodes leave mid-execution | Churn tolerance validated |

---

## Core Model

### The Insight

Fitness evaluation (backtest) is **99% of compute time** and **embarrassingly parallel**:
1000 genomes independently evaluated = 1000x speedup with enough nodes.
Selection/crossover/mutation is **trivial** (<1%) and **deterministic** — every node
can do it independently from the same data.

ML training: gradient computation per batch is independent. Gradient averaging
over all batches is deterministic. Every node can compute the same average from
the same set of gradients.

### Single-Machine Illusion

```
Population Gen G: [genome_0 ... genome_N-1]     ← shared state (gossip)
                            │
    ┌───────────────────────┼───────────────────────┐
    ▼                       ▼                       ▼
 Node A (50 evals)     Node B (50 evals)       Node C (50 evals)
    │                       │                       │
    └───────────┬───────────┴───────────┬───────────┘
                ▼                       ▼
        gossip state               gossip state
     [fitness_0..49]            [fitness_50..99]
                │                       │
                └───────────┬───────────┘
                            ▼
              All 100 fitness values received
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
         Node A                         Node B
    "I have all fitness values →     "I have all fitness values →
     deterministic selection →        deterministic selection →
     Generation G+1 (same)"           Generation G+1 (same)"
                            │
                            ▼
                  Generation G+1 shared
                  via gossip → repeat
```

- **Phase 1 (distributed)**: evaluation tasks claimed via gossip, results published
- **Phase 2 (automatic, per-node)**: when all evaluations in a generation are complete,
  each node independently runs selection/crossover/mutation with **shared RNG seed**
  → same next generation, no communication needed

### Why This Is Not Master/Slave

| Aspect | Master/Slave | NeoEthos Pool |
|---|---|---|
| Coordinator | Single point of failure | None — gossip consensus |
| Scalability | Master bandwidth bound | Gossip is O(log n) |
| Trust | Master must be trusted | Every node verifies independently |
| Churn | Slave loss = wasted work | Other nodes pick up tasks |
| State | Master has full state | Every node has full shared state |

---

## Pool State

Each peer maintains:

```rust
struct PoolState {
    // Job universe — all known work descriptions
    jobs: HashMap<JobId, JobState>,

    // Current generation and its population (for GA jobs)
    generations: HashMap<JobId, Generation>,

    // Peer capabilities and pressure
    peers: HashMap<EndpointId, PeerInfo>,

    // Commitment ledger — who owes what compute
    commitments: HashMap<EndpointId, Commitment>,
}

struct JobState {
    ad: JobAd,
    // Which genome indices in current generation are claimed/evaluated
    eval_claims: HashMap<GenomeIndex, Vec<EvalClaim>>,
    // Completed evaluations
    evals_complete: BTreeSet<GenomeIndex>,
    // Completed generations
    generation: u64,
    // Is this job converged or cancelled
    status: JobStatus,
}

struct Generation {
    generation: u64,
    genomes: Vec<GenomeDescriptor>,  // serialized genome params
    shared_rng_seed: u64,           // deterministic selection
    fitnesses: Vec<Option<Fitness>>, // None = not yet evaluated
    status: GenStatus,               // Evaluating | Aggregating | Done
}

struct PeerInfo {
    cpu_cores: u32,
    ram_budget_mb: u64,
    gpu: bool,
    pressure: f32,       // 0.0-1.0 (Dask-style)
    ram_used_mb: u64,
    cpu_load: f32,
    active_task_count: u32,
    last_seen: Instant,
    commitment: Option<Commitment>,
}

struct Commitment {
    cpu_cores_min: u32,
    ram_mb_min: u64,
    duration_hours: u32,  // minimum hours pledged
    start_time: Instant,
    fulfilled: bool,
}
```

---

## Gossip Messages

All serde JSON over the existing gossip topic `neoethos-mesh-announce-v1-000000`.
Messages are tagged for dispatch.

### 1. Heartbeat (replaces old Announce, 20s interval)

```rust
Heartbeat {
    node_id: EndpointId,
    cpu_cores: u32,
    ram_budget_mb: u64,
    gpu: bool,
    app_online: bool,
    pressure: f32,          // <0.60 normal, 0.60-0.70 spill, 0.70-0.80 pause, >0.80 abort
    ram_used_mb: u64,
    cpu_load: f32,
    active_tasks: u32,
    commitment: Option<Commitment>,
    ts: u64,
}
```

### 2. JobAd — A peer announces a new pool job

```rust
JobAd {
    job_id: [u8; 32],       // BLAKE3(symbol || base_tf || work_type || announcer)
    symbol: String,
    base_tf: String,
    work_type: String,      // "discovery" | "training"
    population_size: u32,   // genomes per generation (GA) or batch count (ML)
    announcer: EndpointId,
    ts: u64,
}
```

### 3. EvalClaim — A peer claims genome evaluations

```rust
EvalClaim {
    job_id: [u8; 32],
    generation: u64,
    genome_indices: Vec<u32>,    // which genomes in this generation
    worker: EndpointId,
    ram_allocated_mb: u64,
    max_duration_secs: u64,      // estimated time
    batch_id: u64,               // unique per worker per generation
    ts: u64,
}
```

### 4. EvalResult — Evaluation results published to all

```rust
EvalResult {
    job_id: [u8; 32],
    generation: u64,
    batch_id: u64,
    worker: EndpointId,
    results: Vec<GenomeFitness>,  // (index, fitness_value, optional metadata blob ticket)
    fitness_blob_ticket: Option<String>,  // iroh-blobs ticket for full results
    ts: u64,
}

struct GenomeFitness {
    index: u32,
    fitness: f64,
    trades_ticket: Option<String>,  // iroh-blobs for full trade log
}
```

(For training jobs, `GenomeFitness` is replaced by `GradientBatch` — same structure
but carries gradient vectors instead of fitness.)

### 5. GenerationAd — Announce next generation

```rust
GenerationAd {
    job_id: [u8; 32],
    generation: u64,
    population_hash: [u8; 32],  // BLAKE3 of all genomes + rng_seed
    genome_count: u32,
    rng_seed: u64,
    parent_fitnesses: Vec<(u32, f64)>,  // summary of previous gen
    blob_ticket: String,         // iroh-blobs ticket for full population
    ts: u64,
}
```

Any peer can independently verify: given the parent population + fitness values,
deterministic selection with `rng_seed` must produce exactly this child population.
If two peers produce conflicting GenerationAds, the one with the lower node_id
(or the one first seen) wins — the other is detectable as incorrect.

### 6. GradientShare — For ML training (replaces EvalResult for training)

```rust
GradientShare {
    job_id: [u8; 32],
    epoch: u64,
    worker: EndpointId,
    batch_id: u32,
    gradient_blob_ticket: String,  // iroh-blobs with gradient tensor
    sample_count: u32,             // how many samples in this batch
    ts: u64,
}
```

### 7. GradientsReady — Trigger for averaging phase

```rust
GradientsReady {
    job_id: [u8; 32],
    epoch: u64,
    total_batches: u32,
    received_batches: u32,
    ts: u64,
}
```

Every node independently computes gradient average when `received_batches == total_batches`.

### 8. GenerationGradients — Averaged gradients for next epoch

```rust
GenerationGradients {
    job_id: [u8; 32],
    epoch: u64,
    gradient_average_hash: [u8; 32],
    blob_ticket: String,
    ts: u64,
}
```

---

## GA Discovery — Full Lifecycle

### 1. Job Creation

Node A has "EURUSD M1 discovery" queued:
```rust
JobAd {
    job_id: J = BLAKE3("EURUSD" || "M1" || "discovery" || node_a_id),
    symbol: "EURUSD",
    base_tf: "M1",
    work_type: "discovery",
    population_size: 1000,
    announcer: node_a_id,
}
```

All peers receive JobAd → add to local pool state. If this is the first generation,
the announcer also publishes `GenerationAd` with the initial random population.

### 2. Evaluation Phase

Each peer reads the shared population. The population is 1000 genomes.
Each node independently decides how many it can evaluate based on its RAM budget:

| Node | RAM Budget | Max Concurrent | Claims |
|---|---|---|---|
| A (16GB) | 12288 MB | 4 | eval_0..49 |
| B (32GB) | 24576 MB | 8 | eval_50..149 |
| C (8GB) | 6144 MB | 2 | eval_150..199 |
| D (4GB RPi) | 3072 MB | 1 | passes (full) |

Each publishes `EvalClaim` with its chosen indices. Conflicts are allowed —
overlapping claims produce more diversity (different random initialization
→ different results → both valid).

Each node runs local engine on claimed genomes:
1. Fetch genome params from shared generation state
2. Run backtest with indicator subset defined by genome
3. Compute fitness
4. Publish `EvalResult` with fitness values

### 3. Aggregation Phase

Every node tracks `EvalResult` messages. When `evals_complete` reaches
`population_size`:

1. Gather all fitness values from gossip state
2. Select parents (tournament/rank selection with shared RNG seed)
3. Apply crossover + mutation (deterministic with shared RNG seed)
4. Assemble Generation G+1
5. **First to compute**: publish `GenerationAd { generation: G+1, rng_seed: S }`
6. **Others**: receive `GenerationAd`, verify that it matches their own computed
   result. If yes → accept. If no → conflict resolution (node_id comparison or
   vote).

```
Time savings: 1000 evaluations distributed across N nodes
    if (N == 20) → 50 evals/node → 20x faster
    if (N == 200) → 5 evals/node → 200x faster
    Aggregation time: ~50ms (negligible)
```

### 4. Convergence & Completion

When fitness stops improving (tracked via generation history), the job is complete.
Any node can publish `JobComplete { job_id }` with the best genome(s).

Complete portfolios are distributed via iroh-blobs tickets, stored locally in
the federation inbox, and pass through Strategy Lab and tail-risk gates before
any live use.

---

## ML Training — Full Lifecycle

### 1. Job Creation

Similar to GA but `work_type: "training"` and `population_size` is replaced by
`batch_count` (number of dataset partitions).

### 2. Gradient Computation Phase

1. Dataset is partitioned into `N` batches (shared via iroh-blobs)
2. Each node claims batches via `EvalClaim` (reused message type)
3. Each node runs forward pass on its batches, computes gradients
4. Publishes `GradientShare { gradient_blob_ticket, sample_count }`

### 3. Gradient Averaging Phase

When all `N` gradient shares are received:
1. Every node computes: `avg_gradient = Σ(g_i × count_i) / Σ(count_i)`
2. Since all nodes see the same gradients → same average → same weight update
3. Publishes `GenerationGradients` as verification (others can check)

### 4. Next Epoch

Model weights are updated identically on all nodes → next epoch begins.

This is **Moshpit SGD** style: no parameter server, no coordinator, pure gossip-based
all-reduce. Verified by (Rybkin et al., NeurIPS 2021).

---

## Memory-Aware Backpressure

Each node self-regulates and publishes its state every 20s via Heartbeat:

| Pressure | Color | Action |
|---|---|---|
| `< 0.60` | ✅ Green | Normal — claim new evaluation tasks |
| `0.60–0.70` | ⚠️ Yellow | Spill to disk — no new claims, finish current |
| `0.70–0.80` | 🔴 Red | PAUSE — suspend compute, complete current backtests |
| `> 0.80` | 💀 Critical | ABORT — kill lowest priority task, publish abort notice |

**Pressure calculation** (each node independently):
```
pressure = ram_used_mb / ram_budget_mb
ram_budget_mb = min(available_ram * 0.80, user_config_max)
```

**CPU throttling**: if CPU temperature > 85°C → pause. If load > 0.90 → no new claims.

**Memory overhead per task** (node estimates locally):
```
discovery_eval_overhead = 500 MB  // bars + indicators per backtest
training_batch_overhead = 2000 MB // model weights + gradient buffer

max_concurrent = max(1, (ram_budget_mb - reserve_mb) / task_overhead)
```

**On abort**: worker publishes `EvalResult` with `results=[]` and
`abort_reason: "OOM"`. The claimed indices return to available pool. Other
nodes re-claim them.

---

## Commitment-Based Participation

To enforce the "υποχρεωτική συνεισφορά" requirement:

### How It Works

1. **Commitment declaration**: node includes `commitment: Some(Commitment{...})`
   in its Heartbeat when it first joins
2. **Minimums**: configurable, e.g. 2 cores + 4GB RAM for 48 hours minimum
3. **Ledger**: every node tracks every other node's commitment in `PoolState.commitments`
4. **Verification**: gossip detects if a node claims commitment but never produces
   eval results. The node's `reputation` (informational metric) drops.

### Access Gating

- A node with an active, fulfilled commitment receives all gossip messages
- A node with NO commitment receives JobAds and GenerationAds but its EvalClaims
  are **de-prioritized** by other peers (not enforced in code — peers simply
  choose not to validate results from non-contributors)
- This is **social/economic enforcement**, not cryptographic: if a node consumes
  results without contributing, others see it via gossip and can ignore it

### Commitment Tracking

```rust
struct Commitment {
    cpu_cores_min: u32,     // minimum cores pledged
    ram_mb_min: u64,        // minimum RAM pledged
    duration_hours: u32,    // minimum hours
    start_time: Instant,
}

struct Contribution {
    node_id: EndpointId,
    tasks_completed: u64,       // total eval tasks contributed
    cpu_hours: f64,             // cumulative compute
    last_contribution: Instant,
}
```

A node's contribution is visible to all peers via gossip. The system is
**self-policing**: non-contributors eventually get ignored because their
results are never requested or validated.

---

## Churn Resilience

Based on (Laredo et al., 2008): P2P EA with gossip maintains convergence even
with high churn.

| Scenario | Behavior |
|---|---|
| Node leaves mid-evaluation | Its claimed indices time out (TTL in EvalClaim) → re-available. Other nodes re-claim |
| Node leaves during aggregation | Other nodes still have all results → they compute Generation G+1 independently |
| Network partition | Both sides continue independently, produce different generations. On reconnection: conflict resolution via node_id ordering or last-committed gen |
| New node joins mid-job | Receives full state via gossip replay + iroh-blobs for latest generation. Can start evaluating immediately |
| Sybil attack (1000 fake nodes) | All results are deterministically verified locally. Invalid results are discarded. No reputation needed |

### Claim TTL

Each `EvalClaim` includes `max_duration_secs`. If no `EvalResult` is received
within that window + grace period (30s), the claim is considered abandoned and
the indices become available again.

---

## Conflict Resolution

### During Evaluation Phase (EvalClaim overlap)

Two nodes may claim the same genome indices simultaneously (gossip lag).
**Both results are accepted** — different random initialization in the backtest
produces different fitness values. Both contribute to the generation's fitness
pool. On aggregation, the sharing node takes the best fitness for each index
(or any — deterministic tiebreaker via index ordering).

### During Aggregation Phase (GenerationAd conflict)

If two nodes publish different `GenerationAd` for the same generation:
1. Both are valid — they used different RNG seeds or selection outcomes
2. Tiebreaker: lower `node_id` wins
3. Or: deterministic `min(hash(gen_ad_a), hash(gen_ad_b))` wins
4. Loser's results are discarded; its node must catch up by fetching the winner's
   population via iroh-blobs

This is eventually consistent: within N rounds, all honest nodes converge to
the same generation because they all use the same deterministic tiebreaker.

---

## Protocol Flow — Full Example

```
t=0   Node A: heartbeat { cpu_cores: 8, ram_budget: 12288, pressure: 0.35 }
      Node B: heartbeat { cpu_cores: 16, ram_budget: 24576, pressure: 0.40 }

t=1   Node A: JobAd { job_id: J, "EURUSD M1 discovery", pop_size: 500 }
      Node A: GenerationAd { gen: 0, rng_seed: 42, blob_ticket: "..." }

t=2   Node B: receives JobAd + GenerationAd
      Node B: fetches population blob via iroh-blobs
      Node B: EvalClaim { job: J, gen: 0, indices: [0..99], ram: 8192 }
      Node A: EvalClaim { job: J, gen: 0, indices: [100..249], ram: 6144 }

t=3   Node A runs 150 backtests, Node B runs 100
      (both heartbeat normal pressure)

t=10  Node B: EvalResult { job: J, gen: 0, batch: 1, results: [0..99 fitnesses] }
t=12  Node A: EvalResult { job: J, gen: 0, batch: 2, results: [100..249 fitnesses] }

t=15  Node C joins: receives gossip state, fetches latest generation
      Node C: heartbeat { cpu_cores: 4, ram_budget: 6144, pressure: 0.20 }

t=20  All 500 evaluations complete (accumulated from multiple nodes)
      Node A: "I have all 500 fitnesses → computing Gen 1..."
      Node B: "I have all 500 fitnesses → computing Gen 1..."
      (Both independently compute same Gen 1 via deterministic RNG)

t=21  Node A: GenerationAd { gen: 1, rng_seed: 43, blob_ticket: "..." }
      Node B: receives, verifies hash matches its own → accepts
      Node C: receives, fetches population → starts evaluating Gen 1

...repeat until convergence...

t=500 JobComplete { job: J, best_genome: index 42, fitness: 0.873 }
      Best portfolio blob shared via iroh-blobs → all peers store in inbox
```

---

## Changes to Existing Code

### `mesh/src/main.rs`

| Component | Change |
|---|---|
| `Announce` → `Heartbeat` | Add `pressure`, `ram_budget_mb`, `active_tasks`, `commitment`. Keep backward-compat fields |
| Pool state | New `PoolState` struct with `jobs`, `generations`, `peers`, `commitments` |
| Gossip dispatch | Match on tagged messages → route to handler per type |
| Worker loop | Replace coordinator/worker → pool-based evaluation claiming |
| Aggregate logic | Track eval completions, trigger aggregation when generation complete |
| iroh-blobs | Add dependency for population/gradient/blob transfer |
| Peer table | Extend with pressure, commitments, active tasks |
| `MeshReq`/`MeshResp` | Keep for fallback direct transfer (iroh-blobs not available) |
| Commitment tracking | Heartbeat handler updates commitment ledger |

### `crates/neoethos-app/src/app_services/federation.rs`

| Component | Change |
|---|---|
| `/pool/jobs` | New endpoint: supplies local pending jobs as JobAds |
| `/pool/status` | Pool-specific status (current generation, evaluations pending) |
| Federation inbox | Broader: also accepts pool results via local IPC |
| `/hardware` | Already correct — source of truth for Heartbeat fields |

### `mesh/Cargo.toml`

| Change |
|---|
| Add `iroh-blobs` dependency |
| Add `blake3` for job_id generation |
| (Keep all existing dependencies) |

---

## References

1. Laredo et al., "EvAg: a scalable peer-to-peer evolutionary algorithm",
   *Genetic Programming and Evolvable Machines* 11(2), 2010.
2. Biazzini & Montresor, "Gossiping Differential Evolution: A Decentralized
   Heuristic for Function Optimization in P2P Networks", ICPADS 2010.
3. Wickramasinghe et al., "Peer-to-peer evolutionary algorithms with adaptive
   autonomous selection", GECCO 2007.
4. Rybkin et al., "Moshpit SGD: Communication-Efficient Decentralized Training
   on Heterogeneous Unreliable Devices", NeurIPS 2021.
5. Sharma et al., "Tiered Gossip Learning: Communication-Frugal and Scalable
   Collaborative Learning", ICLR 2026.
6. Laredo et al., "Resilience to churn of a peer-to-peer evolutionary algorithm",
   *IJHPSA* 1(4), 2008.
7. Jelasity et al., "Gossip-based aggregation in large dynamic networks",
   *ACM TOCS* 23(3), 2005.
8. Anderson, "BOINC: A Platform for Volunteer Computing", *J. Grid Computing*,
   2020.
9. Estrada et al., "A distributed evolutionary method to design scheduling
   policies for volunteer computing", GECCO 2008.

---

## Implementation reality — grounded in NeoEthos's actual code (2026-07-03)

This addendum reconciles the spec above with what the engine actually is, what
is **done today**, and what the **major next project** is — so the deep work
is built carefully, never rushed into the core (a mistake there costs months).

### DONE today (safe, mesh-only, verified)

- **Swarm-capacity aggregation** — the "one machine" view. Each node announces
  its REAL hardware from the app's `/hardware` (`coresLogical`, `ram.totalMb`,
  `gpu.available`); the mesh sums self + all live peers and reports
  `nodes / total_cores / total_ram_gb / total_gpus`, logged every 30 s and
  written to `<temp>/neoethos_mesh_swarm.json` for the UI. Verified: two
  independent nodes report `nodes=2, total_cores=24` (12+12). This is the
  Heartbeat capacity fields the spec calls for, live now.
- **Combo-level distribution** (Phase 0): a whole discovery per node — COVERAGE
  speedup + DIVERSITY, already verified end-to-end for discovery.

### The exact hook for distributed evaluation (Phase 1 — the big one)

`crates/neoethos-search/src/genetic/search_engine.rs`, generation loop (~1451):

```
for generation in 0..generations {
    let metrics = evaluate_genes_cached(features, ohlcv, &genes, &eval_cfg, &eval_cache)?; // 99%
    apply_metrics(&mut genes, &metrics, eval_cfg.growth_objective);
    // selection / crossover / mutation → next `genes`   (1%)
}
```

Distributing = evaluate only an assigned SLICE of `genes`, publish fitness,
gather the rest from peers before `apply_metrics`. The 1% aggregation then runs
identically on every node.

### Hard problems the spec must not gloss over

1. **Deterministic aggregation is NOT free today.** The GA calls `rand::rng()`
   in selection/mutation paths → non-deterministic. Every node computing the
   SAME next generation requires switching to a **shared seeded RNG**. This is
   a real, careful change to the core engine.
2. **Identical feature cube on every node** — evaluation needs byte-identical
   `features`/`ohlcv`; must be hash-verified across machines.
3. **Latency vs. generation time** — per-generation network sync only pays off
   on SLOW combos (M1/M5, 24 h+). Distributed evaluation is **opt-in for slow
   combos**; fast combos stay local. (This matches the exact combos you care
   about — so the constraint is a feature, not a limit.)
4. Churn (slice TTL + reassign) and fitness verification as the spec describes.

### Honest phasing

- **Phase 0 — DONE**: capacity aggregation + combo distribution (verified).
- **Phase 1 — distributed evaluation** (slow combos): the major next project.
  Build in its own effort with the house discipline — **in-process simulation
  (two evaluators, one process) → 2 machines → many** — because it touches the
  core GA. This is where a real M1 run drops from 24 h to hours.
- **Phase 2 — ML training** via gradient averaging (Moshpit-SGD / ring
  all-reduce). Evaluate `burn_p2p` (P2P for the Burn framework we already use)
  and `p2panda-net` (gossip built ON iroh) as building blocks — each a real
  integration, not a shortcut.

### Operator refinement (2026-07-03) — the ISLAND MODEL resolves all three

The operator's insight redirects Phase 1 from fragile per-generation
evaluation-sync to a robust **island model**, which resolves the three hard
problems above:

- **On #2 (OOM per node) — ALREADY GUARANTEED.** The engine's never-OOM
  invariant (`cubecl_eval.rs::auto_tune_memory_budgets`, ~1691) caps each
  node's resident RAM to ~25% of ITS available RAM, clamped [256 MB, 4 GB]:
  *"a 1 GB box still runs (tiny chunks, slow but never OOM)"*. So growing the
  GLOBAL search (more generations / islands) never grows any single node's
  memory — each node holds only ITS island, sized to ITS hardware. Small
  8 GB nodes participate safely by construction. This is the invariant that
  makes the whole vision possible.
- **On #1 (deterministic sync) — the island model makes it EASY.** Islands do
  NOT need byte-identical generations. Each node evolves its own island
  independently and, every N generations, gossips its **elites** (a few best
  genomes) as migrants; other nodes import them. Migration is tiny and
  occasional, so "near-real-time" sync is trivial — no cross-node determinism
  required. This is the EvAg / GoDE approach the research confirmed, and it
  sidesteps the hardest part of the naive design.
- **On #3 (help everywhere) — resource-adaptive breadth.** More nodes = more
  islands = broader, more diverse search — automatically, on ANY combo (not
  just slow ones), because islands are independent (no per-generation latency
  tax). The app reads the swarm capacity (already written to
  `neoethos_mesh_swarm.json`) and **auto-scales search breadth** (island count
  / candidate count / combos in flight) to the visible total resources — while
  each node stays never-OOM. The extra compute is always usable.

**Revised Phase 1 (island model over the mesh):**
1. Each node runs one or more GA islands on a shared combo, each sized to its
   own RAM (never-OOM, already guaranteed).
2. Elites migrate over gossip every N generations (small, infrequent).
3. The app scales the number of islands / search breadth to the swarm's total
   capacity. NOTE: the GA has no island primitive today — importing migrant
   genomes into the population is the core-engine change, but it is FAR smaller
   and safer than per-generation evaluation-sync, and needs no shared RNG.
4. Build it simulation-first (two islands in one process, migration between
   them) → 2 machines → many.

**Bottom line:** the swarm *sees itself as one machine* today. *Computing as
one machine* becomes the island model — OOM-safe by the existing invariant,
migration-synced (not determinism-synced), and resource-adaptive — which is a
smaller, safer core change than the naive design, and is the specified Phase 1.
