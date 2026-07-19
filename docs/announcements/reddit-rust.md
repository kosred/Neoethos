# Post for r/rust

**Title:**
NeoEthos: a pure-Rust algorithmic trading engine (genetic search + native ML ensemble + Tauri desktop) built solo on a 6-core mini PC — AGPL, no Python at runtime

**Body:**

After migrating off a Python/Rust hybrid, the whole hot path is Rust — I
wanted to share the architecture because several pieces might interest
people building compute-heavy desktop apps:

**Single-process desktop app.** The Tauri v2 shell links the engine crates
in-process and serves an axum HTTP API on an ephemeral loopback port inside
the same process — the React UI reads the port via a Tauri command. No
separate backend, no fixed port, no supervisor. A whole class of
"spawn/health-check/SSE-reconnect" bugs disappeared when the second process
did.

**Workspace layout.** Eight crates (`core`, `data`, `models`, `search`,
`trader`, `app`, `cli`, the Tauri shell) plus two *isolated* sidecar
workspaces with their own lockfiles: a P2P mesh on iroh 1.0 (QUIC + relays,
gossip peer discovery, distributes strategy-search work across trusted
machines with no server or port-forwarding) and an MCP client on rmcp.
Root `exclude` keeps their dependency trees from ever touching the pinned
engine stack.

**Compute.** A genetic search evaluates strategy populations over a
multi-timeframe feature cube (~350 engineered features); evaluation runs on
CPU (rayon) or GPU (cubecl/wgpu or CUDA) with a hybrid splitter. The ML
ensemble is native: tree boosters (XGBoost/LightGBM/CatBoost via FFI) +
Burn neural nets (transformer, N-BEATS, TiDE, TabNet, KAN) + a candle-based
DQN — no embedded Python anywhere.

**Never-OOM as an invariant.** Peak memory is a function of *available
hardware*, never of user parameters: the feature cube streams to a mmap'd
store when it won't fit in RAM, budgets adapt at runtime, and the engine
chunks down to a single genome rather than crash. A force-killed run's
orphaned stores are swept on the next start.

**TUI too.** A full ratatui terminal UI (live candlesticks, discovery
progress, logs) shares the exact same engine and config resolution as the
GUI, so the two can't drift.

It's a real shipped app (Windows installers; Linux builds from source),
AGPL-3.0, zero telemetry — every outbound connection is enumerated in
PRIVACY.md. The trading domain comes with an honest disclaimer: nothing it
finds is a promise of profit; the entire design center is *refusing* to
trade strategies that fail out-of-sample validation.

Repo: https://github.com/kosred/Neoethos

Happy to go deep on any of it — the Tauri in-process pattern, the
GPU/CPU split, Burn vs. tree ensembles, or the iroh mesh.
