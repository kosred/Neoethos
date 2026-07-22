# GPU rental runbook (A6000 / A4000)

Written for a rented box billed by the hour: every step below is something
that can be done *before* the meter starts, or a check that fails fast.

## Why this exists

Discovery, training and evaluation run **in-process inside the desktop app**.
The GPU evaluator lives behind cargo features (`gpu-*`), so a default build has
no GPU code in it at all. Installing the normal Windows build on a GPU box
gives you a **silently CPU-only** run — the card sits idle and nothing says so.

Two things must BOTH be true for GPU work to happen:

| | Answers | Where to check |
|---|---|---|
| A card is present | "does this machine have a GPU?" | Hardware screen → **Card detected** |
| The lane is compiled in | "can this binary use one?" | Hardware screen → **GPU lane in this build** |

The Hardware screen states both and warns when they disagree. `GET /hardware`
returns the same under `gpuSupport`.

## Before renting (on your own machine)

Nothing here needs the rented box:

- [ ] `git pull` — the box will clone this repo.
- [ ] Decide the lane: **`gpu-nvidia`** (CUDA) for A6000/A4000.
- [ ] Note the data you need up there. The feature cube is rebuilt on the box,
      but the *price history* must be present — copy `data/` or re-download.

## On the box, in order

```bash
# 0. Sanity: the driver must see the card BEFORE anything else.
nvidia-smi                     # name + total memory. If this fails, stop.

# 1. Toolchain (see BUILDING.md for the full list).
#    CUDA toolkit + libtorch are required by the gpu-nvidia lane.

# 2. Build the CLI first — it is the fastest path to a GPU run and needs no
#    frontend toolchain. Same engine, same features as the app by design.
cargo build --release -p neoethos-cli --features gpu-nvidia

# 3. Prove the lane is live BEFORE starting a long run.
#    The auto-tuner logs its budgets at discovery start; grep for them.
./target/release/neoethos-cli discover --symbol EURCAD --base H1 \
  --generations 5 --population 32 2>&1 | grep -i "auto-tuned memory budgets"
```

That log line is the receipt. It prints `min_vram_gb` and `vram_budget_mb`:

- `min_vram_gb = unknown` → the card was **not** seen by the engine. Fix that
  before anything else; the run would be CPU-only.
- On a 48 GB A6000 expect `vram_budget_mb ≈ 29000`. (It used to be capped at
  24576 by a flat ceiling — that cap is gone; the budget now scales with the
  card, minus a driver reserve.)

Only once that reads correctly, start the real run.

## The desktop app on the box (optional)

The CLI is enough for discovery. If you want the UI there too:

```bash
cd desktop
npm install
npx tauri build -- --features gpu-nvidia
```

Then open the **Hardware** screen: it must show *Card detected: yes* AND
*GPU lane in this build: CUDA*. If it shows "not compiled", the feature flag
did not reach the build — do not start a long run.

## Known lane issues

- **`gpu-vulkan` does not build on Windows right now** (2026-07-21): the
  dependency chain reaches `wgpu-hal`, whose DirectX12 backend fails against
  `windows 0.62` (`ID3D12Heap` no longer satisfies `Param<ID3D12Heap>`). This
  is upstream, not our wiring — the same chain resolves fine. **`gpu-nvidia`
  (CUDA) does not go through wgpu at all**, so an A6000 build is unaffected,
  and on Linux the D3D12 backend is never compiled.

## Things that will bite (learned the hard way)

- **Multi-GPU sharding is deliberately disabled.** cubecl panics
  (`Memory page 0 doesn't exist`) on the multi-device path, so each combo is
  pinned to ONE card. Combo-level parallelism (one combo per card,
  concurrently) is the supported way to use several cards.
- **The feature cube is built on the CPU/host**, then uploaded and kept
  resident in VRAM (`get_or_upload`, LRU-bounded to half the VRAM budget).
  Give the box enough RAM for the host side or it streams to disk.
- **Closing the app kills the run** — results are written at the very end.
  On a rented box, prefer the CLI under `screen`/`tmux`.
- **Bill control**: hibernate/destroy the box the moment the run finishes.
  Discovery writes its artifacts to `cache/`; copy them off first.

## Benchmarking + tuning the GPU (do this once per new card)

Task 6 of the GPU remediation. The harness measures GPU eval throughput and
peak memory, but **only after** proving the GPU output matches the CPU
reference bar-for-bar — a faster path that disagrees is a failure, not a win.

```bash
# 1. Correctness gate FIRST. This fails loud (not skips) under require-GPU.
NEOETHOS_REQUIRE_GPU=1 cargo test -p neoethos-search --features gpu-nvidia \
  gpu_cpu_parity -- --nocapture

# 2. Throughput, once parity is green. Prints one row per workload shape with
#    cpu_ms, gpu_ms, speedup, evals/s, cube MB, and a per-row parity verdict.
#    Exits non-zero if ANY shape breaks parity.
NEOETHOS_REQUIRE_GPU=1 cargo run --release -p neoethos-search \
  --features gpu-nvidia --example gpu_eval_bench
```

Read the output before trusting any number:

- **`NEOETHOS_REQUIRE_GPU true` + a real adapter line** — the run used the GPU.
  If instead it panicked with "require-GPU set but the GPU lane failed", the
  card/driver is not usable yet; fix that before reading timings.
- **`parity OK`** on every row is the precondition for looking at `speedup`.
  A `parity FAIL` row means the kernel is wrong on this hardware — stop and
  report it; do not tune around it.
- **`speedup` below 1.0x on the `small` shape is expected** — kernel launch +
  transfer overhead dominates tiny populations. The GPU earns its keep on the
  `medium`/`large` shapes. This is why tiny workloads stay on the CPU lane.

Only after parity is green across all shapes, tune within the memory guardrail:
sweep `NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB` and the stream/batch knobs, re-run the
bench, and keep a setting only if it (a) keeps parity, (b) stays under the
card's VRAM budget (see the `auto-tuned memory budgets` log line), and (c)
actually speeds up the `medium`/`large` shapes. Record the winning values by
**capability class** (e.g. "48 GB discrete NVIDIA"), never by marketing model
name, so the setting ports to the next card of the same class.

## After the run

```bash
# Copy artifacts back BEFORE destroying the box.
#   cache/discovery/<SYMBOL>_<TF>.json                 portfolio
#   cache/discovery/<SYMBOL>_<TF>.live_portfolio.json  what Autopilot loads
#   cache/discovery/<SYMBOL>_<TF>.quality.json         per-candidate metrics
```
