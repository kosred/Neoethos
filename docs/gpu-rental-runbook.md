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

## After the run

```bash
# Copy artifacts back BEFORE destroying the box.
#   cache/discovery/<SYMBOL>_<TF>.json                 portfolio
#   cache/discovery/<SYMBOL>_<TF>.live_portfolio.json  what Autopilot loads
#   cache/discovery/<SYMBOL>_<TF>.quality.json         per-candidate metrics
```
