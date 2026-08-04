# Rented-card runbook — one session, every open verification

Budget context: ~12 EUR remaining. An RTX 3090 on vast.ai runs $0.20-0.30/hr →
~40 GPU-hours available. This session needs **4-6 hours**. Everything below is
ordered so that if the session dies early, the most valuable results are
already on disk.

Standing rules (from hard experience, do not skip):
- **AMD host only** — Intel Xeon hosts SIGILL in feature building.
- Connect via **direct port** (`public_ipaddr:direct_port_start`), never the
  `ssh*.vast.ai` proxy. Generate an **ephemeral key** for the session; the
  personal key has a passphrase.
- `vastai create instance` needs an explicit `start`.
- **Destroy the instance and delete the ephemeral key when done.** The last
  40 EUR died to instances left running idle.
- `scp` of large files dies silently — use `rsync` or split.
- **mold breaks every CUDA binary** — never link with mold on the GPU box.
- Write ALL run logs to files under `~/logs/` — do not grep-filter live
  output; the answer has repeatedly been in lines a filter dropped.

## Phase 0 — provision (15 min)

```bash
# on the box
apt-get update && apt-get install -y cmake ninja-build libstdc++-12-dev python3-pip
# cmake must be >= 3.28 for LightGBM; check:
cmake --version
nvcc --version && nvidia-smi
git clone https://github.com/kosred/Neoethos.git && cd Neoethos
```

## Phase 1 — the 10-minute checks that close open verifications (30 min)

These are pure `cargo check` — cheap, and each closes a question no Windows
box could answer.

```bash
# 1a. Master compiles with the CUDA feature (was never verified anywhere):
cargo check -p neoethos-search --features gpu-cuda --tests 2>&1 | tee ~/logs/check-master-cuda.log

# 1b. The burn-0.22/cubecl-0.11 branch — the port that went 974 -> 0 on Vulkan
#     but was NEVER compiled with gpu-cuda (no nvcc existed on the dev box):
git fetch origin wf62103d94/cubecl-011-integration 2>/dev/null || true
git checkout wf62103d94/cubecl-011-integration
cargo check -p neoethos-search --features gpu-cuda --tests 2>&1 | tee ~/logs/check-burn022-cuda.log
git checkout master
```

If 1a fails, STOP and report — nothing downstream is meaningful.
If 1b fails, note the error list and continue; it blocks only the burn-0.22
migration, not tonight's search.

## Phase 2 — build + parity (45 min)

```bash
cargo build --release -p neoethos-cli --features gpu-nvidia 2>&1 | tee ~/logs/build.log

# Parity gate — the measured-exact lane must stay exact:
NEOETHOS_REQUIRE_GPU=1 cargo test -p neoethos-search --release --features gpu-cuda gpu_ -- --nocapture 2>&1 | tee ~/logs/parity.log
```

Parity must be green before any run is believed. The f32 CubeCL lane measured
54% wrong once; the whole point of prototype B is bit-exactness.

## Phase 3 — the discovery run that was never possible before (2-3 h)

What changed since the last rented run, all landed on master:
- the 300k stop cliff is fixed (scoring/walk-forward/live now compute the
  SAME stop; before, a gene was scored on ~6 pips and traded on ~18),
- the stop pip resolution no longer falls back to 0.0001 on JPY/metals,
- kill zones resolve once from config,
- the promotion gate reads config.yaml,
- filter floors displayed = enforced.

So this is the FIRST run whose artifacts describe the system that produced
them. Baseline before touching knobs:

```bash
# Run 1 — baseline, config as shipped (population 100), for comparison:
./target/release/neoethos-cli discover --symbol EURUSD 2>&1 | tee ~/logs/run1-baseline.log

# Run 2 — the population lever (measured: 42M cand-bars/s at pop 256,
# 966M at 131k; `fits` ceiling ~16.8k at H1 bars). Set in config.yaml:
#   models.prop_search_population: 4096
./target/release/neoethos-cli discover --symbol EURUSD 2>&1 | tee ~/logs/run2-pop4096.log
```

Compare candidate counts, GPU utilisation (`nvidia-smi dmon -s u` in a second
shell, logged), and wall time. Run 2 SEARCHES MORE — it is not just faster;
its results are expected to differ.

## Phase 4 — model training with the fixed libraries (1-2 h)

Everything is on master now — LightGBM CUDA learner + OpenMP,
XGBoost CUDA, and the stop/gate/kill-zone fixes. No branch-hopping needed. Then:

```bash
./target/release/neoethos-cli train --symbol EURUSD 2>&1 | tee ~/logs/train.log
```

Watch for:
- XGBoost logs `device = cuda` (not the deprecated gpu_hist),
- LightGBM does NOT Fatal at fit — the device_type="gpu" OpenCL bug is fixed
  on master (fit_internal now consults effective_device_type, and the vendored
  build enables the CUDA learner + OpenMP). If it Fatals anyway, that is a NEW
  bug: capture the log,
- the artifact records which device each model trained on.

Label sanity: with the current asymmetric label geometry the class prior is
~66/34 and constant predictors score exactly the prior. If validation accuracy
== class prior to many decimals, the model learned nothing — that is the known
label-geometry issue, not a training failure.

## Phase 5 — teardown (5 min)

```bash
# pull the logs and any artifacts you want to keep FIRST:
rsync -avz box:~/logs/ ./rented-logs-$(date +%Y%m%d)/
vastai destroy instance <id>
# delete the ephemeral key from ~/.ssh and the vast console
```

## What NOT to spend card money on tonight

- The barrier-surface measurement — CPU-bound Python, runs on the home box.
- The models' label-geometry fix — code work, no GPU needed.
- gpu-rocm — needs an AMD card, different rental, different day.
- Anything on the burn-0.22 branch beyond the 10-minute check in Phase 1b.
