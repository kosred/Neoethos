# A6000 measurement run — 2026-07-27

Raw benchmark reports from the first real measurement of Prototypes A, B and C
on the target card. Every JSON here was written by `neoethos-cli bench`; nothing
was edited by hand.

**Hardware.** Rented NVIDIA RTX A6000, 48 GB, driver 580.173.02, CUDA 12.2,
36 CPU cores. Rental cost of the entire session, including the earlier
correctness gate on an RTX 3060 Ti: **$0.64**.

**Inputs.** Two real EURUSD series exported from the project's own store with
`bench-prepare --symbol`, plus the deterministic tiny fixture. Every prototype
saw byte-identical inputs.

## Results

| report | workload | parity | median | candidate-bars/s |
|---|---|---|---|---|
| `tiny-a` | 256 x 4 096 | ✗ 0.36 % | 0.0086 s | 122 M |
| `tiny-b` | 256 x 4 096 | ✓ | 0.0252 s | 42 M |
| `tiny-c` | 256 x 4 096 | ✗ 0.36 % | 0.0125 s | 84 M |
| `snap-a` | EURUSD H1, 256 x 20 000 | ✗ 1.8 % | 0.0326 s | 157 M |
| `snap-b` | EURUSD H1, 256 x 20 000 | ✗ 0.62 % on 1 of 256 | 0.0960 s | 53 M |
| `snap-b-nofma` | EURUSD H1, 256 x 20 000 | ✓ | 0.0983 s | 52 M |
| `snap-c` | EURUSD H1, 256 x 20 000 | ✗ 1.8 % | 0.0497 s | 103 M |
| `m5-a` | EURUSD M5, 256 x 200 000 | ✗ **54 %** | 0.3280 s | 156 M |
| `m5-b` | EURUSD M5, 256 x 200 000 | ✓ | 1.0164 s | 50 M |
| `m5-c` | EURUSD M5, 256 x 200 000 | ✗ **54 %** | 0.5080 s | 101 M |

`snap-b` and `snap-b-nofma` are the same engine and the same input; the second
was compiled with `-fmad=false`. All `m5-b` and `snap-b-nofma` figures are from
the non-contracting build.

## What the numbers say

**The f32 CubeCL lane is not usable for strategy selection at production series
length.** Prototypes A and C share the fused f32 accumulation and produce
identical wrong values. The error compounds with the series:

| bars | net-profit error |
|---|---|
| 4 096 | 0.36 % |
| 20 000 | 1.8 % |
| 200 000 | **54 %** |

At 200 000 bars candidate 0 is reported as +3 940.88 against a canonical
+8 506.33. That is not rounding noise; it reorders survivors. Prototype A is
the fused evaluator the production discovery GPU lane already uses, so this is
a finding about the shipped product, not only about a prototype.

**Prototype B reproduces the canonical semantics exactly at every scale, once
FMA contraction is disabled.** With contraction on, one candidate in 256
diverged by 0.62 % at 20 000 bars — a boundary comparison flipped by a fused
multiply-add. Disabling it costs about 2 % of throughput and buys exactness at
4 096, 20 000 and 200 000 bars.

**Against a CPU baseline on the same box** (production CPU evaluator, 36 cores,
rayon, same M5 workload): 31.4 M candidate-bars/s.

| engine | candidate-bars/s | vs 36-core CPU | correct? |
|---|---|---|---|
| A (f32) | 146–156 M | 4.7x | no |
| C (f32) | 101 M | 3.2x | no |
| **B (f64, no FMA)** | **50 M** | **1.6x** | **yes** |

Among engines that reproduce the semantics there is currently only one
candidate, so this is not yet an A/B/C choice — it is a statement that two of
the three are disqualified on correctness at this scale.

## What these numbers do not establish

- One symbol, one population size (256), two timeframes. No sweep.
- No Nsight Systems or Nsight Compute pass, no occupancy or VRAM figures.
- The CPU baseline is a 36-core rented box, not the operator's 6-core machine.
  A per-core extrapolation suggests roughly 10x there, but that is an estimate,
  not a measurement.
- No architecture has been selected. The stop gate in
  `docs/gpu-native-redesign.md` requires a recorded human decision, and nothing
  here substitutes for it.
