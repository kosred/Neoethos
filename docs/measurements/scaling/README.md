# Population scaling sweep — RTX A6000, 2026-07-28

## The headline

Prototype B's throughput is dominated by how full the card is, not by precision.
The reduce kernel runs one thread per candidate, so a small population leaves an
A6000 almost idle. Measured, GPU-only timing, parity verified at every point:

| population | bars | GPU time | candidate-bars/s | parity |
|---|---|---|---|---|
| 256 | 4 096 | 0.0252 s | 42 M | ✓ |
| 16 384 | 4 096 | 0.0796 s | 843 M | ✓ |
| 65 536 | 4 096 | 0.2896 s | 927 M | ✓ |
| 131 072 | 4 096 | 0.5559 s | 966 M | ✓ |

**23x more throughput from filling the card, with no code change and no loss of
exactness.** The earlier "B is only 1.6x a CPU" figure was taken at population
256, where the measurement was dominated by launch latency rather than work.

The card never hit its memory ceiling: 131 072 candidates x 4 096 bars = 537 M
candidate-bars ran with an event capacity of 520 M sized from free VRAM.

## The CPU comparison is not settled

Same shapes, CPU reference derived by subtracting GPU time from total wall:

| population | GPU A (f32) | GPU B (exact) | CPU, 48 cores | B / CPU |
|---|---|---|---|---|
| 16 384 | 860 M | 843 M | 40 M | 20.9x |
| 65 536 | 1 009 M | 927 M | 127 M | 7.3x |
| 131 072 | 842 M | 966 M | 174 M | 5.5x |

Treat these ratios as **upper bounds, not results.** The CPU figure is obtained
by subtraction and still contains process startup and fixture construction, so
it understates the CPU and therefore overstates the GPU's advantage. The CPU
column is also still climbing at the largest shape, meaning it has not
saturated. A clean CPU-only timing mode is required before any multiple is
quoted as fact.

## A false result this sweep produced, and the guard that now prevents it

The first run of this sweep varied `--population` against a snapshot and
reported throughput rising from 147 M to 10 114 M candidate-bars/s. None of it
was real: a snapshot carries its own population, so all four runs evaluated the
same 256 candidates, and the "scaling" came entirely from the caller multiplying
a requested population by the bar count. The tell was that GPU time stayed flat
at 0.032 s while the workload supposedly grew 64-fold.

`--execute-snapshot` now refuses `--population`, `--bars` and `--features`
outright. A parameter that cannot change the measurement must not be quietly
absorbed into it.

## Open items

- Clean CPU-only benchmark mode, so the ratio can be stated rather than bounded.
- Real-data snapshots at large populations: the parity evidence at 20 000 and
  200 000 bars is at population 256, and the scaling evidence is on the
  synthetic tiny fixture. They have not yet been demonstrated together.
- The reduce kernel is still one thread per candidate. Filling the card hides
  that, it does not fix it: an affine-scan reformulation would remove the
  dependence on population size altogether.
