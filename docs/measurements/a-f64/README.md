# Does double precision fix Prototype A? — RTX 2080 Ti, 2026-07-28

## The question

Every previous conclusion about precision came from comparing Prototype A (f32,
CubeCL) against Prototype B (f64, native CUDA). Nobody had ever run **A itself**
in double precision. That gap mattered: A already has every call site in the
discovery pipeline, while B exists only in the bench harness. If precision alone
fixed A, there would be no integration work left to do.

All three lanes were run on **one card, one set of snapshots**, so the FP64
throughput penalty is identical for every f64 lane and the comparison is not
confounded by hardware. Population 256, real EURUSD data, parity measured
against the canonical CPU engine.

## The measurement

| bars | A-f32 | A-f64 | B (f64, native CUDA) |
|---|---|---|---|
| 4 096 | ✗ 95.0 M cand-bars/s | ✗ 13.5 M | **✓ 49.7 M** |
| 20 000 | ✗ 136.4 M | ✗ 15.4 M | **✓ 47.8 M** |
| 200 000 | ✗ 138.9 M | ✗ 11.5 M | **✓ 47.4 M** |

✓ = reproduces the CPU exactly. ✗ = first divergence reported below.

## What f64 fixed, and what it did not

It fixed the catastrophic error. At 200 000 bars the f32 lane reports a net
profit of **3 940.88 against the canonical 8 506.33 — 54 % wrong**, and that is
the *first* metric compared, i.e. the headline number for the first candidate.
With f64 the first divergence moves far down the comparison and shrinks to
**20 876.09 expected against 20 916.23 actual, 0.19 %**. The accumulation error
that made the f32 lane unusable is gone.

It did not make A exact. Parity is still false at every size. The residual sits
in a later, path-dependent statistic rather than in the P&L sum, and it does not
shrink with series length the way an accumulation error does (8 % at 4 096 bars,
0.62 % at 20 000, 0.19 % at 200 000), which is the signature of a different
cause — most plausibly the signal/confidence boundary, which was deliberately
left in f32 so that this experiment changed exactly one thing.

## The decision this settles

**A-f64 is not a substitute for B.** It is 3–4x *slower* than B while still not
being exact, and B is exact. Double precision costs A roughly 9–12x its own f32
throughput.

The more useful finding is why. Both A-f64 and B are f64 on the same silicon, so
the ~4x gap between them is not precision — it is the kernel shape. A walks each
candidate with one serial, branchy thread per gene; B is warp-cooperative. This
is direct evidence for the diagnosis recorded on 2026-07-24 that the GPU problem
is the per-gene serial walk rather than occupancy or precision alone.

So the ranking is unambiguous: B for correctness and speed, A-f32 only for work
where a 54 % error is acceptable — which is nothing in this system.

## Caveats

- One card class. The RTX 2080 Ti runs FP64 at 1:32 of FP32. A consumer card at
  1:64 would penalise both f64 lanes further; a datacenter card at 1:2 would lift
  both. Because A-f64 and B are *both* f64, the ratio between them should be
  roughly hardware-independent — it is algorithmic — but that has not been
  measured on a second card class.
- One symbol, one population (256), three sizes. No Nsight pass, no occupancy or
  VRAM profile.
- Prototype C was not re-measured. It shares A's f32 root cause and produced
  identical wrong values on 2026-07-27; it remains outside the discovery path.

Total rental cost for this measurement: **$0.16**.
