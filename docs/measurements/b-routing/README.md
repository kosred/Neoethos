# Prototype B under the discovery entry point — RTX 2080 Ti, 2026-07-28

## What was tested

`bench --prototype a` drives `try_evaluate_population_cuda`, the entry point
every discovery lane funnels through — GA scoring, the Monte-Carlo quality
screen, walk-forward and CPCV. The command line is unchanged from the earlier
measurement, so the comparison is exact: same binary invocation, same snapshots,
same card. Only the routing inside that function is new.

## Correctness: confirmed

| bars | before routing | after routing |
|---|---|---|
| 4 096 | parity **False** | parity **True** |
| 20 000 | parity **False** | parity **True** |
| 200 000 | parity **False** (54 % wrong) | parity **True** |

The discovery entry point now reproduces the canonical CPU engine exactly at
every size tested. Run with `NEOETHOS_REQUIRE_GPU=1`, so a silent CPU fallback
would have failed the run rather than passed it quietly.

## Throughput: an adapter cost the standalone engine does not have

| bars | via the discovery entry point | Prototype B directly |
|---|---|---|
| 4 096 | 2.7 M cand-bars/s | 49.5 M |
| 20 000 | 11.9 M | 49.7 M |
| 200 000 | 36.5 M | 47.6 M |

Both columns run the same kernel and produce the same exact numbers, so the gap
is entirely adapter overhead: it creates a fresh `PopulationSession` and
re-uploads the whole dataset on every call. The cost is fixed per call, which is
why it is invisible at 200 000 bars (23 % lost) and dominant at 4 096 (18x).

This is not a rounding detail for discovery. The Monte-Carlo quality screen
calls the evaluator **once per surviving candidate** — a real AUDUSD H4 run put
7 793 candidates through it — and the GA calls it every generation. A per-call
session build and dataset upload is exactly the wrong shape for that workload.

The fix is the one the CubeCL lane already uses: keep the session and the
uploaded dataset resident, keyed by dataset identity, and re-upload only the
genes and scenarios that actually change between calls. Until that lands, the
routed lane is correct but leaves most of the card's throughput unused.

## Not yet covered

- A full discovery run end to end. This proves the evaluation entry point, not
  the pipeline around it.
- One card, one symbol, population 256.

Rental cost for this test: $0.14. Instance destroyed, ephemeral key deleted.
