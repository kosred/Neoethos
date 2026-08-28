# Branch triage — 2026-08-10

**Verdict in one line: nothing in the unmerged branches blocks the 4090 run, and
none of them can be merged.**

## Why merging is impossible

Every unmerged tip is rooted at a master that predates commit `64722aa7`
(474 files, +110,851 / −12,675). Measured, not estimated:

| Branch | `git diff --stat master..branch` |
|---|---|
| `wf62103d94/cubecl-011-integration` | 2091 files, +19,445, **−420,485** |
| `feat/confidence-at-entry` | 2083 files, +17,711, **−419,847** |
| `worktree-wf_7479823c-62a-1` | 2182 files, +20,222, **−455,026** |
| `slice3-dedup` | 2081 files, +17,608, **−418,943** |

Those ~420,000 deletions are today's work. A merge would not add their content;
it would revert the day. Anything wanted from them must be **cherry-picked as
individual commits**, and each one checked against what was rebuilt today.

## The cubecl question — settled, and it is good news

Six branches (9 commits each, 2026-08-03) port `cubecl_eval.rs`,
`prototype_c_engine/device.rs`, `prototype_c_gpu.rs`, `signal_trace_gpu.rs` and
`trade_trace_gpu.rs` off the `ArrayArg`/`ScalarArg` launch API removed in
cubecl 0.11.

Master still contains **212** `ArrayArg`/`ScalarArg` sites, which reads alarming
until you check the pin: `crates/neoethos-search/Cargo.toml:13` and `:21` pin
**cubecl 0.10.0**, and `Cargo.lock` agrees. 0.10 is the version that *has* those
APIs, so master is internally consistent and its GPU build is not broken.

**These six are an UPGRADE to 0.11, not a repair.** The recorded position is
that burn 0.21 / cubecl 0.10 are the stable maxima. Task #11 is marked completed
in the task list; what completed was the port *on a branch*, not on master.
Correct that record rather than trusting it.

Note the six tips are **divergent, not a chain** — six separate stages of the
same port, each 9 commits ahead. Cherry-picking means reconstructing the
sequence, not taking one tip.

## Deleted — codex, by operator instruction ("old, no point going in")

All four predate the 2026-08-09 audit, which covered the same ground across 323
verified items:

- `codex/audit-remediation` (was `473210ff`)
- `codex/audit-remediation-pre-v054` (was `4d71c0ce`)
- `codex/gpu-native-ml-stage3` (was `8896627d`)
- `codex/audit-remediation-writable` (was `3250cd51`) — its worktree under
  `Documents/Codex/2026-07-13/` was removed with it

24 unmerged tips remain.

## Superseded — content rebuilt today, do not cherry-pick

- `worktree-wf_7479823c-62a-1` — "wire the RiskManager gate, armed by nothing".
  This is ledger item W3; the RiskyModeManager kill switch was wired on
  2026-08-09 with the tier accumulators and the pre-send ceiling.
- `feat/confidence-at-entry` — "the snapshot erased the session profile on the
  cost path". The session-spread profile was wired into production the same day,
  at the single point every discovery setting flows through.
- `worktree-wf_7479823c-62a-3` — the news blackout window. The three news knobs
  it configures were **deleted** on 2026-08-09; the item has no subject left.

Verify each against the current file before discarding, but the burden of proof
is now on re-adding them.

## Worth cherry-picking, in this order

1. **`slice3-dedup`** — "build xgboost from official source with CUDA". Directly
   relevant to the card: it is the difference between an xgboost that uses the
   GPU and one that does not.
2. **`worktree-wf_7937dcbd-6c6-1` / `-2`** — strict mode refusing a different
   engine instead of falling through, and the ATR trailing stop that never
   ratcheted on the wgpu path. Both are GPU-correctness, both small.
3. **`wf/capability-report-honest`** — the GPU capability report describing what
   is actually there. Pairs with the device self-report work.
4. **`worktree-wf_90cc53c5-4fa-1`** — "charge what the device allocates, so the
   cheapest candidate is not the one that lies about memory".
5. **The six cubecl branches** — only when moving to cubecl 0.11 is a decision
   taken on purpose, with a measurement behind it.

## Deferred by the operator's own sequencing

- `worktree-wf_04b89097-88c-5` — the compile-only HIP seam, and `-4` its
  portability roadmap. AMD comes after everything else is closed and verified.
- `dependabot/.../brace-expansion-5.0.7` — trivial, take it whenever.

## What this means for the 4090

Nothing here gates the card. Build master as it stands, with `nvcc` as the real
compiler for the 253 vendor files no compiler has yet touched, read the whole
log including warnings, and run M15 end to end. The cherry-picks above make the
card *better*; none of them makes it *possible*.
