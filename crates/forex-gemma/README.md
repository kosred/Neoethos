# forex-gemma

Local on-device LLM helper for the **forex-ai** trading bot. Loads
the Gemma-4-E4B Uncensored model
(`HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive`) via
mistral.rs and exposes:

1. A **conversational helper** that answers app / trading
   questions (strict topic gate, read-only tools).
2. A **`GemmaExpert`** that plugs into the existing
   `SoftVotingEnsemble` as just-another expert with a tiny
   weight — voice, not veto.

> **Phase G0 — scaffolding only.** Trait surface, schema-versioned
> config + audit, functional jailbreak regex gate, ensemble
> integration stubs. Real inference / embedding gate / ensemble
> adapter follow as focused commits.

## Operator directive (2026-05-18) — Gemma's role

> «Πάνω από όλα το Gemma δεν θα έχει κυρίαρχο ρόλο αλλά ακόμα μια ψήφο»

What that means in the codebase:

- Gemma is **NOT** a meta-decider, gate-keeper, or "main AI".
- Gemma is **another expert** in `SoftVotingEnsemble`, equal to
  GBDT, NEAT, linear, CRFM-NES, etc.
- Its vote weight is the same as peers (or less, if backtests
  show lower accuracy).
- It does **NOT** filter peer predictions, does **NOT** have
  veto power, does **NOT** run "above" the others.
- It does **NOT** execute trades directly through the chat
  helper. There is no `submit_order` tool. The ensemble's
  soft-vote decides trades like every other day.

The conversational helper (read-only Q&A) is a separate
deliverable — it answers questions about the bot, explains why
a decision was made, looks up positions, etc. Both deliverables
ship in this crate but they're independent.

## Status / phasing

| Phase | What lands | Status |
|-------|-----------|--------|
| G0 | Crate scaffolding · trait surface · config + audit schema · jailbreak regex gate · anchor corpus placeholder · `GemmaExpert` stubs · `gemma-helper` feature in `forex-app` | **DONE** |
| G1 | mistral.rs runtime · Q5/Q4 GGUF load · token streaming | Pending |
| G2 | Embedding gate (multilingual-e5-small via candle) · post-filter · session watchdog | Pending |
| G3 | Read-only tools (positions, quotes, risk config, model status, logs) for Q&A | Pending |
| G4 | Models → Gemma push bridge for **conversational context** (not for execution) | Pending |
| G5 | Tavily web search tool for Q&A | Pending |
| **G6** | **`GemmaExpert` ensemble integration — Gemma becomes one more expert with `initial_ensemble_weight = 0.0`** | Pending |
| G7 | JSONL audit log writer (schema-versioned, hashes by default) | Pending |
| G8 | REST + SSE API surface for the Flutter client (chat UI only — no trade endpoints) | Pending |

**G6 used to be "gated trading tools".** It now means "Gemma joins
the ensemble". The trading-tools idea was rejected by the operator
2026-05-18 along with the directive above. See
`src/expert.rs` module docs for the new integration shape.

## Why a separate crate?

The helper drags new deps (mistral.rs, candle for embeddings,
Tavily HTTP client, regex) that the rest of `forex-app` doesn't
need. Keeping the crate optional behind the `gemma-helper`
feature in `forex-app` means:

- Builds without the helper stay lean (no LLM weights, no
  candle, no mistral.rs).
- Memory budget stays predictable for users who don't want a
  local LLM (a Q5 4B model is ~2.5 GB resident).
- The trait surface lets the ensemble integrate via a one-way
  dependency (`forex-gemma → forex-models`, never the reverse).

## Topic gate is load-bearing for Q&A

The chosen checkpoint
(`HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive`) has had
refusal training **deliberately stripped**. That means the
system prompt is signal, not enforcement — the helper will say
yes to anything if we let it. So the topic gate (G2) does the
real work for the **conversational helper** path:

1. **Jailbreak regex pre-filter** (Layer 2.1 — *live in G0*) —
   literal patterns ("ignore previous", "developer mode",
   "DAN", Greek variants).
2. **Multilingual embedding similarity** (Layer 2.2) — against
   a curated anchor corpus (in-scope vs out-of-scope sentences
   in EN+EL).
3. **System prompt** (Layer 2.3) — polite layer, defence in
   depth.
4. **Post-filter** (Layer 2.4) — re-check Gemma's response
   before streaming to the user.
5. **Sliding session watchdog** (Layer 2.5) — tighten thresholds
   when soft refusals stack up.

The gate does **not** apply to the `GemmaExpert` ensemble
inference path — that path receives `DataFrame` features, not
free-form user text, and produces a deterministic
`(direction, confidence)` reply parsed from a fixed-template
prompt. No jailbreak surface there.

## Look-ahead bias discipline

The `ToolContext` carries a `past_data_cutoff_unix_ms` field
that every time-series tool **must** respect — only data with
timestamp `< past_data_cutoff_unix_ms` may flow into Gemma's
context. The most recently formed bar must be fully closed
before its data becomes visible. Same discipline the GPU-
migration audit enforces on the training side.

The `GemmaExpert` inference path inherits this from the
ensemble's existing bar-closed gating (the ensemble already
operates on closed bars only); no extra check needed on the
expert side beyond what `forex-models::ensemble_inference`
already provides.

## Schema versioning

Every operator-facing artifact this crate writes
(`gemma_config.toml`, `gemma_audit.jsonl`, the anchor corpus
file) uses `forex_core::SchemaVersion` and implements
`forex_core::HasSchemaVersion`. Adding fields with
`#[serde(default)]` is non-breaking; renaming / typing changes
bump the version and ship a migration. Same convention as
`broker_credentials.toml`.

## Build features

| Feature | What it enables | Status |
|---------|----------------|--------|
| `runtime-mistralrs` | Pulls mistral.rs, makes `StubGemmaRuntime` real | Declared, no impl in G0 |
| `gate-embedding` | Pulls candle, makes `EmbeddingGate` real | Declared, no impl in G0 |
| `search-tavily` | Tavily client for `web_search` tool | Declared, no impl in G0 |

(Future) `ensemble-integration` — adds the optional
`forex-models` dep and turns `GemmaExpert` into a real
`forex_models::ensemble_inference::ExpertModel`. Wired in G6.

All features are off by default. The bare crate (G0) compiles
without any of them and provides functional trait stubs.

## Περίληψη στα Ελληνικά

`forex-gemma` είναι το νέο crate για το local Gemma-4-E4B
helper. Στο G0 (αυτό το commit) δίνει:

- trait surface για όλα τα layers (runtime, gate, tools, audit,
  bridge, **expert**, api)
- schema-versioned config + audit + anchor corpus
- **functional** jailbreak regex gate (Layer 2.1 του topic
  gate)
- `GemmaExpert` stub με τη σχεδιαστική πρόθεση (G6 wiring στο
  `forex-models::ensemble_inference::ExpertModel`)
- in-memory test backends για bridge + audit

Το Gemma είναι **ακόμα μία ψήφος** στο `SoftVotingEnsemble`,
όχι meta-decider. Trading flow περνά από το ensemble όπως
ισχύει σήμερα — όχι από τον helper. Όλα τα πραγματικά LLM bits
έρχονται σε επόμενα commits (G1-G8) πίσω από feature flags.
Default OFF στο forex-app μέσω του `gemma-helper` feature.

## Bundled model

The installer ships **Gemma-4-E4B-Uncensored-HauhauCS-Aggressive** (Q4_K_M
quantization, ~5.0 GB on disk) bundled inside the packager resources.
The runtime resolves the file at startup via the chain in
`runtime::resolve_bundled_model_path`:

1. `FOREX_AI_GEMMA_MODEL_PATH` env override — dev convenience.
2. `<exe_dir>/resources/models/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
   — installed bundle path.
3. `<repo_root>/resources/models/<filename>` — dev tree fallback.
4. `%LOCALAPPDATA%\forex-ai\models\<filename>` (Windows) /
   `$HOME/.forex-ai/models/<filename>` (POSIX) — user-data swap-in.

First hit wins. Missing-everywhere returns an actionable
`GemmaError::ConfigInvalid` that names every candidate path tried plus
the HuggingFace download URL.

### Installer-prep download

Before running `cargo build --release` for an installer artifact, run
the bundled PowerShell helper to fetch the model from HuggingFace:

```powershell
.\scripts\fetch-gemma-model.ps1            # default Q4_K_M (~5.0 GB)
.\scripts\fetch-gemma-model.ps1 -Quant Q5_K_M   # bigger / higher quality (~5.4 GB)
.\scripts\fetch-gemma-model.ps1 -Force      # re-download
```

The script:
- Checks `C:` drive free space (aborts if < 50 GB).
- Streams the file to `resources/models/`.
- Prints SHA-256 hash for verification.
- Resumes via `.tmp` rename so partial downloads can't end up
  half-bundled.

### Why bundle (not first-run download)?

Operator directive 2026-05-18: "Το μοντέλο να το κατεβάσει και να το
πακετάρει μαζί με την εφαρμογή." A first-run download adds latency
and a network dependency to the user's first session; bundling means
the helper is ready from the moment the installer finishes.

### Swapping the quant after install

The operator can drop a different GGUF into `%LOCALAPPDATA%\forex-ai\models\`
with the canonical filename and the runtime picks it up on next start
(per resolution chain step 4). Useful when:
- A higher-quality quant (`Q5_K_M`, `Q6_K_P`, `Q8_K_P`) is preferred.
- A smaller quant is needed on a low-RAM machine (`Q3_K_M`, `IQ3_M`).
- A future HauhauCS release supersedes this version.

### Disk + memory budgets

| | Disk (file) | RAM (loaded) |
|---|---|---|
| Q4_K_M (default) | ~5.0 GB | ~5.5 GB |
| Q5_K_M | ~5.4 GB | ~5.9 GB |
| Q6_K_P | ~5.9 GB | ~6.4 GB |
| Q8_K_P | ~7.6 GB | ~8.1 GB |
