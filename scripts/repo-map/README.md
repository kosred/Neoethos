# Repository map

Local code navigation and repeatable mapping on GitHub-hosted runners. No AI API, broker session,
GPU execution or production source rewrite is involved. The workflow adds
diagnostics without changing the manual-only build/test CI.

## Resume without repeating work

Before implementation or reinstalling tooling, use the same query entry point:

```sh
python scripts/repo-map/query_map.py --resume "duplicate gate" --limit 3
```

On this Windows checkout use `.git/repo-map-venv/Scripts/python.exe`. This mode
combines current Git state, Beads Rust in-progress/ready/recently closed work,
closed-inclusive task search, and read-only CASS session search scoped to the
configured project workspaces. Text output defaults to 6,000 characters, including
only short transcript snippets; task notes show their latest recorded tail.
A failed tool or timed-out
history lookup returns a nonzero exit; it must not be interpreted as no history.
The code-map cache and its `--local` interface remain unchanged.

The stores have distinct jobs:

- `.beads/issues.jsonl` is the one Git-tracked task ledger: decisions, attempts,
  outcomes, validation references and next actions. Use `br show ID --json` for
  the whole record, `br search "topic" --all --json --limit 3` before opening a
  new task, and `br ready --json --limit 3` to select unfinished work. Update an
  existing issue as work changes; close only after its acceptance criteria pass.
  Run `br sync --flush-only` before committing. Database, locks and tool-generated
  local backups are ignored. A new checkout can reconstruct them from JSONL with
  `br sync --import-only`; never replace the ledger with a new empty workspace.
- CASS indexes existing Codex transcripts locally, including archived sessions.
  Search results retain the source path and canonical message index for checking
  what actually happened. Expand a selected hit with `cass view SOURCE
  --message-index LINE_NUMBER --source SOURCE_ID --conversation-id CONVERSATION_ID
  -C 0 --json --timeout 5000` (one command, using the installed prefix).
  Search's `line_number` is **not** a physical JSONL line: `-n` can open unrelated
  content. A message can be large; inspect only the relevant bounded content.
  Historical claims are leads; verify the referenced code/commit and distinguish
  a failed attempt from a proved result. Sessions in another worktree require
  its exact workspace filter; configure all known project paths. This installation
  includes the primary checkout and both recorded spellings of phase0-bootstrap.
  A no-hit search in one workspace is not a global absence.
- Git and the existing code map establish current source state. Task completion
  and past transcript claims never substitute for current source or execution
  evidence. Reopen completed work with the regression/change that justifies it.

Installed versions: Beads Rust **0.7.4** (`br.exe` in the existing user
`.local/bin`) and CASS **0.10.0** (Linux executable under local WSL). Release
archives were checked against the SHA-256 digests published by GitHub. Native
Windows CASS 0.10.0 reports a passing selftest then crashes during thread shutdown;
the Linux executable passed, so do not reinstall the Windows build to fix PATH.
These are local tools; no LLM API or remote history service is configured.

Machine-specific executable prefixes live in ignored `.codex/continuity.json`
as `br_command` and `cass_command` argument arrays. `history_workspaces` contains
exact workspace paths; `cass_index_report` names the one local JSON index report.
On a fresh machine, install
the pinned official releases, set these arrays, configure CASS sources for the
existing Codex session directories, and save an initial `cass index --json` report.
Subsequent resume searches use `--no-maintenance`: they cannot rebuild the index,
download models or launch detached catch-up jobs. This machine stores the
CASS configuration at `/home/konst/.config/neoethos/cass/sources.toml` and data at
`/home/konst/.local/share/neoethos/cass` in WSL; command prefixes set those paths.
`CASS_RAW_MIRROR=0` avoids a second copy of the raw transcripts; `CASS_AUTO_REFRESH=0`
prevents detached catch-up jobs. Indexing uses the configured local Codex sources;
retrieval is lexical, with no embedding download.
The original Codex logs remain necessary evidence; the index is not their backup.

When newer transcript evidence is needed, inspect free disk space and run one
incremental pass, preserving its JSON result even on nonzero exit. In PowerShell:

```powershell
$historyConfig = Get-Content .codex/continuity.json -Raw | ConvertFrom-Json
$historyExe = $historyConfig.cass_command[0]
$historyArguments = $historyConfig.cass_command | Select-Object -Skip 1
$historyResult = & $historyExe @historyArguments index --json --background
$historyExit = $LASTEXITCODE
$historyResult | Set-Content -LiteralPath $historyConfig.cass_index_report -Encoding utf8
# Inspect historyExit and the report before retrying.
```

The resume command exposes that report's coverage warnings separately from index
freshness. CASS can report a ready index even when some sources were rejected.
The initial corpus has 12 files above the default 100 MiB source budget; one is
1.44 GB, beyond this release's maximum configurable budget. Do not rebuild or
raise the limit repeatedly. Read `indexing_stats.connector_diagnostics` for exact
paths and use an `rg` excerpt on a relevant rejected original, for example in
PowerShell: `rg -n -m 3 -o '.{0,100}TechnicalIdentifier.{0,100}' SOURCE |
Select-Object -First 3` (one command, using a Windows path). Limit displayed matches
as well as matching lines: one JSONL line can contain many matches.
Coverage warnings and stale timestamps remain visible at the start of the result.
These warnings allow useful partial retrieval; unavailable tools and timeouts fail.
If disk usage grows, `cass index --gc --json` reclaims only merge-retired segments
after the tool's grace period, without rescanning sources. Never delete original logs.

There is no second handoff/TODO document to maintain. `AGENTS.md` makes this
resume procedure persistent across sessions, and the same issue is updated during
work so interruption does not lose the last recorded state.

## Use from a phone

After merging the workflow, open **Actions → Repository map → Run workflow**.
The default run creates the structural map, LOC report and duplication report.
Select **semantic_rust** for a separate, heavier Rust indexing job.
Code pushes on master and `codex/repo-map-*` branches run the structural job.
Pull requests run the new-duplicate gate without regenerating the full map.

Open the run summary, download its `repository-map-<commit>` artifact, unzip it
and open `index.html` in a browser. It is a standalone searchable connection
viewer: no web server or external JavaScript service. Artifact download/viewing
support varies between phone browsers. The SQLite/JSON files are also usable by
code assistants without loading the entire source repository into context.
The query database is in `repository-graph-<commit>`, while duplication details
are in `repository-duplicates-<commit>`. The viewer/coverage download stays small;
an assistant can separately download the database and retrieve bounded results.

The run's `README.md`, `checks.json` and `coverage.json` are authoritative about
what succeeded, failed, timed out or was not indexed. Check the separate Rust
artifact for semantic references; a green structural job does not certify Rust
semantic coverage. Tools and analysis run on GitHub compute and use its Actions
quota; zero AI API calls does not mean zero compute/storage cost.

## Outputs

- `inventory.json`: every tracked file, content hash, byte count, physical text
  lines and separate project/vendor counts. Binary files have zero text lines.
- `loc.json`: cloc's per-file code/comment/blank counts, excluding vendor.
  Embedded Rust tests still count as code; this is not a production-only metric.
- `graph.sqlite` / `index.html`: file ownership, definitions, imports/macros,
  call sites, possible name targets, declared Cargo dependencies and features.
- `architecture.md` / `architecture.json`: every project/vendor component's
  size, largest project files, declared package dependencies/cycles, indexed
  Rust document coverage, cross-component static references, duplicate file
  pairs, and remaining parser/configuration errors with source locations.
- `coverage.json`: parser coverage, grammar errors, inventory-only files and
  semantic document coverage when requested.
- `identical-files.json`: exact file matches of at least 20 physical lines.
- `duplicates/`: jscpd copy/paste/renaming candidates under the project root,
  using `.jscpd.json` and Git ignore rules (local runs also see untracked source),
  ignoring comments and identifier differences (12 lines / 80 tokens minimum,
  16 MB per-file ceiling); no model/semantic embeddings mode. Review reported
  scanner statistics for the actual format coverage. Similarity is not a bug verdict.
- `cargo-metadata.stdout`: root workspace metadata, without dependency fetching.
  TOML mapping separately inventories non-vendor manifests, including isolated
  workspaces. Neither is a resolved all-feature dependency graph.
- Semantic artifact: `index.scip`, JSON export, a graph augmented with indexed
  definitions/references, and tool stdout/stderr.

## Evidence limits

The syntax graph parses Rust, C/C++/CUDA, TypeScript/TSX, JavaScript, Python and
shell files. JSON/TOML/YAML structures, workflow jobs/actions and npm declarations
are parsed without executing commands. Other formats and vendor source files are
inventoried and explicitly marked, not silently certified. Invalid configuration
fixtures remain visible; a parse error is not automatically a production bug.
Rust cfg attributes and context remain recorded without claiming that every
branch is active. Macros are recorded without expanding them. CUDA uses the C++
grammar; unsupported constructs are counted as parse errors.

`possible_target` edges match names only. Method dispatch, aliases and generics
may make them ambiguous; zero candidates does not prove dead code. SCIP provides
static Rust symbol resolution for the repository nightly with default features,
not a complete runtime call graph or CPU/CUDA/HIP coverage. Its indexer may need
build scripts, proc macros and native dependencies. Failures are retained and
reported. No mode automatically deletes code or authorizes trading/promotion.

## Local or assistant queries

Serena 1.7.0 provides local Rust/TypeScript definitions and references through
language servers. The shared settings are in `.serena/project.yml`; caches stay
ignored. This checkout has `rust-analyzer` and `rust-src` installed for the pinned
`nightly-2026-04-07`. Five Cargo workspaces are linked. Build scripts and proc
macros remain enabled, but automatic check-on-save is disabled. The first Rust
request can take several minutes; a startup warning or empty result is not proof
that a symbol has no callers. Native dependencies, inactive GPU features and
generated code still limit coverage. The existing SCIP profiles cover additional
configurations explicitly; neither tool is a complete runtime call graph.

The local `.codex/config.toml` registers Serena (read-only tool allowlist,
`codex` context, `no-memories` mode, 300-second client tool timeout) and jscpd MCP.
Serena's default response budget is 6,000 characters. Memory/onboarding files and
browser dashboards are disabled. Codex must load the trusted project config;
already-running sessions may need a new session to receive the added tools.
Definitions and cross-file references were tested over MCP for Rust and TS;
jscpd's snippet and per-file clone queries were also exercised.

Install the pinned tools on another Windows checkout with `uv tool install
serena-agent==1.7.0`, `rustup component add rust-analyzer rust-src --toolchain
nightly-2026-04-07`, and `npm install --prefix
"$env:LOCALAPPDATA/neoethos/tools/code-intelligence" jscpd@5.4.1` (each is one
command). The Codex server commands here point to the installed `serena.exe` and
native `jscpd-windows-x64-msvc/bin/jscpd.exe`. Trust only the intended project in
both Codex and Serena; custom language-server initialization requires Serena trust.

The single `.jscpd.json` covers Rust, TS/TSX, JS, Python, C/C++ and CUDA extensions,
excluding vendor, temporary and generated build directories. The root `/tmp/`
exclusion lives in `.gitignore`: jscpd matches its own ignore globs against full
paths, so `tmp/**` would also exclude Linux baseline checkouts under `/tmp`,
while `./vendor/**` would miss vendor in an absolute baseline scan. The tests
exercise both relative and absolute scan roots. `check_duplication`
tests a proposed block against current indexed source; `get_file_clones` locates
existing overlaps. Use a small `limit`, and refresh with
`check_current_directory(limit=0)` after changes. Token thresholds are 12 lines
and 80 tokens, so shorter or substantially rewritten equivalents can be missed.
This token scan uses no model or embeddings API.

To reject newly introduced copies while leaving historical candidates visible:

```sh
python scripts/repo-map/run_checks.py --duplicates-only --baseline-ref HEAD --output .git/repo-map/duplicate-check
```

Use `git rev-parse --absolute-git-dir` for the output prefix in linked worktrees.
CI compares PRs with the base commit and pushes with the previous commit, using
the same detector/configuration. Baselines are generated from Git; no growing
baseline file is committed. The local pre-commit hook runs this gate and requires
tracked working-tree contents to match the index, preventing partially staged
changes from hiding a duplicate. Stage or stash remaining tracked edits first;
the hook never does that automatically. Existing Git LFS hooks are preserved.
Missing tools, failed scans and invalid baseline refs fail the check. CI failure
does not by itself configure GitHub branch protection.

Python 3.11+ is required (workflow uses 3.12).

For routine local discovery, reuse this mapper with its automatically refreshed
cache. No GitHub run or AI API is needed:

```sh
python scripts/repo-map/query_map.py --local CTraderDealSnapshot --exact
python scripts/repo-map/query_map.py --local financial_report --direction in
```

Install the requirements below in a virtual environment first. This Windows
checkout uses `.git/repo-map-venv/Scripts/python.exe` as its Python interpreter.
The generated cache stays in the worktree's Git directory under `repo-map/`,
outside tracked source. Each query checks the commit and hashes of all tracked
working-tree contents. Unchanged contents reuse the database; changes rebuild
it with **the existing `map_repo.py`**, not a second indexer. Refresh failures
refuse to serve stale results; details stay in the cache's `build.log`.

Local results default to 8 matches, 12 connections **total**, and at most 6,000
characters. Use `--exact`, `--direction`, `--limit`, `--edge-limit`, or
`--max-chars` to narrow output; `--format json` retains structured details.
An explicit database remains a read-only historical query with JSON output by
default. Source reading and reasoning still consume model context; indexing
itself makes no model calls. Untracked files are excluded until added to Git;
use `rg` for those. A structural map is not complete semantic/runtime knowledge.

```sh
python -m pip install -r scripts/repo-map/requirements.txt
python scripts/repo-map/map_repo.py --output /tmp/repository-map
python scripts/repo-map/query_map.py /tmp/repository-map/graph.sqlite resident_search
python scripts/repo-map/query_map.py /tmp/repository-map/graph.sqlite LiveTrading --direction in --limit 20
```

The database records the commit and every edge's evidence/path/line. Query
results are bounded, read-only and suitable for retrieving a relevant subgraph.
The compressed HTML is a compact symbol overview for current browsers with
`DecompressionStream` support; full individual call sites stay in SQLite.
Regenerate after source changes; do not use an old map as evidence for new code.
Reports live in Actions artifacts (30 days), not as generated files committed
into the already large source tree. Download a checkpoint for longer retention.

## Completed Rust checkpoint

Mapping branches (`codex/repo-map-*`) generate seven fresh static profiles for
the current commit: the root CPU workspace, `mcp`, `mesh`, the two cTrader tools,
CUDA features and HIP features. The aggregate report validates each profile's
commit, workspace and name, then combines it with same-commit structural, LOC
and clone reports. It does not depend on a historical workflow artifact.

These profiles explicitly skip native build-script outputs and proc-macro
builds. They provide source visibility; they do not validate a CPU/GPU build
or execution. Each profile retains its exact feature configuration, commit,
failures and logs. Missing or failed profiles fail the aggregate while retaining
available diagnostics. The report states unique Rust document coverage and
keeps missing files visible. The default `master` run remains structural; a
manual semantic run can include actual build outputs when prerequisites permit.

`reuse_scip.py --checkpoint ... --source-commit ...` remains available for a
completed historical semantic index. It rejects changed application/configuration
content, a dirty checkout or incomplete checkpoint checks. Reuse is valid only
when all non-mapping tracked contents still match the index's source commit.

The original indexer emitted duplicate-symbol warnings. Symbols defined in
multiple documents remain ambiguous; the graph does not arbitrarily assign
one definition. Local symbols are scoped to both document and profile.
The compact viewer omits local variables/external-only symbols; all occurrences
remain in SQLite.

Rust reports/viewer are in `repository-map-rust-reports-<commit>`. The database
is gzip-compressed into numbered `repository-map-rust-db-00-<commit>` artifacts,
each at most 24 MiB. Download all existing parts and unzip them into one folder.
An assistant or Python environment can restore and query the full database:

```sh
python scripts/repo-map/pack_graph.py /tmp/rust-graph.sqlite /tmp/downloaded-parts --restore
python scripts/repo-map/query_map.py /tmp/rust-graph.sqlite resident_search --limit 20
```

The viewer works independently of database restoration. A fresh manual semantic
run additionally retains the raw SCIP index/export as a separate archival
artifact. Indexer stderr and SCIP diagnostics are retained for investigating
unresolved native dependencies, proc macros and inactive configurations.
