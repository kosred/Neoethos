# Repository map

Local code navigation and repeatable mapping on GitHub-hosted runners. No AI API, broker session,
GPU execution or production source rewrite is involved. The workflow adds
diagnostics without changing the manual-only build/test CI.

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
"$env:LOCALAPPDATA/neoethos/tools/code-intelligence" jscpd@5.3.3` (each is one
command). The Codex server commands here point to the installed `serena.exe` and
native `jscpd-windows-x64-msvc/bin/jscpd.exe`. Trust only the intended project in
both Codex and Serena; custom language-server initialization requires Serena trust.

The single `.jscpd.json` covers Rust, TS/TSX, JS, Python, C/C++ and CUDA extensions,
excluding vendor, temporary and generated build directories. `check_duplication`
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
