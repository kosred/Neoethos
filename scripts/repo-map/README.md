# Repository map

Read-only, repeatable mapping on GitHub-hosted runners. No AI API, broker session,
GPU execution or production source rewrite is involved. The workflow adds
diagnostics without changing the manual-only build/test CI.

## Use from a phone

After merging the workflow, open **Actions → Repository map → Run workflow**.
The default run creates the structural map, LOC report and duplication report.
Select **semantic_rust** for a separate, heavier Rust indexing job.
Code pushes on master and `codex/repo-map-*` branches run the structural job.

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
- `duplicates/`: jscpd copy/paste/renaming candidates on tracked project paths,
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

Python 3.11+ is required (workflow uses 3.12).

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
