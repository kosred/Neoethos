# Repository map

Read-only, repeatable mapping on GitHub-hosted runners. No AI API, broker session,
GPU execution or production source rewrite is involved. The workflow adds
diagnostics without changing the manual-only build/test CI.

## Use from a phone

After merging the workflow, open **Actions → Repository map → Run workflow**.
The default run creates the structural map, LOC report and duplication report.
Select **semantic_rust** for a separate, heavier Rust indexing job. The first
`codex/repo-map-*` branch push runs both jobs to validate setup before merging.
Normal master code pushes run only the lightweight structural job.

Open the run summary, download its `repository-map-<commit>` artifact, unzip it
and open `index.html` in a browser. It is a standalone searchable connection
viewer: no web server or external JavaScript service. Artifact download/viewing
support varies between phone browsers. The SQLite/JSON files are also usable by
code assistants without loading the entire source repository into context.

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

The syntax graph parses Rust, C/C++/CUDA and TypeScript/TSX files. Other languages
and vendor files are inventoried and explicitly marked, not silently certified.
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
