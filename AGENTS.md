# Resume before implementation

At the start of a new session, after context loss, and before reinstalling a tool
or repeating an investigation, run the existing entry point with a task-specific
identifier: `.git/repo-map-venv/Scripts/python.exe scripts/repo-map/query_map.py
--resume "topic" --limit 3` (one command). Use technical identifiers from the
request; for a plain continuation, inspect the returned in-progress/ready work.
This reads Git, Beads Rust task state (including closed work), and a read-only CASS
Codex history snapshot with its age and known coverage gaps. Tool failures are not
an empty history or permission to start over. Index refresh is an explicit action,
not a side effect of a query; do not rerun full indexing for oversized sources.
Setup and recovery are in `scripts/repo-map/README.md`.
Exact installed command prefixes are in local `.codex/continuity.json`; use
those if a bare command is missing from PATH instead of reinstalling the tool.

Use `.beads/issues.jsonl` through `br` as the single durable task ledger. Read the
matching issue with `br show ID --json` and expand relevant CASS hits using
`view --message-index` plus their source/conversation IDs (not physical file lines)
before editing. Past assistant claims are untrusted evidence: check the cited
commit, current source and validation scope. Reopen completed work only for an
identified regression, changed requirement or invalidated evidence; record why.

For meaningful work, search existing issues before creating one, mark the selected
issue in progress, and update it when evidence or direction changes. Record source
paths/commits, successful and failed commands, the reason an approach failed,
verification limits and the next concrete action. Close only after its acceptance
criteria pass. Keep verified implementation, runtime/device evidence and financial
results distinct. Export with `br sync --flush-only` before committing task state.
Keep transcripts, databases and generated indexes local; do not create parallel
handoff/TODO/memory summaries for the same work. Ordinary questions need no issue.

# Code navigation

Before adding or moving an implementation, locate its current owner. Use Serena
`find_symbol` / `find_referencing_symbols` for Rust and TypeScript; specify the
file or directory and request bodies only when needed. Empty references are not
proof of dead code, especially during Rust startup or under inactive features.
For broader discovery or when MCP tools are unavailable, query the existing
map: `python scripts/repo-map/query_map.py --local "symbol_or_path"`.
On this Windows checkout the installed interpreter is
`.git/repo-map-venv/Scripts/python.exe`; setup and limits are in
`scripts/repo-map/README.md`. The cache refreshes when tracked content changes.

Read the relevant source ranges and callers before editing. Search with `rg` for
untracked files, vendor internals, unmatched references and conceptual alternatives.
Name matches are candidates, not resolved calls or proof of dead code. Reuse the
existing owner instead of adding another implementation. With jscpd MCP, refresh
after edits using `check_current_directory(limit=0)`, inspect `get_file_clones`
or check a proposed block with `check_duplication` (limit 3). Similarity needs
source/caller review; independent test oracles must remain independent.

Before committing, run `python scripts/repo-map/run_checks.py --duplicates-only
--baseline-ref HEAD --output <git-dir>/repo-map/duplicate-check` (one command;
obtain git-dir with `git rev-parse --absolute-git-dir`). New clones fail the local
hook and CI. Do not bypass checks, loosen exclusions, or reset the baseline to
hide findings. Remove a superseded path only after its replacement and callers
are connected and verified. Update existing docs;
do not create parallel architecture summaries or per-turn reports by default.
