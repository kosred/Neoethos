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
