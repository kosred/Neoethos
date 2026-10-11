# Code navigation

For cross-module discovery or before adding an implementation, query the existing
map first: `python scripts/repo-map/query_map.py --local "symbol_or_path"`.
On this Windows checkout the installed interpreter is
`.git/repo-map-venv/Scripts/python.exe`; setup and limits are in
`scripts/repo-map/README.md`. The cache refreshes when tracked content changes.

Read the relevant source ranges and callers before editing. Search with `rg` for
untracked files, vendor internals, unmatched references and conceptual alternatives.
Name matches are candidates, not resolved calls or proof of dead code. Reuse the
existing owner instead of adding another implementation. Update existing docs;
do not create parallel architecture summaries or per-turn reports by default.
