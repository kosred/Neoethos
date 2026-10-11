#!/usr/bin/env python3
"""Retrieve bounded connections; --local maintains the existing mapper's cache."""
import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def source_digest(root):
    """Hash tracked working-tree contents, including unstaged edits and deletions."""
    digest = hashlib.sha256()
    for raw in sorted(set(git(root, "ls-files", "-z").split(b"\0")) - {b""}):
        path = root / os.fsdecode(raw)
        digest.update(raw + b"\0")
        if path.is_symlink():
            content = b"symlink\0" + os.fsencode(path.readlink())
        elif path.is_file():
            content = b"file\0" + path.read_bytes()
        else:
            content = b"missing\0"
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def local_database(root):
    root = Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).strip())
    git_dir = Path(os.fsdecode(git(root, "rev-parse", "--absolute-git-dir")).strip())
    output = git_dir / "repo-map"
    database = output / "graph.sqlite"
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    digest = source_digest(root)
    try:
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
    except sqlite3.Error:
        metadata = {}
    if metadata.get("local_source_digest") != digest or metadata.get("commit") != commit:
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "build.log"
        print("Refreshing local repository map...", file=sys.stderr)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("map_repo.py")),
                 "--root", str(root), "--output", str(output)],
                stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
        if result.returncode:
            raise RuntimeError(f"Map refresh failed; no stale result served. See {log_path}")
        if digest != source_digest(root) or commit != git(root, "rev-parse", "HEAD").decode().strip():
            raise RuntimeError("Source changed during indexing; retry the query.")
        with closing(sqlite3.connect(database)) as db, db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", ("local_source_digest", digest))
    return database


def query_graph(database, query, direction="both", limit=8, edge_limit=12, exact=False):
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        result = {"metadata": dict(db.execute("SELECT key,value FROM metadata")), "matches": []}
        where = "lower(name)=lower(?)" if exact else "instr(lower(name),lower(?))>0 OR instr(lower(path),lower(?))>0"
        params = [query] if exact else [query, query]
        nodes = db.execute(f"SELECT * FROM nodes WHERE {where} ORDER BY "
                           "CASE WHEN lower(name)=lower(?) THEN 0 ELSE 1 END, "
                           "CASE WHEN kind='call_site' THEN 1 ELSE 0 END, id LIMIT ?",
                           [*params, query, limit + 1]).fetchall()
        result["matches_truncated"] = len(nodes) > limit
        remaining = edge_limit
        for node in nodes[:limit]:
            item = {**dict(node), "connections": [], "connections_truncated": False}
            for side in (["in", "out"] if direction == "both" else [direction]):
                column, peer = ("target", "source") if side == "in" else ("source", "target")
                edges = db.execute(
                    f"SELECT e.*, n.name AS peer_name, n.path AS peer_path, n.line AS peer_line "
                    f"FROM edges e JOIN nodes n ON n.id=e.{peer} WHERE e.{column}=? "
                    "ORDER BY e.kind, e.path, e.line, e.source, e.target LIMIT ?",
                    (node["id"], remaining + 1)).fetchall()
                item["connections_truncated"] |= len(edges) > remaining
                selected = edges[:remaining]
                item["connections"].extend({"direction": side, **dict(edge)} for edge in selected)
                remaining -= len(selected)
            result["matches"].append(item)
        return result


def compact(result, max_chars=6000):
    def short(value):
        return " ".join(str(value).split())[:180]
    meta = result["metadata"]
    lines = [f"commit={meta.get('commit')} dirty={meta.get('working_tree_dirty')}",
             "Scope: tracked files; vendor inventory only. Name candidates are NOT resolved calls.",
             "Untracked files are not indexed; use rg for those and for unmatched references."]
    for node in result["matches"]:
        lines.append(f"{node['path']}:{node['line']}-{node['end_line']} [{node['kind']}] {short(node['name'])}")
        if node["context"]:
            lines.append("  context: " + short(node["context"]))
        for edge in node["connections"]:
            lines.append(f"  {edge['direction']} {edge['kind']} [{edge['evidence']}] "
                         f"{edge['path']}:{edge['line']} -> {short(edge['peer_name'])}")
        if node["connections_truncated"]:
            lines.append("  ... more connections; narrow query/direction or raise --edge-limit")
    if result["matches_truncated"]:
        lines.append("... more matches; use --exact or narrow the query")
    if not result["matches"]:
        lines.append("No indexed match; this is not proof that an implementation is absent.")
    text = "\n".join(lines)
    suffix = "\n... output capped; narrow the query"
    return text if len(text) <= max_chars else text[:max_chars - len(suffix)] + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, nargs="?")
    parser.add_argument("query", help="Substring of a symbol, package or path")
    parser.add_argument("--local", action="store_true", help="Refresh/reuse .git/repo-map locally")
    parser.add_argument("--direction", choices=["in", "out", "both"], default="both")
    parser.add_argument("--limit", type=int, default=8, help="Maximum matching nodes")
    parser.add_argument("--edge-limit", type=int, default=12, help="Total connection budget across all matches")
    parser.add_argument("--exact", action="store_true", help="Match a symbol name exactly")
    parser.add_argument("--format", choices=["text", "json"], help="Default: text for --local, JSON for an explicit database")
    parser.add_argument("--max-chars", type=int, default=6000, help="Text output character budget")
    args = parser.parse_args()
    if args.local == (args.database is not None):
        parser.error("Use either --local QUERY or DATABASE QUERY")
    if not 1 <= args.limit <= 500 or not 0 <= args.edge_limit <= 500 or args.max_chars < 300:
        parser.error("Require limit 1..500, edge-limit 0..500 and max-chars >= 300")
    try:
        database = local_database(Path.cwd()) if args.local else args.database
        result = query_graph(database, args.query, args.direction, args.limit, args.edge_limit, args.exact)
    except (OSError, sqlite3.Error, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
    output_format = args.format or ("text" if args.local else "json")
    print(compact(result, args.max_chars) if output_format == "text" else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
