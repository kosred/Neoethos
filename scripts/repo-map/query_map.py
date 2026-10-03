#!/usr/bin/env python3
"""Query a downloaded repository graph without loading it into an AI context."""
import argparse
import json
import sqlite3
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("database", type=Path)
parser.add_argument("query", help="Substring of a symbol, package or path")
parser.add_argument("--direction", choices=["in", "out", "both"], default="both")
parser.add_argument("--limit", type=int, default=40)
args = parser.parse_args()
if not 1 <= args.limit <= 500:
    parser.error("limit must be between 1 and 500")
with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as db:
    db.row_factory = sqlite3.Row
    result = {"metadata": dict(db.execute("SELECT key,value FROM metadata")), "matches": []}
    # Literal substring search; user text is never interpolated into SQL.
    for node in db.execute("SELECT * FROM nodes WHERE instr(lower(name),lower(?))>0 OR instr(lower(path),lower(?))>0 ORDER BY CASE WHEN lower(name)=lower(?) THEN 0 ELSE 1 END, id LIMIT ?",
                           (args.query, args.query, args.query, args.limit)).fetchall():
        item = dict(node)
        item["connections"] = []
        for direction in (["in", "out"] if args.direction == "both" else [args.direction]):
            column = "target" if direction == "in" else "source"
            for edge in db.execute(f"SELECT * FROM edges WHERE {column}=? LIMIT ?", (node["id"], args.limit)):
                item["connections"].append({"direction": direction, **dict(edge)})
        result["matches"].append(item)
    print(json.dumps(result, indent=2))
