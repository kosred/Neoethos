#!/usr/bin/env python3
"""Read-only repository inventory and evidence-labelled graph. No AI or broker calls."""
import argparse
import collections
import base64
import gzip
import hashlib
import json
import sqlite3
import subprocess
import tomllib
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_cpp
import tree_sitter_rust
import tree_sitter_typescript

PARSERS = {
    "rust": Parser(Language(tree_sitter_rust.language())),
    "cpp": Parser(Language(tree_sitter_cpp.language())),
    "typescript": Parser(Language(tree_sitter_typescript.language_typescript())),
    "tsx": Parser(Language(tree_sitter_typescript.language_tsx())),
}
LANGUAGES = {".rs": "rust", ".cu": "cpp", ".cuh": "cpp", ".c": "cpp",
             ".h": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
             ".ts": "typescript", ".tsx": "tsx"}
DEFINITIONS = {"function_item", "function_signature_item", "struct_item", "enum_item",
               "trait_item", "type_item", "const_item", "static_item", "mod_item",
               "function_definition", "class_specifier", "struct_specifier",
               "function_declaration", "method_definition", "class_declaration",
               "interface_declaration", "type_alias_declaration"}
CALLS = {"call_expression"}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def text(node, data):
    return data[node.start_byte:node.end_byte].decode("utf-8", "replace")


def walk(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.named_children))


def name_of(node, data):
    name = node.child_by_field_name("name")
    if name:
        return text(name, data)
    declarator = node.child_by_field_name("declarator")
    if declarator:
        for child in walk(declarator):
            if child.type in {"identifier", "field_identifier"}:
                return text(child, data)
    return None


def called_name(node, data):
    if node.type in {"identifier", "field_identifier", "property_identifier"}:
        return text(node, data)
    for field_name in ("field", "property", "name", "function"):
        child = node.child_by_field_name(field_name)
        if child:
            return called_name(child, data)
    return None


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = set()
        self.pending = []
        self.files = []
        self.coverage = []

    def node(self, key, kind, name, path="", line=0, end_line=0, context=""):
        self.nodes.setdefault(key, {"id": key, "kind": kind, "name": name, "path": path,
                                   "line": line, "end_line": end_line, "context": context})
        return key

    def edge(self, source, target, kind, evidence, path="", line=0, detail=""):
        self.edges.add((source, target, kind, evidence, path, line, detail))


def parse_source(graph, path, data, language):
    """Syntax edges are NEVER presented as resolved runtime calls."""
    tree = PARSERS[language].parse(data)
    errors = sum(n.type == "ERROR" or n.is_missing for n in walk(tree.root_node))
    graph.coverage.append({"path": path, "language": language, "parse_errors": errors,
                           "status": "syntax_with_errors" if errors else "syntax_only"})
    # Attributes remain literal conditions; no cfg/feature branch is assumed active.
    def visit(node, owner, context):
        if node.type in {"impl_item", "mod_item", "class_declaration", "trait_item"}:
            head = text(node, data).split("{", 1)[0].strip()
            context = (context + " / " + head)[-1500:]
        if node.type in DEFINITIONS:
            name = name_of(node, data)
            if name:
                conditions = []
                previous = node.prev_named_sibling
                while previous and previous.type in {"attribute_item", "line_comment", "block_comment"}:
                    if previous.type == "attribute_item":
                        conditions.append(text(previous, data))
                    previous = previous.prev_named_sibling
                ctx = context + " " + " ".join(reversed(conditions))
                key = f"syntax:{path}:{node.start_byte}"
                graph.node(key, node.type, name, path, node.start_point.row + 1,
                           node.end_point.row + 1, ctx.strip())
                graph.edge(owner, key, "contains", "syntax", path, node.start_point.row + 1)
                owner, context = key, ctx
        if node.type in CALLS:
            target = node.child_by_field_name("function")
            if target:
                raw = text(target, data)[:240]
                name = called_name(target, data)
                graph.pending.append((owner, name or "<unresolved>", raw, path, node.start_point.row + 1))
        if node.type in {"macro_invocation", "use_declaration", "import_statement"}:
            raw = text(node, data)
            key = graph.node(f"statement:{path}:{node.start_byte}", node.type, raw[:300],
                             path, node.start_point.row + 1, node.end_point.row + 1, context)
            graph.edge(owner, key, "declares", "syntax", path, node.start_point.row + 1)
        for child in node.named_children:
            visit(child, owner, context)
    visit(tree.root_node, "file:" + path, "")


def resolve_candidates(graph):
    names = collections.defaultdict(list)
    for key, node in graph.nodes.items():
        if node["kind"] in DEFINITIONS:
            names[node["name"]].append(key)
    for owner, name, raw, path, line in graph.pending:
        # Preserve every call occurrence, including external/dynamic/unresolved targets.
        occurrence = graph.node(f"call:{path}:{line}:{len(graph.edges)}", "call_site", raw,
                                path, line, line)
        graph.edge(owner, occurrence, "calls_syntax", "syntax", path, line, raw)
        candidates = names.get(name, [])
        # High fan-out names are retained as unresolved, never arbitrarily bound.
        if len(candidates) <= 20:
            for candidate in candidates:
                graph.edge(occurrence, candidate, "possible_target", "name_candidate",
                           path, line, "Unresolved by syntax; not a proof of use or dead code")


def manifest_graph(graph, root, paths):
    manifests = {}
    for path in paths:
        if path.endswith("Cargo.toml") and not path.startswith("vendor/"):
            try:
                manifests[path] = tomllib.loads((root / path).read_text())
            except (ValueError, UnicodeError) as error:
                graph.coverage.append({"path": path, "status": "manifest_error", "detail": str(error)})
    workspace = manifests.get("Cargo.toml", {}).get("workspace", {}).get("dependencies", {})
    for path, document in manifests.items():
        package = document.get("package", {})
        if not package:
            continue
        key = graph.node("package:" + path, "package", package.get("name", path), path)
        graph.edge(key, "file:" + path, "manifest", "toml", path)
        sections = [("", document)] + [(target, cfg) for target, cfg in document.get("target", {}).items()]
        for target, section in sections:
            for kind in ("dependencies", "dev-dependencies", "build-dependencies"):
                for alias, spec in section.get(kind, {}).items():
                    original = spec
                    if isinstance(spec, dict) and spec.get("workspace"):
                        spec = workspace.get(alias, spec)
                        folder = root
                    else:
                        folder = (root / path).parent
                    if isinstance(spec, dict) and "path" in spec:
                        resolved = (folder / spec["path"] / "Cargo.toml").resolve()
                        try:
                            dependency_path = resolved.relative_to(root).as_posix()
                        except ValueError:
                            dependency_path = str(resolved)
                        dep = graph.node("package:" + dependency_path, "package",
                                         manifests.get(dependency_path, {}).get("package", {}).get("name", alias),
                                         dependency_path)
                    else:
                        dep = graph.node("dependency:" + alias + ":" + json.dumps(spec, sort_keys=True),
                                         "external_dependency", alias)
                    graph.edge(key, dep, "declared_dependency", "toml", path, 0,
                               json.dumps({"kind": kind, "target": target, "spec": original}, sort_keys=True))
        for name, values in document.get("features", {}).items():
            feature = graph.node(key + ":feature:" + name, "feature", name, path,
                                 context=json.dumps(values))
            graph.edge(key, feature, "declares_feature", "toml", path)


def field(obj, snake):
    parts = snake.split("_")
    camel = parts[0] + "".join(p.title() for p in parts[1:])
    return obj.get(snake, obj.get(camel))


def import_scip(graph, index_path):
    index = json.loads(index_path.read_text())
    tracked = {f["path"] for f in graph.files}
    for document in index.get("documents", []):
        path = field(document, "relative_path")
        if not path or path.startswith("vendor/"):
            continue
        if tracked and path not in tracked:
            graph.coverage.append({"path": path, "status": "scip_untracked_not_imported"})
            continue
        file_node = graph.node("file:" + path, "file", path, path)
        graph.coverage.append({"path": path, "status": "scip_indexed",
                               "profile": "repo-nightly-default-features", "runtime_verified": False})
        def symbol_id(symbol):
            return "scip:" + (path + ":" if symbol.startswith("local ") else "") + symbol
        for symbol in document.get("symbols", []):
            raw = symbol.get("symbol", "")
            if not raw:
                continue
            key = graph.node(symbol_id(raw), "scip_symbol", field(symbol, "display_name") or raw,
                             context="repo-nightly-default-features")
            for relationship in symbol.get("relationships", []):
                other_raw = relationship.get("symbol", "")
                if other_raw:
                    other = graph.node(symbol_id(other_raw), "scip_symbol", other_raw)
                    graph.edge(key, other, "symbol_relationship", "scip", path, 0,
                               json.dumps(relationship, sort_keys=True))
        for occurrence in document.get("occurrences", []):
            raw = occurrence.get("symbol", "")
            if not raw:
                continue
            key = graph.node(symbol_id(raw), "scip_symbol", raw,
                             context="repo-nightly-default-features")
            span = occurrence.get("range", [])
            line = span[0] + 1 if span else 0
            role = field(occurrence, "symbol_roles") or 0
            if role & 1 and not graph.nodes[key]["path"]:
                graph.nodes[key].update(path=path, line=line, end_line=line)
            graph.edge(file_node, key, "defines" if role & 1 else "references", "scip", path, line,
                       "Static symbol reference; not automatically a runtime call")


def save(graph, output, commit, dirty=False):
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "graph.sqlite"
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as db:
        db.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE nodes(id INTEGER PRIMARY KEY, stable_key TEXT, kind TEXT, name TEXT, path TEXT,
                               line INTEGER, end_line INTEGER, context TEXT);
            CREATE TABLE edges(source INTEGER, target INTEGER, kind TEXT, evidence TEXT,
                               path TEXT, line INTEGER, detail TEXT);
            CREATE INDEX node_name ON nodes(name); CREATE INDEX node_path ON nodes(path);
            CREATE INDEX edge_source ON edges(source); CREATE INDEX edge_target ON edges(target);
        """)
        identifiers = {key: index for index, key in enumerate(sorted(graph.nodes))}
        db.executemany("INSERT INTO metadata VALUES (?, ?)", [("commit", commit), ("working_tree_dirty", str(dirty)),
                       ("schema", "1"), ("scope", "tracked files; vendor inventoried, syntax excluded")])
        db.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?)",
                       ((identifiers[key], key, n["kind"], n["name"], n["path"], n["line"], n["end_line"], n["context"])
                        for key, n in graph.nodes.items()))
        db.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?,?)",
                       ((identifiers[a], identifiers[b], *rest) for a, b, *rest in sorted(graph.edges)))
    summary = {"commit": commit, "working_tree_dirty": dirty, "files": len(graph.files), "nodes": len(graph.nodes),
               "edges": len(graph.edges), "whole_repo_semantically_verified": False,
               "file_groups": dict(collections.Counter(f["group"] for f in graph.files)),
               "physical_text_lines": sum(f["lines"] for f in graph.files),
               "coverage": dict(collections.Counter(c["status"] for c in graph.coverage)),
               "edge_evidence": dict(collections.Counter(e[3] for e in graph.edges)),
               "limitations": ["Name candidates are not resolved calls or dead-code proof.",
                   "Syntax includes inactive cfg branches and tests; macros are not expanded.",
                   "CUDA parsed as C++ can contain grammar errors; these remain recorded.",
                   "SCIP covers its indexed Rust configuration only; no GPU/runtime execution proof.",
                   "No automatic deletions, model API calls, or broker access."]}
    for name, value in [("summary.json", summary), ("inventory.json", graph.files),
                        ("coverage.json", graph.coverage)]:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    hashes = collections.defaultdict(list)
    for item in graph.files:
        if item["group"] == "project" and item["lines"] >= 20:
            hashes[item["sha256"]].append(item["path"])
    (output / "identical-files.json").write_text(json.dumps([v for v in hashes.values() if len(v) > 1], indent=2))
    # Standalone HTML: no CDN, server, fetch(), model API, or local-file CORS issue.
    visible = {key for key, n in graph.nodes.items() if n["kind"] != "call_site"}
    call_owners = {b: a for a, b, kind, *_ in graph.edges if kind == "calls_syntax"}
    compact_edges = set()
    for a, b, kind, evidence, path, line, _ in graph.edges:
        if kind == "possible_target":
            a, kind = call_owners[a], "possible_call"
        if a in visible and b in visible:
            compact_edges.add((identifiers[a], identifiers[b], kind, evidence, path, line))
    data = {"summary": summary,
            "nodes": [{**graph.nodes[key], "id": identifiers[key]} for key in sorted(visible)],
            "edges": sorted(compact_edges)}
    # The full call-site graph remains in SQLite; the phone viewer is a compressed symbol overview.
    for n in data["nodes"]:
        n.pop("context")
    payload = json.dumps(base64.b64encode(gzip.compress(json.dumps(data).encode(), mtime=0)).decode())
    template = Path(__file__).with_name("viewer.html").read_text()
    (output / "index.html").write_text(template.replace("/*REPO_MAP_DATA*/", payload))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scip-json", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = sorted(p.decode() for p in git(root, "ls-files", "-z").split(b"\0") if p)
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    graph = Graph()
    for path in paths:
        file_node = graph.node("file:" + path, "file", path, path)
        source = root / path
        if source.is_symlink():
            graph.coverage.append({"path": path, "status": "symlink_not_followed"})
            data = str(source.readlink()).encode()
            graph.files.append({"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                "lines": 0, "group": "vendor" if path.startswith("vendor/") else "project",
                                "binary": False, "symlink": True})
            continue
        data = source.read_bytes()
        binary = b"\0" in data
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            binary = True
        vendor = path.startswith("vendor/")
        graph.files.append({"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                            "lines": 0 if binary else data.count(b"\n") + int(bool(data) and not data.endswith(b"\n")),
                            "group": "vendor" if vendor else "project", "binary": binary})
        language = LANGUAGES.get(Path(path).suffix)
        if vendor or binary or not language:
            graph.coverage.append({"path": path, "status": "vendor_inventory_only" if vendor else
                                   "binary_inventory_only" if binary else "inventory_only"})
        else:
            parse_source(graph, path, data, language)
    manifest_graph(graph, root, paths)
    resolve_candidates(graph)
    if args.scip_json:
        import_scip(graph, args.scip_json)
    dirty = bool(git(root, "status", "--porcelain", "--untracked-files=no"))
    summary = save(graph, args.output, commit, dirty)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
