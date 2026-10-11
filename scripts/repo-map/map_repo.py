#!/usr/bin/env python3
"""Read-only repository inventory and evidence-labelled graph. No AI or broker calls."""
import argparse
import collections
import base64
import gzip
import hashlib
import fnmatch
import json
import sqlite3
import subprocess
import tomllib
import yaml
from contextlib import closing
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_cpp
import tree_sitter_rust
import tree_sitter_typescript
import tree_sitter_python
import tree_sitter_bash

PARSERS = {
    "rust": Parser(Language(tree_sitter_rust.language())),
    "cpp": Parser(Language(tree_sitter_cpp.language())),
    "typescript": Parser(Language(tree_sitter_typescript.language_typescript())),
    "tsx": Parser(Language(tree_sitter_typescript.language_tsx())),
    "python": Parser(Language(tree_sitter_python.language())),
    "bash": Parser(Language(tree_sitter_bash.language())),
}
LANGUAGES = {".rs": "rust", ".cu": "cpp", ".cuh": "cpp", ".c": "cpp",
             ".h": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
             ".ts": "typescript", ".tsx": "tsx", ".js": "typescript",
             ".py": "python", ".sh": "bash"}
DEFINITIONS = {"function_item", "function_signature_item", "struct_item", "enum_item",
               "trait_item", "type_item", "const_item", "static_item", "mod_item",
               "function_definition", "class_specifier", "struct_specifier",
               "function_declaration", "method_definition", "class_declaration",
               "interface_declaration", "type_alias_declaration", "class_definition"}
CALLS = {"call_expression", "call", "command"}


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
    if node.type in {"identifier", "field_identifier", "property_identifier", "word", "command_name"}:
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
        self.semantic_source_commit = None

    def node(self, key, kind, name, path="", line=0, end_line=0, context=""):
        self.nodes.setdefault(key, {"id": key, "kind": kind, "name": name, "path": path,
                                   "line": line, "end_line": end_line, "context": context})
        return key

    def edge(self, source, target, kind, evidence, path="", line=0, detail=""):
        self.edges.add((source, target, kind, evidence, path, line, detail))


def parse_source(graph, path, data, language):
    """Syntax edges are NEVER presented as resolved runtime calls."""
    tree = PARSERS[language].parse(data)
    error_nodes = [n for n in walk(tree.root_node) if n.type == "ERROR" or n.is_missing]
    errors = len(error_nodes)
    graph.coverage.append({"path": path, "language": language, "parse_errors": errors,
                           "error_locations": [{"line": n.start_point.row + 1,
                                                "column": n.start_point.column + 1,
                                                "kind": n.type} for n in error_nodes],
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
            target = node.child_by_field_name("function") or node.child_by_field_name("name")
            if target:
                raw = text(target, data)[:240]
                name = called_name(target, data)
                graph.pending.append((owner, name or "<unresolved>", raw, path, node.start_point.row + 1))
        if node.type in {"macro_invocation", "use_declaration", "import_statement", "import_from_statement"}:
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
        if 'workspace' not in document:
            continue
        ws = document['workspace']
        key = graph.node('workspace:' + path, 'cargo_workspace', path, path)
        parent = Path(path).parent
        patterns = ws.get('members', ['.'] if document.get('package') else [])
        for member, member_document in manifests.items():
            folder = Path(member).parent
            try:
                relative = folder.relative_to(parent).as_posix()
            except ValueError:
                continue
            if any(fnmatch.fnmatch(relative, pattern) for pattern in patterns) and not any(
                    fnmatch.fnmatch(relative, pattern) for pattern in ws.get('exclude', [])):
                target = graph.node('package:' + member, 'package', member_document.get('package', {}).get('name', member), member)
                graph.edge(key, target, 'workspace_member', 'toml', path, 0,
                           json.dumps({'default_members': ws.get('default-members', 'all declared members')}))
        for registry, patches in document.get('patch', {}).items():
            for name, spec in patches.items():
                if isinstance(spec, dict) and 'path' in spec:
                    target_path = (parent / spec['path'] / 'Cargo.toml').as_posix()
                    target = graph.node('package:' + target_path, 'package', name, target_path)
                    graph.edge(key, target, 'patch_override', 'toml', path, 0, registry)
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


def jsonc_loads(data):
    """Allow TS-config comments/trailing commas while preserving quoted strings."""
    source = data.decode()
    cleaned, index, quoted, escaped = [], 0, False, False
    while index < len(source):
        char = source[index]
        if quoted:
            cleaned.append(char)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
        elif source.startswith('//', index):
            end = source.find('\n', index)
            end = len(source) if end < 0 else end
            cleaned.extend(' ' * (end - index)); index = end
            continue
        elif source.startswith('/*', index):
            end = source.find('*/', index + 2)
            if end < 0:
                raise ValueError('Unterminated JSONC comment')
            cleaned.extend('\n' if c == '\n' else ' ' for c in source[index:end + 2])
            index = end + 2
            continue
        elif char == ',':
            next_index = index + 1
            while next_index < len(source) and source[next_index].isspace():
                next_index += 1
            if next_index < len(source) and source[next_index] in '}]':
                char = ' '
        cleaned.append(char); index += 1
    # Remove commas preceding comments and a closing delimiter after comment stripping.
    clean = ''.join(cleaned)
    if clean != source:
        return jsonc_loads(clean.encode())
    return json.loads(clean)


def parse_configuration(graph, path, data):
    """Map parsed configuration structure, never execute scripts or expand expressions."""
    suffix = Path(path).suffix
    try:
        if suffix == ".toml":
            document = tomllib.loads(data.decode())
        elif suffix == ".json":
            document = jsonc_loads(data) if Path(path).name.startswith('tsconfig') else json.loads(data)
        else:
            # BaseLoader keeps GitHub's 'on' as a string and never constructs Python objects.
            document = yaml.load(data.decode(), Loader=yaml.BaseLoader)
    except (ValueError, UnicodeError, yaml.YAMLError) as error:
        graph.coverage.append({"path": path, "status": "configuration_error", "detail": str(error)})
        return
    graph.coverage.append({"path": path, "status": "configuration_parsed", "format": suffix})
    if not isinstance(document, dict):
        return
    for name, value in document.items():
        key = graph.node("config:" + path + ":" + str(name), "configuration_section", str(name), path)
        graph.edge("file:" + path, key, "configuration_section", "parsed_configuration", path)
    if path.startswith(".github/workflows/"):
        for job, spec in (document.get("jobs") or {}).items():
            if not isinstance(spec, dict):
                continue
            key = graph.node("job:" + path + ":" + str(job), "workflow_job", str(job), path,
                             context=json.dumps({k: spec[k] for k in ("if", "runs-on", "permissions") if k in spec}))
            graph.edge("file:" + path, key, "declares_job", "parsed_configuration", path)
            needs = spec.get("needs", [])
            if isinstance(needs, str):
                needs = [needs]
            for dependency in needs:
                target = graph.node("job:" + path + ":" + str(dependency), "workflow_job", str(dependency), path)
                graph.edge(key, target, "needs_job", "parsed_configuration", path)
            for step in spec.get("steps", []):
                if isinstance(step, dict) and "uses" in step:
                    action = str(step["uses"])
                    target = graph.node("action:" + action, "workflow_action", action)
                    graph.edge(key, target, "uses_action", "parsed_configuration", path)
    if Path(path).name == "package.json":
        for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            for name, version in (document.get(section) or {}).items():
                target = graph.node("npm:" + name + ":" + str(version), "npm_dependency", name,
                                    context=str(version))
                graph.edge("file:" + path, target, "declared_npm_dependency", "parsed_configuration", path, 0, section)
        for name, command in (document.get("scripts") or {}).items():
            key = graph.node("npm-script:" + path + ":" + name, "npm_script", name, path, context=str(command))
            graph.edge("file:" + path, key, "declares_script", "parsed_configuration", path)


def field(obj, snake):
    parts = snake.split("_")
    camel = parts[0] + "".join(p.title() for p in parts[1:])
    return obj.get(snake, obj.get(camel))


def import_scip(graph, index_path, prefix='', profile='repo-nightly-default-features'):
    index = json.loads(index_path.read_text())
    tracked = {f["path"] for f in graph.files}
    for document in index.get("documents", []):
        path = field(document, "relative_path")
        if path and prefix:
            import posixpath
            path = posixpath.normpath(posixpath.join(prefix, path))
        if not path or path.startswith("vendor/"):
            continue
        if tracked and path not in tracked:
            graph.coverage.append({"path": path, "status": "scip_untracked_not_imported"})
            continue
        file_node = graph.node("file:" + path, "file", path, path)
        graph.coverage.append({"path": path, "status": "scip_indexed",
                               "profile": profile, "runtime_verified": False,
                               "occurrences": len(document.get("occurrences", [])),
                               "definitions": sum(bool((field(o, "symbol_roles") or 0) & 1)
                                                  for o in document.get("occurrences", [])),
                               "diagnostics": [d for o in document.get("occurrences", [])
                                               for d in o.get("diagnostics", [])]})
        def symbol_id(symbol):
            return "scip:" + (path + ":" + profile + ":" if symbol.startswith("local ") else "") + symbol
        for symbol in document.get("symbols", []):
            raw = symbol.get("symbol", "")
            if not raw:
                continue
            key = graph.node(symbol_id(raw), "scip_symbol", field(symbol, "display_name") or raw,
                             context=profile)
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
                             context=profile)
            span = occurrence.get("range", [])
            line = span[0] + 1 if span else 0
            role = field(occurrence, "symbol_roles") or 0
            if role & 1 and not graph.nodes[key]["path"]:
                graph.nodes[key].update(path=path, line=line, end_line=line)
            graph.edge(file_node, key, "defines" if role & 1 else "references", "scip", path, line,
                       profile + ": Static symbol reference; not automatically a runtime call")
    definitions = collections.defaultdict(set)
    for source, target, kind, evidence, path, line, _ in graph.edges:
        if kind == 'defines' and evidence == 'scip':
            definitions[target].add(path)
    for key, locations in definitions.items():
        if len(locations) > 1:
            graph.nodes[key].update(path='', line=0, end_line=0,
                                    context='Ambiguous SCIP definition across documents; inspect defines edges')


def save(graph, output, commit, dirty=False):
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "graph.sqlite"
    db_path.unlink(missing_ok=True)
    with closing(sqlite3.connect(db_path)) as db, db:
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
                       ("schema", "1"), ("scope", "tracked files; vendor inventoried, syntax excluded"),
                       ("semantic_source_commit", graph.semantic_source_commit or "")])
        db.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?)",
                       ((identifiers[key], key, n["kind"], n["name"], n["path"], n["line"], n["end_line"], n["context"])
                        for key, n in graph.nodes.items()))
        db.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?,?)",
                       ((identifiers[a], identifiers[b], *rest) for a, b, *rest in sorted(graph.edges)))
    summary = {"commit": commit, "working_tree_dirty": dirty, "files": len(graph.files), "nodes": len(graph.nodes),
               "edges": len(graph.edges), "whole_repo_semantically_verified": False,
               "semantic_source_commit": graph.semantic_source_commit,
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
    scip_definitions = {b for _, b, kind, evidence, *_ in graph.edges if kind == 'defines' and evidence == 'scip'}
    visible = {key for key, n in graph.nodes.items() if n["kind"] != "call_site" and
               (n['kind'] != 'scip_symbol' or (key in scip_definitions and ':local ' not in key))}
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
    parser.add_argument("--scip-json", type=Path, action='append', default=[])
    parser.add_argument("--scip-prefix", action='append', default=[])
    parser.add_argument("--scip-profile", action='append', default=[])
    parser.add_argument("--scip-source-commit")
    args = parser.parse_args()
    root = args.root.resolve()
    paths = sorted(p.decode() for p in git(root, "ls-files", "-z").split(b"\0") if p)
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    graph = Graph()
    graph.semantic_source_commit = args.scip_source_commit
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
        if not vendor and not binary and Path(path).suffix in {".toml", ".json", ".yml", ".yaml"}:
            parse_configuration(graph, path, data)
        elif vendor or binary or not language:
            graph.coverage.append({"path": path, "status": "vendor_inventory_only" if vendor else
                                   "binary_inventory_only" if binary else "inventory_only"})
        else:
            parse_source(graph, path, data, language)
    manifest_graph(graph, root, paths)
    resolve_candidates(graph)
    for i, index_path in enumerate(args.scip_json):
        import_scip(graph, index_path, args.scip_prefix[i] if i < len(args.scip_prefix) else '',
                    args.scip_profile[i] if i < len(args.scip_profile) else 'repo-nightly-default-features')
    dirty = bool(git(root, "status", "--porcelain", "--untracked-files=no"))
    summary = save(graph, args.output, commit, dirty)
    from report_map import report
    report(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
