import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import map_repo as mapper
from run_checks import run


class GraphEvidenceTests(unittest.TestCase):
    def test_trait_methods_cfg_macros_and_generic_call_are_not_falsely_resolved(self):
        source = b'''#[cfg(feature="gpu")]
        mod gpu {
            pub trait Run { fn execute(&self); }
            impl Run for Engine { fn execute(&self) { helper::<Engine>(); self.verify(); } }
            impl Other { fn execute(&self) { self.verify(); } }
            fn helper<T>() { external!(); }
        }'''
        graph = mapper.Graph()
        graph.node("file:sample.rs", "file", "sample.rs", "sample.rs")
        mapper.parse_source(graph, "sample.rs", source, "rust")
        mapper.resolve_candidates(graph)
        helpers = [n for n in graph.nodes.values() if n["name"] == "helper"]
        self.assertEqual(len(helpers), 1)
        self.assertIn('feature="gpu"', helpers[0]["context"])
        self.assertTrue(any(e[2] == "possible_target" and e[1] == helpers[0]["id"] for e in graph.edges))
        self.assertTrue(any(n["kind"] == "macro_invocation" for n in graph.nodes.values()))
        self.assertFalse(any(e[3] == "scip" for e in graph.edges))
        self.assertEqual(graph.coverage[0]["status"], "syntax_only")
        self.assertEqual(len([n for n in graph.nodes.values() if n["name"] == "execute"]), 3)

    def test_unresolved_external_and_grammar_errors_remain_visible(self):
        graph = mapper.Graph()
        graph.node("file:x.rs", "file", "x.rs", "x.rs")
        mapper.parse_source(graph, "x.rs", b"fn f() { not_in_this_repo(); } fn broken( {", "rust")
        mapper.resolve_candidates(graph)
        self.assertGreater(graph.coverage[0]["parse_errors"], 0)
        self.assertTrue(any(n["kind"] == "call_site" and n["name"] == "not_in_this_repo" for n in graph.nodes.values()))
        self.assertFalse(any(e[2] == "possible_target" for e in graph.edges))

    def test_workspace_dependency_inheritance_and_target_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text('[workspace.dependencies]\nshared={path="crates/shared"}\n')
            (root / "crates/app").mkdir(parents=True)
            (root / "crates/shared").mkdir()
            (root / "crates/app/Cargo.toml").write_text('[package]\nname="app"\n[target.\'cfg(windows)\'.dependencies]\nshared={workspace=true}\n')
            (root / "crates/shared/Cargo.toml").write_text('[package]\nname="shared"\n')
            graph = mapper.Graph()
            mapper.manifest_graph(graph, root, ["Cargo.toml", "crates/app/Cargo.toml", "crates/shared/Cargo.toml"])
            dependencies = [e for e in graph.edges if e[2] == "declared_dependency"]
            self.assertEqual(len(dependencies), 1)
            self.assertEqual(dependencies[0][1], "package:crates/shared/Cargo.toml")
            self.assertEqual(json.loads(dependencies[0][6])["target"], "cfg(windows)")

    def test_scip_local_symbols_are_document_scoped_and_references_are_not_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index.json"
            index.write_text(json.dumps({"documents": [
                {"relative_path": "a.rs", "occurrences": [{"symbol": "local 0", "range": [0, 0, 1], "symbol_roles": 1}]},
                {"relativePath": "b.rs", "occurrences": [{"symbol": "local 0", "range": [3, 0, 1], "symbolRoles": 0}]},
            ]}))
            graph = mapper.Graph()
            mapper.import_scip(graph, index)
            self.assertIn("scip:a.rs:local 0", graph.nodes)
            self.assertIn("scip:b.rs:local 0", graph.nodes)
            self.assertEqual({e[2] for e in graph.edges}, {"defines", "references"})
            summary = mapper.save(graph, Path(tmp) / "out", "abc", dirty=True)
            self.assertFalse(summary["whole_repo_semantically_verified"])
            with sqlite3.connect(Path(tmp) / "out/graph.sqlite") as db:
                self.assertEqual(db.execute("SELECT count(*) FROM edges").fetchone()[0], 2)
                self.assertEqual(db.execute("SELECT value FROM metadata WHERE key='working_tree_dirty'").fetchone()[0], "True")

    def test_tool_failure_is_recorded_without_reading_stdout_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = run("missing", ["/definitely-absent-repo-map-tool"], Path(tmp), 1)
            self.assertEqual(record["status"], "unavailable")
            record = run("failed", ["python3", "-c", "raise SystemExit(7)"], Path(tmp), 1)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
