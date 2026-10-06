import json
import sqlite3
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import map_repo as mapper
from run_checks import run
from report_map import cycles
from pack_graph import pack, restore
from reuse_scip import check_source, validate_profile


class GraphEvidenceTests(unittest.TestCase):
    def test_profile_provenance_rejects_stale_commit_wrong_workspace_or_profile(self):
        valid = {'source_commit': 'current', 'prefix': 'mcp', 'profile': 'mcp'}
        validate_profile(valid, 'current', 'mcp', 'mcp')
        for field, value in [('source_commit', 'old'), ('prefix', 'mesh'), ('profile', 'cpu-static')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_profile({**valid, field: value}, 'current', 'mcp', 'mcp')

    def test_tsconfig_comments_and_trailing_commas_preserve_strings(self):
        self.assertEqual(mapper.jsonc_loads(b'{"url":"https://host/a,}",/* c */"values":[1, // c\n],}'),
                         {'url': 'https://host/a,}', 'values': [1]})
        with self.assertRaises(ValueError):
            mapper.jsonc_loads(b'{/* missing end')

    def test_multiple_scip_profiles_keep_local_symbols_separate_and_global_definitions_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / 'index.json'
            index.write_text(json.dumps({'documents': [
                {'relative_path': 'src/lib.rs', 'occurrences': [{'symbol': 'local 0', 'range': [0, 0, 1], 'symbol_roles': 1},
                                                             {'symbol': 'shared global', 'range': [1, 0, 1], 'symbol_roles': 1}]},
                {'relative_path': 'src/other.rs', 'occurrences': [{'symbol': 'shared global', 'range': [2, 0, 1], 'symbol_roles': 1}]},
            ]}))
            graph = mapper.Graph()
            mapper.import_scip(graph, index, prefix='mcp', profile='cpu')
            mapper.import_scip(graph, index, prefix='mcp', profile='gpu')
            self.assertIn('scip:mcp/src/lib.rs:cpu:local 0', graph.nodes)
            self.assertIn('scip:mcp/src/lib.rs:gpu:local 0', graph.nodes)
            self.assertEqual(graph.nodes['scip:shared global']['path'], '')
            self.assertEqual({e[4] for e in graph.edges if e[1]=='scip:shared global' and e[2]=='defines'}, {'mcp/src/lib.rs','mcp/src/other.rs'})

    def test_python_shell_and_workflow_connections_have_explicit_evidence(self):
        graph = mapper.Graph()
        for path, code, language in [('tool.py', b'def helper(): pass\ndef main(): helper()\n', 'python'),
                                     ('tool.sh', b'helper() { echo ok; }\nhelper\n', 'bash')]:
            graph.node('file:' + path, 'file', path, path)
            mapper.parse_source(graph, path, code, language)
        mapper.resolve_candidates(graph)
        self.assertEqual(len([n for n in graph.nodes.values() if n['name'] == 'helper' and n['kind'] in mapper.DEFINITIONS]), 2)
        self.assertTrue(any(e[2] == 'possible_target' and e[4] == 'tool.py' for e in graph.edges))
        self.assertTrue(any(e[2] == 'possible_target' and e[4] == 'tool.sh' for e in graph.edges))
        mapper.parse_configuration(graph, '.github/workflows/test.yml', b'on: push\njobs:\n  first:\n    steps:\n      - uses: actions/checkout@pinned\n  second:\n    needs: first\n')
        self.assertTrue(any(e[2] == 'needs_job' for e in graph.edges))
        self.assertTrue(any(e[2] == 'uses_action' for e in graph.edges))
        mapper.parse_configuration(graph, 'broken.json', b'{invalid')
        self.assertEqual(graph.coverage[-1]['status'], 'configuration_error')

    def test_dependency_cycles_include_self_loops_but_not_acyclic_paths(self):
        self.assertEqual(cycles({'a': {'b'}, 'b': {'a', 'c'}, 'd': {'d'}, 'e': {'c'}}), [['a', 'b'], ['d']])

    def test_database_parts_restore_exactly_and_reject_missing_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / 'source.sqlite'; db.write_bytes(bytes(range(256)) * 200)
            with mock.patch('pack_graph.CHUNK_BYTES', 100):
                count = pack(db, root / 'parts')
            self.assertGreater(count, 1)
            restore(root / 'parts', root / 'restored.sqlite')
            self.assertEqual(db.read_bytes(), (root / 'restored.sqlite').read_bytes())
            (root / 'parts/graph.sqlite.gz.part00').unlink()
            with self.assertRaises(ValueError):
                restore(root / 'parts', root / 'bad.sqlite')

    def test_checkpoint_reuse_rejects_changed_application_or_dirty_files(self):
        with mock.patch('reuse_scip.subprocess.check_output', side_effect=[b'scripts/repo-map/report_map.py\n', b'']):
            self.assertEqual(check_source('abc'), ['scripts/repo-map/report_map.py'])
        with mock.patch('reuse_scip.subprocess.check_output', side_effect=[b'crates/app/src/lib.rs\n', b'']):
            with self.assertRaises(ValueError):
                check_source('abc')
        with mock.patch('reuse_scip.subprocess.check_output', side_effect=[b'', b' M Cargo.toml']):
            with self.assertRaises(ValueError):
                check_source('abc')

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
            self.assertIn("scip:a.rs:repo-nightly-default-features:local 0", graph.nodes)
            self.assertIn("scip:b.rs:repo-nightly-default-features:local 0", graph.nodes)
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
