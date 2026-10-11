#!/usr/bin/env python3
"""Deterministic summaries over inventory and static graph evidence; no AI calls."""
import collections
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def component(path):
    parts = Path(path).parts
    if not parts:
        return '<external>'
    if parts[0] in ('crates', 'vendor', 'tools') and len(parts) > 1:
        return '/'.join(parts[:2])
    return parts[0] if len(parts) > 1 else '<root>'


def cycles(adjacency):
    """Tarjan SCCs on the small declared-package graph, not runtime recursion."""
    indices, low, stack, active, found = {}, {}, [], set(), []
    def visit(node):
        indices[node] = low[node] = len(indices)
        stack.append(node); active.add(node)
        for target in sorted(adjacency.get(node, ())):
            if target not in indices:
                visit(target); low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            group = []
            while True:
                target = stack.pop(); active.remove(target); group.append(target)
                if target == node:
                    break
            if len(group) > 1 or node in adjacency.get(node, ()):
                found.append(sorted(group))
    for node in sorted(set(adjacency) | {x for v in adjacency.values() for x in v}):
        if node not in indices:
            visit(node)
    return sorted(found)


def report(output):
    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text())
    inventory = json.loads((output / 'inventory.json').read_text())
    coverage = json.loads((output / 'coverage.json').read_text())
    groups, extensions = {}, {}
    for f in inventory:
        for key, mapping in ((component(f['path']), groups), (Path(f['path']).suffix or '<none>', extensions)):
            row = mapping.setdefault(key, {'files': 0, 'physical_text_lines': 0, 'bytes': 0})
            row['files'] += 1; row['physical_text_lines'] += f['lines']; row['bytes'] += f['bytes']
    covered = {c['path'] for c in coverage if c['status'] == 'scip_indexed'}
    rust = [f['path'] for f in inventory if f['group'] == 'project' and f['path'].endswith('.rs')]
    adjacency = collections.defaultdict(set)
    cross = collections.Counter()
    with closing(sqlite3.connect((output / 'graph.sqlite').resolve().as_uri() + '?mode=ro', uri=True)) as db:
        ambiguities = db.execute("SELECT count(*) FROM (SELECT target FROM edges WHERE kind='defines' AND evidence='scip' GROUP BY target HAVING count(DISTINCT path)>1)").fetchone()[0]
        for source, target in db.execute("SELECT a.path,b.path FROM edges e JOIN nodes a ON a.id=e.source JOIN nodes b ON b.id=e.target WHERE e.kind='declared_dependency' AND b.kind='package'"):
            adjacency[source].add(target)
        # Static references grouped by the document owning a symbol's definition.
        for source, target, count in db.execute("SELECT e.path,n.path,count(*) FROM edges e JOIN nodes n ON n.id=e.target WHERE e.kind='references' AND e.evidence='scip' AND n.path!='' AND e.path!=n.path GROUP BY e.path,n.path"):
            a, b = component(source), component(target)
            if a != b:
                cross[(a, b)] += count
    result = {'commit': summary['commit'], 'semantic_source_commit': summary.get('semantic_source_commit'),
              'components': groups, 'extensions': extensions,
              'largest_project_files': sorted((f for f in inventory if f['group'] == 'project'), key=lambda f: (-f['lines'], f['path']))[:40],
              'declared_package_dependencies': {k: sorted(v) for k, v in sorted(adjacency.items())},
              'declared_package_cycles': cycles(adjacency),
              'cross_component_static_references': [{'source': a, 'target': b, 'occurrences': n} for (a, b), n in cross.most_common()],
              'rust_semantic_coverage': {'tracked_project_files': len(rust), 'indexed_files': len(set(rust) & covered),
                                       'not_indexed_files': sorted(set(rust) - covered),
                                       'meaning': 'A document being indexed does not prove every symbol/cfg/macro resolved.'},
              'parser_and_configuration_errors': [c for c in coverage if c['status'] in ('syntax_with_errors', 'configuration_error', 'manifest_error')],
              'inventory_only_project_files': [c for c in coverage if c['status'] == 'inventory_only'],
              'scip_diagnostics': [{'path': c['path'], 'diagnostics': c['diagnostics']} for c in coverage if c.get('diagnostics')],
              'ambiguous_scip_definition_symbols': ambiguities,
              'semantic_profiles': dict(collections.Counter(c.get('profile') for c in coverage if c['status']=='scip_indexed')),
              'limits': summary['limitations']}
    result['indexer_warnings'] = []
    for log in sorted(output.rglob('rust-scip.stderr')):
        content = log.read_text(errors='replace').splitlines()
        result['indexer_warnings'].append({'log': log.relative_to(output).as_posix(),
                                          'duplicate_symbol_warnings': sum('Duplicate symbol:' in line for line in content),
                                          'error_or_failure_lines': [line[:1000] for line in content if any(term in line.lower() for term in ('error:', 'failed to', 'panicked'))]})
    clone_path = output / 'duplicates/jscpd-report.json'
    if clone_path.exists():
        clones = json.loads(clone_path.read_text())
        pairs = collections.Counter()
        for clone in clones.get('duplicates', []):
            names = [clone.get(key, {}).get('name', '') for key in ('firstFile', 'secondFile')]
            if all(names):
                pairs[tuple(sorted(names))] += 1
        result['duplication'] = {'candidate_blocks': len(clones.get('duplicates', [])),
                                 'scanner_statistics': clones.get('statistics', {}).get('total', {}),
                                 'file_pairs': [{'files': list(pair), 'candidate_blocks': n} for pair, n in pairs.most_common()],
                                 'meaning': 'Similarity includes tests and intentional backend parallels; no automatic deletion.'}
    loc_path = output / 'loc.json'
    if loc_path.exists():
        loc = json.loads(loc_path.read_text())
        result['cloc_project_totals'] = loc.get('SUM', {})
        result['cloc_meaning'] = 'Tracked non-vendor files recognized by cloc, including tests, documentation and data; not production-only LOC.'
    (output / 'architecture.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Repository architecture evidence', '', 'Commit `' + summary['commit'] + '`.', '',
             '## Components', '', '| Component | Files | Physical text lines |', '|---|---:|---:|']
    for name, row in sorted(groups.items(), key=lambda pair: (-pair[1]['physical_text_lines'], pair[0])):
        lines.append(f"| {name} | {row['files']:,} | {row['physical_text_lines']:,} |")
    r = result['rust_semantic_coverage']
    lines += ['', '## Rust semantic coverage', '', f"{r['indexed_files']} of {r['tracked_project_files']} tracked Rust files have an indexed document.",
              'Indexing is configuration-specific. Diagnostics and missing documents are listed in architecture.json.', '',
              '## Largest project files', '', '| File | Physical text lines |', '|---|---:|']
    for f in result['largest_project_files'][:20]:
        lines.append(f"| {f['path']} | {f['lines']:,} |")
    lines += ['', '## Evidence gaps', '', f"{len(result['parser_and_configuration_errors'])} files have parser/configuration errors.",
              f"{len(result['inventory_only_project_files'])} project files have inventory only.",
              'Vendor components have exact inventory/hash/line counts; their source has not been semantically analyzed.',
              f"{ambiguities} SCIP symbols have definitions in multiple documents; these are kept ambiguous rather than selecting one source.",
              'Static reference observations may repeat across profiles. Static profiles skip build outputs/proc macros and do not validate a CPU/GPU build.',
              'Declared package cycles combine all target/dev/build conditions; they are not proof of a failing build.', '',
              '## Limits', ''] + ['- ' + limit for limit in result['limits']]
    if 'cloc_project_totals' in result:
        lines += ['', '## Line counts', '', json.dumps(result['cloc_project_totals']), result['cloc_meaning']]
    if 'duplication' in result:
        lines += ['', '## Duplication candidates', '', str(result['duplication']['candidate_blocks']) + ' candidate blocks.',
                  result['duplication']['meaning']]
    (output / 'architecture.md').write_text('\n'.join(lines) + '\n')
    return result
