#!/usr/bin/env python3
"""Reuse a completed index only when all non-mapping tracked contents are unchanged."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from run_checks import run
from pack_graph import pack


def check_source(source_commit):
    changed = subprocess.check_output(['git', 'diff', '--name-only', source_commit, 'HEAD', '--']).decode().splitlines()
    rejected = [p for p in changed if not (p.startswith('scripts/repo-map/') or
                p in ('.github/workflows/repo-map.yml', '.github/workflows/repo-map-continuation.yml'))]
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip()
    if rejected or dirty:
        raise ValueError('Index provenance check failed: non-mapping tracked content changed: ' + repr(rejected))
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--structural', type=Path)
    args = parser.parse_args()
    changes = check_source(args.source_commit)
    previous = json.loads((args.checkpoint / 'checks.json').read_text())
    if any(previous.get(k, {}).get('status') != 'completed' for k in ('rust_scip', 'scip_json', 'graph')):
        raise ValueError('Checkpoint did not complete index generation/export/graph checks.')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.structural:
        import shutil
        structural = json.loads((args.structural / 'summary.json').read_text())
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        if structural['commit'] != head or structural['working_tree_dirty']:
            raise ValueError('Structural reports do not match this clean commit.')
        shutil.copyfile(args.structural / 'loc.json', args.output / 'loc.json')
        shutil.copytree(args.structural / 'duplicates', args.output / 'duplicates', dirs_exist_ok=True)
    record = run('graph', [sys.executable, str(Path(__file__).with_name('map_repo.py')),
                 '--output', str(args.output), '--scip-json', str(args.checkpoint / 'scip-json.stdout'),
                 '--scip-source-commit', args.source_commit], args.output, 600)
    checks = {'reused_index': {'status': 'completed', 'source_commit': args.source_commit,
                              'verified_changes_limited_to_mapping_tools': changes}, 'graph': record}
    if record['status'] == 'completed':
        checks['database_parts'] = {'status': 'completed', 'parts': pack(args.output / 'graph.sqlite', args.output / 'parts')}
    (args.output / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    # Preserve actual indexer stderr for build/proc-macro warning investigation.
    for name in ('rust-scip.stderr', 'rust-scip.stdout'):
        source = args.checkpoint / name
        if source.exists():
            import shutil
            shutil.copyfile(source, args.output / name)
    return int(record['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
