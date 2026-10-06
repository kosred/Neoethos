#!/usr/bin/env python3
"""Assemble current static profiles, or strictly validate reuse of a completed index."""
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


def validate_profile(profile, head, prefix, name):
    if profile.get('source_commit') != head or profile.get('prefix') != prefix or profile.get('profile') != name:
        raise ValueError('Additional index does not match current commit/workspace/profile: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit')
    parser.add_argument('--profiles-only', action='store_true',
                        help='Assemble fresh same-commit static profiles without a historical checkpoint.')
    parser.add_argument('--structural', type=Path)
    parser.add_argument('--additional-indices', type=Path)
    args = parser.parse_args()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    if args.profiles_only:
        if not args.additional_indices or args.checkpoint or args.source_commit:
            parser.error('--profiles-only requires --additional-indices and no checkpoint/source-commit.')
        changes = check_source(head)
        source_commit = head
    else:
        if not args.checkpoint or not args.source_commit:
            parser.error('Checkpoint reuse requires --checkpoint and --source-commit.')
        changes = check_source(args.source_commit)
        source_commit = args.source_commit
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
    command = [sys.executable, str(Path(__file__).with_name('map_repo.py')),
               '--output', str(args.output), '--scip-source-commit', source_commit]
    if args.checkpoint:
        command += ['--scip-json', str(args.checkpoint / 'scip-json.stdout'),
                    '--scip-prefix', '', '--scip-profile', 'repo-nightly-default-features']
    profiles = {}
    if args.additional_indices:
        import shutil
        from probe_indices import PROFILES
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        for name, (prefix, _) in PROFILES.items():
            folder = args.additional_indices / ('rust-profile-' + name)
            path = folder / 'checks.json'
            if not path.exists():
                profiles[name] = {'status': 'missing_profile'}
                continue
            profile = json.loads(path.read_text()); profiles[name] = profile
            validate_profile(profile, head, prefix, name)
            destination = args.output / 'profiles' / name; destination.mkdir(parents=True, exist_ok=True)
            for filename in ('checks.json', 'rust-scip.stderr', 'rust-scip.stdout', 'config.json'):
                if (folder / filename).exists():
                    shutil.copyfile(folder / filename, destination / filename)
            if all(profile.get(k, {}).get('status') == 'completed' for k in ('index', 'export')):
                command += ['--scip-json', str(folder / 'scip-json.stdout'), '--scip-prefix', prefix,
                            '--scip-profile', name]
    # Copy original indexer diagnostics before report generation so warnings are summarized.
    import shutil
    if args.checkpoint:
        for name in ('rust-scip.stderr', 'rust-scip.stdout'):
            source = args.checkpoint / name
            if source.exists():
                shutil.copyfile(source, args.output / name)
    record = run('graph', command, args.output, 900)
    profile_failed = any(any(p.get(k, {}).get('status') != 'completed' for k in ('index', 'export')) for p in profiles.values())
    checks = {'graph': record, 'additional_profiles': profiles}
    if args.profiles_only:
        checks['fresh_static_indices'] = {'status': 'failed' if profile_failed else 'completed', 'source_commit': head,
                                         'build_outputs_and_proc_macros_verified': False}
    else:
        checks['reused_index'] = {'status': 'completed', 'source_commit': source_commit,
                                 'verified_changes_limited_to_mapping_tools': changes}
    if record['status'] == 'completed':
        checks['database_parts'] = {'status': 'completed', 'parts': pack(args.output / 'graph.sqlite', args.output / 'parts')}
    (args.output / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    return int(record['status'] != 'completed' or profile_failed)


if __name__ == '__main__':
    raise SystemExit(main())
