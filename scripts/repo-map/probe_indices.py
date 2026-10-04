#!/usr/bin/env python3
"""Generate bounded, explicitly static Rust indexes for isolated workspaces and GPU cfgs."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from run_checks import run

PROFILES = {
    'mcp': ('mcp', []),
    'mesh': ('mesh', []),
    'history-probe': ('tools/ctrader-history-probe', []),
    'boundary-harness': ('tools/ctrader-network-boundary-harness', []),
    'cuda-static': ('', ['neoethos-cli/gpu-nvidia-full', 'neoethos-cli/gpu-b-native', 'neoethos-data/gpu-cuda-device-fixtures', 'neoethos-gpu-cuda/cuda-device-fixtures']),
    'hip-static': ('', ['neoethos-data/gpu-hip-session', 'neoethos-data/gpu-hip-smc', 'neoethos-gpu-cuda/hip-device-fixtures', 'neoethos-models/gpu-rocm']),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    prefix, features = PROFILES[args.profile]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    # CLI SCIP always requests build outputs. Explicit override avoids native compilation
    # and leaves proc macros/generated OUT_DIR items unresolved, recorded as a limit.
    config = {'cargo': {'features': features, 'buildScripts': {'overrideCommand': ['true']}},
              'procMacro': {'enable': False}}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    os.chdir(root / prefix)
    checks = {'profile': args.profile, 'prefix': prefix, 'source_commit': commit,
              'features': features, 'build_scripts': 'skipped_by_explicit_static_override',
              'runtime_verified': False, 'proc_macros_and_generated_outputs_verified': False}
    checks['index'] = run('rust-scip', ['rustup', 'run', 'nightly-2026-04-07', 'rust-analyzer', 'scip', '.',
                           '--config-path', str(output / 'config.json'), '--output', str(output / 'index.scip'),
                           '--exclude-vendored-libraries'], output, 900)
    if checks['index']['status'] == 'completed' and (output / 'index.scip').is_file():
        checks['export'] = run('scip-json', ['scip', 'print', '--json', str(output / 'index.scip')], output, 120)
    elif checks['index']['status'] == 'completed':
        checks['index']['status'] = 'missing_index'
    (output / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps({k: v for k, v in checks.items() if k not in ('index', 'export')}, indent=2))
    return int(any(checks.get(k, {}).get('status') != 'completed' for k in ('index', 'export')))


if __name__ == '__main__':
    raise SystemExit(main())
