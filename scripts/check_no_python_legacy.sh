#!/usr/bin/env bash
# Pure-Rust workspace guard. Refuses any Python artifact / PyO3 wiring
# outside the explicit allow list (docs/ + vendor/, both of which can
# carry third-party Python tooling that's not part of the project's
# runtime — e.g. lightgbm-upstream Python wrappers under
# `vendor/lightgbm3-sys/lightgbm/`).
#
# V0.4 audit Task #62 — extended coverage:
#   - `.pyc`, `.pyd`, `__pycache__/` (compiled / bytecode artifacts that
#      leak in via dev environments even after the `.py` source is gone)
#   - `*.service` / shell scripts referencing `PYTHONPATH=`, `python -m`,
#     `venv/`, or `pip install` (catches stale Python systemd units and
#     installer fossils that survived the v0.5 pure-Rust migration)
#
# Wire this script into CI (`.github/workflows/ci.yml`) so a poisoned
# branch fails the build instead of leaking onto main. Local devs can
# run it as a pre-commit hook:
#   ln -s ../../scripts/check_no_python_legacy.sh .git/hooks/pre-commit
set -euo pipefail

status=0

is_allowed_path() {
  case "$1" in
    docs/*|vendor/*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

tracked_files=$(git ls-files -- . ':(exclude)docs/**' ':(exclude)vendor/**')
while IFS= read -r file; do
  if [[ -z "$file" ]]; then
    continue
  fi

  if is_allowed_path "$file"; then
    continue
  fi

  lower_file=${file,,}
  case "$lower_file" in
    *.py|*.pyc|*.pyd|*/__pycache__/*|__pycache__/*|*/pyproject.toml|pyproject.toml|*pyo3*|python/*|*/python/*|python-bindings/*|*/python-bindings/*|pybindings/*|*/pybindings/*)
      printf 'Legacy Python/PyO3 artifact is not allowed outside docs/vendor: %s\n' "$file" >&2
      status=1
      ;;
  esac
done <<< "$tracked_files"

active_manifests=$(git ls-files -- 'Cargo.toml' 'crates/**/Cargo.toml' ':(exclude)docs/**' ':(exclude)vendor/**')
while IFS= read -r manifest; do
  if [[ -z "$manifest" ]]; then
    continue
  fi

  if is_allowed_path "$manifest"; then
    continue
  fi

  if grep -nE '(^|[[:space:]])pyo3([[:space:]]|=|$)' "$manifest" >&2; then
    printf 'PyO3 dependency is not allowed in active Cargo manifest: %s\n' "$manifest" >&2
    status=1
  fi
done <<< "$active_manifests"

# Scan service files, shell scripts, and Dockerfiles for stale Python
# runtime references that survived the migration. Allow PYTHONPATH /
# python invocations only inside docs/ + vendor/.
runtime_files=$(git ls-files -- '*.service' '*.sh' 'Dockerfile*' \
  ':(exclude)scripts/check_no_python_legacy.sh' \
  ':(exclude)docs/**' ':(exclude)vendor/**')
while IFS= read -r rfile; do
  if [[ -z "$rfile" ]]; then
    continue
  fi
  if is_allowed_path "$rfile"; then
    continue
  fi
  if grep -nE 'PYTHONPATH=|python[0-9.]*[[:space:]]+-m[[:space:]]+forex_bot|^[[:space:]]*pip[[:space:]]+install|/venv/|/\.venv/' \
       "$rfile" >&2; then
    printf 'Stale Python runtime reference in %s — V0.4 migration removed Python from the runtime.\n' "$rfile" >&2
    status=1
  fi
done <<< "$runtime_files"

exit "$status"
