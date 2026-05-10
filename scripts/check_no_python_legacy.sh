#!/usr/bin/env bash
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
    *.py|*/pyproject.toml|pyproject.toml|*pyo3*|python/*|*/python/*|python-bindings/*|*/python-bindings/*|pybindings/*|*/pybindings/*)
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

exit "$status"
