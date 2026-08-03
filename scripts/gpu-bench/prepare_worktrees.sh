#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-cache/gpu-bench/worktrees}"
CANDIDATE_SHA="${2:?candidate SHA required}"
LEGACY_SHA="${3:-2be1408ee3986026fdbb2a5a74aaaf6ac67e5209}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"

ensure_worktree() {
  local name="$1"
  local sha="$2"
  local path="$ROOT/$name"
  if [[ -e "$path/.git" || -f "$path/.git" ]]; then
    actual="$(git -C "$path" rev-parse HEAD)"
    if [[ "$actual" != "$sha" ]]; then
      printf '%s worktree is at %s, expected %s\n' "$name" "$actual" "$sha" >&2
      exit 30
    fi
    return
  fi
  git -C "$REPO_ROOT" worktree add --detach "$path" "$sha"
}

ensure_worktree legacy "$LEGACY_SHA"
ensure_worktree candidate "$CANDIDATE_SHA"

git -C "$ROOT/legacy" diff --quiet
git -C "$ROOT/candidate" diff --quiet

cat > "$ROOT/worktrees.json" <<JSON
{
  "legacy": {"sha": "$LEGACY_SHA", "path": "$ROOT/legacy"},
  "candidate": {"sha": "$CANDIDATE_SHA", "path": "$ROOT/candidate"}
}
JSON

printf 'Prepared pinned worktrees at %s\n' "$ROOT"
printf 'Candidate benchmark binary build command:\n'
printf '  cargo build --release -p neoethos-cli --features gpu-nvidia --manifest-path %q/Cargo.toml\n' "$ROOT/candidate"
