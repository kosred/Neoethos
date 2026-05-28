#!/usr/bin/env bash
# Sync the cTrader Open API proto definitions from the official
# Spotware upstream repository.
#
#   https://github.com/spotware/openapi-proto-messages
#
# Why this script exists:
# Phase A (commit 88bf8742, 2026-05-27) modelled the entire
# `SymbolFinancials` struct from proto comments — but never verified
# against real broker bytes. The real-data capture in commit e60972ad
# (2026-05-28) discovered that proto enums are sent as INTEGER
# discriminants on the JSON proxy wire, not SCREAMING_SNAKE_CASE
# strings as the file comments imply. ALL parses had been failing
# silently in production.
#
# The invariant: our `crates/neoethos-app/proto/*.proto` MUST match
# the upstream Spotware spec byte-for-byte (modulo line endings). This
# script enforces that invariant — run it before any Phase A/B/C/D
# work and after every Spotware release. CI should run it as a
# verification step (TODO: wire it in).
#
# Usage:
#   scripts/sync-ctrader-proto.sh         # diff-only, no writes
#   scripts/sync-ctrader-proto.sh --write # overwrite local files
#
# Requires:
#   - gh (GitHub CLI) authenticated for read-only access
#   - md5sum, base64
set -euo pipefail

UPSTREAM_REPO="spotware/openapi-proto-messages"
LOCAL_DIR="crates/neoethos-app/proto"
PROTO_FILES=(
    OpenApiCommonMessages
    OpenApiCommonModelMessages
    OpenApiMessages
    OpenApiModelMessages
)
WRITE=0
if [ "${1:-}" = "--write" ]; then
    WRITE=1
fi

# Resolve repo root so the script works from any cwd.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "ERROR: $LOCAL_DIR not found — run from repo root." >&2
    exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

drift=0
for stem in "${PROTO_FILES[@]}"; do
    f="$stem.proto"
    upstream_path="$TMPDIR/$f"

    # Fetch upstream content via the GitHub API (raw bytes via
    # base64-decode is faster than the raw.githubusercontent CDN for
    # text files and avoids the redirect rabbit-hole).
    gh api "repos/$UPSTREAM_REPO/contents/$f" --jq '.content' \
        | base64 -d > "$upstream_path"

    local_hash=$(tr -d '\r' < "$LOCAL_DIR/$f" | md5sum | awk '{print $1}')
    upstream_hash=$(md5sum "$upstream_path" | awk '{print $1}')

    if [ "$local_hash" = "$upstream_hash" ]; then
        echo "  ✓ $f : in sync"
    else
        drift=1
        echo "  ✗ $f : DRIFT (local=$local_hash upstream=$upstream_hash)"
        if [ $WRITE -eq 1 ]; then
            cp "$upstream_path" "$LOCAL_DIR/$f"
            echo "      → overwrote local with upstream"
        else
            echo "      diff (first 20 lines):"
            diff -u <(tr -d '\r' < "$LOCAL_DIR/$f") "$upstream_path" | head -20 | sed 's/^/        /'
        fi
    fi
done

if [ $drift -eq 0 ]; then
    echo
    echo "All 4 proto files content-identical to upstream Spotware ($UPSTREAM_REPO)."
    exit 0
fi

if [ $WRITE -eq 1 ]; then
    echo
    echo "Sync complete. Review the changes:"
    echo "  git diff $LOCAL_DIR/"
    exit 0
fi

echo
echo "Drift detected. Run with --write to overwrite, or accept manually."
exit 1
