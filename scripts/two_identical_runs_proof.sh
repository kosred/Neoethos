#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# two_identical_runs_proof.sh — SLICE 5: can a run be reproduced, and can it
# name its own arithmetic?
#
# Runs the SAME discovery twice with identical config + environment and
# compares every artifact. If the two runs differ, the diff of the two
# `*.profile.json` files names the knob that differed — that is the whole
# point of the DiscoveryRunProfile.execution section landed in this slice.
#
# WRITTEN, NOT RUN, by the slice-5 agent (2026-08-08). Intended for the GPU
# box (Linux, repo at ~/Neoethos, data at ~/.local/share/neoethos/data), but
# it works on any Linux host with the repo + data. DO NOT run it while a
# build/parity job is occupying the card.
#
# PRECONDITIONS the script VERIFIES (it refuses to "prove" anything on a
# setup that cannot be deterministic, and tells you exactly what to fix):
#   1. `models.search_runtime.seed` must be set in the CANONICAL user config
#      (the one `Settings::load()` resolves — on Linux typically
#      ~/.config/neoethos/config.yaml, falling back to ./config.yaml in the
#      working directory). The CLI installs runtime overrides ONCE at startup
#      from that config; `--config` on the subcommand does NOT reach the
#      OnceLock installers, so the seed must live in the canonical file.
#      Verified after run A via `.determinism_policy` in the profile.
#   2. No persistent seen-signature file (`.execution.seen_memory.file_path`
#      must be null) — cross-RUN memory makes run B search differently.
#   3. The discovery ledger must either be disabled or use a RELATIVE cache
#      dir (the default `cache/search`), because each run executes in its own
#      scratch working directory, which isolates relative ledger paths.
#
# KNOWN NON-DETERMINISM THIS SCRIPT SIDESTEPS (do not remove the guards):
#   - Wall-clock coupling: the GA convergence early-stop fires only after
#     `convergence_min_elapsed_fraction` of the per-combo time budget. Keep
#     `--generations` small and `models.prop_search_max_hours` at 0 so the
#     early-stop is generation-count-driven, not wall-clock-driven.
#   - Thread scheduling: pin `models.backtest_runtime.rayon_threads` in the
#     config if the comparison ever flickers; the profile records the
#     effective thread count either way (`.execution.effective_rayon_threads`).
#
# USAGE (all overridable via env):
#   SYMBOL=EURUSD BASE=M5 POPULATION=256 GENERATIONS=5 \
#   DATA_ROOT="$HOME/.local/share/neoethos/data" \
#   FEATURES=gpu-nvidia \
#   bash scripts/two_identical_runs_proof.sh
#
# EXIT CODES: 0 = REPRODUCIBLE, 1 = DIVERGENT, 2 = preconditions not met.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYMBOL="${SYMBOL:-EURUSD}"
BASE="${BASE:-M5}"
POPULATION="${POPULATION:-256}"
GENERATIONS="${GENERATIONS:-5}"
DATA_ROOT="${DATA_ROOT:-$HOME/.local/share/neoethos/data}"
# GPU box: FEATURES=gpu-nvidia. CPU-only proof: leave empty.
FEATURES="${FEATURES:-}"
PROOF_ROOT="${PROOF_ROOT:-$REPO_ROOT/proof-runs/$(date -u +%Y%m%dT%H%M%SZ)}"

command -v jq >/dev/null || { echo "FATAL: jq is required (apt install jq)"; exit 2; }
[ -d "$DATA_ROOT" ] || { echo "FATAL: DATA_ROOT $DATA_ROOT does not exist"; exit 2; }

echo "── building neoethos-cli (release${FEATURES:+, features=$FEATURES}) ──"
(
    cd "$REPO_ROOT"
    if [ -n "$FEATURES" ]; then
        cargo build --release -p neoethos-cli --features "$FEATURES"
    else
        cargo build --release -p neoethos-cli
    fi
)
BIN="$REPO_ROOT/target/release/neoethos-cli"
[ -x "$BIN" ] || { echo "FATAL: $BIN missing after build"; exit 2; }

run_once() {
    # Each run gets its OWN working directory so every relative cache path
    # (discovery ledger `cache/search`, fx-rate store fallbacks, …) is
    # per-run state, not shared state. The data root is passed absolute.
    local tag="$1"
    local dir="$PROOF_ROOT/$tag"
    mkdir -p "$dir"
    echo "── run $tag → $dir ──"
    (
        cd "$dir"
        # Identical environment for both runs by construction: the runs
        # inherit the same shell env. Anything ambient that differs will be
        # visible in the profile's `.execution` section afterwards.
        "$BIN" discover \
            --root "$DATA_ROOT" \
            --symbol "$SYMBOL" \
            --base "$BASE" \
            --population "$POPULATION" \
            --generations "$GENERATIONS" \
            --out "portfolio.json" \
            >"run.log" 2>&1
    ) || {
        echo "FATAL: run $tag failed — tail of $dir/run.log:"
        tail -n 40 "$dir/run.log"
        exit 2
    }
}

run_once a
run_once b

PROFILE_A="$PROOF_ROOT/a/portfolio.json.profile.json"
PROFILE_B="$PROOF_ROOT/b/portfolio.json.profile.json"
[ -f "$PROFILE_A" ] || { echo "FATAL: run a produced no profile at $PROFILE_A"; exit 2; }
[ -f "$PROFILE_B" ] || { echo "FATAL: run b produced no profile at $PROFILE_B"; exit 2; }

echo "── verifying the runs were reproducible-BY-CONSTRUCTION ──"
# Serialized shape (snake_case tag): {"mode":"deterministic","seed":42}
policy=$(jq -c '.determinism_policy' "$PROFILE_A")
case "$policy" in
    *'"mode":"deterministic"'*) echo "   determinism_policy = $policy" ;;
    *)
        echo "PRECONDITION FAILED: determinism_policy = $policy (need Deterministic)."
        echo "Set the seed in the CANONICAL user config the CLI loads at startup:"
        echo "    models:"
        echo "      search_runtime:"
        echo "        seed: 42"
        echo "(on this box: ~/.config/neoethos/config.yaml; a --config flag on the"
        echo " subcommand does NOT reach the startup installers — see"
        echo " crates/neoethos-cli/src/main.rs:28-33)."
        exit 2
        ;;
esac
seen_file=$(jq -c '.execution.seen_memory.file_path' "$PROFILE_A")
if [ "$seen_file" != "null" ]; then
    echo "PRECONDITION FAILED: a persistent seen-signature file is configured"
    echo "($seen_file). Cross-run memory makes run B search differently by design."
    echo "Unset models.seen_signature_runtime.file_path for the proof."
    exit 2
fi
ledger=$(jq -r '.discovery_ledger_enabled' "$PROFILE_A")
ledger_dir=$(jq -r '.discovery_ledger_cache_dir' "$PROFILE_A")
if [ "$ledger" = "true" ] && [ "${ledger_dir#/}" != "$ledger_dir" ]; then
    echo "PRECONDITION FAILED: discovery ledger enabled with ABSOLUTE cache dir"
    echo "($ledger_dir) — that is shared cross-run state. Disable the ledger or"
    echo "use a relative cache dir (per-run scratch cwd isolates it)."
    exit 2
fi

echo "── comparing artifacts (jq -S normalized, sha256) ──"
divergent=0
compare() {
    local rel="$1"
    local fa="$PROOF_ROOT/a/$rel" fb="$PROOF_ROOT/b/$rel"
    if [ ! -f "$fa" ] && [ ! -f "$fb" ]; then
        return 0 # produced by neither run — consistent
    fi
    if [ ! -f "$fa" ] || [ ! -f "$fb" ]; then
        echo "   DIVERGENT (presence): $rel exists in only one run"
        divergent=1
        return 0
    fi
    local ha hb
    ha=$(jq -S . "$fa" | sha256sum | cut -d' ' -f1)
    hb=$(jq -S . "$fb" | sha256sum | cut -d' ' -f1)
    if [ "$ha" = "$hb" ]; then
        echo "   identical: $rel ($ha)"
    else
        echo "   DIVERGENT: $rel"
        divergent=1
    fi
}

compare "portfolio.json.profile.json"
compare "portfolio.json"
compare "portfolio.json.funnel.json"
compare "portfolio.json.quality.json"
compare "portfolio.json.trades.json"
compare "portfolio.json.live_portfolio.json"

if [ "$divergent" -eq 0 ]; then
    echo "VERDICT: REPRODUCIBLE — two identical-config runs produced identical artifacts."
    exit 0
fi

echo "VERDICT: DIVERGENT — identical configs did not reproduce."
echo "The profile names the arithmetic. First, diff the execution sections:"
echo "    diff <(jq -S .execution '$PROFILE_A') <(jq -S .execution '$PROFILE_B')"
if ! diff <(jq -S '.execution' "$PROFILE_A") <(jq -S '.execution' "$PROFILE_B") >/dev/null; then
    echo "── execution-section diff (a vs b) ──"
    diff <(jq -S '.execution' "$PROFILE_A") <(jq -S '.execution' "$PROFILE_B") || true
    echo "→ the runs did NOT share an execution environment; the lines above are the cause."
else
    echo "── execution sections are IDENTICAL ──"
    echo "→ every recorded knob matched, yet results differ: an UNRECORDED source"
    echo "  of nondeterminism exists. That is a slice-5 census failure — find the"
    echo "  knob, add it to ExecutionEnvironmentProfile, and extend the census"
    echo "  table in crates/neoethos-search/src/discovery_tests.rs"
    echo "  (every_env_knob_is_classified_and_recorded_in_the_run_profile)."
fi
exit 1
