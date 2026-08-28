#!/usr/bin/env bash
set -euo pipefail

cd /workspace/forex-ai
accept=audit/chunk-4e-app-acceptance
root="$accept/runtime-root-clean"
base=http://127.0.0.1:7423

cancel_terminal=0
for _ in $(seq 1 240); do
  curl --fail --silent --show-error "$base/engines/status" > "$accept/status-after-cancel.json"
  state=$(jq -r '.canonicalNativeResearch.state' "$accept/status-after-cancel.json")
  if [[ "$state" != "Queued" && "$state" != "Running" ]]; then
    cancel_terminal=1
    break
  fi
  sleep 0.25
done
test "$cancel_terminal" = 1
test "$(jq -r '.canonicalNativeResearch.state' "$accept/status-after-cancel.json")" = Cancelled
test "$(jq -r '.discovery' "$accept/status-after-cancel.json")" = Idle
test "$(jq -r '.training' "$accept/status-after-cancel.json")" = Idle
test ! -e "$root/research/native-discovery/v1"

set +e
./target/debug/neoethos-cli native-research start \
  --contract-relative-path research/contracts/canonical-native-device-happy.json \
  --expected-sha256 da3d7ba03cd99621e5f36f8dd41943865e58b849c8f8982af9bae63a8f8d1828 \
  --population 10 --population-auto false --max-indicators 5 \
  --api-base "$base" > "$accept/cli-start-normal.log" 2>&1
cli_start_code=$?
set -e
test "$cli_start_code" = 0

: > "$accept/normal-status-poll.jsonl"
normal_terminal=0
for _ in $(seq 1 1200); do
  curl --fail --silent --show-error "$base/engines/status" > "$accept/status-normal-latest.json"
  jq -c '{discovery,training,canonicalNativeResearch}' "$accept/status-normal-latest.json" \
    >> "$accept/normal-status-poll.jsonl"
  state=$(jq -r '.canonicalNativeResearch.state' "$accept/status-normal-latest.json")
  if [[ "$state" != "Queued" && "$state" != "Running" ]]; then
    cp "$accept/status-normal-latest.json" "$accept/status-normal-terminal.json"
    normal_terminal=1
    break
  fi
  sleep 0.25
done
test "$normal_terminal" = 1
test "$(jq -r '.canonicalNativeResearch.state' "$accept/status-normal-terminal.json")" = Published
test "$(jq -r '.discovery' "$accept/status-normal-terminal.json")" = Idle
test "$(jq -r '.training' "$accept/status-normal-terminal.json")" = Idle
if jq -e 'select(.training != "Idle")' "$accept/normal-status-poll.jsonl" >/dev/null; then
  echo 'legacy Training became non-idle during Native Research' >&2
  exit 1
fi

relative_path=$(jq -r '.canonicalNativeResearch.published.relativePath' "$accept/status-normal-terminal.json")
byte_count=$(jq -r '.canonicalNativeResearch.published.byteCount' "$accept/status-normal-terminal.json")
reported_sha=$(jq -r '.canonicalNativeResearch.published.fileSha256' "$accept/status-normal-terminal.json")
evidence_sha=$(jq -r '.canonicalNativeResearch.published.evidenceIdentitySha256' "$accept/status-normal-terminal.json")
test "$relative_path" = "research/native-discovery/v1/cngr1-$evidence_sha.json"
artifact="$root/$relative_path"
test -f "$artifact"
test "$(stat -c %s "$artifact")" = "$byte_count"
test "$(sha256sum "$artifact" | cut -d ' ' -f 1)" = "$reported_sha"

set +e
./target/debug/neoethos-cli native-research status --api-base "$base" \
  > "$accept/cli-status-published.log" 2>&1
cli_final_status_code=$?
set -e
test "$cli_final_status_code" = 0

printf 'CANCEL_TERMINAL=Cancelled\nCLI_START=%s\nNORMAL_TERMINAL=Published\nCLI_FINAL_STATUS=%s\nARTIFACT=%s\nARTIFACT_BYTES=%s\nARTIFACT_SHA256=%s\nEVIDENCE_SHA256=%s\n' \
  "$cli_start_code" "$cli_final_status_code" "$relative_path" "$byte_count" \
  "$reported_sha" "$evidence_sha" > "$accept/normal-sequence.status"

printf '%s\n' '--- cancelled terminal ---'
jq '{discovery,training,canonicalNativeResearch}' "$accept/status-after-cancel.json"
printf '%s\n' '--- CLI normal start ---'
cat "$accept/cli-start-normal.log"
printf '%s\n' '--- published terminal ---'
jq '{discovery,training,canonicalNativeResearch}' "$accept/status-normal-terminal.json"
printf '%s\n' '--- CLI published status ---'
cat "$accept/cli-status-published.log"
printf '%s\n' '--- persisted artifact ---'
sha256sum "$artifact"
stat -c '%n %s bytes' "$artifact"
