#!/usr/bin/env bash
set -euo pipefail

cd /workspace/forex-ai
accept=audit/chunk-4e-app-acceptance
body='{"contractArtifact":{"relativePath":"research/contracts/canonical-native-device-happy.json","expectedSha256":"da3d7ba03cd99621e5f36f8dd41943865e58b849c8f8982af9bae63a8f8d1828"},"population":1000,"populationAuto":false,"maxIndicators":5}'

start_code=$(curl --silent --show-error -o "$accept/start-cancel.json" -w '%{http_code}' \
  -H 'content-type: application/json' --data "$body" \
  http://127.0.0.1:7423/engines/native-research/start)
busy_code=$(curl --silent --show-error -o "$accept/start-busy.json" -w '%{http_code}' \
  -H 'content-type: application/json' --data "$body" \
  http://127.0.0.1:7423/engines/native-research/start)
token=$(jq -r '.leaseToken // empty' "$accept/start-cancel.json")
test -n "$token"
wrong_token=$((token + 1))
wrong_code=$(curl --silent --show-error -o "$accept/cancel-wrong-token.json" -w '%{http_code}' \
  -H 'content-type: application/json' --data "{\"leaseToken\":\"$wrong_token\"}" \
  http://127.0.0.1:7423/engines/native-research/cancel)

set +e
./target/debug/neoethos-cli native-research status --api-base http://127.0.0.1:7423 \
  > "$accept/cli-status-live.log" 2>&1
cli_status_code=$?
./target/debug/neoethos-cli native-research cancel --api-base http://127.0.0.1:7423 \
  > "$accept/cli-cancel.log" 2>&1
cli_cancel_code=$?
set -e

printf 'START_HTTP=%s\nBUSY_HTTP=%s\nWRONG_TOKEN_HTTP=%s\nLEASE_TOKEN=%s\nWRONG_TOKEN=%s\nCLI_STATUS=%s\nCLI_CANCEL=%s\n' \
  "$start_code" "$busy_code" "$wrong_code" "$token" "$wrong_token" \
  "$cli_status_code" "$cli_cancel_code" > "$accept/cancel-sequence.status"

printf '%s\n' '--- start ---'
cat "$accept/start-cancel.json"
printf '\n%s\n' '--- busy ---'
cat "$accept/start-busy.json"
printf '\n%s\n' '--- wrong token ---'
cat "$accept/cancel-wrong-token.json"
printf '\n%s\n' '--- cli status ---'
cat "$accept/cli-status-live.log"
printf '%s\n' '--- cli cancel ---'
cat "$accept/cli-cancel.log"

test "$start_code" = 202
test "$busy_code" = 409
test "$wrong_code" = 409
test "$cli_status_code" = 0
test "$cli_cancel_code" = 0
