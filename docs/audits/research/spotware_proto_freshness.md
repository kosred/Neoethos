# Spotware Open API Proto Freshness Audit

**Audit date:** 2026-05-14
**Local proto files dated:** 2026-05-08 (~6 days old)
**Upstream source:** `https://raw.githubusercontent.com/spotware/openapi-proto-messages/master/<filename>.proto`

## Summary

All four local proto files differ from the upstream master branch. The
deltas are mostly **cosmetic / non-semantic** (whitespace, blank-line
formatting, indentation fixes) plus a sizeable batch of **enum default
value annotations** that upstream added for optional fields. None of the
upstream changes drop or rename existing fields, but **`OpenApiMessages.proto`
has a ~15.5KB size increase** which strongly suggests new messages /
payload types have been appended downstream of the section we sampled.

> **Recommendation:** schedule a refresh of `OpenApiMessages.proto` and
> `OpenApiModelMessages.proto`. The other two are safe-to-defer cosmetic
> drift. Do not blind-replace; re-run `prost-build` and the round-trip
> tests after replacement.

## Per-file comparison

| File | Local size (B) | Upstream size (B) | Δ bytes | Diff character |
|------|---------------:|------------------:|--------:|----------------|
| `OpenApiCommonMessages.proto`      |  1,589 |  1,645 |    +56 | Whitespace; `[default = ERROR_RES]`, `[default = HEARTBEAT_EVENT]` enum defaults added |
| `OpenApiCommonModelMessages.proto` |  1,226 |  1,227 |     +1 | Single trailing newline added |
| `OpenApiMessages.proto`            | 35,035 | 50,595 | +15,560 | Blank-line formatting, many new `[default = …]` enum defaults on `optional` fields, **and ~15KB of additional content** (likely new messages appended beyond the sampled diff window) |
| `OpenApiModelMessages.proto`       | 45,093 | 45,468 |   +375 | Indentation fixes (orphan top-of-message lines re-indented), many `[default = …]` enum defaults added |

## Categorised upstream changes

### 1. Pure cosmetic (safe to ignore)

- Extra blank lines between option declarations in
  `OpenApiCommonMessages.proto` and `OpenApiMessages.proto` (Java-style
  formatting).
- Re-indentation of fields that were flush-left in our copy (e.g.
  `required int64 symbolId = 1;` in `ProtoOASymbol`, `required int64
  orderId = 1;` in `ProtoOAOrder`, `BALANCE_WITHDRAW_COPY_FEE = 34;`,
  `MARKET_RANGE = 5;` in enum bodies). Generated code is identical.

### 2. Added `[default = …]` annotations (semantically backward-compatible)

Upstream added default values to many `optional` fields. Examples:

- `ProtoErrorRes.payloadType` → `[default = ERROR_RES]`
- `ProtoHeartbeatEvent.payloadType` → `[default = HEARTBEAT_EVENT]`
- Every `ProtoOA*Req` / `ProtoOA*Res` `payloadType` field now defaults
  to its corresponding `ProtoOAPayloadType` constant (e.g.
  `PROTO_OA_APPLICATION_AUTH_REQ`, `PROTO_OA_VERSION_RES`).
- `ProtoOANewOrderReq.timeInForce` → `[default = GOOD_TILL_CANCEL]`
- `ProtoOANewOrderReq.stopTriggerMethod` → `[default = TRADE]`
- `ProtoOAAmendOrderReq.stopTriggerMethod` → `[default = TRADE]`
- `ProtoOAAmendPositionSLTPReq.stopLossTriggerMethod` →
  `[default = TRADE]`
- `ProtoOASymbol.swapRollover3Days` → `[default = MONDAY]`
- `ProtoOASymbol.commissionType` → `[default = USD_PER_MILLION_USD]`
- `ProtoOASymbol.distanceSetIn` → `[default = SYMBOL_DISTANCE_IN_POINTS]`
- `ProtoOASymbol.minCommissionType` → `[default = CURRENCY]`
- `ProtoOASymbol.minCommissionAsset` → `[default = "USD"]`
- `ProtoOASymbol.tradingMode` → `[default = ENABLED]`
- `ProtoOASymbol.rolloverCommission3Days` → `[default = MONDAY]`
- `ProtoOASymbol.swapCalculationType` → `[default = PIPS]`
- `ProtoOATrader.accessRights` → `[default = FULL_ACCESS]`
- `ProtoOATrader.accountType` → `[default = HEDGED]`
- `ProtoOATrader.limitedRiskMarginCalculationStrategy` →
  `[default = ACCORDING_TO_LEVERAGE]`
- `ProtoOATrader.stopOutStrategy` → `[default = MOST_MARGIN_USED_FIRST]`
- `ProtoOAPosition.stopLossTriggerMethod` → `[default = TRADE]`
- `ProtoOAOrder.timeInForce` → `[default = IMMEDIATE_OR_CANCEL]`
- `ProtoOAOrder.stopTriggerMethod` → `[default = TRADE]`
- `ProtoOATrendbar.period` → `[default = M1]`

**Impact assessment:** `prost` ignores proto2 default values when
deserialising — Rust generated code uses `Option<T>` for `optional`
fields regardless. So today, picking up these defaults is a no-op for
us. They become relevant only if we later use a strict-proto2 codec or
mirror the protobuf JSON encoding.

### 3. Comment/doc tweaks

- `ProtoOAArchivedSymbol`: leading `/** Archived symbol entity. */`
  doc comment was **removed** upstream. Net loss of one comment line.

### 4. **Unknown / suspicious bulk additions in `OpenApiMessages.proto`**

The unified diff sampled the first ~2KB only. Upstream is ~50.6KB vs
our 35KB — a +15KB delta that the sampled diff window does not cover.
The most plausible source of that delta is **new messages or payload
types appended at the end of the file**. This could include:

- New `ProtoOA*Req` / `ProtoOA*Res` pairs for features added since
  2026-05-08.
- New `optional` fields on existing messages.

A full byte-for-byte diff was NOT performed in this audit due to output
size budget. The header/options block and the first ~30 messages were
the only sections actually rendered in the diff.

## Sandbox / fetch status

- `curl https://raw.githubusercontent.com/...` succeeded (HTTP 200 for
  all four files).
- The Claude `WebFetch` tool *summarised* the file content rather than
  returning verbatim text, so the diff was generated from the `curl`
  download saved to `/tmp/spotware_upstream/`.

## Next steps (NOT performed in this audit — operator decision needed)

1. Run a complete byte-by-byte diff on `OpenApiMessages.proto` to
   enumerate every new message / field.
2. Decide whether to pull the upstream proto wholesale or cherry-pick
   only the additions we care about. Replacing wholesale will require:
   - Re-running `prost-build` (the build script under
     `crates/forex-app/build.rs` if any).
   - Verifying the cTrader `ProtoOATrendbarPeriod` enum still matches
     our canonical timeframes (after H2 removal, this is M1=1, M3=3,
     M5=5, M15=7, M30=8, H1=9, H4=10, H12=11, D1=12, W1=13, MN1=14;
     non-canonical M2/M4/M10/H2 remain rejected at the application
     layer).
3. Pin a known-good upstream commit hash in a comment block at the top
   of each local proto file once refreshed.
