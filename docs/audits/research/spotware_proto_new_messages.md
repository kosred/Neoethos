# Spotware OpenApiMessages.proto — New Messages After Refresh

**Refresh date:** 2026-05-14
**Upstream source:** `https://raw.githubusercontent.com/spotware/openapi-proto-messages/master/OpenApiMessages.proto`
**Size delta:** 35,035 B (local 2026-05-08) -> 50,595 B (upstream 2026-05-14) = +15,560 B
**Top-level symbol delta:** 59 -> 89 (+30 new `message` declarations, 0 removed)

## Method

Both files were tokenized with `grep -nE '^(message|enum) '` to enumerate
top-level declarations. `comm -13 local upstream` produced the additions.
For each new message the leading doc comment and field block were
inspected to determine the message's semantic role.

## New message types (30 net-new)

Each entry below is **NEW upstream**; none of them exist in our local copy.
They are grouped by domain. Field counts are taken from the upstream master.

### A. Account session lifecycle (3 new)

| Message | Role | Payload type default |
|---|---|---|
| `ProtoOAAccountLogoutReq`     | Client request to log out of an account session.                 | `PROTO_OA_ACCOUNT_LOGOUT_REQ` |
| `ProtoOAAccountLogoutRes`     | Server ack of logout.                                             | `PROTO_OA_ACCOUNT_LOGOUT_RES` |
| `ProtoOAAccountDisconnectEvent` | Server-initiated event: existing account session was dropped server-side; client must re-auth. | `PROTO_OA_ACCOUNT_DISCONNECT_EVENT` |

**Why we care:** today the only signal for a stale session is failed
ProtoOAErrorRes on the next request. With `ProtoOAAccountDisconnectEvent`
the broker can proactively tell us to re-auth, which is exactly what our
reconnect/reconcile loop in `forex-app` would want to listen for.

### B. Margin call management (5 new)

| Message | Role |
|---|---|
| `ProtoOAMarginCallListReq`       | Request the existing margin-call thresholds configured on the account. |
| `ProtoOAMarginCallListRes`       | Response: list of configured thresholds (uses `ProtoOAMarginCall`). |
| `ProtoOAMarginCallUpdateReq`     | Client request to upsert a margin-call threshold. |
| `ProtoOAMarginCallUpdateRes`     | Server ack of the threshold update. |
| `ProtoOAMarginCallUpdateEvent`   | Async event: a margin-call threshold configuration was changed. |
| `ProtoOAMarginCallTriggerEvent`  | Async event: account margin level crossed `marginLevelThreshold` (rate-limited to once per 10 min). |

**Why we care:** the trading-side of `forex-app` currently has no
in-band margin-call wiring — we infer margin pressure only from the
periodic `ProtoOATraderUpdateEvent`/`ProtoOAGetAccountListByAccessTokenRes`
snapshots. These messages let us subscribe to threshold-trigger events
directly, which is materially safer for the risk-management module
(`crates/forex-app/src/risk/*`). **DO NOT integrate in this batch.**

### C. Depth-of-market / Level 2 subscription (3 new)

| Message | Role |
|---|---|
| `ProtoOASubscribeDepthQuotesReq`   | Subscribe to L2 quote stream for symbol IDs. |
| `ProtoOASubscribeDepthQuotesRes`   | Server ack. |
| `ProtoOAUnsubscribeDepthQuotesReq` | Unsubscribe. |
| `ProtoOAUnsubscribeDepthQuotesRes` | Server ack. |
| `ProtoOADepthEvent`                | Streaming L2 delta event: `repeated ProtoOADepthQuote newQuotes`, `repeated uint64 deletedQuotes`. |

**Why we care:** today we consume only the top-of-book via
`ProtoOASpotEvent`. The L2 depth stream would let us add a market-impact
estimator before submitting larger orders. **DO NOT integrate in this batch.**

### D. Deal / Order detail lookups (6 new)

| Message | Role |
|---|---|
| `ProtoOADealListByPositionIdReq` | Pull all deals for a specific `positionId` over a `[fromTimestamp, toTimestamp]` window. |
| `ProtoOADealListByPositionIdRes` | Response with the deal list. |
| `ProtoOADealOffsetListReq`       | Given a `dealId`, return the pair of deal sets it offset / was offset by. |
| `ProtoOADealOffsetListRes`       | Response with `offsettingDeals` + `offsetByDeals`. |
| `ProtoOAOrderDetailsReq`         | Fetch a single order plus all of its child deals by `orderId`. |
| `ProtoOAOrderDetailsRes`         | Response: order + deals + (optional) the originating position. |
| `ProtoOAOrderListByPositionIdReq` | Pull all orders for a specific position over a time window. |
| `ProtoOAOrderListByPositionIdRes` | Response with the order list. |

**Why we care:** today the reconcile pass in `forex-app` walks the
whole deal history (`ProtoOADealListReq`) and then filters client-side
by position. These give us narrow-window queries that are much cheaper
when reconciling after a single position close. **DO NOT integrate in
this batch.**

### E. Unrealized PnL on demand (2 new)

| Message | Role |
|---|---|
| `ProtoOAGetPositionUnrealizedPnLReq` | Ask the server for the current unrealized PnL on all open positions. |
| `ProtoOAGetPositionUnrealizedPnLRes` | Response: `repeated ProtoOAPositionUnrealizedPnL` + `moneyDigits` scale. |

**Why we care:** today unrealized PnL is computed locally from
`(currentPrice - entryPrice) * volume * pipValue * direction`. Having an
authoritative server-side value lets us cross-check our calculation on
every reconcile tick and alarm when they diverge (suggests stale market
data or wrong conversion FX). **DO NOT integrate in this batch.**

### F. Dynamic leverage lookup (2 new)

| Message | Role |
|---|---|
| `ProtoOAGetDynamicLeverageByIDReq` | Resolve a dynamic leverage entity by `leverageId` (the field added to `ProtoOASymbol` field 35). |
| `ProtoOAGetDynamicLeverageByIDRes` | Response: the tier ladder for the leverage entity. |

**Why we care:** `ProtoOASymbol.leverageId` exists in our local copy
(field 35 in `OpenApiModelMessages.proto`), but until now we had no way
to resolve the leverage tier ladder it points to. Required if we want
to compute initial margin correctly for accounts on dynamic leverage.

### G. Symbol categories (2 new)

| Message | Role |
|---|---|
| `ProtoOASymbolCategoryListReq` | Request the symbol-category taxonomy for the account. |
| `ProtoOASymbolCategoryListRes` | Response: `repeated ProtoOASymbolCategory`. |

**Why we care:** could improve our universe selector UI; not on the
critical path.

### H. CTID profile lookup (2 new)

| Message | Role |
|---|---|
| `ProtoOAGetCtidProfileByTokenReq` | Resolve the cTID profile (user identity) for an access token. GDPR-limited fields. |
| `ProtoOAGetCtidProfileByTokenRes` | Response. |

**Why we care:** purely identity / display; no immediate use in
forex-app's automated trading flow.

## Cross-cutting observations

- **All new messages use the `payloadType = 1 [default = PROTO_OA_*_*]`
  proto2 pattern** consistent with the existing schema. No new
  payload-discrimination mechanism.
- **No existing message was removed or renamed.** The diff is purely
  additive, so the refresh is **backward compatible** at the wire level.
- **No new enum types** are introduced in `OpenApiMessages.proto`. The
  new `PROTO_OA_*` payload-type constants live in
  `OpenApiModelMessages.proto::ProtoOAPayloadType`, which is included in
  the same refresh.
- All five domains B (margin call), C (depth), D (order/deal detail),
  E (unrealized PnL), and F (dynamic leverage) are realistic candidates
  for future feature work in `forex-app`. Of these, **B (margin call
  events) and E (server-side unrealized PnL)** have the highest
  risk-management value and should be the first to integrate when this
  feature batch comes around.

## Build impact

- `crates/forex-app/build.rs` invokes `protoc` via
  `protobuf-codegen` v4.31 and generates Rust types into
  `$OUT_DIR/protobuf_generated`. The new messages will appear as Rust
  structs in the generated module after the next build with no source
  changes required.
- No existing Rust call sites in `crates/forex-app/src/**` reference any
  of the 30 new symbols (verified by name match) — pure additions.

## Refresh action taken (2026-05-14)

| File | Action |
|---|---|
| `OpenApiCommonMessages.proto`      | Replaced with upstream master (cosmetic + 2 default annotations). |
| `OpenApiCommonModelMessages.proto` | Replaced with upstream master (trailing newline only). |
| `OpenApiMessages.proto`            | Replaced with upstream master (+30 new messages, cosmetic on rest). |
| `OpenApiModelMessages.proto`       | Replaced with upstream master (cosmetic indentation + many `[default = …]` annotations; no new symbols). |

No `forex-app` source code was modified. Integration of the 30 new
message types is deferred to a separate work item.
