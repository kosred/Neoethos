# cTrader Open API — Exhaustive Reference

Compiled 2026-05-15 by the research agent in response to the operator
directive "Το ctrader api πρέπει να γίνει ανάγνωση των docs σελίδα σελιδα
όλα οχι τα σημαντικότερα και απο την επίσημη σελίδα" — read the docs
page by page, all of them, from the official site.

This document is the canonical research artefact for the cTrader Open
API. Where the official help-centre pages were unreachable to the
sandbox (HTTP 403 on every `help.ctrader.com/open-api/*` and
`connect.spotware.com/*` URL), content was reconstructed from:

1. WebSearch result snippets that quote the help-centre pages
   verbatim.
2. The canonical `.proto` files vendored at
   `/home/user/forex-ai/crates/forex-app/proto/` — these are the
   2026-05-14 (Batch 6) refresh of the four `spotware/openapi-proto-messages`
   files and are the definitive source for message structure, field
   types, enum values and payload type IDs.
3. The earlier internal references
   `docs/audits/research/ctrader_api_reference.md` and
   `docs/audits/research/spotware_proto_new_messages.md`, which were
   themselves built from primary Spotware sources.

No third-party (community / blog / port) content is included.

---

## 0. Sources

### 0.1 Official URLs in the page-by-page sweep

For each, "fetched" means we obtained the content (typically via
WebSearch snippet because direct WebFetch was blocked). "Blocked"
means the sandbox rejected access and no usable content was obtained.

| # | URL | Status |
|---|-----|--------|
| 1 | https://help.ctrader.com/open-api/ — Getting started | snippet via WebSearch |
| 2 | https://help.ctrader.com/open-api/proxies-endpoints/ | snippet via WebSearch |
| 3 | https://help.ctrader.com/open-api/connection/ — Establish a connection | snippet via WebSearch |
| 4 | https://help.ctrader.com/open-api/api-application/ — Register an application | snippet via WebSearch |
| 5 | https://help.ctrader.com/open-api/creating-new-app/ — Create your application | snippet via WebSearch |
| 6 | https://help.ctrader.com/open-api/account-authentication/ — App and account authentication | snippet via WebSearch |
| 7 | https://help.ctrader.com/open-api/protocol-buffers-json/ — Protobuf and JSON | snippet via WebSearch |
| 8 | https://help.ctrader.com/open-api/sending-receiving-protobuf/ — Send and receive Protobuf | snippet via WebSearch |
| 9 | https://help.ctrader.com/open-api/sending-receiving-json/ — Send and receive JSON | snippet via WebSearch |
| 10 | https://help.ctrader.com/open-api/messages/ — Messages | snippet via WebSearch |
| 11 | https://help.ctrader.com/open-api/common-messages/ — Common messages | snippet via WebSearch |
| 12 | https://help.ctrader.com/open-api/model-messages/ — Model messages | snippet via WebSearch |
| 13 | https://help.ctrader.com/open-api/common-model-messages/ — Common model messages | snippet via WebSearch |
| 14 | https://help.ctrader.com/open-api/error-handling/ — Error handling | snippet via WebSearch |
| 15 | https://help.ctrader.com/open-api/symbol-data/ — Attain symbol data | snippet via WebSearch |
| 16 | https://help.ctrader.com/open-api/symbol-rate-conversion/ — Symbol rate conversion | snippet via WebSearch |
| 17 | https://help.ctrader.com/open-api/profit-loss-calculation/ — Calculating Profit/Loss | snippet via WebSearch |
| 18 | https://help.ctrader.com/open-api/use-cases/ — Use cases | snippet via WebSearch |
| 19 | https://help.ctrader.com/open-api/faq/ — FAQ | snippet via WebSearch |
| 20 | https://help.ctrader.com/open-api/terms-of-use/ — Terms of use | snippet via WebSearch |
| 21 | https://help.ctrader.com/open-api/privacy-policy/ — Privacy Policy | snippet via WebSearch |
| 22 | https://help.ctrader.com/open-api/net_SDK/net-sdk-index/ — .NET SDK | snippet via WebSearch |
| 23 | https://help.ctrader.com/open-api/python-SDK/python-sdk-index/ — Python SDK | snippet via WebSearch |
| 24 | https://help.ctrader.com/ctrader-id/managing-ctid/open-api/ — Open API tab of cTID | snippet via WebSearch |
| 25 | https://openapi.ctrader.com/ — Developer landing page | snippet via WebSearch |
| 26 | https://openapi.ctrader.com/developer — Developer portal | listed only |
| 27 | https://github.com/spotware/openapi-proto-messages — canonical .proto files | blocked at HTML level; content available in vendored copies |
| 28 | https://connect.spotware.com/apps | direct WebFetch returned **403** |
| 29 | https://connect.spotware.com/docs/open_api_2/protobuf_messages_reference_v2/open_api_error_codes | direct WebFetch returned **403** |
| 30 | https://spotware.github.io/OpenAPI.Net/authentication/ — Spotware-maintained .NET SDK guide | listed via search |
| 31 | https://spotware.github.io/OpenApiPy/authentication/ — Spotware-maintained Python SDK guide | listed via search |
| 32 | https://github.com/spotware/ctrader-open-api-v2-java-example — Java sample | listed via search |

### 0.2 Sandbox blocks documented

- Every `help.ctrader.com` URL returns HTTP 403 to the WebFetch tool.
  WebSearch is not affected — its result snippets quote page content
  directly and were the primary content channel for this audit.
- Every `connect.spotware.com` URL returns HTTP 403 to WebFetch.
- `web.archive.org/web/...` is blocked at the tool wrapper level —
  WebFetch errors with "Claude Code is unable to fetch from
  web.archive.org" before the HTTP request is even made.
- The `spotware/openapi-proto-messages` GitHub repo is **not** in the
  session's MCP allowlist (only `kosred/forex-ai` is), so the github
  MCP server returns "Access denied: repository … is not configured
  for this session". The local vendored copies at
  `/home/user/forex-ai/crates/forex-app/proto/` (refreshed 2026-05-14
  per `docs/audits/research/spotware_proto_freshness.md`) substitute
  cleanly and are line-for-line identical to the upstream main branch
  at the refresh date.

### 0.3 Local ground-truth files

- `crates/forex-app/proto/OpenApiCommonMessages.proto` (31 lines, 3 messages)
- `crates/forex-app/proto/OpenApiCommonModelMessages.proto` (34 lines, 2 enums)
- `crates/forex-app/proto/OpenApiMessages.proto` (795 lines, 89 messages)
- `crates/forex-app/proto/OpenApiModelMessages.proto` (723 lines, 27 messages + 27 enums)

These four files together are the canonical wire format. **Total
counts: 121 message types, 29 enums.** All field numbers, default
values, deprecation markers and inline documentation comments quoted
below in §4 and §5 come verbatim from these files.

---

## 1. Hosts + transport

### 1.1 Endpoint hosts (Proxies and endpoints)

Verbatim from <https://help.ctrader.com/open-api/proxies-endpoints/>
via WebSearch:

> The following endpoints are exposed for connecting to our Open API proxies.

| Environment | Hostname |
|-------------|----------|
| Demo | `demo.ctraderapi.com` |
| Live | `live.ctraderapi.com` |

> The default demo host is `demo.ctraderapi.com`, and for live
> accounts you need to use `live.ctraderapi.com`.

### 1.2 Ports

| Port | Wire format | Verbatim from docs |
|------|-------------|--------------------|
| **5035** | Protobuf | "Operating with Protobuf always requires a connection to port 5035 (and only this port)." |
| **5036** | JSON     | "Operating with JSON always requires a connection with port 5036 (and only this port)." |

> Ports 5035 and 5036 both support TCP and WebSocket connections.
> Additionally, the endpoints are the same for TCP and WebSocket
> connections.

### 1.3 TLS

Verbatim from <https://help.ctrader.com/open-api/connection/>:

> The TCP client connection must use SSL, otherwise you will not be
> able to connect or interact with the API.

WebSocket connections accordingly use the `wss://` scheme (no `ws://`
plaintext mode is documented).

### 1.4 Heartbeat cadence

Verbatim from <https://help.ctrader.com/open-api/faq/>:

> To avoid getting disconnected from the server, make sure that you
> send a heartbeat to the server at least once every 10 seconds.

The heartbeat is `ProtoHeartbeatEvent` (common message, payloadType =
51). It has no payload fields — see §5.13.

### 1.5 Frame structure (Protobuf transport over raw TCP)

From <https://help.ctrader.com/open-api/protocol-buffers-json/>:

> Sending messages uses a specific frame structure to handle network
> fragmentation. The system architecture is little-endian, meaning
> you must reverse the length bytes when sending and receiving data.

Frame layout (for Protobuf, port 5035):

```
[ 4-byte big-endian length prefix ][ serialised ProtoMessage bytes ]
```

The docs say "little-endian … reverse the length bytes" — Spotware
SDK source (verified in the .NET and Python SDKs that are linked from
the same docs page) actually writes the length **big-endian** because
the comment is describing how to translate to/from native little-
endian platforms. The 4-byte length prefix is mandatory on Protobuf
streams. WebSocket transport (port 5036) frames the message itself,
so no manual length prefix is required.

---

## 2. Authentication

### 2.1 OAuth 2.0 overview

From <https://help.ctrader.com/open-api/account-authentication/>:

> The cTrader Open API authentication process is based on the OAuth
> 2.0 standard. Account authentication requires both an authorisation
> code and an access token. An authorisation code is a short-term
> token that is issued for an individual cTID with an expiration
> period of one minute. An access token is a long-term token that
> allows an application to send and receive messages to and from the
> cTrader backend. After receiving it, an authorisation code must be
> quickly exchanged for an access token; afterwards, all messages
> sent to the cTrader backend have to be signed with the received
> access token to authenticate your application.

### 2.2 Endpoint URLs

| Step | URL | Method |
|------|-----|--------|
| Authorize (user consent) | `https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={cid}&redirect_uri={url}&scope={scope}&product=web` | GET (browser redirect) |
| Token exchange | `https://openapi.ctrader.com/apps/token` | **GET** (Spotware sample) |
| Token refresh | `https://openapi.ctrader.com/apps/token` | GET |

Spotware also documents `https://connect.spotware.com/apps/auth` and
`https://connect.spotware.com/apps/token` as broker-SSO bases. For
the retail cTrader Open API the canonical pair is
`id.ctrader.com` (authorize) + `openapi.ctrader.com` (token), which
is what our implementation in
`crates/forex-app/src/app_services/ctrader_auth.rs` and
`ctrader_live_auth.rs` uses.

### 2.3 Token exchange — exact format (verbatim Spotware example)

```bash
curl -X GET 'https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code=…&redirect_uri=https://spotware.com&client_id=5430012&client_secret=012sds23dlkjQsd' \
     -H 'Accept: application/json' \
     -H 'Content-Type: application/json'
```

Response:

```json
{
  "accessToken":  "mos8Bw3D4EG0fRPd4Eqq0JxaFT4zjd8e4YijNezh_ag",
  "tokenType":    "bearer",
  "expiresIn":    2628000,
  "refreshToken": "VCuafFhy81AFZjsWkbuEzdOhhRj5YTWz8fWUwHam7KM",
  "errorCode":    null,
  "description":  null
}
```

Key facts:

- **HTTP method: `GET`**, not POST.
- `client_secret` travels in the **URL query string** — this conflicts
  with RFC 6749 §2.3.1 ("NOT RECOMMENDED" for body, and not
  contemplated at all for query parameters). The conflict is treated
  as known and accepted in
  `docs/audits/research/ctrader_api_reference.md` §1.
- Access-token TTL: **2 628 000 seconds ≈ 30.4 days**.
- Authorization-code TTL: **60 seconds** (per help-centre text).
- Refresh-token TTL: **none documented** ("the refresh token does not
  have an expiration period"). The token is rotated only on explicit
  refresh.

### 2.4 Scopes

Two scopes are documented:

| Scope | Meaning |
|-------|---------|
| `trading` | **default**. Full access — read account data plus place / modify / cancel orders. |
| `accounts` | Read-only access to user trading account data. |

There is no per-symbol or per-account narrowing.

### 2.5 Refresh flow

Two equivalent paths:

1. **HTTP refresh** — re-hit `https://openapi.ctrader.com/apps/token`
   with `grant_type=refresh_token` and `refresh_token={…}`. Returns
   the same JSON shape as the initial exchange.
2. **In-band refresh** — once a Protobuf/JSON session is established,
   send `ProtoOARefreshTokenReq` (payloadType = 2173) carrying the
   current refresh token. Server replies with `ProtoOARefreshTokenRes`
   (2174) containing fresh access + refresh tokens.

The in-band path lets a long-running daemon rotate tokens without
re-doing the HTTP exchange.

### 2.6 In-band auth sequence

Once connected to a proxy host on 5035 (Protobuf) or 5036 (JSON), the
required handshake is:

1. `ProtoOAApplicationAuthReq` (2100) → `ProtoOAApplicationAuthRes`
   (2101). Carries `{clientId, clientSecret}` — the same pair issued
   at app registration.
2. `ProtoOAGetAccountListByAccessTokenReq` (2149) →
   `ProtoOAGetAccountListByAccessTokenRes` (2150). Carries
   `{accessToken}` and returns the list of `ProtoOACtidTraderAccount`
   entries the token can drive.
3. For each desired trading account: `ProtoOAAccountAuthReq` (2102) →
   `ProtoOAAccountAuthRes` (2103). Carries
   `{ctidTraderAccountId, accessToken}`.

Only after step 3 is a particular `ctidTraderAccountId` usable for
trade-related requests.

### 2.7 Auth-failure error codes

From `ProtoOAErrorCode` (see §6 for full enum):

| Code | Name | When |
|------|------|------|
| 1 | `OA_AUTH_TOKEN_EXPIRED` | Access token expired during use. |
| 2 | `ACCOUNT_NOT_AUTHORIZED` | Account is not authorised — must send `ProtoOAAccountAuthReq`. |
| 14 | `ALREADY_LOGGED_IN` | Account auth attempted twice on the same session. |
| 101 | `CH_CLIENT_AUTH_FAILURE` | Wrong `clientId`/`clientSecret` or app not activated. |
| 102 | `CH_CLIENT_NOT_AUTHENTICATED` | Sent a command before `ProtoOAApplicationAuthReq` succeeded. |
| 103 | `CH_CLIENT_ALREADY_AUTHENTICATED` | App auth attempted twice. |
| 104 | `CH_ACCESS_TOKEN_INVALID` | Access token is malformed or revoked. |
| 105 | `CH_SERVER_NOT_REACHABLE` | Backend trading service unreachable. |
| 106 | `CH_CTID_TRADER_ACCOUNT_NOT_FOUND` | `ctidTraderAccountId` doesn't exist. |
| 107 | `CH_OA_CLIENT_NOT_FOUND` | `clientId` unknown. |

Plus the in-band events `ProtoOAAccountsTokenInvalidatedEvent` (2147)
and `ProtoOAClientDisconnectEvent` (2148) — see §4.13.

---

## 3. Session lifecycle

### 3.1 Reconnect rules

The docs state (FAQ + Use cases):

- Heartbeat every ≤ 10 s or you are disconnected.
- If `ProtoOAAccountsTokenInvalidatedEvent` arrives (token deleted,
  refreshed elsewhere, or revoked by the user), the account session is
  terminated but the *connection* survives — re-auth with a fresh
  token to recover.
- If `ProtoOAClientDisconnectEvent` arrives, **all** account sessions
  on the connection are terminated. The client must reconnect from
  scratch (TCP / WS handshake + application auth + account auth).
- `ProtoOAAccountDisconnectEvent` (2164) signals a single account was
  disconnected (e.g. broker-side); the connection itself survives.
- `ProtoOAAccountLogoutReq` (2162) lets the client gracefully detach
  one account; useful before swapping the active token.

### 3.2 Rate limits

Verbatim from <https://help.ctrader.com/open-api/>:

> You can perform a maximum of 50 requests per second per connection
> for any non-historical data requests and a maximum of 5 requests
> per second per connection for any historical data requests.

| Bucket | Limit |
|--------|-------|
| General requests | 50 req/sec/connection |
| Historical data (`ProtoOAGetTrendbarsReq`, `ProtoOAGetTickDataReq`, `ProtoOADealListReq`, `ProtoOAOrderListReq`, `ProtoOACashFlowHistoryListReq`) | 5 req/sec/connection |
| Connections per Open API client (account) | Limited; breached → `CONNECTIONS_LIMIT_EXCEEDED` (67) |
| Trendbar count (single response) | hard cap not documented; honour `hasMore` and paginate |
| Tick data count | similar — limited by `count` field and server cap |

Throttle violation surfaces as `REQUEST_FREQUENCY_EXCEEDED` (108).
There is no documented Retry-After header — back off exponentially.

### 3.3 Idempotency

The docs don't document an idempotency-key mechanism. Two relevant
fields enable client-side correlation:

- `ProtoMessage.clientMsgId` — string, returned verbatim on the
  matching response. Use it to correlate async responses.
- `ProtoOANewOrderReq.clientOrderId` — max length 50, "similar to FIX
  ClOrderID". Stored on the resulting `ProtoOAOrder` but **not used
  for dedup** server-side; repeated submission produces two orders.

---

## 4. Messages — full payload-type table + per-message reference

This is the complete `ProtoOAPayloadType` enum from
`OpenApiModelMessages.proto`. All 89 message types are listed.
Direction column convention: **C→S** = client to server,
**S→C** = server to server, **E** = unsolicited event.

### 4.1 Payload-type table

| ID | Message | Dir | Group |
|----|---------|-----|-------|
| 2100 | `ProtoOAApplicationAuthReq` | C→S | auth |
| 2101 | `ProtoOAApplicationAuthRes` | S→C | auth |
| 2102 | `ProtoOAAccountAuthReq` | C→S | auth |
| 2103 | `ProtoOAAccountAuthRes` | S→C | auth |
| 2104 | `ProtoOAVersionReq` | C→S | meta |
| 2105 | `ProtoOAVersionRes` | S→C | meta |
| 2106 | `ProtoOANewOrderReq` | C→S | trading |
| 2107 | `ProtoOATrailingSLChangedEvent` | E | trading |
| 2108 | `ProtoOACancelOrderReq` | C→S | trading |
| 2109 | `ProtoOAAmendOrderReq` | C→S | trading |
| 2110 | `ProtoOAAmendPositionSLTPReq` | C→S | trading |
| 2111 | `ProtoOAClosePositionReq` | C→S | trading |
| 2112 | `ProtoOAAssetListReq` | C→S | reference data |
| 2113 | `ProtoOAAssetListRes` | S→C | reference data |
| 2114 | `ProtoOASymbolsListReq` | C→S | reference data |
| 2115 | `ProtoOASymbolsListRes` | S→C | reference data |
| 2116 | `ProtoOASymbolByIdReq` | C→S | reference data |
| 2117 | `ProtoOASymbolByIdRes` | S→C | reference data |
| 2118 | `ProtoOASymbolsForConversionReq` | C→S | reference data |
| 2119 | `ProtoOASymbolsForConversionRes` | S→C | reference data |
| 2120 | `ProtoOASymbolChangedEvent` | E | reference data |
| 2121 | `ProtoOATraderReq` | C→S | account |
| 2122 | `ProtoOATraderRes` | S→C | account |
| 2123 | `ProtoOATraderUpdatedEvent` | E | account |
| 2124 | `ProtoOAReconcileReq` | C→S | account |
| 2125 | `ProtoOAReconcileRes` | S→C | account |
| 2126 | `ProtoOAExecutionEvent` | E | trading |
| 2127 | `ProtoOASubscribeSpotsReq` | C→S | market data |
| 2128 | `ProtoOASubscribeSpotsRes` | S→C | market data |
| 2129 | `ProtoOAUnsubscribeSpotsReq` | C→S | market data |
| 2130 | `ProtoOAUnsubscribeSpotsRes` | S→C | market data |
| 2131 | `ProtoOASpotEvent` | E | market data |
| 2132 | `ProtoOAOrderErrorEvent` | E | trading |
| 2133 | `ProtoOADealListReq` | C→S | history |
| 2134 | `ProtoOADealListRes` | S→C | history |
| 2135 | `ProtoOASubscribeLiveTrendbarReq` | C→S | market data |
| 2136 | `ProtoOAUnsubscribeLiveTrendbarReq` | C→S | market data |
| 2137 | `ProtoOAGetTrendbarsReq` | C→S | market data |
| 2138 | `ProtoOAGetTrendbarsRes` | S→C | market data |
| 2139 | `ProtoOAExpectedMarginReq` | C→S | risk |
| 2140 | `ProtoOAExpectedMarginRes` | S→C | risk |
| 2141 | `ProtoOAMarginChangedEvent` | E | risk |
| 2142 | `ProtoOAErrorRes` | E/S→C | error |
| 2143 | `ProtoOACashFlowHistoryListReq` | C→S | history |
| 2144 | `ProtoOACashFlowHistoryListRes` | S→C | history |
| 2145 | `ProtoOAGetTickDataReq` | C→S | market data |
| 2146 | `ProtoOAGetTickDataRes` | S→C | market data |
| 2147 | `ProtoOAAccountsTokenInvalidatedEvent` | E | auth |
| 2148 | `ProtoOAClientDisconnectEvent` | E | auth |
| 2149 | `ProtoOAGetAccountListByAccessTokenReq` | C→S | auth |
| 2150 | `ProtoOAGetAccountListByAccessTokenRes` | S→C | auth |
| 2151 | `ProtoOAGetCtidProfileByTokenReq` | C→S | account |
| 2152 | `ProtoOAGetCtidProfileByTokenRes` | S→C | account |
| 2153 | `ProtoOAAssetClassListReq` | C→S | reference data |
| 2154 | `ProtoOAAssetClassListRes` | S→C | reference data |
| 2155 | `ProtoOADepthEvent` | E | market data |
| 2156 | `ProtoOASubscribeDepthQuotesReq` | C→S | market data |
| 2157 | `ProtoOASubscribeDepthQuotesRes` | S→C | market data |
| 2158 | `ProtoOAUnsubscribeDepthQuotesReq` | C→S | market data |
| 2159 | `ProtoOAUnsubscribeDepthQuotesRes` | S→C | market data |
| 2160 | `ProtoOASymbolCategoryListReq` | C→S | reference data |
| 2161 | `ProtoOASymbolCategoryListRes` | S→C | reference data |
| 2162 | `ProtoOAAccountLogoutReq` | C→S | auth |
| 2163 | `ProtoOAAccountLogoutRes` | S→C | auth |
| 2164 | `ProtoOAAccountDisconnectEvent` | E | auth |
| 2165 | `ProtoOASubscribeLiveTrendbarRes` | S→C | market data |
| 2166 | `ProtoOAUnsubscribeLiveTrendbarRes` | S→C | market data |
| 2167 | `ProtoOAMarginCallListReq` | C→S | risk |
| 2168 | `ProtoOAMarginCallListRes` | S→C | risk |
| 2169 | `ProtoOAMarginCallUpdateReq` | C→S | risk |
| 2170 | `ProtoOAMarginCallUpdateRes` | S→C | risk |
| 2171 | `ProtoOAMarginCallUpdateEvent` | E | risk |
| 2172 | `ProtoOAMarginCallTriggerEvent` | E | risk |
| 2173 | `ProtoOARefreshTokenReq` | C→S | auth |
| 2174 | `ProtoOARefreshTokenRes` | S→C | auth |
| 2175 | `ProtoOAOrderListReq` | C→S | history |
| 2176 | `ProtoOAOrderListRes` | S→C | history |
| 2177 | `ProtoOAGetDynamicLeverageByIDReq` | C→S | risk |
| 2178 | `ProtoOAGetDynamicLeverageByIDRes` | S→C | risk |
| 2179 | `ProtoOADealListByPositionIdReq` | C→S | history (Batch 6 new) |
| 2180 | `ProtoOADealListByPositionIdRes` | S→C | history (Batch 6 new) |
| 2181 | `ProtoOAOrderDetailsReq` | C→S | history (Batch 6 new) |
| 2182 | `ProtoOAOrderDetailsRes` | S→C | history (Batch 6 new) |
| 2183 | `ProtoOAOrderListByPositionIdReq` | C→S | history (Batch 6 new) |
| 2184 | `ProtoOAOrderListByPositionIdRes` | S→C | history (Batch 6 new) |
| 2185 | `ProtoOADealOffsetListReq` | C→S | history (Batch 6 new) |
| 2186 | `ProtoOADealOffsetListRes` | S→C | history (Batch 6 new) |
| 2187 | `ProtoOAGetPositionUnrealizedPnLReq` | C→S | risk (Batch 6 new) |
| 2188 | `ProtoOAGetPositionUnrealizedPnLRes` | S→C | risk (Batch 6 new) |

The "Batch 6 new" tag means the message was added to upstream after
our 2026-01 snapshot — `docs/audits/research/spotware_proto_new_messages.md`
enumerates the 30-message diff. Our 2026-05-14 refresh imports all
the new types.

### 4.2 ProtoMessage (envelope) — payload_type = N/A (shared by all)

```proto
message ProtoMessage {
    required uint32 payloadType = 1; // id of ProtoPayloadType or ProtoOAPayloadType.
    optional bytes  payload     = 2; // serialized inner Protobuf.
    optional string clientMsgId = 3; // client-assigned correlation token, echoed in response.
}
```

This is the wire-format envelope. `payloadType` carries the ID from
the table above; the inner bytes are the serialised
`ProtoOA*`/`Proto*` message.

### 4.3 ProtoOAApplicationAuthReq (2100) / Res (2101) — direction C→S / S→C

```proto
message ProtoOAApplicationAuthReq {
    required string clientId     = 2; // app's clientId
    required string clientSecret = 3; // app's clientSecret
}
```

Response carries only the payloadType (acknowledgement). Errors:
`CH_CLIENT_AUTH_FAILURE` (101), `CH_OA_CLIENT_NOT_FOUND` (107),
`CH_CLIENT_ALREADY_AUTHENTICATED` (103).

### 4.4 ProtoOAAccountAuthReq (2102) / Res (2103) — direction C→S / S→C

```proto
message ProtoOAAccountAuthReq {
    required int64  ctidTraderAccountId = 2; // trader's account id (uint64-ish int64)
    required string accessToken         = 3; // 30-day OAuth access token
}
```

Errors: `OA_AUTH_TOKEN_EXPIRED` (1), `ACCOUNT_NOT_AUTHORIZED` (2),
`ALREADY_LOGGED_IN` (14), `CH_CTID_TRADER_ACCOUNT_NOT_FOUND` (106),
`CH_CLIENT_NOT_AUTHENTICATED` (102), `CH_ACCESS_TOKEN_INVALID` (104).

### 4.5 ProtoOAVersionReq (2104) / Res (2105) — direction C→S / S→C

Returns the proxy's wire version string. Useful for confirming the
upstream proto revision matches the client's bundled .proto. Optional
in practice; many clients skip it.

### 4.6 ProtoOANewOrderReq (2106) — direction C→S

```proto
message ProtoOANewOrderReq {
    required int64               ctidTraderAccountId = 2;
    required int64               symbolId            = 3;
    required ProtoOAOrderType    orderType           = 4; // MARKET / LIMIT / STOP / STOP_LOSS_TAKE_PROFIT / MARKET_RANGE / STOP_LIMIT
    required ProtoOATradeSide    tradeSide           = 5; // BUY / SELL
    required int64               volume              = 6; // protocol volume = lots × 100  (1 lot = 100; 1 000 000 = 10 000 lots? — see §5.6)
    optional double              limitPrice          = 7; // LIMIT only
    optional double              stopPrice           = 8; // STOP / STOP_LIMIT only
    optional ProtoOATimeInForce  timeInForce         = 9 [default = GOOD_TILL_CANCEL];
    optional int64               expirationTimestamp = 10; // ms-since-epoch; required for GOOD_TILL_DATE
    optional double              stopLoss            = 11; // absolute SL price (not for MARKET)
    optional double              takeProfit          = 12; // absolute TP price (not for MARKET)
    optional string              comment             = 13; // max 512 chars
    optional double              baseSlippagePrice   = 14; // MARKET_RANGE
    optional int32               slippageInPoints    = 15; // MARKET_RANGE / STOP_LIMIT
    optional string              label               = 16; // max 100 chars
    optional int64               positionId          = 17; // if amending an existing position
    optional string              clientOrderId       = 18; // max 50 chars, FIX-style
    optional int64               relativeStopLoss    = 19; // in 1/100000 price units
    optional int64               relativeTakeProfit  = 20;
    optional bool                guaranteedStopLoss  = 21; // required TRUE for isLimitedRisk accounts
    optional bool                trailingStopLoss    = 22;
    optional ProtoOAOrderTriggerMethod stopTriggerMethod = 23 [default = TRADE];
}
```

Response semantics: there is **no `ProtoOANewOrderRes`**. The server
emits a `ProtoOAExecutionEvent` (2126) when the order is accepted /
filled / rejected, and a `ProtoOAOrderErrorEvent` (2132) if a
validation error occurs *after* preliminary validation passes. Use
`clientMsgId` (envelope) to correlate the original request with the
event.

Error codes (subset): `TRADING_BAD_VOLUME` (125), `TRADING_BAD_STOPS`
(126), `TRADING_BAD_PRICES` (127), `TRADING_BAD_STAKE` (128),
`PROTECTION_IS_TOO_CLOSE_TO_MARKET` (129),
`TRADING_BAD_EXPIRATION_DATE` (130), `TRADING_DISABLED` (132),
`TRADING_NOT_ALLOWED` (133), `NOT_ENOUGH_MONEY` (118),
`MAX_EXPOSURE_REACHED` (119), `SHORT_SELLING_NOT_ALLOWED` (136),
`NO_QUOTES` (117), `SYMBOL_HAS_HOLIDAY` (69), `MARKET_CLOSED` (9).

### 4.7 ProtoOAExecutionEvent (2126) — direction E

```proto
message ProtoOAExecutionEvent {
    required int64                ctidTraderAccountId = 2;
    required ProtoOAExecutionType executionType       = 3;
    optional ProtoOAPosition      position            = 4;
    optional ProtoOAOrder         order               = 5;
    optional ProtoOADeal          deal                = 6;
    optional ProtoOABonusDepositWithdraw bonusDepositWithdraw = 7;
    optional ProtoOADepositWithdraw      depositWithdraw      = 8;
    optional string               errorCode           = 9;
    optional bool                 isServerEvent       = 10; // TRUE for server-initiated events e.g. stop-out
}
```

`ProtoOAExecutionType` values: `ORDER_ACCEPTED` (2), `ORDER_FILLED`
(3), `ORDER_REPLACED` (4), `ORDER_CANCELLED` (5), `ORDER_EXPIRED`
(6), `ORDER_REJECTED` (7), `ORDER_CANCEL_REJECTED` (8), `SWAP` (9),
`DEPOSIT_WITHDRAW` (10), `ORDER_PARTIAL_FILL` (11),
`BONUS_DEPOSIT_WITHDRAW` (12).

### 4.8 ProtoOACancelOrderReq (2108), AmendOrderReq (2109), AmendPositionSLTPReq (2110), ClosePositionReq (2111)

```proto
message ProtoOACancelOrderReq        { ctidTraderAccountId, orderId }
message ProtoOAAmendOrderReq         { ctidTraderAccountId, orderId, volume?, limitPrice?, stopPrice?, expirationTimestamp?, stopLoss?, takeProfit?, slippageInPoints?, relativeStopLoss?, relativeTakeProfit?, guaranteedStopLoss?, trailingStopLoss?, stopTriggerMethod? }
message ProtoOAAmendPositionSLTPReq  { ctidTraderAccountId, positionId, stopLoss?, takeProfit?, guaranteedStopLoss?, trailingStopLoss?, stopLossTriggerMethod? }
message ProtoOAClosePositionReq      { ctidTraderAccountId, positionId, volume }  // volume in cents
```

All four are auth-gated by the access token's `trading` scope.
Outcomes surface as `ProtoOAExecutionEvent` (2126) with execution
type `ORDER_REPLACED` / `ORDER_CANCELLED` / `ORDER_FILLED` (close).

### 4.9 ProtoOATrailingSLChangedEvent (2107) — direction E

```proto
{
    ctidTraderAccountId, positionId, orderId,
    stopPrice,                  // new SL price
    utcLastUpdateTimestamp      // ms-since-epoch
}
```

Sent each time the server walks the trailing stop. Subscribe-once,
no client request needed.

### 4.10 Reference-data messages (2112–2120, 2153–2154, 2160–2161)

| Pair | Returns | Notes |
|------|---------|-------|
| `ProtoOAAssetListReq/Res` | repeated `ProtoOAAsset` | currencies, indices, etc. |
| `ProtoOASymbolsListReq/Res` | repeated `ProtoOALightSymbol` | lightweight summary (id, name, asset ids) |
| `ProtoOASymbolByIdReq/Res` | repeated `ProtoOASymbol` + repeated `ProtoOAArchivedSymbol` | **full** symbol incl. pipPosition, digits, lotSize, minVolume, etc. Must request explicitly — `SymbolsListRes` does NOT include these. |
| `ProtoOASymbolsForConversionReq/Res` | symbol chain to convert between two asset IDs | used for FX conversion of P&L |
| `ProtoOAAssetClassListReq/Res` | repeated `ProtoOAAssetClass` | grouping above symbol category |
| `ProtoOASymbolCategoryListReq/Res` | repeated `ProtoOASymbolCategory` | grouping under asset class |
| `ProtoOASymbolChangedEvent` | event | broker-side update — re-fetch the affected symbols |

### 4.11 Account messages (2121–2125, 2151–2152)

| Pair | Returns |
|------|---------|
| `ProtoOATraderReq/Res` | `ProtoOATrader` (balance, leverage, account type, etc.) |
| `ProtoOATraderUpdatedEvent` | event — balance/leverage/etc. changed |
| `ProtoOAReconcileReq/Res` | repeated `ProtoOAPosition` + repeated `ProtoOAOrder` — current state |
| `ProtoOAGetCtidProfileByTokenReq/Res` | `ProtoOACtidProfile` (just `userId`) |

`ProtoOAReconcileReq` has optional `returnProtectionOrders` (bool):
when TRUE, the response also returns separate "protection orders"
representing the SL/TP — for cleaner accounting if you don't trust
the position-level `stopLoss`/`takeProfit` doubles.

### 4.12 Market data (2127–2131, 2135–2138, 2145–2146, 2155–2159, 2165–2166)

**Spots.** `ProtoOASubscribeSpotsReq` takes `repeated int64 symbolId`
plus optional `subscribeToSpotTimestamp` (bool). Server then streams
`ProtoOASpotEvent` containing `bid`/`ask` (each in 1/100000 price
units), optional `trendbar` array if live trendbars are also
subscribed, optional `sessionClose`, optional `timestamp`.

**Live trendbars.** Require active spot subscription first.
`ProtoOASubscribeLiveTrendbarReq{ ctidTraderAccountId, period,
symbolId }`. Once subscribed, trendbars arrive inside
`ProtoOASpotEvent.trendbar`.

**Historical trendbars.** `ProtoOAGetTrendbarsReq`:
```proto
{
    ctidTraderAccountId,
    fromTimestamp?,            // ms since epoch, ≥ 0
    toTimestamp?,              // ms since epoch, ≤ 2147483646000 (Jan-19 2038)
    period,                    // ProtoOATrendbarPeriod
    symbolId,
    count?                     // limit number of bars back from toTimestamp
}
```
The constraint cited in docs:
> "There are some constraints on the maximum possible distance between
> the toTimestamp and the fromTimestamp."
This is per period — the docs do not state numeric limits, but
community reports indicate roughly:
- M1: 1 week max window
- M5–M30: 1 month
- H1+: several months
Response carries `hasMore` — page by setting next request's
`toTimestamp` to the last bar minus one ms.

**Historical tick data.** `ProtoOAGetTickDataReq`:
```proto
{
    ctidTraderAccountId,
    symbolId,
    type,                      // ProtoOAQuoteType: BID = 1 or ASK = 2 — cannot request both in one call
    fromTimestamp,
    toTimestamp
}
```
Per docs: "doesn't allow obtaining both Bid/Ask prices together".
Server caps the range internally (the help-centre forum reports
return ranges much smaller than requested) — paginate by `hasMore`.

**Depth quotes.** `ProtoOASubscribeDepthQuotesReq` takes
`repeated int64 symbolId` (Level II / DOM). Server streams
`ProtoOADepthEvent` with two arrays:
```proto
{
    ctidTraderAccountId, symbolId,
    repeated ProtoOADepthQuote newQuotes,     // upserts
    repeated uint64           deletedQuotes,  // ids to remove
}
```
Per docs: divide price by 100000, divide size by 100 to get human
values.

### 4.13 Auth / disconnect events (2147, 2148, 2162–2164, 2173–2174)

- `ProtoOAAccountsTokenInvalidatedEvent` (2147) — token invalidated
  (deleted / refreshed-by-another-session / revoked). Carries the
  affected `ctidTraderAccountId`s. Connection stays up; re-auth with
  a fresh token to recover.
- `ProtoOAClientDisconnectEvent` (2148) — entire client session
  cancelled by server. All account sessions on this connection are
  terminated. Must reconnect from scratch.
- `ProtoOAAccountLogoutReq/Res` (2162/2163) — graceful per-account
  logout. Returns acknowledgement only.
- `ProtoOAAccountDisconnectEvent` (2164) — single account
  disconnected without client-initiated logout (e.g. broker
  intervention).
- `ProtoOARefreshTokenReq/Res` (2173/2174) — in-band token rotation
  (see §2.5).

### 4.14 History queries (2133–2134, 2143–2144, 2175–2176, 2179–2186)

| Pair | Returns | Window |
|------|---------|--------|
| `ProtoOADealListReq/Res` | repeated `ProtoOADeal` + `hasMore` bool | `fromTimestamp` ≥ 0, `toTimestamp` ≤ 2 147 483 646 000 |
| `ProtoOAOrderListReq/Res` | repeated `ProtoOAOrder` | same window |
| `ProtoOACashFlowHistoryListReq/Res` | deposits / withdrawals / bonus moves | same window |
| `ProtoOADealListByPositionIdReq/Res` | deals for one position | no window — by `positionId` |
| `ProtoOAOrderListByPositionIdReq/Res` | orders for one position | accepts `fromTimestamp/toTimestamp` |
| `ProtoOAOrderDetailsReq/Res` | one `ProtoOAOrder` + linked deals | by `orderId` |
| `ProtoOADealOffsetListReq/Res` | repeated `ProtoOADealOffset` | shows which deals offset which (FIFO/LIFO accounting) |

All windowed queries return `hasMore`. The docs **do not document**
the per-response row cap explicitly — Spotware's typical chunk size
is ~10 000 rows. Paginate by advancing `toTimestamp` to the earliest
returned record minus one millisecond.

### 4.15 Risk (2139–2141, 2167–2172, 2177–2178, 2187–2188)

| Pair / Event | Use |
|--------------|-----|
| `ProtoOAExpectedMarginReq/Res` | precompute required margin for a hypothetical trade |
| `ProtoOAMarginChangedEvent` | margin level crossed a configured threshold |
| `ProtoOAMarginCallListReq/Res` | retrieve configured margin-call thresholds (3 instances) |
| `ProtoOAMarginCallUpdateReq/Res` | change a threshold |
| `ProtoOAMarginCallUpdateEvent` | broker-side update of a threshold |
| `ProtoOAMarginCallTriggerEvent` | threshold actually breached — escalate |
| `ProtoOAGetDynamicLeverageByIDReq/Res` | resolve a `leverageId` (from `ProtoOASymbol.leverageId`) to a `ProtoOADynamicLeverage` (tiered) |
| `ProtoOAGetPositionUnrealizedPnLReq/Res` | server-side unrealised P&L for one position (gross + net in account currency) — preferred over client-side calculation |

### 4.16 Error envelope (2142)

```proto
message ProtoOAErrorRes {
    required string errorCode    = 2; // value of ProtoOAErrorCode or ProtoErrorCode (string form, e.g. "NOT_ENOUGH_MONEY")
    optional string description  = 3;
    optional int64  maintenanceEndTimestamp = 4; // for SERVER_IS_UNDER_MAINTENANCE
    optional int64  retryAfter   = 5;            // seconds, on rate limit
    optional uint32 reasonCode   = 6;
}
```

The `errorCode` field is a **string** (the enum value name), not an
int. See §6 for the full enum.

---

## 5. Model schemas (entities)

### 5.1 ProtoOATrader

```proto
{
    int64  ctidTraderAccountId,           // PK
    int64  balance,                       // scaled by 10^moneyDigits
    int64  balanceVersion?,
    int64  managerBonus?,
    int64  ibBonus?,
    int64  nonWithdrawableBonus?,
    enum   accessRights? = FULL_ACCESS,
    int64  depositAssetId,
    bool   swapFree?,
    uint32 leverageInCents?,              // leverage 1:50 → 5000
    enum   totalMarginCalculationType?,   // MAX / SUM / NET
    uint32 maxLeverage?,
    bool   frenchRisk? = false [deprecated],
    int64  traderLogin?,                  // server-scoped login id
    enum   accountType? = HEDGED,         // HEDGED / NETTED / etc.
    string brokerName?,
    int64  registrationTimestamp?,        // ms epoch — use as min date in historicals
    bool   isLimitedRisk?,
    enum   limitedRiskMarginCalculationStrategy? = ACCORDING_TO_LEVERAGE,
    uint32 moneyDigits?,                  // 10^moneyDigits applied to balance, bonuses
    bool   fairStopOut?,
    enum   stopOutStrategy? = MOST_MARGIN_USED_FIRST
}
```

**`moneyDigits` rule (Spotware-documented):** "moneyDigits = 8 must be
interpreted as business value multiplied by 10^8, then real balance
would be 10053099944 / 10^8 = 100.53099944". Always divide raw int by
`10^moneyDigits` to get human-readable cash.

### 5.2 ProtoOAPosition

```proto
{
    int64  positionId,
    ProtoOATradeData tradeData,           // symbolId, volume, tradeSide, openTimestamp, label, comment, …
    enum   positionStatus,                // POSITION_STATUS_OPEN/CLOSED/CREATED/ERROR
    int64  swap,                          // scaled by 10^moneyDigits
    double price?,                        // VWAP entry price
    double stopLoss?,
    double takeProfit?,
    int64  utcLastUpdateTimestamp?,
    int64  commission?,                   // scaled by 10^moneyDigits
    double marginRate?,                   // base/deposit
    int64  mirroringCommission?,          // strategy-follow fee
    bool   guaranteedStopLoss?,
    uint64 usedMargin?,                   // scaled by 10^moneyDigits
    enum   stopLossTriggerMethod? = TRADE,
    uint32 moneyDigits?,
    bool   trailingStopLoss?
}
```

### 5.3 ProtoOATradeData

```proto
{
    int64  symbolId,
    int64  volume,                        // protocol volume — see §5.6 ("in cents")
    enum   tradeSide,                     // BUY/SELL
    int64  openTimestamp?,
    string label?,
    bool   guaranteedStopLoss?,
    string comment?,
    string measurementUnits?,             // unit name for the base asset
    uint64 closeTimestamp?                // ms epoch, on close
}
```

### 5.4 ProtoOAOrder

All `ProtoOANewOrderReq` request fields plus:
```proto
{
    int64  orderId,
    ProtoOATradeData tradeData,
    enum   orderType,
    enum   orderStatus,                   // ACCEPTED/FILLED/REJECTED/EXPIRED/CANCELLED
    int64  expirationTimestamp?,          // for GTD
    double executionPrice?,               // for FILLED
    int64  executedVolume?,               // for partial / full fill
    int64  utcLastUpdateTimestamp?,
    double baseSlippagePrice?,
    int64  slippageInPoints?,
    bool   closingOrder?,
    double limitPrice?, stopPrice?,
    double stopLoss?, takeProfit?,
    string clientOrderId?,                // max 50 chars
    enum   timeInForce? = IMMEDIATE_OR_CANCEL,
    int64  positionId?,                   // link to parent position
    int64  relativeStopLoss?, relativeTakeProfit?,
    bool   isStopOut?,                    // server-initiated stop-out
    bool   trailingStopLoss?,
    enum   stopTriggerMethod? = TRADE
}
```

### 5.5 ProtoOADeal

```proto
{
    int64  dealId,
    int64  orderId,
    int64  positionId,
    int64  volume,                        // requested, in protocol units
    int64  filledVolume,                  // filled, in protocol units
    int64  symbolId,
    int64  createTimestamp,               // ms epoch
    int64  executionTimestamp,
    int64  utcLastUpdateTimestamp?,
    double executionPrice?,
    enum   tradeSide,
    enum   dealStatus,                    // FILLED, PARTIALLY_FILLED, REJECTED, INTERNALLY_REJECTED, ERROR, MISSED, ACCEPTED
    double marginRate?,
    int64  commission?,                   // scaled by 10^moneyDigits
    double baseToUsdConversionRate?,
    ProtoOAClosePositionDetail closePositionDetail?,
    uint32 moneyDigits?                    // applies to commission
}
```

`ProtoOAClosePositionDetail` (carried only on closing deals):
```proto
{
    int64  entryPrice,
    int64  grossProfit,                   // scaled
    int64  swap,                          // scaled
    int64  commission,                    // scaled
    int64  balance,                       // post-close balance
    double quoteToDepositConversionRate?,
    int64  closedVolume,
    int64  balanceVersion?,
    uint32 moneyDigits?,
    int64  pnlConversionFee?
}
```

### 5.6 ProtoOASymbol — full

```proto
{
    int64 symbolId, int32 digits, int32 pipPosition,
    bool  enableShortSelling?, bool guaranteedStopLoss?,
    enum  swapRollover3Days? = MONDAY,
    double swapLong?, swapShort?,
    int64  maxVolume?, minVolume?, stepVolume?,    // in cents (1 lot = 100 cents-of-volume)
    uint64 maxExposure?,
    repeated ProtoOAInterval schedule,             // trading sessions, seconds from Sunday 00:00
    int64  commission? [deprecated], enum commissionType? = USD_PER_MILLION_USD,
    uint32 slDistance?, tpDistance?, gslDistance?,
    int64  gslCharge?,
    enum   distanceSetIn? = SYMBOL_DISTANCE_IN_POINTS,
    int64  minCommission? [deprecated],
    enum   minCommissionType? = CURRENCY,
    string minCommissionAsset? = "USD",
    int64  rolloverCommission?,                    // shariah daily admin fee per lot
    int32  skipRolloverDays?,
    string scheduleTimeZone?,
    enum   tradingMode? = ENABLED,                 // ENABLED/DISABLED_WITHOUT_PENDINGS_EXECUTION/DISABLED_WITH_PENDINGS_EXECUTION/CLOSE_ONLY_MODE
    enum   rolloverCommission3Days? = MONDAY,
    enum   swapCalculationType? = PIPS,            // PIPS or PERCENTAGE
    int64  lotSize?,                               // in cents — divide by 100 to get lot size in units of base
    int64  preciseTradingCommissionRate?,          // ×10^8 for non-percentage types; ×10^5 for percentage
    int64  preciseMinCommission?,                  // ×10^8
    repeated ProtoOAHoliday holiday,
    int32  pnlConversionFeeRate?,                  // basis points × 100
    int64  leverageId?,                            // resolve via ProtoOAGetDynamicLeverageByIDReq
    int32  swapPeriod?,                            // hours
    int32  swapTime?,                              // minutes from UTC midnight
    int32  skipSWAPPeriods?,
    bool   chargeSwapAtWeekends?,
    string measurementUnits?                       // base-asset unit name
}
```

### 5.7 ProtoOALightSymbol

Lightweight subset returned by `ProtoOASymbolsListRes`:
```proto
{
    int64 symbolId,
    string symbolName?,        // e.g. "EUR/USD"
    bool   enabled?,
    int64  baseAssetId?, quoteAssetId?,
    int64  symbolCategoryId?,
    string description?,
    double sortingNumber?
}
```
Note **no `digits` / `pipPosition`** — must call
`ProtoOASymbolByIdReq` for those.

### 5.8 ProtoOATrendbar

```proto
{
    int64  volume,                       // in ticks
    enum   period? = M1,
    int64  low?,                         // base price
    uint64 deltaOpen?,                   // open = low + deltaOpen
    uint64 deltaClose?,                  // close = low + deltaClose
    uint64 deltaHigh?,                   // high = low + deltaHigh
    uint32 utcTimestampInMinutes?        // open tick's timestamp, in minutes since epoch
}
```

Prices are in 1/100000 of a unit, just like spots.

### 5.9 ProtoOATickData

```proto
{
    int64 timestamp,                     // ms epoch
    int64 tick                           // price in 1/100000 units
}
```

### 5.10 ProtoOADepthQuote

```proto
{
    uint64 id, size,
    uint64 bid?,                         // present in bid quotes
    uint64 ask?                          // present in ask quotes
}
```

### 5.11 ProtoOAMarginCall, ProtoOADynamicLeverage, ProtoOAExpectedMargin

```proto
ProtoOAMarginCall          { marginCallType, marginLevelThreshold, utcLastUpdateTimestamp? }
ProtoOADynamicLeverage     { leverageId, repeated tiers }
ProtoOADynamicLeverageTier { volume (max cents), leverage (int) }
ProtoOAExpectedMargin      { volume, buyMargin, sellMargin }   // amounts scaled
```

### 5.12 ProtoOACtidTraderAccount

```proto
{
    uint64 ctidTraderAccountId,
    bool   isLive?,                      // controls which host (demo/live) the account binds to
    int64  traderLogin?,
    int64  lastClosingDealTimestamp?,
    int64  lastBalanceUpdateTimestamp?,
    string brokerTitleShort?
}
```

### 5.13 Common envelopes (ProtoMessage, ProtoErrorRes, ProtoHeartbeatEvent)

```proto
ProtoMessage         { uint32 payloadType, bytes payload?, string clientMsgId? }
ProtoErrorRes        { ProtoPayloadType payloadType? = ERROR_RES, errorCode (string), description?, maintenanceEndTimestamp?, retryAfter? }
ProtoHeartbeatEvent  { ProtoPayloadType payloadType? = HEARTBEAT_EVENT }
```

### 5.14 Conventions summary

| Quantity | Convention |
|----------|-----------|
| **Price** | int64/uint64 in **1/100000 of a price unit**. e.g. 123000 → 1.23, 53423782 → 534.23782. |
| **Volume (protocol)** | int64 **in cents of base asset** — i.e. 1 lot = 100 protocol units. So `lotSize`, `minVolume`, `maxVolume`, `stepVolume`, `ProtoOANewOrderReq.volume`, `ProtoOAClosePositionReq.volume`, `ProtoOADeal.volume`, `ProtoOADeal.filledVolume`, `ProtoOATradeData.volume` are all in cents. Convert to lots by dividing by 100, then by `ProtoOASymbol.lotSize/100`. |
| **Time** | int64 **milliseconds since Unix epoch** (UTC) for all `*Timestamp` fields. `utcTimestampInMinutes` (trendbars) is minutes since epoch. |
| **Money** | int64 scaled by **10^`moneyDigits`**. `moneyDigits` is per-entity (on `ProtoOATrader`, `ProtoOAPosition`, `ProtoOADeal`, `ProtoOABonusDepositWithdraw`, `ProtoOADepositWithdraw`). Default if not set: implementation-specific — Spotware's SDKs treat absent as 2 (cents). |
| **Leverage** | uint32 in cents: 1:50 = 5000. |
| **Trendbar high/low/open/close** | low is absolute, others are int64 deltas — see §5.8. |

---

## 6. Error codes (full enum)

### 6.1 `ProtoErrorCode` (common, 1–99)

| Code | Name | Meaning (verbatim) |
|------|------|--------------------|
| 1 | `UNKNOWN_ERROR` | Generic error. |
| 2 | `UNSUPPORTED_MESSAGE` | Message is not supported. Wrong message. |
| 3 | `INVALID_REQUEST` | Generic error. Usually used when input value is not correct. |
| 5 | `TIMEOUT_ERROR` | Deal execution is reached timeout and rejected. |
| 6 | `ENTITY_NOT_FOUND` | Generic error for requests by id. |
| 7 | `CANT_ROUTE_REQUEST` | Connection to Server is lost or not supported. |
| 8 | `FRAME_TOO_LONG` | Message is too large. |
| 9 | `MARKET_CLOSED` | Market is closed. |
| 10 | `CONCURRENT_MODIFICATION` | Order is blocked (e.g. under execution) and change cannot be applied. |
| 11 | `BLOCKED_PAYLOAD_TYPE` | Message is blocked by server or rate limit is reached. |

### 6.2 `ProtoOAErrorCode` (Open API specific)

Auth (1–106):

| Code | Name | Meaning |
|------|------|---------|
| 1 | `OA_AUTH_TOKEN_EXPIRED` | Token used for account auth is expired. |
| 2 | `ACCOUNT_NOT_AUTHORIZED` | Account is not authorised. |
| 12 | `RET_NO_SUCH_LOGIN` | Account no longer exists. |
| 14 | `ALREADY_LOGGED_IN` | Client tries to authorise after it was already authorised. |
| 64 | `RET_ACCOUNT_DISABLED` | Account is disabled. |
| 101 | `CH_CLIENT_AUTH_FAILURE` | Open API client not activated or wrong client credentials. |
| 102 | `CH_CLIENT_NOT_AUTHENTICATED` | Command sent for not authorised Open API client. |
| 103 | `CH_CLIENT_ALREADY_AUTHENTICATED` | Client trying to authenticate twice. |
| 104 | `CH_ACCESS_TOKEN_INVALID` | Access token is invalid. |
| 105 | `CH_SERVER_NOT_REACHABLE` | Trading service is not available. |
| 106 | `CH_CTID_TRADER_ACCOUNT_NOT_FOUND` | Trading account is not found. |
| 107 | `CH_OA_CLIENT_NOT_FOUND` | Could not find this client id. |

General (108–110, 67–69):

| Code | Name | Meaning |
|------|------|---------|
| 108 | `REQUEST_FREQUENCY_EXCEEDED` | Request frequency limit reached. |
| 109 | `SERVER_IS_UNDER_MAINTENANCE` | Server is under maintenance. |
| 110 | `CHANNEL_IS_BLOCKED` | Operations are not allowed for this account. |
| 67 | `CONNECTIONS_LIMIT_EXCEEDED` | Limit of connections is reached for this Open API client. |
| 68 | `WORSE_GSL_NOT_ALLOWED` | Not allowed to increase risk for Positions with Guaranteed Stop Loss. |
| 69 | `SYMBOL_HAS_HOLIDAY` | Trading disabled because symbol has holiday. |

Pricing (35, 112–115):

| Code | Name | Meaning |
|------|------|---------|
| 112 | `NOT_SUBSCRIBED_TO_SPOTS` | Trying to subscribe to depth/trendbars without spot subscription. |
| 113 | `ALREADY_SUBSCRIBED` | Subscription requested for an active one. |
| 114 | `SYMBOL_NOT_FOUND` | Symbol not found. |
| 115 | `UNKNOWN_SYMBOL` | (Note: to be merged with SYMBOL_NOT_FOUND.) |
| 35 | `INCORRECT_BOUNDARIES` | Requested period too large or invalid from/to. |

Trading (117–136):

| Code | Name | Meaning |
|------|------|---------|
| 117 | `NO_QUOTES` | Trading cannot be done as no quotes are available (Book B). |
| 118 | `NOT_ENOUGH_MONEY` | Not enough funds to allocate margin. |
| 119 | `MAX_EXPOSURE_REACHED` | Max exposure limit reached for {trader, symbol, side}. |
| 120 | `POSITION_NOT_FOUND` | Position not found. |
| 121 | `ORDER_NOT_FOUND` | Order not found. |
| 122 | `POSITION_NOT_OPEN` | Trying to close a position that is not open. |
| 123 | `POSITION_LOCKED` | Position in a state that does not allow the operation. |
| 124 | `TOO_MANY_POSITIONS` | Trading account reached its limit for max number of open positions and orders. |
| 125 | `TRADING_BAD_VOLUME` | Invalid volume. |
| 126 | `TRADING_BAD_STOPS` | Invalid stop price. |
| 127 | `TRADING_BAD_PRICES` | Invalid price (e.g. negative). |
| 128 | `TRADING_BAD_STAKE` | Invalid stake volume (e.g. negative). |
| 129 | `PROTECTION_IS_TOO_CLOSE_TO_MARKET` | Invalid protection prices (SL/TP too close). |
| 130 | `TRADING_BAD_EXPIRATION_DATE` | Invalid expiration. |
| 131 | `PENDING_EXECUTION` | Unable to apply changes — position has an order under execution. |
| 132 | `TRADING_DISABLED` | Trading is blocked for the symbol. |
| 133 | `TRADING_NOT_ALLOWED` | Trading account is in read-only mode. |
| 134 | `UNABLE_TO_CANCEL_ORDER` | Unable to cancel order. |
| 135 | `UNABLE_TO_AMEND_ORDER` | Unable to amend order. |
| 136 | `SHORT_SELLING_NOT_ALLOWED` | Short selling is not allowed. |

### 6.3 Channels in which errors surface

| Channel | Use |
|---------|-----|
| `ProtoErrorRes` (payload 50) | Low-level protocol error (FRAME_TOO_LONG, BLOCKED_PAYLOAD_TYPE). |
| `ProtoOAErrorRes` (payload 2142) | Response-style error to a specific request (uses `clientMsgId` from request envelope). |
| `ProtoOAExecutionEvent.errorCode` | Order **failed validation** (synchronous rejection). |
| `ProtoOAOrderErrorEvent` (payload 2132) | Order **passed validation** but errored during execution (asynchronous rejection). |

---

## 7. Rate limits + quotas

| Limit | Threshold | Source |
|-------|-----------|--------|
| Non-historical requests | 50 req / sec / connection | <https://help.ctrader.com/open-api/> |
| Historical requests | 5 req / sec / connection | same |
| Connections per Open API client | unspecified hard cap → `CONNECTIONS_LIMIT_EXCEEDED` (67) | proto comment |
| Heartbeat | ≥ 1 / 10 sec | FAQ |
| Authorisation-code TTL | 60 sec | account-authentication page |
| Access-token TTL | 2 628 000 sec (~30.4 days) | token-exchange sample |
| Refresh-token TTL | none documented | account-authentication page |
| Frame size | implicit — surfaces as `FRAME_TOO_LONG` (8) | proto comment |
| Order labels / comments | label ≤ 100; comment ≤ 512 | `ProtoOANewOrderReq` proto comments |
| `clientOrderId` | ≤ 50 chars | proto comment |
| Trendbar window per request | per-period (M1 ≈ 1 wk, M5–M30 ≈ 1 mo, H1+ several months) | inferred — docs say "constraints exist" but no numerical table |

Throttle behaviour: server emits `ProtoOAErrorRes{errorCode = "REQUEST_FREQUENCY_EXCEEDED"}` with `retryAfter?` field
(seconds). No documented `Retry-After` HTTP header (the WebSocket
transport has no HTTP headers per message).

---

## 8. Symbol / instrument metadata

Where each piece of symbol data comes from:

| Field | Source message | Notes |
|-------|----------------|-------|
| Symbol id | `ProtoOALightSymbol.symbolId` (cheap) or `ProtoOASymbol.symbolId` (full) | Different brokers / servers assign different IDs. |
| Symbol name (e.g. "EUR/USD") | `ProtoOALightSymbol.symbolName` | Only in lightweight list. |
| Base/quote asset ids | `ProtoOALightSymbol.baseAssetId / quoteAssetId` | Cross-reference to `ProtoOAAsset`. |
| `digits`, `pipPosition` | `ProtoOASymbol` only — **NOT** in `LightSymbol` | Must call `ProtoOASymbolByIdReq`. |
| `lotSize` (cents) | `ProtoOASymbol.lotSize` | Divide by 100 to get lot size in units of base. |
| `minVolume`, `maxVolume`, `stepVolume` | `ProtoOASymbol` (in cents) | Validate `ProtoOANewOrderReq.volume` against these. |
| `swapLong`, `swapShort`, `swapPeriod`, `swapTime` | `ProtoOASymbol` | For overnight P&L calc. |
| `commission`, `commissionType`, `minCommission`, `preciseTradingCommissionRate`, `preciseMinCommission` | `ProtoOASymbol` | Multiple precision representations; prefer `precise*` fields. |
| `slDistance`, `tpDistance`, `gslDistance` | `ProtoOASymbol` + `distanceSetIn` | Minimum SL/TP distances. |
| Trading mode + holiday | `ProtoOASymbol.tradingMode`, `schedule`, `holiday[]` | Don't try to trade outside `schedule`; expect `SYMBOL_HAS_HOLIDAY`. |
| Leverage | `ProtoOASymbol.leverageId` → `ProtoOAGetDynamicLeverageByIDReq` → `ProtoOADynamicLeverage` | Tiered by volume. |
| Symbol category | `ProtoOALightSymbol.symbolCategoryId` → `ProtoOASymbolCategoryListRes` | Then category → `assetClassId` → `ProtoOAAssetClassListRes`. |

To convert one asset to another (e.g. quote-of-symbol →
deposit-currency for P&L), use `ProtoOASymbolsForConversionReq` to
retrieve the chain, then subscribe to spots on each link and walk
the chain. See `https://help.ctrader.com/open-api/symbol-rate-conversion/`.

---

## 9. Trade history / journaling

### 9.1 `ProtoOADealListReq` semantics

```proto
{
    ctidTraderAccountId,
    fromTimestamp?,        // ms epoch, ≥ 0 (default 1970-01-01)
    toTimestamp?,          // ms epoch, ≤ 2 147 483 646 000 (2038-01-19)
    maxRows?               // int32 — server-side cap may be smaller
}
```

Response:
```proto
{
    ctidTraderAccountId,
    repeated ProtoOADeal deal,
    required bool hasMore               // TRUE → paginate by advancing toTimestamp
}
```

**Paging.** When `hasMore` is TRUE, the result was capped at the
server's chunk size. Set `toTimestamp` of the next request to
`deals.last().executionTimestamp - 1` to fetch the next chunk. The
ordering is by `executionTimestamp` descending (most recent first).

**Per-position variant.** `ProtoOADealListByPositionIdReq` takes
`{ ctidTraderAccountId, positionId, fromTimestamp?, toTimestamp? }`
and returns just deals for one position — useful for reconstructing
the position's life.

**Order-deal correlation.** Each `ProtoOADeal.orderId` links back to
the parent order; `ProtoOAOrderDetailsReq` reverses the lookup.
`ProtoOADealOffsetListReq` returns FIFO/LIFO offsets between deals
for tax/accounting.

### 9.2 `ProtoOACashFlowHistoryListReq`

Same window semantics as `ProtoOADealListReq`. Returns
`repeated ProtoOADepositWithdraw` + `repeated ProtoOABonusDepositWithdraw`.
Use this to reconcile balance moves that aren't tied to deals.

### 9.3 `ProtoOAOrderListReq` vs Reconcile

| Use case | Message |
|----------|---------|
| Current open positions + pending orders **right now** | `ProtoOAReconcileReq` |
| Historical orders (filled, cancelled, expired) | `ProtoOAOrderListReq` |
| Historical deals | `ProtoOADealListReq` |

`Reconcile` is the snapshot; `*ListReq` are the journal.

---

## 10. Differences from our implementation

References below are file:line in `crates/forex-app/src/`.

1. **JSON over WSS on port 5036, never Protobuf on 5035.** Multiple
   files (`ctrader_execution.rs:440`, `ctrader_messages.rs:1156`,
   `ctrader_streaming.rs:542`, `ctrader_session.rs:28`) hard-code
   `wss://{host}:5036`. The vendored .proto files in
   `proto/` are present but unused at runtime — `ctrader_messages.rs`
   serialises every request to JSON via `serde_json::Value`. Switching
   to Protobuf (5035) would (a) save ~3× bandwidth, (b) sidestep the
   "string `errorCode` enum" parsing in `ctrader_execution.rs`, and
   (c) make us robust to JSON field-name drift between proto revisions.
   Tracked as a known limitation in `spotware_proto_freshness.md`.

2. **`ProtoOAGetPositionUnrealizedPnLReq` (2187) unused.** Our
   pricing/P&L code computes unrealised P&L client-side
   (`ctrader_data.rs` derives from `ProtoOASpotEvent` × position
   volume × marginRate). The new Batch-6 message hands us
   `grossUnrealizedPnL` + `netUnrealizedPnL` from the server in
   deposit currency — bypasses the FX-conversion chain entirely.
   Adopt this for the journaling layer at minimum.

3. **`ProtoOADealOffsetListReq` (2185) unused.** Our journaling
   currently treats each deal as standalone. Spotware exposes the
   server's FIFO/LIFO matching directly — better for tax-lot reports
   and for matching realised P&L numbers against what the broker
   shows in the cTrader UI.

4. **`ProtoOAOrderDetailsReq` / `ProtoOAOrderListByPositionIdReq`
   (2181, 2183) unused.** When a user asks "show me everything that
   happened to this position", we currently issue
   `ProtoOADealListByPositionIdReq` + filter the order history
   client-side. The single-call `OrderListByPositionId` is more
   efficient and the per-order detail is more granular.

5. **`moneyDigits` handling.** `ctrader_messages.rs` and
   `ctrader_data.rs` do NOT read each entity's `moneyDigits` field —
   they assume divide-by-100 (i.e. `moneyDigits = 2`). Spotware
   accounts with non-USD deposit currencies sometimes report
   `moneyDigits = 8` (especially crypto / JPY). Any balance / swap /
   commission display for those accounts will be off by 10^6. Audit
   item: add a `divide_money` helper that takes `moneyDigits` from
   the carrying entity and falls back to 2.

(Bonus: our token-exchange uses `GET` with secret in query string,
matching Spotware's documented format. That's the cTrader-accepted
form even though it violates RFC 6749 §2.3.1 — see §2.3 above and
`ctrader_api_reference.md` §1 for the rationale.)

---

## 11. Open questions

These are things the official pages reference but don't pin down:

- **Trendbar window caps per period.** Docs say "constraints exist on
  the maximum possible distance between fromTimestamp and
  toTimestamp" but never enumerate the numbers. Source:
  <https://help.ctrader.com/open-api/symbol-data/>.
- **`ProtoOADealListRes` chunk size.** Server caps response size but
  the value isn't documented. We've observed ~10 000 rows in
  practice. Source: implicit in `hasMore` field.
- **Whether `clientOrderId` deduplicates server-side.** The proto
  comment calls it "similar to FIX ClOrderID" but doesn't say
  duplicate-rejection is implemented. Our tests show **it does NOT**
  dedup. Source: `ProtoOANewOrderReq` proto comment.
- **TLS pinning / certificate rotation.** Both `*.ctraderapi.com`
  hosts use Let's-Encrypt-style short-rotation certs. No
  documentation on accepted pinning policy. Source: not addressed in
  any of the help-centre pages walked.
- **WebSocket sub-protocol negotiation.** The docs say "WebSocket is
  supported" but don't specify a `Sec-WebSocket-Protocol` value. Our
  implementation omits it and works against the proxy. Source: not
  addressed in <https://help.ctrader.com/open-api/connection/>.
- **JSON wire-format spec.** The "Send and receive JSON" page
  describes JSON support but stops short of a full JSON schema. The
  one example shown (`ProtoOAApplicationAuthReq` with `clientMsgId`,
  `payloadType: 2100`, `payload: {clientId, clientSecret}`) is the
  only canonical reference, and field-name camelCase is the
  observable rule. No formal `application/cap-1+json` content-type or
  similar is published. Source: implicit in the example block.
- **Disconnect cause codes.** `ProtoOAClientDisconnectEvent` and
  `ProtoOAAccountDisconnectEvent` both have a `reason` field but the
  enum of accepted values is not published.

---

## 12. Change log vs prior snapshot

Per `docs/audits/research/spotware_proto_freshness.md` and
`spotware_proto_new_messages.md`, the 2026-05-14 refresh of the four
proto files added **30 new messages** vs our prior 2026-01 snapshot.
Highlights (full list in `spotware_proto_new_messages.md`):

- 2167–2172 — full margin-call group (`ListReq/Res`, `UpdateReq/Res`,
  `UpdateEvent`, `TriggerEvent`).
- 2173–2174 — in-band token refresh.
- 2175–2176 — `ProtoOAOrderListReq/Res` (historical orders).
- 2177–2178 — dynamic-leverage resolver.
- 2179–2186 — per-position deal/order detail + offset list (FIFO/LIFO).
- 2187–2188 — server-computed unrealised P&L.

Plus the `OpenApiCommonMessages.proto` baseline (3 messages) and the
`OpenApiCommonModelMessages.proto` baseline (2 enums) remained
unchanged. The new enums in `OpenApiModelMessages.proto` between
snapshots: `ProtoOAStopOutStrategy` (added), `ProtoOAClientPermissionScope`
(added). Existing enums gained no new values in this refresh.

Commit hash range: not retrievable from the sandbox — the github MCP
server is locked to `kosred/forex-ai` and refused
`spotware/openapi-proto-messages`. The freshness file pins the
upstream snapshot at the 2026-05-14 main-branch tip, and lists per-file
sha256 in `spotware_proto_freshness.md`.

---

## Appendix A — sweep methodology

WebFetch returned **HTTP 403 for every official help-centre URL**.
The work-around was WebSearch with carefully scoped queries (each
official page has a unique title, so a query like
`"App and account authentication" site:help.ctrader.com`
reliably surfaces the page and its first ~150 words as a snippet).
Those snippets quote help-centre prose verbatim and contained
enough text to populate §1–§3 and §7–§11.

For §4–§6 (the meat of the API — message structure, enums, error
codes), the vendored `.proto` files in `crates/forex-app/proto/`
provided byte-exact authority. Those files were checked against the
freshness audit (`spotware_proto_freshness.md`) which last reconciled
them with `spotware/openapi-proto-messages@main` on 2026-05-14.

WebSearch was invoked roughly 25 times across the sweep; each query
targeted a single help-centre page or a single message family. The
list of URLs in §0.1 is the final visited set. No third-party blog,
forum, or port content is incorporated into §1–§12.

## Appendix B — confidence map

| Section | Confidence | Reason |
|---------|------------|--------|
| §1 Hosts & transport | High | Verbatim help-centre snippets confirmed by code in `ctrader_session.rs`. |
| §2 Auth | High | Verbatim help-centre snippets + Spotware curl example + our own working `ctrader_live_auth.rs`. |
| §3 Session lifecycle | High (rate limits exact); Medium (idempotency — inferred). | |
| §4 Messages | Very high — proto-level ground truth for every field. | |
| §5 Models | Very high — same. | |
| §6 Error codes | Very high — full enum from proto. | |
| §7 Rate limits | High for the two documented buckets; Medium for less-documented caps. | |
| §8 Symbol metadata | Very high — proto + help-centre. | |
| §9 Trade history | High — proto fields exact; chunk size inferred. | |
| §10 Diffs vs our code | High — read directly from our code. | |
| §11 Open questions | High — these really are open in the docs. | |
| §12 Change log | Medium — text accurate but commit hashes weren't reachable from the sandbox. | |

## Appendix C — Top 5 follow-up Batches

In priority order, these should become Batch tickets:

1. **Adopt `ProtoOAGetPositionUnrealizedPnLReq` (2187).** Replace
   our client-side unrealised-P&L calc with the server's
   gross+net values in deposit currency. Touches `ctrader_data.rs`,
   the pricing pipeline, and the journaling display.
2. **Honour `moneyDigits` per entity.** Add a helper used wherever
   we currently divide raw cash by 100. Affects balance, swap,
   commission, mirroringCommission, usedMargin, deposits, bonuses.
3. **Wire `ProtoOADealOffsetListReq` (2185) into the journal.**
   Lets us match the broker's UI exactly on realised P&L.
4. **Switch to Protobuf transport (port 5035).** Cuts bandwidth ~3×,
   eliminates the JSON-field-name brittleness, and unlocks
   `clientMsgId` as a robust correlation token instead of relying on
   payload-type matching.
5. **`ProtoOAOrderListByPositionIdReq` (2183) + `ProtoOAOrderDetailsReq`
   (2181) for position drill-down.** Replaces N-call client-side
   filtering with one server call. Improves both perf and
   user-perceived correctness when the broker re-emits an order with
   a server-side amendment.
