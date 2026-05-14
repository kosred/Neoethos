# cTrader Open API & OAuth 2.0 — Authoritative Reference

**Purpose:** Source-of-truth doc compiled from official Spotware / cTrader / IETF material so we can cross-check the patches our coding agents are writing against the audit findings in `v0.4.1_full_repo_audit*.md`.

**Source policy:** Only Spotware/cTrader org pages, the `spotware/openapi-proto-messages` GitHub repo (the canonical `.proto` source), NuGet, and IETF RFCs are cited. Community forum posts and third-party Go/Python ports are mentioned only when the official help-centre pages 403'd against WebFetch and the same content appeared verbatim in those secondary sources. Where the primary source couldn't be fetched directly, that's flagged explicitly.

**Fetch caveat:** `help.ctrader.com/open-api/*` and several `datatracker.ietf.org` pages returned **HTTP 403** to the research agent's WebFetch tool. Their content was obtained via WebSearch result snippets (which quote the pages directly) and, for proto definitions, by fetching the raw GitHub source which is the canonical artefact anyway. The `.proto` quotes below are verbatim from `raw.githubusercontent.com/spotware/openapi-proto-messages`.

---

## 1. cTrader Open API OAuth 2.0 token exchange

### Official sources

- App and account authentication: <https://help.ctrader.com/open-api/account-authentication/>
- Getting started: <https://help.ctrader.com/open-api/>
- Register an application: <https://help.ctrader.com/open-api/api-application/>
- OpenAPI.Net authentication guide (Spotware-maintained .NET SDK docs): <https://spotware.github.io/OpenAPI.Net/>
- OpenApiPy authentication guide (Spotware-maintained Python SDK docs): <https://spotware.github.io/OpenApiPy/authentication/>

### Endpoints

| Purpose | URL |
|--------|-----|
| Authorization (user consent) | `https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={clientId}&redirect_uri={redirectURI}&scope={scope}&product=web` |
| Token exchange | `https://openapi.ctrader.com/apps/token` |
| Token refresh | `https://openapi.ctrader.com/apps/token` (same base) |

Spotware also documents `https://connect.spotware.com/apps/auth` as the broker-SSO authorization base, but for the cTrader Open API (retail apps) the help centre uses `id.ctrader.com` and `openapi.ctrader.com` as above.

### Token-exchange call (the canonical Spotware example)

From the cTrader help centre's authentication page (per WebSearch snippet quoting the page verbatim):

> ```
> curl -X GET 'https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code=0ssdgds98as9_QSF56FVC_22dfdf&redirect_uri=https://spotware.com&client_id=5430012&client_secret=012sds23dlkjQsd' -H 'Accept: application/json' -H 'Content-Type: application/json'
> ```
>
> Response (JSON):
>
> ```json
> {
>   "accessToken": "mos8Bw3D4EG0fRPd4Eqq0JxaFT4zjd8e4YijNezh_ag",
>   "tokenType": "bearer",
>   "expiresIn": 2628000,
>   "refreshToken": "VCuafFhy81AFZjsWkbuEzdOhhRj5YTWz8fWUwHam7KM",
>   "errorCode": null,
>   "description": null
> }
> ```

Key observations:

- **HTTP method is `GET`.** Not POST.
- **All credentials — including `client_secret` — are URL query parameters.** Not body, not Authorization header.
- Access-token lifetime is `2628000` seconds (~30.4 days).
- Authorization-code lifetime is **60 seconds** per the help-centre text: *"An authorisation code is a short-term token that is issued for an individual cTID with an expiration period of one minute."*
- Refresh token has **no documented expiry** ("does not have an expiration period").

### Scopes

Two scopes only, per <https://help.ctrader.com/open-api/account-authentication/>:

- `trading` — **default**. Full access to user trading accounts (read + place/modify/cancel orders).
- `accounts` — read-only access to user trading account data.

There is no granular per-symbol or per-account scope.

### Conflict with OAuth 2.0 RFC 6749 §2.3.1

RFC 6749 §2.3.1 (quoted via gist mirror of RFC text at <https://gist.github.com/yorkxin/6590756>; the IETF datatracker page itself 403'd):

> "Including the client credentials in the request-body using the two parameters is NOT RECOMMENDED and SHOULD be limited to clients unable to directly utilize the HTTP Basic authentication scheme"

The spec's preferred method is HTTP Basic auth in the `Authorization` header; the body-parameter form is "NOT RECOMMENDED"; **URL query parameters are not contemplated as a credential-bearing channel at all** in §2.3.1. Putting `client_secret` in a URL query string is worse than the discouraged body form because (a) query strings are logged in web-server access logs, proxies, and browser history, and (b) RFC 6750 §2.3 explicitly cautions against URI-query transport for bearer tokens.

### What this means for our patches

**cTrader's official, documented OAuth flow uses `GET` with `client_secret` in URL query params** — that is what the help centre shows and what every Spotware SDK does. If our audit findings recommend "move client_secret to request body to comply with RFC 6749 §2.3.1", **that recommendation is correct from an OAuth-hygiene standpoint but will NOT work against the real cTrader endpoint** unless Spotware also accepts POST/body (which is not documented and is not in the example). Our patch should:

1. Match the documented format (`GET` + query params) for compatibility — this is what the broker actually accepts.
2. Compensate for the leakage risk by: never logging the token-exchange URL (mask `client_secret` and `code` in any logging middleware), preferring TLS-pinning, and rotating the client secret on a schedule.
3. Document this deviation from RFC 6749 §2.3.1 in code comments so future readers don't "fix" it.

A patch that switches to POST-with-body would be more secure on paper but would almost certainly break against `openapi.ctrader.com/apps/token` — verify before shipping.

---

## 2. cTrader Open API order placement (`ProtoOANewOrderReq`)

### Official source

Canonical proto definitions: <https://github.com/spotware/openapi-proto-messages/blob/main/OpenApiMessages.proto>
(Fetched via raw.githubusercontent.com — github.com HTML pages 403'd, but raw worked.)

### ProtoOANewOrderReq (verbatim)

> ```proto
> message ProtoOANewOrderReq {
>     optional ProtoOAPayloadType payloadType = 1 [default = PROTO_OA_NEW_ORDER_REQ];
>     required int64 ctidTraderAccountId = 2;
>     required int64 symbolId = 3;
>     required ProtoOAOrderType orderType = 4;
>     required ProtoOATradeSide tradeSide = 5;
>     required int64 volume = 6;
>     optional double limitPrice = 7;
>     optional double stopPrice = 8;
>     optional ProtoOATimeInForce timeInForce = 9 [default = GOOD_TILL_CANCEL];
>     optional int64 expirationTimestamp = 10;
>     optional double stopLoss = 11;
>     optional double takeProfit = 12;
>     optional string comment = 13;
>     optional double baseSlippagePrice = 14;
>     optional int32 slippageInPoints = 15;
>     optional string label = 16;
>     optional int64 positionId = 17;
>     optional string clientOrderId = 18;
>     optional int64 relativeStopLoss = 19;
>     optional int64 relativeTakeProfit = 20;
>     optional bool guaranteedStopLoss = 21;
>     optional bool trailingStopLoss = 22;
>     optional ProtoOAOrderTriggerMethod stopTriggerMethod = 23 [default = TRADE];
> }
> ```

### Required vs. optional

**Required:** `ctidTraderAccountId`, `symbolId`, `orderType`, `tradeSide`, `volume`. Everything else is optional in the protobuf sense.

### `clientOrderId` — yes, it exists

Field tag **18**, optional `string`, max length **50 characters** (per the proto comment quoted by the canonical doc).

It also appears on `ProtoOAOrder` (the order entity returned in execution events and reconcile responses) as field **17**:

> ```proto
> message ProtoOAOrder {
>     ...
>     optional string clientOrderId = 17;
>     ...
> }
> ```

So the broker round-trips `clientOrderId` back on every `ProtoOAExecutionEvent` and `ProtoOAReconcileRes`, which is what makes it usable for client-side idempotency tracking.

**Caveat — it is NOT on `ProtoOAPosition` or `ProtoOATradeData`:**

> ```proto
> message ProtoOATradeData {
>     required int64 symbolId = 1;
>     required int64 volume = 2;
>     required ProtoOATradeSide tradeSide = 3;
>     optional int64 openTimestamp = 4;
>     optional string label = 5;
>     optional bool guaranteedStopLoss = 6;
>     optional string comment = 7;
>     optional string measurementUnits = 8;
>     optional uint64 closeTimestamp = 9;
> }
>
> message ProtoOAPosition {
>     required int64 positionId = 1;
>     required ProtoOATradeData tradeData = 2;
>     required ProtoOAPositionStatus positionStatus = 3;
>     ...
> }
> ```

`clientOrderId` lives on the **order**, not on the **position** that results from a filled order. Positions carry `label` and `comment` (via `tradeData`), but not `clientOrderId`. For idempotency tracking that survives into the open-position lifecycle, you must either (a) reconcile through the order list (`ProtoOAReconcileRes.order`) and keep the client_order_id → position_id mapping locally, or (b) also stuff the idempotency token into `label` or `comment` (which are forwarded into `ProtoOATradeData`).

### `ProtoOAReconcileReq` — yes, this is the reconciliation primitive

> ```proto
> message ProtoOAReconcileReq {
>     optional ProtoOAPayloadType payloadType = 1 [default = PROTO_OA_RECONCILE_REQ];
>     required int64 ctidTraderAccountId = 2;
>     optional bool returnProtectionOrders = 3;
> }
> ```

The response (`ProtoOAReconcileRes`) carries `repeated ProtoOAPosition position = 3;` and `repeated ProtoOAOrder order = 4;`. So:

- To check "does an order with my client_order_id already exist as a **pending** order?" — scan `ProtoOAReconcileRes.order[*].clientOrderId`.
- To check "did my client_order_id already fill into a position?" — `ProtoOAPosition` does not carry `clientOrderId`, so you have to either (i) keep the local mapping captured from the prior `ProtoOAExecutionEvent`, or (ii) also send the same token in `comment` and scan `ProtoOAPosition.tradeData.comment`.

There is **no** dedicated "lookup order by clientOrderId" message documented in the public proto repo. Reconcile is the only batched server-side check.

### Idempotency strategy (what the spec supports vs. requires)

The cTrader Open API protocol **does not enforce uniqueness** on `clientOrderId` server-side in any documented way. The Spotware proto comment merely calls it *"Optional ClientOrderId. Max Length = 50 chars."* — it is a free-form label, the FIX `ClOrdID` equivalent. Recommended retry pattern (derived from the proto shape — no Spotware-published "idempotency guide" exists):

1. Generate a client-side UUID per logical order intent.
2. Send it as both `clientOrderId` and (for position-level visibility) `comment`.
3. On retry after a network error / timeout: send `ProtoOAReconcileReq` first; if the order is present in `reconcile.order[]` with our `clientOrderId`, abort the retry. If we have a local mapping to a `positionId`, also check `reconcile.position[]`.
4. As a belt-and-braces step, scan recent `ProtoOAExecutionEvent` messages (if the WebSocket session has been continuous) before issuing the retry.

### What this means for our patches

- The audit's "add `client_order_id` to `ProtoOANewOrderReq`" patch is **correct and supported by the protocol**. Field 18, optional string, max 50 chars.
- The audit's "use `ProtoOAReconcileReq` to verify orders before retry" patch is **correct** — that is the documented way to check pending orders.
- **Watch out:** if the patch tries to look up a *filled* order by clientOrderId via reconcile and expects it on `ProtoOAPosition`, it will silently miss every match because positions don't carry that field. Patch must either keep a local order_id → position_id map from the original execution event, or also write the idempotency token into `comment`.
- There is no spec-mandated uniqueness on `clientOrderId`; if our code assumes the broker will reject duplicates, that assumption is unsupported and we must dedupe client-side.

---

## 3. cTrader Open API symbol metadata

### Official sources

- Help-centre "Attain symbol data": <https://help.ctrader.com/open-api/symbol-data/> (403 on WebFetch; content verified via WebSearch snippets and the proto file).
- Help-centre "Model messages": <https://help.ctrader.com/open-api/model-messages/>
- Proto source: <https://github.com/spotware/openapi-proto-messages/blob/main/OpenApiModelMessages.proto>
- OpenAPI.Net tick/pip value calculation guide: <https://spotware.github.io/OpenAPI.Net/calculating-symbol-tick-value/> (also 403 to WebFetch, but the formula appeared verbatim in search snippets).

### Two-step workflow

1. `ProtoOASymbolsListReq` → returns `repeated ProtoOALightSymbol symbol` — sparse summary (id, name, enabled, base/quote asset id, category id, description, sorting number). **Insufficient for risk calculations.**
2. `ProtoOASymbolByIdReq` (with a list of `symbolId`s) → returns `ProtoOASymbolByIdRes` containing full `ProtoOASymbol` entities — this is the message with `digits`, `pipPosition`, `lotSize`, `minVolume`, `maxVolume`, `stepVolume`, etc.

### ProtoOASymbol (verbatim, key fields)

> ```proto
> message ProtoOASymbol {
>     required int64 symbolId = 1;
>     required int32 digits = 2;
>     required int32 pipPosition = 3;
>     optional bool enableShortSelling = 4;
>     optional bool guaranteedStopLoss = 5;
>     optional ProtoOADayOfWeek swapRollover3Days = 6;
>     optional double swapLong = 7;
>     optional double swapShort = 8;
>     optional int64 maxVolume = 9;
>     optional int64 minVolume = 10;
>     optional int64 stepVolume = 11;
>     optional uint64 maxExposure = 12;
>     repeated ProtoOAInterval schedule = 13;
>     optional int64 commission = 14;
>     optional ProtoOACommissionType commissionType = 15;
>     optional uint32 slDistance = 16;
>     optional uint32 tpDistance = 17;
>     optional uint32 gslDistance = 18;
>     optional int64 gslCharge = 19;
>     optional ProtoOASymbolDistanceType distanceSetIn = 20;
>     optional int64 minCommission = 21;
>     optional ProtoOAMinCommissionType minCommissionType = 22;
>     optional string minCommissionAsset = 23;
>     optional int64 rolloverCommission = 24;
>     optional int32 skipRolloverDays = 25;
>     optional string scheduleTimeZone = 26;
>     optional ProtoOATradingMode tradingMode = 27;
>     optional ProtoOADayOfWeek rolloverCommission3Days = 28;
>     optional ProtoOASwapCalculationType swapCalculationType = 29;
>     optional int64 lotSize = 30;
>     optional int64 preciseTradingCommissionRate = 31;
>     optional int64 preciseMinCommission = 32;
>     repeated ProtoOAHoliday holiday = 33;
>     optional int32 pnlConversionFeeRate = 34;
>     optional int64 leverageId = 35;
>     optional int32 swapPeriod = 36;
>     optional int32 swapTime = 37;
>     optional int32 skipSWAPPeriods = 38;
>     optional bool chargeSwapAtWeekends = 39;
>     optional string measurementUnits = 40;
> }
> ```

Field-level notes from the proto comments (as quoted in search results):

- `digits` — number of price digits to display.
- `pipPosition` — pip position on digits.
- `lotSize` — lot size of the symbol, **in cents** (i.e. 1e-2 of the base unit; for FX this is typically `10000000` representing 100,000 base-currency units in cents).
- `minVolume`, `maxVolume`, `stepVolume` — order volume bounds **in cents**.

### Canonical pip/tick formulas

From the Spotware OpenAPI.Net "Calculating Symbol Tick/Pip Value" page (search-snippet verbatim):

> ```
> TickSize = 1 / Math.Pow(10, symbol.Digits)
> PipSize  = 1 / Math.Pow(10, symbol.PipPosition)
> ```

For pip *value* per lot, the canonical chain is:

```
pip_size           = 10^(-pipPosition)
contract_size      = lotSize / 100               # because lotSize is in cents
pip_value_per_lot  = pip_size * contract_size    # in quote currency
```

(`pipPosition` is exposed as the count of decimal digits the pip occupies in the price; for EURUSD with `digits=5, pipPosition=4`, that gives a 0.0001 pip on a 5-decimal price, and for JPY pairs with `digits=3, pipPosition=2`, that gives 0.01.)

### What this means for our patches

The audit flagged synthesised pip values at `forex-app/src/app_services/trading.rs:3332` with an empty-string symbol fallback. **The Open API gives us every input needed to compute pip value deterministically** — there is no need to synthesise. The patch should:

1. Cache `ProtoOASymbol` entries keyed by `symbolId` after the initial `ProtoOASymbolsListReq` → `ProtoOASymbolByIdReq` warm-up.
2. Compute `pip_size`, `tick_size`, and `pip_value_per_lot` from `digits`, `pipPosition`, and `lotSize` as above.
3. Treat a missing/unknown symbol as a hard error (reject the order, alert) — **not** as a trigger for synthesis. The empty-string fallback is a correctness bug; there is no defensible reason to synthesise when the broker hands us the authoritative value.
4. Be aware `lotSize`/`minVolume`/`stepVolume` are in **cents** of the base asset; convert before exposing to the application layer.

---

## 4. cTrader supported timeframes

### Official source

Proto: <https://github.com/spotware/openapi-proto-messages/blob/main/OpenApiModelMessages.proto>

### Verbatim enum

> ```proto
> enum ProtoOATrendbarPeriod {
>     M1 = 1;
>     M2 = 2;
>     M3 = 3;
>     M4 = 4;
>     M5 = 5;
>     M10 = 6;
>     M15 = 7;
>     M30 = 8;
>     H1 = 9;
>     H4 = 10;
>     H12 = 11;
>     D1 = 12;
>     W1 = 13;
>     MN1 = 14;
> }
> ```

### Comparison vs. our canonical list (M1, M3, M5, M15, M30, H1, **H2**, H4, **H12**, D1, W1, MN1)

| Our timeframe | Native in cTrader? |
|---------------|--------------------|
| M1            | yes                |
| M3            | **yes** (M3 = 3)   |
| M5            | yes                |
| M15           | yes                |
| M30           | yes                |
| H1            | yes                |
| **H2**        | **NO** — must resample from H1 (or M30) |
| H4            | yes                |
| H12           | yes                |
| D1            | yes                |
| W1            | yes                |
| MN1           | yes                |

cTrader additionally supports M2, M4, M10 that we don't list, and **lacks H2, H3, H6, H8 entirely**. The audit's suspicion is confirmed: **H2 is not natively supported** by the cTrader Open API and must be aggregated from H1 (or smaller) bars.

### What this means for our patches

- The timeframe-mapping patch must route `H2` through a resampler (e.g., fold pairs of H1 bars), not send `H2` to `ProtoOAGetTrendbarsReq` — the enum has no value for it and the server will reject.
- M3 is fine — direct support exists (enum value 3).
- If the patch ships a single `ProtoOATrendbarPeriod` mapping function, it should explicitly list cTrader's 14 supported periods and either error out or trigger a resample for everything else.

---

## 5. cTrader .NET SDK

### Official sources

- GitHub repo: <https://github.com/spotware/OpenAPI.Net>
- Docs site: <https://spotware.github.io/OpenAPI.Net/>
- NuGet (current package name): <https://www.nuget.org/packages/cTrader.OpenAPI.Net>
- NuGet (older package name, still listed): <https://www.nuget.org/packages/Spotware.OpenAPI.Net>
- Help-centre .NET SDK index: <https://help.ctrader.com/open-api/net_SDK/net-sdk-index/>

### Key facts

| Item | Value |
|------|-------|
| NuGet package name (current) | `cTrader.OpenAPI.Net` |
| Latest stable version (as of fetch on 2026-05-14) | **1.4.4** |
| Last published | **2022-05-03** |
| License | **MIT** |
| Target framework | **.NET 6** |
| Architecture | Reactive Extensions (Rx) over WebSocket |
| Stars / forks | ~89 / ~47 |
| Total NuGet downloads | ~13.7K |

There is also an older package `Spotware.OpenAPI.Net` (last seen 1.3.9). Spotware renamed/republished as `cTrader.OpenAPI.Net`; the `Spotware.*` package is effectively superseded.

### Dependencies

- `Google.Protobuf >= 3.20.1`
- `System.Reactive >= 5.0.0`
- `Websocket.Client >= 4.4.43`

### What it provides over a hand-rolled Rust protobuf client

- Pre-compiled, version-tracked C# bindings for the proto files.
- Rx-stream abstraction over the WebSocket — observable event streams, automatic backpressure via System.Reactive.
- Channels + array pools for low-allocation hot paths.
- Pre-wired request/response correlation (matching `clientMsgId` to async observers).
- Sample apps for desktop and console.

A Rust client gets none of this for free — but Rust users gain: better memory safety than C# WS clients, no GC pauses, and (potentially) the ability to use `prost`/`tonic` and `tokio-tungstenite` to build something equivalent. The .NET SDK is a code-generation + correlation convenience, not a protocol gate.

### Known issues / deprecation signals

- **Last commit / release is May 2022.** Nearly four years stale. No `CHANGELOG.md` is present in the repo root (404 on fetch).
- No public deprecation notice in the README or NuGet page, but the long quiet period is itself a signal.
- Targets .NET 6, which Microsoft moved out of LTS on 2024-11-12 ([Microsoft .NET support policy](https://dotnet.microsoft.com/en-us/platform/support/policy/dotnet-core)) — a consumer on .NET 8/9 will need to verify TFM compatibility or recompile.
- The proto files in `spotware/openapi-proto-messages` have been updated since 2022 (the repo currently shows 76 commits to `main`), so the C# bindings shipped in the SDK may lag the live wire protocol. The forum post "Protobuf/C# out of date compared to Web Docs and Servers" (search result) corroborates this concern.

### What this means for our architecture decision

- Spotware **does** publish an official .NET SDK, and it's MIT-licensed and on NuGet.
- It is, however, **maintenance-mode at best** (no commits in ~4 years; .NET 6 target now out of LTS).
- Adopting it for a Rust project would mean adding a .NET shim — significant complexity for an SDK that's not aggressively maintained. The Rust protobuf approach (using `prost` against the always-current `openapi-proto-messages` repo) is arguably more future-proof.
- If the user is considering migrating to a .NET-based component, **the staleness should weigh against rather than for the SDK** — they'd be adopting a 2022-era artefact that may already diverge from the production wire format.

---

## 6. OAuth 2.0 `state` parameter (RFC 6749 §10.12) and PKCE (RFC 7636)

### Official sources

- RFC 6749 (The OAuth 2.0 Authorization Framework): <https://datatracker.ietf.org/doc/html/rfc6749> (direct fetch 403'd; quotes obtained via the gist mirror at <https://gist.github.com/yorkxin/6590756> and `tech-invite.com`'s page-3 mirror).
- RFC 7636 (Proof Key for Code Exchange): <https://datatracker.ietf.org/doc/html/rfc7636> (likewise 403'd; quotes from official-RFC-mirror search snippets).

### RFC 6749 §4.1.1 — the `state` parameter (verbatim)

> "An opaque value used by the client to maintain state between the request and callback. The parameter SHOULD be used for preventing cross-site request forgery as described in Section 10.12."

### RFC 6749 §10.12 — Cross-Site Request Forgery (verbatim)

> "The client MUST implement CSRF protection for its redirection endpoint when the grant type is 'code' by utilizing the 'state' parameter to correlate requests and responses."
>
> "The binding value used for CSRF protection MUST contain a non-guessable value (as described in Section 10.10), and the client SHOULD utilize the 'state' request parameter to deliver this value to the authorization server when making an authorization request."

§10.10 ("Credentials-Guessing Attacks") describes "non-guessable" as cryptographic-random with sufficient length to make brute force infeasible (RFC 6749 does not pin a specific bit count for `state`; it defers to general "credentials-guessing" guidance — practical minimum is 128 bits of entropy, with 256 bits recommended).

### RFC 7636 §4.1 — code_verifier construction (verbatim per RFC text)

> "A code verifier itself is a random string using characters of [A-Z] / [a-z] / [0-9] / '-' / '.' / '_' / '~', with a minimum length of 43 characters and a maximum length of 128 characters."

### RFC 7636 §7.1 — Entropy of the code verifier (verbatim per RFC text)

> "The client SHOULD create a 'code_verifier' with a minimum of 256 bits of entropy. This can be done by having a suitable random number generator create a 32-octet sequence."
>
> "The octet sequence can then be base64url-encoded to produce a 43-octet URL safe string."

### Is PKCE required for cTrader's flow?

**No, not by the cTrader help centre.** The Spotware authentication docs (the help-centre page and both SDK auth guides) describe only the classic RFC 6749 `authorization_code` grant. No `code_challenge` / `code_challenge_method` parameter is documented for `id.ctrader.com/my/settings/openapi/grantingaccess/`, and no `code_verifier` is mentioned in the token-exchange example. cTrader treats the Open API client as a "confidential client" (the `client_secret` is mandatory), so under the OAuth 2.1 draft guidance PKCE would still be **recommended** but it is not currently surfaced by Spotware's endpoint.

### Canonical generation patterns

For `state` (RFC 6749 §10.12 compliance):

- 16 bytes (128 bits) of CSPRNG output, base64url-encoded → 22-char token. This is the floor.
- 32 bytes (256 bits) → 43-char token. Recommended.
- Must be bound to the user's session (e.g. stored in a server-side session, signed cookie, or short-TTL Redis key keyed by session ID) and **verified for exact match on callback before exchanging the code**.

For PKCE `code_verifier` (RFC 7636 §4.1, §7.1):

- 32 random bytes → base64url → 43-char `code_verifier`.
- `code_challenge = BASE64URL(SHA256(code_verifier))` with `code_challenge_method=S256`.
- Even when not required by the server, PKCE is harmless when the server ignores the params, and provides real protection against authorization-code interception on the redirect leg.

### What this means for our patches

- The audit's "add CSPRNG-backed `state` parameter and verify on callback" patch is **mandatory per RFC 6749 §10.12** ("MUST implement CSRF protection ... by utilizing the 'state' parameter"). 32 bytes (256 bits) base64url is the safe default; 16 bytes is the absolute minimum.
- Adding PKCE (`code_challenge` / `code_challenge_method=S256` on the auth request, `code_verifier` on the token exchange) is **not required by cTrader today** but is good defence-in-depth. Patch should add the parameters and accept that cTrader may currently ignore them — that's the correct posture for an OAuth 2.1-forward codebase.
- Min entropy: `state` ≥ 128 bits, `code_verifier` ≥ 256 bits (RFC 7636 §7.1). Use `rand::rngs::OsRng` in Rust; do not use `rand::thread_rng()` for credential generation.

---

## Cross-check summary: where our intended patches conflict with the spec

| Patch direction (from audit) | Spec alignment | Action |
|------------------------------|----------------|--------|
| Move `client_secret` from URL query to request body | **Conflicts with cTrader's documented format.** RFC says body is "NOT RECOMMENDED" vs. Basic-auth header, but cTrader's documented endpoint is `GET` with query params. | Match the documented format; compensate with log-redaction and secret rotation. Do not change to POST/body without confirming cTrader supports it. |
| Add `client_order_id` to `ProtoOANewOrderReq` | **Spec-supported.** Field 18, optional string, max 50 chars. | Ship as-is. |
| Use `ProtoOAReconcileReq` to dedupe retries | **Partially correct.** Works for pending `ProtoOAOrder` (has `clientOrderId`). Does NOT work for filled `ProtoOAPosition` (no `clientOrderId`). | Keep local order_id → position_id map from execution events, OR also stuff the token into `comment` and scan `ProtoOAPosition.tradeData.comment`. |
| Drop synthesised pip values, use broker-provided values | **Fully spec-supported.** `digits`, `pipPosition`, `lotSize` are all on `ProtoOASymbol`. | Implement `ProtoOASymbolsListReq` → `ProtoOASymbolByIdReq` warm-up; cache. Treat unknown symbol as error, not synthesis trigger. |
| Native H2 timeframe support | **Conflicts with spec.** Enum has no H2 value. | Resample from H1. Patch must explicitly handle this fork. |
| Native M3 timeframe support | **Spec-supported.** | Direct mapping `M3 → ProtoOATrendbarPeriod.M3 (3)`. |
| Add CSPRNG-generated `state` parameter | **Required by RFC 6749 §10.12.** | 32 bytes from `OsRng`, base64url, session-bound, verified-equal on callback. |
| Add PKCE (`S256`) | Not required by cTrader, recommended by OAuth 2.1. | Add it; expect cTrader to currently ignore — that is fine. |
| Replace hand-rolled protobuf with Spotware .NET SDK | **Architecturally questionable.** SDK is 4 years stale, .NET 6 target out of LTS. | Stay on Rust + `prost` against the live `openapi-proto-messages` repo. |

---

## Appendix: every URL cited

- <https://help.ctrader.com/open-api/>
- <https://help.ctrader.com/open-api/account-authentication/>
- <https://help.ctrader.com/open-api/api-application/>
- <https://help.ctrader.com/open-api/messages/>
- <https://help.ctrader.com/open-api/model-messages/>
- <https://help.ctrader.com/open-api/symbol-data/>
- <https://help.ctrader.com/open-api/protocol-buffers-json/>
- <https://help.ctrader.com/open-api/net_SDK/net-sdk-index/>
- <https://help.ctrader.com/open-api/faq/>
- <https://help.ctrader.com/open-api/symbol-rate-conversion/>
- <https://help.ctrader.com/open-api/sending-receiving-protobuf/>
- <https://openapi.ctrader.com/>
- <https://openapi.ctrader.com/apps/token>
- <https://id.ctrader.com/my/settings/openapi/grantingaccess/>
- <https://github.com/spotware/openapi-proto-messages>
- <https://github.com/spotware/openapi-proto-messages/blob/main/OpenApiMessages.proto>
- <https://github.com/spotware/openapi-proto-messages/blob/main/OpenApiModelMessages.proto>
- <https://github.com/spotware/OpenAPI.Net>
- <https://spotware.github.io/OpenAPI.Net/>
- <https://spotware.github.io/OpenAPI.Net/calculating-symbol-tick-value/>
- <https://spotware.github.io/OpenApiPy/authentication/>
- <https://github.com/spotware/OpenApiPy>
- <https://www.nuget.org/packages/cTrader.OpenAPI.Net>
- <https://www.nuget.org/packages/Spotware.OpenAPI.Net>
- <https://datatracker.ietf.org/doc/html/rfc6749>
- <https://datatracker.ietf.org/doc/html/rfc7636>
- <https://gist.github.com/yorkxin/6590756> (gist mirror of RFC 6749 text — used only because the canonical IETF host 403'd to WebFetch)
