# `ProtoOAOrderListByPositionIdRes` fixtures

> **STATUS 2026-08-09 (dead-code purge, batch D2).** This directory is
> currently **EMPTY of fixtures** — none was ever captured. The tests that
> would have consumed them lived in
> `crates/neoethos-app/src/app_services/ctrader_history.rs`, deleted in D2
> because every public item in that file had zero callers. What survives is
> the parser, `ctrader_account::parse_order_list_by_position_id_response`,
> and its request builder, `ctrader_messages::build_order_list_by_position_id_request`
> — both now unreferenced protocol surface kept as a WIRE item for the
> 2026-05-15 "full cTrader API coverage" directive. This file is retained
> because it is the only surviving description of the wire shape.

This directory is *intended* to hold **real, captured**
`ProtoOAOrderListByPositionIdRes` JSON envelopes.
Synthetic broker payloads are forbidden here per the 2026-05-15
operator directive (no-silent-fallback / no-synthetic-data).

## Schema reference

Source of truth: the upstream Spotware proto
(`https://github.com/spotware/openapi-proto-messages` —
message `ProtoOAOrderListByPositionIdRes`, payload type **2184**, and the
per-order `ProtoOAOrder` body, same shape as the `order` array inside
`ProtoOAReconcileRes`). The local `crates/neoethos-app/proto/` copy was
deleted in D2 along with the protoc codegen that nothing consumed, and
`docs/audits/research/ctrader_api_full_reference.md` does not exist in this
repo.

```json
{
  "clientMsgId": "<operator-supplied or echoed from the request>",
  "payloadType": 2184,
  "payload": {
    "ctidTraderAccountId": <int64 — required>,
    "order": [
      {
        "orderId": <int64 — required>,
        "tradeData": {
          "symbolId": <int64 — required>,
          "volume": <int64 — required; cTrader protocol units (1/100 lot)>,
          "tradeSide": "BUY" | "SELL",
          "openTimestamp": <int64 — unix ms>,
          "label": <string — optional>,
          "comment": <string — optional>
        },
        "orderType": "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT" | ...,
        "orderStatus": "ORDER_STATUS_FILLED" | "ORDER_STATUS_CANCELLED" | ...,
        "limitPrice": <double — optional>,
        "stopPrice": <double — optional>,
        "stopLoss": <double — optional>,
        "takeProfit": <double — optional>,
        "clientOrderId": <string — optional>
      },
      ...
    ],
    "hasMore": <bool — optional; if true the page was capped>
  }
}
```

The request side accepts `fromTimestamp` / `toTimestamp` in unix ms
(both optional) and the response is filtered locally to the same
window. That local clamp lived in
`ctrader_history::fetch_orders_by_position_id_with_transport`
(`filter_orders_to_window`) and was deleted with the file — a future wiring
must re-implement it. A captured fixture should still cover at least 2 orders
straddling the requested window so the clamping codepath gets exercised.

## Expected fixture file names

The follow-up test suite is `#[ignore = "needs cTrader fixture"]`
until the operator captures and lands the following files (commit
them under this directory):

| File | Test that consumes it | Captured scenario |
|------|------------------------|-------------------|
| `position_full_chain.json` | `fetch_orders_by_position_id_clamps_to_position_real_fixture` | A position that has accumulated at least 3 historical orders (e.g. one filling order + one TP amend + one SL amend, or one filling order + one partial close + one stop-out). The returned `order` array should contain ONLY orders whose `tradeData.positionId` matches the requested position — the test asserts this invariant against the broker's own response (no client-side filter). |

The file is the **raw JSON envelope** the cTrader Open API sends on
the WebSocket (port `5036`) — i.e. the same string our
`ProductionCTraderOpenApiTransport` reads off the socket and passes
to `parse_order_list_by_position_id_response`. Keep `clientMsgId`
as returned by the broker (do not hand-edit). The host
(`live.ctraderapi.com` or `demo.ctraderapi.com`) and the
originating account id should be recorded in the commit message
but **not** in the file body — re-using the broker's verbatim
bytes is the entire point.

## Capture procedure

1. Start the bot against a demo (or live) cTrader account with at
   least one position that has accumulated multiple amends or
   partial closes. The demo host is `demo.ctraderapi.com:5036`;
   auth bundle as usual via the cTrader OAuth flow.
2. Wrap the `ProductionCTraderOpenApiTransport::send_sequence`
   call with a one-line `tracing::debug!(target:
   "neoethos_app::ctrader_capture", raw = %text, "captured
   order_list_by_position response")` immediately after the
   `responses.push(text.to_string())` for payload type 2184. The
   captured `text` is exactly the bytes this README references —
   no re-serialization, no field reordering.
3. Trigger a `ProtoOAOrderListByPositionIdReq` for the target
   position by either:
   - Letting the post-execution drill-down fire after a fresh
     order fills (see `trading/orders.rs::execute_ctrader_request`
     — the request goes out on the success path after
     `refresh_ctrader_runtime_after_execution`), or
   - Cancelling / closing the position so the pre-cancel /
     pre-close drill-down fires (see `trading/orders.rs::
     cancel_selected_order` / `close_selected_position`).
4. Redirect the trace output to a file:
   `RUST_LOG=neoethos_app::ctrader_capture=debug cargo run ...`.
5. Extract the matching log line, save the JSON body to
   `position_full_chain.json` in this directory, and remove the
   temporary capture tracing line.
6. Commit the file with a short note in the commit body
   identifying the broker host, the originating account id
   (range only — e.g. "demo account 100…1234"), the position id,
   and the capture date.

## Why this matters (cross-reference)

A `fetch_orders_by_position_id` helper would replace the client-side N-call
filter pattern below. (The 2026-05 implementation was deleted in D2 as dead
code; only the builder and parser remain.) The operator-facing flow today is
still:

```
ProtoOAOrderListReq  (whole account)
  → parse N orders
  → iterate, filter by tradeData.positionId == target
```

After this batch:

```
ProtoOAOrderListByPositionIdReq  (one position)
  → parse exactly the orders the broker thinks are linked
```

The second form is one network round trip instead of one + N
client-side iterations, and — more importantly — exposes any
broker-side amendments (e.g. a server-side stop-out that re-emits
the parent order with a fresh `orderStatus`) that the client-side
filter could otherwise miss because the local snapshot was stale.

The captured fixture exists so the test can prove that the helper
preserves the broker's filtering decision verbatim (no client-side
re-filter, no silent drop of orders whose `tradeData.positionId`
field is structured differently than expected).
