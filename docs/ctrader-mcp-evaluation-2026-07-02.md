# cTrader MCP — Evaluation (2026-07-02)

**Task**: assess Spotware's official MCP servers as a future transport for the
NeoEthos Supervisor and for external AI agents (Claude Code, Codex, …).

## What exists (verified 2026-07-02)

Spotware ships **cTrader AI Agent Connect** — https://mcp.spotware.com/ — two
official MCP servers plus a skills library:

| Server | Connects to | Capabilities |
|---|---|---|
| **Remote MCP** | cTrader **Web** | trading, account analysis, market data workflows |
| **Local MCP** | cTrader **Windows/Mac desktop** | charts, indicators, plugins, UI layout, price alerts |

Compatible clients include **Claude Code**, Codex, Cursor, Windsurf, Gemini
CLI. Announcement: https://www.spotware.com/news/ctrader-launches-official-mcp-servers/

## How it relates to NeoEthos

NeoEthos already talks to cTrader through the **Open API** (OAuth + WebSocket)
— orders, positions, bars, ticks, account. The MCP servers do NOT replace that
path (the engine needs deterministic, low-latency, programmatic access). Where
MCP adds value:

1. **Supervisor phase 2 — capabilities our Open API path lacks**: desktop
   chart control, indicator/plugin management, price alerts, and any future
   surface Spotware exposes MCP-first. The Supervisor's action whitelist could
   grow `mcp_call` actions routed through an MCP client.
2. **External agents**: a user can point Claude Code (or any MCP client) at
   BOTH the NeoEthos API and cTrader's MCP — e.g. "compare what NeoEthos's
   journal says with the cTrader statement" without custom glue.
3. **Skills library**: Spotware's prebuilt AI workflows may cover operations
   (statement export, alert setup) we'd otherwise implement by hand.

## Integration sketch (when we pick this up)

- Add an MCP **client** to the supervisor. **SDK RESOLVED (2026-07-03,
  operator-supplied)**: the official Rust SDK is mature —
  https://github.com/modelcontextprotocol/rust-sdk, crates `rmcp` +
  `rmcp-macros`, v2.1.0 (Jul 2026), client+server, tokio, stdio/child-process/
  streamable-HTTP transports, production adopters (Goose/Block, Apollo,
  containerd). Condition (b) of the verdict below is now SATISFIED.
- **Isolation discipline** (per docs/p2p-mesh-design-2026-07-03.md §1): if
  the rmcp dependency tree conflicts with the app's pinned stack, ship the
  MCP client as a sidecar speaking to the app's HTTP API instead of linking
  it into `neoethos-app` — same pattern as `neoethos-mesh`.
- New whitelisted action `{"action":"mcp_call","tool":...,"args":...}` gated
  the same way as everything else (T2 reversible / T3 approval).
- Credential note: the remote MCP authenticates against cTrader Web — separate
  consent surface from our OAuth token; never share tokens across paths.

## Verdict

**Adopt later, deliberately.** No immediate engine value (Open API covers
trading), clear future value for supervisor/desktop workflows and third-party
agent interop. Revisit when: (a) the servers exit early-access and document
stable tool schemas, ~~(b) a mature Rust MCP client exists~~ **(b) SATISFIED
2026-07-03 — official `rmcp` v2.1.0, see Integration sketch**, (c) a concrete
workflow needs a capability only MCP exposes. Two of three remain.
