# NeoEthos MCP sidecar

Bridges the app's **Codex / Supervisor** to **Model Context Protocol** tools —
cTrader's official MCP, a filesystem server, web search, or any MCP server —
using the official Rust SDK (`rmcp`).

## Why a separate program

Same doctrine as the P2P mesh: `rmcp` brings its own dependency tree, and the
trading engine sits on a delicately pinned stack. So this is an **isolated
binary** with its **own workspace + `Cargo.lock`**, excluded from the main
workspace. It talks to the app only over localhost HTTP; a bug here crashes
only this process — the engine never links `rmcp`.

Built on **rmcp 2.1** (edition 2024).

## Run

```bash
cp mcp_servers.example.json mcp_servers.json   # then edit it
cargo run --release                            # or ./target/release/neoethos-mcp --config PATH
```

It connects to every configured MCP server on startup and serves a tiny local
HTTP API:

| Endpoint | Purpose |
|---|---|
| `GET /health` | which servers connected |
| `GET /tools` | every tool across all servers (`{server, name, description}`) |
| `POST /call` | invoke `{ "server": "...", "tool": "...", "args": {...} }` |

## How the app uses it

The Supervisor gains an `mcp_tools` action (list what's available) and an
`mcp_call` action (invoke a tool) that POST to this sidecar — so the same
whitelisted, guard-railed action path governs MCP calls as everything else.
A cTrader-MCP call, a filesystem read, or a web search all flow through here.

## Transports

- **`http`** — remote servers by URL (e.g. cTrader's `mcp.spotware.com`).
  Streamable-HTTP client.
- **`stdio`** — local servers spawned as a child process (e.g. the official
  `@modelcontextprotocol/server-filesystem` via `npx`).

## Security

- The cTrader MCP authenticates against cTrader Web — a **separate** consent
  surface from the app's Open API OAuth token. Never share tokens across paths.
- Filesystem servers are scoped to the directory you configure — scope tightly.
- This sidecar binds to `127.0.0.1` only; it is not exposed to the network.

## License

AGPL-3.0-or-later. `rmcp` and its dependencies keep their own licenses.
