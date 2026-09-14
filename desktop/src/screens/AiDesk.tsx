import { useEffect, useState } from "react";
import {
  codexStatus, codexStart, codexLogout, codexChat, supervisorChat,
  mcpStatus, mcpConfigGet, mcpConfigSave,
} from "../api";
import { usePoll } from "../hooks";
import Supervisor from "./Supervisor";

type Turn = { role: "you" | "ai"; text: string; mode: ChatMode };

// The ONE place to talk to the LLM (operator request 2026-07-11 — there
// used to be two chat boxes: this one and another on the Supervisor
// screen). The mode toggle routes each message:
//  - Assistant  → plain ChatGPT chat (market questions, no system access)
//  - Supervisor → the tool-aware supervisor (reads full system state,
//    standing directives, and can ACT through the same whitelisted,
//    guard-railed actions as its autonomous loop)
type ChatMode = "assistant" | "supervisor";

export default function AiDesk() {
  const { data: status, error: statusError, loading, reload } = usePoll(codexStatus, 4000);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  // Which ChatGPT account (email) to connect. Optional — passed as the
  // OAuth `login_hint` so the sign-in page targets that account. Left
  // empty ⇒ ChatGPT shows its own account picker.
  const [email, setEmail] = useState("");

  const authed = !!status?.authenticated && !statusError;

  const login = async () => {
    setBusy(true);
    setMsg("Starting ChatGPT (Codex) login — approve in the browser that opens…");
    try {
      const r = await codexStart(email.trim() || undefined);
      if (r?.authorizeUrl) setMsg(`Open this URL to authorize: ${r.authorizeUrl}`);
      await reload();
    } catch (e) {
      setMsg(`Login failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };
  const logout = async () => {
    setBusy(true);
    try {
      await codexLogout();
      setMsg("Logged out.");
      await reload();
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  };

  // Switch the connected ChatGPT account (change the email): log out of the
  // current one, then immediately start a fresh login so a DIFFERENT account
  // can authorise. The account/email is whatever you sign in with here — it is
  // never hardcoded; it comes from the ChatGPT token you authorise.
  const switchAccount = async () => {
    setBusy(true);
    setMsg("Switching account — logging out, then opening a fresh ChatGPT login…");
    try {
      await codexLogout();
      await reload();
      const r = await codexStart(email.trim() || undefined);
      if (r?.authorizeUrl) {
        setMsg(`Sign in with the account you want to use: ${r.authorizeUrl}`);
      } else {
        setMsg("Approve the new ChatGPT account in the browser that opened.");
      }
      await reload();
    } catch (e) {
      setMsg(`Switch failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const [mode, setMode] = useState<ChatMode>("supervisor");

  const send = async () => {
    const prompt = input.trim();
    if (!prompt || busy || !authed) return;
    setInput("");
    setTurns((t) => [...t, { role: "you", text: prompt, mode }]);
    setBusy(true);
    try {
      if (mode === "supervisor") {
        const r = await supervisorChat(prompt);
        setTurns((t) => [...t, { role: "ai", text: `${r.reply}\n\nAction results: ${r.summary}`, mode }]);
      } else {
        const r = await codexChat(prompt);
        setTurns((t) => [...t, { role: "ai", text: r?.response ?? "(no response)", mode }]);
      }
    } catch (e) {
      setTurns((t) => [...t, { role: "ai", text: `Error: ${e}`, mode }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <Supervisor aiReady={authed} />
      <details style={{ marginTop: 20 }}>
      <summary>AI chat &amp; connection · {statusError ? "status unavailable" : loading ? "checking" : authed ? "signed in" : "not signed in"}</summary>
      <h2>AI Desk</h2>
      <p className="sub">Market briefing &amp; assistant via your ChatGPT subscription (Codex)</p>

      <div className="settings-grid">
        <div className="kv"><span>Status</span><b className={authed ? "buy" : "sell"}>{statusError ? "unknown" : loading ? "checking…" : authed ? "signed in" : "not signed in"}</b></div>
        <div className="kv"><span>Account</span><b style={{ fontSize: 12 }}>{status?.email ?? "—"}</b></div>
      </div>
      {statusError && <div className="banner warn" role="alert">Cannot check the AI connection: {statusError} <button onClick={() => void reload()}>Retry</button></div>}
      <div className="settings-grid" style={{ marginTop: 8 }}>
        <label className="kv" style={{ alignItems: "center" }}>
          <span title="Optional. The ChatGPT account (email) to connect. Leave empty to pick it in the browser.">
            ChatGPT email
          </span>
          <input
            type="email"
            value={email}
            placeholder="you@example.com (optional)"
            disabled={busy}
            autoComplete="email"
            onChange={(e) => setEmail(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !busy && (authed ? switchAccount() : login())}
            style={{ flex: 1, minWidth: 220 }}
          />
        </label>
      </div>
      <p className="muted" style={{ fontSize: 11, margin: "4px 0 0" }}>
        You sign in inside ChatGPT's own page — NeoEthos never sees your password.
        The email just tells it which account to use.
      </p>
      <div className="btn-row">
        {authed ? (
          <>
            <button onClick={switchAccount} disabled={busy} title="Log out and sign in with a different ChatGPT account / email.">Switch account</button>
            <button onClick={logout} disabled={busy}>Log out</button>
          </>
        ) : (
          <button className="primary" onClick={login} disabled={busy}>{busy ? "Working…" : "Connect ChatGPT"}</button>
        )}
      </div>
      {msg && <div className="banner info">{msg}</div>}

      <div className="btn-row" style={{ marginTop: 12, gap: 6 }}>
        <button
          className={mode === "assistant" ? "primary" : ""}
          disabled={busy}
          onClick={() => setMode("assistant")}
          title="Plain ChatGPT chat — market questions, strategy talk. No system access."
        >💬 Assistant</button>
        <button
          className={mode === "supervisor" ? "primary" : ""}
          disabled={busy}
          onClick={() => setMode("supervisor")}
          title="Tool-aware supervisor — reads the full system state (engines, journal, autopilot) and can ACT through the whitelisted, guard-railed actions. Same brain as the autonomous loop below."
        >🧭 Supervisor</button>
        <span className="muted small" style={{ alignSelf: "center" }}>
          {mode === "supervisor"
            ? "reads the whole system and can act (guard-railed)"
            : "plain chat — no system access"}
        </span>
      </div>
      <div className="chat">
        {turns.length === 0 && <p className="muted">Ask about the markets, a strategy, or your account — or switch to Supervisor to steer the system.</p>}
        {turns.map((t, i) => (
          <div key={i} className={`chat-turn ${t.role}`}>
            <b>{t.role === "you" ? "You" : "AI"} · {t.mode}</b>
            <div style={{ whiteSpace: "pre-wrap" }}>{t.text}</div>
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          value={input}
          placeholder={
            !authed
              ? "Connect ChatGPT first"
              : mode === "supervisor"
                ? "π.χ. Τι βλέπεις στα live engines; Ξεκίνα discovery στο GBPUSD M15…"
                : "Type a message…"
          }
          disabled={!authed || busy}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          style={{ flex: 1 }}
        />
        <button className="primary" onClick={send} disabled={!authed || busy || !input.trim()}>Send</button>
      </div>

      </details>
      <details style={{ marginTop: 20 }}>
      <summary>External tools · MCP configuration</summary>
      <McpTools />
      </details>
    </div>
  );
}

// ── MCP tool servers (external tools for the Supervisor) ────────────────────
// Config editor + live status for the MCP sidecar. Tools connected here
// (cTrader remote, MT5 bridges, web search, …) become available to the
// Supervisor's ACTION framework — trade-affecting calls still require your
// approval click. The sidecar reads mcp_servers.json at app start.
function McpTools() {
  const { data: st, error: statusError } = usePoll(mcpStatus, 15000);
  const [content, setContent] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    mcpConfigGet()
      .then((r) => { if (alive) { setContent(r.content); setLoaded(true); setLoadError(""); } })
      .catch((error) => { if (alive) setLoadError(String(error)); });
    return () => { alive = false; };
  }, [attempt]);

  const save = async () => {
    setBusy(true);
    try {
      const r = await mcpConfigSave(content);
      setMsg(`✓ Saved. ${r?.note ?? ""}`);
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const tools = st?.tools ?? [];
  return (
    <div>
      <h2>
        MCP tool servers{" "}
        <span className={`badge ${st?.reachable ? "live" : "demo"}`}>
          {statusError ? "STATUS UNAVAILABLE" : !st ? "CHECKING" : st.reachable ? `CONNECTED · ${tools.length} tools` : "SIDECAR OFF"}
        </span>
      </h2>
      <p className="muted small">
        External tools (MCP servers) the <b>Supervisor</b> can use: the official cTrader remote,
        MetaTrader&nbsp;5 bridges, web search, filesystem… Add servers below (JSON) — applied on the
        next app start. Supervisor calls outside the read-only allowlist are queued for approval.
        Connect only tools you trust; an external server has its own permissions and behaviour.
      </p>
      {statusError && <div className="banner warn" role="alert">MCP status could not be refreshed: {statusError}</div>}
      {loadError && <div className="banner warn" role="alert">Could not load MCP configuration: {loadError} <button onClick={() => setAttempt((value) => value + 1)}>Retry</button></div>}
      {tools.length > 0 && (
        <p className="muted small">
          Available: {tools.slice(0, 12).map((tool) => `${tool.server}/${tool.name}`).join(", ")}
          {tools.length > 12 ? ` … +${tools.length - 12} more` : ""}
        </p>
      )}
      <div className="ticket">
        <textarea
          value={content}
          aria-label="MCP configuration JSON"
          disabled={!loaded || busy}
          onChange={(e) => setContent(e.target.value)}
          spellCheck={false}
          style={{ width: "100%", minHeight: 160, fontFamily: "monospace", fontSize: 12 }}
          placeholder='{ "port": 7431, "servers": [ { "name": "ctrader", "transport": "http", "url": "https://mcp.spotware.com/mcp" } ] }'
        />
        <div className="btn-row" style={{ marginTop: 8 }}>
          <button className="primary" disabled={busy || !loaded || !content.trim()} onClick={save}>
            Save MCP config
          </button>
        </div>
        {msg && <div className="banner info" style={{ marginTop: 8 }}>{msg}</div>}
      </div>
    </div>
  );
}
