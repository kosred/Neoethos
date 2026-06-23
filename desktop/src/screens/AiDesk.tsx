import { useState } from "react";
import { codexStatus, codexStart, codexLogout, codexChat } from "../api";
import { usePoll } from "../hooks";

type Turn = { role: "you" | "ai"; text: string };

export default function AiDesk() {
  const { data: status, reload } = usePoll(codexStatus, 4000);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const authed = !!status?.authenticated;

  const login = async () => {
    setBusy(true);
    setMsg("Starting ChatGPT (Codex) login — approve in the browser that opens…");
    try {
      const r = await codexStart();
      if (r?.authorize_url) setMsg(`Open this URL to authorize: ${r.authorize_url}`);
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

  const send = async () => {
    const prompt = input.trim();
    if (!prompt) return;
    setInput("");
    setTurns((t) => [...t, { role: "you", text: prompt }]);
    setBusy(true);
    try {
      const r = await codexChat(prompt);
      setTurns((t) => [...t, { role: "ai", text: r?.response ?? "(no response)" }]);
    } catch (e) {
      setTurns((t) => [...t, { role: "ai", text: `Error: ${e}` }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>AI Desk</h1>
      <p className="sub">Market briefing &amp; assistant via your ChatGPT subscription (Codex)</p>

      <div className="settings-grid">
        <div className="kv"><span>Status</span><b className={authed ? "buy" : "sell"}>{authed ? "connected" : "not connected"}</b></div>
        <div className="kv"><span>Account</span><b style={{ fontSize: 12 }}>{status?.email ?? "—"}</b></div>
      </div>
      <div className="btn-row">
        {authed ? (
          <button onClick={logout} disabled={busy}>Log out</button>
        ) : (
          <button className="primary" onClick={login} disabled={busy}>{busy ? "Working…" : "Connect ChatGPT"}</button>
        )}
      </div>
      {msg && <div className="banner info">{msg}</div>}

      <div className="chat">
        {turns.length === 0 && <p className="muted">Ask about the markets, a strategy, or your account.</p>}
        {turns.map((t, i) => (
          <div key={i} className={`chat-turn ${t.role}`}>
            <b>{t.role === "you" ? "You" : "AI"}</b>
            <div>{t.text}</div>
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          value={input}
          placeholder={authed ? "Type a message…" : "Connect ChatGPT first"}
          disabled={!authed || busy}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          style={{ flex: 1 }}
        />
        <button className="primary" onClick={send} disabled={!authed || busy}>Send</button>
      </div>
    </div>
  );
}
