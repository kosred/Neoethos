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
  // Which ChatGPT account (email) to connect. Optional — passed as the
  // OAuth `login_hint` so the sign-in page targets that account. Left
  // empty ⇒ ChatGPT shows its own account picker.
  const [email, setEmail] = useState("");

  const authed = !!status?.authenticated;

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
