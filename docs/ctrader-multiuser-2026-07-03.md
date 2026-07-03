# cTrader multi-user — research (2026-07-03)

**Question:** can other users connect THEIR cTrader accounts through the app,
using the app's dev API credentials — without those credentials changing or
being exposed?

## Verdict: multi-user is supported BY DESIGN (per-user tokens)

cTrader Open API uses OAuth 2.0 exactly as you'd hope:

- The app keeps **one** `client_id` + `client_secret` (the developer's — ours).
- **Each user authorizes once**: they are sent to cTrader's authorization URI,
  log in with THEIR OWN cTID, and grant the app access to their account(s).
- Each user receives their **own** access token (~30-day) + refresh token
  (no expiry). The app refreshes silently thereafter — no re-auth needed.
- The `client_id`/`client_secret` stay the same for everyone; only the
  per-user tokens differ.

So nothing about cTrader blocks multi-user. What the APP needs is to support
**multiple stored user tokens** (one per connected user/account) instead of a
single one, plus UI to add / switch / remove accounts. That is an app feature,
not a cTrader limitation. Our Broker Setup already runs the OAuth flow; the
work is making token storage per-account and adding the switch UI.

Sources: [App & account authentication](https://help.ctrader.com/open-api/account-authentication/),
[Register an application](https://help.ctrader.com/open-api/api-application/).

## The real caveat: client_secret in a serverless desktop app

We have **no server** (by design — the whole ethos). So the OAuth token
exchange (`https://openapi.ctrader.com/apps/token`, which needs the
`client_secret`) happens on the user's machine, meaning the **secret is
embedded in the distributed binary** (our `embedded_credentials`). A
determined user can extract it. This is the well-known limitation of shipping
an OAuth *confidential client* as a desktop app.

**How bad is it, honestly?**
- Bounded: the secret does NOT grant access to anyone's account. Every account
  needs its own per-user OAuth consent, and each user's tokens live only on
  their machine. An extracted secret lets someone *impersonate our app in new
  OAuth flows* (users would still have to consent), not read existing users'
  data.
- But it IS a real exposure: a leaked secret could be abused for phishing-style
  "authorize NeoEthos" prompts, and Spotware could rate-limit/revoke the app.

**Options (for a later decision — needs one more research pass on cTrader
PKCE support):**
1. **Accept the bounded risk** (simplest; matches many open-source desktop
   OAuth apps). Document it.
2. **PKCE** — if cTrader supports the Authorization Code + PKCE flow for
   *public* clients, the client_secret is not needed in the client at all.
   MUST verify cTrader supports PKCE before relying on it.
3. **A minimal dev-run token-exchange proxy** — keeps the secret server-side,
   but breaks the no-server principle and puts you in the request path. Avoid
   unless PKCE is unavailable and the risk is judged unacceptable.

## Recommendation

- Build the **multi-account app feature** (per-user token store + add/switch
  UI) — this is the real work and is unblocked.
- Before shipping multi-user publicly, do a focused check: **does cTrader Open
  API support PKCE for public clients?** If yes, adopt it and stop embedding
  the secret. If no, accept + document the bounded exposure (option 1).
- Never log or transmit the secret; keep it out of any telemetry (we have none
  anyway — see PRIVACY.md).
