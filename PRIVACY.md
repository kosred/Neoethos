# Privacy & Trust

NeoEthos is built on a simple rule: **your machine, your data, your keys.**
This document is a complete, verifiable audit of what the software does with
your data. Because the code is AGPL, every claim here can be checked line by
line — don't trust, verify.

## What we collect

**Nothing.** There is:

- ❌ **no telemetry** — the app never reports usage, metrics or statistics anywhere
- ❌ **no analytics** — no Google Analytics, no tracking pixels, no fingerprinting
- ❌ **no crash reporting** — errors go to your local log files only
- ❌ **no accounts** — NeoEthos has no server, no sign-up, no user database
- ❌ **no auto-update phone-home** — updates are manual, from GitHub Releases
- ❌ **no ads, no bundled software, no miners, no malware** — audit the source

## Where your data lives

Everything stays on your machine:

| Data | Location | Leaves your machine? |
|---|---|---|
| Configuration | `config.yaml` | Never |
| Broker credentials | `broker_credentials.toml` + local secure store | Only to your broker (OAuth) |
| Price history | your configured data directory | Never |
| Discovered strategies & models | data/cache directories | Never (unless YOU share via Federation) |
| Trade journal | data directory | Never |
| Logs | local log directory | Never |

## Every outbound connection, enumerated

The app talks to the network **only** for these purposes, each one under your
control:

1. **Your broker (cTrader / Spotware Open API)** — OAuth sign-in, market data,
   order execution. Only after *you* configure credentials. This is the point
   of the app.
2. **Economic calendar (ForexFactory weekly JSON)** — powers the news gate
   that pauses trading around high-impact events. Can be disabled in
   Settings → News gate.
3. **News headlines (public RSS feeds)** — fetched only when you press
   Refresh on the News screen.
4. **ChatGPT (AI Desk / Supervisor)** — **off by default**. Works only if you
   sign in with *your own* OpenAI account; your prompts go to OpenAI under
   their terms. Never required for trading.
5. **Federation** — **off by default**. Connects only to coordinator URLs
   *you* type in, protected by a shared token. What you share (compute +
   strategy artifacts) is your explicit choice.

That's the complete list. Anything else would be a bug — please report it.

## Your keys

Broker credentials and OAuth tokens are stored locally and transmitted only
to the broker itself over TLS. NeoEthos has no middleman server that could
see, store or leak them — because there is no server at all.

## License as a guarantee

The AGPL-3.0 license isn't just legal text — it's the enforcement mechanism
for everything above. Anyone can read the code, and anyone who modifies and
serves it must publish their changes. Privacy claims you can't audit are
marketing; these you can.
