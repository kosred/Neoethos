# Post for r/algotrading

**Title:**
I spent 2 years building an open-source (AGPL) strategy-discovery + live-trading engine in pure Rust on a €300 mini PC — it enforces walk-forward, CPCV, PBO, permutation and plateau tests before any strategy can touch money

**Body:**

I'm a small retail trader from Greece. I couldn't afford the institutional
tooling, so I built it: **NeoEthos** — a desktop app (Tauri + React, engine
100% Rust) that discovers trading strategies with a genetic search and
refuses to let anything trade until it survives five independent
overfitting tests:

- walk-forward (unseen time)
- CPCV — combinatorially purged cross-validation (resampling)
- PBO/CSCV (López de Prado's probability of backtest overfitting)
- a permutation test (no profit allowed on destroyed data)
- a parameter-plateau test (±15% perturbation must keep the edge)

Other things it does that I haven't seen together in open source:

- **Backtest↔live parity as a tested invariant** — the live engine replicates
  the backtest's trailing stops, weekend kill zones, news gate and costs
  exactly, and a parity harness verifies it
- **Risk-constrained Kelly sizing** (Busseti/Ryu/Boyd) solved on the full
  empirical R-multiple distribution — fat left tails shrink position size
  automatically
- **Prop-firm challenge simulator** — first-passage Monte Carlo against
  FTMO-style barriers, sweeping risk-per-trade to find the challenge-optimal
  size (spoiler: it's not the Kelly size)
- **Auto-cull with a permanent blacklist** — live losers get retired forever
  and a fresh discovery is queued automatically
- **Federation** — SETI@home-style: friends can pool compute for discovery,
  now shipped as an isolated **P2P mesh sidecar** (iroh/QUIC — automatic
  relay connectivity and peer discovery, no server, no port-forwarding);
  every imported result still passes every local gate
- Native ML ensemble in Rust (XGBoost/LightGBM/CatBoost/Burn NNs) that may
  only *veto* trades, never set direction — and only after proving edge on
  live experience data

It trades through the cTrader Open API (your own account, your keys, no
middleman). No telemetry, no tracking, nothing leaves your machine — there's
a PRIVACY.md you can verify against the source.

**License: AGPL-3.0.** Free forever. I'm not selling signals, courses or a
subscription — I built this because small traders deserve the same
discipline institutions have.

Honest caveats: it's v0.5.6, Windows installers are prebuilt (Linux builds
from source), forex-focused via cTrader, and no strategy it finds is a
promise of profit — the whole point is that it tells you the *truth* about
an edge, including when there isn't one.

Repo + release: https://github.com/kosred/Neoethos

Happy to answer anything about the validation stack, the Rust architecture,
or the mistakes I made along the way (many).
