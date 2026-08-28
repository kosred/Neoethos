# Show HN submission

**Title (80 chars max):**
Show HN: NeoEthos – open-source Rust trading engine with enforced anti-overfitting

**URL:** https://github.com/kosred/Neoethos

**First comment (post immediately after submitting):**

Author here. I'm a retail trader from Greece; I built this over two years on
a 6-core €300 mini PC because the serious tooling (walk-forward validation,
genetic strategy search, Kelly-aware sizing, prop-firm risk simulation)
lives behind institutional walls.

Technical bits HN might find interesting:

- Pure Rust hot path (migrated off a Python/Rust hybrid): genetic search
  over ~350 engineered features, deterministic backtester, native ML
  ensemble (XGBoost/LightGBM/CatBoost + Burn neural nets), single-process
  Tauri desktop app with the engine linked in-process.

- Strategies must survive five independent overfitting tests before export:
  walk-forward, CPCV, PBO/CSCV (López de Prado), a permutation test, and a
  parameter-plateau test. Most candidates die. That's the point.

- Backtest↔live parity is a tested invariant, not a hope: the live engine
  replicates trailing stops, weekend kill zones, news gates and costs
  exactly, verified by a parity harness.

- Position sizing is risk-constrained Kelly (Busseti/Ryu/Boyd) solved on the
  full empirical R-multiple distribution by bisection — rare catastrophic
  losses shrink the size automatically, no learned model needed.

- "Federation": a SETI@home-style mode where trusted peers pool discovery
  compute — shipped as an isolated P2P sidecar on iroh (QUIC + relays,
  automatic NAT traversal, gossip peer discovery; no server, no
  port-forwarding). Any instance can coordinate; imported results still
  pass every local validation gate.

- Never-OOM discipline: peak memory adapts to available hardware, never to
  user parameters; the engine chunks down instead of crashing.

AGPL-3.0, no telemetry (PRIVACY.md enumerates every outbound connection),
no signals or subscriptions for sale. It is not a money printer and does not
pretend to be one — it's an honesty machine for people who want to know
whether their edge is real.
