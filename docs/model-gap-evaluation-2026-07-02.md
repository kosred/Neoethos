# Model-gap proposals — evaluation & verdict (2026-07-02)

Ten model additions were proposed (operator dialogue with an external
assistant) for the twin goals: **4%/month prop-firm** + **risky compounding**.
Each was checked against the actual codebase and the cited research before
deciding. Verdict legend: ✅ built · 🟡 already covered · ⏸ deferred
(evidence-gated) · ❌ rejected (doctrine).

## PropFirm-side proposals

| Proposal | Verdict | Why |
|---|---|---|
| Cox PH survival model (P(gene survives N days)) | 🟡 covered | Survival is **measured, not predicted** here: the challenge simulator (first-passage MC, `challenge_sim.rs`) reports pass/bust per sizing; scoring v4/v5 already penalise worst-day and drawdown in the GA itself. A Cox model over gene covariates needs far more (gene, lifetime) observations than exist and would be pure overfit surface. Revisit only when live-experience data is deep. |
| Volatility forecaster feeding sizing | ⏸ deferred | ATR/vol features are already in the cube; `risky_mode` has a volatility-sigma pause. A *forecast* wired into live sizing is a new money path — must pass the accept-only-if-beats-genes bar via the experience store first. |
| Regime-adaptive ensemble weights (RAGe-ENS, MSES 2026 — verified real) | ⏸ deferred | The blend already regime-gates (genes decide direction, meta gates). Dynamic per-regime ensemble weights = the MoE roadmap item; do it when the regime router lands, not before the fresh v5 archive exists. |
| Drawdown-recovery model | ✅ built (deterministic) | No model needed: **time-under-water** is computable exactly. `tail_risk` now reports `underwater_p95_trades` — p95 of the longest below-peak streak across reshuffles (+ UI card). |
| Causal break detector → re-discovery trigger | ⏸ deferred | The behaviour-level trigger just shipped (auto-cull → blacklist → auto-rediscovery, Settings-gated). `ConceptDriftMonitor` exists unwired and is the natural next step once live experience accumulates labels; causal-graph detection stays research-grade. |

## Risky-side proposals

| Proposal | Verdict | Why |
|---|---|---|
| QR-DQN distributional RL → CVaR sizing | ✅ built (deterministic equivalent) | The decision-relevant quantity is "does the left tail shrink my size?" — answered exactly by solving the Busseti/Ryu/Boyd bound on the **full empirical R-multiple distribution**: `risk_constrained_kelly_empirical` (core). Fat tails shrink f automatically; `tail_risk.rck_risk_pct` now uses it. A quantile-DQN would add a trained model + live-path risk for the same signal, on a 6-core box where neural training is already the weak lane. |
| Risk-constrained Kelly (Busseti/Ryu/Boyd) | ✅ built earlier same day | `domain::kelly` + tail-risk surfacing (advisory; the §7.1 30–50% risky band stays untouched). |
| Multivariate Kelly across correlated genes | ⏸ deferred | Joint return-distribution estimation across engines is fragile at our sample sizes; the correlation cap (0.7) + portfolio risk budget already bound joint exposure. Revisit with much deeper live data. |
| VAE out-of-distribution boundary detector | ⏸ deferred | Concept is sound (reduce risk in unseen regimes); an IsolationForest OOD veto on the live feature row is the cheap version and IsolationForest already exists in the model zoo — but it is a live entry gate ⇒ evidence-gate through the experience store like every ML influence. |
| PPO joint direction+sizing | ❌ rejected | Violates the locked decision: **ML never sets direction — genes do.** Sizing influence is allowed only as gated confirmation/scaling, never as a joint actor. |

## The principle applied

Every accepted item is **deterministic, testable, and advisory** — it changes
what the operator *knows*, not what the bot *does*, unless the operator sets
the number themselves. Every learned-model influence on live money remains
behind the experience-store evidence gate. This is the same discipline that
took live trading from 15% WR (unvalidated strategies) to parity-verified
OOS trading.
