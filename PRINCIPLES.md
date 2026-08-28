# The Ethos — engineering principles

NeoEthos exists because the serious tools live behind institutional walls
while the small trader gets a chart and a prayer. Refusing that means more
than writing code — it means a discipline. These are the rules the codebase
actually enforces, not aspirations.

## 1. The math tells the truth

Every strategy must survive **five independent ways of being wrong** before
it may touch money:

- **Walk-forward** — does the edge hold on time it never saw?
- **CPCV** (combinatorially purged cross-validation) — does it hold across resamplings?
- **PBO/CSCV** (probability of backtest overfitting, López de Prado) — is the in-sample champion still good out-of-sample, or did we just pick the luckiest?
- **Permutation test** — does it still "profit" on structurally destroyed data? (Then it's noise, and it dies.)
- **Plateau test** — do ±15% parameter perturbations keep most of the edge, or is it a knife-edge overfit?

No cherry-picking, no survivorship editing, no "it looked good in the demo".

## 2. Fail loud

No silent fallbacks on any path that touches money or data. A missing
symbol, a NaN cost, an unreadable config — the system stops and says exactly
what is wrong and what to do, instead of trading on a guess.

## 3. Parity is sacred

The live engine must replicate the backtest **exactly**: the same trailing
stops, the same weekend kill zones, the same news gate, the same costs. A
strategy validated under one set of rules and traded under another is a lie
with extra steps. A parity harness verifies this, and parity bugs outrank
features.

## 4. Retire, never delete

User data is never deleted by the system. Losing strategies are **retired**
into a permanent blacklist — kept as a record, never selectable, never
re-discoverable. Memory of failure is part of the edge.

## 5. Models advise, genes decide

Machine learning never sets trade direction — the discovered, validated
strategy rules do. ML may only *veto or scale* (regime gates, confirmation),
and only after proving on live experience data that it adds edge.
Evidence-gated, never faith-gated.

## 6. Never OOM

Peak memory is a function of **available hardware**, never of user
parameters. If the machine is small, the engine gets slower — it does not
crash. Chunk down to a single unit of work if needed; degrade gracefully;
fail loud only when physics says no.

## 7. No invented numbers

Every figure in the UI comes from the engine. Nothing is synthesized,
smoothed, or beautified for presentation. If a number is unknown, the UI
shows that it is unknown.

## 8. Survival before growth

Position sizing derives from measured edge and survival constraints
(risk-constrained Kelly: maximum growth *subject to* a drawdown-probability
bound), not from hope. The risky mode states its ruin probability out loud
and makes you acknowledge it.

## 9. Your machine, your keys, your data

No telemetry, no server, no middleman. See [PRIVACY.md](PRIVACY.md) — and
verify it in the source.

## 10. Community over capital

The rich buy server farms; we share. Federation lets people who trust each
other pool coverage — while every imported result still passes every local
gate before any real money. The advantage that can't be bought is each
other.

---

*These principles are enforceable in code review: a PR that violates one is
a PR that doesn't merge, whoever wrote it.*
