# Contributing to NeoEthos

Thank you — a project like this survives on people, not capital.

## Ways to help (no code required)

- **Run it and report** — install the [latest release](../../releases/latest),
  trade on a *demo* account, and open issues for anything confusing or broken.
  Real-world reports are the most valuable contribution there is.
- **Lend your cores** — join a federation group (Advanced → Federation) and
  contribute discovery compute to people you trust.
- **Translate** — the UI is English/Greek today; more languages welcome.
- **Documentation** — if something in [BUILDING.md](BUILDING.md) didn't work
  on your machine, a PR fixing it helps the next person.
- **Support the project** — see the Support section in the README.

## Contributing code

1. Read [PRINCIPLES.md](PRINCIPLES.md) first. PRs that violate a principle
   (silent fallbacks, invented UI numbers, parity breaks, memory that scales
   with user parameters) will be declined regardless of how clever they are.
2. Fork, branch, and keep PRs focused — one concern per PR.
3. **Verification is mandatory**: `cargo check` on touched crates, tests for
   new logic, and `npx tsc --noEmit` + `npm run build` under `desktop/` for
   UI changes.
4. Never edit validation gates, scoring, or live-trading paths without tests
   pinning the old and new behaviour. Scoring changes require a
   `SCORING_VERSION` bump with a changelog comment.
5. By contributing you agree your work is licensed **AGPL-3.0-or-later**.

## Good first areas

- Broker adapters (the trait exists; cTrader is the reference implementation)
- More timeframe/session presets and symbol metadata
- Federation Phase 1 (see `docs/p2p-mesh-design-2026-07-03.md` — sidecar only)
- TUI polish

## Security

If you find a vulnerability (especially anything touching credentials or
order execution), please **do not** open a public issue — use GitHub's
private vulnerability reporting on this repository.
