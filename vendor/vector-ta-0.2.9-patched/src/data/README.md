# VectorTA regression fixture

`2018-09-01-2024-Bitfinex_Spot-4h.vortex` is the immutable numeric-regression
fixture used by VectorTA's unit tests and benchmarks. It is not broker-real
NeoEthos market data and must never be used for costs, fills, PnL, promotion,
or live-trading decisions.

- Upstream repository: `https://github.com/VectorAlpha-dev/VectorTA`
- Upstream commit at capture: `802518e2392c5d011744b75e56e108e97a0682b4`
- Source path: `src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv`
- Source SHA-256: `1a4a908a43d4203ff136018e514dabfa179b6ed09b24c87631e247b4c5fafb2b`
- Vortex SHA-256: `a642e73d40c9653081ac4ddf5cc5c9be107a7dc86f7edcc307f90f8a969fcc28`
- Rows: `15,577`
- Physical schema: non-null `timestamp: i64`, `open/high/low/close/volume: f64`
- Legacy column order: timestamp, open, close, high, low, volume
- Timestamp convention: external/unknown; the fixture is formula-only and
  therefore deliberately ineligible for canonical broker replay.

The source CSV was used only by a one-time converter. It is intentionally not
stored in this repository. Runtime and test consumers reopen only Vortex.
