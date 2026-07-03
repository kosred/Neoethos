# Building NeoEthos from source

This guide takes you from a clean machine to a running build of NeoEthos — the
**Tauri + React desktop app** and/or the **terminal UI (TUI)**. Everything is
pure Rust on the hot path; there is no Python runtime and no separate backend
process to manage — the desktop app links the whole engine in-process.

> NeoEthos is licensed under the **AGPL-3.0-or-later** (see [LICENSE](LICENSE)).
> If you modify it and run it as a network service, you must offer your users
> the corresponding source.

---

## 1. What you'll build

| Target | Command | Output |
|---|---|---|
| Desktop installer | `cd desktop && npx tauri build` | `target/release/bundle/nsis/*.exe` + `msi/*.msi` (Windows), platform bundle elsewhere |
| Desktop (dev, hot-reload) | `cd desktop && npx tauri dev` | runs the app against the Vite dev server |
| Terminal UI (TUI) | `cargo run --release -p neoethos-cli` | live candlesticks, discovery, logs in your terminal |
| Headless engine checks | `cargo build --release -p neoethos-app` | the in-process API/engine library + binary |

The desktop bundle is self-contained: `npx tauri build` first runs the frontend
build (`npm run build`) and then compiles the Rust binary with the web assets
embedded. A plain `cargo build` on `neoethos-desktop` will **not** embed the UI —
always go through Tauri for a shippable app.

---

## 1.5 Hardware guide — what you need for what

NeoEthos was developed on a **6-core, €300 mini PC** — that is the honest
baseline, not a marketing minimum. The engine follows a **never-OOM**
discipline: peak memory adapts to *available* hardware, so a small machine
gets slower, never crashes.

| Use case | CPU | RAM | Disk | GPU |
|---|---|---|---|---|
| **Run the app + live trading** | 4 cores | 8 GB | ~5 GB (app + price data) | none |
| **Strategy discovery (comfortable)** | 6–8 cores | 16–32 GB | ~20 GB data/cache | none needed |
| **Serious discovery coverage** (many pairs × timeframes) | 16–32 cores | 64 GB | 50 GB+ | optional |
| **Build from source** | any | 16 GB | **~150 GB free** for `target/` on a full release build (then `cargo clean`) | — |
| **ML training (GPU lane)** | — | 32 GB+ | — | dedicated NVIDIA (CUDA); tested on RTX/A-series under Linux |

Notes from real-world experience:

- **Discovery scales with COVERAGE, not clock speed** — more (symbol,
  timeframe) combos in parallel beat a deeper search on one. This is also
  why the built-in **Federation** (Advanced → Federation) exists: several
  small machines out-search one big one.
- **Dense timeframes (M1–M5) are data-volume-bound**: on 6 cores a full M5
  history run can take ~13 h. Use `Max rows` in Settings to cap it, or
  federate.
- **GPU**: CPU is the default and the reliable path. A *dedicated* NVIDIA
  card under **Linux + CUDA** accelerates discovery evaluation; the
  Windows Vulkan GPU lane is currently blocked by an upstream `wgpu`
  incompatibility — leave compute on `auto`/`cpu` on Windows. **Never** pick
  `gpu` on a shared-RAM integrated GPU (it competes with the engine's RAM).
- **Laptops are fine** for running/trading; sustained discovery runs hot —
  a desktop/mini-PC with decent airflow is kinder for multi-hour searches.

---

## 2. Prerequisites (all platforms)

- **Rust** — stable toolchain, recent enough for the **2024 edition** (Rust **1.85+**). Install via [rustup](https://rustup.rs/). On Windows use the **MSVC** toolchain (`x86_64-pc-windows-msvc`), not GNU.
- **Node.js 20+** (Node 20.19+ or 22.12+ — required by Vite 8) and npm. [nodejs.org](https://nodejs.org/) or nvm.
- **CMake 3.28+** and a **C/C++ compiler** — several native crates (e.g. `lightgbm3-sys`, `polars`) build C/C++ at compile time.
- **Git**.

### Platform-specific system dependencies

**Windows 10/11**
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with the **“Desktop development with C++”** workload (MSVC + Windows SDK).
- **WebView2** runtime — preinstalled on Windows 11; on older Windows install the [Evergreen runtime](https://developer.microsoft.com/microsoft-edge/webview2/).
- **NSIS** is invoked by Tauri to produce the `.exe` installer. Tauri fetches it automatically; a system install lives at `C:\Program Files (x86)\NSIS`.

**Linux (Debian/Ubuntu)**
```bash
sudo apt update && sudo apt install -y \
  build-essential curl wget file cmake pkg-config \
  libssl-dev libwebkit2gtk-4.1-dev librsvg2-dev \
  libxdo-dev libayatana-appindicator3-dev
```
(Adjust package names for Fedora/Arch — the equivalents are `webkit2gtk4.1`, `openssl`, `librsvg2`, etc.)

**macOS**
```bash
xcode-select --install     # Command Line Tools (clang, make)
brew install cmake node    # if not already present
```

---

## 3. Get the source

```bash
git clone https://github.com/kosred/Neoethos.git   # or your fork
cd Neoethos
```

---

## 4. Build the desktop app

```bash
cd desktop
npm install                # one-time: pull the frontend deps
npx tauri build            # full release installer (~10–15 min on first build)
```

Installers land in:
- `target/release/bundle/nsis/NeoEthos_<version>_x64-setup.exe`
- `target/release/bundle/msi/NeoEthos_<version>_x64_en-US.msi`

For day-to-day development with hot-reload:

```bash
cd desktop
npx tauri dev
```

Frontend only (type-check + bundle, no Rust): `cd desktop && npm run build`.

---

## 5. Build & run the terminal UI

No Node needed — it's pure Rust:

```bash
cargo run --release -p neoethos-cli
```

The TUI hosts strategy discovery, model training and live candlesticks in the
terminal, driving the same engine as the desktop app.

---

## 6. First run & cTrader setup

NeoEthos trades through the **cTrader Open API**. To connect your account:

1. Create a cTrader Open API application in the [Spotware / cTrader developer portal](https://openapi.ctrader.com/) to obtain a **client id** and **client secret**, and register the OAuth redirect the app uses.
2. Launch NeoEthos → **Broker Setup** → enter the client id/secret and **Re-authenticate**. This runs the OAuth flow in your browser once; the token is stored locally and refreshed silently thereafter.
3. Pick the account to trade (a **Demo** account is strongly recommended until you have validated an edge).

Credentials are persisted to `broker_credentials.toml` on your machine and never
leave it. Authentication is automatic on subsequent launches.

---

## 7. Where data & config live (disk usage)

Charts are streamed live from the broker and are **not** written to disk. The
things that actually use disk are:

- **`target/`** — the Rust build cache. This is by far the largest consumer (several GB after a full build). Safe to `cargo clean` when you need the space.
- **Downloaded price history** — OHLCV you fetch for Discovery/Training, under the configured **data directory** (see Settings → Data). This is the data the engine trains on; deleting it means re-downloading.
- **Feature cubes** — temporary multi-timeframe feature stores written under the OS temp dir (`%TEMP%/neoethos_feature_store/…` on Windows) during discovery and reclaimed when the run ends.
- **Models & exported strategies/portfolios** — under the data directory; these are your results — keep them.

The app auto-prunes old installer bundles (keeping the latest) so `target/release/bundle` doesn't grow without bound.

---

## 8. GPU acceleration (optional, advanced)

A standard build runs the ML ensemble and genetic discovery on **CPU** out of the
box — nothing special is required. GPU acceleration is selected at **runtime**
(Settings → compute mode: auto/cpu/gpu), not via a build feature flag, and is
auto-detected where available.

Heavy CUDA acceleration (dedicated NVIDIA cards, libtorch/CUDA toolchain, VPS
deployment) is an advanced topic with its own environment requirements and is not
needed to build or run the app locally. If you have a shared-RAM integrated GPU,
prefer CPU or `auto` — a discovery run can otherwise exhaust shared memory.

---

## 9. Troubleshooting

- **`npx tauri build` succeeds but the app shows a blank window** — you likely ran a bare `cargo build`. Rebuild through Tauri so the frontend is embedded.
- **Linker/WebView errors on Linux** — re-check the `libwebkit2gtk-4.1-dev` and appindicator packages in §2.
- **`lightgbm`/native crate build fails** — install CMake 3.28+ and a working C++ toolchain; on Linux you may also need `libstdc++` dev headers.
- **cTrader shows a connection/auth error after install** — this is usually an expired OAuth token: open **Broker Setup → Re-authenticate**. The app surfaces the broker's real error code to tell you which case it is.
- **First build is slow** — the initial release build compiles the full dependency tree (10–15 min is normal). Subsequent builds are incremental.

---

## 10. Contributing

By contributing you agree your contributions are licensed under the project's
AGPL-3.0-or-later. Please keep the engine honest: no invented numbers in the UI,
fail loud with actionable errors on integration paths, and never let peak memory
scale with user parameters instead of available hardware.
