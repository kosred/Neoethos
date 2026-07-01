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
git clone https://github.com/kosred/forex-ai.git   # or your fork
cd forex-ai
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
