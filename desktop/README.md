# NeoEthos desktop

The React/TypeScript frontend runs inside the Tauri v2 desktop shell. Tauri links
the Rust engine in-process; Vite alone does not provide the application backend.

Use Node.js 24 for development and the existing TypeScript tests. The repository's
`rust-toolchain.toml` selects the Rust nightly. Native Tauri development also needs
the platform dependencies listed in [BUILDING.md](../BUILDING.md).

Run these commands from `desktop/`:

```sh
npm ci
npm test
npm run lint
npm run build
```

`npm test` runs the existing Node test suite without a broker account or a Tauri
process. `npm run build` type-checks and bundles the frontend; it does not build
the native desktop application. Both checks and lint run in the Stage 1 GitHub
Actions workflow.

For desktop development with the engine and hot reload:

```sh
npm run tauri -- dev
```

The `dev:tauri` pre-script builds the outbound MCP sidecar before starting Vite.
For frontend-only development, `npm run dev` starts Vite on port 5173, but screens
that call Tauri commands need the native host.

For platform bundles and the optional GPU build, follow
[BUILDING.md](../BUILDING.md). The favicon reuses the existing native application
icon; no separate template icon or duplicate branding asset is needed.
