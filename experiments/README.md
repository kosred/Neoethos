# experiments/

Sub-projects that are NOT part of the v0.5.0 ship target.

Anything in this folder is intentionally **off the workspace**:

* Not in `Cargo.toml` `[workspace] members`.
* Not built by `cargo build --workspace`.
* Not bundled by `cargo packager`.
* Not part of any GitHub Release asset.

It exists as a holding area for prototypes that may graduate back to
a production crate later — or may be deleted outright. Anyone reading
the repo for the first time should treat `experiments/` as "ignore
me, the real code is in `crates/`".

## What's here

### `experiments/forex-flutter-ui/`

Flutter desktop/mobile UI scaffold from a 2026-05-18 spike that mirrored
the 14-tab egui layout under a Dart/Flutter shell. Status when parked
(2026-05-19): rendered the 14 sidebar entries with `PendingStub`
placeholders, talked to the Rust backend over REST + SSE through `dio` +
`http`, and ran on Windows desktop. Did not implement any of the panel
content; functional gap vs. the egui UI was ~95%.

**Why parked:** v0.5.0 picks egui as the single GUI target. The egui UI
is production-ready, tested live against the cTrader Open API on
2026-05-19, and is what end users install via the NSIS installer. The
Flutter spike was useful exploration but pulling two GUIs through the
v0.5.0 ship gate would double the verification surface without
delivering any user-facing capability the egui UI doesn't already.

If a future ticket revisits the Flutter UI it should:

1. Re-validate against the current `neoethos-app` REST + SSE surface
   (which has shifted since the 2026-05-18 scaffold).
2. Decide whether to keep `flutter_riverpod` + `go_router` or migrate
   to whatever the Flutter ecosystem standard is at that future date.
3. Set a clear feature-parity bar before re-introducing it as a workspace
   member.

Until then: leave it here, do not import it from anything in `crates/`.
