# neoethos Flutter UI

Flutter desktop/mobile front-end for neoethos.

The UI is a thin client over the Rust backend. Business logic, broker auth,
order/risk guards, data bootstrap, Gemma, and chart data stay in Rust and are
exposed through `neoethos-app --server` on `http://127.0.0.1:7423`.

## Current Contract

- `lib/api/backend_client.dart` calls the Rust REST routes.
- `lib/startup/backend_supervisor.dart` locates and starts the backend with
  `--server` when needed.
- The user-facing brand is `neoethos`.
- The internal backend binary remains `neoethos-app`.

## Setup

```powershell
flutter pub get
flutter analyze
flutter test
flutter run -d windows
```

## Notes

- Flutter must not copy legacy Rust UI behavior.
- Server responses are the source of truth for screen state.
- Route or DTO changes should be verified against
  `crates/neoethos-app/src/server/mod.rs`.
