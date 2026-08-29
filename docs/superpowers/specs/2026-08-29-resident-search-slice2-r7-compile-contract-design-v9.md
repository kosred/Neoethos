# Resident Search Slice 2 R7 compile-contract correction v9

**Status:** approved design authority for R7 only

**Authoritative source base:** `6bff01cf2abbf711972f4fcac5348fe14753e5a1`

**Authority publication:** the design, implementation plan, and v9 SHA
manifest land together in one docs-only commit whose sole parent is the source
base above. That commit changes exactly those three paths. Its commit/tree IDs
are recorded at handoff because a Git commit cannot contain its own ID. Later
R7 implementation starts from that docs commit, verifies its parent and the
manifest-bound document bytes, and expects only the pre-existing `vendor/` and
ICE receipt to remain untracked.

**Supersession boundary:** this document replaces only the R7 executable
move-only-opacity design in the older Slice 2 design. It does not rewrite,
supersede, or weaken R1-R6, R8, R9, the v5 design receipt, or the R6 v8 plan.
The following pre-existing artifacts remain byte-identical:

- `docs/superpowers/specs/2026-08-28-resident-search-slice2-archive-knn-design.md`
  with normalized-LF SHA-256
  `52b166cc52a09358e47e9da3ce1daad5a692783fea820027fb4db491d2b1431a`;
- `docs/superpowers/plans/2026-08-29-resident-search-slice2-r6-combined-preallocation.md`
  with normalized-LF SHA-256
  `e3e876948d98bee1f6f8bdf351011ad3480a40d46505f378da759d0223e25a27`;
- `audit/resident-search-slice2-design-v5.sha256` with normalized-LF SHA-256
  `3f370ffe7561dc26e99b1834b482d0399a188befa2bd68da2b661e771b7de144`;
- `audit/resident-search-slice2-design-v8.sha256` with normalized-LF SHA-256
  `413263ceaa403e486ed626571407a1af3f00ce6ee1fe5ebe02504eef1454e443`.

For every normalized-LF digest in this authority, the algorithm is exact:
read the file as bytes; reject a UTF-8 BOM and reject invalid UTF-8; replace
each CRLF pair with LF, then replace every remaining bare CR with LF; perform
no other Unicode or whitespace normalization; preserve all other bytes,
including whether the file ends in a newline; and SHA-256 the resulting UTF-8
bytes. Raw digests named elsewhere hash the original bytes without this
transformation.

## Outcome and non-claims

R7 v9 creates a compiler-observed contract for the real nominal Slice 2 V3
state API. It proves, on the pinned host toolchain, that:

1. the supported public owner states are move-only at the external crate
   boundary;
2. each opaque state exposes no field;
3. the staged-state constructor remains crate-private;
4. a ranked state cannot be built with struct update syntax;
5. the novelty calibration receipt cannot be passed as the distinct full-run
   deadline receipt;
6. the Search façade re-exports the exact GPU nominal types rather than mirror
   structs; and
7. the Search façade is compiler-absent when the non-default contract feature
   is off; the GPU module's feature-off absence is proven separately by exact
   source and metadata gates.

R7 v9 does **not** prove a CUDA allocation, kernel launch, stream/event
dependency, trim-map ownership, archive pointer, population pointer, runtime
transition, cleanup path, numerical result, deadline, production binding,
headless readiness, GPU readiness, or application readiness. The V3 nominal
types are the required production path, but the later CUDA implementation must
still anchor its real combined-admission and transition signatures to them.
Until that later source/device review exists, R7 is host compile evidence only.

## Why v8 R7 could false-green

The superseded R7 text had four independent defects:

- its device-fixture feature pulled `gpu-b-native`, `cuda`, `cust`, VectorTA's
  native CUDA feature, and the `nvcc`/`cuobjdump` build path before the UI
  diagnostics could be observed;
- most named types and raw fields did not exist in production source, so a
  fixture mirror or an unresolved import could masquerade as opacity;
- the nested workspace did not repeat the repository's VectorTA patch and did
  not bind its target inventory, metadata origin, lockfile, compiler JSON, or
  environment; and
- ten feature-on cases could not prove that the compile-only surface was
  absent with the feature disabled.

The corrected design uses one canonical defining module, a re-export-only
Search façade, dependency-empty GPU feature, eleven isolated UI cases, exact
compiler-JSON identities and spans, an exact public-API receipt, and explicit
offline/vendor limitations.

## Honest mapping from superseded cases

The old raw-field names asserted CUDA internals that are not present yet. They
are deliberately not preserved as misleading filenames.

| Superseded v8 target | v9 target | Honest v9 claim |
| --- | --- | --- |
| `fail_read_trim_map_e0616` | `fail_read_chain_inner_e0616` | the canonical generation-chain representation is private |
| `fail_read_trim_event_e0616` | `fail_read_ranked_inner_e0616` | the canonical ranked-state representation is private |
| `fail_read_archive_pointer_e0616` | `fail_read_staged_inner_e0616` | the canonical staged-state representation is private |
| `fail_read_population_field_e0616` | `fail_read_pending_inner_e0616` | the canonical terminal-pending representation is private |
| `fail_construct_ranked_receipt_e0451` | `fail_construct_ranked_state_e0451` | the real public ranked owner cannot be constructed by FRU |

The clone, Copy, private-constructor, calibration-versus-deadline, and positive
typed-surface cases retain their semantic purpose. Raw trim/event/archive/
population ownership moves to later CUDA source and runtime evidence.

## Canonical production nominal module

The only defining source is the new file:

`crates/neoethos-gpu-cuda/src/resident_search_slice2_v3.rs`

`crates/neoethos-gpu-cuda/src/lib.rs` declares it exactly once:

```rust
#[cfg(any(
    feature = "cuda",
    feature = "resident-search-slice2-compile-contract"
))]
pub mod resident_search_slice2_v3;
```

Neither `resident_search_v2.rs`, `resident_archive_knn_v2.rs`, the R6 host
admission module, a UI source, nor a fixture module may redefine any V3 type.
Future CUDA code imports these same nominal items from
`crate::resident_search_slice2_v3`.

### Exact GPU public-item allowlist

The public children of
`neoethos_gpu_cuda::resident_search_slice2_v3` are exactly:

1. `ResidentArchiveKnnCalibrationReceiptV2`;
2. `ResidentSearchGenerationChainV3`;
3. `ResidentSearchRankEnqueuedV3`;
4. `ResidentSearchArchiveStagedV3`;
5. `ResidentSearchTerminalPendingV3`;
6. `ResidentSearchTerminalReceiptV3`;
7. `ResidentSearchTryCompleteV3`;
8. `ResidentSearchTransitionErrorV3`; and
9. `ResidentSearchRejectedAuthorityV3<A>`.

`ResidentSearchTryCompleteV3` is the sole public enum and has exactly:

```rust
pub enum ResidentSearchTryCompleteV3 {
    NotReady(ResidentSearchTerminalPendingV3),
    Complete(ResidentSearchTerminalReceiptV3),
}
```

Calibration, chain, ranked, staged, pending, terminal receipt, and transition
error are opaque public structs with exactly one private field named `inner`.
The rejection carrier has exactly the private fields `error` and `authority`.
The ranked and staged wire receipts remain crate-private and are not duplicated
as public UI types.

The declaration-only deadline marker remains the existing real type in
`neoethos-search/src/gpu_resident_current_config_plan_v1.rs`:

```rust
pub struct FullResidentDiscoveryDeadlineReceiptV1 {
    _not_minted_in_slice2: core::convert::Infallible,
}
```

### Exact public methods

The six allowed inherent methods are:

```rust
impl ResidentSearchGenerationChainV3 {
    pub fn enqueue_score_and_rank_v3(
        self,
    ) -> Result<
        ResidentSearchRankEnqueuedV3,
        ResidentSearchRejectedAuthorityV3<Self>,
    >;

    pub fn enqueue_terminal_seal_v3(
        self,
    ) -> Result<
        ResidentSearchTerminalPendingV3,
        ResidentSearchRejectedAuthorityV3<Self>,
    >;
}

impl ResidentSearchRankEnqueuedV3 {
    pub fn enqueue_stage_archive_from_rank_v3(
        self,
    ) -> Result<
        ResidentSearchArchiveStagedV3,
        ResidentSearchRejectedAuthorityV3<Self>,
    >;
}

impl ResidentSearchArchiveStagedV3 {
    pub fn enqueue_evolve_and_publish_v3(
        self,
    ) -> Result<
        ResidentSearchGenerationChainV3,
        ResidentSearchRejectedAuthorityV3<Self>,
    >;
}

impl ResidentSearchTerminalPendingV3 {
    pub fn try_complete_v3(
        self,
    ) -> Result<ResidentSearchTryCompleteV3, ResidentSearchTransitionErrorV3>;
}

impl<A> ResidentSearchRejectedAuthorityV3<A> {
    pub fn into_parts_v3(self) -> (ResidentSearchTransitionErrorV3, A);
}
```

Semicolons above specify signatures, not trait declarations. The Rust source
contains fail-closed bodies over uninhabited private internals until a later
CUDA implementation is reviewed.

The only non-public constructor named by R7 is:

```rust
impl ResidentSearchArchiveStagedV3 {
    pub(crate) fn from_ranked_v3(
        ranked: ResidentSearchRankEnqueuedV3,
    ) -> ResidentSearchArchiveStagedV3;
}
```

The real later archive-stage transition must call this same constructor. There
is no public constructor or sealer for calibration or any owner.

No other public inherent method, free function, field, trait, const, static,
type alias, macro, constructor, raw accessor, state conversion, or wrapper is
allowed. The defining crate may add no explicit local impl of `Clone`, `Copy`,
`Default`, `Deref`, `AsRef`, `Borrow`, `From`, or `Into`, and there may be no
non-identity conversion between states. Standard-library reflexive blanket
impls such as `From<T> for T`, `Into<T> for T`, and `Borrow<T> for T` are
external language/library facts; the design neither forbids nor counts them as
a local API.

## Search re-export façade and no-mirror proof

`neoethos-search/src/lib.rs` contains one inline public module:

```rust
#[cfg(any(
    feature = "gpu-b-native",
    feature = "resident-search-slice2-compile-contract"
))]
pub mod resident_search_slice2_v3 {
    pub use crate::gpu_resident_current_config_plan_v1::
        FullResidentDiscoveryDeadlineReceiptV1;
    pub use neoethos_gpu_cuda::resident_search_slice2_v3::{
        ResidentArchiveKnnCalibrationReceiptV2,
        ResidentSearchArchiveStagedV3,
        ResidentSearchGenerationChainV3,
        ResidentSearchRankEnqueuedV3,
        ResidentSearchRejectedAuthorityV3,
        ResidentSearchTerminalPendingV3,
        ResidentSearchTerminalReceiptV3,
        ResidentSearchTransitionErrorV3,
        ResidentSearchTryCompleteV3,
    };
}
```

The façade contains only those two `pub use` declarations. It contains no
local struct, enum, impl, function, trait, alias, const, static, macro, wrapper,
or constructor. The positive UI fixture assigns values typed through the GPU
path to sinks typed through the Search path for all nine GPU items. A mirror
with the same spelling therefore fails the positive case.

## Dependency-empty host compile feature

The exact new GPU feature is:

```toml
resident-search-slice2-compile-contract = []
```

It is non-default, has no `dep:` or feature edge, and neither includes nor is
included by `cuda`, `cuda-device-fixtures`, or the R6 host-contract feature.
The existing GPU feature values remain:

```toml
default = []
resident-search-slice2-host-contract = []
cuda = ["dep:cust", "dep:vector-ta"]
cuda-device-fixtures = ["cuda"]
```

The Search dependency becomes explicit about disabled defaults:

```toml
neoethos-gpu-cuda = {
    path = "../neoethos-gpu-cuda",
    optional = true,
    default-features = false,
}
```

The exact Search feature is:

```toml
resident-search-slice2-compile-contract = [
    "dep:neoethos-gpu-cuda",
    "neoethos-gpu-cuda/resident-search-slice2-compile-contract",
]
```

It does not imply `gpu`, `gpu-b-adapter`, `gpu-b-native`, `gpu-cuda`,
`neoethos-data/gpu-cuda`, `cuda`, or `cuda-device-fixtures`. It is absent from
every default/application/production aggregate.

Only the existing cfg gates for `canonical_discovery_config_digest_v1` and
`gpu_resident_current_config_plan_v1` add the compile-contract alternative:

```rust
#[cfg(any(
    test,
    feature = "gpu-b-adapter",
    feature = "resident-search-slice2-compile-contract"
))]
```

The R9 device feature remains a separate future CUDA concern. R7 does not add
or enable it.

## Outer integration target

`crates/neoethos-search/Cargo.toml` adds exactly:

```toml
[[test]]
name = "resident_search_slice2_compile_contract"
path = "tests/resident_search_slice2_compile_contract.rs"
required-features = ["resident-search-slice2-compile-contract"]
```

The runner contains exactly one test function named
`resident_search_slice2_compile_contract_v9`. It invokes the nested compiler
cases, metadata checks, source ratchets, hash ledger, and API receipt. It never
creates or calls a CUDA resource.

Tests-first sequencing is explicit. The provisional RED runner initially
contains only the exact case ledger, sanitized process/config boundary,
Cargo-JSON parser, and case executor. It reaches the positive invocation and
fails because the canonical API is missing; raw streams and hashes capture that
compiler-observed mismatch. It must not create placeholder receipts or run the
final receipt/API/source/hash preflight first. After the canonical API and real
receipts exist, the committed GREEN runner installs all final preflights before
the cases and fails closed on any missing or drifting receipt.

## Standalone nested workspace

The exact nested manifest path is:

`crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml`

Its non-target portion is:

```toml
[package]
name = "neoethos-resident-search-slice2-ui"
version = "0.0.0"
edition = "2024"
publish = false
autolib = false
autobins = false
autoexamples = false
autotests = false
autobenches = false

[workspace]
members = ["."]
resolver = "3"

[features]
default = []
resident-search-slice2-compile-contract = [
    "neoethos-search/resident-search-slice2-compile-contract",
]

[dependencies]
neoethos-search = { path = "../../..", default-features = false }
neoethos-gpu-cuda = { path = "../../../../neoethos-gpu-cuda", default-features = false }

[patch.crates-io]
vector-ta = { path = "../../../../../vendor/vector-ta-0.2.9-patched" }
```

The direct no-default GPU dependency remains present so the positive source can
name the GPU-path types, but the fixture feature does not activate the GPU
feature directly. Cargo feature unification therefore exposes the GPU module
only if the Search feature really forwards its declared GPU edge; deleting or
breaking that forwarding edge makes the positive case fail.

The VectorTA patch is load-bearing even though CUDA is disabled. Search
unconditionally traverses `neoethos-data`, which enables the patched
`vector-ta` with `nightly-avx`. The active VectorTA node must have exactly the
non-CUDA feature closure, `source = null`, and canonical manifest path
`$REPO/vendor/vector-ta-0.2.9-patched/Cargo.toml`. Its
`cuda-build-native`, `cust`, and `cust_derive` closure must be absent.

The manifest has exactly eleven explicit `[[bin]]` tables, each containing
only `name` and `path`. Metadata must report exactly those eleven bin targets
and no lib, test, example, or bench target.

## Exact UI matrix

The first ten cases enable the compile-contract feature. The last case omits
all features. Every negative except the feature-off case has exactly one
top-level authored primary error. The feature-off case also has exactly one
top-level authored primary error, through the Search façade.

| # | Exact bin | Exact source | Feature | Code | Exact primary span after canonical path resolution |
| ---: | --- | --- | --- | --- | --- |
| 1 | `pass_typed_surface` | `pass/typed_surface.rs` | on | success | no warning/error diagnostic for the selected fixture package/bin |
| 2 | `fail_clone_owner_e0599` | `fail/clone_owner_e0599.rs` | on | `E0599` | line 8, columns 13-18, token `clone` |
| 3 | `fail_copy_owner_e0277` | `fail/copy_owner_e0277.rs` | on | `E0277` | line 10, columns 18-25, token `chain()` |
| 4 | `fail_read_chain_inner_e0616` | `fail/read_chain_inner_e0616.rs` | on | `E0616` | line 8, columns 21-26, token `inner` |
| 5 | `fail_read_ranked_inner_e0616` | `fail/read_ranked_inner_e0616.rs` | on | `E0616` | line 8, columns 21-26, token `inner` |
| 6 | `fail_read_staged_inner_e0616` | `fail/read_staged_inner_e0616.rs` | on | `E0616` | line 8, columns 21-26, token `inner` |
| 7 | `fail_read_pending_inner_e0616` | `fail/read_pending_inner_e0616.rs` | on | `E0616` | line 8, columns 21-26, token `inner` |
| 8 | `fail_call_staged_constructor_e0624` | `fail/call_staged_constructor_e0624.rs` | on | `E0624` | line 8, columns 44-58, token `from_ranked_v3` |
| 9 | `fail_construct_ranked_state_e0451` | `fail/construct_ranked_state_e0451.rs` | on | `E0451` | line 8, columns 46-54, token `ranked()` in the FRU tail |
| 10 | `fail_novelty_receipt_as_full_deadline_e0308` | `fail/novelty_receipt_as_full_deadline_e0308.rs` | on | `E0308` | line 11, columns 27-40, token `calibration()` |
| 11 | `fail_feature_gate_off_e0432` | `fail/feature_gate_off_e0432.rs` | off | `E0432` | line 1, columns 22-47, token `resident_search_slice2_v3` |

Each negative has a same-stem tracked `.stderr`. The positive has none. The
source templates and line layout are frozen in the v9 implementation plan;
blank lines count. Any source edit that moves a primary span requires a new
versioned receipt.

The positive names all nine GPU nominal types, the deadline marker, both
`ResidentSearchTryCompleteV3` variants, and all six public method signatures.
It assigns GPU-path suppliers to Search-path sinks for all nine nominal types.
Suppliers panic if executed; `cargo check` type-checks them but never runs or
links a UI binary. It calls no constructor and creates no GPU resource.

## Exactly eleven compiler invocations

For cases 1-10 the runner substitutes only the exact bin name into:

```text
cargo +nightly-2026-04-07 check --manifest-path crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml --locked --offline -j 7 --no-default-features --features resident-search-slice2-compile-contract --bin <exact-bin> --message-format=json --color never
```

Case 11 uses:

```text
cargo +nightly-2026-04-07 check --manifest-path crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml --locked --offline -j 7 --no-default-features --bin fail_feature_gate_off_e0432 --message-format=json --color never
```

There is no `--bins`, package-wide check, inferred target, default feature, or
twelfth UI invocation. The pass runs first, then the nine feature-on negatives
in table order, and the feature-off case last.

## Fresh targets and exact environment

The outer command uses a newly generated, asserted-nonexistent GUID target.
The runner uses two further asserted-nonexistent target roots: one shared
sequentially by cases 1-10 and one used only by case 11. A third fresh root is
used for rustdoc JSON/API evidence. Sharing the feature-on target bounds disk
use without permitting the feature-off observation to reuse feature-on state.
The outer launcher sanitizes the environment before Cargo can compile the
outer test or any build script; the runner applies the same contract again to
every metadata, check, and rustdoc child.

Every child command has exactly these controlled values:

```text
CARGO_INCREMENTAL=0
RUSTFLAGS=-Dwarnings
CARGO_NET_OFFLINE=true
CARGO_TERM_COLOR=never
RUST_BACKTRACE=0
```

The runner uses `Command::env_clear()`, and the outer PowerShell launcher uses
`ProcessStartInfo.Environment.Clear()`. Each copies only the case-insensitive
Windows host/bootstrap allowlist `SystemRoot`, `WINDIR`, `ComSpec`, `PATH`,
`PATHEXT`, `TEMP`, `TMP`, `USERPROFILE`, `HOMEDRIVE`, `HOMEPATH`,
`LOCALAPPDATA`, `APPDATA`, `PROGRAMDATA`, `ProgramFiles`,
`ProgramFiles(x86)`, `ProgramW6432`, `NUMBER_OF_PROCESSORS`,
`PROCESSOR_ARCHITECTURE`, `PROCESSOR_IDENTIFIER`, `RUSTUP_HOME`, `CARGO_HOME`,
`VSINSTALLDIR`, `VCINSTALLDIR`, `VCToolsInstallDir`, `WindowsSdkDir`,
`WindowsSDKVersion`, `UCRTVersion`, `UniversalCRTSdkDir`, `INCLUDE`, `LIB`, and
`LIBPATH` when present. It then adds the controlled values above, canonical
`CARGO`, and its exact child `CARGO_TARGET_DIR`. No other `CARGO_*`, `RUST*`,
`CC*`, `CXX*`, `AR*`,
`CUDA*`, `NVCC*`, or target-linker variable outside the explicit lists may
survive.

In particular `CARGO_ENCODED_RUSTFLAGS`, `RUSTC`, `RUSTDOC`,
`RUSTC_WRAPPER`, `RUSTC_WORKSPACE_WRAPPER`, `CARGO_BUILD_RUSTC_WRAPPER`,
`CARGO_BUILD_TARGET`, `RUSTDOCFLAGS`, `CARGO_ENCODED_RUSTDOCFLAGS`,
`RUSTUP_TOOLCHAIN`, `CC`, `CXX`, and `AR` are absent. `CUDA_PATH`, `CUDA_HOME`,
`CUDACXX`, and `CUDAOBJDUMP` are then set to distinct nonexistent sentinel
paths under the fresh target. Hidden CUDA activation therefore fails instead
of finding a machine-global toolkit. The runner records the allowlisted key
names, the SHA-256 of each non-secret value, the canonical Cargo executable,
and the detected host C++ compiler identity without emitting raw environment
values.

Before any Cargo process, both layers enumerate every Cargo configuration file
Cargo could discover from the canonical working directory through the
filesystem root and from the effective `CARGO_HOME`. The prepared checkout
inventory is exactly:

```text
de69922d58cddec2c0383536b40ea2491a23c56a34510c5bde7488d13964fdb8  $REPO/.cargo/config.toml
5ee03587848a82cc5b50a2d41ae2cd7a56c6da1a3e320c865a54262442757b38  <one enclosing-workspace>/.cargo/config.toml
```

There is no discovered legacy `.cargo/config`, no Cargo-home config, and no
third file. The launcher binds canonical paths plus raw hashes and fails before
Cargo on any difference. The two reviewed files contain only the known
`BINDGEN_EXTRA_CLANG_ARGS`, job-count, and target-rustflags settings; the
controlled `RUSTFLAGS` environment has higher priority. Any discovered target,
runner, linker, `rustc`, wrapper, source replacement, registry override, or
force-overriding `[env]` setting outside those exact hashed bytes fails closed.
No invocation adds `--config`.

The toolchain identity must equal:

```text
rustc 1.96.0-nightly (bcded3316 2026-04-06)
commit-hash: bcded331651b60a0383b3ff51db4f24c4495ac53
host: x86_64-pc-windows-msvc
LLVM version: 22.1.2
```

The channel remains `nightly-2026-04-07`. A C++17-capable host compiler is
still required because the GPU crate's CUDA-off build script compiles the
honest stub/layout sources. That host compile is not CUDA evidence.

The sanitized launcher resolves `cargo.exe` and `rustc.exe` once to canonical
application paths. Under the same sanitized environment used for the build, it
captures `cargo +nightly-2026-04-07 -Vv` and
`rustc +nightly-2026-04-07 -vV`; the latter must match the frozen identity
above. The canonical Cargo application path is exactly
`C:\Users\konst\.cargo\bin\cargo.exe`. Its stdout, after only CRLF/bare-CR to LF
normalization and with the final LF preserved, is exactly 337 UTF-8 bytes with
SHA-256 `7d4a0723c4202c639b08fdf5a12b01f4cd6eaad342126018e401c6c01ce794a3`:

```text
cargo 1.96.0-nightly (888f67534 2026-03-30)
release: 1.96.0-nightly
commit-hash: 888f675344eb1cf2308fd53183e667bdd2c58e51
commit-date: 2026-03-30
host: x86_64-pc-windows-msvc
libgit2: 1.9.2 (sys:0.20.4 vendored)
libcurl: 8.19.0-DEV (sys:0.4.87+curl-8.19.0 vendored ssl:Schannel)
os: Windows 10.0.26200 (Windows 11 Professional) [64-bit]
```

Cargo identity stderr is empty. The launcher exact-matches this complete
receipt before any metadata, check, rustdoc, generate-lockfile, or outer-test
Cargo invocation. It retains the raw streams separately because their host
line endings are evidence too. The outer command and every runner child use
that same canonical Cargo path; a different path, byte, executable, proxy
result, host triple, or toolchain fails.

## Compiler-JSON observation boundary

Every Cargo or tool-identity invocation captures stdout and process stderr as
two independent raw byte streams before either is decoded. Each stream's byte
count and SHA-256 are retained, valid UTF-8 is required for classification, and
the complete streams are losslessly re-emitted after capture; truncation,
merging, `tail`, or line dropping is forbidden. Metadata stdout is parsed as
one complete JSON document. Check/rustdoc stdout is parsed as newline-delimited
Cargo JSON. Process stderr is classified by origin and severity into
INFO/WARNING/ERROR/other events, but it is never confused with a tracked UI
`.stderr` file: the latter is derived only from the selected compiler
diagnostic's JSON `rendered` value.

For each check/rustdoc Cargo stream, the runner parses every nonblank stdout
line as Cargo JSON. Non-JSON stdout is a failure. For each
`compiler-message` it records:

- normalized `package_id`;
- target name, kind, crate types, and canonical `src_path`;
- diagnostic level and optional code;
- every top-level span, including `is_primary`, canonical file path, byte,
  line, and column boundaries;
- labels and rendered text; and
- child diagnostics without counting child-note spans as a second top-level
  authored primary error.

It also requires the final `build-finished` message and the correct success
bit: true for the positive, false for every negative.

The selected diagnostic must belong to package
`neoethos-resident-search-slice2-ui 0.0.0`, the exact selected bin target of
kind `bin`, and the exact canonical source in the table. A negative must have
exactly one top-level primary authored error with the specified code and span.
Code-less, span-less terminal summaries such as rustc's abort count may be
recorded but cannot satisfy or add an authored-primary result. Any additional
top-level authored primary error, unresolved dependency, wrong package/target,
warning promoted from the selected bin, skipped target, wrong exit status, or
stderr drift fails.

The positive requires zero warning/error `compiler-message` events for the
selected fixture package/bin. Cap-linted diagnostics from locked dependencies
are preserved and classified by package; they are not erased and the complete
build is not described as globally warning-free.

### Only allowed normalization

Rendered diagnostics are normalized in this order:

1. CRLF and bare CR become LF;
2. path separators become `/`;
3. the exact canonical current target, fixture, and repository prefixes become
   `$TARGET`, `$FIXTURE`, and `$REPO` respectively, applied longest-prefix
   first so the enclosing repository cannot mask either child root; and
4. ANSI SGR color sequences are rejected, with stripping permitted only to
   produce a useful mismatch report.

No message text, code, label, quote, line, column, byte offset, package name,
target name, compiler wording, or arbitrary path suffix is normalized. Joined
rendered output for the one authored primary error is compared byte-for-byte
with the same-stem `.stderr` after this normalization.

## Exact API and source receipts

Nightly rustdoc JSON is produced under the compile-contract feature into the
third fresh target for both crates. The runner filters only the two public
modules named `resident_search_slice2_v3` and writes/compares the normalized
tracked receipt:

`crates/neoethos-search/tests/ui/resident_search_slice2/api-surface-v3.txt`

Rows cover module, public type, enum variant and payload, inherent method
signature, generic parameter, and Search re-export origin. The expected rows
are exactly the nine GPU items, two `ResidentSearchTryCompleteV3` variants, six
methods, and the Search façade's nine GPU origins plus deadline marker. Any
extra child is a failure. Explicit local impls of `Clone`, `Copy`, `Default`,
`Deref`, `AsRef`, `Borrow`, `From`, or `Into` are rejected; external reflexive
blanket impls are discarded by origin and are not treated as local children.
Compiler-positive cross-path assignments independently prove nominal re-export
identity.

The receipt is UTF-8, LF-terminated, ordinally sorted, and contains exactly
these rows after rustdoc IDs and whitespace are reduced to the named nominal
paths below:

```text
gpu|enum|ResidentSearchTryCompleteV3
gpu|method|ResidentSearchArchiveStagedV3|enqueue_evolve_and_publish_v3|fn(self)->Result<ResidentSearchGenerationChainV3,ResidentSearchRejectedAuthorityV3<Self>>
gpu|method|ResidentSearchGenerationChainV3|enqueue_score_and_rank_v3|fn(self)->Result<ResidentSearchRankEnqueuedV3,ResidentSearchRejectedAuthorityV3<Self>>
gpu|method|ResidentSearchGenerationChainV3|enqueue_terminal_seal_v3|fn(self)->Result<ResidentSearchTerminalPendingV3,ResidentSearchRejectedAuthorityV3<Self>>
gpu|method|ResidentSearchRankEnqueuedV3|enqueue_stage_archive_from_rank_v3|fn(self)->Result<ResidentSearchArchiveStagedV3,ResidentSearchRejectedAuthorityV3<Self>>
gpu|method|ResidentSearchRejectedAuthorityV3<A>|into_parts_v3|fn(self)->(ResidentSearchTransitionErrorV3,A)
gpu|method|ResidentSearchTerminalPendingV3|try_complete_v3|fn(self)->Result<ResidentSearchTryCompleteV3,ResidentSearchTransitionErrorV3>
gpu|module|resident_search_slice2_v3
gpu|struct|ResidentArchiveKnnCalibrationReceiptV2
gpu|struct|ResidentSearchArchiveStagedV3
gpu|struct|ResidentSearchGenerationChainV3
gpu|struct|ResidentSearchRankEnqueuedV3
gpu|struct|ResidentSearchRejectedAuthorityV3<A>
gpu|struct|ResidentSearchTerminalPendingV3
gpu|struct|ResidentSearchTerminalReceiptV3
gpu|struct|ResidentSearchTransitionErrorV3
gpu|variant|ResidentSearchTryCompleteV3::Complete(ResidentSearchTerminalReceiptV3)
gpu|variant|ResidentSearchTryCompleteV3::NotReady(ResidentSearchTerminalPendingV3)
search|module|resident_search_slice2_v3
search|reexport|FullResidentDiscoveryDeadlineReceiptV1|crate::gpu_resident_current_config_plan_v1::FullResidentDiscoveryDeadlineReceiptV1
search|reexport|ResidentArchiveKnnCalibrationReceiptV2|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentArchiveKnnCalibrationReceiptV2
search|reexport|ResidentSearchArchiveStagedV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchArchiveStagedV3
search|reexport|ResidentSearchGenerationChainV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchGenerationChainV3
search|reexport|ResidentSearchRankEnqueuedV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchRankEnqueuedV3
search|reexport|ResidentSearchRejectedAuthorityV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchRejectedAuthorityV3
search|reexport|ResidentSearchTerminalPendingV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchTerminalPendingV3
search|reexport|ResidentSearchTerminalReceiptV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchTerminalReceiptV3
search|reexport|ResidentSearchTransitionErrorV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchTransitionErrorV3
search|reexport|ResidentSearchTryCompleteV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchTryCompleteV3
```

The parser resolves rustdoc import IDs back to their canonical paths before
emitting rows. It never infers identity from equal spelling alone.

Source ratchets require:

- exactly one definition of every allowlisted GPU type, all in
  `resident_search_slice2_v3.rs`;
- exactly one GPU module declaration with the exact cfg;
- the Search façade body contains only the two exact `pub use` declarations;
- exactly the private `inner`/`error`/`authority` field names and the one
  crate-private constructor;
- no public raw accessor or constructor token;
- zero diff in `resident_search_v2.rs`,
  `resident_search_slice2_admission_v2.rs`, `build.rs`, root `Cargo.toml`, and
  root `Cargo.lock`; and
- root `Cargo.lock` remains raw SHA-256
  `725cc6fb8645a0d7d9cd11f32bab01dcc8cc3de0497a9df5472886e20eb2167f`.

## Metadata, lock, and vendor identity

The nested `Cargo.lock` is tracked. A tracked
`r7-v9-receipt.sha256` binds the raw bytes of the nested manifest, nested lock,
all eleven sources, all ten `.stderr` files, `api-surface-v3.txt`, and the
canonical type/facade sources. No placeholder digest is accepted.

Feature-on metadata must prove:

- exactly the eleven explicit bin targets and no auto target;
- fixture default features are empty;
- the active Search and GPU nodes contain only the compile-contract features
  relevant to R7;
- `neoethos-gpu-cuda` lacks `cuda` and `cuda-device-fixtures`;
- `neoethos-data` lacks `gpu-cuda`;
- the active `vector-ta` node has `nightly-avx` but not
  `cuda-build-native`;
- no active package/node named `cust`, `cust_raw`, or `find_cuda_helper`;
- VectorTA has `source = null` and canonical manifest path
  `$REPO/vendor/vector-ta-0.2.9-patched/Cargo.toml`; and
- build-script JSON contains no CUDA/cudart/driver link library or CUDA link
  search path. The stub archive and a `rerun-if-env-changed=CUDA_PATH` line are
  not CUDA links.

Cargo JSON cannot observe arbitrary child-process launches. Non-attempt of
`nvcc`/`cuobjdump` is a separate bounded inference: reviewed unchanged build
source keeps those calls behind absent CUDA/native features, the run starts
from fresh targets, documented CUDA tool overrides point to nonexistent
sentinels before the outer build starts, and the build completes successfully.
The sentinel is a fail-closed backstop for the reviewed tool lookup; it is not
runtime tracing and no broader claim about arbitrary process execution is made.

Feature-off metadata must show neither crate's compile-contract feature and
the compiler case must observe the exact façade `E0432`.

The two nested metadata commands are exactly:

```text
cargo +nightly-2026-04-07 metadata --manifest-path crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml --locked --offline --format-version 1 --no-default-features --features resident-search-slice2-compile-contract
cargo +nightly-2026-04-07 metadata --manifest-path crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml --locked --offline --format-version 1 --no-default-features
```

The outer target declaration is read from this exact no-dependency metadata
command and must report the one exact required feature:

```text
cargo +nightly-2026-04-07 metadata --manifest-path crates/neoethos-search/Cargo.toml --locked --offline --format-version 1 --no-deps
```

The two API commands share the separate fresh rustdoc target and are exactly:

```text
cargo +nightly-2026-04-07 rustdoc --locked --offline -j 7 -p neoethos-gpu-cuda --lib --no-default-features --features resident-search-slice2-compile-contract --message-format=json --color never -- -Dwarnings -Z unstable-options --output-format json
cargo +nightly-2026-04-07 rustdoc --locked --offline -j 7 -p neoethos-search --lib --no-default-features --features resident-search-slice2-compile-contract --message-format=json --color never -- -Dwarnings -Z unstable-options --output-format json
```

They must produce exactly `$TARGET/doc/neoethos_gpu_cuda.json` and
`$TARGET/doc/neoethos_search.json`. Metadata and rustdoc invocations are
topology/API preludes and are not counted among the eleven UI `cargo check`
cases.

The prepared checkout's current VectorTA directory is entirely untracked but
has this v9 provenance snapshot:

```text
files: 1077
canonical tree-manifest SHA-256: def4551c993af6e9149c6a93fee1733a43c77629d132d28eee1c1fc16bd224b5
```

The tree-manifest algorithm sorts regular files by ordinal forward-slash
relative path, emits lowercase `sha256  relative/path\n` for each file, and
hashes the UTF-8 manifest bytes. This digest proves only the currently
provisioned external bytes; it does not make them part of Git history.

The root manifest additionally names eleven untracked local patch directories:
the exact-content-bound VectorTA path above plus existence-only checks for the
other ten currently inactive paths `vendor/lightgbm3`, `vendor/lightgbm3-sys`,
`vendor/xgboost_lib-sys`, `vendor/sklears-core`, `vendor/rlkit`,
`vendor/cubecl-runtime-0.10.0-patched`,
`vendor/cubecl-cuda-0.10.0-patched`,
`vendor/cubek-matmul-0.2.0-patched`,
`vendor/cubek-convolution-0.2.0-patched`, and
`vendor/catboost-rust-0.3.8-patched`. At this base, `git ls-files vendor`
returns zero while `vendor/` contains 6,185 files and 316,511,667 bytes. Cargo
may need to load all eleven root patch manifests before the outer integration
test starts. Therefore:

- prepared-checkout evidence is allowed only when the exact VectorTA
  1,077-file content receipt passes and each of the other ten named patch
  directories and its `Cargo.toml` exists and is readable; this is explicitly
  not an exact-content receipt for those ten inactive paths;
- a missing or mismatched patch, vendor digest, nested lock, metadata origin,
  offline cache, host compiler, or pinned toolchain fails closed; and
- fresh-clone, self-contained-offline, CI-portable, or repository-reproducible
  R7 evidence remains **blocked** until the whole required vendor closure is
  tracked or externally provisioned through a committed digest/provenance and
  bootstrap gate.

## Exact outer command

From the repository root on the prepared checkout, after the provenance gate:

```powershell
$ErrorActionPreference = "Stop"
$r7Repo = (Resolve-Path -LiteralPath .).Path
$r7Run = Join-Path $r7Repo ("target\resident-search-slice2-r7-v9-" + [Guid]::NewGuid().ToString("N"))
if (Test-Path -LiteralPath $r7Run) { throw "R7 run root already exists" }
$null = New-Item -ItemType Directory -Path $r7Run
$r7Outer = Join-Path $r7Run "outer-target"
if (Test-Path -LiteralPath $r7Outer) { throw "R7 outer target already exists" }

$r7Cargo = (Resolve-Path -LiteralPath (@(Get-Command cargo.exe -CommandType Application -All)[0].Source)).Path
$r7Rustc = (Resolve-Path -LiteralPath (@(Get-Command rustc.exe -CommandType Application -All)[0].Source)).Path
if ($r7Cargo -cne "C:\Users\konst\.cargo\bin\cargo.exe") {
    throw "canonical Cargo path mismatch"
}

$r7ConfigCandidates = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
$r7Cursor = [System.IO.DirectoryInfo]$r7Repo
while ($null -ne $r7Cursor) {
    $null = $r7ConfigCandidates.Add((Join-Path $r7Cursor.FullName ".cargo\config.toml"))
    $null = $r7ConfigCandidates.Add((Join-Path $r7Cursor.FullName ".cargo\config"))
    $r7Cursor = $r7Cursor.Parent
}
$r7CargoHome = if ($env:CARGO_HOME) {
    [System.IO.Path]::GetFullPath($env:CARGO_HOME)
} else {
    Join-Path $env:USERPROFILE ".cargo"
}
$null = $r7ConfigCandidates.Add((Join-Path $r7CargoHome "config.toml"))
$null = $r7ConfigCandidates.Add((Join-Path $r7CargoHome "config"))
$r7Configs = @($r7ConfigCandidates | Where-Object { Test-Path -LiteralPath $_ })
if ($r7Configs.Count -ne 2) { throw "unexpected Cargo config inventory" }
$r7RepoConfig = Join-Path $r7Repo ".cargo\config.toml"
if (-not ($r7Configs -contains $r7RepoConfig)) { throw "missing repository Cargo config" }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $r7RepoConfig).Hash.ToLowerInvariant() -ne
    "de69922d58cddec2c0383536b40ea2491a23c56a34510c5bde7488d13964fdb8") {
    throw "repository Cargo config drift"
}
$r7EnclosingConfig = @($r7Configs | Where-Object { $_ -ne $r7RepoConfig })
if ($r7EnclosingConfig.Count -ne 1 -or
    (Get-FileHash -Algorithm SHA256 -LiteralPath $r7EnclosingConfig[0]).Hash.ToLowerInvariant() -ne
    "5ee03587848a82cc5b50a2d41ae2cd7a56c6da1a3e320c865a54262442757b38") {
    throw "enclosing Cargo config drift"
}

$r7Allowed = @(
    "SystemRoot", "WINDIR", "ComSpec", "PATH", "PATHEXT", "TEMP", "TMP",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA",
    "PROGRAMDATA", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
    "RUSTUP_HOME", "CARGO_HOME", "VSINSTALLDIR", "VCINSTALLDIR",
    "VCToolsInstallDir", "WindowsSdkDir", "WindowsSDKVersion", "UCRTVersion",
    "UniversalCRTSdkDir", "INCLUDE", "LIB", "LIBPATH"
)
$r7Env = @{}
foreach ($r7Name in $r7Allowed) {
    $r7Value = [Environment]::GetEnvironmentVariable($r7Name)
    if ($null -ne $r7Value) { $r7Env[$r7Name] = $r7Value }
}
$r7Env["CARGO"] = $r7Cargo
$r7Env["CARGO_TARGET_DIR"] = $r7Outer
$r7Env["CARGO_INCREMENTAL"] = "0"
$r7Env["RUSTFLAGS"] = "-Dwarnings"
$r7Env["CARGO_NET_OFFLINE"] = "true"
$r7Env["CARGO_TERM_COLOR"] = "never"
$r7Env["RUST_BACKTRACE"] = "0"
$r7Env["CUDA_PATH"] = Join-Path $r7Run "missing-cuda-path"
$r7Env["CUDA_HOME"] = Join-Path $r7Run "missing-cuda-home"
$r7Env["CUDACXX"] = Join-Path $r7Run "missing-nvcc.exe"
$r7Env["CUDAOBJDUMP"] = Join-Path $r7Run "missing-cuobjdump.exe"
foreach ($r7Sentinel in @("CUDA_PATH", "CUDA_HOME", "CUDACXX", "CUDAOBJDUMP")) {
    if (Test-Path -LiteralPath $r7Env[$r7Sentinel]) { throw "CUDA sentinel exists" }
}

function Invoke-R7CapturedProcess {
    param([string]$Name, [string]$File, [string[]]$Arguments)
    if ($Name -notmatch "^[a-z0-9-]+$") { throw "invalid evidence name" }
    $r7Info = [System.Diagnostics.ProcessStartInfo]::new()
    $r7Info.FileName = $File
    $r7Info.WorkingDirectory = $r7Repo
    $r7Info.UseShellExecute = $false
    $r7Info.CreateNoWindow = $true
    $r7Info.RedirectStandardOutput = $true
    $r7Info.RedirectStandardError = $true
    $r7Info.Environment.Clear()
    foreach ($r7Pair in $r7Env.GetEnumerator()) {
        $r7Info.Environment[$r7Pair.Key] = [string]$r7Pair.Value
    }
    foreach ($r7Argument in $Arguments) { $null = $r7Info.ArgumentList.Add($r7Argument) }
    $r7Process = [System.Diagnostics.Process]::new()
    $r7Process.StartInfo = $r7Info
    if (-not $r7Process.Start()) { throw "failed to start $Name" }
    $r7Stdout = [System.IO.MemoryStream]::new()
    $r7Stderr = [System.IO.MemoryStream]::new()
    $r7StdoutCopy = $r7Process.StandardOutput.BaseStream.CopyToAsync($r7Stdout)
    $r7StderrCopy = $r7Process.StandardError.BaseStream.CopyToAsync($r7Stderr)
    $r7Process.WaitForExit()
    $r7StdoutCopy.GetAwaiter().GetResult()
    $r7StderrCopy.GetAwaiter().GetResult()
    $r7StdoutBytes = $r7Stdout.ToArray()
    $r7StderrBytes = $r7Stderr.ToArray()
    $r7StdoutPath = Join-Path $r7Run "$Name.stdout.raw"
    $r7StderrPath = Join-Path $r7Run "$Name.stderr.raw"
    [System.IO.File]::WriteAllBytes($r7StdoutPath, $r7StdoutBytes)
    [System.IO.File]::WriteAllBytes($r7StderrPath, $r7StderrBytes)
    $r7StreamRows = @(
        "{0} {1} {2}.stdout.raw" -f
            (Get-FileHash -Algorithm SHA256 -LiteralPath $r7StdoutPath).Hash.ToLowerInvariant(),
            $r7StdoutBytes.Length, $Name
        "{0} {1} {2}.stderr.raw" -f
            (Get-FileHash -Algorithm SHA256 -LiteralPath $r7StderrPath).Hash.ToLowerInvariant(),
            $r7StderrBytes.Length, $Name
    )
    [System.IO.File]::WriteAllLines(
        (Join-Path $r7Run "$Name.streams.sha256"),
        $r7StreamRows,
        [System.Text.UTF8Encoding]::new($false)
    )
    $r7Utf8 = [System.Text.UTF8Encoding]::new($false, $true)
    $r7StdoutText = $r7Utf8.GetString($r7StdoutBytes)
    $r7StderrText = $r7Utf8.GetString($r7StderrBytes)
    [Console]::Out.Write($r7StdoutText)
    [Console]::Error.Write($r7StderrText)
    [pscustomobject]@{
        ExitCode = $r7Process.ExitCode
        Stdout = $r7StdoutText
        Stderr = $r7StderrText
    }
}

$r7RustcId = Invoke-R7CapturedProcess "rustc-id" $r7Rustc @("+nightly-2026-04-07", "-vV")
if ($r7RustcId.ExitCode -ne 0 -or
    $r7RustcId.Stdout -notmatch "rustc 1\.96\.0-nightly \(bcded3316 2026-04-06\)" -or
    $r7RustcId.Stdout -notmatch "commit-hash: bcded331651b60a0383b3ff51db4f24c4495ac53" -or
    $r7RustcId.Stdout -notmatch "host: x86_64-pc-windows-msvc" -or
    $r7RustcId.Stdout -notmatch "LLVM version: 22\.1\.2") {
    throw "pinned rustc identity mismatch"
}
$r7CargoId = Invoke-R7CapturedProcess "cargo-id" $r7Cargo @("+nightly-2026-04-07", "-Vv")
$r7ExpectedCargoId = @(
    "cargo 1.96.0-nightly (888f67534 2026-03-30)",
    "release: 1.96.0-nightly",
    "commit-hash: 888f675344eb1cf2308fd53183e667bdd2c58e51",
    "commit-date: 2026-03-30",
    "host: x86_64-pc-windows-msvc",
    "libgit2: 1.9.2 (sys:0.20.4 vendored)",
    "libcurl: 8.19.0-DEV (sys:0.4.87+curl-8.19.0 vendored ssl:Schannel)",
    "os: Windows 10.0.26200 (Windows 11 Professional) [64-bit]"
) -join "`n"
$r7ExpectedCargoId += "`n"
$r7ObservedCargoId = $r7CargoId.Stdout.Replace("`r`n", "`n").Replace("`r", "`n")
if ($r7CargoId.ExitCode -ne 0 -or $r7CargoId.Stderr.Length -ne 0 -or
    $r7ObservedCargoId -cne $r7ExpectedCargoId) {
    throw "pinned Cargo identity mismatch"
}

$r7OuterResult = Invoke-R7CapturedProcess "outer" $r7Cargo @(
    "+nightly-2026-04-07", "test", "--locked", "--offline", "-j", "7",
    "-p", "neoethos-search", "--no-default-features", "--features",
    "resident-search-slice2-compile-contract", "--test",
    "resident_search_slice2_compile_contract", "--",
    "resident_search_slice2_compile_contract_v9", "--exact", "--nocapture"
)
if ($r7OuterResult.ExitCode -ne 0) { throw "R7 outer contract failed" }
```

The captured Cargo identity output is part of the run receipt and must be
identical on the repeat run. The runner records the sanitized environment key
set and value hashes, config paths/hashes, canonical executables, and detected
C++ compiler identity under `$r7Run`; it emits no raw secret-bearing values.
The complete raw outer stdout/stderr streams are classified and reviewed in
INFO, WARNING, ERROR, then other order. The run root and outer target are not
deleted automatically; evidence collection and any later exact-path cleanup
are separate reviewed actions.

## R7 implementation path allowlist

Only these tracked paths may change in the later R7 implementation commit:

- `crates/neoethos-gpu-cuda/Cargo.toml`;
- `crates/neoethos-gpu-cuda/src/lib.rs`;
- `crates/neoethos-gpu-cuda/src/resident_search_slice2_v3.rs`;
- `crates/neoethos-search/Cargo.toml`;
- `crates/neoethos-search/src/lib.rs`;
- `crates/neoethos-search/tests/resident_search_slice2_compile_contract.rs`;
- `crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml`;
- `crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.lock`;
- `crates/neoethos-search/tests/ui/resident_search_slice2/r7-v9-receipt.sha256`;
- `crates/neoethos-search/tests/ui/resident_search_slice2/api-surface-v3.txt`;
- the exact eleven `.rs` files and ten `.stderr` files listed above.

No root manifest/lock, build script, native source, existing R6 file, existing
Search implementation, prepared-run path, readiness flag, device-fixture path,
vendor file, or ICE receipt is in scope.

## Acceptance

R7 v9 is accepted only when all of the following describe the same committed
bytes:

1. tests-first missing-API RED is captured and cannot be mistaken for an
   expected UI diagnostic;
2. the exact 11-case matrix passes on the pinned host toolchain;
3. metadata, source, hash, lock, no-link, API, and feature-off ratchets pass;
4. all INFO/WARNING/ERROR events are retained and classified, including
   dependency warnings;
5. two independent reviewers return P0/P1/P2 = 0/0/0 on the same commit;
6. `resident_search_v2.rs`, R6, build/native code, root lock, vendor, and the
   ICE receipt remain untouched; and
7. the result is described only as prepared-checkout host compile evidence,
   with fresh-clone portability and CUDA production binding still blocked.
