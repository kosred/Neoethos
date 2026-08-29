# Resident Search Slice 2 R7 Compile Contract v9 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an exact, dependency-non-CUDA compiler contract for the one
canonical Slice 2 V3 nominal state API, its Search re-export identity, external
opacity, move-only boundary, and Search-façade feature-off absence. The GPU
feature-off gate remains a source/metadata ratchet, not a second compiler case.

**Architecture:** Define the nominal states once in a new GPU production module
compiled by either CUDA or one empty host compile-contract feature. Re-export
those exact types through a declaration-only Search façade. A standalone
nested workspace runs one positive and nine feature-on negative `cargo check`
cases plus one separately targeted feature-off check, while the outer
integration runner verifies compiler JSON, API JSON, metadata, source, lock,
vendor, and no-link receipts.

**Tech Stack:** Rust 2024, Cargo JSON/metadata/rustdoc JSON,
`nightly-2026-04-07`, SHA-256, PowerShell host orchestration.

**Design authority:**
`docs/superpowers/specs/2026-08-29-resident-search-slice2-r7-compile-contract-design-v9.md`

**Authority handoff prerequisite:** this plan is executable only from the
reviewed docs-only v9 authority commit delivered with the design and
`audit/resident-search-slice2-design-v9.sha256`. That commit has sole parent
`6bff01cf2abbf711972f4fcac5348fe14753e5a1` and changes exactly those three
paths. Its commit/tree IDs and the two new-document normalized-LF hashes come
from the v9 handoff; they are not placeholders that an implementer may choose.

---

## File map and hard scope

Create or modify only:

- Modify: `crates/neoethos-gpu-cuda/Cargo.toml`
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Create: `crates/neoethos-gpu-cuda/src/resident_search_slice2_v3.rs`
- Modify: `crates/neoethos-search/Cargo.toml`
- Modify: `crates/neoethos-search/src/lib.rs`
- Create: `crates/neoethos-search/tests/resident_search_slice2_compile_contract.rs`
- Create: `crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml`
- Create: `crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.lock`
- Create: `crates/neoethos-search/tests/ui/resident_search_slice2/r7-v9-receipt.sha256`
- Create: `crates/neoethos-search/tests/ui/resident_search_slice2/api-surface-v3.txt`
- Create: the exact eleven fixture sources and ten same-stem `.stderr` files
  listed in Task 3.

Do not modify the root `Cargo.toml`, root `Cargo.lock`, `build.rs`, native code,
`resident_search_v2.rs`, `resident_search_slice2_admission_v2.rs`, any R6 file,
vendor content, the historical ICE receipt, or an existing v5-v8 design
artifact.

Every Cargo/rustc identity, generate-lockfile, metadata, check, rustdoc, and
outer-test process in this plan uses the exact sanitized launcher/config gate
from the v9 design. Capture stdout and process stderr independently as complete
raw byte streams with byte counts and SHA-256; require valid UTF-8, preserve and
classify INFO/WARNING/ERROR/other by origin, and review the whole streams. Never
merge, truncate, or `tail` them. Process stderr is evidence output and is never
used as a tracked UI `.stderr`; those receipts come only from Cargo JSON
`diagnostic.rendered`.

## Chunk 1: Provenance and tests-first RED

### Task 1: Prove the exact prepared-checkout authority boundary

**Files:** Read only.

- [ ] **Step 1: Verify the isolated repository and source base**

Run:

```powershell
git rev-parse --show-toplevel
git rev-parse HEAD
git rev-parse HEAD^
git show -s --format=%T HEAD
git diff-tree --no-commit-id --name-only -r HEAD
git status --short
```

Expected root ends in
`target/codex-stage/resident-search-novelty-slice2/repo45`; HEAD and its tree
equal the recorded v9 authority handoff; `HEAD^` is exactly
`6bff01cf2abbf711972f4fcac5348fe14753e5a1`; the commit changes exactly the v9
design, this plan, and the v9 SHA manifest; and only the pre-existing `vendor/`
plus `rustc-ice-2026-08-28T23_28_56-10040.txt` are untracked.

- [ ] **Step 2: Verify protected normalized-LF digests**

First reject a BOM or invalid UTF-8, then apply the exact CRLF/bare-CR and
trailing-newline-preserving normalized-LF routine from the v9 design. Verify
every row in `audit/resident-search-slice2-design-v9.sha256`, including the two
new authority documents. The protected expected rows are:

```text
52b166cc52a09358e47e9da3ce1daad5a692783fea820027fb4db491d2b1431a  docs/superpowers/specs/2026-08-28-resident-search-slice2-archive-knn-design.md
e3e876948d98bee1f6f8bdf351011ad3480a40d46505f378da759d0223e25a27  docs/superpowers/plans/2026-08-29-resident-search-slice2-r6-combined-preallocation.md
3f370ffe7561dc26e99b1834b482d0399a188befa2bd68da2b661e771b7de144  audit/resident-search-slice2-design-v5.sha256
413263ceaa403e486ed626571407a1af3f00ce6ee1fe5ebe02504eef1454e443  audit/resident-search-slice2-design-v8.sha256
```

- [ ] **Step 3: Verify the pinned compiler exactly**

Use the sanitized identity launcher from the exact outer command, not an
ambient wrapper. Its rustc observation is equivalent to:

```powershell
rustc +nightly-2026-04-07 -vV
```

Expected commit hash
`bcded331651b60a0383b3ff51db4f24c4495ac53`, release
`1.96.0-nightly`, host `x86_64-pc-windows-msvc`, LLVM `22.1.2`.

Before any generate-lockfile, metadata, check, rustdoc, or outer-test Cargo
invocation, require canonical path
`C:\Users\konst\.cargo\bin\cargo.exe`, empty identity stderr, and exact
LF-normalized `cargo +nightly-2026-04-07 -Vv` stdout including its final LF:

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

The normalized receipt is exactly 337 UTF-8 bytes, SHA-256
`7d4a0723c4202c639b08fdf5a12b01f4cd6eaad342126018e401c6c01ce794a3`.
Retain the raw identity streams separately and exact-match the complete
normalized bytes, not selected fields.

- [ ] **Step 4: Fail closed on missing external patch closure**

Resolve all eleven `[patch.crates-io]` paths in the root manifest and the
nested VectorTA path. Require the exact VectorTA directory and `Cargo.toml`, and
existence/readability only for each other inactive patch directory and its
`Cargo.toml`; do not describe those ten checks as content identity. Stop before
Cargo if any path is absent. Record that `git ls-files vendor` is zero at this
base; never label the result fresh-clone or self-contained.

- [ ] **Step 5: Verify the prepared VectorTA tree identity**

Compute the canonical sorted per-file manifest defined by the v9 design.
Expected 1,077 files and digest:

```text
def4551c993af6e9149c6a93fee1733a43c77629d132d28eee1c1fc16bd224b5
```

Stop if it differs. Do not alter or add vendor files.

### Task 2: Add exact host feature and target topology without the API

**Files:**

- Modify: `crates/neoethos-gpu-cuda/Cargo.toml`
- Modify: `crates/neoethos-search/Cargo.toml`
- Modify: `crates/neoethos-search/src/lib.rs`

- [ ] **Step 1: Add the empty GPU feature**

Add only:

```toml
resident-search-slice2-compile-contract = []
```

Keep the R6, CUDA, device-fixture, and default values byte-equivalent.

- [ ] **Step 2: Make the Search GPU dependency explicitly no-default**

Use:

```toml
neoethos-gpu-cuda = { path = "../neoethos-gpu-cuda", optional = true, default-features = false }
```

- [ ] **Step 3: Add the exact Search feature**

```toml
resident-search-slice2-compile-contract = [
    "dep:neoethos-gpu-cuda",
    "neoethos-gpu-cuda/resident-search-slice2-compile-contract",
]
```

Do not use `gpu-b-adapter`, `gpu-b-native`, `cuda`, or a device fixture.

- [ ] **Step 4: Extend only the two metadata-module cfg gates**

For `canonical_discovery_config_digest_v1` and
`gpu_resident_current_config_plan_v1`, use exactly:

```rust
#[cfg(any(
    test,
    feature = "gpu-b-adapter",
    feature = "resident-search-slice2-compile-contract"
))]
```

Do not add the façade or canonical GPU module yet.

- [ ] **Step 5: Register the exact outer target**

```toml
[[test]]
name = "resident_search_slice2_compile_contract"
path = "tests/resident_search_slice2_compile_contract.rs"
required-features = ["resident-search-slice2-compile-contract"]
```

### Task 3: Create the nested workspace and exact fixture sources

**Files:** Create the nested fixture tree only.

- [ ] **Step 1: Write the exact nested manifest**

Use the non-target manifest from the v9 design, followed by these exact bin
tables in this order:

The fixture compile feature forwards only
`neoethos-search/resident-search-slice2-compile-contract`. Do not add a direct
GPU feature edge: the direct no-default GPU dependency exists only so the
positive source can name GPU paths, and Cargo feature unification must expose
them through Search's forwarding edge.

```toml
[[bin]]
name = "pass_typed_surface"
path = "pass/typed_surface.rs"

[[bin]]
name = "fail_clone_owner_e0599"
path = "fail/clone_owner_e0599.rs"

[[bin]]
name = "fail_copy_owner_e0277"
path = "fail/copy_owner_e0277.rs"

[[bin]]
name = "fail_read_chain_inner_e0616"
path = "fail/read_chain_inner_e0616.rs"

[[bin]]
name = "fail_read_ranked_inner_e0616"
path = "fail/read_ranked_inner_e0616.rs"

[[bin]]
name = "fail_read_staged_inner_e0616"
path = "fail/read_staged_inner_e0616.rs"

[[bin]]
name = "fail_read_pending_inner_e0616"
path = "fail/read_pending_inner_e0616.rs"

[[bin]]
name = "fail_call_staged_constructor_e0624"
path = "fail/call_staged_constructor_e0624.rs"

[[bin]]
name = "fail_construct_ranked_state_e0451"
path = "fail/construct_ranked_state_e0451.rs"

[[bin]]
name = "fail_novelty_receipt_as_full_deadline_e0308"
path = "fail/novelty_receipt_as_full_deadline_e0308.rs"

[[bin]]
name = "fail_feature_gate_off_e0432"
path = "fail/feature_gate_off_e0432.rs"
```

- [ ] **Step 2: Write the positive source**

`pass/typed_surface.rs`:

```rust
use neoethos_gpu_cuda::resident_search_slice2_v3 as gpu;
use neoethos_search::resident_search_slice2_v3 as search;

fn gpu_calibration() -> gpu::ResidentArchiveKnnCalibrationReceiptV2 { panic!() }
fn gpu_chain() -> gpu::ResidentSearchGenerationChainV3 { panic!() }
fn gpu_ranked() -> gpu::ResidentSearchRankEnqueuedV3 { panic!() }
fn gpu_staged() -> gpu::ResidentSearchArchiveStagedV3 { panic!() }
fn gpu_pending() -> gpu::ResidentSearchTerminalPendingV3 { panic!() }
fn gpu_terminal() -> gpu::ResidentSearchTerminalReceiptV3 { panic!() }
fn gpu_try_complete() -> gpu::ResidentSearchTryCompleteV3 { panic!() }
fn gpu_error() -> gpu::ResidentSearchTransitionErrorV3 { panic!() }
fn gpu_rejection() -> gpu::ResidentSearchRejectedAuthorityV3<gpu::ResidentSearchGenerationChainV3> {
    panic!()
}
fn full_deadline() -> search::FullResidentDiscoveryDeadlineReceiptV1 { panic!() }

fn take_calibration(_: search::ResidentArchiveKnnCalibrationReceiptV2) {}
fn take_chain(_: search::ResidentSearchGenerationChainV3) {}
fn take_ranked(_: search::ResidentSearchRankEnqueuedV3) {}
fn take_staged(_: search::ResidentSearchArchiveStagedV3) {}
fn take_pending(_: search::ResidentSearchTerminalPendingV3) {}
fn take_terminal(_: search::ResidentSearchTerminalReceiptV3) {}
fn take_try_complete(_: search::ResidentSearchTryCompleteV3) {}
fn take_error(_: search::ResidentSearchTransitionErrorV3) {}
fn take_rejection(
    _: search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchGenerationChainV3>,
) {}
fn take_deadline(_: search::FullResidentDiscoveryDeadlineReceiptV1) {}

fn main() {
    take_calibration(gpu_calibration());
    take_chain(gpu_chain());
    take_ranked(gpu_ranked());
    take_staged(gpu_staged());
    take_pending(gpu_pending());
    take_terminal(gpu_terminal());
    take_try_complete(gpu_try_complete());
    take_error(gpu_error());
    take_rejection(gpu_rejection());
    take_deadline(full_deadline());

    let _: fn(search::ResidentSearchGenerationChainV3) -> Result<
        search::ResidentSearchRankEnqueuedV3,
        search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchGenerationChainV3>,
    > = search::ResidentSearchGenerationChainV3::enqueue_score_and_rank_v3;
    let _: fn(search::ResidentSearchGenerationChainV3) -> Result<
        search::ResidentSearchTerminalPendingV3,
        search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchGenerationChainV3>,
    > = search::ResidentSearchGenerationChainV3::enqueue_terminal_seal_v3;
    let _: fn(search::ResidentSearchRankEnqueuedV3) -> Result<
        search::ResidentSearchArchiveStagedV3,
        search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchRankEnqueuedV3>,
    > = search::ResidentSearchRankEnqueuedV3::enqueue_stage_archive_from_rank_v3;
    let _: fn(search::ResidentSearchArchiveStagedV3) -> Result<
        search::ResidentSearchGenerationChainV3,
        search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchArchiveStagedV3>,
    > = search::ResidentSearchArchiveStagedV3::enqueue_evolve_and_publish_v3;
    let _: fn(search::ResidentSearchTerminalPendingV3) -> Result<
        search::ResidentSearchTryCompleteV3,
        search::ResidentSearchTransitionErrorV3,
    > = search::ResidentSearchTerminalPendingV3::try_complete_v3;
    let _: fn(
        search::ResidentSearchRejectedAuthorityV3<search::ResidentSearchGenerationChainV3>,
    ) -> (
        search::ResidentSearchTransitionErrorV3,
        search::ResidentSearchGenerationChainV3,
    ) = search::ResidentSearchRejectedAuthorityV3::<
        search::ResidentSearchGenerationChainV3,
    >::into_parts_v3;

    let _ = search::ResidentSearchTryCompleteV3::NotReady;
    let _ = search::ResidentSearchTryCompleteV3::Complete;
}
```

`cargo check` never executes the panic suppliers. They exist only to type-check
nominal identity without a constructor or GPU resource.

- [ ] **Step 3: Write clone and Copy negatives with exact blank lines**

`fail/clone_owner_e0599.rs`:

```rust
use neoethos_search::resident_search_slice2_v3::ResidentSearchGenerationChainV3;

fn chain() -> ResidentSearchGenerationChainV3 {
    loop {}
}

fn main() {
    chain().clone();
}
```

`fail/copy_owner_e0277.rs`:

```rust
use neoethos_search::resident_search_slice2_v3::ResidentSearchGenerationChainV3;

fn chain() -> ResidentSearchGenerationChainV3 {
    loop {}
}

fn require_copy<T: Copy>(_: T) {}

fn main() {
    require_copy(chain());
}
```

- [ ] **Step 4: Write the four exact `inner` opacity negatives**

Use this template without adding or deleting blank lines:

```rust
use neoethos_search::resident_search_slice2_v3::<EXACT_TYPE>;

fn value() -> <EXACT_TYPE> {
    loop {}
}

fn main() {
    let _ = value().inner;
}
```

Replace `<EXACT_TYPE>` once per file with, respectively:

```text
ResidentSearchGenerationChainV3
ResidentSearchRankEnqueuedV3
ResidentSearchArchiveStagedV3
ResidentSearchTerminalPendingV3
```

Use the corresponding exact filenames from the bin table. The primary span is
line 8, columns 21-26 in all four.

- [ ] **Step 5: Write the private-constructor negative**

`fail/call_staged_constructor_e0624.rs`:

```rust
use neoethos_search::resident_search_slice2_v3::{ResidentSearchArchiveStagedV3, ResidentSearchRankEnqueuedV3};

fn ranked() -> ResidentSearchRankEnqueuedV3 {
    loop {}
}

fn main() {
    let _ = ResidentSearchArchiveStagedV3::from_ranked_v3(ranked());
}
```

Expected primary span: line 8, columns 44-58.

- [ ] **Step 6: Write the ranked-state FRU negative**

`fail/construct_ranked_state_e0451.rs`:

```rust
use neoethos_search::resident_search_slice2_v3::ResidentSearchRankEnqueuedV3;

fn ranked() -> ResidentSearchRankEnqueuedV3 {
    loop {}
}

fn main() {
    let _ = ResidentSearchRankEnqueuedV3 { ..ranked() };
}
```

Expected primary span: line 8, columns 46-54.

- [ ] **Step 7: Write the receipt-separation negative**

`fail/novelty_receipt_as_full_deadline_e0308.rs`:

```rust
use neoethos_gpu_cuda::resident_search_slice2_v3::ResidentArchiveKnnCalibrationReceiptV2;
use neoethos_search::resident_search_slice2_v3::FullResidentDiscoveryDeadlineReceiptV1;

fn calibration() -> ResidentArchiveKnnCalibrationReceiptV2 {
    loop {}
}

fn require_full_deadline(_: FullResidentDiscoveryDeadlineReceiptV1) {}

fn main() {
    require_full_deadline(calibration());
}
```

Expected primary span: line 11, columns 27-40.

- [ ] **Step 8: Write the separate feature-off negative**

`fail/feature_gate_off_e0432.rs`:

```rust
use neoethos_search::resident_search_slice2_v3::ResidentSearchGenerationChainV3 as _;

fn main() {}
```

Expected primary span: line 1, columns 22-47. This case is compiled without
the fixture feature.

### Task 4: Implement the runner skeleton and capture the intended RED

**Files:**

- Create: `crates/neoethos-search/tests/resident_search_slice2_compile_contract.rs`

This provisional RED runner deliberately has no tracked `.stderr`, API, hash,
source-uniqueness, or final metadata-receipt preflight yet. It exists only long
enough to prove that the exact compiler executor reaches the missing canonical
API. The committed GREEN runner adds all fail-closed preflights after the real
receipts have been captured in Task 7.

- [ ] **Step 1: Encode the exact case ledger**

Define one constant table with bin, relative source, feature mode, expected
code, line, start column, end column, and stderr path. Assert table length 11,
unique bin/path pairs, exactly one positive, nine feature-on negatives, and one
feature-off negative.

- [ ] **Step 2: Add path and environment guards**

Resolve the repository and fixture roots without shell interpolation. Refuse
symlinks escaping the repository. Create two asserted-nonexistent temporary
target roots under the exact outer target: UI-on and UI-off. Sanitize the exact
environment from the design and install nonexistent CUDA sentinels. Apply the
exact two-file Cargo-config inventory/hash gate before spawning Cargo.

- [ ] **Step 3: Add strict Cargo-JSON parsing**

Parse every stdout line, require `build-finished`, bind package/target/src path,
and count only top-level authored primary spans. Preserve dependency warning
events separately. Capture raw stdout and process stderr independently and
classify the complete streams; keep process stderr distinct from UI receipts.
Apply only the four permitted diagnostic normalization classes.

- [ ] **Step 4: Add only the provisional exact case executor**

Implement exact-bin, exact-feature-mode execution and make any result other than
the frozen case expectation a harness failure. For this RED only, select the
positive first and require that its missing-API diagnostic is reported as a
mismatch. Do not add receipt completeness, API/source, final metadata, or hash
guards yet; any such preflight would make the intended compiler RED
unreachable.

- [ ] **Step 5: Generate the nested lock offline**

Run once, without CUDA and without updating the root lock:

```powershell
cargo +nightly-2026-04-07 generate-lockfile --manifest-path crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml --offline
```

Review both complete raw streams. Verify the nested lock resolves the local
patched VectorTA package without `source`/`checksum` and does not change root
`Cargo.lock`.

- [ ] **Step 6: Run the exact outer command before adding the API**

Use the exact sanitized/captured command from the v9 design with a fresh outer
target. Expected: the executor reaches the positive compiler invocation, sees
the canonical module/import missing, rejects that diagnostic because success
was expected, and makes the outer harness fail. A preflight/placeholder failure
does not count. No expected negative UI error is allowed to satisfy this RED.
Persist both raw outer streams, child streams, hashes, and classified event
ledger as the tests-first receipt.

- [ ] **Step 7: Stop on any CUDA/native activation**

If metadata or logs show `gpu-b-native`, `cuda`, `cuda-device-fixtures`,
`cuda-build-native`, `cust`, `cust_raw`, `find_cuda_helper`, `nvcc`,
`cuobjdump`, CUDA libraries, or CUDA link paths, stop and repair topology before
adding production declarations.

## Chunk 2: Minimal canonical API and exact GREEN

### Task 5: Add the one canonical nominal module

**Files:**

- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Create: `crates/neoethos-gpu-cuda/src/resident_search_slice2_v3.rs`

- [ ] **Step 1: Register the module under the exact one-time cfg**

Use the exact declaration in the v9 design. Do not add a dead-code allow at
crate or module scope.

- [ ] **Step 2: Declare exactly the nine public types**

Each opaque type has only private `inner`; rejection has only private `error`
and `authority`. Use private uninhabited internal representations so no owner
or calibration receipt can be minted yet. Do not derive or implement Clone,
Copy, Default, Deref, AsRef, Borrow, From, or Into locally. The API parser
ignores standard-library reflexive blanket impls by origin but rejects every
explicit local impl and every non-identity state conversion.

- [ ] **Step 3: Implement only the six public signatures**

Use fail-closed bodies over the uninhabited private internals. The methods must
not allocate, call FFI, touch a device, set readiness, return fabricated
success, or create a runnable owner.

- [ ] **Step 4: Add the one crate-private staged constructor**

Use exact name, argument, return type, and visibility
`pub(crate) fn from_ranked_v3(...)`. Add no other constructor or sealer.

- [ ] **Step 5: Format only the touched Rust files**

Run rustfmt directly with the pinned toolchain on the exact new/modified Rust
files. Inspect the entire diff; do not run a bulk workspace rewrite.

### Task 6: Add the pure Search façade

**Files:**

- Modify: `crates/neoethos-search/src/lib.rs`

- [ ] **Step 1: Add the exact inline façade**

Copy the exact cfg and two `pub use` declarations from the v9 design. Do not
define a local type or wrapper.

- [ ] **Step 2: Prove source uniqueness before Cargo**

Use `rg` to count every public V3 type definition across both crates. Expected:
one each, all in `resident_search_slice2_v3.rs`. The Search façade may contain
only re-export occurrences.

- [ ] **Step 3: Prove protected production files are unchanged**

Run:

```powershell
git diff --exit-code 6bff01cf -- crates/neoethos-gpu-cuda/src/resident_search_v2.rs crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs crates/neoethos-gpu-cuda/build.rs Cargo.toml Cargo.lock
```

Expected exit 0 and no output.

### Task 7: Capture exact diagnostics and receipts

**Files:**

- Create/finalize: ten `.stderr` files
- Create/finalize: `api-surface-v3.txt`
- Create/finalize: `r7-v9-receipt.sha256`

- [ ] **Step 1: Run the positive alone in a fresh feature-on target**

Use the exact case command for `pass_typed_surface`. Expected success, final
`build-finished.success=true`, and zero selected-package warning/error
diagnostics. Preserve and classify all dependency warnings.

- [ ] **Step 2: Run each feature-on negative alone in table order**

For each, require nonzero rustc result, `build-finished.success=false`, and the
one exact top-level primary code/span. Stop at the first mismatch. Never bless
an unresolved import or dependency error.

- [ ] **Step 3: Normalize and write the ten stderr receipts**

Use only CRLF, separator, exact-root, and color handling from the design.
Review each full rendered diagnostic before accepting it. Do not normalize
compiler wording, lines, columns, labels, or secondary definition spans. These
tracked files are derived from JSON `diagnostic.rendered`, never from the Cargo
process's separately captured stderr stream.

- [ ] **Step 4: Run the feature-off case in its separate fresh target**

Omit `--features`. Expected exactly one authored primary `E0432` at line 1,
columns 22-47 through the Search façade. Any success, warning-only failure,
GPU-crate absence error, or second primary error fails.

- [ ] **Step 5: Produce rustdoc JSON in the third fresh target**

Run exactly, with the third fresh target in `CARGO_TARGET_DIR`:

```powershell
cargo +nightly-2026-04-07 rustdoc --locked --offline -j 7 -p neoethos-gpu-cuda --lib --no-default-features --features resident-search-slice2-compile-contract --message-format=json --color never -- -Dwarnings -Z unstable-options --output-format json
cargo +nightly-2026-04-07 rustdoc --locked --offline -j 7 -p neoethos-search --lib --no-default-features --features resident-search-slice2-compile-contract --message-format=json --color never -- -Dwarnings -Z unstable-options --output-format json
```

Require exactly `doc/neoethos_gpu_cuda.json` and
`doc/neoethos_search.json`. Normalize the filtered rows and write
`api-surface-v3.txt`. Compare against the exact allowlist; reject extra public
children or banned impl families. Preserve and classify both commands' complete
raw stdout/stderr streams.

- [ ] **Step 6: Finalize the raw SHA ledger**

Hash nested manifest/lock, 11 sources, 10 stderr files, API receipt, canonical
GPU module, and Search façade source. Write sorted lowercase raw SHA-256 rows to
`r7-v9-receipt.sha256`. The runner must recompute every row and reject missing,
extra, duplicate, malformed, or placeholder entries.

- [ ] **Step 7: Install the final fail-closed GREEN preflight**

Only after Steps 1-6 have produced reviewed real receipts, replace the
provisional RED path with the committed preflight. Before compiler cases it
verifies the authority commit/manifest, protected hashes and root-lock hash,
exact fixture target inventory, nested lock/receipt completeness, exact
VectorTA tree digest and metadata origin, existence-only status of the other ten
inactive root patch manifests, source uniqueness/private shape, exact API
receipt, Search-to-GPU feature forwarding, feature-on/off resolve closures,
Cargo config/environment identities, and no-CUDA link evidence. Missing or
drifting data is a harness failure, never a skip. Then rerun the positive and
all ten negatives through the same executor and require the frozen outcomes.

### Task 8: Run topology, no-link, and full harness verification

**Files:** Read only after receipts are frozen.

- [ ] **Step 1: Inspect feature-on nested metadata**

Run:

```powershell
cargo +nightly-2026-04-07 metadata --manifest-path crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml --locked --offline --format-version 1 --no-default-features --features resident-search-slice2-compile-contract
```

Check the active resolve graph, not lockfile substrings. Expected VectorTA
`source=null`, exact vendor manifest path, `nightly-avx` present,
`cuda-build-native` absent, and no active cust family. Prove that the GPU
compile feature is reached through Search's forwarding edge; the fixture has no
direct GPU feature edge.

- [ ] **Step 2: Inspect feature-off nested metadata**

Run:

```powershell
cargo +nightly-2026-04-07 metadata --manifest-path crates/neoethos-search/tests/ui/resident_search_slice2/Cargo.toml --locked --offline --format-version 1 --no-default-features
```

Expected both compile-contract features absent and the same no-CUDA closure.
The compiler claim here is only the Search-façade `E0432`; GPU-module absence is
an exact source/metadata-gate ratchet.

- [ ] **Step 3: Verify the exact target inventory**

Metadata must report exactly 11 `bin` targets with the exact source paths and
no lib/test/example/bench target. `autolib`, all four other `auto*` flags, and
resolver/member identity must match the design.

Also run:

```powershell
cargo +nightly-2026-04-07 metadata --manifest-path crates/neoethos-search/Cargo.toml --locked --offline --format-version 1 --no-deps
```

Require the outer target's path/name/kind and
`required-features=["resident-search-slice2-compile-contract"]` exactly.

- [ ] **Step 4: Run the exact outer command in a new target**

Expected one outer test passed, 0 failed. The runner reports 1 positive, 9
feature-on negatives, 1 feature-off negative, exact API/metadata/source/hash
ratchets, and no CUDA edge. Read the complete INFO/WARNING/ERROR output; do not
use `tail` and do not call the whole log warning-free if dependencies warned.
Build-script JSON proves no CUDA link library/search path. Non-attempt of
`nvcc`/`cuobjdump` is the bounded source-gate/fresh-target/sentinel inference
defined by the design, not a claim that Cargo JSON traces arbitrary processes.

- [ ] **Step 5: Repeat from another new outer target**

Expected identical normalized receipts and counts. A target-cache-dependent
result is a failure.

- [ ] **Step 6: Recheck root lock and protected hashes**

Root lock raw SHA remains
`725cc6fb8645a0d7d9cd11f32bab01dcc8cc3de0497a9df5472886e20eb2167f`.
The four protected normalized-LF digests from Task 1 remain exact.

## Chunk 3: Independent review and atomic handoff

### Task 9: Obtain two independent zero-finding reviews

**Files:** Exact final diff and complete logs.

- [ ] **Step 1: Dispatch a spec/API reviewer**

Ask the reviewer to compare the final bytes against the v9 design: nominal
identity, public/private API, exact 11 cases/spans, feature-off proof, no
mirrors, and non-claims. Require P0/P1/P2 counts.

- [ ] **Step 2: Dispatch a Cargo/false-green reviewer**

Ask the reviewer to attack package/target/path binding, JSON parsing,
normalization, target freshness, feature unification, nested patch/lock,
untracked vendor provenance, no-link evidence, warning accounting, and allowed
path scope. Require P0/P1/P2 counts.

- [ ] **Step 3: Repair and repeat on the same bytes**

Any P1/P2 reopens implementation. After a repair, rerun all exact verification
and both reviews. Approval from an earlier byte set does not transfer.

### Task 10: Commit only the exact R7 boundary

**Files:** The allowlist from this plan only.

- [ ] **Step 1: Inspect status and staged paths**

Verify `vendor/` and the ICE file remain untracked/untouched. Stage every and
only allowed R7 path. Do not stage logs outside the approved receipt paths.

- [ ] **Step 2: Run verification-before-completion**

Repeat the exact outer harness, protected hashes, root-lock hash, source
uniqueness, and `git diff --check` against the exact staged bytes.

- [ ] **Step 3: Commit atomically**

Suggested subject:

```text
test(search): bind exact R7 compile opacity contract
```

Record commit/tree hashes, exact command, test counts, dependency warning
counts, vendor/provisioning status, and both 0/0/0 reviews.

- [ ] **Step 4: Report the bounded result honestly**

State: prepared-checkout host compiler contract proven. State separately:
CUDA production binding, device execution, runtime ownership, R8/R9, readiness,
and fresh-clone/self-contained portability remain unproven or blocked. Do not
start a paid GPU instance for R7.
