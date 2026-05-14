# Rust Ecosystem Reference — Audit Cross-Check

Research conducted 2026-05-14 against authoritative upstream sources (GitHub raw content for
official crates / `rust-lang/rust` stdlib / `rust-lang/nomicon`). docs.rs and
doc.rust-lang.org HTML pages were not directly reachable from this sandbox; in every case the
underlying source-of-truth (rustdoc comments in the crate's `lib.rs` / `request.rs` / `atomic.rs`,
or the nomicon's markdown source) was fetched at the tagged release.

Crate / source versions resolved live from `crates.io` API:

| Crate     | Latest stable | Released   | Source pulled                                                  |
|-----------|---------------|------------|----------------------------------------------------------------|
| reqwest   | 0.13.3        | 2026-04-27 | `seanmonstar/reqwest @ v0.13.3`                                |
| tokio     | 1.52.3        | 2026-05-08 | `tokio-rs/tokio` (Send/Sync semantics unchanged in 1.x)        |
| zeroize   | 1.8.2         | 2025-09-29 | `RustCrypto/utils/zeroize @ master` (HEAD == 1.8.2)            |
| secrecy   | (compared)    | n/a        | `iqlusioninc/crates/secrecy @ main`                            |
| rand_distr| 0.6.0         | 2026-02-10 | `rust-random/rand_distr @ master` (HEAD == 0.6.0)              |
| std (Rust)| edition 2024  | n/a        | `rust-lang/rust @ master` (`library/std/src/env.rs`, `library/core/src/sync/atomic.rs`) |
| Nomicon   | latest        | n/a        | `rust-lang/nomicon @ master` (`src/atomics.md`)                |

---

## 1. `reqwest` — OAuth 2.0 token exchange

### Latest stable

`reqwest 0.13.3`, published 2026-04-27 (`crates.io` API, `max_stable_version`).
Repo: <https://github.com/seanmonstar/reqwest>, source ref `v0.13.3`.

Note: starting in `v0.13.0`, the `form` Cargo feature became **opt-in** (per the v0.13.0
"Breaking changes" block in `CHANGELOG.md`):

> - `query` and `form` are now crate features, disabled by default.

So `Cargo.toml` must list `reqwest = { version = "0.13", features = ["form", "json"] }`.

### Correct API for form-encoded POST

From `src/async_impl/request.rs` lines 396-448 (rustdoc + impl):

> Send a form body.
>
> Sets the body to the url encoded serialization of the passed value,
> and also sets the `Content-Type: application/x-www-form-urlencoded`
> header.

```rust
#[cfg(feature = "form")]
#[cfg_attr(docsrs, doc(cfg(feature = "form")))]
pub fn form<T: Serialize + ?Sized>(mut self, form: &T) -> RequestBuilder {
    // ...
    match serde_urlencoded::to_string(form) {
        Ok(body) => {
            req.headers_mut()
                .entry(CONTENT_TYPE)
                .or_insert(HeaderValue::from_static(
                    "application/x-www-form-urlencoded",
                ));
            *req.body_mut() = Some(body.into());
        }
        // ...
    }
}
```

Source URL: <https://github.com/seanmonstar/reqwest/blob/v0.13.3/src/async_impl/request.rs#L396-L448>

So the **idiomatic, authoritative pattern** for an OAuth token exchange is:

```rust
let resp = client.post(token_url)
    .form(&[
        ("grant_type",   "authorization_code"),
        ("code",         code),
        ("client_id",    client_id),
        ("client_secret", client_secret),
        ("redirect_uri", redirect_uri),
    ])
    .send().await?;
```

A slice of `(&str, &str)` tuples implements `Serialize`, so it goes through
`serde_urlencoded::to_string`, producing a properly percent-encoded body. **Do not** assemble a
custom `body(...)` string — that bypasses encoding (`&` / `+` / `=` in secrets break the wire
format and may leak via panic messages).

### Logging-leak concern

This is **a real, recently-fixed bug** in reqwest. The v0.13.3 CHANGELOG.md entry
(<https://github.com/seanmonstar/reqwest/blob/v0.13.3/CHANGELOG.md>) explicitly says:

> ## v0.13.3
> - Fix logging in resolver to only show host, not full URL.

Diffing `src/connect.rs` between v0.13.2 and v0.13.3 confirms the fix:

```
- v0.13.2 line 784: log::debug!("proxy({proxy:?}) intercepts '{dst:?}'");
+ v0.13.3 line 784: log::debug!("proxy({proxy:?}) intercepts '{:?}'", dst.host());

- v0.13.2 line 929: log::debug!("starting new connection: {dst:?}");
+ v0.13.3 line 929: log::debug!("starting new connection '{:?}'", dst.host());
```

URL `Display`/`Debug` includes path **and** query string. For an OAuth flow, however, the
audit's specific concern — `client_secret` in the URL — only applies if someone wrongly
constructs a `GET` with `?client_secret=...`. For a POST with `.form(...)`, the secret is
in the body, and reqwest never `log::debug!`s the body. **Action:** require reqwest >= 0.13.3
and confirm the codebase uses POST + `.form(...)` (not GET with query params).

**Verdict:** Audit recommendation to use `.form(&[...])` is correct. The logging-leak risk
is **real but mitigated** by (a) using POST + body and (b) pinning reqwest >= 0.13.3.

---

## 2. `std::sync::atomic::Ordering` — variants and Relaxed semantics

Source: `rust-lang/rust/library/core/src/sync/atomic.rs`
(<https://github.com/rust-lang/rust/blob/master/library/core/src/sync/atomic.rs>).

### Variant docs (lines 447-507, verbatim rustdoc)

> Memory orderings specify the way atomic operations synchronize memory.
> In its weakest `Ordering::Relaxed`, only the memory directly touched by the
> operation is synchronized.

- `Relaxed`: "No ordering constraints, only atomic operations. Corresponds to `memory_order_relaxed` in C++20."
- `Release`: "When coupled with a store, all previous operations become ordered before any load of this value with `Acquire` (or stronger) ordering."
- `Acquire`: "When coupled with a load, if the loaded value was written by a store operation with `Release` (or stronger) ordering, then all subsequent operations become ordered after that store."
- `AcqRel`: "Has the effects of both `Acquire` and `Release` together... This ordering is only applicable for operations that combine both loads and stores."
- `SeqCst`: "Like `Acquire`/`Release`/`AcqRel`... with the additional guarantee that all threads see all sequentially consistent operations in the same order."

### The canonical std example uses `Relaxed` for a counter (atomic.rs lines 223-236)

```rust
//! Keep a global count of live threads:
//!
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! static GLOBAL_THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);
//!
//! // Note that Relaxed ordering doesn't synchronize anything
//! // except the global thread counter itself.
//! let old_thread_count = GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::Relaxed);
```

Source URL: <https://github.com/rust-lang/rust/blob/master/library/core/src/sync/atomic.rs#L223-L236>

This is the **official std-library-endorsed pattern** for a global monotonically-increasing
counter — exactly the `client_order_id` use case. `Relaxed` is correct.

### Nomicon corroboration

`rust-lang/nomicon/src/atomics.md` lines 223-232
(<https://github.com/rust-lang/nomicon/blob/master/src/atomics.md>):

> ## Relaxed
>
> Relaxed accesses are the absolute weakest. They can be freely re-ordered and
> provide no happens-before relationship. **Still, relaxed operations are still
> atomic. That is, they don't count as data accesses and any read-modify-write
> operations done to them occur atomically.** Relaxed operations are appropriate for
> things that you definitely want to happen, but don't particularly otherwise care
> about. For instance, **incrementing a counter can be safely done by multiple
> threads using a relaxed `fetch_add` if you're not using the counter to
> synchronize any other accesses.**

(emphasis added)

**Verdict:** Audit F-APP3-001 is **incorrect** in claiming `Relaxed` is unsafe for a
unique-id counter. `Ordering::Relaxed` on `fetch_add` guarantees:
1. The increment is atomic (no torn reads/writes).
2. Each `fetch_add` returns a distinct prior value (the operation is linearizable / has a
   total modification order).
3. No memory ordering with respect to *other* memory locations is guaranteed — irrelevant
   for a self-contained counter.

`SeqCst` would impose a global total order on this operation versus all other `SeqCst`
operations, which is overkill and a measurable perf hit on weakly-ordered hardware
(ARM, RISC-V). The official std example uses `Relaxed` for exactly this pattern.

---

## 3. `zeroize` — proper usage for secrets

### Latest stable

`zeroize 1.8.2`, published 2025-09-29 (`crates.io` API).
Repo: <https://github.com/RustCrypto/utils/tree/master/zeroize>.

### Is `Zeroizing<String>` correct?

From `zeroize/src/lib.rs` (master == 1.8.2) lines 186-209
(<https://github.com/RustCrypto/utils/blob/master/zeroize/src/lib.rs>):

> ## Stack/Heap Zeroing Notes
>
> However, be aware several operations in Rust can unintentionally leave
> copies of data in memory. This includes but is not limited to:
>
> - Moves and `Copy`
> - **Heap reallocation when using `Vec` and `String`**
> - Borrowers of a reference making copies of the data
>
> The `Zeroize` impls for `Vec`, `String` and `CString` zeroize the entire
> capacity of their backing buffer, but cannot guarantee copies of the data
> were not previously made by buffer reallocation. It's therefore important
> when attempting to zeroize such buffers to initialize them to the correct
> capacity, and take care to prevent subsequent reallocation.
>
> **The `secrecy` crate provides higher-level abstractions for eliminating
> usage patterns which can cause reallocations:**
>
> <https://crates.io/crates/secrecy>

So `zeroize::Zeroizing<String>` **works but has a known pitfall**: any operation on the
inner `String` that causes a `realloc` (e.g. `push_str` past capacity) leaves a copy of the
old buffer on the heap that is never zeroed. The upstream maintainers themselves point at
the `secrecy` crate for higher-level safety.

### `secrecy::SecretString` comparison

From `iqlusioninc/crates/secrecy/src/lib.rs`
(<https://github.com/iqlusioninc/crates/blob/main/secrecy/src/lib.rs>) lines 210-215:

```rust
/// Secret string type.
///
/// This is a type alias for [`SecretBox<str>`] which supports some helpful trait impls.
///
/// Notably it has a [`From<String>`] impl which is the preferred method for construction.
pub type SecretString = SecretBox<str>;
```

Crucially this is `SecretBox<str>` — a `Box<str>`, **not** a `String`. A `Box<str>` cannot
reallocate (no extra capacity). It also implements `Debug` as `[REDACTED]` (line 144) and
opts out of `serde::Serialize` by default to prevent accidental exfiltration.

### Recommendation

For API keys / OAuth secrets / passwords stored as immutable strings:

- **Prefer `secrecy::SecretString` (`SecretBox<str>`)** over `zeroize::Zeroizing<String>`.
  - No realloc risk (Box<str> has no spare capacity).
  - `Debug` is redacted by default.
  - `expose_secret()` makes every access site grep-auditable.
- `zeroize::Zeroizing<[u8; N]>` (fixed-size buffer) is fine for cryptographic key material
  on the stack.
- `zeroize::Zeroizing<String>` is acceptable **only** if the secret is set once and never
  mutated after construction — and the audit should verify no `push_str` / `+=` operations.

**Verdict:** The audit recommendation (use zeroize) is **on the right track but should be
refined** to recommend `secrecy::SecretString` for `String`-backed secrets, with
`zeroize::Zeroizing` reserved for fixed-size buffers.

---

## 4. `std::env::set_var` safety (Rust 2024 edition)

Source: `rust-lang/rust/library/std/src/env.rs` lines 301-363
(<https://github.com/rust-lang/rust/blob/master/library/std/src/env.rs>).

### Verbatim `# Safety` section

> ## Safety
>
> This function is safe to call in a single-threaded program.
>
> This function is also always safe to call on Windows, in single-threaded
> and multi-threaded programs.
>
> **In multi-threaded programs on other operating systems, the only safe option is
> to not use `set_var` or `remove_var` at all.**
>
> The exact requirement is: you must ensure that there are no other threads concurrently
> writing or **reading**(!) the environment through functions or global variables other
> than the ones in this module. The problem is that these operating systems do not provide
> a thread-safe way to read the environment, and most C libraries, including libc itself,
> do not advertise which functions read from the environment. Even functions from the Rust
> standard library may read the environment without going through this module, e.g. for
> DNS lookups from `std::net::ToSocketAddrs`. No stable guarantee is made about which
> functions may read from the environment in future versions of a library. All this makes
> it not practically possible for you to guarantee that no other thread will read the
> environment, so the only safe option is to not use `set_var` or `remove_var` in
> multi-threaded programs at all.

The function signature itself confirms the 2024 edition deprecation-to-unsafe:

```rust
#[rustc_deprecated_safe_2024(
    audit_that = "the environment access only happens in single-threaded code"
)]
#[stable(feature = "env", since = "1.0.0")]
pub unsafe fn set_var<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
```

### Safe alternatives

The std docs explicitly suggest:

> To pass an environment variable to a child process, you can instead use `Command::env`.

Pragmatic alternatives for our codebase:

1. **Set env vars before any threads spawn.** In `main()`, before `tokio::runtime` /
   ratatui startup. This still requires `unsafe { ... }` but is sound.
2. **Don't use process env for config at all.** Pass values via a `Config` struct injected
   into the relevant modules. This is the *only* fully-safe option per the std docs ("the
   only safe option is to not use `set_var` ... at all").
3. **For subprocess env injection**, use `std::process::Command::env(key, value)` —
   this is fully safe and does not touch the parent process's env.

For a TUI app that mutates env mid-flight (the F-CLI2-001 scenario): there is **no fully
safe answer**. The only sound pattern is to:
  - Collect all required env vars up front in `main()` before spawning anything.
  - Or move that config to a non-env channel (file, in-memory `Arc<RwLock<Config>>`, ...).

**Verdict:** Audit F-CLI2-001 is **correct**. The recommended fix is *not* "wrap in
`unsafe`" — that just silences the compiler. The fix is to **stop mutating env at runtime**
and pass config through an internal channel.

---

## 5. `rand_distr::Normal` vs hand-rolled Box-Muller

### Latest stable

`rand_distr 0.6.0`, published 2026-02-10 (`crates.io` API). MSRV 1.85, Edition 2024.
Repo: <https://github.com/rust-random/rand_distr>.

### Algorithm in upstream `Normal`

From `src/normal.rs` lines 18-101
(<https://github.com/rust-random/rand_distr/blob/master/src/normal.rs#L18-L101>):

> The standard Normal distribution `N(0, 1)`.
>
> This is equivalent to `Normal::new(0.0, 1.0)`, but faster.
> [...]
> ## Notes
> Implemented via the ZIGNOR variant[^1] of the Ziggurat method.
>
> [^1]: Jurgen A. Doornik (2005). *An Improved Ziggurat Method to Generate Normal Random
>       Samples*. Nuffield College, Oxford

This is **not Box-Muller** — it uses the Ziggurat method, which:

- Samples from the true `N(0,1)` density (no log-domain blow-up).
- Has a correctly-handled tail via the `zero_case` rejection loop (lines 67-90), so very
  large samples occur with the correct (vanishingly small) probability — not the corrupted
  spike that a Box-Muller implementation produces when `u1` rounds to zero and `sqrt(-2 *
  ln(u1))` explodes.
- The 0.6.0 CHANGELOG additionally lists fixes for related distributions:
  > - Fix panic in `FisherF::new` on almost zero parameters
  > - Fix panic in `NormalInverseGaussian::new` with very large `alpha`; this is a
  >   Value-breaking change
  > - Fix hang and debug assertion in `Zipf::new` on invalid parameters

`Normal::new(0.0, 1.0).unwrap()` is the canonical API. Per the rustdoc example:

```rust
use rand_distr::{Normal, Distribution};
let normal = Normal::new(2.0, 3.0).unwrap();
let v = normal.sample(&mut rand::rng());
```

For a `StandardNormal` (μ=0, σ=1) sampler, the slightly faster path is:

```rust
use rand_distr::StandardNormal;
let val: f64 = rand::rng().sample(StandardNormal);
```

### Verdict on F-MODELS7-001

The 30+ sigma outliers in CRFMNES are a **classic Box-Muller pathology** caused by `u1`
underflowing to a near-zero `f64`, making `sqrt(-2 * ln(u1))` produce values like 38σ that
should occur with probability `~10^-316`. A "tighter clamp" on the output is a **bandaid**
that distorts the tail distribution. The correct fix is to **replace the hand-rolled
Box-Muller with `rand_distr::StandardNormal` (or `Normal::new(0.0, 1.0).unwrap().sample(rng)`)**,
which uses the Ziggurat method and has correct tails by construction.

**Verdict:** Audit recommendation (tighter clamp) is **wrong / incomplete**. Replace
Box-Muller wholesale with `rand_distr::StandardNormal` — there is zero performance reason
to keep a hand-rolled implementation.

---

## 6. `fetch_add` semantics — focused re-verification

Source: `library/core/src/sync/atomic.rs` lines 3134-3163
(<https://github.com/rust-lang/rust/blob/master/library/core/src/sync/atomic.rs#L3134>):

```rust
/// Adds to the current value, returning the previous value.
///
/// This operation wraps around on overflow.
///
/// `fetch_add` takes an [`Ordering`] argument which describes the memory ordering
/// of this operation. All ordering modes are possible. Note that using
/// [`Acquire`] makes the store part of this operation [`Relaxed`], and
/// using [`Release`] makes the load part [`Relaxed`].
///
/// **Note**: This method is only available on platforms that support atomic operations on
/// `<integer type>`.
pub fn fetch_add(&self, val: $int_type, order: Ordering) -> $int_type {
    // SAFETY: data races are prevented by atomic intrinsics.
    unsafe { atomic_add(self.as_ptr(), val, order) }
}
```

The C++20 memory model (which Rust mirrors verbatim per atomic.rs line 437-438:
"Rust's memory orderings are the same as those of C++20") defines a **single total modification
order** for every atomic object, *regardless of memory ordering*. Each `fetch_add` reads the
value at one point in that total order and writes its successor — so two concurrent
`fetch_add(1, Relaxed)` calls cannot return the same value. The "ordering" argument
controls visibility of *other* memory accesses around the atomic — not the atomicity of the
operation itself.

This was already established in §2 via the nomicon quote ("read-modify-write operations
done to them occur atomically"). Re-confirmed.

**Verdict:** F-APP3-001 is **invalidated**. `Ordering::Relaxed` on `fetch_add` for a
unique-counter pattern is correct, idiomatic, and matches the official std example.

---

## Final Summary Table

| Audit Finding                               | Recommended Patch                                      | Docs Verdict | Action                                                                                                              |
|---------------------------------------------|--------------------------------------------------------|--------------|---------------------------------------------------------------------------------------------------------------------|
| F-APP3-001 (Ordering::Relaxed on counter)   | Change to SeqCst                                        | **No**       | **Revert.** Keep `Relaxed` — matches the official std example for `GLOBAL_THREAD_COUNT` and is endorsed by the nomicon. |
| F-CORE2-027 (API key plaintext in memory)   | Wrap in `zeroize::Zeroizing<String>`                   | **Revise**   | **Refine.** Use `secrecy::SecretString` (`SecretBox<str>`) instead — avoids `String` realloc pitfall called out by the zeroize upstream docs. Reserve `zeroize::Zeroizing<[u8; N]>` for fixed-size buffers. |
| F-MODELS7-001 (Box-Muller 30σ outliers)     | Tighter clamp on Box-Muller output                     | **No**       | **Revise.** Replace hand-rolled Box-Muller with `rand_distr::StandardNormal` (Ziggurat method, correct tails by construction). Clamping is a bandaid that distorts the distribution. |
| F-CLI2-001 (`unsafe { env::set_var }`)      | Add `unsafe` wrapper                                   | **Revise**   | **Refine.** Per std docs, on Linux/macOS the *only* safe option is to **never** mutate env in a multi-threaded program. Either move all env mutation to `main()` before any thread spawns, or remove env-mutation entirely and use an in-memory `Arc<RwLock<Config>>` / pass via `Command::env` for subprocesses. |
| (Implicit) reqwest URL-leak via debug logs  | Audit OAuth POST construction                          | **Yes**      | **Keep.** Pin `reqwest >= 0.13.3` (changelog: "Fix logging in resolver to only show host, not full URL") and ensure token exchanges use `.post(url).form(&[...])` so secrets travel in the body — reqwest never logs the body. Reject any GET-with-query OAuth code. |

---

## Citations index

- reqwest `RequestBuilder::form` source: <https://github.com/seanmonstar/reqwest/blob/v0.13.3/src/async_impl/request.rs#L396-L448>
- reqwest CHANGELOG (v0.13.3 logging fix): <https://github.com/seanmonstar/reqwest/blob/v0.13.3/CHANGELOG.md>
- reqwest connect.rs (logging call sites): <https://github.com/seanmonstar/reqwest/blob/v0.13.3/src/connect.rs>
- std `Ordering` enum + `Relaxed` example: <https://github.com/rust-lang/rust/blob/master/library/core/src/sync/atomic.rs#L223-L508>
- std `fetch_add` rustdoc: <https://github.com/rust-lang/rust/blob/master/library/core/src/sync/atomic.rs#L3134-L3163>
- Nomicon, "Relaxed" section: <https://github.com/rust-lang/nomicon/blob/master/src/atomics.md#L223-L232>
- std `env::set_var` Safety section: <https://github.com/rust-lang/rust/blob/master/library/std/src/env.rs#L301-L363>
- zeroize realloc caveat + secrecy pointer: <https://github.com/RustCrypto/utils/blob/master/zeroize/src/lib.rs#L186-L209>
- secrecy `SecretString = SecretBox<str>` definition: <https://github.com/iqlusioninc/crates/blob/main/secrecy/src/lib.rs#L210-L215>
- rand_distr `Normal` / `StandardNormal` (Ziggurat): <https://github.com/rust-random/rand_distr/blob/master/src/normal.rs#L18-L101>
- rand_distr 0.6.0 CHANGELOG: <https://github.com/rust-random/rand_distr/blob/master/CHANGELOG.md>
