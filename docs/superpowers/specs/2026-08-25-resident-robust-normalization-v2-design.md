# Resident Robust Normalization V2 Design

## Scope

Add the semantic-v2 robust-normalization transform to the strict native CUDA
feature-store lane without host feature materialization. The canonical split
is shared Discovery semantics, while Data is the sole authority that may seal
it from an exact pinned/canonical row count and the already-installed runtime
configuration. gpu-cuda may transform only the already-filled final bar-major
store before canonical content hashing.

## Frozen semantics

For each column, use explicitly valid and finite cells in the exact training
range. Sort them with Rust `f64::total_cmp` order, compute the median and
`1.4826 * MAD`, and use population standard deviation over the same sorted
values only when the MAD scale does not exceed
`32 * f64::EPSILON * max(max_abs, 1)`. A scale at or below that floor is
degenerate. Apply `(value - median) / scale` to every valid row, clip to
`[-10, 10]`, preserve existing invalid reasons and canonical NaN payloads, and
convert newly non-finite results to `NonFinite` plus canonical NaN.

## Ownership and flow

1. Data reads the row count only from `PinnedCanonicalSeriesV1` or a sealed
   `CanonicalOhlcvFrame`, reads enabled/disabled only from the process-wide
   configuration installed once at startup, and derives
   `0..floor(0.8 * rows)` internally. No public constructor accepts a range,
   mode, fit value, hash, feature byte, context, or stream.
2. The move-only split travels with Data's phase-zero pinned workspace
   preflight. Once the exact ordered feature schema exists, Data consumes it
   once and freezes row count, range, semantic version, scratch extent and
   fit-metadata extent. The resulting component receipt is then bound to the
   moved run admission's exact context and stream identities.
3. After every producer batch has been packed and retired, gpu-cuda normalizes
   the final bar-major f64/u4 allocation in place, before canonical SHA-256.
4. Scratch is reused in batches of at most 64 columns and lives through the
   normalization-ready event. Six canonical u64 words per column (48 bytes)
   carry training start/end, median bits, scale bits, valid count and the
   degenerate flag. The device writes a semantic-v2 SHA-256 of those canonical
   words into the retired scratch prefix: the versioned domain
   `neoethos.resident-robust-normalization.fit-metadata.semantic-v2\0` followed
   by every column's six words in column order and big-endian u64 encoding.
   The exact event is synchronized
   before one 4-byte verdict and one 32-byte fit digest are read. Feature
   values and validity payloads never cross D2H. Fit metadata and its event
   remain owned by the sealed store through Search-consumer completion, so
   fit bytes are charged once to steady residency rather than only to peak.
5. Disabled mode retains the canonical split identity but has zero padded,
   scratch and fit extents, zero native launches, zero events and zero device
   readback. It uses a versioned in-tree disabled-state digest rather than a
   caller-provided placeholder.

## Deterministic device algorithm

Use a fixed-size, power-of-two scratch segment per active column. Invalid or
padded training cells receive a positive-NaN sentinel ordered after every
finite value. CUDA bitonic compare/swap uses the exact signed-bit transform
implemented by Rust `f64::total_cmp`. One ordered per-column thread performs
the CPU-order mean and variance reductions. The same scratch is then replaced
with absolute deviations, sorted again, and finalized. Strict NVCC precision
flags already disable FMA contraction and preserve precise division/sqrt.
Packed u4 storage is allocated as `round_up(logical_bytes, 4)` and both Rust
and native code require an exact 4-byte multiple and aligned base pointer.
Every packed read and compare-and-swap seed uses an aligned 32-bit atomic load;
no byte load or plain non-atomic word read is authoritative.

## Failure and completion boundary

An uninstalled runtime configuration, empty range, fewer than 64 training
rows, empty holdout, zero valid training cells, valid non-finite training
cells, extent/alignment overflow, context/stream drift, missing or duplicate
event, verdict-before-event, second application, digest mismatch, or runtime
receipt mismatch fails closed. Capability/census authority must not advance
until source, compile and real-device gates validate the connected component.
The current complete-workspace factory still has unresolved producer and
identity receipts, so this slice connects only the honest typed
carrier/adapter and resident component path; the CLI bail remains unchanged.
