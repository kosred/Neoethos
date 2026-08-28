# S2 independent Philox oracle candidate

Status: standalone and intentionally **not** wired into Cargo, production
routing, or the CUDA implementation. Integration waits for the GeneView gate.

The fixture answers two independent questions:

1. Does a scalar implementation produce the official D. E. Shaw Research
   Philox4x32-10 known answers and additional boundary answers?
2. Does the proposed S2 logical draw tuple map to the frozen counter/key and
   official Random123 output?

## Files

- `UPSTREAM_RANDOM123_RECEIPT.md`: pinned upstream source identity and hashes.
- `upstream_philox4x32_10_kat.txt`: exact official zero/all-one/non-zero KATs.
- `random123_reference_generator.cpp`: standalone generator using only the
  pinned upstream `Random123/philox.h`.
- `philox4x32_10_boundary_v1.tsv`: five non-zero/boundary counter-key vectors.
- `address_mapping_candidate_v1.tsv`: ten complete logical-address vectors.
- `standalone_oracle_v1.rs`: dependency-free scalar verifier compiled directly
  with `rustc --test`; it has no NeoEthos imports.
- `SHA256SUMS`: immutable handoff hashes for every fixture artifact except the
  checksum file itself.

## Proposed address mapping under test

For tuple `(search_seed, run_identity_sha256, generation, candidate_identity,
operator_identity, decision_slot, rejection_attempt)`:

```text
draw_index = (decision_slot << 32) | rejection_attempt
counter = [candidate_low32, candidate_high32, generation, draw_index_low32]
key = [seed_low32 XOR run_word0_le XOR operator_identity,
       seed_high32 XOR run_word1_le XOR draw_index_high32]
```

The corpus covers high/low candidate words, generation `0x80000000` and
`0xffffffff`, operator separation, decision-slot/retry separation, all-maxima,
and two identities with identical first eight bytes but different remaining
24 bytes.

Important: this candidate V1 mapping binds only the first eight bytes of the
named 32-byte run identity. The `tail_unbound_*` pair deliberately exposes that
fact and produces the same counter/key/output. Do not promote this candidate
mapping as a full-SHA-256 binding without an explicit design decision.

## Standalone verification

From this directory on the VPS:

```sh
/root/.rustup/toolchains/nightly-2026-04-07-x86_64-unknown-linux-gnu/bin/rustc \
  --edition=2024 --test standalone_oracle_v1.rs \
  -o /tmp/s2-independent-philox-oracle-v1
/tmp/s2-independent-philox-oracle-v1 --nocapture
sha256sum -c SHA256SUMS
```

Recompute the generated corpora against the pinned official header:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -pedantic \
  -I/tmp/neoethos-random123-upstream/include \
  random123_reference_generator.cpp \
  -o /tmp/random123_reference_generator
/tmp/random123_reference_generator direct > /tmp/direct.tsv
/tmp/random123_reference_generator address > /tmp/address.tsv
diff -u philox4x32_10_boundary_v1.tsv /tmp/direct.tsv
diff -u address_mapping_candidate_v1.tsv /tmp/address.tsv
```

This fixture is an independent primitive/address oracle only. It does not make
the legacy CPU genetic search an oracle for complete GPU GA semantics.
