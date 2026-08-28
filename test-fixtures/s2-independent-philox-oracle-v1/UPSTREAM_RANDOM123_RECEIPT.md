# Upstream Random123 receipt

This receipt pins the independent source used to generate the S2 Philox4x32-10
goldens. No NeoEthos Rust or CUDA helper was used.

- Official repository: <https://github.com/DEShawResearch/random123>
- Context7 library identity: `/deshawresearch/random123`
- Commit: `9545ff6413f258be2f04c1d319d99aaef7521150`
- Describe: `v1.14.0-5-g9545ff6`
- Commit date: `2022-01-17T16:45:51-05:00`
- Commit subject: `added Ampere CC 80 and 86 to the core counting logic`
- License: upstream `LICENSE`, D. E. Shaw Research BSD-style license

## Pinned source objects

| Path | Git blob OID | SHA-256 |
|---|---|---|
| `tests/kat_vectors` | `cafa286d803793aa8c13e91ba17fc31c9ecbbe69` | `aab5ebabf40003f63d6d87b24cbd2c8a02652e00cf8bad64226fd50586929183` |
| `include/Random123/philox.h` | `7bf4d195772358a87b8fbb33667783b5caba61a4` | `6c2ef219a855885499a73b338d5f41dafe079618b2dae2f60ea86ee785d771e2` |
| `LICENSE` | — | `5fd0885ab205878bd90c19144677286252eb9b988c1545a3a910edfe1e27b6df` |

Permalinks:

- <https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/tests/kat_vectors#L27-L29>
- <https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/include/Random123/philox.h>

`upstream_philox4x32_10_kat.txt` is the exact three-line extraction produced by:

```sh
grep '^philox4x32 10 ' tests/kat_vectors
```

Its SHA-256 is
`fde9fb8458ae5c1a31c818731c633021a7fa47d666c6837a075377e11063b501`.
The three upstream cases cover zero counter/key, all-one counter/key, and a
non-zero digits-of-pi counter/key.

## Reproduction

```sh
git clone https://github.com/DEShawResearch/random123.git /tmp/random123
git -C /tmp/random123 checkout --detach 9545ff6413f258be2f04c1d319d99aaef7521150
g++ -std=c++17 -Wall -Wextra -Werror -pedantic \
  -I/tmp/random123/include random123_reference_generator.cpp \
  -o /tmp/random123_reference_generator
/tmp/random123_reference_generator direct
/tmp/random123_reference_generator address
```

The two generated TSV files are frozen receipts. Regeneration must use the
pinned commit and must be reviewed as a fixture change.
