# Stage 7 MATPOWER inputs

These three files are unmodified copies from the official MATPOWER 8.1
release. They are public network inputs only; they do not contain the
renewable, storage, reserve, or time-series data used by Wang et al.

## Immutable provenance

- Repository: <https://github.com/MATPOWER/matpower>
- Release: `8.1`
- Tag object: `3f8ecfdbc79b07697d6b45f8d868ac1c2d27f788`
- Resolved commit: `1a828c7af590714499284e36ee9c81273388c594`
- Retrieval date: 2026-08-03 UTC
- Version DOI: <https://doi.org/10.5281/zenodo.15871662>
- License: three-clause BSD; retained in `MATPOWER-LICENSE.txt`

| Local file | Upstream Git blob | SHA-256 |
|---|---|---|
| `case1354pegase.m` | `d6ede376f35af472b45b93ae771209c483427c26` | `9400ce5d5add70e654cf7513285920d4adc5dd87d649b0cc54f964bf5601a103` |
| `case2868rte.m` | `0223116b52b3bd10786ccd61a808c440826aacdc` | `07e5a9e26eacfc66730879c6959e7af621a97cb2697e68c26fbcc9bdcb78c101` |
| `case9241pegase.m` | `cc9816b188ef38725c1e7c5b04cb9555b6b8a78e` | `d88aa6d3a280b4fadd8130463291bf3511e5d0c5dac91bc37383f1c711bc8d01` |

No source case was edited. Stage 7 applies a separate, versioned structural
reconstruction policy at model-construction time. In particular, the public
files contain zero thermal ratings, inactive generators, and angle-difference
limits that do not map directly to the paper's printed DCOPF. Every such
transformation is declared in
`configs/benchmarks/stage_7_small_medium.json` and recorded in the raw run
evidence.

The benchmark results must cite MATPOWER and the additional PEGASE/RTE sources
requested in the headers of these files.
