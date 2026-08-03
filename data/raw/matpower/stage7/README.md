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

| Local file | Upstream Git blob | Canonical Git-blob SHA-256 (LF bytes) |
|---|---|---|
| `case1354pegase.m` | `d6ede376f35af472b45b93ae771209c483427c26` | `1b08b25a2f6c1d540d090009dfaff41ff2b05784a2d8d302a7ad695821557b89` |
| `case2868rte.m` | `0223116b52b3bd10786ccd61a808c440826aacdc` | `2b30e8943daf84ccb111cee30f19f4917afc9c3772cab3ce9eaf6193988a6861` |
| `case9241pegase.m` | `cc9816b188ef38725c1e7c5b04cb9555b6b8a78e` | `593a58ecddb5af509ff94410a6630f81021b48fa31da0694ff516acfa9ea5f3b` |

The SHA-256 values are computed over the canonical Git blob bytes, whose text
uses LF line endings. A Windows checkout may materialize CRLF working-tree
bytes and therefore have a different raw-file SHA-256 without any source
content drift; the recorded Git blob identity and canonical digest remain the
portable provenance checks.

No source case was edited. Stage 7 applies a separate, versioned structural
reconstruction policy at model-construction time. In particular, the public
files contain zero thermal ratings, inactive generators, and angle-difference
limits that do not map directly to the paper's printed DCOPF. Every such
transformation is declared in
`configs/benchmarks/stage_7_small_medium.json` and recorded in the raw run
evidence.

The benchmark results must cite MATPOWER and the additional PEGASE/RTE sources
requested in the headers of these files.
