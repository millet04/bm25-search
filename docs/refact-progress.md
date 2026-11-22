## Refactoring Progress

This file records decisions and completed steps so the refactoring can be resumed after an interruption.

### Decisions

- Preserve the existing public algorithm names and behavior for compatibility.
- Add CSR implementations with a `_CSR` suffix, such as `BM25_CSR`.
- Use `0.05` as the default BM25T convergence tolerance (`eps`), matching `docs/index.md`.
- For BM25T, follow the objective and formulas documented in `docs/index.md`.
- If BM25T does not converge safely within `max_iter`, keep the initial `k` value.

### Completed

- [x] Review and clarify `docs/refact.md`.
- [x] Define the CSR naming and compatibility policy.
- [x] Correct BM25T's default `eps` value to `0.05`.
- [x] Make BM25T use term frequencies instead of term presence when calculating `c`.
- [x] Add finite-value, positive-domain, derivative, and step safeguards to BM25T optimization.
- [x] Audit and fix term-frequency extraction in `BM11`, `BM15`, `BM25`, `BM25F`, `BM25L`, `BM25Plus`, and `TfIdf`.
- [x] Rebuild the extension and run smoke checks for all existing algorithm classes with repeated terms.
- [x] Add the shared CSR storage and scoring path.
- [x] Add and validate `BM25_CSR` against the existing `BM25` scores.
- [x] Add and validate CSR implementations for `BM11`, `BM15`, `BM25L`, `BM25Plus`, and `TFIDF`.
- [x] Add and validate `BM25T_CSR` and `BM25F_CSR` by converting the exact TF results of the existing implementations into CSR storage.

### Implementation approach

`BM25T` and `BM25F` will reuse their existing model-building logic before conversion. This preserves BM25T's term-specific optimization and BM25F's field weighting exactly while exposing the final matrix in CSR form.

### Next steps

- [x] Build and validate the BM25T changes against the formulas in `docs/index.md` using a local `uv` environment with only `setuptools` and `pybind11` installed.
- [x] Add CSR implementations for all algorithms listed in `docs/refact.md`.
