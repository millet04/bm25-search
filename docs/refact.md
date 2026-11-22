## Refactoring Tasks

The following tasks are intended to improve the `bm25-search` library.
Refactor the library according to these instructions.

### 0. Instructions
- You do not have to complete all the tasks at once. Handle them one at a time and report back after each task.
- Obtain approval before deleting or making significant changes to critical code or behavior.
- Test each algorithm after creating or modifying it to ensure that it works correctly.

### Task 1: Refactor BM25T
- `BM25T` (`model/BM25T.cpp`) uses the Newton–Raphson method to find the optimal value of the parameter `k`.
- However, the current implementation is fragile and contains bugs, even though it appears to work in some cases.
- Review and refactor the implementation to make the algorithm robust.

### Task 2: Compressed Sparse Row Matrix
- All algorithms in `bm25-search` were developed using sparse matrices, but the current implementation cannot handle large volumes of data efficiently.
- Create CSR (Compressed Sparse Row) versions of every algorithm currently included in the library, including `BM25`, `BM25L`, `BM25T`, `BM25F`, `BM25Plus`, `BM11`, `BM15`, and `TfIdf`.
- Preserve the existing public names and behavior so that code using the current library remains compatible. Treat the existing implementations as the sparse-matrix versions.
- Name the new CSR implementations by adding a `_CSR` suffix, such as `BM25_CSR`, `BM25L_CSR`, and `TfIdf_CSR`.
