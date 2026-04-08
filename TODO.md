# TODO: linalg-tester Implementation Gaps

## 1. Safety & Integrity (Phase 1 Refactor Incomplete)
- [x] **Sentinel Checking:** Implement `src/core/sentinel.h` and `src/core/sentinel.cpp` for out-of-bounds write detection.
- [x] **Buffer Allocation:** Update generators to use `alloc_with_sentinel` when `ld_pad > 0` or `|inc| > 1`.
- [x] **Validation:** Update test drivers (e.g., `gemm.cpp`, `trsm.cpp`) to call `check_sentinels` and report results.
- [x] **Reporting:** Update `report_result` in `src/core/report.cpp` to include a `SentinelResult` parameter as planned.

## 2. Numerical Accuracy (Error Metrics)
- [x] **NaN/Inf Handling:** Implement explicit NaN/Inf comparison logic in `src/core/error_metrics.cpp` to verify that NaN/Inf propagate correctly.
- [x] **Zero Reference:** Report absolute error separately when the reference value is zero but the computed value is nonzero.

## 3. Complex Number Support (Phase 5)
- [x] **Infrastructure:** Add `MpfrComplexMatrix` and complex arithmetic helpers.
- [x] **Conversion:** Update conversion library API to export `custom_to_mpfr_complex` and `mpfr_to_custom_complex`.
- [x] **BLAS Routines:** Add complex-only BLAS routines (HEMM, HERK, HER2K, etc.) to the dispatch table.
- [x] **Level 1 ABI:** Handle platform-dependent complex return value conventions (e.g., hidden first argument).

## 4. Complex LAPACK Support (NEW)
- [x] **Infrastructure:** Implement complex variants of LAPACK utilities in `src/core/mpfr_lapack_complex_utils.h`.
- [x] **Generators:** Add `gen_hermitian_positive_definite_array` and fix `get_eps` for complex types.
- [x] **Testers:** Add 33 complex LAPACK test drivers as separate files (7 factorizations, 7 solvers, 8 eigenvalue/SVD, 11 auxiliary).
- [x] **Verification:** All 328 regression tests pass (237 real + 32 complex BLAS + 59 complex LAPACK).

## 5. LAPACK Infrastructure (Phase 6+ Gaps)
- [x] **Reporting Refactor:** Replace the `ErrorResult` hack in `src/lapack/lapack_common.h` with a proper `report_lapack_result` that displays residuals and orthogonality correctly in all formats (Text, JSON, CSV).
- [x] **Structural Verification:** Implement checks for matrix structure (e.g., unit lower triangular L in GETRF, upper triangular R in GEQRF).
- [x] **Generator Relocation:** Move `gen_positive_definite_array` from `src/core/mpfr_lapack_utils.h` to `src/core/generators.cpp`.

## 6. PBLAS Support (NEW)
- [x] **Infrastructure:** Add support for ScaLAPACK-style 2D block-cyclic data distribution.
- [x] **Context:** Implement `PblasCtx` to handle BLACS context, process grid (M x N), and block sizes (MB, NB).
- [x] **Generators:** Implement distributed matrix generators that partition MPFR reference matrices across the process grid.
- [x] **Testers:** Add test drivers for core PBLAS routines (P_GEMM, P_TRSM, etc.) using MPI-aware verification.
- [x] **Verification:** Implement global-to-local and local-to-global MPFR gather/scatter for error computation.
- [x] **Verification:** All 442 regression tests pass (328 BLAS/LAPACK + 8 BLACS + 106 PBLAS real+complex).

## 7. BLACS Support (NEW)
- [x] **Infrastructure:** Wrappers for context creation, process grid management (`BLACS_GRIDINIT`, `BLACS_GRIDEXIT`).
- [x] **Communication:** Point-to-point (`GESD2D`/`GERV2D`) and broadcast (`GEBS2D`/`GEBR2D`) verification.
- [x] **Topology:** Test different process topologies (1D row/column, 2D square/rectangular).
- [x] **Verification:** All 442 regression tests pass (328 BLAS/LAPACK + 8 BLACS + 106 PBLAS).

## 8. ScaLAPACK Support (NEW)
- [ ] **Infrastructure:** Global-to-local coordinate mapping (`INDXG2L`, `INDXG2P`).
- [ ] **Routines:** Implement test drivers for ScaLAPACK factorizations (`P_GETRF`, `P_POTRF`) and solvers (`P_GESV`).
- [ ] **Verification:** Distributed residual computation using MPFR references gathered from the process grid.
- [ ] **Redistribution:** Test matrix redistribution routines (`P_GEMR2D`).

## 9. Build & CI
- [x] **Regression Tests:** Add an automated regression test script (`test/run_regression.sh`) that verifies all routines against a reference BLAS/LAPACK.
- [x] **CI Pipeline:** Set up a GitHub Actions pipeline to build and run the regression tests on Linux.
