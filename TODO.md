# TODO: linalg-tester Implementation Gaps

## 1. Safety & Integrity (Phase 1 Refactor Incomplete)
- [x] **Sentinel Checking:** Implement `src/core/sentinel.h` and `src/core/sentinel.cpp` for out-of-bounds write detection.
- [ ] **Buffer Allocation:** Update generators to use `alloc_with_sentinel` when `ld_pad > 0` or `|inc| > 1`.
- [ ] **Validation:** Update test drivers (e.g., `gemm.cpp`, `trsm.cpp`) to call `check_sentinels` and report results.
- [x] **Reporting:** Update `report_result` in `src/core/report.cpp` to include a `SentinelResult` parameter as planned.

## 2. Numerical Accuracy (Error Metrics)
- [x] **NaN/Inf Handling:** Implement explicit NaN/Inf comparison logic in `src/core/error_metrics.cpp` to verify that NaN/Inf propagate correctly.
- [x] **Zero Reference:** Report absolute error separately when the reference value is zero but the computed value is nonzero.

## 3. Complex Number Support (Phase 5 Missing)
- [ ] **Infrastructure:** Add `MpfrComplexMatrix` and complex arithmetic helpers.
- [ ] **Conversion:** Update conversion library API to export `custom_to_mpfr_complex` and `mpfr_to_custom_complex`.
- [ ] **Routines:** Add complex-only BLAS routines (HEMM, HERK, HER2K, etc.) to the dispatch table.
- [ ] **Level 1 ABI:** Handle platform-dependent complex return value conventions (e.g., hidden first argument).

## 4. LAPACK Infrastructure (Phase 6+ Gaps)
- [x] **Reporting Refactor:** Replace the `ErrorResult` hack in `src/lapack/lapack_common.h` with a proper `report_lapack_result` that displays residuals and orthogonality correctly in all formats (Text, JSON, CSV).
- [x] **Structural Verification:** Implement checks for matrix structure (e.g., unit lower triangular L in GETRF, upper triangular R in GEQRF).
- [x] **Generator Relocation:** Move `gen_positive_definite_array` from `src/core/mpfr_lapack_utils.h` to `src/core/generators.cpp`.

## 5. Build & CI
- [x] **Regression Tests:** Add an automated regression test script (`test/run_regression.sh`) that verifies all routines against a reference BLAS/LAPACK.
- [x] **CI Pipeline:** Set up a GitHub Actions pipeline to build and run the regression tests on Linux.
