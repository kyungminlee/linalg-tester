#pragma once

#include <cstddef>
#include <mpfr.h>

/* ------------------------------------------------------------------ */
/* Conversion function pointer types                                    */
/* ------------------------------------------------------------------ */

/* Convert one custom-type element (at src) into dst (already init'd). */
extern "C" typedef void (*custom_to_mpfr_fn)(mpfr_t dst, const void *src);

/* Convert mpfr_t src into one custom-type element at dst.
   dst must point to typesize bytes of writable storage.              */
extern "C" typedef void (*mpfr_to_custom_fn)(void *dst, mpfr_t src, mpfr_rnd_t rnd);

/* ------------------------------------------------------------------ */
/* BLAS function pointer types (gfortran ABI: hidden char lengths last)*/
/* ------------------------------------------------------------------ */

extern "C" typedef void (*gemm_fn_t)(
    const char *transa, const char *transb,
    const int  *m,      const int  *n,      const int  *k,
    const void *alpha,
    const void *A,      const int  *lda,
    const void *B,      const int  *ldb,
    const void *beta,
    void       *C,      const int  *ldc,
    std::size_t transa_len, std::size_t transb_len
);

extern "C" typedef void (*trsm_fn_t)(
    const char *side,   const char *uplo,
    const char *transa, const char *diag,
    const int  *m,      const int  *n,
    const void *alpha,
    const void *A,      const int  *lda,
    void       *B,      const int  *ldb,
    std::size_t side_len, std::size_t uplo_len,
    std::size_t transa_len, std::size_t diag_len
);

/* ------------------------------------------------------------------ */
/* Runtime context                                                      */
/* ------------------------------------------------------------------ */

struct TesterCtx {
    mpfr_prec_t       prec;
    std::size_t       typesize;
    custom_to_mpfr_fn to_mpfr;
    mpfr_to_custom_fn from_mpfr;
};

/* ------------------------------------------------------------------ */
/* Error result                                                         */
/* ------------------------------------------------------------------ */

struct ErrorResult {
    double max_relative;      /* max_ij |e_ij| / |ref_ij|           */
    double normwise_relative; /* ||E||_F / ||ref||_F                 */
};

/* Residual norms for sparse solver testing: ||b - Ax|| / ||b||        */
struct ResidualResult {
    double l1;    /* ||r||_1  / ||b||_1    */
    double l2;    /* ||r||_2  / ||b||_2    */
    double linf;  /* ||r||_inf / ||b||_inf */
};

/* ------------------------------------------------------------------ */
/* Public entry points                                                  */
/* ------------------------------------------------------------------ */

ErrorResult reference_test_gemm(
    const TesterCtx &ctx,
    char transa, char transb,
    int m, int n, int k,
    const void *alpha,
    const void *A,
    const void *B,
    const void *beta,
    const void *C_in,
    const void *C_out
);

ErrorResult reference_test_trsm(
    const TesterCtx &ctx,
    char side, char uplo, char transa, char diag,
    int m, int n,
    const void *alpha,
    const void *A,
    const void *B_in,
    const void *X_out
);

/* Compute residual norms for sparse Ax=b.
 * irn, jcn: 1-based COO indices.
 * a_vals:   nnz values (custom type).
 * b:        original RHS (custom type, n elements).
 * x:        computed solution (custom type, n elements).
 * Returns ||b - Ax||/||b|| in L1, L2, Linf.                          */
ResidualResult reference_sparse_residual(
    const TesterCtx &ctx,
    int n, int nnz,
    const int *irn, const int *jcn,
    const void *a_vals,
    const void *b,
    const void *x
);
