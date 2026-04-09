/* mirror_gen.h -- MPFR-first generators and native materializers */

#pragma once

#include "../src/core/mpfr_types.h"
#include "../src/core/mpfr_complex_types.h"
#include "../src/core/tester_ctx.h"
#include <mpfr.h>

/* ------------------------------------------------------------------ */
/* MPFR-first generators: populate MpfrMatrix/MpfrScalar directly      */
/* ------------------------------------------------------------------ */

void gen_mpfr_random_matrix(MpfrMatrix &dst, mpfr_prec_t prec, unsigned *seed);
void gen_mpfr_random_vector(MpfrMatrix &dst, mpfr_prec_t prec, unsigned *seed);
void gen_mpfr_random_scalar(MpfrScalar &dst, mpfr_prec_t prec, unsigned *seed);

void gen_mpfr_triangular_matrix(MpfrMatrix &dst, char uplo, char diag,
                                 mpfr_prec_t prec, unsigned *seed);
void gen_mpfr_symmetric_matrix(MpfrMatrix &dst, char uplo,
                                mpfr_prec_t prec, unsigned *seed);

/* Band storage (ldab x n, where ldab = kl + ku + 1) */
void gen_mpfr_band_matrix(MpfrMatrix &dst, int m, int kl, int ku,
                           mpfr_prec_t prec, unsigned *seed);

/* Symmetric band storage (ldab x n, where ldab = k + 1) */
void gen_mpfr_symmetric_band_matrix(MpfrMatrix &dst, int n, int k, char uplo,
                                     mpfr_prec_t prec, unsigned *seed);

/* Triangular band storage */
void gen_mpfr_triangular_band_matrix(MpfrMatrix &dst, int n, int k,
                                      char uplo, char diag,
                                      mpfr_prec_t prec, unsigned *seed);

/* Packed symmetric (n*(n+1)/2 x 1 matrix for storage) */
void gen_mpfr_packed_symmetric(MpfrMatrix &dst, int n, char uplo,
                                mpfr_prec_t prec, unsigned *seed);

/* Packed triangular */
void gen_mpfr_packed_triangular(MpfrMatrix &dst, int n, char uplo, char diag,
                                 mpfr_prec_t prec, unsigned *seed);

/* Symmetric positive-definite: A = random_symmetric + n*I */
void gen_mpfr_positive_definite_matrix(MpfrMatrix &dst, mpfr_prec_t prec,
                                        unsigned *seed);

/* Diagonal-dominant symmetric (for indefinite solvers like SYSV) */
void gen_mpfr_diag_dominant_symmetric(MpfrMatrix &dst, char uplo,
                                       mpfr_prec_t prec, unsigned *seed);

/* GBSV band storage: ldab = 2*kl+ku+1 (extra rows for pivoting) */
void gen_mpfr_gbsv_band_matrix(MpfrMatrix &dst, int n, int kl, int ku,
                                 mpfr_prec_t prec, unsigned *seed);

/* Tridiagonal with diagonal dominance (for GTSV) */
void gen_mpfr_tridiagonal(MpfrMatrix &dl, MpfrMatrix &d, MpfrMatrix &du,
                            int n, mpfr_prec_t prec, unsigned *seed);

/* Complex MPFR-first generators */
void gen_mpfr_random_complex_matrix(MpfrComplexMatrix &dst,
                                     mpfr_prec_t prec, unsigned *seed);
void gen_mpfr_random_complex_vector(MpfrComplexMatrix &dst,
                                     mpfr_prec_t prec, unsigned *seed);
void gen_mpfr_random_complex_scalar(MpfrComplexScalar &dst,
                                     mpfr_prec_t prec, unsigned *seed);

void gen_mpfr_hermitian_matrix(MpfrComplexMatrix &dst, char uplo,
                                mpfr_prec_t prec, unsigned *seed);
void gen_mpfr_triangular_complex_matrix(MpfrComplexMatrix &dst,
                                         char uplo, char diag,
                                         mpfr_prec_t prec, unsigned *seed);

/* Hermitian band storage */
void gen_mpfr_hermitian_band_matrix(MpfrComplexMatrix &dst, int n, int k,
                                     char uplo, mpfr_prec_t prec,
                                     unsigned *seed);

/* Packed hermitian */
void gen_mpfr_packed_hermitian(MpfrComplexMatrix &dst, int n, char uplo,
                                mpfr_prec_t prec, unsigned *seed);

/* Hermitian positive-definite: A = random_hermitian + n*I */
void gen_mpfr_hermitian_positive_definite(MpfrComplexMatrix &dst,
                                           mpfr_prec_t prec, unsigned *seed);

/* Diagonal-dominant hermitian (for HESV) */
void gen_mpfr_diag_dominant_hermitian(MpfrComplexMatrix &dst, char uplo,
                                       mpfr_prec_t prec, unsigned *seed);

/* Diagonal-dominant complex symmetric (for CSYSV) */
void gen_mpfr_diag_dominant_complex_symmetric(MpfrComplexMatrix &dst,
                                               char uplo, mpfr_prec_t prec,
                                               unsigned *seed);

/* Complex band storage for CGBSV: ldab = 2*kl+ku+1 */
void gen_mpfr_complex_gbsv_band_matrix(MpfrComplexMatrix &dst, int n,
                                         int kl, int ku, mpfr_prec_t prec,
                                         unsigned *seed);

/* Complex tridiagonal with diagonal dominance (for CGTSV) */
void gen_mpfr_complex_tridiagonal(MpfrComplexMatrix &dl, MpfrComplexMatrix &d,
                                    MpfrComplexMatrix &du, int n,
                                    mpfr_prec_t prec, unsigned *seed);

/* Complex symmetric matrix */
void gen_mpfr_complex_symmetric_matrix(MpfrComplexMatrix &dst, char uplo,
                                        mpfr_prec_t prec, unsigned *seed);

/* ------------------------------------------------------------------ */
/* Native materializers: MPFR -> void* native array                    */
/* ------------------------------------------------------------------ */

/* Column-major matrix with leading dimension ld */
void *mpfr_mat_to_native(const MpfrMatrix &src, int ld, const TesterCtx &ctx);

/* Strided vector with increment inc */
void *mpfr_vec_to_native(const MpfrMatrix &src, int inc, const TesterCtx &ctx);

/* Single scalar */
void *mpfr_scalar_to_native(const MpfrScalar &src, const TesterCtx &ctx);

/* Complex column-major matrix */
void *mpfr_complex_mat_to_native(const MpfrComplexMatrix &src, int ld,
                                  const TesterCtx &ctx);

/* Complex strided vector */
void *mpfr_complex_vec_to_native(const MpfrComplexMatrix &src, int inc,
                                  const TesterCtx &ctx);

/* Complex scalar */
void *mpfr_complex_scalar_to_native(const MpfrComplexScalar &src,
                                     const TesterCtx &ctx);
