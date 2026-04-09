/* scalapack_common.h -- Shared ScaLAPACK testing infrastructure */
/* Provides MPFR reference solvers (LU, Cholesky), gather utilities,
   and PDGEMM helper for distributed residual computation. */

#pragma once

#include "../pblas/pblas_common.h"
#include "../lapack/lapack_common.h"
#include "../core/mpfr_types.h"
#include "../core/mpfr_complex_types.h"
#include "../core/mpfr_lapack_utils.h"
#include "../core/mpfr_lapack_complex_utils.h"
#include "../core/mpfr_complex.h"
#include "../core/error_metrics.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

/* ------------------------------------------------------------------ */
/* Gather: local block-cyclic -> global column-major                   */
/* (Inverse of scatter_global_to_local)                                */
/* ------------------------------------------------------------------ */

inline void gather_local_to_global(void *global, int ldg,
                                    const void *local, int lld,
                                    int m, int n, int mb, int nb,
                                    int myrow, int mycol,
                                    int nprow, int npcol,
                                    std::size_t typesize)
{
    int loc_m = numroc(m, mb, myrow, 0, nprow);
    int loc_n = numroc(n, nb, mycol, 0, npcol);

    for (int jl = 0; jl < loc_n; ++jl) {
        int jg = indxl2g(jl, nb, mycol, 0, npcol);
        for (int il = 0; il < loc_m; ++il) {
            int ig = indxl2g(il, mb, myrow, 0, nprow);
            std::memcpy(
                static_cast<char *>(global) +
                    (static_cast<std::size_t>(jg) * ldg + ig) * typesize,
                static_cast<const char *>(local) +
                    (static_cast<std::size_t>(jl) * lld + il) * typesize,
                typesize);
        }
    }
}

/* ------------------------------------------------------------------ */
/* MPFR LU factorization with partial pivoting (in-place)              */
/* Returns 0-based ipiv (row swap indices).                            */
/* ------------------------------------------------------------------ */

inline void mpfr_lu_factorize(MpfrMatrix &A, std::vector<int> &ipiv)
{
    int n = A.rows();
    ipiv.resize(n);
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar tmp(prec), abs_val(prec), max_val(prec);

    for (int k = 0; k < n; ++k) {
        /* Find pivot: max |A(i,k)| for i >= k */
        int max_row = k;
        mpfr_abs(max_val.get(), A.at(k, k), MPFR_RNDN);
        for (int i = k + 1; i < n; ++i) {
            mpfr_abs(abs_val.get(), A.at(i, k), MPFR_RNDN);
            if (mpfr_cmp(abs_val.get(), max_val.get()) > 0) {
                mpfr_set(max_val.get(), abs_val.get(), MPFR_RNDN);
                max_row = i;
            }
        }
        ipiv[k] = max_row;

        /* Swap rows k and max_row */
        if (max_row != k) {
            for (int j = 0; j < n; ++j)
                mpfr_swap(A.at(k, j), A.at(max_row, j));
        }

        /* Compute multipliers and update submatrix */
        if (!mpfr_zero_p(A.at(k, k))) {
            for (int i = k + 1; i < n; ++i)
                mpfr_div(A.at(i, k), A.at(i, k), A.at(k, k), MPFR_RNDN);
            for (int j = k + 1; j < n; ++j) {
                for (int i = k + 1; i < n; ++i) {
                    mpfr_mul(tmp.get(), A.at(i, k), A.at(k, j), MPFR_RNDN);
                    mpfr_sub(A.at(i, j), A.at(i, j), tmp.get(), MPFR_RNDN);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* MPFR LU solve: X = A^{-1} * B using high-precision LU              */
/* A is n-by-n, B is n-by-nrhs, X is n-by-nrhs                       */
/* ------------------------------------------------------------------ */

inline void mpfr_lu_solve(MpfrMatrix &X, const MpfrMatrix &A_orig,
                           const MpfrMatrix &B)
{
    int n = A_orig.rows(), nrhs = B.cols();
    mpfr_prec_t prec = mpfr_get_prec(A_orig.at(0, 0));

    /* Copy A for factorization */
    MpfrMatrix A(n, n, prec);
    mpfr_mat_copy(A, A_orig);

    /* LU factorize */
    std::vector<int> ipiv;
    mpfr_lu_factorize(A, ipiv);

    /* Copy B to X */
    mpfr_mat_copy(X, B);

    /* Apply row swaps (0-based ipiv) */
    for (int k = 0; k < n; ++k) {
        if (ipiv[k] != k) {
            for (int j = 0; j < nrhs; ++j)
                mpfr_swap(X.at(k, j), X.at(ipiv[k], j));
        }
    }

    /* Forward substitution: L * Y = P * B (L is unit lower) */
    MpfrScalar tmp(prec);
    for (int j = 0; j < nrhs; ++j) {
        for (int k = 0; k < n; ++k) {
            for (int i = k + 1; i < n; ++i) {
                mpfr_mul(tmp.get(), A.at(i, k), X.at(k, j), MPFR_RNDN);
                mpfr_sub(X.at(i, j), X.at(i, j), tmp.get(), MPFR_RNDN);
            }
        }
    }

    /* Backward substitution: U * X = Y */
    for (int j = 0; j < nrhs; ++j) {
        for (int k = n - 1; k >= 0; --k) {
            mpfr_div(X.at(k, j), X.at(k, j), A.at(k, k), MPFR_RNDN);
            for (int i = 0; i < k; ++i) {
                mpfr_mul(tmp.get(), A.at(i, k), X.at(k, j), MPFR_RNDN);
                mpfr_sub(X.at(i, j), X.at(i, j), tmp.get(), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* MPFR Cholesky factorization (lower triangle, in-place)              */
/* Returns false if A is not positive definite.                        */
/* ------------------------------------------------------------------ */

inline bool mpfr_cholesky_factor(MpfrMatrix &L, const MpfrMatrix &A_orig,
                                   char uplo = 'L')
{
    int n = A_orig.rows();
    mpfr_prec_t prec = mpfr_get_prec(A_orig.at(0, 0));
    mpfr_mat_copy(L, A_orig);

    MpfrScalar tmp(prec), sum(prec);

    if (uplo == 'L') {
        for (int j = 0; j < n; ++j) {
            mpfr_set(sum.get(), L.at(j, j), MPFR_RNDN);
            for (int k = 0; k < j; ++k) {
                mpfr_mul(tmp.get(), L.at(j, k), L.at(j, k), MPFR_RNDN);
                mpfr_sub(sum.get(), sum.get(), tmp.get(), MPFR_RNDN);
            }
            if (mpfr_sgn(sum.get()) <= 0) return false;
            mpfr_sqrt(L.at(j, j), sum.get(), MPFR_RNDN);

            for (int i = j + 1; i < n; ++i) {
                mpfr_set(sum.get(), L.at(i, j), MPFR_RNDN);
                for (int k = 0; k < j; ++k) {
                    mpfr_mul(tmp.get(), L.at(i, k), L.at(j, k), MPFR_RNDN);
                    mpfr_sub(sum.get(), sum.get(), tmp.get(), MPFR_RNDN);
                }
                mpfr_div(L.at(i, j), sum.get(), L.at(j, j), MPFR_RNDN);
            }
            /* Zero upper part */
            for (int i = 0; i < j; ++i)
                mpfr_set_d(L.at(i, j), 0.0, MPFR_RNDN);
        }
    } else {
        /* Upper Cholesky: A = U^T * U */
        for (int j = 0; j < n; ++j) {
            mpfr_set(sum.get(), L.at(j, j), MPFR_RNDN);
            for (int k = 0; k < j; ++k) {
                mpfr_mul(tmp.get(), L.at(k, j), L.at(k, j), MPFR_RNDN);
                mpfr_sub(sum.get(), sum.get(), tmp.get(), MPFR_RNDN);
            }
            if (mpfr_sgn(sum.get()) <= 0) return false;
            mpfr_sqrt(L.at(j, j), sum.get(), MPFR_RNDN);

            for (int i = j + 1; i < n; ++i) {
                mpfr_set(sum.get(), L.at(j, i), MPFR_RNDN);
                for (int k = 0; k < j; ++k) {
                    mpfr_mul(tmp.get(), L.at(k, j), L.at(k, i), MPFR_RNDN);
                    mpfr_sub(sum.get(), sum.get(), tmp.get(), MPFR_RNDN);
                }
                mpfr_div(L.at(j, i), sum.get(), L.at(j, j), MPFR_RNDN);
            }
            /* Zero lower part */
            for (int i = j + 1; i < n; ++i)
                mpfr_set_d(L.at(i, j), 0.0, MPFR_RNDN);
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* MPFR Cholesky solve: X = A^{-1} * B using Cholesky                 */
/* ------------------------------------------------------------------ */

inline void mpfr_cholesky_solve(MpfrMatrix &X, const MpfrMatrix &A_orig,
                                  const MpfrMatrix &B, char uplo = 'L')
{
    int n = A_orig.rows(), nrhs = B.cols();
    mpfr_prec_t prec = mpfr_get_prec(A_orig.at(0, 0));
    MpfrScalar tmp(prec);

    /* Cholesky factor */
    MpfrMatrix L(n, n, prec);
    mpfr_cholesky_factor(L, A_orig, uplo);

    /* Copy B to X */
    mpfr_mat_copy(X, B);

    if (uplo == 'L') {
        /* Forward sub: L * Y = B */
        for (int j = 0; j < nrhs; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int i = 0; i < k; ++i) {
                    mpfr_mul(tmp.get(), L.at(k, i), X.at(i, j), MPFR_RNDN);
                    mpfr_sub(X.at(k, j), X.at(k, j), tmp.get(), MPFR_RNDN);
                }
                mpfr_div(X.at(k, j), X.at(k, j), L.at(k, k), MPFR_RNDN);
            }
        }
        /* Backward sub: L^T * X = Y */
        for (int j = 0; j < nrhs; ++j) {
            for (int k = n - 1; k >= 0; --k) {
                for (int i = k + 1; i < n; ++i) {
                    mpfr_mul(tmp.get(), L.at(i, k), X.at(i, j), MPFR_RNDN);
                    mpfr_sub(X.at(k, j), X.at(k, j), tmp.get(), MPFR_RNDN);
                }
                mpfr_div(X.at(k, j), X.at(k, j), L.at(k, k), MPFR_RNDN);
            }
        }
    } else {
        /* Forward sub: U^T * Y = B */
        for (int j = 0; j < nrhs; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int i = 0; i < k; ++i) {
                    mpfr_mul(tmp.get(), L.at(i, k), X.at(i, j), MPFR_RNDN);
                    mpfr_sub(X.at(k, j), X.at(k, j), tmp.get(), MPFR_RNDN);
                }
                mpfr_div(X.at(k, j), X.at(k, j), L.at(k, k), MPFR_RNDN);
            }
        }
        /* Backward sub: U * X = Y */
        for (int j = 0; j < nrhs; ++j) {
            for (int k = n - 1; k >= 0; --k) {
                for (int i = k + 1; i < n; ++i) {
                    mpfr_mul(tmp.get(), L.at(k, i), X.at(i, j), MPFR_RNDN);
                    mpfr_sub(X.at(k, j), X.at(k, j), tmp.get(), MPFR_RNDN);
                }
                mpfr_div(X.at(k, j), X.at(k, j), L.at(k, k), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Complex division: (re_out, im_out) = (a_re, a_im) / (b_re, b_im)   */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_div(mpfr_t re_out, mpfr_t im_out,
                               const mpfr_t a_re, const mpfr_t a_im,
                               const mpfr_t b_re, const mpfr_t b_im,
                               mpfr_rnd_t rnd)
{
    mpfr_prec_t prec = mpfr_get_prec(a_re);
    mpfr_t denom, t1, t2;
    mpfr_init2(denom, prec); mpfr_init2(t1, prec); mpfr_init2(t2, prec);

    /* denom = b_re^2 + b_im^2 */
    mpfr_mul(t1, b_re, b_re, rnd);
    mpfr_fma(denom, b_im, b_im, t1, rnd);

    /* re = (a_re*b_re + a_im*b_im) / denom */
    mpfr_mul(t1, a_re, b_re, rnd);
    mpfr_fma(t1, a_im, b_im, t1, rnd);
    mpfr_div(re_out, t1, denom, rnd);

    /* im = (a_im*b_re - a_re*b_im) / denom */
    mpfr_mul(t1, a_im, b_re, rnd);
    mpfr_mul(t2, a_re, b_im, rnd);
    mpfr_sub(t1, t1, t2, rnd);
    mpfr_div(im_out, t1, denom, rnd);

    mpfr_clear(denom); mpfr_clear(t1); mpfr_clear(t2);
}

/* ------------------------------------------------------------------ */
/* Complex MPFR LU factorization with partial pivoting (in-place)      */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_lu_factorize(MpfrComplexMatrix &A,
                                        std::vector<int> &ipiv)
{
    int n = A.rows();
    ipiv.resize(n);
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    MpfrScalar abs_cur(prec), abs_max(prec);
    MpfrComplexScalar tmp(prec);

    for (int k = 0; k < n; ++k) {
        int max_row = k;
        mpfr_complex_abs(abs_max.get(), A.re(k, k), A.im(k, k), MPFR_RNDN);
        for (int i = k + 1; i < n; ++i) {
            mpfr_complex_abs(abs_cur.get(), A.re(i, k), A.im(i, k), MPFR_RNDN);
            if (mpfr_cmp(abs_cur.get(), abs_max.get()) > 0) {
                mpfr_set(abs_max.get(), abs_cur.get(), MPFR_RNDN);
                max_row = i;
            }
        }
        ipiv[k] = max_row;

        if (max_row != k) {
            for (int j = 0; j < n; ++j) {
                mpfr_swap(A.re(k, j), A.re(max_row, j));
                mpfr_swap(A.im(k, j), A.im(max_row, j));
            }
        }

        bool pivot_nonzero = !mpfr_zero_p(A.re(k, k)) || !mpfr_zero_p(A.im(k, k));
        if (pivot_nonzero) {
            for (int i = k + 1; i < n; ++i)
                mpfr_complex_div(A.re(i, k), A.im(i, k),
                                  A.re(i, k), A.im(i, k),
                                  A.re(k, k), A.im(k, k), MPFR_RNDN);

            for (int j = k + 1; j < n; ++j) {
                for (int i = k + 1; i < n; ++i) {
                    mpfr_complex_mul(tmp.re(), tmp.im(),
                                      A.re(i, k), A.im(i, k),
                                      A.re(k, j), A.im(k, j), MPFR_RNDN);
                    mpfr_sub(A.re(i, j), A.re(i, j), tmp.re(), MPFR_RNDN);
                    mpfr_sub(A.im(i, j), A.im(i, j), tmp.im(), MPFR_RNDN);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Complex MPFR LU solve: X = A^{-1} * B                              */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_lu_solve(MpfrComplexMatrix &X,
                                    const MpfrComplexMatrix &A_orig,
                                    const MpfrComplexMatrix &B)
{
    int n = A_orig.rows(), nrhs = B.cols();
    mpfr_prec_t prec = mpfr_get_prec(A_orig.re(0, 0));

    MpfrComplexMatrix A(n, n, prec);
    mpfr_complex_mat_copy(A, A_orig);

    std::vector<int> ipiv;
    mpfr_complex_lu_factorize(A, ipiv);

    mpfr_complex_mat_copy(X, B);

    /* Apply row swaps */
    for (int k = 0; k < n; ++k) {
        if (ipiv[k] != k) {
            for (int j = 0; j < nrhs; ++j) {
                mpfr_swap(X.re(k, j), X.re(ipiv[k], j));
                mpfr_swap(X.im(k, j), X.im(ipiv[k], j));
            }
        }
    }

    /* Forward substitution */
    MpfrComplexScalar tmp(prec);
    for (int j = 0; j < nrhs; ++j) {
        for (int k = 0; k < n; ++k) {
            for (int i = k + 1; i < n; ++i) {
                mpfr_complex_mul(tmp.re(), tmp.im(),
                                  A.re(i, k), A.im(i, k),
                                  X.re(k, j), X.im(k, j), MPFR_RNDN);
                mpfr_sub(X.re(i, j), X.re(i, j), tmp.re(), MPFR_RNDN);
                mpfr_sub(X.im(i, j), X.im(i, j), tmp.im(), MPFR_RNDN);
            }
        }
    }

    /* Backward substitution */
    for (int j = 0; j < nrhs; ++j) {
        for (int k = n - 1; k >= 0; --k) {
            mpfr_complex_div(X.re(k, j), X.im(k, j),
                              X.re(k, j), X.im(k, j),
                              A.re(k, k), A.im(k, k), MPFR_RNDN);
            for (int i = 0; i < k; ++i) {
                mpfr_complex_mul(tmp.re(), tmp.im(),
                                  A.re(i, k), A.im(i, k),
                                  X.re(k, j), X.im(k, j), MPFR_RNDN);
                mpfr_sub(X.re(i, j), X.re(i, j), tmp.re(), MPFR_RNDN);
                mpfr_sub(X.im(i, j), X.im(i, j), tmp.im(), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Complex MPFR Cholesky factorization (Hermitian, lower)              */
/* ------------------------------------------------------------------ */

inline bool mpfr_complex_cholesky_factor(MpfrComplexMatrix &L,
                                           const MpfrComplexMatrix &A_orig,
                                           char uplo = 'L')
{
    int n = A_orig.rows();
    mpfr_prec_t prec = mpfr_get_prec(A_orig.re(0, 0));
    mpfr_complex_mat_copy(L, A_orig);

    MpfrComplexScalar tmp(prec);
    MpfrScalar sum_re(prec), t(prec);

    if (uplo == 'L') {
        for (int j = 0; j < n; ++j) {
            /* L(j,j) = sqrt(A(j,j) - sum|L(j,k)|^2) -- real since HPD */
            mpfr_set(sum_re.get(), L.re(j, j), MPFR_RNDN);
            for (int k = 0; k < j; ++k) {
                mpfr_complex_abs_sq(t.get(), L.re(j, k), L.im(j, k), MPFR_RNDN);
                mpfr_sub(sum_re.get(), sum_re.get(), t.get(), MPFR_RNDN);
            }
            if (mpfr_sgn(sum_re.get()) <= 0) return false;
            mpfr_sqrt(L.re(j, j), sum_re.get(), MPFR_RNDN);
            mpfr_set_d(L.im(j, j), 0.0, MPFR_RNDN);

            for (int i = j + 1; i < n; ++i) {
                /* L(i,j) = (A(i,j) - sum L(i,k)*conj(L(j,k))) / L(j,j) */
                mpfr_set(tmp.re(), L.re(i, j), MPFR_RNDN);
                mpfr_set(tmp.im(), L.im(i, j), MPFR_RNDN);
                for (int k = 0; k < j; ++k) {
                    MpfrComplexScalar cjk(prec);
                    mpfr_complex_conj(cjk.re(), cjk.im(),
                                       L.re(j, k), L.im(j, k), MPFR_RNDN);
                    MpfrComplexScalar prod(prec);
                    mpfr_complex_mul(prod.re(), prod.im(),
                                      L.re(i, k), L.im(i, k),
                                      cjk.re(), cjk.im(), MPFR_RNDN);
                    mpfr_sub(tmp.re(), tmp.re(), prod.re(), MPFR_RNDN);
                    mpfr_sub(tmp.im(), tmp.im(), prod.im(), MPFR_RNDN);
                }
                mpfr_div(L.re(i, j), tmp.re(), L.re(j, j), MPFR_RNDN);
                mpfr_div(L.im(i, j), tmp.im(), L.re(j, j), MPFR_RNDN);
            }
            /* Zero upper */
            for (int i = 0; i < j; ++i) {
                mpfr_set_d(L.re(i, j), 0.0, MPFR_RNDN);
                mpfr_set_d(L.im(i, j), 0.0, MPFR_RNDN);
            }
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Complex MPFR Cholesky solve: X = A^{-1} * B                        */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_cholesky_solve(MpfrComplexMatrix &X,
                                          const MpfrComplexMatrix &A_orig,
                                          const MpfrComplexMatrix &B,
                                          char uplo = 'L')
{
    int n = A_orig.rows(), nrhs = B.cols();
    mpfr_prec_t prec = mpfr_get_prec(A_orig.re(0, 0));

    MpfrComplexMatrix L(n, n, prec);
    mpfr_complex_cholesky_factor(L, A_orig, uplo);
    mpfr_complex_mat_copy(X, B);

    MpfrComplexScalar tmp(prec);

    if (uplo == 'L') {
        /* Forward sub: L * Y = B */
        for (int j = 0; j < nrhs; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int i = 0; i < k; ++i) {
                    mpfr_complex_mul(tmp.re(), tmp.im(),
                                      L.re(k, i), L.im(k, i),
                                      X.re(i, j), X.im(i, j), MPFR_RNDN);
                    mpfr_sub(X.re(k, j), X.re(k, j), tmp.re(), MPFR_RNDN);
                    mpfr_sub(X.im(k, j), X.im(k, j), tmp.im(), MPFR_RNDN);
                }
                /* L(k,k) is real */
                mpfr_div(X.re(k, j), X.re(k, j), L.re(k, k), MPFR_RNDN);
                mpfr_div(X.im(k, j), X.im(k, j), L.re(k, k), MPFR_RNDN);
            }
        }
        /* Backward sub: L^H * X = Y */
        for (int j = 0; j < nrhs; ++j) {
            for (int k = n - 1; k >= 0; --k) {
                for (int i = k + 1; i < n; ++i) {
                    /* conj(L(i,k)) * X(i,j) */
                    MpfrComplexScalar clik(prec);
                    mpfr_complex_conj(clik.re(), clik.im(),
                                       L.re(i, k), L.im(i, k), MPFR_RNDN);
                    mpfr_complex_mul(tmp.re(), tmp.im(),
                                      clik.re(), clik.im(),
                                      X.re(i, j), X.im(i, j), MPFR_RNDN);
                    mpfr_sub(X.re(k, j), X.re(k, j), tmp.re(), MPFR_RNDN);
                    mpfr_sub(X.im(k, j), X.im(k, j), tmp.im(), MPFR_RNDN);
                }
                mpfr_div(X.re(k, j), X.re(k, j), L.re(k, k), MPFR_RNDN);
                mpfr_div(X.im(k, j), X.im(k, j), L.re(k, k), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* PDGEMM function pointer type (for distributed residual computation) */
/* ------------------------------------------------------------------ */

extern "C" typedef void (*sc_pgemm_fn_t)(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const void *alpha,
    const void *A, const int *ia, const int *ja, const int *desca,
    const void *B, const int *ib, const int *jb, const int *descb,
    const void *beta,
    void *C, const int *ic, const int *jc, const int *descc,
    std::size_t, std::size_t
);

/* Load PDGEMM/PZGEMM from the library for residual computation.
   sym is the test routine symbol, e.g. "pdsyev_" -> extract 'd' -> "pdgemm_" */
inline sc_pgemm_fn_t load_pgemm(void *lib, const char *sym)
{
    char pfx = sym[1]; /* e.g. 'd' from "pdsyev_" */
    char gemm_sym[16];
    std::snprintf(gemm_sym, sizeof(gemm_sym), "p%cgemm_", pfx);
    return reinterpret_cast<sc_pgemm_fn_t>(try_load_sym(lib, gemm_sym));
}

/* Gather distributed X to global, compute proper solve residual.
   ||A*X - B||_1 / (||A||_1 * ||X||_1 * n * eps)                 */
inline double scalapack_gather_solve_residual(
    const MpfrMatrix &A_orig,  /* full global A */
    const MpfrMatrix &B_orig,  /* full global B */
    const void *X_loc,         /* local portion of computed solution */
    int lld_x,
    int n, int nrhs,
    const PblasCtx &pc,
    const TesterCtx &ctx)
{
    double eps = get_eps(ctx);
    void *X_global = std::calloc(static_cast<std::size_t>(n) * nrhs, ctx.typesize);
    gather_local_to_global(X_global, n, X_loc, lld_x,
                            n, nrhs, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol,
                            pc.bc.nprow, pc.bc.npcol, ctx.typesize);

    MpfrMatrix X_mpfr(n, nrhs, ctx.prec);
    custom_to_mpfr_mat(X_mpfr, X_global, n, ctx);
    std::free(X_global);

    return mpfr_solve_residual(A_orig, X_mpfr, B_orig, eps);
}

/* Complex variant */
inline double scalapack_gather_solve_residual_complex(
    const MpfrComplexMatrix &A_orig,
    const MpfrComplexMatrix &B_orig,
    const void *X_loc,
    int lld_x,
    int n, int nrhs,
    const PblasCtx &pc,
    const TesterCtx &ctx)
{
    double eps = get_eps(ctx);
    void *X_global = std::calloc(static_cast<std::size_t>(n) * nrhs, ctx.typesize);
    gather_local_to_global(X_global, n, X_loc, lld_x,
                            n, nrhs, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol,
                            pc.bc.nprow, pc.bc.npcol, ctx.typesize);

    MpfrComplexMatrix X_mpfr(n, nrhs, ctx.prec);
    custom_to_mpfr_complex_mat(X_mpfr, X_global, n, ctx);
    std::free(X_global);

    return mpfr_complex_solve_residual(A_orig, X_mpfr, B_orig, eps);
}
