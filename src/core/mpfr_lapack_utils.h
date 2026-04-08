/* mpfr_lapack_utils.h -- MPFR utilities for LAPACK residual computation */

#pragma once

#include "mpfr_types.h"
#include "generators.h"
#include <algorithm>
#include <cstdlib>

/* ------------------------------------------------------------------ */
/* MPFR general matrix multiply: C = alpha*A*B + beta*C                */
/* A is m-by-k, B is k-by-n, C is m-by-n                              */
/* ------------------------------------------------------------------ */

inline void mpfr_mat_mul(MpfrMatrix &C,
                          mpfr_t alpha,
                          const MpfrMatrix &A,
                          const MpfrMatrix &B,
                          mpfr_t beta)
{
    int m = A.rows(), k = A.cols(), n = B.cols();
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar acc(prec), tmp(prec), bc(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
            for (int p = 0; p < k; ++p)
                mpfr_fma(acc.get(), A.at(i, p), B.at(p, j), acc.get(), MPFR_RNDN);
            mpfr_mul(tmp.get(), alpha, acc.get(), MPFR_RNDN);
            mpfr_mul(bc.get(), beta, C.at(i, j), MPFR_RNDN);
            mpfr_add(C.at(i, j), tmp.get(), bc.get(), MPFR_RNDN);
        }
    }
}

/* Simplified: C = A * B (no alpha/beta) */
inline void mpfr_mat_mul_simple(MpfrMatrix &C,
                                 const MpfrMatrix &A,
                                 const MpfrMatrix &B)
{
    int m = A.rows(), k = A.cols(), n = B.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar acc(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(C.at(i, j), 0.0, MPFR_RNDN);
            for (int p = 0; p < k; ++p)
                mpfr_fma(C.at(i, j), A.at(i, p), B.at(p, j),
                         C.at(i, j), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* MPFR matrix transpose: B = A^T                                      */
/* ------------------------------------------------------------------ */

inline void mpfr_mat_transpose(MpfrMatrix &B, const MpfrMatrix &A)
{
    for (int j = 0; j < A.cols(); ++j)
        for (int i = 0; i < A.rows(); ++i)
            mpfr_set(B.at(j, i), A.at(i, j), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* MPFR matrix subtraction: C = A - B                                  */
/* ------------------------------------------------------------------ */

inline void mpfr_mat_sub(MpfrMatrix &C,
                          const MpfrMatrix &A,
                          const MpfrMatrix &B)
{
    for (int j = 0; j < A.cols(); ++j)
        for (int i = 0; i < A.rows(); ++i)
            mpfr_sub(C.at(i, j), A.at(i, j), B.at(i, j), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* MPFR matrix 1-norm: max over columns of sum of |a_ij|               */
/* ------------------------------------------------------------------ */

inline double mpfr_mat_norm1(const MpfrMatrix &A)
{
    int m = A.rows(), n = A.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar col_sum(prec), abs_val(prec), max_norm(prec);
    mpfr_set_d(max_norm.get(), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j) {
        mpfr_set_d(col_sum.get(), 0.0, MPFR_RNDN);
        for (int i = 0; i < m; ++i) {
            mpfr_abs(abs_val.get(), A.at(i, j), MPFR_RNDN);
            mpfr_add(col_sum.get(), col_sum.get(), abs_val.get(), MPFR_RNDN);
        }
        if (mpfr_cmp(col_sum.get(), max_norm.get()) > 0)
            mpfr_set(max_norm.get(), col_sum.get(), MPFR_RNDN);
    }
    return mpfr_get_d(max_norm.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* MPFR Frobenius norm                                                 */
/* ------------------------------------------------------------------ */

inline double mpfr_mat_normF(const MpfrMatrix &A)
{
    int m = A.rows(), n = A.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar sum(prec);
    mpfr_set_d(sum.get(), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            mpfr_fma(sum.get(), A.at(i, j), A.at(i, j), sum.get(), MPFR_RNDN);

    mpfr_sqrt(sum.get(), sum.get(), MPFR_RNDN);
    return mpfr_get_d(sum.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* MPFR infinity norm: max over rows of sum of |a_ij|                  */
/* ------------------------------------------------------------------ */

inline double mpfr_mat_normI(const MpfrMatrix &A)
{
    int m = A.rows(), n = A.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar row_sum(prec), abs_val(prec), max_norm(prec);
    mpfr_set_d(max_norm.get(), 0.0, MPFR_RNDN);

    for (int i = 0; i < m; ++i) {
        mpfr_set_d(row_sum.get(), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j) {
            mpfr_abs(abs_val.get(), A.at(i, j), MPFR_RNDN);
            mpfr_add(row_sum.get(), row_sum.get(), abs_val.get(), MPFR_RNDN);
        }
        if (mpfr_cmp(row_sum.get(), max_norm.get()) > 0)
            mpfr_set(max_norm.get(), row_sum.get(), MPFR_RNDN);
    }
    return mpfr_get_d(max_norm.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* MPFR max absolute element                                           */
/* ------------------------------------------------------------------ */

inline double mpfr_mat_normM(const MpfrMatrix &A)
{
    int m = A.rows(), n = A.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar abs_val(prec), max_val(prec);
    mpfr_set_d(max_val.get(), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_abs(abs_val.get(), A.at(i, j), MPFR_RNDN);
            if (mpfr_cmp(abs_val.get(), max_val.get()) > 0)
                mpfr_set(max_val.get(), abs_val.get(), MPFR_RNDN);
        }
    }
    return mpfr_get_d(max_val.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* MPFR symmetric 1-norm (only reads one triangle)                     */
/* ------------------------------------------------------------------ */

inline double mpfr_symmat_norm1(const MpfrMatrix &A, char uplo)
{
    int n = A.rows();
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar abs_val(prec), max_norm(prec);
    mpfr_set_d(max_norm.get(), 0.0, MPFR_RNDN);

    /* col_sums accumulates column sums using the symmetry */
    MpfrMatrix col_sums(n, 1, prec);
    for (int j = 0; j < n; ++j)
        mpfr_set_d(col_sums.at(j, 0), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j) {
        if (uplo == 'U') {
            for (int i = 0; i <= j; ++i) {
                mpfr_abs(abs_val.get(), A.at(i, j), MPFR_RNDN);
                mpfr_add(col_sums.at(j, 0), col_sums.at(j, 0), abs_val.get(), MPFR_RNDN);
                if (i != j)
                    mpfr_add(col_sums.at(i, 0), col_sums.at(i, 0), abs_val.get(), MPFR_RNDN);
            }
        } else {
            for (int i = j; i < n; ++i) {
                mpfr_abs(abs_val.get(), A.at(i, j), MPFR_RNDN);
                mpfr_add(col_sums.at(j, 0), col_sums.at(j, 0), abs_val.get(), MPFR_RNDN);
                if (i != j)
                    mpfr_add(col_sums.at(i, 0), col_sums.at(i, 0), abs_val.get(), MPFR_RNDN);
            }
        }
    }
    for (int j = 0; j < n; ++j)
        if (mpfr_cmp(col_sums.at(j, 0), max_norm.get()) > 0)
            mpfr_set(max_norm.get(), col_sums.at(j, 0), MPFR_RNDN);

    return mpfr_get_d(max_norm.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Set matrix to identity                                              */
/* ------------------------------------------------------------------ */

inline void mpfr_mat_set_identity(MpfrMatrix &A)
{
    for (int j = 0; j < A.cols(); ++j)
        for (int i = 0; i < A.rows(); ++i)
            mpfr_set_d(A.at(i, j), (i == j) ? 1.0 : 0.0, MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Copy matrix: dst = src                                              */
/* ------------------------------------------------------------------ */

inline void mpfr_mat_copy(MpfrMatrix &dst, const MpfrMatrix &src)
{
    for (int j = 0; j < src.cols(); ++j)
        for (int i = 0; i < src.rows(); ++i)
            mpfr_set(dst.at(i, j), src.at(i, j), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Apply row permutation from LAPACK IPIV (1-based, sequential swaps)  */
/* P * A: for i = 0..n-1, swap rows i and ipiv[i]-1                   */
/* ------------------------------------------------------------------ */

inline void mpfr_apply_ipiv_rows(MpfrMatrix &A, const int *ipiv, int n, bool forward = true)
{
    if (forward) {
        for (int i = 0; i < n; ++i) {
            int ip = ipiv[i] - 1; /* LAPACK is 1-based */
            if (ip != i) {
                for (int j = 0; j < A.cols(); ++j) {
                    mpfr_swap(A.at(i, j), A.at(ip, j));
                }
            }
        }
    } else {
        /* Reverse order: for i = n-1..0, swap rows i and ipiv[i]-1 */
        for (int i = n - 1; i >= 0; --i) {
            int ip = ipiv[i] - 1;
            if (ip != i) {
                for (int j = 0; j < A.cols(); ++j) {
                    mpfr_swap(A.at(i, j), A.at(ip, j));
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Extract L and U from GETRF in-place factored matrix                 */
/* A is m-by-n; L is m-by-min(m,n) unit lower; U is min(m,n)-by-n     */
/* ------------------------------------------------------------------ */

inline void mpfr_extract_LU(const MpfrMatrix &A, MpfrMatrix &L, MpfrMatrix &U)
{
    int m = A.rows(), n = A.cols();
    int mn = std::min(m, n);

    for (int j = 0; j < mn; ++j) {
        for (int i = 0; i < m; ++i) {
            if (i < j)
                mpfr_set_d(L.at(i, j), 0.0, MPFR_RNDN);
            else if (i == j)
                mpfr_set_d(L.at(i, j), 1.0, MPFR_RNDN);
            else
                mpfr_set(L.at(i, j), A.at(i, j), MPFR_RNDN);
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < mn; ++i) {
            if (i <= j)
                mpfr_set(U.at(i, j), A.at(i, j), MPFR_RNDN);
            else
                mpfr_set_d(U.at(i, j), 0.0, MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Extract lower/upper triangular from a matrix                        */
/* ------------------------------------------------------------------ */

inline void mpfr_extract_triangle(const MpfrMatrix &A, MpfrMatrix &T, char uplo)
{
    int m = A.rows(), n = A.cols();
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (uplo == 'L') {
                if (i >= j) mpfr_set(T.at(i, j), A.at(i, j), MPFR_RNDN);
                else mpfr_set_d(T.at(i, j), 0.0, MPFR_RNDN);
            } else {
                if (i <= j) mpfr_set(T.at(i, j), A.at(i, j), MPFR_RNDN);
                else mpfr_set_d(T.at(i, j), 0.0, MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Accumulate Householder reflectors into explicit Q matrix            */
/* Given reflectors stored in columns of V and tau array:              */
/* Q = (I - tau_1 v_1 v_1^T) * ... * (I - tau_k v_k v_k^T)          */
/* For QR: reflectors below diagonal of A, v_i has 1 at position i    */
/* ------------------------------------------------------------------ */

inline void mpfr_accumulate_Q_from_QR(MpfrMatrix &Q,
                                       const MpfrMatrix &A,
                                       const MpfrMatrix &tau_vec,
                                       int m, int n)
{
    int k = std::min(m, n);
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));

    mpfr_mat_set_identity(Q);

    /* Apply reflectors in reverse order: Q = H_1 * H_2 * ... * H_k
       We build Q by applying H_k, H_{k-1}, ..., H_1 from the right.
       Actually: Q = H_1 H_2 ... H_k, so we start from H_k and work backward.
       Q = I, then for i=k-1..0: Q = (I - tau_i v_i v_i^T) * Q */
    MpfrScalar dot(prec), scale(prec);

    for (int i = k - 1; i >= 0; --i) {
        /* Build v_i: v(j) = 0 for j<i, v(i) = 1, v(j) = A(j,i) for j>i */

        /* Apply H_i = I - tau_i * v_i * v_i^T to Q from the left:
           Q = Q - tau_i * v_i * (v_i^T * Q) */
        /* For each column j of Q:
           Q(:,j) = Q(:,j) - tau_i * v_i * (v_i . Q(:,j)) */
        for (int j = 0; j < m; ++j) {
            /* Compute v_i^T * Q(:,j) */
            mpfr_set(dot.get(), Q.at(i, j), MPFR_RNDN); /* v(i)=1 */
            for (int r = i + 1; r < m; ++r)
                mpfr_fma(dot.get(), A.at(r, i), Q.at(r, j), dot.get(), MPFR_RNDN);

            /* scale = tau_i * dot */
            mpfr_mul(scale.get(), tau_vec.at(i, 0), dot.get(), MPFR_RNDN);

            /* Q(i,j) -= scale * 1 */
            mpfr_sub(Q.at(i, j), Q.at(i, j), scale.get(), MPFR_RNDN);
            /* Q(r,j) -= scale * A(r,i) for r > i */
            for (int r = i + 1; r < m; ++r) {
                MpfrScalar tmp(prec);
                mpfr_mul(tmp.get(), scale.get(), A.at(r, i), MPFR_RNDN);
                mpfr_sub(Q.at(r, j), Q.at(r, j), tmp.get(), MPFR_RNDN);
            }
        }
    }
}

/* Same for LQ: reflectors stored in rows of A, above diagonal        */
/* Q = (I - tau_1 v_1 v_1^T) * ... * (I - tau_k v_k v_k^T)          */
/* For LQ: v_i has 1 at position i, v(j) = A(i,j) for j > i         */

inline void mpfr_accumulate_Q_from_LQ(MpfrMatrix &Q,
                                       const MpfrMatrix &A,
                                       const MpfrMatrix &tau_vec,
                                       int m, int n)
{
    int k = std::min(m, n);
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));

    /* Q is n-by-n */
    mpfr_mat_set_identity(Q);

    MpfrScalar dot(prec), scale(prec);

    /* LAPACK GELQF: Q = H(k) * ... * H(2) * H(1) (1-based)
       So iterate forward i=0..k-1, applying H(i) from the left:
       result = H(k-1)*...*H(1)*H(0) = H(k)*...*H(2)*H(1) in 1-based */
    for (int i = 0; i < k; ++i) {
        /* v_i: v(j) = 0 for j<i, v(i) = 1, v(j) = A(i,j) for j>i */
        for (int j = 0; j < n; ++j) {
            mpfr_set(dot.get(), Q.at(i, j), MPFR_RNDN);
            for (int r = i + 1; r < n; ++r)
                mpfr_fma(dot.get(), A.at(i, r), Q.at(r, j), dot.get(), MPFR_RNDN);

            mpfr_mul(scale.get(), tau_vec.at(i, 0), dot.get(), MPFR_RNDN);

            mpfr_sub(Q.at(i, j), Q.at(i, j), scale.get(), MPFR_RNDN);
            for (int r = i + 1; r < n; ++r) {
                MpfrScalar tmp(prec);
                mpfr_mul(tmp.get(), scale.get(), A.at(i, r), MPFR_RNDN);
                mpfr_sub(Q.at(r, j), Q.at(r, j), tmp.get(), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Compute orthogonality metric: ||Q^T Q - I||_1 / (n * eps)          */
/* ------------------------------------------------------------------ */

inline double mpfr_orthogonality(const MpfrMatrix &Q, double eps)
{
    int m = Q.rows(), n = Q.cols();
    mpfr_prec_t prec = mpfr_get_prec(Q.at(0, 0));

    /* Compute Q^T * Q */
    MpfrMatrix Qt(n, m, prec);
    mpfr_mat_transpose(Qt, Q);

    MpfrMatrix QtQ(n, n, prec);
    mpfr_mat_mul_simple(QtQ, Qt, Q);

    /* Subtract identity */
    for (int i = 0; i < n; ++i) {
        MpfrScalar one(prec);
        mpfr_set_d(one.get(), 1.0, MPFR_RNDN);
        mpfr_sub(QtQ.at(i, i), QtQ.at(i, i), one.get(), MPFR_RNDN);
    }

    double norm = mpfr_mat_norm1(QtQ);
    return norm / (std::max(m, n) * eps);
}

/* ------------------------------------------------------------------ */
/* Compute solve residual: ||AX - B||_1 / (||A||_1 * ||X||_1 * n * eps) */
/* ------------------------------------------------------------------ */

inline double mpfr_solve_residual(const MpfrMatrix &A,
                                   const MpfrMatrix &X,
                                   const MpfrMatrix &B,
                                   double eps)
{
    int n = A.rows();
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));

    MpfrMatrix AX(n, X.cols(), prec);
    mpfr_mat_mul_simple(AX, A, X);

    MpfrMatrix R(n, X.cols(), prec);
    mpfr_mat_sub(R, AX, B);

    double norm_R = mpfr_mat_norm1(R);
    double norm_A = mpfr_mat_norm1(A);
    double norm_X = mpfr_mat_norm1(X);

    if (norm_A == 0.0 || norm_X == 0.0) return 0.0;
    return norm_R / (norm_A * norm_X * n * eps);
}

/* ------------------------------------------------------------------ */
/* Machine epsilon for the type under test (based on typesize)         */
/* ------------------------------------------------------------------ */

inline double get_eps(const TesterCtx &ctx)
{
    /* Approximate: for IEEE types, eps ~ 2^{-p+1} where p is significand bits */
    if (ctx.typesize == 4) return 5.96e-8;          /* float: 2^-23 */
    if (ctx.typesize == 8) return 1.11e-16;          /* double: 2^-52 */
    if (ctx.typesize == 16) return 9.63e-35;         /* quad: 2^-112 (IEEE) or ~1e-31 (long double 80-bit) */
    /* Conservative default for unknown types */
    return 1.11e-16;
}

/* gen_positive_definite_array is declared in generators.h */
