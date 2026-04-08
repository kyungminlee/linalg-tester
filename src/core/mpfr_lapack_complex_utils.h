/* mpfr_lapack_complex_utils.h -- MPFR utilities for complex LAPACK residual computation */

#pragma once

#include "mpfr_complex_types.h"
#include "mpfr_complex.h"
#include "generators.h"
#include <algorithm>
#include <cstdlib>

/* ------------------------------------------------------------------ */
/* Complex matrix multiply: C = A * B                                  */
/* A is m-by-k, B is k-by-n, C is m-by-n                              */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_mat_mul_simple(MpfrComplexMatrix &C,
                                         const MpfrComplexMatrix &A,
                                         const MpfrComplexMatrix &B)
{
    int m = A.rows(), k = A.cols(), n = B.cols();

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(C.re(i, j), 0.0, MPFR_RNDN);
            mpfr_set_d(C.im(i, j), 0.0, MPFR_RNDN);
            for (int p = 0; p < k; ++p)
                mpfr_complex_fma(C.re(i, j), C.im(i, j),
                                 A.re(i, p), A.im(i, p),
                                 B.re(p, j), B.im(p, j),
                                 MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Complex matrix adjoint (conjugate transpose): B = A^H               */
/* A is m-by-n, B is n-by-m                                           */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_mat_adjoint(MpfrComplexMatrix &B,
                                      const MpfrComplexMatrix &A)
{
    for (int j = 0; j < A.cols(); ++j)
        for (int i = 0; i < A.rows(); ++i)
            mpfr_complex_conj(B.re(j, i), B.im(j, i),
                              A.re(i, j), A.im(i, j),
                              MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Complex matrix subtraction: C = A - B                               */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_mat_sub(MpfrComplexMatrix &C,
                                  const MpfrComplexMatrix &A,
                                  const MpfrComplexMatrix &B)
{
    for (int j = 0; j < A.cols(); ++j)
        for (int i = 0; i < A.rows(); ++i)
            mpfr_complex_sub(C.re(i, j), C.im(i, j),
                             A.re(i, j), A.im(i, j),
                             B.re(i, j), B.im(i, j),
                             MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Complex matrix 1-norm: max over columns of sum of |A(i,j)|          */
/* ------------------------------------------------------------------ */

inline double mpfr_complex_mat_norm1(const MpfrComplexMatrix &A)
{
    int m = A.rows(), n = A.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    MpfrScalar col_sum(prec), abs_val(prec), max_norm(prec);
    mpfr_set_d(max_norm.get(), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j) {
        mpfr_set_d(col_sum.get(), 0.0, MPFR_RNDN);
        for (int i = 0; i < m; ++i) {
            mpfr_complex_abs(abs_val.get(),
                             A.re(i, j), A.im(i, j),
                             MPFR_RNDN);
            mpfr_add(col_sum.get(), col_sum.get(), abs_val.get(), MPFR_RNDN);
        }
        if (mpfr_cmp(col_sum.get(), max_norm.get()) > 0)
            mpfr_set(max_norm.get(), col_sum.get(), MPFR_RNDN);
    }
    return mpfr_get_d(max_norm.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Complex Frobenius norm: sqrt(sum |A(i,j)|^2)                        */
/* ------------------------------------------------------------------ */

inline double mpfr_complex_mat_normF(const MpfrComplexMatrix &A)
{
    int m = A.rows(), n = A.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    MpfrScalar sum(prec);
    mpfr_set_d(sum.get(), 0.0, MPFR_RNDN);

    /* Accumulate re^2 + im^2 for each element using fma */
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_fma(sum.get(), A.re(i, j), A.re(i, j), sum.get(), MPFR_RNDN);
            mpfr_fma(sum.get(), A.im(i, j), A.im(i, j), sum.get(), MPFR_RNDN);
        }
    }

    mpfr_sqrt(sum.get(), sum.get(), MPFR_RNDN);
    return mpfr_get_d(sum.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Set complex matrix to identity                                      */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_mat_set_identity(MpfrComplexMatrix &A)
{
    for (int j = 0; j < A.cols(); ++j) {
        for (int i = 0; i < A.rows(); ++i) {
            mpfr_set_d(A.re(i, j), (i == j) ? 1.0 : 0.0, MPFR_RNDN);
            mpfr_set_d(A.im(i, j), 0.0, MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Copy complex matrix: dst = src                                      */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_mat_copy(MpfrComplexMatrix &dst,
                                   const MpfrComplexMatrix &src)
{
    for (int j = 0; j < src.cols(); ++j) {
        for (int i = 0; i < src.rows(); ++i) {
            mpfr_set(dst.re(i, j), src.re(i, j), MPFR_RNDN);
            mpfr_set(dst.im(i, j), src.im(i, j), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Apply row permutation from LAPACK IPIV to complex matrix            */
/* P * A: for i = 0..n-1, swap rows i and ipiv[i]-1                   */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_apply_ipiv_rows(MpfrComplexMatrix &A,
                                          const int *ipiv, int n,
                                          bool forward = true)
{
    MpfrMatrix &re_ = A.real_part();
    MpfrMatrix &im_ = A.imag_part();

    if (forward) {
        for (int i = 0; i < n; ++i) {
            int ip = ipiv[i] - 1; /* LAPACK is 1-based */
            if (ip != i) {
                for (int j = 0; j < A.cols(); ++j) {
                    mpfr_swap(re_.at(i, j), re_.at(ip, j));
                    mpfr_swap(im_.at(i, j), im_.at(ip, j));
                }
            }
        }
    } else {
        /* Reverse order: for i = n-1..0, swap rows i and ipiv[i]-1 */
        for (int i = n - 1; i >= 0; --i) {
            int ip = ipiv[i] - 1;
            if (ip != i) {
                for (int j = 0; j < A.cols(); ++j) {
                    mpfr_swap(re_.at(i, j), re_.at(ip, j));
                    mpfr_swap(im_.at(i, j), im_.at(ip, j));
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Extract L and U from GETRF in-place factored complex matrix         */
/* A is m-by-n; L is m-by-min(m,n) unit lower; U is min(m,n)-by-n     */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_extract_LU(const MpfrComplexMatrix &A,
                                     MpfrComplexMatrix &L,
                                     MpfrComplexMatrix &U)
{
    int m = A.rows(), n = A.cols();
    int mn = std::min(m, n);

    for (int j = 0; j < mn; ++j) {
        for (int i = 0; i < m; ++i) {
            if (i < j) {
                mpfr_set_d(L.re(i, j), 0.0, MPFR_RNDN);
                mpfr_set_d(L.im(i, j), 0.0, MPFR_RNDN);
            } else if (i == j) {
                mpfr_set_d(L.re(i, j), 1.0, MPFR_RNDN);
                mpfr_set_d(L.im(i, j), 0.0, MPFR_RNDN);
            } else {
                mpfr_set(L.re(i, j), A.re(i, j), MPFR_RNDN);
                mpfr_set(L.im(i, j), A.im(i, j), MPFR_RNDN);
            }
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < mn; ++i) {
            if (i <= j) {
                mpfr_set(U.re(i, j), A.re(i, j), MPFR_RNDN);
                mpfr_set(U.im(i, j), A.im(i, j), MPFR_RNDN);
            } else {
                mpfr_set_d(U.re(i, j), 0.0, MPFR_RNDN);
                mpfr_set_d(U.im(i, j), 0.0, MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Extract lower/upper triangular from complex matrix                  */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_extract_triangle(const MpfrComplexMatrix &A,
                                           MpfrComplexMatrix &T,
                                           char uplo)
{
    int m = A.rows(), n = A.cols();
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (uplo == 'L') {
                if (i >= j) {
                    mpfr_set(T.re(i, j), A.re(i, j), MPFR_RNDN);
                    mpfr_set(T.im(i, j), A.im(i, j), MPFR_RNDN);
                } else {
                    mpfr_set_d(T.re(i, j), 0.0, MPFR_RNDN);
                    mpfr_set_d(T.im(i, j), 0.0, MPFR_RNDN);
                }
            } else {
                if (i <= j) {
                    mpfr_set(T.re(i, j), A.re(i, j), MPFR_RNDN);
                    mpfr_set(T.im(i, j), A.im(i, j), MPFR_RNDN);
                } else {
                    mpfr_set_d(T.re(i, j), 0.0, MPFR_RNDN);
                    mpfr_set_d(T.im(i, j), 0.0, MPFR_RNDN);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Accumulate complex Householder reflectors into explicit Q (QR)      */
/* H_i = I - tau_i * v_i * v_i^H  (conjugate transpose)               */
/* Q = H_1 * H_2 * ... * H_k; built by applying from H_k backward    */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_accumulate_Q_from_QR(MpfrComplexMatrix &Q,
                                               const MpfrComplexMatrix &A,
                                               const MpfrComplexMatrix &tau_vec,
                                               int m, int n)
{
    int k = std::min(m, n);
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));

    mpfr_complex_mat_set_identity(Q);

    /* Temporaries for complex dot product and scale */
    MpfrComplexScalar dot(prec), scale(prec), tmp(prec);

    for (int i = k - 1; i >= 0; --i) {
        /* Apply H_i = I - tau_i * v_i * v_i^H to Q from the left:
           For each column j of Q:
             dot = conj(v_i)^T * Q(:,j)
             Q(:,j) -= tau_i * v_i * dot
           where v(i)=1, v(r)=A(r,i) for r>i */
        for (int j = 0; j < m; ++j) {
            /* Compute dot = conj(v_i)^T * Q(:,j)
               dot = conj(1) * Q(i,j) + sum_{r>i} conj(A(r,i)) * Q(r,j)
               dot = Q(i,j) + sum_{r>i} conj(A(r,i)) * Q(r,j) */
            mpfr_set(dot.re(), Q.re(i, j), MPFR_RNDN);
            mpfr_set(dot.im(), Q.im(i, j), MPFR_RNDN);
            for (int r = i + 1; r < m; ++r) {
                /* dot += conj(A(r,i)) * Q(r,j) */
                MpfrComplexScalar cv(prec);
                mpfr_complex_conj(cv.re(), cv.im(),
                                  A.re(r, i), A.im(r, i),
                                  MPFR_RNDN);
                mpfr_complex_fma(dot.re(), dot.im(),
                                 cv.re(), cv.im(),
                                 Q.re(r, j), Q.im(r, j),
                                 MPFR_RNDN);
            }

            /* scale = tau_i * dot */
            mpfr_complex_mul(scale.re(), scale.im(),
                             tau_vec.re(i, 0), tau_vec.im(i, 0),
                             dot.re(), dot.im(),
                             MPFR_RNDN);

            /* Q(i,j) -= scale * 1 */
            mpfr_sub(Q.re(i, j), Q.re(i, j), scale.re(), MPFR_RNDN);
            mpfr_sub(Q.im(i, j), Q.im(i, j), scale.im(), MPFR_RNDN);

            /* Q(r,j) -= scale * v(r) = scale * A(r,i) for r > i */
            for (int r = i + 1; r < m; ++r) {
                mpfr_complex_mul(tmp.re(), tmp.im(),
                                 scale.re(), scale.im(),
                                 A.re(r, i), A.im(r, i),
                                 MPFR_RNDN);
                mpfr_sub(Q.re(r, j), Q.re(r, j), tmp.re(), MPFR_RNDN);
                mpfr_sub(Q.im(r, j), Q.im(r, j), tmp.im(), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Accumulate complex Householder reflectors into explicit Q (LQ)      */
/* H_i = I - conj(tau_i) * v_i * v_i^H                                */
/* For LQ: v(i)=1, v(r)=A(i,r) for r>i (stored in rows)              */
/* Iterate forward i=0..k-1                                            */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_accumulate_Q_from_LQ(MpfrComplexMatrix &Q,
                                               const MpfrComplexMatrix &A,
                                               const MpfrComplexMatrix &tau_vec,
                                               int m, int n)
{
    int k = std::min(m, n);
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));

    /* Q is n-by-n */
    mpfr_complex_mat_set_identity(Q);

    MpfrComplexScalar dot(prec), scale(prec), tmp(prec), ctau(prec);

    /* LAPACK ZGELQF: Q = H(k)^H * ... * H(1)^H where H(i) = I - tau*v*v^H.
       LAPACK stores conj(v) in A rows: A(i,r) = conj(v(r)) for r > i.
       So v(r) = conj(A(i,r)).
       H(i)^H = I - conj(tau) * v * v^H.
       Apply from left: Q(:,j) -= conj(tau) * v * (v^H * Q(:,j))
       Inner product: v^H * Q = conj(v)^T * Q. Since conj(v(r))=A(i,r):
         dot = Q(i,j) + sum_{r>i} A(i,r) * Q(r,j)  [NO conjugation of A]
       Application: v(r) = conj(A(i,r)):
         Q(r,j) -= scale * conj(A(i,r))             [conjugate A] */
    for (int i = 0; i < k; ++i) {
        mpfr_complex_conj(ctau.re(), ctau.im(),
                          tau_vec.re(i, 0), tau_vec.im(i, 0),
                          MPFR_RNDN);

        for (int j = 0; j < n; ++j) {
            /* dot = v^H * Q(:,j) = Q(i,j) + sum A(i,r)*Q(r,j) */
            mpfr_set(dot.re(), Q.re(i, j), MPFR_RNDN);
            mpfr_set(dot.im(), Q.im(i, j), MPFR_RNDN);
            for (int r = i + 1; r < n; ++r) {
                mpfr_complex_fma(dot.re(), dot.im(),
                                 A.re(i, r), A.im(i, r),
                                 Q.re(r, j), Q.im(r, j),
                                 MPFR_RNDN);
            }

            /* scale = conj(tau_i) * dot */
            mpfr_complex_mul(scale.re(), scale.im(),
                             ctau.re(), ctau.im(),
                             dot.re(), dot.im(),
                             MPFR_RNDN);

            /* Q(i,j) -= scale * v(i) = scale * 1 */
            mpfr_sub(Q.re(i, j), Q.re(i, j), scale.re(), MPFR_RNDN);
            mpfr_sub(Q.im(i, j), Q.im(i, j), scale.im(), MPFR_RNDN);

            /* Q(r,j) -= scale * v(r) = scale * conj(A(i,r)) for r > i */
            for (int r = i + 1; r < n; ++r) {
                MpfrComplexScalar cv(prec);
                mpfr_complex_conj(cv.re(), cv.im(),
                                  A.re(i, r), A.im(i, r),
                                  MPFR_RNDN);
                mpfr_complex_mul(tmp.re(), tmp.im(),
                                 scale.re(), scale.im(),
                                 cv.re(), cv.im(),
                                 MPFR_RNDN);
                mpfr_sub(Q.re(r, j), Q.re(r, j), tmp.re(), MPFR_RNDN);
                mpfr_sub(Q.im(r, j), Q.im(r, j), tmp.im(), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Complex orthogonality metric: ||Q^H*Q - I||_1 / (max(m,n) * eps)   */
/* ------------------------------------------------------------------ */

inline double mpfr_complex_orthogonality(const MpfrComplexMatrix &Q,
                                          double eps)
{
    int m = Q.rows(), n = Q.cols();
    mpfr_prec_t prec = mpfr_get_prec(Q.re(0, 0));

    /* Compute Q^H */
    MpfrComplexMatrix Qh(n, m, prec);
    mpfr_complex_mat_adjoint(Qh, Q);

    /* Compute Q^H * Q */
    MpfrComplexMatrix QhQ(n, n, prec);
    mpfr_complex_mat_mul_simple(QhQ, Qh, Q);

    /* Subtract identity */
    for (int i = 0; i < n; ++i) {
        MpfrScalar one(prec);
        mpfr_set_d(one.get(), 1.0, MPFR_RNDN);
        mpfr_sub(QhQ.re(i, i), QhQ.re(i, i), one.get(), MPFR_RNDN);
    }

    double norm = mpfr_complex_mat_norm1(QhQ);
    return norm / (std::max(m, n) * eps);
}

/* ------------------------------------------------------------------ */
/* Complex solve residual: ||AX-B||_1 / (||A||_1 * ||X||_1 * n * eps) */
/* ------------------------------------------------------------------ */

inline double mpfr_complex_solve_residual(const MpfrComplexMatrix &A,
                                           const MpfrComplexMatrix &X,
                                           const MpfrComplexMatrix &B,
                                           double eps)
{
    int n = A.rows();
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));

    MpfrComplexMatrix AX(n, X.cols(), prec);
    mpfr_complex_mat_mul_simple(AX, A, X);

    MpfrComplexMatrix R(n, X.cols(), prec);
    mpfr_complex_mat_sub(R, AX, B);

    double norm_R = mpfr_complex_mat_norm1(R);
    double norm_A = mpfr_complex_mat_norm1(A);
    double norm_X = mpfr_complex_mat_norm1(X);

    if (norm_A == 0.0 || norm_X == 0.0) return 0.0;
    return norm_R / (norm_A * norm_X * n * eps);
}

/* ------------------------------------------------------------------ */
/* Hermitian 1-norm (only reads one triangle, uses |z| for entries)    */
/* ------------------------------------------------------------------ */

inline double mpfr_hermat_norm1(const MpfrComplexMatrix &A, char uplo)
{
    int n = A.rows();
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    MpfrScalar abs_val(prec), max_norm(prec);
    mpfr_set_d(max_norm.get(), 0.0, MPFR_RNDN);

    /* col_sums accumulates column sums using Hermitian symmetry */
    MpfrMatrix col_sums(n, 1, prec);
    for (int j = 0; j < n; ++j)
        mpfr_set_d(col_sums.at(j, 0), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j) {
        if (uplo == 'U') {
            for (int i = 0; i <= j; ++i) {
                mpfr_complex_abs(abs_val.get(),
                                 A.re(i, j), A.im(i, j),
                                 MPFR_RNDN);
                mpfr_add(col_sums.at(j, 0), col_sums.at(j, 0), abs_val.get(), MPFR_RNDN);
                if (i != j)
                    mpfr_add(col_sums.at(i, 0), col_sums.at(i, 0), abs_val.get(), MPFR_RNDN);
            }
        } else {
            for (int i = j; i < n; ++i) {
                mpfr_complex_abs(abs_val.get(),
                                 A.re(i, j), A.im(i, j),
                                 MPFR_RNDN);
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
