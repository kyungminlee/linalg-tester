#include "reference.h"

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstdio>
#include <memory>

/* Column-major index */
static inline std::size_t IDX(int i, int j, int ld)
{
    return static_cast<std::size_t>(j) * ld + i;
}

/* ------------------------------------------------------------------ */
/* RAII wrapper for an MPFR matrix (column-major)                       */
/* ------------------------------------------------------------------ */

class MpfrMatrix {
public:
    MpfrMatrix(int rows, int cols, mpfr_prec_t prec)
        : rows_(rows), cols_(cols)
    {
        std::size_t n = static_cast<std::size_t>(rows) * cols;
        data_ = new mpfr_t[n];
        for (std::size_t i = 0; i < n; ++i)
            mpfr_init2(data_[i], prec);
    }

    ~MpfrMatrix() {
        std::size_t n = static_cast<std::size_t>(rows_) * cols_;
        for (std::size_t i = 0; i < n; ++i)
            mpfr_clear(data_[i]);
        delete[] data_;
    }

    /* Non-copyable, non-movable (mpfr_t is an array type internally) */
    MpfrMatrix(const MpfrMatrix &) = delete;
    MpfrMatrix &operator=(const MpfrMatrix &) = delete;

    mpfr_t &at(int i, int j)       { return data_[IDX(i, j, rows_)]; }
    const mpfr_t &at(int i, int j) const { return data_[IDX(i, j, rows_)]; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

private:
    mpfr_t *data_;
    int     rows_;
    int     cols_;
};

/* RAII wrapper for a single mpfr_t */
class MpfrScalar {
public:
    explicit MpfrScalar(mpfr_prec_t prec) { mpfr_init2(val_, prec); }
    ~MpfrScalar() { mpfr_clear(val_); }

    MpfrScalar(const MpfrScalar &) = delete;
    MpfrScalar &operator=(const MpfrScalar &) = delete;

    mpfr_t &get() { return val_; }

private:
    mpfr_t val_;
};

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

/* Convert a column-major custom matrix to an MpfrMatrix. */
static void custom_to_mpfr_mat(MpfrMatrix &dst, const void *src,
                                int ld, const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(src);
    for (int j = 0; j < dst.cols(); ++j)
        for (int i = 0; i < dst.rows(); ++i)
            ctx.to_mpfr(dst.at(i, j),
                        p + IDX(i, j, ld) * ctx.typesize);
}

/* Compute error metrics between MPFR reference and opaque custom output. */
static ErrorResult compute_error_metrics(
    const MpfrMatrix &ref, const void *custom_out,
    int ld, const TesterCtx &ctx)
{
    mpfr_prec_t prec = ctx.prec;
    MpfrScalar val(prec), diff(prec), absdiff(prec), absref(prec);
    MpfrScalar ratio(prec), sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);

    const char *p = static_cast<const char *>(custom_out);
    for (int j = 0; j < ref.cols(); ++j) {
        for (int i = 0; i < ref.rows(); ++i) {
            ctx.to_mpfr(val.get(), p + IDX(i, j, ld) * ctx.typesize);

            mpfr_sub(diff.get(), val.get(), ref.at(i, j), MPFR_RNDN);
            mpfr_abs(absdiff.get(), diff.get(), MPFR_RNDN);
            mpfr_abs(absref.get(),  ref.at(i, j), MPFR_RNDN);

            mpfr_fma(sum_sq_diff.get(), diff.get(), diff.get(),
                     sum_sq_diff.get(), MPFR_RNDN);
            mpfr_fma(sum_sq_ref.get(),  ref.at(i, j), ref.at(i, j),
                     sum_sq_ref.get(),  MPFR_RNDN);

            if (mpfr_zero_p(absref.get()))
                continue;
            mpfr_div(ratio.get(), absdiff.get(), absref.get(), MPFR_RNDN);
            if (mpfr_cmp(ratio.get(), max_rel.get()) > 0)
                mpfr_set(max_rel.get(), ratio.get(), MPFR_RNDN);
        }
    }

    ErrorResult result;
    result.max_relative = mpfr_get_d(max_rel.get(), MPFR_RNDN);

    if (mpfr_zero_p(sum_sq_ref.get())) {
        result.normwise_relative = 0.0;
    } else {
        mpfr_sqrt(sum_sq_diff.get(), sum_sq_diff.get(), MPFR_RNDN);
        mpfr_sqrt(sum_sq_ref.get(),  sum_sq_ref.get(),  MPFR_RNDN);
        mpfr_div(ratio.get(), sum_sq_diff.get(), sum_sq_ref.get(), MPFR_RNDN);
        result.normwise_relative = mpfr_get_d(ratio.get(), MPFR_RNDN);
    }

    return result;
}

/* ------------------------------------------------------------------ */
/* MPFR reference GEMM                                                  */
/* C_ref = alpha * op(A) * op(B) + beta * C_in                         */
/* ------------------------------------------------------------------ */

static void mpfr_gemm_ref(MpfrMatrix &C_ref,
                           char transa, char transb,
                           int m, int n, int k,
                           mpfr_t alpha,
                           const MpfrMatrix &A,
                           const MpfrMatrix &B,
                           mpfr_t beta,
                           const MpfrMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);

    MpfrScalar acc(prec), alpha_acc(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
            for (int p = 0; p < k; ++p) {
                const mpfr_t &a_ip = (transa == 'N')
                    ? A.at(i, p) : A.at(p, i);
                const mpfr_t &b_pj = (transb == 'N')
                    ? B.at(p, j) : B.at(j, p);
                mpfr_fma(acc.get(), a_ip, b_pj, acc.get(), MPFR_RNDN);
            }
            mpfr_mul(alpha_acc.get(), alpha, acc.get(), MPFR_RNDN);
            mpfr_mul(beta_c.get(),   beta,  C_in.at(i, j), MPFR_RNDN);
            mpfr_add(C_ref.at(i, j), alpha_acc.get(), beta_c.get(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* MPFR reference TRSM                                                  */
/* Solves op(A)*X=alpha*B (side='L') or X*op(A)=alpha*B (side='R').    */
/* X is modified in place (starts as a copy of B_in).                  */
/* ------------------------------------------------------------------ */

static void mpfr_trsm_ref(MpfrMatrix &X,
                            char side, char uplo, char transa, char diag,
                            int m, int n,
                            mpfr_t alpha,
                            const MpfrMatrix &A)
{
    int k = (side == 'L') ? m : n;
    mpfr_prec_t prec = mpfr_get_prec(alpha);

    /* Scale X by alpha */
    {
        MpfrScalar tmp(prec);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                mpfr_mul(X.at(i, j), alpha, X.at(i, j), MPFR_RNDN);
    }

    /* Build working matrix Aw: extract relevant triangle, zero other,
       set unit diagonal if diag=='U'.                                  */
    MpfrMatrix Aw(k, k, prec);
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < k; ++i) {
            if ((uplo == 'U' && j >= i) || (uplo == 'L' && j <= i))
                mpfr_set(Aw.at(i, j), A.at(i, j), MPFR_RNDN);
            else
                mpfr_set_d(Aw.at(i, j), 0.0, MPFR_RNDN);

            if (diag == 'U' && i == j)
                mpfr_set_d(Aw.at(i, j), 1.0, MPFR_RNDN);
        }
    }

    /* If transa != 'N', transpose Aw in place. */
    if (transa != 'N') {
        MpfrMatrix Awt(k, k, prec);
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < k; ++i)
                mpfr_set(Awt.at(i, j), Aw.at(j, i), MPFR_RNDN);
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < k; ++i)
                mpfr_set(Aw.at(i, j), Awt.at(i, j), MPFR_RNDN);
    }

    /* After possible transpose: is Aw effectively upper triangular?
       solve_upper = (uplo=='U') XOR (transa!='N')                    */
    bool solve_upper = ((uplo == 'U') != (transa != 'N'));

    MpfrScalar tmp(prec);

    if (side == 'L') {
        /* Solve Aw * X[:,j] = X[:,j] for each column j */
        for (int j = 0; j < n; ++j) {
            if (solve_upper) {
                for (int i = k - 1; i >= 0; --i) {
                    for (int p = i + 1; p < k; ++p) {
                        mpfr_mul(tmp.get(), Aw.at(i, p), X.at(p, j), MPFR_RNDN);
                        mpfr_sub(X.at(i, j), X.at(i, j), tmp.get(), MPFR_RNDN);
                    }
                    mpfr_div(X.at(i, j), X.at(i, j), Aw.at(i, i), MPFR_RNDN);
                }
            } else {
                for (int i = 0; i < k; ++i) {
                    for (int p = 0; p < i; ++p) {
                        mpfr_mul(tmp.get(), Aw.at(i, p), X.at(p, j), MPFR_RNDN);
                        mpfr_sub(X.at(i, j), X.at(i, j), tmp.get(), MPFR_RNDN);
                    }
                    mpfr_div(X.at(i, j), X.at(i, j), Aw.at(i, i), MPFR_RNDN);
                }
            }
        }
    } else {
        /* side == 'R': solve X[i,:] * Aw = X[i,:] for each row i.
           x*Aw=b → column j of Aw:
             x[j]*Aw[j,j] = b[j] - sum_{p!=j, Aw[p,j]!=0} x[p]*Aw[p,j]
           solve_upper (Aw upper): Aw[p,j]=0 for p>j → forward in j
           solve_lower  (Aw lower): Aw[p,j]=0 for p<j → backward in j */
        for (int i = 0; i < m; ++i) {
            if (solve_upper) {
                for (int j = 0; j < k; ++j) {
                    for (int p = 0; p < j; ++p) {
                        mpfr_mul(tmp.get(), X.at(i, p), Aw.at(p, j), MPFR_RNDN);
                        mpfr_sub(X.at(i, j), X.at(i, j), tmp.get(), MPFR_RNDN);
                    }
                    mpfr_div(X.at(i, j), X.at(i, j), Aw.at(j, j), MPFR_RNDN);
                }
            } else {
                for (int j = k - 1; j >= 0; --j) {
                    for (int p = j + 1; p < k; ++p) {
                        mpfr_mul(tmp.get(), X.at(i, p), Aw.at(p, j), MPFR_RNDN);
                        mpfr_sub(X.at(i, j), X.at(i, j), tmp.get(), MPFR_RNDN);
                    }
                    mpfr_div(X.at(i, j), X.at(i, j), Aw.at(j, j), MPFR_RNDN);
                }
            }
        }
    }
}

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
    const void *C_out)
{
    transa = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    transb = static_cast<char>(std::toupper(static_cast<unsigned char>(transb)));

    mpfr_prec_t prec = ctx.prec;

    MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
    ctx.to_mpfr(mpfr_alpha.get(), alpha);
    ctx.to_mpfr(mpfr_beta.get(),  beta);

    int rows_A = (transa == 'N') ? m : k;
    int cols_A = (transa == 'N') ? k : m;
    int rows_B = (transb == 'N') ? k : n;
    int cols_B = (transb == 'N') ? n : k;

    MpfrMatrix A_mpfr(rows_A, cols_A, prec);
    MpfrMatrix B_mpfr(rows_B, cols_B, prec);
    MpfrMatrix C_in_mpfr(m, n, prec);
    MpfrMatrix C_ref(m, n, prec);

    custom_to_mpfr_mat(A_mpfr,    A,    rows_A, ctx);
    custom_to_mpfr_mat(B_mpfr,    B,    rows_B, ctx);
    custom_to_mpfr_mat(C_in_mpfr, C_in, m,      ctx);

    mpfr_gemm_ref(C_ref, transa, transb, m, n, k,
                  mpfr_alpha.get(), A_mpfr, B_mpfr, mpfr_beta.get(), C_in_mpfr);

    return compute_error_metrics(C_ref, C_out, m, ctx);
}

ErrorResult reference_test_trsm(
    const TesterCtx &ctx,
    char side, char uplo, char transa, char diag,
    int m, int n,
    const void *alpha,
    const void *A,
    const void *B_in,
    const void *X_out)
{
    side   = static_cast<char>(std::toupper(static_cast<unsigned char>(side)));
    uplo   = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo)));
    transa = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    diag   = static_cast<char>(std::toupper(static_cast<unsigned char>(diag)));

    mpfr_prec_t prec = ctx.prec;
    int k = (side == 'L') ? m : n;

    MpfrScalar mpfr_alpha(prec);
    ctx.to_mpfr(mpfr_alpha.get(), alpha);

    MpfrMatrix A_mpfr(k, k, prec);
    MpfrMatrix X_ref(m, n, prec);

    custom_to_mpfr_mat(A_mpfr, A,    k, ctx);
    custom_to_mpfr_mat(X_ref,  B_in, m, ctx);

    mpfr_trsm_ref(X_ref, side, uplo, transa, diag,
                  m, n, mpfr_alpha.get(), A_mpfr);

    return compute_error_metrics(X_ref, X_out, m, ctx);
}
