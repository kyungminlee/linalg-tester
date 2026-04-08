/* hetrd.cpp -- LAPACK complex HETRD accuracy tester (Hermitian tridiagonal reduction) */
/* A = Q * T * Q^H where T is real tridiagonal, Q is complex unitary */

#include "../factorizations.h"
#include "../lapack_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/mpfr_lapack_utils.h"  // for get_eps

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Reconstruct Q from complex Householder reflectors produced by HETRD.*/
/* H_i = I - tau_i * v_i * v_i^H                                      */
/*                                                                     */
/* uplo='L': reflectors stored below first subdiagonal of A columns.  */
/*   For i=0..n-2: v(j)=0 for j<=i, v(i+1)=1, v(j)=A(j,i) for j>i+1 */
/*   Q = H_0 H_1 ... H_{n-2}                                         */
/*                                                                     */
/* uplo='U': reflectors stored above first superdiagonal of A columns.*/
/*   For i=n-1..1: v(j)=0 for j>=i, v(i-1)=1, v(j)=A(j,i) for j<i-1 */
/*   Q = H_{n-1} H_{n-2} ... H_1                                     */
/* ------------------------------------------------------------------ */

static void accumulate_Q_hetrd(MpfrComplexMatrix &Q, const MpfrComplexMatrix &A,
                                const MpfrComplexMatrix &tau_vec,
                                int n, char uplo)
{
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    int k = n - 1; /* number of reflectors */
    if (k <= 0) {
        mpfr_complex_mat_set_identity(Q);
        return;
    }

    mpfr_complex_mat_set_identity(Q);
    MpfrComplexScalar dot(prec), scale(prec), tmp(prec);

    if (uplo == 'L') {
        /* Q = H_0 H_1 ... H_{n-2}, build from right:
           start with I, apply H_{n-2}, ..., H_0 from the left */
        for (int i = k - 1; i >= 0; --i) {
            /* v starts at row i+1: v(i+1)=1, v(j)=A(j,i) for j>i+1 */
            int v_start = i + 1;

            for (int j = 0; j < n; ++j) {
                /* dot = conj(v_i)^T * Q(:,j)
                   dot = conj(1)*Q(v_start,j) + sum_{r>v_start} conj(A(r,i))*Q(r,j) */
                mpfr_set(dot.re(), Q.re(v_start, j), MPFR_RNDN);
                mpfr_set(dot.im(), Q.im(v_start, j), MPFR_RNDN);
                for (int r = v_start + 1; r < n; ++r) {
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

                /* Q(v_start,j) -= scale */
                mpfr_sub(Q.re(v_start, j), Q.re(v_start, j), scale.re(), MPFR_RNDN);
                mpfr_sub(Q.im(v_start, j), Q.im(v_start, j), scale.im(), MPFR_RNDN);

                for (int r = v_start + 1; r < n; ++r) {
                    /* Q(r,j) -= scale * A(r,i) */
                    mpfr_complex_mul(tmp.re(), tmp.im(),
                                     scale.re(), scale.im(),
                                     A.re(r, i), A.im(r, i),
                                     MPFR_RNDN);
                    mpfr_sub(Q.re(r, j), Q.re(r, j), tmp.re(), MPFR_RNDN);
                    mpfr_sub(Q.im(r, j), Q.im(r, j), tmp.im(), MPFR_RNDN);
                }
            }
        }
    } else {
        /* uplo='U': Q = H_{n-1} H_{n-2} ... H_1
           Reflector i (for column i, i=1..n-1) has tau index i-1.
           v ends at row i-1: v(i-1)=1, v(j)=A(j,i) for j<i-1, v(j)=0 for j>=i */
        for (int i = 1; i < n; ++i) {
            int ti = i - 1;
            int v_end = i - 1; /* v(v_end) = 1 */

            for (int j = 0; j < n; ++j) {
                /* dot = conj(v)^T * Q(:,j)
                   dot = conj(1)*Q(v_end,j) + sum_{r<v_end} conj(A(r,i))*Q(r,j) */
                mpfr_set(dot.re(), Q.re(v_end, j), MPFR_RNDN);
                mpfr_set(dot.im(), Q.im(v_end, j), MPFR_RNDN);
                for (int r = 0; r < v_end; ++r) {
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
                                 tau_vec.re(ti, 0), tau_vec.im(ti, 0),
                                 dot.re(), dot.im(),
                                 MPFR_RNDN);

                /* Q(v_end,j) -= scale */
                mpfr_sub(Q.re(v_end, j), Q.re(v_end, j), scale.re(), MPFR_RNDN);
                mpfr_sub(Q.im(v_end, j), Q.im(v_end, j), scale.im(), MPFR_RNDN);

                for (int r = 0; r < v_end; ++r) {
                    /* Q(r,j) -= scale * A(r,i) */
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
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_hetrd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int n = params.n;
    int lda = n + params.ld_pad;

    /* Real typesize is half of complex typesize */
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        /* Generate Hermitian matrix (n-by-n, ld=n) */
        void *A_her = gen_hermitian_array(n, uplo, ctx.typesize,
                                          ctx.from_mpfr_complex, prec, &seed_A);

        /* Copy into lda-padded layout */
        void *A = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
        {
            char *dst = static_cast<char *>(A);
            const char *src = static_cast<const char *>(A_her);
            for (int j = 0; j < n; ++j)
                std::memcpy(dst + static_cast<std::size_t>(j) * lda * ctx.typesize,
                            src + static_cast<std::size_t>(j) * n * ctx.typesize,
                            static_cast<std::size_t>(n) * ctx.typesize);
        }

        /* Save A_orig as full Hermitian MPFR matrix */
        MpfrComplexMatrix A_orig(n, n, prec);
        {
            const char *p = static_cast<const char *>(A);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    ctx.to_mpfr_complex(A_orig.re(i, j), A_orig.im(i, j),
                                        p + IDX(i, j, lda) * ctx.typesize);
            /* Fill the other triangle by Hermitian symmetry: A(i,j) = conj(A(j,i)) */
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < j; ++i) {
                    if (uplo == 'U') {
                        mpfr_complex_conj(A_orig.re(j, i), A_orig.im(j, i),
                                          A_orig.re(i, j), A_orig.im(i, j),
                                          MPFR_RNDN);
                    } else {
                        mpfr_complex_conj(A_orig.re(i, j), A_orig.im(i, j),
                                          A_orig.re(j, i), A_orig.im(j, i),
                                          MPFR_RNDN);
                    }
                }
        }

        /* Allocate d, e (real), tau (complex) */
        void *d_arr = std::malloc(static_cast<std::size_t>(n) * real_typesize);
        void *e_arr = std::malloc(static_cast<std::size_t>(std::max(n - 1, 1)) * real_typesize);
        void *tau   = std::malloc(static_cast<std::size_t>(std::max(n - 1, 1)) * ctx.typesize);

        /* Workspace query */
        int lwork_query = -1;
        int info = 0;
        void *work_query = std::malloc(ctx.typesize);
        auto *fn = reinterpret_cast<void (*)(
            const char *, const int *, void *, const int *,
            void *, void *, void *,
            void *, const int *, int *, std::size_t)>(load_sym(lib, sym));
        fn(&uplo, &n, A, &lda, d_arr, e_arr, tau,
           work_query, &lwork_query, &info, (std::size_t)1);
        int lwork = query_lwork_complex(work_query, ctx);
        std::free(work_query);

        void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);

        /* Call HETRD */
        info = 0;
        fn(&uplo, &n, A, &lda, d_arr, e_arr, tau,
           work, &lwork, &info, (std::size_t)1);

        /* Convert factored A (complex) to MPFR */
        MpfrComplexMatrix A_fact(n, n, prec);
        custom_to_mpfr_complex_mat(A_fact, A, lda, ctx);

        /* Convert d (real) to MPFR */
        MpfrMatrix d_mpfr(n, 1, prec);
        {
            const char *dp = static_cast<const char *>(d_arr);
            for (int i = 0; i < n; ++i)
                ctx.to_mpfr(d_mpfr.at(i, 0), dp + static_cast<std::size_t>(i) * real_typesize);
        }

        /* Convert e (real) and tau (complex) to MPFR */
        int ne = (n > 1) ? (n - 1) : 0;
        MpfrMatrix e_mpfr(std::max(ne, 1), 1, prec);
        MpfrComplexMatrix tau_mpfr(std::max(ne, 1), 1, prec);
        if (ne > 0) {
            const char *ep = static_cast<const char *>(e_arr);
            const char *tp = static_cast<const char *>(tau);
            for (int i = 0; i < ne; ++i) {
                ctx.to_mpfr(e_mpfr.at(i, 0), ep + static_cast<std::size_t>(i) * real_typesize);
                ctx.to_mpfr_complex(tau_mpfr.re(i, 0), tau_mpfr.im(i, 0),
                                    tp + static_cast<std::size_t>(i) * ctx.typesize);
            }
        }

        /* Build tridiagonal matrix T as a complex matrix with real d/e on diagonals
           and zero imaginary parts. T is n-by-n. */
        MpfrComplexMatrix T(n, n, prec);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                mpfr_set_d(T.re(i, j), 0.0, MPFR_RNDN);
                mpfr_set_d(T.im(i, j), 0.0, MPFR_RNDN);
            }
        for (int i = 0; i < n; ++i)
            mpfr_set(T.re(i, i), d_mpfr.at(i, 0), MPFR_RNDN);
        for (int i = 0; i < ne; ++i) {
            mpfr_set(T.re(i, i + 1), e_mpfr.at(i, 0), MPFR_RNDN);
            mpfr_set(T.re(i + 1, i), e_mpfr.at(i, 0), MPFR_RNDN);
        }

        /* Reconstruct Q (n-by-n) from complex Householder reflectors */
        MpfrComplexMatrix Q(n, n, prec);
        accumulate_Q_hetrd(Q, A_fact, tau_mpfr, n, uplo);

        /* Compute Q * T * Q^H */
        MpfrComplexMatrix QT(n, n, prec);
        mpfr_complex_mat_mul_simple(QT, Q, T);

        MpfrComplexMatrix Qh(n, n, prec);
        mpfr_complex_mat_adjoint(Qh, Q);

        MpfrComplexMatrix QTQh(n, n, prec);
        mpfr_complex_mat_mul_simple(QTQh, QT, Qh);

        /* Residual: ||A - Q*T*Q^H||_1 / (||A||_1 * n * eps) */
        MpfrComplexMatrix Resid(n, n, prec);
        mpfr_complex_mat_sub(Resid, A_orig, QTQh);
        double norm_resid = mpfr_complex_mat_norm1(Resid);
        double norm_A = mpfr_complex_mat_norm1(A_orig);

        double residual = (norm_A > 0.0) ? norm_resid / (norm_A * n * eps) : 0.0;

        /* Orthogonality: ||Q^H*Q - I||_1 / (n * eps) */
        double ortho = mpfr_complex_orthogonality(Q, eps);

        LapackResult lr;
        lr.residual = residual;
        lr.orthogonality = ortho;
        lr.info = info;

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str), "uplo=%c n=%d", uplo, n);
        report_lapack_result("HETRD", params_str, lr, format);

        std::free(A);
        std::free(A_her);
        std::free(d_arr);
        std::free(e_arr);
        std::free(tau);
        std::free(work);
    }
}
