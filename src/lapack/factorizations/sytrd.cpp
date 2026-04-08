/* sytrd.cpp -- LAPACK SYTRD accuracy tester (symmetric tridiagonal reduction) */
/* A = Q * T * Q^T where T is tridiagonal */

#include "../factorizations.h"
#include "../lapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Reconstruct Q from Householder reflectors produced by SYTRD.       */
/*                                                                     */
/* uplo='L': reflectors stored below first subdiagonal of A columns.  */
/*   For i=0..n-2: v(j)=0 for j<=i, v(i+1)=1, v(j)=A(j,i) for j>i+1 */
/*   Q = H_0 H_1 ... H_{n-2}                                         */
/*                                                                     */
/* uplo='U': reflectors stored above first superdiagonal of A columns.*/
/*   For i=n-1..1: v(j)=0 for j>=i, v(i-1)=1, v(j)=A(j,i) for j<i-1 */
/*   Q = H_{n-1} H_{n-2} ... H_1                                     */
/* ------------------------------------------------------------------ */

static void accumulate_Q_sytrd(MpfrMatrix &Q, const MpfrMatrix &A,
                                const MpfrMatrix &tau_vec,
                                int n, char uplo)
{
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    int k = n - 1; /* number of reflectors */
    if (k <= 0) {
        mpfr_mat_set_identity(Q);
        return;
    }

    mpfr_mat_set_identity(Q);
    MpfrScalar dot(prec), scale(prec);

    if (uplo == 'L') {
        /* Q = H_0 H_1 ... H_{n-2}, build from right:
           start with I, apply H_{n-2}, ..., H_0 from the left */
        for (int i = k - 1; i >= 0; --i) {
            /* v starts at row i+1: v(i+1)=1, v(j)=A(j,i) for j>i+1 */
            int v_start = i + 1;

            for (int j = 0; j < n; ++j) {
                mpfr_set(dot.get(), Q.at(v_start, j), MPFR_RNDN);
                for (int r = v_start + 1; r < n; ++r)
                    mpfr_fma(dot.get(), A.at(r, i), Q.at(r, j), dot.get(), MPFR_RNDN);

                mpfr_mul(scale.get(), tau_vec.at(i, 0), dot.get(), MPFR_RNDN);

                mpfr_sub(Q.at(v_start, j), Q.at(v_start, j), scale.get(), MPFR_RNDN);
                for (int r = v_start + 1; r < n; ++r) {
                    MpfrScalar tmp(prec);
                    mpfr_mul(tmp.get(), scale.get(), A.at(r, i), MPFR_RNDN);
                    mpfr_sub(Q.at(r, j), Q.at(r, j), tmp.get(), MPFR_RNDN);
                }
            }
        }
    } else {
        /* uplo='U': Q = H_{n-1} H_{n-2} ... H_1
           Reflector i (for column i, i=1..n-1) has tau index i-1.
           v ends at row i-1: v(i-1)=1, v(j)=A(j,i) for j<i-1, v(j)=0 for j>=i */
        for (int i = 1; i < n; ++i) {
            /* tau index: i-1 */
            int ti = i - 1;
            int v_end = i - 1; /* v(v_end) = 1 */

            for (int j = 0; j < n; ++j) {
                /* dot = v^T * Q(:,j) */
                mpfr_set(dot.get(), Q.at(v_end, j), MPFR_RNDN);
                for (int r = 0; r < v_end; ++r)
                    mpfr_fma(dot.get(), A.at(r, i), Q.at(r, j), dot.get(), MPFR_RNDN);

                mpfr_mul(scale.get(), tau_vec.at(ti, 0), dot.get(), MPFR_RNDN);

                mpfr_sub(Q.at(v_end, j), Q.at(v_end, j), scale.get(), MPFR_RNDN);
                for (int r = 0; r < v_end; ++r) {
                    MpfrScalar tmp(prec);
                    mpfr_mul(tmp.get(), scale.get(), A.at(r, i), MPFR_RNDN);
                    mpfr_sub(Q.at(r, j), Q.at(r, j), tmp.get(), MPFR_RNDN);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_sytrd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int n = params.n;
    int lda = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        /* Generate symmetric matrix (n-by-n, ld=n) */
        void *A_sym = gen_symmetric_array(n, uplo, ctx.typesize,
                                          ctx.from_mpfr, prec, &seed_A);

        /* Copy into lda-padded layout */
        void *A = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
        {
            char *dst = static_cast<char *>(A);
            const char *src = static_cast<const char *>(A_sym);
            for (int j = 0; j < n; ++j)
                std::memcpy(dst + static_cast<std::size_t>(j) * lda * ctx.typesize,
                            src + static_cast<std::size_t>(j) * n * ctx.typesize,
                            static_cast<std::size_t>(n) * ctx.typesize);
        }

        /* Save A_orig as full symmetric MPFR matrix */
        MpfrMatrix A_orig(n, n, prec);
        {
            const char *p = static_cast<const char *>(A);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    ctx.to_mpfr(A_orig.at(i, j),
                                p + IDX(i, j, lda) * ctx.typesize);
            /* Fill the other triangle by symmetry */
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < j; ++i) {
                    if (uplo == 'U')
                        mpfr_set(A_orig.at(j, i), A_orig.at(i, j), MPFR_RNDN);
                    else
                        mpfr_set(A_orig.at(i, j), A_orig.at(j, i), MPFR_RNDN);
                }
        }

        /* Allocate d, e, tau */
        void *d_arr = std::malloc(static_cast<std::size_t>(n) * ctx.typesize);
        void *e_arr = std::malloc(static_cast<std::size_t>(std::max(n - 1, 1)) * ctx.typesize);
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
        int lwork = query_lwork(work_query, ctx);
        std::free(work_query);

        void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);

        /* Call SYTRD */
        info = 0;
        fn(&uplo, &n, A, &lda, d_arr, e_arr, tau,
           work, &lwork, &info, (std::size_t)1);

        /* Convert factored A, d, e, tau to MPFR */
        MpfrMatrix A_fact(n, n, prec);
        custom_to_mpfr_mat(A_fact, A, lda, ctx);

        MpfrMatrix d_mpfr(n, 1, prec);
        {
            const char *dp = static_cast<const char *>(d_arr);
            for (int i = 0; i < n; ++i)
                ctx.to_mpfr(d_mpfr.at(i, 0), dp + static_cast<std::size_t>(i) * ctx.typesize);
        }

        int ne = (n > 1) ? (n - 1) : 0;
        MpfrMatrix e_mpfr(std::max(ne, 1), 1, prec);
        MpfrMatrix tau_mpfr(std::max(ne, 1), 1, prec);
        if (ne > 0) {
            const char *ep = static_cast<const char *>(e_arr);
            const char *tp = static_cast<const char *>(tau);
            for (int i = 0; i < ne; ++i) {
                ctx.to_mpfr(e_mpfr.at(i, 0), ep + static_cast<std::size_t>(i) * ctx.typesize);
                ctx.to_mpfr(tau_mpfr.at(i, 0), tp + static_cast<std::size_t>(i) * ctx.typesize);
            }
        }

        /* Build tridiagonal matrix T (n-by-n) */
        MpfrMatrix T(n, n, prec);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                mpfr_set_d(T.at(i, j), 0.0, MPFR_RNDN);
        for (int i = 0; i < n; ++i)
            mpfr_set(T.at(i, i), d_mpfr.at(i, 0), MPFR_RNDN);
        for (int i = 0; i < ne; ++i) {
            mpfr_set(T.at(i, i + 1), e_mpfr.at(i, 0), MPFR_RNDN);
            mpfr_set(T.at(i + 1, i), e_mpfr.at(i, 0), MPFR_RNDN);
        }

        /* Reconstruct Q (n-by-n) */
        MpfrMatrix Q(n, n, prec);
        accumulate_Q_sytrd(Q, A_fact, tau_mpfr, n, uplo);

        /* Compute Q * T * Q^T */
        MpfrMatrix QT(n, n, prec);
        mpfr_mat_mul_simple(QT, Q, T);

        MpfrMatrix Qt(n, n, prec);
        mpfr_mat_transpose(Qt, Q);

        MpfrMatrix QTQt(n, n, prec);
        mpfr_mat_mul_simple(QTQt, QT, Qt);

        /* Residual: ||A - Q*T*Q^T||_1 / (||A||_1 * n * eps) */
        MpfrMatrix Resid(n, n, prec);
        mpfr_mat_sub(Resid, A_orig, QTQt);
        double norm_resid = mpfr_mat_norm1(Resid);
        double norm_A = mpfr_mat_norm1(A_orig);

        double residual = (norm_A > 0.0) ? norm_resid / (norm_A * n * eps) : 0.0;

        /* Orthogonality: ||Q^T*Q - I||_1 / (n * eps) */
        double ortho = mpfr_orthogonality(Q, eps);

        LapackResult lr;
        lr.residual = residual;
        lr.orthogonality = ortho;
        lr.info = info;

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str), "uplo=%c n=%d", uplo, n);
        report_lapack_result("SYTRD", params_str, lr, format);

        std::free(A);
        std::free(A_sym);
        std::free(d_arr);
        std::free(e_arr);
        std::free(tau);
        std::free(work);
    }
}
