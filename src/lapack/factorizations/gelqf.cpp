/* gelqf.cpp -- LAPACK GELQF accuracy tester (LQ factorization) */

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
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_gelqf(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;
    int mn = std::min(m, n);

    unsigned seed_A = params.seed;

    /* Generate random m-by-n matrix A */
    void *A = gen_random_array(static_cast<std::size_t>(lda) * n,
                               ctx.typesize, ctx.from_mpfr, prec, &seed_A);

    /* Save A_orig in MPFR */
    MpfrMatrix A_orig(m, n, prec);
    custom_to_mpfr_mat(A_orig, A, lda, ctx);

    /* Allocate tau */
    void *tau = std::malloc(static_cast<std::size_t>(mn) * ctx.typesize);

    /* Workspace query */
    int lwork_query = -1;
    int info = 0;
    void *work_query = std::malloc(ctx.typesize);
    auto *fn = reinterpret_cast<void (*)(
        const int *, const int *, void *, const int *,
        void *, void *, const int *, int *)>(load_sym(lib, sym));
    fn(&m, &n, A, &lda, tau, work_query, &lwork_query, &info);
    int lwork = query_lwork(work_query, ctx);
    std::free(work_query);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);

    /* Call GELQF */
    info = 0;
    fn(&m, &n, A, &lda, tau, work, &lwork, &info);

    /* Convert factored A and tau to MPFR */
    MpfrMatrix A_fact(m, n, prec);
    custom_to_mpfr_mat(A_fact, A, lda, ctx);

    MpfrMatrix tau_mpfr(mn, 1, prec);
    {
        const char *tp = static_cast<const char *>(tau);
        for (int i = 0; i < mn; ++i)
            ctx.to_mpfr(tau_mpfr.at(i, 0), tp + static_cast<std::size_t>(i) * ctx.typesize);
    }

    /* Extract L (lower triangular part of A_fact, m-by-n with only m-by-mn relevant) */
    MpfrMatrix L(m, n, prec);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            if (i >= j)
                mpfr_set(L.at(i, j), A_fact.at(i, j), MPFR_RNDN);
            else
                mpfr_set_d(L.at(i, j), 0.0, MPFR_RNDN);
        }

    /* Reconstruct Q (n-by-n) from Householder reflectors stored in rows */
    MpfrMatrix Q(n, n, prec);
    mpfr_accumulate_Q_from_LQ(Q, A_fact, tau_mpfr, m, n);

    /* Compute L*Q */
    MpfrMatrix LQ(m, n, prec);
    mpfr_mat_mul_simple(LQ, L, Q);

    /* Residual: ||A - L*Q||_1 / (||A||_1 * max(m,n) * eps) */
    MpfrMatrix Resid(m, n, prec);
    mpfr_mat_sub(Resid, A_orig, LQ);
    double norm_resid = mpfr_mat_norm1(Resid);
    double norm_A = mpfr_mat_norm1(A_orig);

    int mn_max = std::max(m, n);
    double residual = (norm_A > 0.0) ? norm_resid / (norm_A * mn_max * eps) : 0.0;

    /* Orthogonality: ||Q^T*Q - I||_1 / (n * eps) */
    double ortho = mpfr_orthogonality(Q, eps);

    LapackResult lr;
    lr.residual = residual;
    lr.orthogonality = ortho;
    lr.info = info;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "m=%d n=%d", m, n);
    report_lapack_result("GELQF", params_str, lr, format);

    std::free(A);
    std::free(tau);
    std::free(work);
}
