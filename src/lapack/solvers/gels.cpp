/* gels.cpp -- LAPACK GELS accuracy tester (least squares solve) */

#include "../solvers.h"
#include "../lapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Test GELS for a specific configuration                               */
/* ------------------------------------------------------------------ */

static void test_gels_case(const TesterCtx &ctx, void *lib, const char *sym,
                           const TestParams &params, const std::string &format,
                           char trans, int m, int n, const char *case_label)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int nrhs = std::min(params.k, 4);
    if (nrhs < 1) nrhs = 1;
    int lda = m + params.ld_pad;
    /* B has max(m,n) rows to hold both input and solution */
    int ldb = std::max(m, n) + params.ld_pad;

    unsigned seed = params.seed;

    /* Generate random m-by-n matrix A */
    void *A = gen_random_array(static_cast<std::size_t>(lda) * n,
                               ctx.typesize, ctx.from_mpfr, prec, &seed);

    /* Save A_orig in MPFR */
    MpfrMatrix A_orig(m, n, prec);
    custom_to_mpfr_mat(A_orig, A, lda, ctx);

    /* Generate random B: max(m,n)-by-nrhs, but only first m rows matter for input */
    void *B = gen_random_array(static_cast<std::size_t>(ldb) * nrhs,
                               ctx.typesize, ctx.from_mpfr, prec, &seed);

    /* Save B_orig (m-by-nrhs) in MPFR */
    MpfrMatrix B_orig(m, nrhs, prec);
    custom_to_mpfr_mat(B_orig, B, ldb, ctx);

    /* Workspace query */
    int lwork = -1;
    int info = 0;
    size_t trans_len = 1;
    auto *fn = reinterpret_cast<void (*)(
        const char *, const int *, const int *, const int *,
        void *, const int *, void *, const int *,
        void *, const int *, int *, size_t)>(load_sym(lib, sym));

    void *work_query = std::malloc(ctx.typesize);
    fn(&trans, &m, &n, &nrhs, A, &lda, B, &ldb,
       work_query, &lwork, &info, trans_len);
    lwork = query_lwork(work_query, ctx);
    std::free(work_query);

    /* Allocate workspace and call GELS */
    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    info = 0;
    fn(&trans, &m, &n, &nrhs, A, &lda, B, &ldb,
       work, &lwork, &info, trans_len);

    /* Extract solution X from B.
       For overdetermined (m >= n, trans='N'): X is first n rows of B.
       For underdetermined (m < n, trans='N'): X is first n rows of B. */
    MpfrMatrix X(n, nrhs, prec);
    custom_to_mpfr_mat(X, B, ldb, ctx);

    /* Reload A_orig (since A was modified in-place by GELS) */
    /* A_orig is already saved above */

    /* For overdetermined (m >= n): least squares, check normal equations
       A^T * (A*X - B) should be zero: ||A^T*r||_1 / (||A||_1 * ||r||_1 * n * eps)
       For underdetermined (m < n): solve residual ||A*X - B|| / (||A|| * ||X|| * m * eps) */
    MpfrMatrix AX(m, nrhs, prec);
    mpfr_mat_mul_simple(AX, A_orig, X);

    MpfrMatrix R(m, nrhs, prec);
    mpfr_mat_sub(R, AX, B_orig);

    double norm_A = mpfr_mat_norm1(A_orig);
    double residual = 0.0;

    if (m >= n) {
        /* Overdetermined: check normal equations A^T * r = 0 */
        MpfrMatrix At(n, m, prec);
        mpfr_mat_transpose(At, A_orig);

        MpfrMatrix AtR(n, nrhs, prec);
        mpfr_mat_mul_simple(AtR, At, R);

        double norm_AtR = mpfr_mat_norm1(AtR);
        double norm_R = mpfr_mat_norm1(R);
        if (norm_A > 0.0 && norm_R > 0.0)
            residual = norm_AtR / (norm_A * norm_R * n * eps);
        else if (norm_R > 0.0)
            residual = norm_AtR / (norm_R * n * eps);
    } else {
        /* Underdetermined: check solve residual */
        double norm_R = mpfr_mat_norm1(R);
        double norm_X = mpfr_mat_norm1(X);
        if (norm_A > 0.0 && norm_X > 0.0)
            residual = norm_R / (norm_A * norm_X * m * eps);
    }

    LapackResult lr;
    lr.residual = residual;
    lr.orthogonality = -1.0;
    lr.info = info;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "trans=%c m=%d n=%d nrhs=%d (%s)",
                  trans, m, n, nrhs, case_label);
    report_lapack_result("GELS", params_str, lr, format);

    std::free(A);
    std::free(B);
    std::free(work);
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_gels(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    /* Overdetermined case: m > n, trans='N' */
    {
        int m = params.m;
        int n = std::min(params.n, params.m);
        /* Ensure m > n for overdetermined; if m == n, increase m */
        if (m <= n) m = n + std::max(n / 2, 1);
        test_gels_case(ctx, lib, sym, params, format, 'N', m, n, "overdetermined");
    }

    /* Underdetermined case: m < n, trans='N' */
    {
        int n = params.n;
        int m = std::min(params.m, params.n);
        /* Ensure m < n for underdetermined; if m == n, increase n */
        if (m >= n) n = m + std::max(m / 2, 1);
        test_gels_case(ctx, lib, sym, params, format, 'N', m, n, "underdetermined");
    }
}
