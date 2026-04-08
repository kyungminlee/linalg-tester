/* gesv.cpp -- LAPACK GESV accuracy tester (general linear solve AX = B) */

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
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_gesv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int n = params.n;
    int nrhs = std::min(params.k, 4);
    if (nrhs < 1) nrhs = 1;
    int lda = n + params.ld_pad;
    int ldb = n + params.ld_pad;

    unsigned seed = params.seed;

    /* Generate random n-by-n matrix A */
    void *A = gen_random_array(static_cast<std::size_t>(lda) * n,
                               ctx.typesize, ctx.from_mpfr, prec, &seed);

    /* Save A_orig in MPFR */
    MpfrMatrix A_orig(n, n, prec);
    custom_to_mpfr_mat(A_orig, A, lda, ctx);

    /* Generate random n-by-nrhs right-hand side B */
    void *B = gen_random_array(static_cast<std::size_t>(ldb) * nrhs,
                               ctx.typesize, ctx.from_mpfr, prec, &seed);

    /* Save B_orig in MPFR */
    MpfrMatrix B_orig(n, nrhs, prec);
    custom_to_mpfr_mat(B_orig, B, ldb, ctx);

    /* Allocate IPIV */
    int *ipiv = new int[n];

    /* Call GESV: void GESV(int *n, int *nrhs, T *A, int *lda, int *ipiv,
                            T *B, int *ldb, int *info) */
    int info = 0;
    auto *fn = reinterpret_cast<void (*)(
        const int *, const int *, void *, const int *,
        int *, void *, const int *, int *)>(load_sym(lib, sym));
    fn(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);

    /* Convert solution X (stored in B) to MPFR */
    MpfrMatrix X(n, nrhs, prec);
    custom_to_mpfr_mat(X, B, ldb, ctx);

    /* Compute solve residual: ||AX - B||_1 / (||A||_1 * ||X||_1 * n * eps) */
    double residual = mpfr_solve_residual(A_orig, X, B_orig, eps);

    LapackResult lr;
    lr.residual = residual;
    lr.orthogonality = -1.0;
    lr.info = info;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "n=%d nrhs=%d", n, nrhs);
    report_lapack_result("GESV", params_str, lr, format);

    std::free(A);
    std::free(B);
    delete[] ipiv;
}
