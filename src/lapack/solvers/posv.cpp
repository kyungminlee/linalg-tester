/* posv.cpp -- LAPACK POSV accuracy tester (symmetric positive definite solve) */

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
/* Test POSV for a single uplo value                                    */
/* ------------------------------------------------------------------ */

static void test_posv_uplo(const TesterCtx &ctx, void *lib, const char *sym,
                           const TestParams &params, const std::string &format,
                           char uplo)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int n = params.n;
    int nrhs = std::min(params.k, 4);
    if (nrhs < 1) nrhs = 1;
    int lda = n + params.ld_pad;
    int ldb = n + params.ld_pad;

    unsigned seed = params.seed;

    /* Generate SPD matrix A (returned with ld = n) */
    void *A_spd = gen_positive_definite_array(n, ctx.typesize, ctx.from_mpfr,
                                               ctx.to_mpfr, prec, &seed);

    /* Copy into padded layout if needed */
    void *A = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
    {
        const char *src = static_cast<const char *>(A_spd);
        char *dst = static_cast<char *>(A);
        for (int j = 0; j < n; ++j)
            std::memcpy(dst + static_cast<std::size_t>(j) * lda * ctx.typesize,
                        src + static_cast<std::size_t>(j) * n * ctx.typesize,
                        static_cast<std::size_t>(n) * ctx.typesize);
    }
    std::free(A_spd);

    /* Save A_orig in MPFR (full symmetric matrix) */
    MpfrMatrix A_orig(n, n, prec);
    custom_to_mpfr_mat(A_orig, A, lda, ctx);

    /* Generate random n-by-nrhs right-hand side B */
    void *B = gen_random_array(static_cast<std::size_t>(ldb) * nrhs,
                               ctx.typesize, ctx.from_mpfr, prec, &seed);

    /* Save B_orig in MPFR */
    MpfrMatrix B_orig(n, nrhs, prec);
    custom_to_mpfr_mat(B_orig, B, ldb, ctx);

    /* Call POSV: void POSV(char *uplo, int *n, int *nrhs, T *A, int *lda,
                            T *B, int *ldb, int *info, size_t uplo_len) */
    int info = 0;
    size_t uplo_len = 1;
    auto *fn = reinterpret_cast<void (*)(
        const char *, const int *, const int *, void *, const int *,
        void *, const int *, int *, size_t)>(load_sym(lib, sym));
    fn(&uplo, &n, &nrhs, A, &lda, B, &ldb, &info, uplo_len);

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
    std::snprintf(params_str, sizeof(params_str),
                  "uplo=%c n=%d nrhs=%d", uplo, n, nrhs);
    report_lapack_result("POSV", params_str, lr, format);

    std::free(A);
    std::free(B);
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_posv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    test_posv_uplo(ctx, lib, sym, params, format, 'U');
    test_posv_uplo(ctx, lib, sym, params, format, 'L');
}
