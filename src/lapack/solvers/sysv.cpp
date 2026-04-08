/* sysv.cpp -- LAPACK SYSV accuracy tester (symmetric indefinite solve) */

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
/* Test SYSV for a single uplo value                                    */
/* ------------------------------------------------------------------ */

static void test_sysv_uplo(const TesterCtx &ctx, void *lib, const char *sym,
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

    /* Generate symmetric matrix A (returned with ld = n) */
    void *A_sym = gen_symmetric_array(n, uplo, ctx.typesize, ctx.from_mpfr,
                                       prec, &seed);

    /* Copy into padded layout and add diagonal dominance */
    void *A = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
    {
        const char *src = static_cast<const char *>(A_sym);
        char *dst = static_cast<char *>(A);
        for (int j = 0; j < n; ++j)
            std::memcpy(dst + static_cast<std::size_t>(j) * lda * ctx.typesize,
                        src + static_cast<std::size_t>(j) * n * ctx.typesize,
                        static_cast<std::size_t>(n) * ctx.typesize);
    }
    std::free(A_sym);

    /* Add diagonal dominance: A(i,i) += n to ensure nonsingularity */
    {
        MpfrScalar diag_val(prec), nval(prec);
        mpfr_set_d(nval.get(), static_cast<double>(n), MPFR_RNDN);
        char *ap = static_cast<char *>(A);
        for (int i = 0; i < n; ++i) {
            ctx.to_mpfr(diag_val.get(),
                        ap + IDX(i, i, lda) * ctx.typesize);
            mpfr_add(diag_val.get(), diag_val.get(), nval.get(), MPFR_RNDN);
            ctx.from_mpfr(ap + IDX(i, i, lda) * ctx.typesize,
                          diag_val.get(), MPFR_RNDN);
        }
    }

    /* Save A_orig in MPFR -- build full symmetric matrix for residual */
    MpfrMatrix A_orig(n, n, prec);
    custom_to_mpfr_mat(A_orig, A, lda, ctx);
    /* Fill in the other triangle from the stored triangle */
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (uplo == 'U' && i > j)
                mpfr_set(A_orig.at(i, j), A_orig.at(j, i), MPFR_RNDN);
            else if (uplo == 'L' && i < j)
                mpfr_set(A_orig.at(i, j), A_orig.at(j, i), MPFR_RNDN);
        }
    }

    /* Generate random n-by-nrhs right-hand side B */
    void *B = gen_random_array(static_cast<std::size_t>(ldb) * nrhs,
                               ctx.typesize, ctx.from_mpfr, prec, &seed);

    /* Save B_orig in MPFR */
    MpfrMatrix B_orig(n, nrhs, prec);
    custom_to_mpfr_mat(B_orig, B, ldb, ctx);

    /* Allocate IPIV */
    int *ipiv = new int[n];

    /* Workspace query */
    int lwork = -1;
    int info = 0;
    size_t uplo_len = 1;
    auto *fn = reinterpret_cast<void (*)(
        const char *, const int *, const int *, void *, const int *,
        int *, void *, const int *, void *, const int *, int *,
        size_t)>(load_sym(lib, sym));

    void *work_query = std::malloc(ctx.typesize);
    fn(&uplo, &n, &nrhs, A, &lda, ipiv, B, &ldb,
       work_query, &lwork, &info, uplo_len);
    lwork = query_lwork(work_query, ctx);
    std::free(work_query);

    /* Allocate workspace and call SYSV */
    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    info = 0;
    fn(&uplo, &n, &nrhs, A, &lda, ipiv, B, &ldb,
       work, &lwork, &info, uplo_len);

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
    report_lapack_result("SYSV", params_str, lr, format);

    std::free(A);
    std::free(B);
    std::free(work);
    delete[] ipiv;
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_sysv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    test_sysv_uplo(ctx, lib, sym, params, format, 'U');
    test_sysv_uplo(ctx, lib, sym, params, format, 'L');
}
