/* csysv.cpp -- LAPACK CSYSV/ZSYSV accuracy tester (complex symmetric indefinite solve) */

#include "../solvers.h"
#include "../lapack_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/mpfr_lapack_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Test CSYSV for a single uplo value                                   */
/* ------------------------------------------------------------------ */

static void test_csysv_uplo(const TesterCtx &ctx, void *lib, const char *sym,
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

    /* Generate complex symmetric matrix A (A = A^T, no conjugation) */
    void *A_sym = gen_complex_symmetric_array(n, uplo, ctx.typesize,
                                               ctx.from_mpfr_complex, prec, &seed);

    /* Copy into padded layout */
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

    /* Add diagonal dominance: A(i,i) += n (complex: add to real part) */
    {
        MpfrComplexScalar diag_val(prec);
        MpfrScalar nval(prec);
        mpfr_set_d(nval.get(), static_cast<double>(n), MPFR_RNDN);
        char *ap = static_cast<char *>(A);
        for (int i = 0; i < n; ++i) {
            ctx.to_mpfr_complex(diag_val.re(), diag_val.im(),
                                ap + IDX(i, i, lda) * ctx.typesize);
            mpfr_add(diag_val.re(), diag_val.re(), nval.get(), MPFR_RNDN);
            /* For complex symmetric, diagonal can have imaginary part -- keep it */
            ctx.from_mpfr_complex(ap + IDX(i, i, lda) * ctx.typesize,
                                  diag_val.re(), diag_val.im(), MPFR_RNDN);
        }
    }

    /* Save A_orig in MPFR -- build full symmetric matrix for residual */
    MpfrComplexMatrix A_orig(n, n, prec);
    custom_to_mpfr_complex_mat(A_orig, A, lda, ctx);
    /* Fill in the other triangle: A(i,j) = A(j,i) (no conjugation for symmetric) */
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (uplo == 'U' && i > j) {
                mpfr_set(A_orig.re(i, j), A_orig.re(j, i), MPFR_RNDN);
                mpfr_set(A_orig.im(i, j), A_orig.im(j, i), MPFR_RNDN);
            } else if (uplo == 'L' && i < j) {
                mpfr_set(A_orig.re(i, j), A_orig.re(j, i), MPFR_RNDN);
                mpfr_set(A_orig.im(i, j), A_orig.im(j, i), MPFR_RNDN);
            }
        }
    }

    /* Generate random complex n-by-nrhs right-hand side B */
    void *B = gen_random_complex_array(static_cast<std::size_t>(ldb) * nrhs,
                                        ctx.typesize, ctx.from_mpfr_complex,
                                        prec, &seed);

    /* Save B_orig in MPFR */
    MpfrComplexMatrix B_orig(n, nrhs, prec);
    custom_to_mpfr_complex_mat(B_orig, B, ldb, ctx);

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
    lwork = query_lwork_complex(work_query, ctx);
    std::free(work_query);

    /* Allocate workspace and call CSYSV/ZSYSV */
    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    info = 0;
    fn(&uplo, &n, &nrhs, A, &lda, ipiv, B, &ldb,
       work, &lwork, &info, uplo_len);

    /* Convert solution X (stored in B) to MPFR */
    MpfrComplexMatrix X(n, nrhs, prec);
    custom_to_mpfr_complex_mat(X, B, ldb, ctx);

    /* Compute solve residual: ||AX - B||_1 / (||A||_1 * ||X||_1 * n * eps) */
    double residual = mpfr_complex_solve_residual(A_orig, X, B_orig, eps);

    LapackResult lr;
    lr.residual = residual;
    lr.orthogonality = -1.0;
    lr.info = info;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "uplo=%c n=%d nrhs=%d", uplo, n, nrhs);
    report_lapack_result("CSYSV", params_str, lr, format);

    std::free(A);
    std::free(B);
    std::free(work);
    delete[] ipiv;
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_csysv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    test_csysv_uplo(ctx, lib, sym, params, format, 'U');
    test_csysv_uplo(ctx, lib, sym, params, format, 'L');
}
