/* gbsv.cpp -- LAPACK GBSV accuracy tester (banded linear solve) */

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

void test_gbsv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int n = params.n;
    int kl = params.kl;
    int ku = params.ku;
    int nrhs = std::min(params.k, 4);
    if (nrhs < 1) nrhs = 1;
    int ldb = n + params.ld_pad;

    /* GBSV requires ldab = 2*kl + ku + 1 (extra kl rows for fill-in) */
    int ldab = 2 * kl + ku + 1;

    unsigned seed = params.seed;

    /* Generate a full dense n-by-n random matrix, then extract band entries */
    void *A_full = gen_random_array(static_cast<std::size_t>(n) * n,
                                    ctx.typesize, ctx.from_mpfr, prec, &seed);

    /* Build full A in MPFR (only band entries are nonzero) for residual */
    MpfrMatrix A_orig(n, n, prec);
    {
        const char *afp = static_cast<const char *>(A_full);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                if (i >= std::max(0, j - ku) && i <= std::min(n - 1, j + kl))
                    ctx.to_mpfr(A_orig.at(i, j),
                                afp + IDX(i, j, n) * ctx.typesize);
                else
                    mpfr_set_d(A_orig.at(i, j), 0.0, MPFR_RNDN);
            }
        }
    }

    /* Add diagonal dominance to ensure nonsingularity */
    {
        MpfrScalar nval(prec);
        mpfr_set_d(nval.get(), static_cast<double>(n), MPFR_RNDN);
        for (int i = 0; i < n; ++i)
            mpfr_add(A_orig.at(i, i), A_orig.at(i, i), nval.get(), MPFR_RNDN);
    }

    /* Build GBSV band storage: AB is ldab-by-n
       Element A(i,j) is stored at AB[kl+ku+i-j, j] (0-based).
       The first kl rows are reserved for fill-in during pivoting. */
    void *AB = std::calloc(static_cast<std::size_t>(ldab) * n, ctx.typesize);
    {
        char *abp = static_cast<char *>(AB);
        for (int j = 0; j < n; ++j) {
            for (int i = std::max(0, j - ku); i <= std::min(n - 1, j + kl); ++i) {
                int row = kl + ku + i - j;
                /* Convert from MPFR to custom format */
                ctx.from_mpfr(abp + IDX(row, j, ldab) * ctx.typesize,
                              A_orig.at(i, j), MPFR_RNDN);
            }
        }
    }

    std::free(A_full);

    /* Generate random n-by-nrhs right-hand side B */
    void *B = gen_random_array(static_cast<std::size_t>(ldb) * nrhs,
                               ctx.typesize, ctx.from_mpfr, prec, &seed);

    /* Save B_orig in MPFR */
    MpfrMatrix B_orig(n, nrhs, prec);
    custom_to_mpfr_mat(B_orig, B, ldb, ctx);

    /* Allocate IPIV */
    int *ipiv = new int[n];

    /* Call GBSV: void GBSV(int *n, int *kl, int *ku, int *nrhs,
                             T *AB, int *ldab, int *ipiv,
                             T *B, int *ldb, int *info) */
    int info = 0;
    auto *fn = reinterpret_cast<void (*)(
        const int *, const int *, const int *, const int *,
        void *, const int *, int *,
        void *, const int *, int *)>(load_sym(lib, sym));
    fn(&n, &kl, &ku, &nrhs, AB, &ldab, ipiv, B, &ldb, &info);

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
                  "n=%d kl=%d ku=%d nrhs=%d", n, kl, ku, nrhs);
    report_lapack_result("GBSV", params_str, lr, format);

    std::free(AB);
    std::free(B);
    delete[] ipiv;
}
