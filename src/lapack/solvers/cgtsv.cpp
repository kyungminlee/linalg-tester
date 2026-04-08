/* cgtsv.cpp -- LAPACK CGTSV/ZGTSV accuracy tester (complex tridiagonal solve) */

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
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_cgtsv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int n = params.n;
    int nrhs = std::min(params.k, 4);
    if (nrhs < 1) nrhs = 1;
    int ldb = n + params.ld_pad;

    unsigned seed = params.seed;

    /* Generate random complex sub-diagonal dl (n-1), diagonal d (n), super-diagonal du (n-1) */
    void *dl = gen_random_complex_array(n - 1, ctx.typesize,
                                         ctx.from_mpfr_complex, prec, &seed);
    void *d  = gen_random_complex_array(n,     ctx.typesize,
                                         ctx.from_mpfr_complex, prec, &seed);
    void *du = gen_random_complex_array(n - 1, ctx.typesize,
                                         ctx.from_mpfr_complex, prec, &seed);

    /* Make diagonal dominant: |d[i]| > |dl[i]| + |du[i]| + n
       Read dl, du into MPFR, compute sum of abs neighbors, set d[i] = (sum + n, 0) */
    {
        MpfrComplexScalar dl_val(prec), du_val(prec);
        MpfrScalar abs_dl(prec), abs_du(prec), sum(prec), nval(prec), d_re(prec);
        mpfr_set_d(nval.get(), static_cast<double>(n), MPFR_RNDN);

        const char *dlp = static_cast<const char *>(dl);
        const char *dup = static_cast<const char *>(du);
        char *dp = static_cast<char *>(d);

        for (int i = 0; i < n; ++i) {
            mpfr_set_d(sum.get(), 0.0, MPFR_RNDN);

            if (i > 0) {
                ctx.to_mpfr_complex(dl_val.re(), dl_val.im(),
                                    dlp + static_cast<std::size_t>(i - 1) * ctx.typesize);
                mpfr_complex_abs(abs_dl.get(), dl_val.re(), dl_val.im(), MPFR_RNDN);
                mpfr_add(sum.get(), sum.get(), abs_dl.get(), MPFR_RNDN);
            }
            if (i < n - 1) {
                ctx.to_mpfr_complex(du_val.re(), du_val.im(),
                                    dup + static_cast<std::size_t>(i) * ctx.typesize);
                mpfr_complex_abs(abs_du.get(), du_val.re(), du_val.im(), MPFR_RNDN);
                mpfr_add(sum.get(), sum.get(), abs_du.get(), MPFR_RNDN);
            }

            /* d[i] = (sum + n, 0) -- real-valued diagonal dominant term */
            mpfr_add(d_re.get(), sum.get(), nval.get(), MPFR_RNDN);
            MpfrScalar d_im(prec);
            mpfr_set_d(d_im.get(), 0.0, MPFR_RNDN);
            ctx.from_mpfr_complex(dp + static_cast<std::size_t>(i) * ctx.typesize,
                                  d_re.get(), d_im.get(), MPFR_RNDN);
        }
    }

    /* Save originals in MPFR complex vectors */
    MpfrComplexMatrix dl_orig(n - 1, 1, prec);
    MpfrComplexMatrix d_orig(n, 1, prec);
    MpfrComplexMatrix du_orig(n - 1, 1, prec);
    {
        const char *dlp = static_cast<const char *>(dl);
        const char *dp  = static_cast<const char *>(d);
        const char *dup = static_cast<const char *>(du);
        for (int i = 0; i < n - 1; ++i)
            ctx.to_mpfr_complex(dl_orig.re(i, 0), dl_orig.im(i, 0),
                                dlp + static_cast<std::size_t>(i) * ctx.typesize);
        for (int i = 0; i < n; ++i)
            ctx.to_mpfr_complex(d_orig.re(i, 0), d_orig.im(i, 0),
                                dp + static_cast<std::size_t>(i) * ctx.typesize);
        for (int i = 0; i < n - 1; ++i)
            ctx.to_mpfr_complex(du_orig.re(i, 0), du_orig.im(i, 0),
                                dup + static_cast<std::size_t>(i) * ctx.typesize);
    }

    /* Build full tridiagonal A in MPFR for residual check */
    MpfrComplexMatrix A_orig(n, n, prec);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            mpfr_set_d(A_orig.re(i, j), 0.0, MPFR_RNDN);
            mpfr_set_d(A_orig.im(i, j), 0.0, MPFR_RNDN);
        }
    for (int i = 0; i < n; ++i) {
        mpfr_set(A_orig.re(i, i), d_orig.re(i, 0), MPFR_RNDN);
        mpfr_set(A_orig.im(i, i), d_orig.im(i, 0), MPFR_RNDN);
    }
    for (int i = 0; i < n - 1; ++i) {
        mpfr_set(A_orig.re(i + 1, i), dl_orig.re(i, 0), MPFR_RNDN);
        mpfr_set(A_orig.im(i + 1, i), dl_orig.im(i, 0), MPFR_RNDN);
        mpfr_set(A_orig.re(i, i + 1), du_orig.re(i, 0), MPFR_RNDN);
        mpfr_set(A_orig.im(i, i + 1), du_orig.im(i, 0), MPFR_RNDN);
    }

    /* Generate random complex n-by-nrhs right-hand side B */
    void *B = gen_random_complex_array(static_cast<std::size_t>(ldb) * nrhs,
                                        ctx.typesize, ctx.from_mpfr_complex,
                                        prec, &seed);

    /* Save B_orig in MPFR */
    MpfrComplexMatrix B_orig(n, nrhs, prec);
    custom_to_mpfr_complex_mat(B_orig, B, ldb, ctx);

    /* Call CGTSV/ZGTSV: void xGTSV(int *n, int *nrhs, T *dl, T *d, T *du,
                                      T *B, int *ldb, int *info) */
    int info = 0;
    auto *fn = reinterpret_cast<void (*)(
        const int *, const int *, void *, void *, void *,
        void *, const int *, int *)>(load_sym(lib, sym));
    fn(&n, &nrhs, dl, d, du, B, &ldb, &info);

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
    std::snprintf(params_str, sizeof(params_str), "n=%d nrhs=%d", n, nrhs);
    report_lapack_result("CGTSV", params_str, lr, format);

    std::free(dl);
    std::free(d);
    std::free(du);
    std::free(B);
}
