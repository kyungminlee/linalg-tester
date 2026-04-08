/* gerc.cpp -- BLAS Level 2 GERC accuracy tester (complex-only) */

#include "../level2.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer -- NO character arguments, NO hidden lengths */
extern "C" typedef void (*gerc_fn_t)(
    const int  *m,      const int  *n,
    const void *alpha,
    const void *x,      const int  *incx,
    const void *y,      const int  *incy,
    void       *A,      const int  *lda
);

/* ------------------------------------------------------------------ */
/* MPFR reference: A_ref = alpha * x * conj(y)^T + A_in               */
/* ------------------------------------------------------------------ */

static void mpfr_gerc_ref(MpfrComplexMatrix &A_ref,
                            int m, int n,
                            const MpfrComplexScalar &alpha,
                            const MpfrComplexMatrix &x,
                            const MpfrComplexMatrix &y,
                            const MpfrComplexMatrix &A_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha.re());
    MpfrComplexScalar tmp(prec), yc(prec);

    for (int j = 0; j < n; ++j) {
        /* Conjugate y[j] once per column */
        mpfr_complex_conj(yc.re(), yc.im(),
                          y.re(j, 0), y.im(j, 0), MPFR_RNDN);
        for (int i = 0; i < m; ++i) {
            /* tmp = x[i] * conj(y[j]) */
            mpfr_complex_mul(tmp.re(), tmp.im(),
                             x.re(i, 0), x.im(i, 0),
                             yc.re(), yc.im(), MPFR_RNDN);
            /* A_ref[i,j] = alpha * tmp + A_in[i,j] */
            mpfr_complex_mul(tmp.re(), tmp.im(),
                             alpha.re(), alpha.im(),
                             tmp.re(), tmp.im(), MPFR_RNDN);
            mpfr_complex_add(A_ref.re(i, j), A_ref.im(i, j),
                             tmp.re(), tmp.im(),
                             A_in.re(i, j), A_in.im(i, j), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_gerc(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "GERC requires --complex\n");
        return;
    }

    auto *fn = reinterpret_cast<gerc_fn_t>(load_sym(lib, sym));

    int m = params.m, n = params.n;
    int incx = params.incx, incy = params.incy;
    mpfr_prec_t prec = ctx.prec;

    int lda = m + params.ld_pad;
    int abs_incx = (incx < 0) ? -incx : incx;
    int abs_incy = (incy < 0) ? -incy : incy;
    int x_alloc = 1 + (m - 1) * abs_incx;
    int y_alloc = 1 + (n - 1) * abs_incy;

    unsigned seed_x  = params.seed;
    unsigned seed_y  = params.seed + 1;
    unsigned seed_A  = params.seed + 2;
    unsigned seed_al = params.seed + 3;

    void *x     = gen_random_complex_array(x_alloc, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
    void *y     = gen_random_complex_array(y_alloc, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_y);
    void *A_in  = gen_random_complex_array(lda * n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
    void *alpha = gen_random_complex_array(1,       ctx.typesize, ctx.from_mpfr_complex, prec, &seed_al);

    unsigned sentinel_seed = 0xDEAD0001;
    void *A_out = alloc_with_sentinel(lda * n, ctx.typesize, sentinel_seed);
    copy_matrix_active(A_out, A_in, m, n, lda, ctx.typesize);

    fn(&m, &n, alpha, x, &incx, y, &incy, A_out, &lda);

    MpfrComplexScalar mpfr_alpha(prec);
    ctx.to_mpfr_complex(mpfr_alpha.re(), mpfr_alpha.im(), alpha);

    MpfrComplexMatrix x_mpfr(m, 1, prec);
    MpfrComplexMatrix y_mpfr(n, 1, prec);
    MpfrComplexMatrix A_in_mpfr(m, n, prec);
    MpfrComplexMatrix A_ref(m, n, prec);

    custom_to_mpfr_complex_vec(x_mpfr, x, incx, ctx);
    custom_to_mpfr_complex_vec(y_mpfr, y, incy, ctx);
    custom_to_mpfr_complex_mat(A_in_mpfr, A_in, lda, ctx);

    mpfr_gerc_ref(A_ref, m, n, mpfr_alpha, x_mpfr, y_mpfr, A_in_mpfr);

    ErrorResult err = compute_error_complex_matrix(A_ref, A_out, lda, ctx);
    SentinelResult sr = check_matrix_sentinels(A_out, m, n, lda, ctx.typesize, sentinel_seed);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d incx=%d incy=%d",
                  m, n, incx, incy);
    report_result("GERC", params_str, err, &sr, format);

    std::free(x); std::free(y); std::free(A_in);
    std::free(A_out); std::free(alpha);
}
