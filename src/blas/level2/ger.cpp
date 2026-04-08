/* ger.cpp -- BLAS Level 2 GER accuracy tester */

#include "../level2.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer -- NO character arguments, NO hidden lengths */
extern "C" typedef void (*ger_fn_t)(
    const int  *m,      const int  *n,
    const void *alpha,
    const void *x,      const int  *incx,
    const void *y,      const int  *incy,
    void       *A,      const int  *lda
);

/* ------------------------------------------------------------------ */
/* MPFR reference: A_ref = alpha * x * y^T + A_in                      */
/* ------------------------------------------------------------------ */

static void mpfr_ger_ref(MpfrMatrix &A_ref,
                          int m, int n,
                          mpfr_t alpha,
                          const MpfrMatrix &x,
                          const MpfrMatrix &y,
                          const MpfrMatrix &A_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar tmp(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            /* A_ref[i,j] = alpha * x[i] * y[j] + A_in[i,j] */
            mpfr_mul(tmp.get(), x.at(i, 0), y.at(j, 0), MPFR_RNDN);
            mpfr_fma(A_ref.at(i, j), alpha, tmp.get(), A_in.at(i, j), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_ger(const TesterCtx &ctx, void *lib, const char *sym,
              const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<ger_fn_t>(load_sym(lib, sym));

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

    void *x     = gen_random_array(x_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
    void *y     = gen_random_array(y_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_y);
    void *A_in  = gen_random_array(lda * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
    void *alpha = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_al);

    unsigned sentinel_seed = 0xDEAD0001;
    void *A_out = alloc_with_sentinel(lda * n, ctx.typesize, sentinel_seed);
    copy_matrix_active(A_out, A_in, m, n, lda, ctx.typesize);

    fn(&m, &n, alpha, x, &incx, y, &incy, A_out, &lda);

    MpfrScalar mpfr_alpha(prec);
    ctx.to_mpfr(mpfr_alpha.get(), alpha);

    MpfrMatrix x_mpfr(m, 1, prec);
    MpfrMatrix y_mpfr(n, 1, prec);
    MpfrMatrix A_in_mpfr(m, n, prec);
    MpfrMatrix A_ref(m, n, prec);

    custom_to_mpfr_vec(x_mpfr, x, incx, ctx);
    custom_to_mpfr_vec(y_mpfr, y, incy, ctx);
    custom_to_mpfr_mat(A_in_mpfr, A_in, lda, ctx);

    mpfr_ger_ref(A_ref, m, n, mpfr_alpha.get(), x_mpfr, y_mpfr, A_in_mpfr);

    ErrorResult err = compute_error_matrix(A_ref, A_out, lda, ctx);
    SentinelResult sr = check_matrix_sentinels(A_out, m, n, lda, ctx.typesize, sentinel_seed);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d incx=%d incy=%d",
                  m, n, incx, incy);
    report_result("GER", params_str, err, &sr, format);

    std::free(x); std::free(y); std::free(A_in);
    std::free(A_out); std::free(alpha);
}
