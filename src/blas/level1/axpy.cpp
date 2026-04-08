/* axpy.cpp -- BLAS Level 1 AXPY accuracy tester */

#include "../level1.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran ABI: void axpy(n, alpha, x, incx, y, incy) -- no hidden char lengths */
extern "C" typedef void (*axpy_fn_t)(
    const int *n, const void *alpha,
    const void *x, const int *incx,
    void *y, const int *incy
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_axpy(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<axpy_fn_t>(load_sym(lib, sym));
    mpfr_prec_t prec = ctx.prec;

    int n    = params.n;
    int incx = params.incx;
    int incy = params.incy;
    int abs_incx = (incx < 0) ? -incx : incx;
    int abs_incy = (incy < 0) ? -incy : incy;
    int alloc_x = 1 + (n - 1) * abs_incx;
    int alloc_y = 1 + (n - 1) * abs_incy;

    unsigned seed_x = params.seed;
    unsigned seed_y = params.seed + 1;
    unsigned seed_a = params.seed + 2;

    void *x_in  = gen_random_array(alloc_x, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
    void *y_in  = gen_random_array(alloc_y, ctx.typesize, ctx.from_mpfr, prec, &seed_y);
    void *alpha = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_a);

    /* Convert inputs to MPFR */
    MpfrMatrix x_mpfr(n, 1, prec), y_mpfr(n, 1, prec);
    custom_to_mpfr_vec(x_mpfr, x_in, incx, ctx);
    custom_to_mpfr_vec(y_mpfr, y_in, incy, ctx);

    MpfrScalar alpha_mpfr(prec);
    ctx.to_mpfr(alpha_mpfr.get(), alpha);

    /* Copy y for the call */
    unsigned sentinel_seed = 0xDEAD0001;
    void *y_out = alloc_with_sentinel(alloc_y, ctx.typesize, sentinel_seed);
    copy_vector_active(y_out, y_in, n, incy, ctx.typesize);

    fn(&n, alpha, x_in, &incx, y_out, &incy);

    /* MPFR reference: y_ref[i] = alpha * x[i] + y_in[i] */
    MpfrMatrix y_ref(n, 1, prec);
    {
        MpfrScalar t(prec);
        for (int i = 0; i < n; ++i) {
            mpfr_mul(t.get(), alpha_mpfr.get(), x_mpfr.at(i, 0), MPFR_RNDN);
            mpfr_add(y_ref.at(i, 0), t.get(), y_mpfr.at(i, 0), MPFR_RNDN);
        }
    }

    ErrorResult err = compute_error_vector(y_ref, y_out, incy, ctx);

    SentinelResult sr = check_vector_sentinels(y_out, n, incy, ctx.typesize, sentinel_seed);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d incy=%d", n, incx, incy);
    report_result("AXPY", params_str, err, &sr, format);

    std::free(x_in);
    std::free(y_in);
    std::free(y_out);
    std::free(alpha);
}
