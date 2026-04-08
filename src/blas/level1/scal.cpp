/* scal.cpp -- BLAS Level 1 SCAL accuracy tester */

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

/* Fortran ABI: void scal(n, alpha, x, incx) -- no hidden char lengths */
extern "C" typedef void (*scal_fn_t)(
    const int *n, const void *alpha, void *x, const int *incx
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_scal(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<scal_fn_t>(load_sym(lib, sym));
    mpfr_prec_t prec = ctx.prec;

    int n    = params.n;
    int incx = params.incx;
    int abs_incx = (incx < 0) ? -incx : incx;
    int alloc_x = 1 + (n - 1) * abs_incx;

    unsigned seed_x = params.seed;
    unsigned seed_a = params.seed + 1;

    void *x_in  = gen_random_array(alloc_x, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
    void *alpha = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_a);

    /* Convert inputs to MPFR */
    MpfrMatrix x_mpfr(n, 1, prec);
    custom_to_mpfr_vec(x_mpfr, x_in, incx, ctx);

    MpfrScalar alpha_mpfr(prec);
    ctx.to_mpfr(alpha_mpfr.get(), alpha);

    /* Copy input for the call */
    unsigned sentinel_seed = 0xDEAD0001;
    void *x_out = alloc_with_sentinel(alloc_x, ctx.typesize, sentinel_seed);
    copy_vector_active(x_out, x_in, n, incx, ctx.typesize);

    fn(&n, alpha, x_out, &incx);

    /* MPFR reference: x_ref[i] = alpha * x[i] */
    MpfrMatrix x_ref(n, 1, prec);
    for (int i = 0; i < n; ++i)
        mpfr_mul(x_ref.at(i, 0), alpha_mpfr.get(), x_mpfr.at(i, 0), MPFR_RNDN);

    ErrorResult err = compute_error_vector(x_ref, x_out, incx, ctx);

    SentinelResult sr = check_vector_sentinels(x_out, n, incx, ctx.typesize, sentinel_seed);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d", n, incx);
    report_result("SCAL", params_str, err, &sr, format);

    std::free(x_in);
    std::free(x_out);
    std::free(alpha);
}
