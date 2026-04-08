/* crscal.cpp -- BLAS Level 1 CSSCAL/ZDSCAL accuracy tester (complex-only) */

#include "../level1.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* CSSCAL/ZDSCAL: Scale complex vector by real scalar.
   x_out(i) = alpha * x(i)
   where alpha is REAL and x is complex. */

extern "C" typedef void (*crscal_fn_t)(
    const int *n,
    const void *alpha,
    void *x, const int *incx
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_crscal(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "CRSCAL requires --complex\n");
        return;
    }

    auto *fn = reinterpret_cast<crscal_fn_t>(load_sym(lib, sym));
    mpfr_prec_t prec = ctx.prec;

    int n    = params.n;
    int incx = params.incx;
    int abs_incx = (incx < 0) ? -incx : incx;
    int alloc_x = 1 + (n - 1) * abs_incx;

    std::size_t real_typesize = ctx.typesize / 2;

    unsigned seed_x = params.seed;
    unsigned seed_a = params.seed + 1;

    void *x_in  = gen_random_complex_array(alloc_x, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
    void *alpha = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_a);

    /* Convert inputs to MPFR */
    MpfrComplexMatrix x_mpfr(n, 1, prec);
    custom_to_mpfr_complex_vec(x_mpfr, x_in, incx, ctx);

    MpfrScalar alpha_mpfr(prec);
    ctx.to_mpfr(alpha_mpfr.get(), alpha);

    /* Copy input for the call (routine modifies in-place) */
    unsigned sentinel_seed = 0xDEAD0001;
    void *x_out = alloc_with_sentinel(alloc_x, ctx.typesize, sentinel_seed);
    copy_vector_active(x_out, x_in, n, incx, ctx.typesize);

    fn(&n, alpha, x_out, &incx);

    /* MPFR reference: x_ref[i] = alpha * x[i] */
    MpfrComplexMatrix x_ref(n, 1, prec);
    for (int i = 0; i < n; ++i)
        mpfr_complex_mul_real(x_ref.re(i, 0), x_ref.im(i, 0),
                              x_mpfr.re(i, 0), x_mpfr.im(i, 0),
                              alpha_mpfr.get(), MPFR_RNDN);

    ErrorResult err = compute_error_complex_vector(x_ref, x_out, incx, ctx);

    SentinelResult sr = check_vector_sentinels(x_out, n, incx, ctx.typesize, sentinel_seed);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d", n, incx);
    report_result("CRSCAL", params_str, err, &sr, format);

    std::free(x_in);
    std::free(x_out);
    std::free(alpha);
}
