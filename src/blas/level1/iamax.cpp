/* iamax.cpp -- BLAS Level 1 IAMAX accuracy tester */

#include "../level1.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <cstdio>
#include <cstdlib>

/* Fortran ABI: integer function iamax(n, x, incx) -- returns int */
extern "C" typedef int (*iamax_fn_t)(
    const int *n, const void *x, const int *incx
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_iamax(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<iamax_fn_t>(load_sym(lib, sym));
    mpfr_prec_t prec = ctx.prec;

    int n    = params.n;
    int incx = params.incx;
    int abs_incx = (incx < 0) ? -incx : incx;
    int alloc_x = 1 + (n - 1) * abs_incx;

    unsigned seed_x = params.seed;

    void *x = gen_random_array(alloc_x, ctx.typesize, ctx.from_mpfr, prec, &seed_x);

    /* Convert input to MPFR */
    MpfrMatrix x_mpfr(n, 1, prec);
    custom_to_mpfr_vec(x_mpfr, x, incx, ctx);

    /* Call the routine */
    int result = fn(&n, x, &incx);

    /* MPFR reference: find 1-based index of max |x[i]| */
    int ref_idx = 1;
    {
        MpfrScalar max_val(prec), cur(prec);
        mpfr_abs(max_val.get(), x_mpfr.at(0, 0), MPFR_RNDN);
        for (int i = 1; i < n; ++i) {
            mpfr_abs(cur.get(), x_mpfr.at(i, 0), MPFR_RNDN);
            if (mpfr_cmp(cur.get(), max_val.get()) > 0) {
                mpfr_set(max_val.get(), cur.get(), MPFR_RNDN);
                ref_idx = i + 1; /* 1-based */
            }
        }
    }

    bool match = compute_error_index(ref_idx, result);

    ErrorResult err;
    err.max_relative = match ? 0.0 : 1.0;
    err.normwise_relative = match ? 0.0 : 1.0;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d", n, incx);
    report_result("IAMAX", params_str, err, format);

    std::free(x);
}
