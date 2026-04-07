/* asum.cpp -- BLAS Level 1 ASUM accuracy tester */

#include "../level1.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ASUM returns a scalar by value. */

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_asum(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
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

    /* Call the routine with the appropriate return type */
    void *result_buf = std::calloc(1, ctx.typesize);

    if (ctx.typesize == 4) {
        auto *fn = reinterpret_cast<float (*)(const int *, const void *,
                                              const int *)>(
            load_sym(lib, sym));
        float result = fn(&n, x, &incx);
        std::memcpy(result_buf, &result, sizeof(float));
    } else if (ctx.typesize == 8) {
        auto *fn = reinterpret_cast<double (*)(const int *, const void *,
                                               const int *)>(
            load_sym(lib, sym));
        double result = fn(&n, x, &incx);
        std::memcpy(result_buf, &result, sizeof(double));
    } else if (ctx.typesize == 16) {
        auto *fn = reinterpret_cast<long double (*)(const int *, const void *,
                                                     const int *)>(
            load_sym(lib, sym));
        long double result = fn(&n, x, &incx);
        std::memcpy(result_buf, &result, ctx.typesize);
    } else {
        std::fprintf(stderr, "ASUM: unsupported typesize %zu, skipping\n",
                     ctx.typesize);
        std::free(x);
        std::free(result_buf);
        return;
    }

    /* MPFR reference: asum = sum_i |x[i]| */
    MpfrScalar asum_ref(prec);
    {
        MpfrScalar acc(prec), t(prec);
        mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
        for (int i = 0; i < n; ++i) {
            mpfr_abs(t.get(), x_mpfr.at(i, 0), MPFR_RNDN);
            mpfr_add(acc.get(), acc.get(), t.get(), MPFR_RNDN);
        }
        mpfr_set(asum_ref.get(), acc.get(), MPFR_RNDN);
    }

    ErrorResult err = compute_error_scalar(asum_ref, result_buf, ctx);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d", n, incx);
    report_result("ASUM", params_str, err, format);

    std::free(x);
    std::free(result_buf);
}
