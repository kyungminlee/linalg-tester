/* dot.cpp -- BLAS Level 1 DOT accuracy tester */

#include "../level1.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* DOT returns a scalar by value. We need type-specific function pointer casts. */

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_dot(const TesterCtx &ctx, void *lib, const char *sym,
              const TestParams &params, const std::string &format)
{
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

    void *x = gen_random_array(alloc_x, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
    void *y = gen_random_array(alloc_y, ctx.typesize, ctx.from_mpfr, prec, &seed_y);

    /* Convert inputs to MPFR */
    MpfrMatrix x_mpfr(n, 1, prec), y_mpfr(n, 1, prec);
    custom_to_mpfr_vec(x_mpfr, x, incx, ctx);
    custom_to_mpfr_vec(y_mpfr, y, incy, ctx);

    /* Call the routine with the appropriate return type */
    void *result_buf = std::calloc(1, ctx.typesize);

    if (ctx.typesize == 4) {
        auto *fn = reinterpret_cast<float (*)(const int *, const void *,
                                              const int *, const void *,
                                              const int *)>(
            load_sym(lib, sym));
        float result = fn(&n, x, &incx, y, &incy);
        std::memcpy(result_buf, &result, sizeof(float));
    } else if (ctx.typesize == 8) {
        auto *fn = reinterpret_cast<double (*)(const int *, const void *,
                                               const int *, const void *,
                                               const int *)>(
            load_sym(lib, sym));
        double result = fn(&n, x, &incx, y, &incy);
        std::memcpy(result_buf, &result, sizeof(double));
    } else if (ctx.typesize == 16) {
        auto *fn = reinterpret_cast<long double (*)(const int *, const void *,
                                                     const int *, const void *,
                                                     const int *)>(
            load_sym(lib, sym));
        long double result = fn(&n, x, &incx, y, &incy);
        std::memcpy(result_buf, &result, ctx.typesize);
    } else {
        std::fprintf(stderr, "DOT: unsupported typesize %zu, skipping\n",
                     ctx.typesize);
        std::free(x);
        std::free(y);
        std::free(result_buf);
        return;
    }

    /* MPFR reference: dot = sum_i x[i] * y[i] */
    MpfrScalar dot_ref(prec);
    {
        MpfrScalar acc(prec), t(prec);
        mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
        for (int i = 0; i < n; ++i) {
            mpfr_mul(t.get(), x_mpfr.at(i, 0), y_mpfr.at(i, 0), MPFR_RNDN);
            mpfr_add(acc.get(), acc.get(), t.get(), MPFR_RNDN);
        }
        mpfr_set(dot_ref.get(), acc.get(), MPFR_RNDN);
    }

    ErrorResult err = compute_error_scalar(dot_ref, result_buf, ctx);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d incy=%d", n, incx, incy);
    report_result("DOT", params_str, err, format);

    std::free(x);
    std::free(y);
    std::free(result_buf);
}
