/* dotc.cpp -- BLAS Level 1 DOTC accuracy tester (complex-only) */

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

/* DOTC: result = sum_i conj(x(i)) * y(i)
   ABI issue: complex return values vary by platform.
   - Hidden arg ABI: void cdotc_(void *result, const int *n, ...)
   - Register ABI:   _Complex T cdotc_(const int *n, ...) */

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_dotc(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "DOTC requires --complex\n");
        return;
    }

    mpfr_prec_t prec = ctx.prec;
    int n = params.n;
    int incx = params.incx;
    int incy = params.incy;
    int abs_incx = (incx < 0) ? -incx : incx;
    int abs_incy = (incy < 0) ? -incy : incy;
    int alloc_x = 1 + (n - 1) * abs_incx;
    int alloc_y = 1 + (n - 1) * abs_incy;

    unsigned seed_x = params.seed;
    unsigned seed_y = params.seed + 1;

    void *x = gen_random_complex_array(alloc_x, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
    void *y = gen_random_complex_array(alloc_y, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_y);

    MpfrComplexMatrix x_mpfr(n, 1, prec), y_mpfr(n, 1, prec);
    custom_to_mpfr_complex_vec(x_mpfr, x, incx, ctx);
    custom_to_mpfr_complex_vec(y_mpfr, y, incy, ctx);

    void *result_buf = std::calloc(1, ctx.typesize);

    if (ctx.complex_return_abi == ComplexReturnABI::Hidden) {
        /* Hidden first argument ABI */
        auto *fn = reinterpret_cast<void (*)(void *, const int *,
                                             const void *, const int *,
                                             const void *, const int *)>(
            load_sym(lib, sym));
        fn(result_buf, &n, x, &incx, y, &incy);
    } else {
        /* Register return ABI - dispatch on typesize */
        if (ctx.typesize == 8) {
            /* complex float */
            struct cf { float re, im; };
            auto *fn = reinterpret_cast<cf (*)(const int *,
                                               const void *, const int *,
                                               const void *, const int *)>(
                load_sym(lib, sym));
            cf result = fn(&n, x, &incx, y, &incy);
            std::memcpy(result_buf, &result, sizeof(cf));
        } else if (ctx.typesize == 16) {
            /* complex double */
            struct cd { double re, im; };
            auto *fn = reinterpret_cast<cd (*)(const int *,
                                               const void *, const int *,
                                               const void *, const int *)>(
                load_sym(lib, sym));
            cd result = fn(&n, x, &incx, y, &incy);
            std::memcpy(result_buf, &result, sizeof(cd));
        } else {
            std::fprintf(stderr, "DOTC: register ABI unsupported for typesize %zu\n",
                         ctx.typesize);
            std::free(x); std::free(y); std::free(result_buf);
            return;
        }
    }

    /* MPFR reference: dotc = sum_i conj(x[i]) * y[i] */
    MpfrComplexScalar dotc_ref(prec);
    {
        MpfrComplexScalar conj_x(prec);
        mpfr_set_d(dotc_ref.re(), 0.0, MPFR_RNDN);
        mpfr_set_d(dotc_ref.im(), 0.0, MPFR_RNDN);
        for (int i = 0; i < n; ++i) {
            mpfr_complex_conj(conj_x.re(), conj_x.im(),
                              x_mpfr.re(i, 0), x_mpfr.im(i, 0), MPFR_RNDN);
            mpfr_complex_fma(dotc_ref.re(), dotc_ref.im(),
                             conj_x.re(), conj_x.im(),
                             y_mpfr.re(i, 0), y_mpfr.im(i, 0), MPFR_RNDN);
        }
    }

    ErrorResult err = compute_error_complex_scalar(dotc_ref, result_buf, ctx);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d incy=%d", n, incx, incy);
    report_result("DOTC", params_str, err, format);

    std::free(x); std::free(y); std::free(result_buf);
}
