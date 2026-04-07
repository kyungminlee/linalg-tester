/* copy.cpp -- BLAS Level 1 COPY accuracy tester */

#include "../level1.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran ABI: void copy(n, x, incx, y, incy) -- no hidden char lengths */
extern "C" typedef void (*copy_fn_t)(
    const int *n, const void *x, const int *incx,
    void *y, const int *incy
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_copy(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<copy_fn_t>(load_sym(lib, sym));
    mpfr_prec_t prec = ctx.prec;

    int n    = params.n;
    int incx = params.incx;
    int incy = params.incy;
    int abs_incx = (incx < 0) ? -incx : incx;
    int abs_incy = (incy < 0) ? -incy : incy;
    int alloc_x = 1 + (n - 1) * abs_incx;
    int alloc_y = 1 + (n - 1) * abs_incy;

    unsigned seed_x = params.seed;

    void *x_in = gen_random_array(alloc_x, ctx.typesize, ctx.from_mpfr, prec, &seed_x);

    /* MPFR reference: y_ref = x_in (exact copy) */
    MpfrMatrix y_ref(n, 1, prec);
    custom_to_mpfr_vec(y_ref, x_in, incx, ctx);

    /* Allocate output y (zero-initialized) */
    void *y_out = std::calloc(static_cast<std::size_t>(alloc_y), ctx.typesize);

    fn(&n, x_in, &incx, y_out, &incy);

    ErrorResult err = compute_error_vector(y_ref, y_out, incy, ctx);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d incy=%d", n, incx, incy);
    report_result("COPY", params_str, err, format);

    std::free(x_in);
    std::free(y_out);
}
