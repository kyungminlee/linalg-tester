/* swap.cpp -- BLAS Level 1 SWAP accuracy tester */

#include "../level1.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran ABI: void swap(n, x, incx, y, incy) -- no hidden char lengths */
extern "C" typedef void (*swap_fn_t)(
    const int *n, void *x, const int *incx,
    void *y, const int *incy
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_swap(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<swap_fn_t>(load_sym(lib, sym));
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

    void *x_in = gen_random_array(alloc_x, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
    void *y_in = gen_random_array(alloc_y, ctx.typesize, ctx.from_mpfr, prec, &seed_y);

    /* MPFR reference: after swap, x should contain original y and vice versa */
    MpfrMatrix x_ref(n, 1, prec), y_ref(n, 1, prec);
    custom_to_mpfr_vec(x_ref, y_in, incy, ctx);  /* x_ref = y_in */
    custom_to_mpfr_vec(y_ref, x_in, incx, ctx);  /* y_ref = x_in */

    /* Copy inputs for the call */
    void *x_out = std::malloc(static_cast<std::size_t>(alloc_x) * ctx.typesize);
    void *y_out = std::malloc(static_cast<std::size_t>(alloc_y) * ctx.typesize);
    std::memcpy(x_out, x_in, static_cast<std::size_t>(alloc_x) * ctx.typesize);
    std::memcpy(y_out, y_in, static_cast<std::size_t>(alloc_y) * ctx.typesize);

    fn(&n, x_out, &incx, y_out, &incy);

    ErrorResult err_x = compute_error_vector(x_ref, x_out, incx, ctx);
    ErrorResult err_y = compute_error_vector(y_ref, y_out, incy, ctx);

    ErrorResult err;
    err.max_relative = std::max(err_x.max_relative, err_y.max_relative);
    err.normwise_relative = std::max(err_x.normwise_relative, err_y.normwise_relative);
    err.max_absolute_at_zero = std::max(err_x.max_absolute_at_zero, err_y.max_absolute_at_zero);
    err.nan_inf_mismatches = err_x.nan_inf_mismatches + err_y.nan_inf_mismatches;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d incy=%d", n, incx, incy);
    report_result("SWAP", params_str, err, format);

    std::free(x_in);
    std::free(y_in);
    std::free(x_out);
    std::free(y_out);
}
