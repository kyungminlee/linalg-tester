/* rotm.cpp -- BLAS Level 1 ROTM accuracy tester */

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

/* Fortran ABI: void rotm(n, x, incx, y, incy, param) -- no hidden char lengths */
extern "C" typedef void (*rotm_fn_t)(
    const int *n, void *x, const int *incx,
    void *y, const int *incy, const void *param
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_rotm(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<rotm_fn_t>(load_sym(lib, sym));
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
    unsigned seed_h = params.seed + 2;

    void *x_in = gen_random_array(alloc_x, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
    void *y_in = gen_random_array(alloc_y, ctx.typesize, ctx.from_mpfr, prec, &seed_y);

    /* Build param array: flag = -1 (all 4 entries specified) + 4 random H entries */
    void *param_buf = std::malloc(5 * ctx.typesize);
    {
        MpfrScalar flag_val(prec);
        mpfr_set_d(flag_val.get(), -1.0, MPFR_RNDN);
        ctx.from_mpfr(param_buf, flag_val.get(), MPFR_RNDN);
    }
    void *h_vals = gen_random_array(4, ctx.typesize, ctx.from_mpfr, prec, &seed_h);
    std::memcpy(static_cast<char *>(param_buf) + ctx.typesize,
                h_vals, 4 * ctx.typesize);

    /* Read param H matrix entries into MPFR.
       BLAS PARAM layout: [flag, H11, H21, H12, H22] (column-major 2x2). */
    MpfrScalar h11(prec), h12(prec), h21(prec), h22(prec);
    const char *hp = static_cast<const char *>(h_vals);
    ctx.to_mpfr(h11.get(), hp);
    ctx.to_mpfr(h21.get(), hp + ctx.typesize);
    ctx.to_mpfr(h12.get(), hp + 2 * ctx.typesize);
    ctx.to_mpfr(h22.get(), hp + 3 * ctx.typesize);

    /* Convert input vectors to MPFR */
    MpfrMatrix x_mpfr(n, 1, prec), y_mpfr(n, 1, prec);
    custom_to_mpfr_vec(x_mpfr, x_in, incx, ctx);
    custom_to_mpfr_vec(y_mpfr, y_in, incy, ctx);

    /* Copy inputs for the call */
    void *x_out = std::malloc(static_cast<std::size_t>(alloc_x) * ctx.typesize);
    void *y_out = std::malloc(static_cast<std::size_t>(alloc_y) * ctx.typesize);
    std::memcpy(x_out, x_in, static_cast<std::size_t>(alloc_x) * ctx.typesize);
    std::memcpy(y_out, y_in, static_cast<std::size_t>(alloc_y) * ctx.typesize);

    fn(&n, x_out, &incx, y_out, &incy, param_buf);

    /* MPFR reference: x_ref[i] = h11*x[i] + h12*y[i],
                       y_ref[i] = h21*x[i] + h22*y[i] */
    MpfrMatrix x_ref(n, 1, prec), y_ref(n, 1, prec);
    {
        MpfrScalar t0(prec), t1(prec);
        for (int i = 0; i < n; ++i) {
            mpfr_mul(t0.get(), h11.get(), x_mpfr.at(i, 0), MPFR_RNDN);
            mpfr_mul(t1.get(), h12.get(), y_mpfr.at(i, 0), MPFR_RNDN);
            mpfr_add(x_ref.at(i, 0), t0.get(), t1.get(), MPFR_RNDN);

            mpfr_mul(t0.get(), h21.get(), x_mpfr.at(i, 0), MPFR_RNDN);
            mpfr_mul(t1.get(), h22.get(), y_mpfr.at(i, 0), MPFR_RNDN);
            mpfr_add(y_ref.at(i, 0), t0.get(), t1.get(), MPFR_RNDN);
        }
    }

    ErrorResult err_x = compute_error_vector(x_ref, x_out, incx, ctx);
    ErrorResult err_y = compute_error_vector(y_ref, y_out, incy, ctx);

    ErrorResult err;
    err.max_relative = std::max(err_x.max_relative, err_y.max_relative);
    err.normwise_relative = std::max(err_x.normwise_relative, err_y.normwise_relative);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d incy=%d flag=-1", n, incx, incy);
    report_result("ROTM", params_str, err, format);

    std::free(x_in);
    std::free(y_in);
    std::free(x_out);
    std::free(y_out);
    std::free(param_buf);
    std::free(h_vals);
}
