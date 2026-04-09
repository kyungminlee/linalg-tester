/* rotm.cpp -- Mirror tester for BLAS Level 1 ROTM */

#include "../mirror_blas1.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

/* Fortran ABI: void rotm(n, x, incx, y, incy, param) */
extern "C" typedef void (*rotm_fn_t)(
    const int *n, void *x, const int *incx,
    void *y, const int *incy, const void *param
);

void mirror_test_rotm(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n    = params.n;
    int incx = params.incx;
    int incy = params.incy;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;
    unsigned seed_y = params.seed + 1;
    unsigned seed_h = params.seed + 2;

    /* Generate canonical MPFR data */
    MpfrMatrix x_mpfr(n, 1, prec);
    MpfrMatrix y_mpfr(n, 1, prec);
    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_vector(y_mpfr, prec, &seed_y);

    /* Build param: 5-element array.  param[0] = -1.0 (flag for full H matrix),
       param[1..4] = random H entries (H11, H21, H12, H22 in BLAS column-major). */
    MpfrScalar flag_mpfr(prec);
    mpfr_set_d(flag_mpfr.get(), -1.0, MPFR_RNDN);

    MpfrScalar h1(prec), h2(prec), h3(prec), h4(prec);
    gen_mpfr_random_scalar(h1, prec, &seed_h);
    gen_mpfr_random_scalar(h2, prec, &seed_h);
    gen_mpfr_random_scalar(h3, prec, &seed_h);
    gen_mpfr_random_scalar(h4, prec, &seed_h);

    auto run_side = [&](const MirrorSide &side,
                         MpfrMatrix &out_x, MpfrMatrix &out_y) {
        void *x_n = mpfr_vec_to_native(x_mpfr, incx, side.ctx);
        void *y_n = mpfr_vec_to_native(y_mpfr, incy, side.ctx);

        /* Build native param array (5 elements) */
        void *param_n = std::malloc(5 * side.ctx.typesize);
        char *pp = static_cast<char *>(param_n);
        side.ctx.from_mpfr(pp,                          flag_mpfr.get(), MPFR_RNDN);
        side.ctx.from_mpfr(pp + side.ctx.typesize,      h1.get(), MPFR_RNDN);
        side.ctx.from_mpfr(pp + 2 * side.ctx.typesize,  h2.get(), MPFR_RNDN);
        side.ctx.from_mpfr(pp + 3 * side.ctx.typesize,  h3.get(), MPFR_RNDN);
        side.ctx.from_mpfr(pp + 4 * side.ctx.typesize,  h4.get(), MPFR_RNDN);

        auto *fn = reinterpret_cast<rotm_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&n, x_n, &incx, y_n, &incy, param_n);

        custom_to_mpfr_vec(out_x, x_n, incx, side.ctx);
        custom_to_mpfr_vec(out_y, y_n, incy, side.ctx);

        std::free(x_n);
        std::free(y_n);
        std::free(param_n);
    };

    MpfrMatrix xa(n, 1, prec), ya(n, 1, prec);
    MpfrMatrix xb(n, 1, prec), yb(n, 1, prec);
    run_side(a, xa, ya);
    run_side(b, xb, yb);

    const MpfrMatrix &ref_x = (config.reference == "a") ? xa : xb;
    const MpfrMatrix &tst_x = (config.reference == "a") ? xb : xa;
    const MpfrMatrix &ref_y = (config.reference == "a") ? ya : yb;
    const MpfrMatrix &tst_y = (config.reference == "a") ? yb : ya;

    ErrorResult err_x = compute_error_mpfr_vector(ref_x, tst_x, prec);
    ErrorResult err_y = compute_error_mpfr_vector(ref_y, tst_y, prec);

    ErrorResult err;
    err.max_relative = std::max(err_x.max_relative, err_y.max_relative);
    err.normwise_relative = std::max(err_x.normwise_relative,
                                      err_y.normwise_relative);
    err.max_absolute_at_zero = std::max(err_x.max_absolute_at_zero,
                                         err_y.max_absolute_at_zero);
    err.nan_inf_mismatches = err_x.nan_inf_mismatches +
                              err_y.nan_inf_mismatches;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d incy=%d flag=-1", n, incx, incy);
    mirror_report_result("ROTM", params_str, err, nullptr, nullptr, config);
}
