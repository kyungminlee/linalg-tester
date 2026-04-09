/* rot.cpp -- Mirror tester for BLAS Level 1 ROT */

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

/* Fortran ABI: void rot(n, x, incx, y, incy, c, s) */
extern "C" typedef void (*rot_fn_t)(
    const int *n, void *x, const int *incx,
    void *y, const int *incy,
    const void *c, const void *s
);

void mirror_test_rot(const MirrorSide &a, const MirrorSide &b,
                      const TestParams &params, const MirrorConfig &config)
{
    int n    = params.n;
    int incx = params.incx;
    int incy = params.incy;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x  = params.seed;
    unsigned seed_y  = params.seed + 1;
    unsigned seed_cs = params.seed + 2;

    /* Generate canonical MPFR data */
    MpfrMatrix x_mpfr(n, 1, prec);
    MpfrMatrix y_mpfr(n, 1, prec);
    MpfrScalar c_mpfr(prec), s_mpfr(prec);

    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_vector(y_mpfr, prec, &seed_y);
    gen_mpfr_random_scalar(c_mpfr, prec, &seed_cs);
    gen_mpfr_random_scalar(s_mpfr, prec, &seed_cs);

    auto run_side = [&](const MirrorSide &side,
                         MpfrMatrix &out_x, MpfrMatrix &out_y) {
        void *x_n = mpfr_vec_to_native(x_mpfr, incx, side.ctx);
        void *y_n = mpfr_vec_to_native(y_mpfr, incy, side.ctx);
        void *c_n = mpfr_scalar_to_native(c_mpfr, side.ctx);
        void *s_n = mpfr_scalar_to_native(s_mpfr, side.ctx);

        auto *fn = reinterpret_cast<rot_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&n, x_n, &incx, y_n, &incy, c_n, s_n);

        custom_to_mpfr_vec(out_x, x_n, incx, side.ctx);
        custom_to_mpfr_vec(out_y, y_n, incy, side.ctx);

        std::free(x_n);
        std::free(y_n);
        std::free(c_n);
        std::free(s_n);
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
                  "n=%d incx=%d incy=%d", n, incx, incy);
    mirror_report_result("ROT", params_str, err, nullptr, nullptr, config);
}
