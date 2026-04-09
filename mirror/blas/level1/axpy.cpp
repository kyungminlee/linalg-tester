/* axpy.cpp -- Mirror tester for BLAS Level 1 AXPY */

#include "../mirror_blas1.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran ABI: void axpy(n, alpha, x, incx, y, incy) */
extern "C" typedef void (*axpy_fn_t)(
    const int *n, const void *alpha,
    const void *x, const int *incx,
    void *y, const int *incy
);

void mirror_test_axpy(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n    = params.n;
    int incx = params.incx;
    int incy = params.incy;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;
    unsigned seed_y = params.seed + 1;
    unsigned seed_a = params.seed + 2;

    /* Generate canonical MPFR data */
    MpfrMatrix x_mpfr(n, 1, prec);
    MpfrMatrix y_mpfr(n, 1, prec);
    MpfrScalar alpha_mpfr(prec);

    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_vector(y_mpfr, prec, &seed_y);
    gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_a);

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
        void *x_n     = mpfr_vec_to_native(x_mpfr, incx, side.ctx);
        void *y_n     = mpfr_vec_to_native(y_mpfr, incy, side.ctx);
        void *alpha_n = mpfr_scalar_to_native(alpha_mpfr, side.ctx);

        auto *fn = reinterpret_cast<axpy_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&n, alpha_n, x_n, &incx, y_n, &incy);

        custom_to_mpfr_vec(result, y_n, incy, side.ctx);

        std::free(x_n);
        std::free(y_n);
        std::free(alpha_n);
    };

    MpfrMatrix res_a(n, 1, prec);
    MpfrMatrix res_b(n, 1, prec);
    run_side(a, res_a);
    run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d incy=%d", n, incx, incy);
    mirror_report_result("AXPY", params_str, err, nullptr, nullptr, config);
}
