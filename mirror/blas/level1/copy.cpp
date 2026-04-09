/* copy.cpp -- Mirror tester for BLAS Level 1 COPY */

#include "../mirror_blas1.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran ABI: void copy(n, x, incx, y, incy) */
extern "C" typedef void (*copy_fn_t)(
    const int *n, const void *x, const int *incx,
    void *y, const int *incy
);

void mirror_test_copy(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n    = params.n;
    int incx = params.incx;
    int incy = params.incy;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;

    /* Generate canonical MPFR data */
    MpfrMatrix x_mpfr(n, 1, prec);
    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
        void *x_n = mpfr_vec_to_native(x_mpfr, incx, side.ctx);

        /* Allocate output y as zeroed buffer */
        int abs_incy = (incy < 0) ? -incy : incy;
        int alloc_y = 1 + (n - 1) * abs_incy;
        void *y_n = std::calloc(static_cast<std::size_t>(alloc_y),
                                side.ctx.typesize);

        auto *fn = reinterpret_cast<copy_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&n, x_n, &incx, y_n, &incy);

        custom_to_mpfr_vec(result, y_n, incy, side.ctx);

        std::free(x_n);
        std::free(y_n);
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
    mirror_report_result("COPY", params_str, err, nullptr, nullptr, config);
}
