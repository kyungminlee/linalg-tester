/* scal.cpp -- Mirror tester for BLAS Level 1 SCAL */

#include "../mirror_blas1.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran ABI: void scal(n, alpha, x, incx) */
extern "C" typedef void (*scal_fn_t)(
    const int *n, const void *alpha,
    void *x, const int *incx
);

void mirror_test_scal(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n    = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;
    unsigned seed_a = params.seed + 1;

    /* Generate canonical MPFR data */
    MpfrMatrix x_mpfr(n, 1, prec);
    MpfrScalar alpha_mpfr(prec);

    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_a);

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
        void *x_n     = mpfr_vec_to_native(x_mpfr, incx, side.ctx);
        void *alpha_n = mpfr_scalar_to_native(alpha_mpfr, side.ctx);

        auto *fn = reinterpret_cast<scal_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&n, alpha_n, x_n, &incx);

        custom_to_mpfr_vec(result, x_n, incx, side.ctx);

        std::free(x_n);
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
                  "n=%d incx=%d", n, incx);
    mirror_report_result("SCAL", params_str, err, nullptr, nullptr, config);
}
