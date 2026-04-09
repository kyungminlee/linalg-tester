/* crscal.cpp -- Mirror tester for BLAS Level 1 CSSCAL/ZDSCAL (complex-only) */

#include "../mirror_blas1.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/mpfr_complex_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*crscal_fn_t)(
    const int *n,
    const void *alpha,
    void *x, const int *incx
);

void mirror_test_crscal(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n    = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;
    unsigned seed_a = params.seed + 1;

    MpfrComplexMatrix x_mpfr(n, 1, prec);
    MpfrScalar alpha_mpfr(prec);  /* alpha is REAL */

    gen_mpfr_random_complex_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_a);

    auto run_side = [&](const MirrorSide &side, MpfrComplexMatrix &result) {
        void *x_n = mpfr_complex_vec_to_native(x_mpfr, incx, side.ctx);

        /* alpha is REAL: allocate typesize/2 bytes */
        std::size_t real_typesize = side.ctx.typesize / 2;
        void *alpha_n = std::malloc(real_typesize);
        side.ctx.from_mpfr(alpha_n, alpha_mpfr.get(), MPFR_RNDN);

        auto *fn = reinterpret_cast<crscal_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&n, alpha_n, x_n, &incx);

        custom_to_mpfr_complex_vec(result, x_n, incx, side.ctx);

        std::free(x_n);
        std::free(alpha_n);
    };

    MpfrComplexMatrix res_a(n, 1, prec);
    MpfrComplexMatrix res_b(n, 1, prec);
    run_side(a, res_a);
    run_side(b, res_b);

    const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_complex_vector(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d", n, incx);
    mirror_report_result("CRSCAL", params_str, err, nullptr, nullptr, config);
}
