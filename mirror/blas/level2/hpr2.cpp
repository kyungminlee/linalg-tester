/* hpr2.cpp -- Mirror tester for BLAS Level 2 HPR2 (complex-only) */

#include "../mirror_blas2.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/mpfr_complex_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*hpr2_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *x,      const int  *incx,
    const void *y,      const int  *incy,
    void       *AP,
    std::size_t uplo_len
);

void mirror_test_hpr2(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int incx = params.incx, incy = params.incy;
    mpfr_prec_t prec = config.prec;
    int count = n * (n + 1) / 2;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_AP = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_y  = params.seed + 2;
        unsigned seed_al = params.seed + 3;

        MpfrComplexMatrix AP_mpfr(count, 1, prec);
        MpfrComplexMatrix x_mpfr(n, 1, prec);
        MpfrComplexMatrix y_mpfr(n, 1, prec);
        MpfrComplexScalar alpha_mpfr(prec);  /* alpha is COMPLEX for HPR2 */

        gen_mpfr_packed_hermitian(AP_mpfr, n, uplo, prec, &seed_AP);
        gen_mpfr_random_complex_vector(x_mpfr, prec, &seed_x);
        gen_mpfr_random_complex_vector(y_mpfr, prec, &seed_y);
        gen_mpfr_random_complex_scalar(alpha_mpfr, prec, &seed_al);

        auto run_side = [&](const MirrorSide &side, MpfrComplexMatrix &result) {
            void *native_AP    = mpfr_complex_vec_to_native(AP_mpfr, 1, side.ctx);
            void *native_x     = mpfr_complex_vec_to_native(x_mpfr, incx, side.ctx);
            void *native_y     = mpfr_complex_vec_to_native(y_mpfr, incy, side.ctx);
            void *native_alpha = mpfr_complex_scalar_to_native(alpha_mpfr, side.ctx);

            auto *fn = reinterpret_cast<hpr2_fn_t>(
                load_sym(side.lib, side.sym.c_str()));
            fn(&uplo, &n,
               native_alpha,
               native_x, &incx,
               native_y, &incy,
               native_AP,
               (std::size_t)1);

            custom_to_mpfr_complex_vec(result, native_AP, 1, side.ctx);

            std::free(native_AP);
            std::free(native_x);
            std::free(native_y);
            std::free(native_alpha);
        };

        MpfrComplexMatrix res_a(count, 1, prec); run_side(a, res_a);
        MpfrComplexMatrix res_b(count, 1, prec); run_side(b, res_b);

        const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_complex_vector(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d incx=%d incy=%d",
                      uplo, n, incx, incy);
        mirror_report_result("HPR2", params_str, err,
                              nullptr, nullptr, config);
    }
}
