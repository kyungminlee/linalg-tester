/* gbmv.cpp -- Mirror tester for BLAS Level 2 GBMV */

#include "../mirror_blas2.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*gbmv_fn_t)(
    const char *trans,
    const int  *m,      const int  *n,
    const int  *kl,     const int  *ku,
    const void *alpha,
    const void *AB,     const int  *ldab,
    const void *x,      const int  *incx,
    const void *beta,
    void       *y,      const int  *incy,
    std::size_t trans_len
);

void mirror_test_gbmv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int kl = params.kl, ku = params.ku;
    int incx = params.incx, incy = params.incy;
    mpfr_prec_t prec = config.prec;

    for (char trans : {'N', 'T', 'C'}) {
        int xlen = (trans == 'N') ? n : m;
        int ylen = (trans == 'N') ? m : n;
        int ldab = kl + ku + 1;

        unsigned seed_AB = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_y  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        MpfrMatrix AB_mpfr(ldab, n, prec);
        MpfrMatrix x_mpfr(xlen, 1, prec);
        MpfrMatrix y_mpfr(ylen, 1, prec);
        MpfrScalar alpha_mpfr(prec), beta_mpfr(prec);

        gen_mpfr_band_matrix(AB_mpfr, m, kl, ku, prec, &seed_AB);
        gen_mpfr_random_vector(x_mpfr, prec, &seed_x);
        gen_mpfr_random_vector(y_mpfr, prec, &seed_y);
        gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_ab);
        gen_mpfr_random_scalar(beta_mpfr, prec, &seed_ab);

        auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
            void *native_AB    = mpfr_mat_to_native(AB_mpfr, ldab, side.ctx);
            void *native_x     = mpfr_vec_to_native(x_mpfr, incx, side.ctx);
            void *native_y     = mpfr_vec_to_native(y_mpfr, incy, side.ctx);
            void *native_alpha = mpfr_scalar_to_native(alpha_mpfr, side.ctx);
            void *native_beta  = mpfr_scalar_to_native(beta_mpfr, side.ctx);

            auto *fn = reinterpret_cast<gbmv_fn_t>(
                load_sym(side.lib, side.sym.c_str()));
            fn(&trans, &m, &n, &kl, &ku,
               native_alpha, native_AB, &ldab,
               native_x, &incx,
               native_beta, native_y, &incy,
               (std::size_t)1);

            custom_to_mpfr_vec(result, native_y, incy, side.ctx);

            std::free(native_AB);
            std::free(native_x);
            std::free(native_y);
            std::free(native_alpha);
            std::free(native_beta);
        };

        MpfrMatrix res_a(ylen, 1, prec); run_side(a, res_a);
        MpfrMatrix res_b(ylen, 1, prec); run_side(b, res_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "trans=%c m=%d n=%d kl=%d ku=%d incx=%d incy=%d",
                      trans, m, n, kl, ku, incx, incy);
        mirror_report_result("GBMV", params_str, err,
                              nullptr, nullptr, config);
    }
}
