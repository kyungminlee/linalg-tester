/* symv.cpp -- Mirror tester for BLAS Level 2 SYMV */

#include "../mirror_blas2.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*symv_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *A,      const int  *lda,
    const void *x,      const int  *incx,
    const void *beta,
    void       *y,      const int  *incy,
    std::size_t uplo_len
);

void mirror_test_symv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int incx = params.incx, incy = params.incy;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
        int lda = n + params.ld_pad;

        unsigned seed_A  = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_y  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        MpfrMatrix A_mpfr(n, n, prec);
        MpfrMatrix x_mpfr(n, 1, prec);
        MpfrMatrix y_mpfr(n, 1, prec);
        MpfrScalar alpha_mpfr(prec), beta_mpfr(prec);

        gen_mpfr_symmetric_matrix(A_mpfr, uplo, prec, &seed_A);
        gen_mpfr_random_vector(x_mpfr, prec, &seed_x);
        gen_mpfr_random_vector(y_mpfr, prec, &seed_y);
        gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_ab);
        gen_mpfr_random_scalar(beta_mpfr, prec, &seed_ab);

        auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
            void *native_A     = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
            void *native_x     = mpfr_vec_to_native(x_mpfr, incx, side.ctx);
            void *native_y     = mpfr_vec_to_native(y_mpfr, incy, side.ctx);
            void *native_alpha = mpfr_scalar_to_native(alpha_mpfr, side.ctx);
            void *native_beta  = mpfr_scalar_to_native(beta_mpfr, side.ctx);

            auto *fn = reinterpret_cast<symv_fn_t>(
                load_sym(side.lib, side.sym.c_str()));
            fn(&uplo, &n,
               native_alpha, native_A, &lda,
               native_x, &incx,
               native_beta, native_y, &incy,
               (std::size_t)1);

            custom_to_mpfr_vec(result, native_y, incy, side.ctx);

            std::free(native_A);
            std::free(native_x);
            std::free(native_y);
            std::free(native_alpha);
            std::free(native_beta);
        };

        MpfrMatrix res_a(n, 1, prec); run_side(a, res_a);
        MpfrMatrix res_b(n, 1, prec); run_side(b, res_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d incx=%d incy=%d",
                      uplo, n, incx, incy);
        mirror_report_result("SYMV", params_str, err,
                              nullptr, nullptr, config);
    }
}
