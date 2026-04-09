/* her.cpp -- Mirror tester for BLAS Level 2 HER (complex-only) */

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

extern "C" typedef void (*her_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *x,      const int  *incx,
    void       *A,      const int  *lda,
    std::size_t uplo_len
);

void mirror_test_her(const MirrorSide &a, const MirrorSide &b,
                      const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
        int lda = n + params.ld_pad;

        unsigned seed_x  = params.seed;
        unsigned seed_A  = params.seed + 1;
        unsigned seed_al = params.seed + 2;

        MpfrComplexMatrix A_mpfr(n, n, prec);
        MpfrComplexMatrix x_mpfr(n, 1, prec);
        MpfrScalar alpha_mpfr(prec);  /* alpha is REAL for HER */

        gen_mpfr_hermitian_matrix(A_mpfr, uplo, prec, &seed_A);
        gen_mpfr_random_complex_vector(x_mpfr, prec, &seed_x);
        gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_al);

        auto run_side = [&](const MirrorSide &side, MpfrComplexMatrix &result) {
            void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, side.ctx);
            void *native_x = mpfr_complex_vec_to_native(x_mpfr, incx, side.ctx);

            /* alpha is REAL: allocate typesize/2 bytes and use from_mpfr */
            std::size_t real_typesize = side.ctx.typesize / 2;
            void *native_alpha = std::malloc(real_typesize);
            side.ctx.from_mpfr(native_alpha, alpha_mpfr.get(), MPFR_RNDN);

            auto *fn = reinterpret_cast<her_fn_t>(
                load_sym(side.lib, side.sym.c_str()));
            fn(&uplo, &n,
               native_alpha,
               native_x, &incx,
               native_A, &lda,
               (std::size_t)1);

            custom_to_mpfr_complex_mat(result, native_A, lda, side.ctx);

            std::free(native_A);
            std::free(native_x);
            std::free(native_alpha);
        };

        MpfrComplexMatrix res_a(n, n, prec); run_side(a, res_a);
        MpfrComplexMatrix res_b(n, n, prec); run_side(b, res_b);

        const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d incx=%d",
                      uplo, n, incx);
        mirror_report_result("HER", params_str, err,
                              nullptr, nullptr, config);
    }
}
