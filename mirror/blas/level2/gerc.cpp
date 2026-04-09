/* gerc.cpp -- Mirror tester for BLAS Level 2 GERC (complex-only) */

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

extern "C" typedef void (*gerc_fn_t)(
    const int  *m,      const int  *n,
    const void *alpha,
    const void *x,      const int  *incx,
    const void *y,      const int  *incy,
    void       *A,      const int  *lda
);

void mirror_test_gerc(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int incx = params.incx, incy = params.incy;
    mpfr_prec_t prec = config.prec;
    int lda = m + params.ld_pad;

    unsigned seed_x  = params.seed;
    unsigned seed_y  = params.seed + 1;
    unsigned seed_A  = params.seed + 2;
    unsigned seed_al = params.seed + 3;

    MpfrComplexMatrix A_mpfr(m, n, prec);
    MpfrComplexMatrix x_mpfr(m, 1, prec);
    MpfrComplexMatrix y_mpfr(n, 1, prec);
    MpfrComplexScalar alpha_mpfr(prec);

    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);
    gen_mpfr_random_complex_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_complex_vector(y_mpfr, prec, &seed_y);
    gen_mpfr_random_complex_scalar(alpha_mpfr, prec, &seed_al);

    auto run_side = [&](const MirrorSide &side, MpfrComplexMatrix &result) {
        void *native_A     = mpfr_complex_mat_to_native(A_mpfr, lda, side.ctx);
        void *native_x     = mpfr_complex_vec_to_native(x_mpfr, incx, side.ctx);
        void *native_y     = mpfr_complex_vec_to_native(y_mpfr, incy, side.ctx);
        void *native_alpha = mpfr_complex_scalar_to_native(alpha_mpfr, side.ctx);

        auto *fn = reinterpret_cast<gerc_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&m, &n,
           native_alpha,
           native_x, &incx,
           native_y, &incy,
           native_A, &lda);

        custom_to_mpfr_complex_mat(result, native_A, lda, side.ctx);

        std::free(native_A);
        std::free(native_x);
        std::free(native_y);
        std::free(native_alpha);
    };

    MpfrComplexMatrix res_a(m, n, prec); run_side(a, res_a);
    MpfrComplexMatrix res_b(m, n, prec); run_side(b, res_b);

    const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d incx=%d incy=%d",
                  m, n, incx, incy);
    mirror_report_result("GERC", params_str, err,
                          nullptr, nullptr, config);
}
