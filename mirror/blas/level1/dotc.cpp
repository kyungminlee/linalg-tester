/* dotc.cpp -- Mirror tester for BLAS Level 1 DOTC (complex-only) */

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

void mirror_test_dotc(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n    = params.n;
    int incx = params.incx;
    int incy = params.incy;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;
    unsigned seed_y = params.seed + 1;

    MpfrComplexMatrix x_mpfr(n, 1, prec);
    MpfrComplexMatrix y_mpfr(n, 1, prec);

    gen_mpfr_random_complex_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_complex_vector(y_mpfr, prec, &seed_y);

    auto run_side = [&](const MirrorSide &side, MpfrComplexScalar &result) {
        void *x_n = mpfr_complex_vec_to_native(x_mpfr, incx, side.ctx);
        void *y_n = mpfr_complex_vec_to_native(y_mpfr, incy, side.ctx);

        void *result_buf = std::calloc(1, side.ctx.typesize);

        if (side.ctx.complex_return_abi == ComplexReturnABI::Hidden) {
            auto *fn = reinterpret_cast<void (*)(void *, const int *,
                                                 const void *, const int *,
                                                 const void *, const int *)>(
                load_sym(side.lib, side.sym.c_str()));
            fn(result_buf, &n, x_n, &incx, y_n, &incy);
        } else {
            if (side.ctx.typesize == 8) {
                struct cf { float re, im; };
                auto *fn = reinterpret_cast<cf (*)(const int *,
                                                   const void *, const int *,
                                                   const void *, const int *)>(
                    load_sym(side.lib, side.sym.c_str()));
                cf r = fn(&n, x_n, &incx, y_n, &incy);
                std::memcpy(result_buf, &r, sizeof(cf));
            } else if (side.ctx.typesize == 16) {
                struct cd { double re, im; };
                auto *fn = reinterpret_cast<cd (*)(const int *,
                                                   const void *, const int *,
                                                   const void *, const int *)>(
                    load_sym(side.lib, side.sym.c_str()));
                cd r = fn(&n, x_n, &incx, y_n, &incy);
                std::memcpy(result_buf, &r, sizeof(cd));
            } else {
                std::fprintf(stderr,
                    "DOTC mirror: register ABI unsupported for typesize %zu\n",
                    side.ctx.typesize);
                std::free(x_n); std::free(y_n); std::free(result_buf);
                return;
            }
        }

        side.ctx.to_mpfr_complex(result.re(), result.im(), result_buf);

        std::free(result_buf);
        std::free(x_n);
        std::free(y_n);
    };

    MpfrComplexScalar res_a(prec), res_b(prec);
    run_side(a, res_a);
    run_side(b, res_b);

    const MpfrComplexScalar &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrComplexScalar &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_complex_scalar(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d incy=%d", n, incx, incy);
    mirror_report_result("DOTC", params_str, err, nullptr, nullptr, config);
}
