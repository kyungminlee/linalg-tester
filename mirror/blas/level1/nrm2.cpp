/* nrm2.cpp -- Mirror tester for BLAS Level 1 NRM2 */

#include "../mirror_blas1.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

void mirror_test_nrm2(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n    = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;

    /* Generate canonical MPFR data */
    MpfrMatrix x_mpfr(n, 1, prec);
    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);

    auto run_side = [&](const MirrorSide &side, MpfrScalar &result) {
        void *x_n = mpfr_vec_to_native(x_mpfr, incx, side.ctx);
        void *result_buf = std::malloc(side.ctx.typesize);

        if (side.ctx.typesize == 4) {
            auto *fn = reinterpret_cast<float (*)(const int *, const void *,
                                                  const int *)>(
                load_sym(side.lib, side.sym.c_str()));
            float r = fn(&n, x_n, &incx);
            std::memcpy(result_buf, &r, sizeof(float));
        } else if (side.ctx.typesize == 8) {
            auto *fn = reinterpret_cast<double (*)(const int *, const void *,
                                                   const int *)>(
                load_sym(side.lib, side.sym.c_str()));
            double r = fn(&n, x_n, &incx);
            std::memcpy(result_buf, &r, sizeof(double));
        } else {
            auto *fn = reinterpret_cast<long double (*)(const int *, const void *,
                                                         const int *)>(
                load_sym(side.lib, side.sym.c_str()));
            long double r = fn(&n, x_n, &incx);
            std::memcpy(result_buf, &r, side.ctx.typesize);
        }

        side.ctx.to_mpfr(result.get(), result_buf);

        std::free(result_buf);
        std::free(x_n);
    };

    MpfrScalar res_a(prec);
    MpfrScalar res_b(prec);
    run_side(a, res_a);
    run_side(b, res_b);

    const MpfrScalar &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrScalar &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_scalar(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d", n, incx);
    mirror_report_result("NRM2", params_str, err, nullptr, nullptr, config);
}
