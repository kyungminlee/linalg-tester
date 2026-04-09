/* iamax.cpp -- Mirror tester for BLAS Level 1 IAMAX */

#include "../mirror_blas1.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* Fortran ABI: integer function iamax(n, x, incx) */
extern "C" typedef int (*iamax_fn_t)(
    const int *n, const void *x, const int *incx
);

void mirror_test_iamax(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n    = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;

    /* Generate canonical MPFR data */
    MpfrMatrix x_mpfr(n, 1, prec);
    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);

    auto run_side = [&](const MirrorSide &side) -> int {
        void *x_n = mpfr_vec_to_native(x_mpfr, incx, side.ctx);

        auto *fn = reinterpret_cast<iamax_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        int idx = fn(&n, x_n, &incx);

        std::free(x_n);
        return idx;
    };

    int idx_a = run_side(a);
    int idx_b = run_side(b);

    ErrorResult err;
    if (idx_a == idx_b) {
        err.max_relative = 0.0;
        err.normwise_relative = 0.0;
        err.max_absolute_at_zero = -1.0;
        err.nan_inf_mismatches = 0;
    } else {
        err.max_relative = std::nan("");
        err.normwise_relative = std::nan("");
        err.max_absolute_at_zero = -1.0;
        err.nan_inf_mismatches = 1;
    }

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d idx_a=%d idx_b=%d", n, incx, idx_a, idx_b);
    mirror_report_result("IAMAX", params_str, err, nullptr, nullptr, config);
}
