/* rotmg.cpp -- Mirror tester for BLAS Level 1 ROTMG */

#include "../mirror_blas1.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

/* Fortran ABI: void rotmg(d1, d2, x1, y1, param) */
extern "C" typedef void (*rotmg_fn_t)(
    void *d1, void *d2, void *x1, const void *y1, void *param
);

void mirror_test_rotmg(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    mpfr_prec_t prec = config.prec;

    unsigned seed_d1 = params.seed;
    unsigned seed_d2 = params.seed + 1;
    unsigned seed_x1 = params.seed + 2;
    unsigned seed_y1 = params.seed + 3;

    /* Generate canonical MPFR data */
    MpfrScalar d1_mpfr(prec), d2_mpfr(prec), x1_mpfr(prec), y1_mpfr(prec);
    gen_mpfr_random_scalar(d1_mpfr, prec, &seed_d1);
    gen_mpfr_random_scalar(d2_mpfr, prec, &seed_d2);
    gen_mpfr_random_scalar(x1_mpfr, prec, &seed_x1);
    gen_mpfr_random_scalar(y1_mpfr, prec, &seed_y1);

    /* Make d1 positive */
    mpfr_abs(d1_mpfr.get(), d1_mpfr.get(), MPFR_RNDN);
    if (mpfr_zero_p(d1_mpfr.get()))
        mpfr_set_d(d1_mpfr.get(), 1.0, MPFR_RNDN);

    /* Run one side: call rotmg, read back d1, d2, x1, and param[0..4] */
    auto run_side = [&](const MirrorSide &side,
                         MpfrScalar &out_d1, MpfrScalar &out_d2,
                         MpfrScalar &out_x1,
                         MpfrScalar &out_p0, MpfrScalar &out_p1,
                         MpfrScalar &out_p2, MpfrScalar &out_p3,
                         MpfrScalar &out_p4) {
        void *d1_n = mpfr_scalar_to_native(d1_mpfr, side.ctx);
        void *d2_n = mpfr_scalar_to_native(d2_mpfr, side.ctx);
        void *x1_n = mpfr_scalar_to_native(x1_mpfr, side.ctx);
        void *y1_n = mpfr_scalar_to_native(y1_mpfr, side.ctx);
        void *param_n = std::calloc(5, side.ctx.typesize);

        auto *fn = reinterpret_cast<rotmg_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(d1_n, d2_n, x1_n, y1_n, param_n);

        side.ctx.to_mpfr(out_d1.get(), d1_n);
        side.ctx.to_mpfr(out_d2.get(), d2_n);
        side.ctx.to_mpfr(out_x1.get(), x1_n);

        const char *pp = static_cast<const char *>(param_n);
        side.ctx.to_mpfr(out_p0.get(), pp);
        side.ctx.to_mpfr(out_p1.get(), pp + side.ctx.typesize);
        side.ctx.to_mpfr(out_p2.get(), pp + 2 * side.ctx.typesize);
        side.ctx.to_mpfr(out_p3.get(), pp + 3 * side.ctx.typesize);
        side.ctx.to_mpfr(out_p4.get(), pp + 4 * side.ctx.typesize);

        std::free(d1_n);
        std::free(d2_n);
        std::free(x1_n);
        std::free(y1_n);
        std::free(param_n);
    };

    MpfrScalar d1a(prec), d2a(prec), x1a(prec);
    MpfrScalar p0a(prec), p1a(prec), p2a(prec), p3a(prec), p4a(prec);
    MpfrScalar d1b(prec), d2b(prec), x1b(prec);
    MpfrScalar p0b(prec), p1b(prec), p2b(prec), p3b(prec), p4b(prec);

    run_side(a, d1a, d2a, x1a, p0a, p1a, p2a, p3a, p4a);
    run_side(b, d1b, d2b, x1b, p0b, p1b, p2b, p3b, p4b);

    /* Compare all 8 outputs: d1, d2, x1, param[0..4] */
    auto pick_ref = [&](const MpfrScalar &va, const MpfrScalar &vb)
        -> const MpfrScalar & {
        return (config.reference == "a") ? va : vb;
    };
    auto pick_tst = [&](const MpfrScalar &va, const MpfrScalar &vb)
        -> const MpfrScalar & {
        return (config.reference == "a") ? vb : va;
    };

    ErrorResult errs[8];
    errs[0] = compute_error_mpfr_scalar(pick_ref(d1a, d1b), pick_tst(d1a, d1b), prec);
    errs[1] = compute_error_mpfr_scalar(pick_ref(d2a, d2b), pick_tst(d2a, d2b), prec);
    errs[2] = compute_error_mpfr_scalar(pick_ref(x1a, x1b), pick_tst(x1a, x1b), prec);
    errs[3] = compute_error_mpfr_scalar(pick_ref(p0a, p0b), pick_tst(p0a, p0b), prec);
    errs[4] = compute_error_mpfr_scalar(pick_ref(p1a, p1b), pick_tst(p1a, p1b), prec);
    errs[5] = compute_error_mpfr_scalar(pick_ref(p2a, p2b), pick_tst(p2a, p2b), prec);
    errs[6] = compute_error_mpfr_scalar(pick_ref(p3a, p3b), pick_tst(p3a, p3b), prec);
    errs[7] = compute_error_mpfr_scalar(pick_ref(p4a, p4b), pick_tst(p4a, p4b), prec);

    ErrorResult err;
    err.max_relative = 0.0;
    err.normwise_relative = 0.0;
    err.max_absolute_at_zero = -1.0;
    err.nan_inf_mismatches = 0;

    for (int i = 0; i < 8; ++i) {
        err.max_relative = std::max(err.max_relative,
                                     errs[i].max_relative);
        err.normwise_relative = std::max(err.normwise_relative,
                                          errs[i].normwise_relative);
        err.max_absolute_at_zero = std::max(err.max_absolute_at_zero,
                                              errs[i].max_absolute_at_zero);
        err.nan_inf_mismatches += errs[i].nan_inf_mismatches;
    }

    mirror_report_result("ROTMG", "", err, nullptr, nullptr, config);
}
