/* rotg.cpp -- Mirror tester for BLAS Level 1 ROTG */

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

/* Fortran ABI: void rotg(a, b, c, s) -- all scalar in/out */
extern "C" typedef void (*rotg_fn_t)(
    void *a, void *b, void *c, void *s
);

void mirror_test_rotg(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    mpfr_prec_t prec = config.prec;

    unsigned seed_a = params.seed;
    unsigned seed_b = params.seed + 1;

    /* Generate canonical MPFR data */
    MpfrScalar a_mpfr(prec), b_mpfr(prec);
    gen_mpfr_random_scalar(a_mpfr, prec, &seed_a);
    gen_mpfr_random_scalar(b_mpfr, prec, &seed_b);

    /* Run one side: call rotg, read back all 4 scalars (a=r, b=z, c, s) */
    auto run_side = [&](const MirrorSide &side,
                         MpfrScalar &out_r, MpfrScalar &out_z,
                         MpfrScalar &out_c, MpfrScalar &out_s) {
        void *a_n = mpfr_scalar_to_native(a_mpfr, side.ctx);
        void *b_n = mpfr_scalar_to_native(b_mpfr, side.ctx);
        void *c_n = std::calloc(1, side.ctx.typesize);
        void *s_n = std::calloc(1, side.ctx.typesize);

        auto *fn = reinterpret_cast<rotg_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(a_n, b_n, c_n, s_n);

        side.ctx.to_mpfr(out_r.get(), a_n);
        side.ctx.to_mpfr(out_z.get(), b_n);
        side.ctx.to_mpfr(out_c.get(), c_n);
        side.ctx.to_mpfr(out_s.get(), s_n);

        std::free(a_n);
        std::free(b_n);
        std::free(c_n);
        std::free(s_n);
    };

    MpfrScalar ra(prec), za(prec), ca(prec), sa(prec);
    MpfrScalar rb(prec), zb(prec), cb(prec), sb(prec);
    run_side(a, ra, za, ca, sa);
    run_side(b, rb, zb, cb, sb);

    /* Compare all four outputs */
    const MpfrScalar &ref_r = (config.reference == "a") ? ra : rb;
    const MpfrScalar &tst_r = (config.reference == "a") ? rb : ra;
    const MpfrScalar &ref_z = (config.reference == "a") ? za : zb;
    const MpfrScalar &tst_z = (config.reference == "a") ? zb : za;
    const MpfrScalar &ref_c = (config.reference == "a") ? ca : cb;
    const MpfrScalar &tst_c = (config.reference == "a") ? cb : ca;
    const MpfrScalar &ref_s = (config.reference == "a") ? sa : sb;
    const MpfrScalar &tst_s = (config.reference == "a") ? sb : sa;

    ErrorResult err_r = compute_error_mpfr_scalar(ref_r, tst_r, prec);
    ErrorResult err_z = compute_error_mpfr_scalar(ref_z, tst_z, prec);
    ErrorResult err_c = compute_error_mpfr_scalar(ref_c, tst_c, prec);
    ErrorResult err_s = compute_error_mpfr_scalar(ref_s, tst_s, prec);

    ErrorResult err;
    err.max_relative = std::max({err_r.max_relative, err_z.max_relative,
                                  err_c.max_relative, err_s.max_relative});
    err.normwise_relative = std::max({err_r.normwise_relative,
                                       err_z.normwise_relative,
                                       err_c.normwise_relative,
                                       err_s.normwise_relative});
    err.max_absolute_at_zero = std::max({err_r.max_absolute_at_zero,
                                          err_z.max_absolute_at_zero,
                                          err_c.max_absolute_at_zero,
                                          err_s.max_absolute_at_zero});
    err.nan_inf_mismatches = err_r.nan_inf_mismatches +
                              err_z.nan_inf_mismatches +
                              err_c.nan_inf_mismatches +
                              err_s.nan_inf_mismatches;

    mirror_report_result("ROTG", "", err, nullptr, nullptr, config);
}
