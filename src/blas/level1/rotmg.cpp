/* rotmg.cpp -- BLAS Level 1 ROTMG accuracy tester */

#include "../level1.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran ABI: void rotmg(d1, d2, x1, y1, param) -- no hidden char lengths */
extern "C" typedef void (*rotmg_fn_t)(
    void *d1, void *d2, void *x1, const void *y1, void *param
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_rotmg(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<rotmg_fn_t>(load_sym(lib, sym));
    mpfr_prec_t prec = ctx.prec;

    unsigned seed_d1 = params.seed;
    unsigned seed_d2 = params.seed + 1;
    unsigned seed_x1 = params.seed + 2;
    unsigned seed_y1 = params.seed + 3;

    /* Generate random inputs; d1 should be positive */
    void *d1 = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_d1);
    void *d2 = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_d2);
    void *x1 = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_x1);
    void *y1 = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_y1);

    /* Make d1 positive by taking abs: read, abs, write back */
    {
        MpfrScalar tmp(prec);
        ctx.to_mpfr(tmp.get(), d1);
        mpfr_abs(tmp.get(), tmp.get(), MPFR_RNDN);
        /* Ensure it is not zero */
        if (mpfr_zero_p(tmp.get()))
            mpfr_set_d(tmp.get(), 1.0, MPFR_RNDN);
        ctx.from_mpfr(d1, tmp.get(), MPFR_RNDN);
    }

    /* Save inputs in MPFR for reconstruction check */
    MpfrScalar d1_in(prec), d2_in(prec), x1_in(prec), y1_in(prec);
    ctx.to_mpfr(d1_in.get(), d1);
    ctx.to_mpfr(d2_in.get(), d2);
    ctx.to_mpfr(x1_in.get(), x1);
    ctx.to_mpfr(y1_in.get(), y1);

    /* Allocate param array (5 elements) */
    void *param_buf = std::calloc(5, ctx.typesize);

    /* Call the routine */
    fn(d1, d2, x1, y1, param_buf);

    /* Read outputs */
    MpfrScalar d1_out(prec), d2_out(prec), x1_out(prec);
    ctx.to_mpfr(d1_out.get(), d1);
    ctx.to_mpfr(d2_out.get(), d2);
    ctx.to_mpfr(x1_out.get(), x1);

    /* Read param[0..4] */
    MpfrScalar p0(prec), p1(prec), p2(prec), p3(prec), p4(prec);
    const char *pp = static_cast<const char *>(param_buf);
    ctx.to_mpfr(p0.get(), pp);
    ctx.to_mpfr(p1.get(), pp + ctx.typesize);
    ctx.to_mpfr(p2.get(), pp + 2 * ctx.typesize);
    ctx.to_mpfr(p3.get(), pp + 3 * ctx.typesize);
    ctx.to_mpfr(p4.get(), pp + 4 * ctx.typesize);

    double flag = mpfr_get_d(p0.get(), MPFR_RNDN);

    /* Construct H matrix based on flag */
    MpfrScalar h11(prec), h12(prec), h21(prec), h22(prec);
    if (flag == -1.0) {
        mpfr_set(h11.get(), p1.get(), MPFR_RNDN);
        mpfr_set(h12.get(), p2.get(), MPFR_RNDN);
        mpfr_set(h21.get(), p3.get(), MPFR_RNDN);
        mpfr_set(h22.get(), p4.get(), MPFR_RNDN);
    } else if (flag == 0.0) {
        mpfr_set_d(h11.get(), 1.0, MPFR_RNDN);
        mpfr_set(h12.get(), p2.get(), MPFR_RNDN);
        mpfr_set(h21.get(), p3.get(), MPFR_RNDN);
        mpfr_set_d(h22.get(), 1.0, MPFR_RNDN);
    } else if (flag == 1.0) {
        mpfr_set(h11.get(), p1.get(), MPFR_RNDN);
        mpfr_set_d(h12.get(), 1.0, MPFR_RNDN);
        mpfr_set_d(h21.get(), -1.0, MPFR_RNDN);
        mpfr_set(h22.get(), p4.get(), MPFR_RNDN);
    } else {
        /* flag == -2: identity */
        mpfr_set_d(h11.get(), 1.0, MPFR_RNDN);
        mpfr_set_d(h12.get(), 0.0, MPFR_RNDN);
        mpfr_set_d(h21.get(), 0.0, MPFR_RNDN);
        mpfr_set_d(h22.get(), 1.0, MPFR_RNDN);
    }

    /*
     * Reconstruction check:
     *   H * [sqrt(d1_in) * x1_in; sqrt(d2_in) * y1_in]
     *     should give [sqrt(d1_out) * x1_out; 0]
     *
     * We compute both sides in MPFR and compare.
     * For d2_in < 0, sqrt is not real, so we skip the check.
     */
    bool skip_reconstruction = (mpfr_sgn(d1_in.get()) < 0) ||
                               (mpfr_sgn(d2_in.get()) < 0) ||
                               (mpfr_sgn(d1_out.get()) < 0);

    ErrorResult err;
    err.max_relative = 0.0;
    err.normwise_relative = 0.0;

    if (!skip_reconstruction) {
        MpfrScalar sqrt_d1_in(prec), sqrt_d2_in(prec), sqrt_d1_out(prec);
        mpfr_sqrt(sqrt_d1_in.get(),  d1_in.get(),  MPFR_RNDN);
        mpfr_sqrt(sqrt_d2_in.get(),  d2_in.get(),  MPFR_RNDN);
        mpfr_sqrt(sqrt_d1_out.get(), d1_out.get(), MPFR_RNDN);

        /* lhs = H * [sqrt(d1_in)*x1_in; sqrt(d2_in)*y1_in] */
        MpfrScalar v0(prec), v1(prec);
        mpfr_mul(v0.get(), sqrt_d1_in.get(), x1_in.get(), MPFR_RNDN);
        mpfr_mul(v1.get(), sqrt_d2_in.get(), y1_in.get(), MPFR_RNDN);

        MpfrScalar lhs0(prec), lhs1(prec), t0(prec), t1(prec);
        mpfr_mul(t0.get(), h11.get(), v0.get(), MPFR_RNDN);
        mpfr_mul(t1.get(), h12.get(), v1.get(), MPFR_RNDN);
        mpfr_add(lhs0.get(), t0.get(), t1.get(), MPFR_RNDN);

        mpfr_mul(t0.get(), h21.get(), v0.get(), MPFR_RNDN);
        mpfr_mul(t1.get(), h22.get(), v1.get(), MPFR_RNDN);
        mpfr_add(lhs1.get(), t0.get(), t1.get(), MPFR_RNDN);

        /* rhs = [sqrt(d1_out)*x1_out; 0] */
        MpfrScalar rhs0(prec);
        mpfr_mul(rhs0.get(), sqrt_d1_out.get(), x1_out.get(), MPFR_RNDN);

        /* Error on first component */
        MpfrScalar diff(prec);
        mpfr_sub(diff.get(), lhs0.get(), rhs0.get(), MPFR_RNDN);
        mpfr_abs(diff.get(), diff.get(), MPFR_RNDN);
        mpfr_abs(rhs0.get(), rhs0.get(), MPFR_RNDN);

        double err0 = 0.0;
        if (!mpfr_zero_p(rhs0.get())) {
            MpfrScalar rel(prec);
            mpfr_div(rel.get(), diff.get(), rhs0.get(), MPFR_RNDN);
            err0 = mpfr_get_d(rel.get(), MPFR_RNDN);
        }

        /* Error on second component (should be 0) */
        double err1 = std::fabs(mpfr_get_d(lhs1.get(), MPFR_RNDN));

        err.max_relative = std::max(err0, err1);
        err.normwise_relative = std::max(err0, err1);
    }

    report_result("ROTMG", "", err, format);

    std::free(d1);
    std::free(d2);
    std::free(x1);
    std::free(y1);
    std::free(param_buf);
}
