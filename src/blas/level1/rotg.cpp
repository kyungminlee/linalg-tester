/* rotg.cpp -- BLAS Level 1 ROTG accuracy tester */

#include "../level1.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran ABI: void rotg(a, b, c, s) -- all scalar pointers, no hidden lengths */
extern "C" typedef void (*rotg_fn_t)(
    void *a, void *b, void *c, void *s
);

/* ------------------------------------------------------------------ */
/* MPFR reference implementation of ROTG                               */
/* ------------------------------------------------------------------ */

static void mpfr_rotg_ref(mpfr_t r, mpfr_t z, mpfr_t c, mpfr_t s,
                           const mpfr_t a_in, const mpfr_t b_in,
                           mpfr_prec_t prec)
{
    MpfrScalar abs_a(prec), abs_b(prec), sigma(prec), denom(prec);
    MpfrScalar a2(prec), b2(prec), sum2(prec);

    mpfr_abs(abs_a.get(), a_in, MPFR_RNDN);
    mpfr_abs(abs_b.get(), b_in, MPFR_RNDN);

    /* r = sigma * sqrt(a^2 + b^2) */
    mpfr_mul(a2.get(), a_in, a_in, MPFR_RNDN);
    mpfr_mul(b2.get(), b_in, b_in, MPFR_RNDN);
    mpfr_add(sum2.get(), a2.get(), b2.get(), MPFR_RNDN);
    mpfr_sqrt(r, sum2.get(), MPFR_RNDN);

    /* sigma = sgn(a) if |a| > |b|, else sgn(b) */
    if (mpfr_cmp(abs_a.get(), abs_b.get()) > 0) {
        if (mpfr_sgn(a_in) >= 0)
            mpfr_set_d(sigma.get(), 1.0, MPFR_RNDN);
        else
            mpfr_set_d(sigma.get(), -1.0, MPFR_RNDN);
    } else {
        if (mpfr_sgn(b_in) >= 0)
            mpfr_set_d(sigma.get(), 1.0, MPFR_RNDN);
        else
            mpfr_set_d(sigma.get(), -1.0, MPFR_RNDN);
    }
    mpfr_mul(r, r, sigma.get(), MPFR_RNDN);

    /* Handle r == 0 */
    if (mpfr_zero_p(r)) {
        mpfr_set_d(c, 1.0, MPFR_RNDN);
        mpfr_set_d(s, 0.0, MPFR_RNDN);
        mpfr_set_d(z, 0.0, MPFR_RNDN);
        return;
    }

    /* c = a / r, s = b / r */
    mpfr_div(c, a_in, r, MPFR_RNDN);
    mpfr_div(s, b_in, r, MPFR_RNDN);

    /* z encoding */
    if (mpfr_cmp(abs_a.get(), abs_b.get()) > 0) {
        mpfr_set(z, s, MPFR_RNDN);
    } else if (!mpfr_zero_p(c)) {
        MpfrScalar one(prec);
        mpfr_set_d(one.get(), 1.0, MPFR_RNDN);
        mpfr_div(z, one.get(), c, MPFR_RNDN);
    } else {
        mpfr_set_d(z, 1.0, MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_rotg(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<rotg_fn_t>(load_sym(lib, sym));
    mpfr_prec_t prec = ctx.prec;

    unsigned seed_a = params.seed;
    unsigned seed_b = params.seed + 1;

    void *a_val = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_a);
    void *b_val = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_b);

    /* Save input copies for MPFR reference */
    MpfrScalar a_in(prec), b_in(prec);
    ctx.to_mpfr(a_in.get(), a_val);
    ctx.to_mpfr(b_in.get(), b_val);

    /* Allocate c, s outputs */
    void *c_val = std::calloc(1, ctx.typesize);
    void *s_val = std::calloc(1, ctx.typesize);

    /* Call the routine: a and b are overwritten with r and z */
    fn(a_val, b_val, c_val, s_val);

    /* MPFR reference */
    MpfrScalar r_ref(prec), z_ref(prec), c_ref(prec), s_ref(prec);
    mpfr_rotg_ref(r_ref.get(), z_ref.get(), c_ref.get(), s_ref.get(),
                  a_in.get(), b_in.get(), prec);

    /* Compare all 4 outputs */
    ErrorResult err_r = compute_error_scalar(r_ref, a_val, ctx);
    ErrorResult err_z = compute_error_scalar(z_ref, b_val, ctx);
    ErrorResult err_c = compute_error_scalar(c_ref, c_val, ctx);
    ErrorResult err_s = compute_error_scalar(s_ref, s_val, ctx);

    ErrorResult err;
    err.max_relative = std::max({err_r.max_relative, err_z.max_relative,
                                  err_c.max_relative, err_s.max_relative});
    err.normwise_relative = std::max({err_r.normwise_relative, err_z.normwise_relative,
                                       err_c.normwise_relative, err_s.normwise_relative});
    err.max_absolute_at_zero = std::max({err_r.max_absolute_at_zero, err_z.max_absolute_at_zero,
                                          err_c.max_absolute_at_zero, err_s.max_absolute_at_zero});
    err.nan_inf_mismatches = err_r.nan_inf_mismatches + err_z.nan_inf_mismatches +
                             err_c.nan_inf_mismatches + err_s.nan_inf_mismatches;

    report_result("ROTG", "", err, format);

    std::free(a_val);
    std::free(b_val);
    std::free(c_val);
    std::free(s_val);
}
