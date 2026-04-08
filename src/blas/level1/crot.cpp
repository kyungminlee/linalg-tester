/* crot.cpp -- BLAS Level 1 CROT/CSROT/ZDROT accuracy tester (complex-only) */

#include "../level1.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* CSROT/ZDROT: Apply real rotation to complex vectors.
   x_out(i) = c*x(i) + s*y(i)
   y_out(i) = -s*x(i) + c*y(i)
   where c, s are REAL scalars and x, y are complex vectors. */

extern "C" typedef void (*crot_fn_t)(
    const int *n,
    void *x, const int *incx,
    void *y, const int *incy,
    const void *c, const void *s
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_crot(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "CROT requires --complex\n");
        return;
    }

    auto *fn = reinterpret_cast<crot_fn_t>(load_sym(lib, sym));
    mpfr_prec_t prec = ctx.prec;

    int n    = params.n;
    int incx = params.incx;
    int incy = params.incy;
    int abs_incx = (incx < 0) ? -incx : incx;
    int abs_incy = (incy < 0) ? -incy : incy;
    int alloc_x = 1 + (n - 1) * abs_incx;
    int alloc_y = 1 + (n - 1) * abs_incy;

    std::size_t real_typesize = ctx.typesize / 2;

    unsigned seed_x = params.seed;
    unsigned seed_y = params.seed + 1;
    unsigned seed_cs = params.seed + 2;

    void *x_in = gen_random_complex_array(alloc_x, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
    void *y_in = gen_random_complex_array(alloc_y, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_y);
    void *c_val = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_cs);
    void *s_val = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_cs);

    /* Convert inputs to MPFR before the call */
    MpfrComplexMatrix x_mpfr(n, 1, prec), y_mpfr(n, 1, prec);
    custom_to_mpfr_complex_vec(x_mpfr, x_in, incx, ctx);
    custom_to_mpfr_complex_vec(y_mpfr, y_in, incy, ctx);

    MpfrScalar c_mpfr(prec), s_mpfr(prec);
    ctx.to_mpfr(c_mpfr.get(), c_val);
    ctx.to_mpfr(s_mpfr.get(), s_val);

    /* Copy inputs for the call (routine modifies in-place) */
    unsigned sentinel_seed_x = 0xDEAD0001;
    unsigned sentinel_seed_y = 0xDEAD0002;
    void *x_out = alloc_with_sentinel(alloc_x, ctx.typesize, sentinel_seed_x);
    void *y_out = alloc_with_sentinel(alloc_y, ctx.typesize, sentinel_seed_y);
    copy_vector_active(x_out, x_in, n, incx, ctx.typesize);
    copy_vector_active(y_out, y_in, n, incy, ctx.typesize);

    fn(&n, x_out, &incx, y_out, &incy, c_val, s_val);

    /* MPFR reference:
       x_ref[i] = c*x[i] + s*y[i]
       y_ref[i] = -s*x[i] + c*y[i] */
    MpfrComplexMatrix x_ref(n, 1, prec), y_ref(n, 1, prec);
    {
        MpfrComplexScalar t0(prec), t1(prec);
        MpfrScalar neg_s(prec);
        mpfr_neg(neg_s.get(), s_mpfr.get(), MPFR_RNDN);
        for (int i = 0; i < n; ++i) {
            /* x_ref[i] = c * x[i] + s * y[i] */
            mpfr_complex_mul_real(t0.re(), t0.im(),
                                  x_mpfr.re(i, 0), x_mpfr.im(i, 0),
                                  c_mpfr.get(), MPFR_RNDN);
            mpfr_complex_mul_real(t1.re(), t1.im(),
                                  y_mpfr.re(i, 0), y_mpfr.im(i, 0),
                                  s_mpfr.get(), MPFR_RNDN);
            mpfr_complex_add(x_ref.re(i, 0), x_ref.im(i, 0),
                             t0.re(), t0.im(), t1.re(), t1.im(), MPFR_RNDN);

            /* y_ref[i] = -s * x[i] + c * y[i] */
            mpfr_complex_mul_real(t0.re(), t0.im(),
                                  x_mpfr.re(i, 0), x_mpfr.im(i, 0),
                                  neg_s.get(), MPFR_RNDN);
            mpfr_complex_mul_real(t1.re(), t1.im(),
                                  y_mpfr.re(i, 0), y_mpfr.im(i, 0),
                                  c_mpfr.get(), MPFR_RNDN);
            mpfr_complex_add(y_ref.re(i, 0), y_ref.im(i, 0),
                             t0.re(), t0.im(), t1.re(), t1.im(), MPFR_RNDN);
        }
    }

    ErrorResult err_x = compute_error_complex_vector(x_ref, x_out, incx, ctx);
    ErrorResult err_y = compute_error_complex_vector(y_ref, y_out, incy, ctx);

    ErrorResult err;
    err.max_relative = std::max(err_x.max_relative, err_y.max_relative);
    err.normwise_relative = std::max(err_x.normwise_relative, err_y.normwise_relative);
    err.max_absolute_at_zero = std::max(err_x.max_absolute_at_zero, err_y.max_absolute_at_zero);
    err.nan_inf_mismatches = err_x.nan_inf_mismatches + err_y.nan_inf_mismatches;

    SentinelResult sr_x = check_vector_sentinels(x_out, n, incx, ctx.typesize, sentinel_seed_x);
    SentinelResult sr_y = check_vector_sentinels(y_out, n, incy, ctx.typesize, sentinel_seed_y);
    SentinelResult sr;
    sr.passed = sr_x.passed && sr_y.passed;
    sr.corrupted_count = sr_x.corrupted_count + sr_y.corrupted_count;
    sr.first_offset = sr_x.passed ? sr_y.first_offset : sr_x.first_offset;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "n=%d incx=%d incy=%d", n, incx, incy);
    report_result("CROT", params_str, err, &sr, format);

    std::free(x_in);
    std::free(y_in);
    std::free(x_out);
    std::free(y_out);
    std::free(c_val);
    std::free(s_val);
}
