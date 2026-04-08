/* spr2.cpp -- BLAS Level 2 SPR2 accuracy tester */

#include "../level2.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer */
extern "C" typedef void (*spr2_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *x,      const int  *incx,
    const void *y,      const int  *incy,
    void       *AP,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference:                                                     */
/*   AP_ref[idx] = alpha*(x[i]*y[j] + y[i]*x[j]) + AP_in[idx]       */
/* ------------------------------------------------------------------ */

static void mpfr_spr2_ref(MpfrMatrix &ap_ref,
                            int n, char uplo,
                            mpfr_t alpha,
                            const MpfrMatrix &x,
                            const MpfrMatrix &y,
                            const MpfrMatrix &ap_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar tmp1(prec), tmp2(prec), sum(prec);
    int pos = 0;

    if (uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                /* tmp1 = x[i]*y[j], tmp2 = y[i]*x[j] */
                mpfr_mul(tmp1.get(), x.at(i, 0), y.at(j, 0), MPFR_RNDN);
                mpfr_mul(tmp2.get(), y.at(i, 0), x.at(j, 0), MPFR_RNDN);
                mpfr_add(sum.get(), tmp1.get(), tmp2.get(), MPFR_RNDN);
                mpfr_mul(sum.get(), alpha, sum.get(), MPFR_RNDN);
                mpfr_add(ap_ref.at(pos, 0), sum.get(), ap_in.at(pos, 0), MPFR_RNDN);
                ++pos;
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                mpfr_mul(tmp1.get(), x.at(i, 0), y.at(j, 0), MPFR_RNDN);
                mpfr_mul(tmp2.get(), y.at(i, 0), x.at(j, 0), MPFR_RNDN);
                mpfr_add(sum.get(), tmp1.get(), tmp2.get(), MPFR_RNDN);
                mpfr_mul(sum.get(), alpha, sum.get(), MPFR_RNDN);
                mpfr_add(ap_ref.at(pos, 0), sum.get(), ap_in.at(pos, 0), MPFR_RNDN);
                ++pos;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_spr2(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<spr2_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int incx = params.incx, incy = params.incy;
    mpfr_prec_t prec = ctx.prec;

    int packed_len = n * (n + 1) / 2;

    for (char uplo : {'U', 'L'}) {
        int abs_incx = (incx < 0) ? -incx : incx;
        int abs_incy = (incy < 0) ? -incy : incy;
        int x_alloc = 1 + (n - 1) * abs_incx;
        int y_alloc = 1 + (n - 1) * abs_incy;

        unsigned seed_AP = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_y  = params.seed + 2;
        unsigned seed_al = params.seed + 3;

        void *AP_in = gen_packed_symmetric_array(n, uplo,
                                                  ctx.typesize, ctx.from_mpfr, prec, &seed_AP);
        void *x     = gen_random_array(x_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
        void *y     = gen_random_array(y_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_y);
        void *alpha = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_al);

        void *AP_out = std::malloc(static_cast<std::size_t>(packed_len) * ctx.typesize);
        std::memcpy(AP_out, AP_in, static_cast<std::size_t>(packed_len) * ctx.typesize);

        fn(&uplo, &n, alpha, x, &incx, y, &incy, AP_out,
           (std::size_t)1);

        MpfrScalar mpfr_alpha(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);

        MpfrMatrix x_mpfr(n, 1, prec);
        MpfrMatrix y_mpfr(n, 1, prec);
        custom_to_mpfr_vec(x_mpfr, x, incx, ctx);
        custom_to_mpfr_vec(y_mpfr, y, incy, ctx);

        /* Read AP_in as a flat vector */
        MpfrMatrix ap_in_mpfr(packed_len, 1, prec);
        {
            const char *p = static_cast<const char *>(AP_in);
            for (int i = 0; i < packed_len; ++i)
                ctx.to_mpfr(ap_in_mpfr.at(i, 0),
                            p + static_cast<std::size_t>(i) * ctx.typesize);
        }

        MpfrMatrix ap_ref(packed_len, 1, prec);
        mpfr_spr2_ref(ap_ref, n, uplo,
                        mpfr_alpha.get(), x_mpfr, y_mpfr, ap_in_mpfr);

        int inc_one = 1;
        ErrorResult err = compute_error_vector(ap_ref, AP_out, inc_one, ctx);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d", uplo, n);
        report_result("SPR2", params_str, err, nullptr, format);

        std::free(AP_in); std::free(AP_out);
        std::free(x); std::free(y); std::free(alpha);
    }
}
