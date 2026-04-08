/* hpr.cpp -- BLAS Level 2 HPR accuracy tester (complex-only) */

#include "../level2.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
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
extern "C" typedef void (*hpr_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *x,      const int  *incx,
    void       *AP,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: AP_ref[idx] = alpha * x[i] * conj(x[j]) + AP_in   */
/*   alpha is REAL. Packed Hermitian storage.                          */
/* ------------------------------------------------------------------ */

static void mpfr_hpr_ref(MpfrComplexMatrix &ap_ref,
                           int n, char uplo,
                           mpfr_t alpha,
                           const MpfrComplexMatrix &x,
                           const MpfrComplexMatrix &ap_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrComplexScalar xixjc(prec), tmp(prec), xjc(prec);
    int pos = 0;

    if (uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                /* xjc = conj(x[j]) */
                mpfr_complex_conj(xjc.re(), xjc.im(),
                                  x.re(j, 0), x.im(j, 0), MPFR_RNDN);
                /* xixjc = x[i] * conj(x[j]) */
                mpfr_complex_mul(xixjc.re(), xixjc.im(),
                                 x.re(i, 0), x.im(i, 0),
                                 xjc.re(), xjc.im(), MPFR_RNDN);
                /* tmp = alpha * xixjc (alpha is real) */
                mpfr_complex_mul_real(tmp.re(), tmp.im(),
                                      xixjc.re(), xixjc.im(),
                                      alpha, MPFR_RNDN);
                /* ap_ref[pos] = tmp + ap_in[pos] */
                mpfr_add(ap_ref.re(pos, 0), tmp.re(), ap_in.re(pos, 0), MPFR_RNDN);
                mpfr_add(ap_ref.im(pos, 0), tmp.im(), ap_in.im(pos, 0), MPFR_RNDN);
                ++pos;
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                mpfr_complex_conj(xjc.re(), xjc.im(),
                                  x.re(j, 0), x.im(j, 0), MPFR_RNDN);
                mpfr_complex_mul(xixjc.re(), xixjc.im(),
                                 x.re(i, 0), x.im(i, 0),
                                 xjc.re(), xjc.im(), MPFR_RNDN);
                mpfr_complex_mul_real(tmp.re(), tmp.im(),
                                      xixjc.re(), xixjc.im(),
                                      alpha, MPFR_RNDN);
                mpfr_add(ap_ref.re(pos, 0), tmp.re(), ap_in.re(pos, 0), MPFR_RNDN);
                mpfr_add(ap_ref.im(pos, 0), tmp.im(), ap_in.im(pos, 0), MPFR_RNDN);
                ++pos;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_hpr(const TesterCtx &ctx, void *lib, const char *sym,
              const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "HPR requires --complex\n");
        return;
    }

    auto *fn = reinterpret_cast<hpr_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = ctx.prec;

    int packed_len = n * (n + 1) / 2;

    /* alpha is REAL for HPR */
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
        int abs_incx = (incx < 0) ? -incx : incx;
        int x_alloc = 1 + (n - 1) * abs_incx;

        unsigned seed_AP = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_al = params.seed + 2;

        void *AP_in = gen_packed_hermitian_array(n, uplo,
                                                  ctx.typesize, ctx.from_mpfr_complex, prec, &seed_AP);
        void *x     = gen_random_complex_array(x_alloc, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
        void *alpha = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_al);

        void *AP_out = std::malloc(static_cast<std::size_t>(packed_len) * ctx.typesize);
        std::memcpy(AP_out, AP_in, static_cast<std::size_t>(packed_len) * ctx.typesize);

        fn(&uplo, &n, alpha, x, &incx, AP_out,
           (std::size_t)1);

        MpfrScalar mpfr_alpha(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);

        MpfrComplexMatrix x_mpfr(n, 1, prec);
        custom_to_mpfr_complex_vec(x_mpfr, x, incx, ctx);

        /* Read AP_in as a flat complex vector of length packed_len */
        MpfrComplexMatrix ap_in_mpfr(packed_len, 1, prec);
        {
            const char *p = static_cast<const char *>(AP_in);
            for (int i = 0; i < packed_len; ++i)
                ctx.to_mpfr_complex(ap_in_mpfr.re(i, 0), ap_in_mpfr.im(i, 0),
                                    p + static_cast<std::size_t>(i) * ctx.typesize);
        }

        MpfrComplexMatrix ap_ref(packed_len, 1, prec);
        mpfr_hpr_ref(ap_ref, n, uplo,
                      mpfr_alpha.get(), x_mpfr, ap_in_mpfr);

        /* Compare packed output as a complex vector with inc=1 */
        int inc_one = 1;
        ErrorResult err = compute_error_complex_vector(ap_ref, AP_out, inc_one, ctx);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d", uplo, n);
        report_result("HPR", params_str, err, nullptr, format);

        std::free(AP_in); std::free(AP_out);
        std::free(x); std::free(alpha);
    }
}
