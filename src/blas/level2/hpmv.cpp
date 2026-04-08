/* hpmv.cpp -- BLAS Level 2 HPMV accuracy tester (complex-only) */

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

/* Fortran-ABI function pointer (no lda for packed) */
extern "C" typedef void (*hpmv_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *AP,
    const void *x,      const int  *incx,
    const void *beta,
    void       *y,      const int  *incy,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* Expand packed Hermitian storage to full MpfrComplexMatrix            */
/* ------------------------------------------------------------------ */

static void expand_packed_herm_to_full(MpfrComplexMatrix &full,
                                        const void *AP,
                                        int n, char uplo,
                                        const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(AP);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            mpfr_set_d(full.re(i, j), 0.0, MPFR_RNDN);
            mpfr_set_d(full.im(i, j), 0.0, MPFR_RNDN);
        }

    if (uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                std::size_t idx = static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(j) * (j + 1) / 2;
                ctx.to_mpfr_complex(full.re(i, j), full.im(i, j),
                                    p + idx * ctx.typesize);
                /* Mirror with conjugation */
                mpfr_complex_conj(full.re(j, i), full.im(j, i),
                                  full.re(i, j), full.im(i, j), MPFR_RNDN);
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                std::size_t idx = static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(j)
                                  * (2 * n - j - 1) / 2;
                ctx.to_mpfr_complex(full.re(i, j), full.im(i, j),
                                    p + idx * ctx.typesize);
                /* Mirror with conjugation */
                mpfr_complex_conj(full.re(j, i), full.im(j, i),
                                  full.re(i, j), full.im(i, j), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* MPFR reference: y_ref = alpha * A * x + beta * y_in                */
/* ------------------------------------------------------------------ */

static void mpfr_hemv_ref(MpfrComplexMatrix &y_ref,
                            int n,
                            const MpfrComplexScalar &alpha,
                            const MpfrComplexMatrix &A_full,
                            const MpfrComplexMatrix &x,
                            const MpfrComplexScalar &beta,
                            const MpfrComplexMatrix &y_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha.re());
    MpfrComplexScalar acc(prec), tmp(prec);

    for (int i = 0; i < n; ++i) {
        mpfr_set_d(acc.re(), 0.0, MPFR_RNDN);
        mpfr_set_d(acc.im(), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j)
            mpfr_complex_fma(acc.re(), acc.im(),
                             A_full.re(i, j), A_full.im(i, j),
                             x.re(j, 0), x.im(j, 0), MPFR_RNDN);
        mpfr_complex_mul(tmp.re(), tmp.im(),
                         alpha.re(), alpha.im(),
                         acc.re(), acc.im(), MPFR_RNDN);
        mpfr_complex_mul(acc.re(), acc.im(),
                         beta.re(), beta.im(),
                         y_in.re(i, 0), y_in.im(i, 0), MPFR_RNDN);
        mpfr_complex_add(y_ref.re(i, 0), y_ref.im(i, 0),
                         tmp.re(), tmp.im(),
                         acc.re(), acc.im(), MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_hpmv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "HPMV requires --complex\n");
        return;
    }

    auto *fn = reinterpret_cast<hpmv_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int incx = params.incx, incy = params.incy;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
        int abs_incx = (incx < 0) ? -incx : incx;
        int abs_incy = (incy < 0) ? -incy : incy;
        int x_alloc = 1 + (n - 1) * abs_incx;
        int y_alloc = 1 + (n - 1) * abs_incy;

        unsigned seed_AP = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_y  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        void *AP    = gen_packed_hermitian_array(n, uplo,
                                                  ctx.typesize, ctx.from_mpfr_complex, prec, &seed_AP);
        void *x     = gen_random_complex_array(x_alloc, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
        void *y_in  = gen_random_complex_array(y_alloc, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_y);
        void *alpha = gen_random_complex_array(1,       ctx.typesize, ctx.from_mpfr_complex, prec, &seed_ab);
        void *beta  = gen_random_complex_array(1,       ctx.typesize, ctx.from_mpfr_complex, prec, &seed_ab);

        unsigned sentinel_seed = 0xDEAD0001;
        void *y_out = alloc_with_sentinel(y_alloc, ctx.typesize, sentinel_seed);
        copy_vector_active(y_out, y_in, n, incy, ctx.typesize);

        fn(&uplo, &n, alpha, AP, x, &incx, beta, y_out, &incy,
           (std::size_t)1);

        MpfrComplexScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr_complex(mpfr_alpha.re(), mpfr_alpha.im(), alpha);
        ctx.to_mpfr_complex(mpfr_beta.re(),  mpfr_beta.im(),  beta);

        MpfrComplexMatrix A_full(n, n, prec);
        expand_packed_herm_to_full(A_full, AP, n, uplo, ctx);

        MpfrComplexMatrix x_mpfr(n, 1, prec);
        MpfrComplexMatrix y_in_mpfr(n, 1, prec);
        MpfrComplexMatrix y_ref(n, 1, prec);

        custom_to_mpfr_complex_vec(x_mpfr, x, incx, ctx);
        custom_to_mpfr_complex_vec(y_in_mpfr, y_in, incy, ctx);

        mpfr_hemv_ref(y_ref, n, mpfr_alpha, A_full, x_mpfr,
                       mpfr_beta, y_in_mpfr);

        ErrorResult err = compute_error_complex_vector(y_ref, y_out, incy, ctx);
        SentinelResult sr = check_vector_sentinels(y_out, n, incy, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d", uplo, n);
        report_result("HPMV", params_str, err, &sr, format);

        std::free(AP); std::free(x); std::free(y_in);
        std::free(y_out); std::free(alpha); std::free(beta);
    }
}
