/* spmv.cpp -- BLAS Level 2 SPMV accuracy tester */

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
extern "C" typedef void (*spmv_fn_t)(
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
/* Expand packed symmetric storage to full MpfrMatrix                  */
/* ------------------------------------------------------------------ */

static void expand_packed_sym_to_full(MpfrMatrix &full,
                                       const void *AP,
                                       int n, char uplo,
                                       const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(AP);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            mpfr_set_d(full.at(i, j), 0.0, MPFR_RNDN);

    if (uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                std::size_t idx = static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(j) * (j + 1) / 2;
                ctx.to_mpfr(full.at(i, j), p + idx * ctx.typesize);
                mpfr_set(full.at(j, i), full.at(i, j), MPFR_RNDN);
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                std::size_t idx = static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(j)
                                  * (2 * n - j - 1) / 2;
                ctx.to_mpfr(full.at(i, j), p + idx * ctx.typesize);
                mpfr_set(full.at(j, i), full.at(i, j), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* MPFR reference: y_ref = alpha * A * x + beta * y_in                */
/* ------------------------------------------------------------------ */

static void mpfr_symv_ref(MpfrMatrix &y_ref,
                            int n,
                            mpfr_t alpha,
                            const MpfrMatrix &A_full,
                            const MpfrMatrix &x,
                            mpfr_t beta,
                            const MpfrMatrix &y_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar acc(prec), tmp(prec);

    for (int i = 0; i < n; ++i) {
        mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j)
            mpfr_fma(acc.get(), A_full.at(i, j), x.at(j, 0),
                     acc.get(), MPFR_RNDN);
        mpfr_mul(acc.get(), alpha, acc.get(), MPFR_RNDN);
        mpfr_mul(tmp.get(), beta, y_in.at(i, 0), MPFR_RNDN);
        mpfr_add(y_ref.at(i, 0), acc.get(), tmp.get(), MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_spmv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<spmv_fn_t>(load_sym(lib, sym));

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

        void *AP    = gen_packed_symmetric_array(n, uplo,
                                                  ctx.typesize, ctx.from_mpfr, prec, &seed_AP);
        void *x     = gen_random_array(x_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
        void *y_in  = gen_random_array(y_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_y);
        void *alpha = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        unsigned sentinel_seed = 0xDEAD0001;
        void *y_out = alloc_with_sentinel(y_alloc, ctx.typesize, sentinel_seed);
        copy_vector_active(y_out, y_in, n, incy, ctx.typesize);

        fn(&uplo, &n, alpha, AP, x, &incx, beta, y_out, &incy,
           (std::size_t)1);

        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_full(n, n, prec);
        expand_packed_sym_to_full(A_full, AP, n, uplo, ctx);

        MpfrMatrix x_mpfr(n, 1, prec);
        MpfrMatrix y_in_mpfr(n, 1, prec);
        MpfrMatrix y_ref(n, 1, prec);

        custom_to_mpfr_vec(x_mpfr, x, incx, ctx);
        custom_to_mpfr_vec(y_in_mpfr, y_in, incy, ctx);

        mpfr_symv_ref(y_ref, n,
                       mpfr_alpha.get(), A_full, x_mpfr,
                       mpfr_beta.get(), y_in_mpfr);

        ErrorResult err = compute_error_vector(y_ref, y_out, incy, ctx);
        SentinelResult sr = check_vector_sentinels(y_out, n, incy, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d", uplo, n);
        report_result("SPMV", params_str, err, &sr, format);

        std::free(AP); std::free(x); std::free(y_in);
        std::free(y_out); std::free(alpha); std::free(beta);
    }
}
