/* syr.cpp -- BLAS Level 2 SYR accuracy tester */

#include "../level2.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer */
extern "C" typedef void (*syr_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *x,      const int  *incx,
    void       *A,      const int  *lda,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: A_ref[i,j] = alpha * x[i] * x[j] + A_in[i,j]      */
/*   Only the uplo triangle is updated.                                */
/* ------------------------------------------------------------------ */

static void mpfr_syr_ref(MpfrMatrix &A_ref,
                          char uplo,
                          int n,
                          mpfr_t alpha,
                          const MpfrMatrix &x,
                          const MpfrMatrix &A_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar tmp(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            bool in_triangle = (uplo == 'U') ? (i <= j) : (i >= j);
            if (in_triangle) {
                mpfr_mul(tmp.get(), x.at(i, 0), x.at(j, 0), MPFR_RNDN);
                mpfr_fma(A_ref.at(i, j), alpha, tmp.get(),
                         A_in.at(i, j), MPFR_RNDN);
            } else {
                /* Copy original value unchanged */
                mpfr_set(A_ref.at(i, j), A_in.at(i, j), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_syr(const TesterCtx &ctx, void *lib, const char *sym,
              const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<syr_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
        int lda = n + params.ld_pad;
        int abs_incx = (incx < 0) ? -incx : incx;
        int x_alloc = 1 + (n - 1) * abs_incx;

        unsigned seed_x  = params.seed;
        unsigned seed_A  = params.seed + 1;
        unsigned seed_al = params.seed + 2;

        void *x     = gen_random_array(x_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
        void *A_in  = gen_random_array(lda * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *alpha = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_al);

        unsigned sentinel_seed = 0xDEAD0001;
        void *A_out = alloc_with_sentinel(lda * n, ctx.typesize, sentinel_seed);
        copy_matrix_active(A_out, A_in, n, n, lda, ctx.typesize);

        fn(&uplo, &n, alpha, x, &incx, A_out, &lda,
           (std::size_t)1);

        MpfrScalar mpfr_alpha(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);

        MpfrMatrix x_mpfr(n, 1, prec);
        MpfrMatrix A_in_mpfr(n, n, prec);
        MpfrMatrix A_ref(n, n, prec);

        custom_to_mpfr_vec(x_mpfr, x, incx, ctx);
        custom_to_mpfr_mat(A_in_mpfr, A_in, lda, ctx);

        mpfr_syr_ref(A_ref, uplo, n, mpfr_alpha.get(), x_mpfr, A_in_mpfr);

        ErrorResult err = compute_error_matrix_triangle(A_ref, A_out, lda, uplo, ctx);
        SentinelResult sr = check_matrix_sentinels(A_out, n, n, lda, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d incx=%d",
                      uplo, n, incx);
        report_result("SYR", params_str, err, &sr, format);

        std::free(x); std::free(A_in); std::free(A_out);
        std::free(alpha);
    }
}
