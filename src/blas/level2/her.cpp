/* her.cpp -- BLAS Level 2 HER accuracy tester (complex-only) */

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
extern "C" typedef void (*her_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *x,      const int  *incx,
    void       *A,      const int  *lda,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: A_ref[i,j] = alpha * x[i] * conj(x[j]) + A_in[i,j]*/
/*   alpha is REAL. Only the uplo triangle is updated.                 */
/* ------------------------------------------------------------------ */

static void mpfr_her_ref(MpfrComplexMatrix &A_ref,
                           char uplo,
                           int n,
                           mpfr_t alpha,
                           const MpfrComplexMatrix &x,
                           const MpfrComplexMatrix &A_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrComplexScalar xixjc(prec), tmp(prec), xjc(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            bool in_triangle = (uplo == 'U') ? (i <= j) : (i >= j);
            if (in_triangle) {
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
                /* A_ref[i,j] = tmp + A_in[i,j] */
                mpfr_complex_add(A_ref.re(i, j), A_ref.im(i, j),
                                 tmp.re(), tmp.im(),
                                 A_in.re(i, j), A_in.im(i, j), MPFR_RNDN);
            } else {
                /* Copy original value unchanged */
                mpfr_set(A_ref.re(i, j), A_in.re(i, j), MPFR_RNDN);
                mpfr_set(A_ref.im(i, j), A_in.im(i, j), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_her(const TesterCtx &ctx, void *lib, const char *sym,
              const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "HER requires --complex\n");
        return;
    }

    auto *fn = reinterpret_cast<her_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = ctx.prec;

    /* alpha is REAL for HER */
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
        int lda = n + params.ld_pad;
        int abs_incx = (incx < 0) ? -incx : incx;
        int x_alloc = 1 + (n - 1) * abs_incx;

        unsigned seed_x  = params.seed;
        unsigned seed_A  = params.seed + 1;
        unsigned seed_al = params.seed + 2;

        void *x     = gen_random_complex_array(x_alloc, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
        void *A_in  = gen_hermitian_array(n, uplo, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *alpha = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_al);

        unsigned sentinel_seed = 0xDEAD0001;
        void *A_out = alloc_with_sentinel(lda * n, ctx.typesize, sentinel_seed);
        copy_matrix_active(A_out, A_in, n, n, lda, ctx.typesize);

        fn(&uplo, &n, alpha, x, &incx, A_out, &lda,
           (std::size_t)1);

        MpfrScalar mpfr_alpha(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);

        MpfrComplexMatrix x_mpfr(n, 1, prec);
        MpfrComplexMatrix A_in_mpfr(n, n, prec);
        MpfrComplexMatrix A_ref(n, n, prec);

        custom_to_mpfr_complex_vec(x_mpfr, x, incx, ctx);
        custom_to_mpfr_complex_mat(A_in_mpfr, A_in, lda, ctx);

        mpfr_her_ref(A_ref, uplo, n, mpfr_alpha.get(), x_mpfr, A_in_mpfr);

        ErrorResult err = compute_error_complex_matrix_triangle(A_ref, A_out, lda, uplo, ctx);
        SentinelResult sr = check_matrix_sentinels(A_out, n, n, lda, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d incx=%d",
                      uplo, n, incx);
        report_result("HER", params_str, err, &sr, format);

        std::free(x); std::free(A_in); std::free(A_out);
        std::free(alpha);
    }
}
