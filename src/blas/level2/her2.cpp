/* her2.cpp -- BLAS Level 2 HER2 accuracy tester (complex-only) */

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
extern "C" typedef void (*her2_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *x,      const int  *incx,
    const void *y,      const int  *incy,
    void       *A,      const int  *lda,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference:                                                     */
/*   A_ref[i,j] = alpha*x[i]*conj(y[j]) + conj(alpha)*y[i]*conj(x[j])*/
/*              + A_in[i,j]                                            */
/*   Only the uplo triangle is updated. alpha is COMPLEX.              */
/* ------------------------------------------------------------------ */

static void mpfr_her2_ref(MpfrComplexMatrix &A_ref,
                            char uplo,
                            int n,
                            const MpfrComplexScalar &alpha,
                            const MpfrComplexMatrix &x,
                            const MpfrComplexMatrix &y,
                            const MpfrComplexMatrix &A_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha.re());
    MpfrComplexScalar t1(prec), t2(prec), sum(prec);
    MpfrComplexScalar alpha_conj(prec), yc(prec), xc(prec);

    /* conj(alpha) */
    mpfr_complex_conj(alpha_conj.re(), alpha_conj.im(),
                      alpha.re(), alpha.im(), MPFR_RNDN);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            bool in_triangle = (uplo == 'U') ? (i <= j) : (i >= j);
            if (in_triangle) {
                /* t1 = alpha * x[i] * conj(y[j]) */
                mpfr_complex_conj(yc.re(), yc.im(),
                                  y.re(j, 0), y.im(j, 0), MPFR_RNDN);
                mpfr_complex_mul(t1.re(), t1.im(),
                                 x.re(i, 0), x.im(i, 0),
                                 yc.re(), yc.im(), MPFR_RNDN);
                mpfr_complex_mul(t1.re(), t1.im(),
                                 alpha.re(), alpha.im(),
                                 t1.re(), t1.im(), MPFR_RNDN);

                /* t2 = conj(alpha) * y[i] * conj(x[j]) */
                mpfr_complex_conj(xc.re(), xc.im(),
                                  x.re(j, 0), x.im(j, 0), MPFR_RNDN);
                mpfr_complex_mul(t2.re(), t2.im(),
                                 y.re(i, 0), y.im(i, 0),
                                 xc.re(), xc.im(), MPFR_RNDN);
                mpfr_complex_mul(t2.re(), t2.im(),
                                 alpha_conj.re(), alpha_conj.im(),
                                 t2.re(), t2.im(), MPFR_RNDN);

                /* sum = t1 + t2 */
                mpfr_complex_add(sum.re(), sum.im(),
                                 t1.re(), t1.im(),
                                 t2.re(), t2.im(), MPFR_RNDN);

                /* A_ref[i,j] = sum + A_in[i,j] */
                mpfr_complex_add(A_ref.re(i, j), A_ref.im(i, j),
                                 sum.re(), sum.im(),
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

void test_her2(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "HER2 requires --complex\n");
        return;
    }

    auto *fn = reinterpret_cast<her2_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int incx = params.incx, incy = params.incy;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
        int lda = n + params.ld_pad;
        int abs_incx = (incx < 0) ? -incx : incx;
        int abs_incy = (incy < 0) ? -incy : incy;
        int x_alloc = 1 + (n - 1) * abs_incx;
        int y_alloc = 1 + (n - 1) * abs_incy;

        unsigned seed_x  = params.seed;
        unsigned seed_y  = params.seed + 1;
        unsigned seed_A  = params.seed + 2;
        unsigned seed_al = params.seed + 3;

        void *x     = gen_random_complex_array(x_alloc, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
        void *y     = gen_random_complex_array(y_alloc, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_y);
        void *A_in  = gen_hermitian_array(n, uplo, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *alpha = gen_random_complex_array(1, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_al);

        unsigned sentinel_seed = 0xDEAD0001;
        void *A_out = alloc_with_sentinel(lda * n, ctx.typesize, sentinel_seed);
        copy_matrix_active(A_out, A_in, n, n, lda, ctx.typesize);

        fn(&uplo, &n, alpha, x, &incx, y, &incy, A_out, &lda,
           (std::size_t)1);

        MpfrComplexScalar mpfr_alpha(prec);
        ctx.to_mpfr_complex(mpfr_alpha.re(), mpfr_alpha.im(), alpha);

        MpfrComplexMatrix x_mpfr(n, 1, prec);
        MpfrComplexMatrix y_mpfr(n, 1, prec);
        MpfrComplexMatrix A_in_mpfr(n, n, prec);
        MpfrComplexMatrix A_ref(n, n, prec);

        custom_to_mpfr_complex_vec(x_mpfr, x, incx, ctx);
        custom_to_mpfr_complex_vec(y_mpfr, y, incy, ctx);
        custom_to_mpfr_complex_mat(A_in_mpfr, A_in, lda, ctx);

        mpfr_her2_ref(A_ref, uplo, n, mpfr_alpha, x_mpfr, y_mpfr, A_in_mpfr);

        ErrorResult err = compute_error_complex_matrix_triangle(A_ref, A_out, lda, uplo, ctx);
        SentinelResult sr = check_matrix_sentinels(A_out, n, n, lda, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d incx=%d incy=%d",
                      uplo, n, incx, incy);
        report_result("HER2", params_str, err, &sr, format);

        std::free(x); std::free(y); std::free(A_in);
        std::free(A_out); std::free(alpha);
    }
}
