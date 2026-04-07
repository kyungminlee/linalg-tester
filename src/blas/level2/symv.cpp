/* symv.cpp -- BLAS Level 2 SYMV accuracy tester */

#include "../level2.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer */
extern "C" typedef void (*symv_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *A,      const int  *lda,
    const void *x,      const int  *incx,
    const void *beta,
    void       *y,      const int  *incy,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: y_ref = alpha * A * x + beta * y_in                 */
/*   A is symmetric; expand from uplo triangle for the computation     */
/* ------------------------------------------------------------------ */

static void mpfr_symv_ref(MpfrMatrix &y_ref,
                           char uplo,
                           int n,
                           mpfr_t alpha,
                           const MpfrMatrix &A,
                           const MpfrMatrix &x,
                           mpfr_t beta,
                           const MpfrMatrix &y_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar acc(prec), tmp(prec);

    for (int i = 0; i < n; ++i) {
        mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j) {
            /* Access A symmetrically: use stored triangle */
            const mpfr_t &a_ij = (uplo == 'U')
                ? ((i <= j) ? A.at(i, j) : A.at(j, i))
                : ((i >= j) ? A.at(i, j) : A.at(j, i));
            mpfr_fma(acc.get(), a_ij, x.at(j, 0), acc.get(), MPFR_RNDN);
        }
        mpfr_mul(acc.get(), alpha, acc.get(), MPFR_RNDN);
        mpfr_mul(tmp.get(), beta, y_in.at(i, 0), MPFR_RNDN);
        mpfr_add(y_ref.at(i, 0), acc.get(), tmp.get(), MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_symv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<symv_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int incx = params.incx, incy = params.incy;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
        int lda = n + params.ld_pad;
        int abs_incx = (incx < 0) ? -incx : incx;
        int abs_incy = (incy < 0) ? -incy : incy;
        int x_alloc = 1 + (n - 1) * abs_incx;
        int y_alloc = 1 + (n - 1) * abs_incy;

        unsigned seed_A  = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_y  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        void *A     = gen_symmetric_array(n, uplo, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *x     = gen_random_array(x_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
        void *y_in  = gen_random_array(y_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_y);
        void *alpha = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        void *y_out = std::malloc(static_cast<std::size_t>(y_alloc) * ctx.typesize);
        std::memcpy(y_out, y_in, static_cast<std::size_t>(y_alloc) * ctx.typesize);

        fn(&uplo, &n, alpha, A, &lda, x, &incx, beta, y_out, &incy,
           (std::size_t)1);

        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_mpfr(n, n, prec);
        MpfrMatrix x_mpfr(n, 1, prec);
        MpfrMatrix y_in_mpfr(n, 1, prec);
        MpfrMatrix y_ref(n, 1, prec);

        custom_to_mpfr_mat(A_mpfr, A, lda, ctx);
        custom_to_mpfr_vec(x_mpfr, x, incx, ctx);
        custom_to_mpfr_vec(y_in_mpfr, y_in, incy, ctx);

        mpfr_symv_ref(y_ref, uplo, n,
                      mpfr_alpha.get(), A_mpfr, x_mpfr,
                      mpfr_beta.get(), y_in_mpfr);

        ErrorResult err = compute_error_vector(y_ref, y_out, incy, ctx);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d incx=%d incy=%d",
                      uplo, n, incx, incy);
        report_result("SYMV", params_str, err, format);

        std::free(A); std::free(x); std::free(y_in);
        std::free(y_out); std::free(alpha); std::free(beta);
    }
}
