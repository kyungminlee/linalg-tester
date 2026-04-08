/* gemv.cpp -- BLAS Level 2 GEMV accuracy tester */

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

/* Fortran-ABI function pointer (hidden char length appended) */
extern "C" typedef void (*gemv_fn_t)(
    const char *trans,
    const int  *m,      const int  *n,
    const void *alpha,
    const void *A,      const int  *lda,
    const void *x,      const int  *incx,
    const void *beta,
    void       *y,      const int  *incy,
    std::size_t trans_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: y_ref = alpha * op(A) * x + beta * y_in            */
/* ------------------------------------------------------------------ */

static void mpfr_gemv_ref(MpfrMatrix &y_ref,
                           char trans,
                           int m, int n,
                           mpfr_t alpha,
                           const MpfrMatrix &A,
                           const MpfrMatrix &x,
                           mpfr_t beta,
                           const MpfrMatrix &y_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    int op_rows = (trans == 'N') ? m : n;
    int op_cols = (trans == 'N') ? n : m;

    MpfrScalar acc(prec), tmp(prec);

    for (int i = 0; i < op_rows; ++i) {
        mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
        for (int j = 0; j < op_cols; ++j) {
            const mpfr_t &a_ij = (trans == 'N')
                ? A.at(i, j) : A.at(j, i);
            mpfr_fma(acc.get(), a_ij, x.at(j, 0), acc.get(), MPFR_RNDN);
        }
        /* y_ref[i] = alpha * acc + beta * y_in[i] */
        mpfr_mul(acc.get(), alpha, acc.get(), MPFR_RNDN);
        mpfr_mul(tmp.get(), beta, y_in.at(i, 0), MPFR_RNDN);
        mpfr_add(y_ref.at(i, 0), acc.get(), tmp.get(), MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_gemv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<gemv_fn_t>(load_sym(lib, sym));

    int m = params.m, n = params.n;
    int incx = params.incx, incy = params.incy;
    mpfr_prec_t prec = ctx.prec;

    for (char trans : {'N', 'T', 'C'}) {
        int xlen = (trans == 'N') ? n : m;
        int ylen = (trans == 'N') ? m : n;

        int lda = m + params.ld_pad;
        int abs_incx = (incx < 0) ? -incx : incx;
        int abs_incy = (incy < 0) ? -incy : incy;
        int x_alloc = 1 + (xlen - 1) * abs_incx;
        int y_alloc = 1 + (ylen - 1) * abs_incy;

        unsigned seed_A  = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_y  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        void *A     = gen_random_array(lda * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *x     = gen_random_array(x_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
        void *y_in  = gen_random_array(y_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_y);
        void *alpha = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        unsigned sentinel_seed = 0xDEAD0001;
        void *y_out = alloc_with_sentinel(y_alloc, ctx.typesize, sentinel_seed);
        copy_vector_active(y_out, y_in, ylen, incy, ctx.typesize);

        fn(&trans, &m, &n, alpha, A, &lda, x, &incx, beta, y_out, &incy,
           (std::size_t)1);

        /* Treat 'C' same as 'T' for real types */
        char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
            ? 'T' : std::toupper(static_cast<unsigned char>(trans));

        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_mpfr(m, n, prec);
        MpfrMatrix x_mpfr(xlen, 1, prec);
        MpfrMatrix y_in_mpfr(ylen, 1, prec);
        MpfrMatrix y_ref(ylen, 1, prec);

        custom_to_mpfr_mat(A_mpfr, A, lda, ctx);
        custom_to_mpfr_vec(x_mpfr, x, incx, ctx);
        custom_to_mpfr_vec(y_in_mpfr, y_in, incy, ctx);

        mpfr_gemv_ref(y_ref, trans_ref, m, n,
                      mpfr_alpha.get(), A_mpfr, x_mpfr,
                      mpfr_beta.get(), y_in_mpfr);

        ErrorResult err = compute_error_vector(y_ref, y_out, incy, ctx);
        SentinelResult sr = check_vector_sentinels(y_out, ylen, incy, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "trans=%c m=%d n=%d incx=%d incy=%d",
                      trans, m, n, incx, incy);
        report_result("GEMV", params_str, err, &sr, format);

        std::free(A); std::free(x); std::free(y_in);
        std::free(y_out); std::free(alpha); std::free(beta);
    }
}
