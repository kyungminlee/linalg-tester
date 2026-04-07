/* syrk.cpp -- BLAS Level 3 SYRK accuracy tester */

#include "../level3.h"
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
extern "C" typedef void (*syrk_fn_t)(
    const char *uplo,  const char *trans,
    const int  *n,     const int  *k,
    const void *alpha,
    const void *A,     const int  *lda,
    const void *beta,
    void       *C,     const int  *ldc,
    std::size_t uplo_len, std::size_t trans_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference:                                                      */
/*   trans='N': C = alpha * A * A^T + beta * C,  A is n-by-k           */
/*   trans='T'/'C': C = alpha * A^T * A + beta * C,  A is k-by-n      */
/* Only the uplo triangle of C is meaningful.                          */
/* ------------------------------------------------------------------ */

static void mpfr_syrk_ref(MpfrMatrix &C_ref,
                           char uplo, char trans,
                           int n, int k,
                           mpfr_t alpha,
                           const MpfrMatrix &A,
                           mpfr_t beta,
                           const MpfrMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar acc(prec), alpha_acc(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);

            for (int p = 0; p < k; ++p) {
                if (trans == 'N') {
                    /* A(i,p) * A(j,p) */
                    mpfr_fma(acc.get(), A.at(i, p), A.at(j, p),
                             acc.get(), MPFR_RNDN);
                } else {
                    /* A(p,i) * A(p,j) */
                    mpfr_fma(acc.get(), A.at(p, i), A.at(p, j),
                             acc.get(), MPFR_RNDN);
                }
            }

            mpfr_mul(alpha_acc.get(), alpha, acc.get(), MPFR_RNDN);
            mpfr_mul(beta_c.get(),    beta,  C_in.at(i, j), MPFR_RNDN);
            mpfr_add(C_ref.at(i, j), alpha_acc.get(), beta_c.get(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_syrk(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<syrk_fn_t>(load_sym(lib, sym));

    int n = params.n, k = params.k;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T', 'C'}) {
        char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
                         ? 'T' : std::toupper(static_cast<unsigned char>(trans));

        int rows_A = (trans_ref == 'N') ? n : k;
        int cols_A = (trans_ref == 'N') ? k : n;

        int lda = rows_A + params.ld_pad;
        int ldc = n      + params.ld_pad;

        unsigned seed_A  = params.seed;
        unsigned seed_C  = params.seed + 1;
        unsigned seed_ab = params.seed + 2;

        void *A     = gen_random_array(lda * cols_A, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *C_in  = gen_random_array(ldc * n,      ctx.typesize, ctx.from_mpfr, prec, &seed_C);
        void *alpha = gen_random_array(1,            ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1,            ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        void *C_out = std::malloc(static_cast<std::size_t>(ldc) * n * ctx.typesize);
        std::memcpy(C_out, C_in, static_cast<std::size_t>(ldc) * n * ctx.typesize);

        fn(&uplo, &trans, &n, &k,
           alpha, A, &lda, beta, C_out, &ldc,
           (std::size_t)1, (std::size_t)1);

        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_mpfr(rows_A, cols_A, prec);
        MpfrMatrix C_in_mpfr(n, n, prec);
        MpfrMatrix C_ref(n, n, prec);

        custom_to_mpfr_mat(A_mpfr,    A,    lda, ctx);
        custom_to_mpfr_mat(C_in_mpfr, C_in, ldc, ctx);

        mpfr_syrk_ref(C_ref, uplo, trans_ref, n, k,
                       mpfr_alpha.get(), A_mpfr,
                       mpfr_beta.get(), C_in_mpfr);

        ErrorResult err = compute_error_matrix_triangle(C_ref, C_out, ldc, uplo, ctx);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c trans=%c", uplo, trans);
        report_result("SYRK", params_str, err, format);

        std::free(A); std::free(C_in);
        std::free(C_out); std::free(alpha); std::free(beta);
    }}
}
