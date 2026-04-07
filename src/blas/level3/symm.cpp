/* symm.cpp -- BLAS Level 3 SYMM accuracy tester */

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
extern "C" typedef void (*symm_fn_t)(
    const char *side,  const char *uplo,
    const int  *m,     const int  *n,
    const void *alpha,
    const void *A,     const int  *lda,
    const void *B,     const int  *ldb,
    const void *beta,
    void       *C,     const int  *ldc,
    std::size_t side_len, std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: C = alpha*A*B + beta*C  (side='L')                  */
/*                 C = alpha*B*A + beta*C  (side='R')                  */
/*   A symmetric (stored in uplo triangle)                             */
/* ------------------------------------------------------------------ */

static void mpfr_symm_ref(MpfrMatrix &C_ref,
                           char side, char uplo,
                           int m, int n,
                           mpfr_t alpha,
                           const MpfrMatrix &A,
                           const MpfrMatrix &B,
                           mpfr_t beta,
                           const MpfrMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    int ka = (side == 'L') ? m : n;

    /* Expand symmetric A to full matrix */
    MpfrMatrix Af(ka, ka, prec);
    for (int j = 0; j < ka; ++j) {
        for (int i = 0; i < ka; ++i) {
            if ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j))
                mpfr_set(Af.at(i, j), A.at(i, j), MPFR_RNDN);
            else
                mpfr_set(Af.at(i, j), A.at(j, i), MPFR_RNDN);
        }
    }

    MpfrScalar acc(prec), alpha_acc(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);

            if (side == 'L') {
                /* C(i,j) = sum_p A(i,p) * B(p,j) */
                for (int p = 0; p < m; ++p)
                    mpfr_fma(acc.get(), Af.at(i, p), B.at(p, j), acc.get(), MPFR_RNDN);
            } else {
                /* C(i,j) = sum_p B(i,p) * A(p,j) */
                for (int p = 0; p < n; ++p)
                    mpfr_fma(acc.get(), B.at(i, p), Af.at(p, j), acc.get(), MPFR_RNDN);
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

void test_symm(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<symm_fn_t>(load_sym(lib, sym));

    int m = params.m, n = params.n;
    mpfr_prec_t prec = ctx.prec;

    for (char side : {'L', 'R'}) {
    for (char uplo : {'U', 'L'}) {
        int ka = (side == 'L') ? m : n;

        int lda = ka + params.ld_pad;
        int ldb = m  + params.ld_pad;
        int ldc = m  + params.ld_pad;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        void *A     = gen_symmetric_array(ka, uplo, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *B     = gen_random_array(ldb * n, ctx.typesize, ctx.from_mpfr, prec, &seed_B);
        void *C_in  = gen_random_array(ldc * n, ctx.typesize, ctx.from_mpfr, prec, &seed_C);
        void *alpha = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        void *C_out = std::malloc(static_cast<std::size_t>(ldc) * n * ctx.typesize);
        std::memcpy(C_out, C_in, static_cast<std::size_t>(ldc) * n * ctx.typesize);

        fn(&side, &uplo, &m, &n,
           alpha, A, &lda, B, &ldb, beta, C_out, &ldc,
           (std::size_t)1, (std::size_t)1);

        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_mpfr(ka, ka, prec);
        MpfrMatrix B_mpfr(m, n, prec);
        MpfrMatrix C_in_mpfr(m, n, prec);
        MpfrMatrix C_ref(m, n, prec);

        custom_to_mpfr_mat(A_mpfr,    A,    lda, ctx);
        custom_to_mpfr_mat(B_mpfr,    B,    ldb, ctx);
        custom_to_mpfr_mat(C_in_mpfr, C_in, ldc, ctx);

        mpfr_symm_ref(C_ref, side, uplo, m, n,
                       mpfr_alpha.get(), A_mpfr, B_mpfr,
                       mpfr_beta.get(), C_in_mpfr);

        ErrorResult err = compute_error_matrix(C_ref, C_out, ldc, ctx);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "side=%c uplo=%c", side, uplo);
        report_result("SYMM", params_str, err, format);

        std::free(A); std::free(B); std::free(C_in);
        std::free(C_out); std::free(alpha); std::free(beta);
    }}
}
