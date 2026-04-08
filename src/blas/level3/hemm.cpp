/* hemm.cpp -- BLAS Level 3 HEMM accuracy tester (complex-only) */

#include "../level3.h"
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
extern "C" typedef void (*hemm_fn_t)(
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
/*   A Hermitian (stored in uplo triangle)                             */
/* ------------------------------------------------------------------ */

static void mpfr_hemm_ref(MpfrComplexMatrix &C_ref,
                            char side, char uplo,
                            int m, int n,
                            const MpfrComplexScalar &alpha,
                            const MpfrComplexMatrix &A,
                            const MpfrComplexMatrix &B,
                            const MpfrComplexScalar &beta,
                            const MpfrComplexMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha.re());
    int ka = (side == 'L') ? m : n;

    /* Expand Hermitian A to full matrix */
    MpfrComplexMatrix Af(ka, ka, prec);
    for (int j = 0; j < ka; ++j) {
        for (int i = 0; i < ka; ++i) {
            if ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j)) {
                mpfr_set(Af.re(i, j), A.re(i, j), MPFR_RNDN);
                mpfr_set(Af.im(i, j), A.im(i, j), MPFR_RNDN);
            } else {
                /* A(i,j) = conj(A(j,i)) */
                mpfr_complex_conj(Af.re(i, j), Af.im(i, j),
                                   A.re(j, i), A.im(j, i), MPFR_RNDN);
            }
        }
    }

    MpfrComplexScalar acc(prec), alpha_acc(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(acc.re(), 0.0, MPFR_RNDN);
            mpfr_set_d(acc.im(), 0.0, MPFR_RNDN);

            if (side == 'L') {
                /* C(i,j) = sum_p Af(i,p) * B(p,j) */
                for (int p = 0; p < m; ++p)
                    mpfr_complex_fma(acc.re(), acc.im(),
                                      Af.re(i, p), Af.im(i, p),
                                      B.re(p, j),  B.im(p, j),
                                      MPFR_RNDN);
            } else {
                /* C(i,j) = sum_p B(i,p) * Af(p,j) */
                for (int p = 0; p < n; ++p)
                    mpfr_complex_fma(acc.re(), acc.im(),
                                      B.re(i, p),  B.im(i, p),
                                      Af.re(p, j), Af.im(p, j),
                                      MPFR_RNDN);
            }

            /* alpha * acc */
            mpfr_complex_mul(alpha_acc.re(), alpha_acc.im(),
                              alpha.re(), alpha.im(),
                              acc.re(), acc.im(), MPFR_RNDN);
            /* beta * C_in(i,j) */
            mpfr_complex_mul(beta_c.re(), beta_c.im(),
                              beta.re(), beta.im(),
                              C_in.re(i, j), C_in.im(i, j), MPFR_RNDN);
            /* C_ref(i,j) = alpha*acc + beta*C_in */
            mpfr_complex_add(C_ref.re(i, j), C_ref.im(i, j),
                              alpha_acc.re(), alpha_acc.im(),
                              beta_c.re(), beta_c.im(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_hemm(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        fprintf(stderr, "HEMM requires --complex\n");
        return;
    }

    auto *fn = reinterpret_cast<hemm_fn_t>(load_sym(lib, sym));

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

        void *A     = gen_hermitian_array(ka, uplo, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *B     = gen_random_complex_array(ldb * n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_B);
        void *C_in  = gen_random_complex_array(ldc * n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_C);
        void *alpha = gen_random_complex_array(1,       ctx.typesize, ctx.from_mpfr_complex, prec, &seed_ab);
        void *beta  = gen_random_complex_array(1,       ctx.typesize, ctx.from_mpfr_complex, prec, &seed_ab);

        unsigned sentinel_seed = 0xDEAD0001;
        void *C_out = alloc_with_sentinel(ldc * n, ctx.typesize, sentinel_seed);
        copy_matrix_active(C_out, C_in, m, n, ldc, ctx.typesize);

        fn(&side, &uplo, &m, &n,
           alpha, A, &lda, B, &ldb, beta, C_out, &ldc,
           (std::size_t)1, (std::size_t)1);

        MpfrComplexScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr_complex(mpfr_alpha.re(), mpfr_alpha.im(), alpha);
        ctx.to_mpfr_complex(mpfr_beta.re(),  mpfr_beta.im(),  beta);

        MpfrComplexMatrix A_mpfr(ka, ka, prec);
        MpfrComplexMatrix B_mpfr(m, n, prec);
        MpfrComplexMatrix C_in_mpfr(m, n, prec);
        MpfrComplexMatrix C_ref(m, n, prec);

        custom_to_mpfr_complex_mat(A_mpfr,    A,    lda, ctx);
        custom_to_mpfr_complex_mat(B_mpfr,    B,    ldb, ctx);
        custom_to_mpfr_complex_mat(C_in_mpfr, C_in, ldc, ctx);

        mpfr_hemm_ref(C_ref, side, uplo, m, n,
                       mpfr_alpha, A_mpfr, B_mpfr,
                       mpfr_beta, C_in_mpfr);

        ErrorResult err = compute_error_complex_matrix(C_ref, C_out, ldc, ctx);
        SentinelResult sr = check_matrix_sentinels(C_out, m, n, ldc, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "side=%c uplo=%c", side, uplo);
        report_result("HEMM", params_str, err, &sr, format);

        std::free(A); std::free(B); std::free(C_in);
        std::free(C_out); std::free(alpha); std::free(beta);
    }}
}
