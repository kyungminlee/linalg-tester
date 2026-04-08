/* herk.cpp -- BLAS Level 3 HERK accuracy tester (complex-only) */

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
extern "C" typedef void (*herk_fn_t)(
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
/*   trans='N': C = alpha * A * A^H + beta * C,  A is n-by-k           */
/*   trans='C': C = alpha * A^H * A + beta * C,  A is k-by-n           */
/* alpha and beta are REAL scalars. Only uplo triangle is meaningful.   */
/* ------------------------------------------------------------------ */

static void mpfr_herk_ref(MpfrComplexMatrix &C_ref,
                            char uplo, char trans,
                            int n, int k,
                            mpfr_t alpha,
                            const MpfrComplexMatrix &A,
                            mpfr_t beta,
                            const MpfrComplexMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrComplexScalar acc(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            mpfr_set_d(acc.re(), 0.0, MPFR_RNDN);
            mpfr_set_d(acc.im(), 0.0, MPFR_RNDN);

            for (int p = 0; p < k; ++p) {
                if (trans == 'N') {
                    /* A(i,p) * conj(A(j,p)) */
                    MpfrComplexScalar conj_ajp(prec);
                    mpfr_complex_conj(conj_ajp.re(), conj_ajp.im(),
                                       A.re(j, p), A.im(j, p), MPFR_RNDN);
                    mpfr_complex_fma(acc.re(), acc.im(),
                                      A.re(i, p), A.im(i, p),
                                      conj_ajp.re(), conj_ajp.im(),
                                      MPFR_RNDN);
                } else {
                    /* conj(A(p,i)) * A(p,j) */
                    MpfrComplexScalar conj_api(prec);
                    mpfr_complex_conj(conj_api.re(), conj_api.im(),
                                       A.re(p, i), A.im(p, i), MPFR_RNDN);
                    mpfr_complex_fma(acc.re(), acc.im(),
                                      conj_api.re(), conj_api.im(),
                                      A.re(p, j), A.im(p, j),
                                      MPFR_RNDN);
                }
            }

            /* alpha * acc  (alpha is real) */
            MpfrComplexScalar alpha_acc(prec);
            mpfr_complex_mul_real(alpha_acc.re(), alpha_acc.im(),
                                   acc.re(), acc.im(), alpha, MPFR_RNDN);
            /* beta * C_in(i,j)  (beta is real) */
            mpfr_complex_mul_real(beta_c.re(), beta_c.im(),
                                   C_in.re(i, j), C_in.im(i, j),
                                   beta, MPFR_RNDN);
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

void test_herk(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        fprintf(stderr, "HERK requires --complex\n");
        return;
    }

    auto *fn = reinterpret_cast<herk_fn_t>(load_sym(lib, sym));

    int n = params.n, k = params.k;
    mpfr_prec_t prec = ctx.prec;
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'C'}) {
        int rows_A = (trans == 'N') ? n : k;
        int cols_A = (trans == 'N') ? k : n;

        int lda = rows_A + params.ld_pad;
        int ldc = n      + params.ld_pad;

        unsigned seed_A  = params.seed;
        unsigned seed_C  = params.seed + 1;
        unsigned seed_al = params.seed + 2;
        unsigned seed_be = params.seed + 3;

        void *A     = gen_random_complex_array(lda * cols_A, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *C_in  = gen_hermitian_array(n, uplo, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_C);
        void *alpha = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_al);
        void *beta  = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_be);

        unsigned sentinel_seed = 0xDEAD0001;
        void *C_out = alloc_with_sentinel(ldc * n, ctx.typesize, sentinel_seed);
        copy_matrix_active(C_out, C_in, n, n, ldc, ctx.typesize);

        fn(&uplo, &trans, &n, &k,
           alpha, A, &lda, beta, C_out, &ldc,
           (std::size_t)1, (std::size_t)1);

        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrComplexMatrix A_mpfr(rows_A, cols_A, prec);
        MpfrComplexMatrix C_in_mpfr(n, n, prec);
        MpfrComplexMatrix C_ref(n, n, prec);

        custom_to_mpfr_complex_mat(A_mpfr,    A,    lda, ctx);
        custom_to_mpfr_complex_mat(C_in_mpfr, C_in, ldc, ctx);

        mpfr_herk_ref(C_ref, uplo, trans, n, k,
                       mpfr_alpha.get(), A_mpfr,
                       mpfr_beta.get(), C_in_mpfr);

        ErrorResult err = compute_error_complex_matrix_triangle(C_ref, C_out, ldc, uplo, ctx);
        SentinelResult sr = check_matrix_sentinels(C_out, n, n, ldc, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c trans=%c", uplo, trans);
        report_result("HERK", params_str, err, &sr, format);

        std::free(A); std::free(C_in);
        std::free(C_out); std::free(alpha); std::free(beta);
    }}
}
