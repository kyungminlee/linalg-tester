/* her2k.cpp -- BLAS Level 3 HER2K accuracy tester (complex-only) */

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
extern "C" typedef void (*her2k_fn_t)(
    const char *uplo,  const char *trans,
    const int  *n,     const int  *k,
    const void *alpha,
    const void *A,     const int  *lda,
    const void *B,     const int  *ldb,
    const void *beta,
    void       *C,     const int  *ldc,
    std::size_t uplo_len, std::size_t trans_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference:                                                      */
/*   trans='N': C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C           */
/*              A,B are n-by-k                                          */
/*   trans='C': C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C           */
/*              A,B are k-by-n                                          */
/* alpha is complex, beta is REAL. Only uplo triangle is meaningful.    */
/* ------------------------------------------------------------------ */

static void mpfr_her2k_ref(MpfrComplexMatrix &C_ref,
                             char uplo, char trans,
                             int n, int k,
                             const MpfrComplexScalar &alpha,
                             const MpfrComplexMatrix &A,
                             const MpfrComplexMatrix &B,
                             mpfr_t beta,
                             const MpfrComplexMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha.re());

    MpfrComplexScalar conj_alpha(prec);
    mpfr_complex_conj(conj_alpha.re(), conj_alpha.im(),
                       alpha.re(), alpha.im(), MPFR_RNDN);

    MpfrComplexScalar ab_acc(prec), ba_acc(prec);
    MpfrComplexScalar term1(prec), term2(prec), sum12(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            mpfr_set_d(ab_acc.re(), 0.0, MPFR_RNDN);
            mpfr_set_d(ab_acc.im(), 0.0, MPFR_RNDN);
            mpfr_set_d(ba_acc.re(), 0.0, MPFR_RNDN);
            mpfr_set_d(ba_acc.im(), 0.0, MPFR_RNDN);

            for (int p = 0; p < k; ++p) {
                if (trans == 'N') {
                    /* ab_acc += A(i,p) * conj(B(j,p)) */
                    MpfrComplexScalar conj_bjp(prec);
                    mpfr_complex_conj(conj_bjp.re(), conj_bjp.im(),
                                       B.re(j, p), B.im(j, p), MPFR_RNDN);
                    mpfr_complex_fma(ab_acc.re(), ab_acc.im(),
                                      A.re(i, p), A.im(i, p),
                                      conj_bjp.re(), conj_bjp.im(),
                                      MPFR_RNDN);
                    /* ba_acc += B(i,p) * conj(A(j,p)) */
                    MpfrComplexScalar conj_ajp(prec);
                    mpfr_complex_conj(conj_ajp.re(), conj_ajp.im(),
                                       A.re(j, p), A.im(j, p), MPFR_RNDN);
                    mpfr_complex_fma(ba_acc.re(), ba_acc.im(),
                                      B.re(i, p), B.im(i, p),
                                      conj_ajp.re(), conj_ajp.im(),
                                      MPFR_RNDN);
                } else {
                    /* ab_acc += conj(A(p,i)) * B(p,j) */
                    MpfrComplexScalar conj_api(prec);
                    mpfr_complex_conj(conj_api.re(), conj_api.im(),
                                       A.re(p, i), A.im(p, i), MPFR_RNDN);
                    mpfr_complex_fma(ab_acc.re(), ab_acc.im(),
                                      conj_api.re(), conj_api.im(),
                                      B.re(p, j), B.im(p, j),
                                      MPFR_RNDN);
                    /* ba_acc += conj(B(p,i)) * A(p,j) */
                    MpfrComplexScalar conj_bpi(prec);
                    mpfr_complex_conj(conj_bpi.re(), conj_bpi.im(),
                                       B.re(p, i), B.im(p, i), MPFR_RNDN);
                    mpfr_complex_fma(ba_acc.re(), ba_acc.im(),
                                      conj_bpi.re(), conj_bpi.im(),
                                      A.re(p, j), A.im(p, j),
                                      MPFR_RNDN);
                }
            }

            /* term1 = alpha * ab_acc */
            mpfr_complex_mul(term1.re(), term1.im(),
                              alpha.re(), alpha.im(),
                              ab_acc.re(), ab_acc.im(), MPFR_RNDN);
            /* term2 = conj(alpha) * ba_acc */
            mpfr_complex_mul(term2.re(), term2.im(),
                              conj_alpha.re(), conj_alpha.im(),
                              ba_acc.re(), ba_acc.im(), MPFR_RNDN);
            /* sum12 = term1 + term2 */
            mpfr_complex_add(sum12.re(), sum12.im(),
                              term1.re(), term1.im(),
                              term2.re(), term2.im(), MPFR_RNDN);
            /* beta_c = beta * C_in(i,j)  (beta is real) */
            mpfr_complex_mul_real(beta_c.re(), beta_c.im(),
                                   C_in.re(i, j), C_in.im(i, j),
                                   beta, MPFR_RNDN);
            /* C_ref(i,j) = sum12 + beta_c */
            mpfr_complex_add(C_ref.re(i, j), C_ref.im(i, j),
                              sum12.re(), sum12.im(),
                              beta_c.re(), beta_c.im(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_her2k(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        fprintf(stderr, "HER2K requires --complex\n");
        return;
    }

    auto *fn = reinterpret_cast<her2k_fn_t>(load_sym(lib, sym));

    int n = params.n, k = params.k;
    mpfr_prec_t prec = ctx.prec;
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'C'}) {
        int rows_AB = (trans == 'N') ? n : k;
        int cols_AB = (trans == 'N') ? k : n;

        int lda = rows_AB + params.ld_pad;
        int ldb = rows_AB + params.ld_pad;
        int ldc = n       + params.ld_pad;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_al = params.seed + 3;
        unsigned seed_be = params.seed + 4;

        void *A     = gen_random_complex_array(lda * cols_AB, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *B     = gen_random_complex_array(ldb * cols_AB, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_B);
        void *C_in  = gen_hermitian_array(n, uplo, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_C);
        void *alpha = gen_random_complex_array(1,             ctx.typesize, ctx.from_mpfr_complex, prec, &seed_al);
        void *beta  = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_be);

        unsigned sentinel_seed = 0xDEAD0001;
        void *C_out = alloc_with_sentinel(ldc * n, ctx.typesize, sentinel_seed);
        copy_matrix_active(C_out, C_in, n, n, ldc, ctx.typesize);

        fn(&uplo, &trans, &n, &k,
           alpha, A, &lda, B, &ldb, beta, C_out, &ldc,
           (std::size_t)1, (std::size_t)1);

        MpfrComplexScalar mpfr_alpha(prec);
        ctx.to_mpfr_complex(mpfr_alpha.re(), mpfr_alpha.im(), alpha);

        MpfrScalar mpfr_beta(prec);
        ctx.to_mpfr(mpfr_beta.get(), beta);

        MpfrComplexMatrix A_mpfr(rows_AB, cols_AB, prec);
        MpfrComplexMatrix B_mpfr(rows_AB, cols_AB, prec);
        MpfrComplexMatrix C_in_mpfr(n, n, prec);
        MpfrComplexMatrix C_ref(n, n, prec);

        custom_to_mpfr_complex_mat(A_mpfr,    A,    lda, ctx);
        custom_to_mpfr_complex_mat(B_mpfr,    B,    ldb, ctx);
        custom_to_mpfr_complex_mat(C_in_mpfr, C_in, ldc, ctx);

        mpfr_her2k_ref(C_ref, uplo, trans, n, k,
                        mpfr_alpha, A_mpfr, B_mpfr,
                        mpfr_beta.get(), C_in_mpfr);

        ErrorResult err = compute_error_complex_matrix_triangle(C_ref, C_out, ldc, uplo, ctx);
        SentinelResult sr = check_matrix_sentinels(C_out, n, n, ldc, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c trans=%c", uplo, trans);
        report_result("HER2K", params_str, err, &sr, format);

        std::free(A); std::free(B); std::free(C_in);
        std::free(C_out); std::free(alpha); std::free(beta);
    }}
}
