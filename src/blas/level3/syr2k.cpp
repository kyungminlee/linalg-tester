/* syr2k.cpp -- BLAS Level 3 SYR2K accuracy tester */

#include "../level3.h"
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

/* Fortran-ABI function pointer */
extern "C" typedef void (*syr2k_fn_t)(
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
/*   trans='N': C = alpha*A*B^T + alpha*B*A^T + beta*C,  A,B n-by-k   */
/*   trans='T'/'C': C = alpha*A^T*B + alpha*B^T*A + beta*C, A,B k-by-n*/
/* Only the uplo triangle of C is meaningful.                          */
/* ------------------------------------------------------------------ */

static void mpfr_syr2k_ref(MpfrMatrix &C_ref,
                            char uplo, char trans,
                            int n, int k,
                            mpfr_t alpha,
                            const MpfrMatrix &A,
                            const MpfrMatrix &B,
                            mpfr_t beta,
                            const MpfrMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar ab_acc(prec), ba_acc(prec), sum(prec);
    MpfrScalar alpha_sum(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            mpfr_set_d(ab_acc.get(), 0.0, MPFR_RNDN);
            mpfr_set_d(ba_acc.get(), 0.0, MPFR_RNDN);

            for (int p = 0; p < k; ++p) {
                if (trans == 'N') {
                    /* A(i,p)*B(j,p) */
                    mpfr_fma(ab_acc.get(), A.at(i, p), B.at(j, p),
                             ab_acc.get(), MPFR_RNDN);
                    /* B(i,p)*A(j,p) */
                    mpfr_fma(ba_acc.get(), B.at(i, p), A.at(j, p),
                             ba_acc.get(), MPFR_RNDN);
                } else {
                    /* A(p,i)*B(p,j) */
                    mpfr_fma(ab_acc.get(), A.at(p, i), B.at(p, j),
                             ab_acc.get(), MPFR_RNDN);
                    /* B(p,i)*A(p,j) */
                    mpfr_fma(ba_acc.get(), B.at(p, i), A.at(p, j),
                             ba_acc.get(), MPFR_RNDN);
                }
            }

            mpfr_add(sum.get(), ab_acc.get(), ba_acc.get(), MPFR_RNDN);
            mpfr_mul(alpha_sum.get(), alpha, sum.get(), MPFR_RNDN);
            mpfr_mul(beta_c.get(),    beta,  C_in.at(i, j), MPFR_RNDN);
            mpfr_add(C_ref.at(i, j), alpha_sum.get(), beta_c.get(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_syr2k(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<syr2k_fn_t>(load_sym(lib, sym));

    int n = params.n, k = params.k;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T', 'C'}) {
        char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
                         ? 'T' : std::toupper(static_cast<unsigned char>(trans));

        int rows_AB = (trans_ref == 'N') ? n : k;
        int cols_AB = (trans_ref == 'N') ? k : n;

        int lda = rows_AB + params.ld_pad;
        int ldb = rows_AB + params.ld_pad;
        int ldc = n       + params.ld_pad;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        void *A     = gen_random_array(lda * cols_AB, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *B     = gen_random_array(ldb * cols_AB, ctx.typesize, ctx.from_mpfr, prec, &seed_B);
        void *C_in  = gen_random_array(ldc * n,       ctx.typesize, ctx.from_mpfr, prec, &seed_C);
        void *alpha = gen_random_array(1,             ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1,             ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        unsigned sentinel_seed = 0xDEAD0001;
        void *C_out = alloc_with_sentinel(ldc * n, ctx.typesize, sentinel_seed);
        copy_matrix_active(C_out, C_in, n, n, ldc, ctx.typesize);

        fn(&uplo, &trans, &n, &k,
           alpha, A, &lda, B, &ldb, beta, C_out, &ldc,
           (std::size_t)1, (std::size_t)1);

        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_mpfr(rows_AB, cols_AB, prec);
        MpfrMatrix B_mpfr(rows_AB, cols_AB, prec);
        MpfrMatrix C_in_mpfr(n, n, prec);
        MpfrMatrix C_ref(n, n, prec);

        custom_to_mpfr_mat(A_mpfr,    A,    lda, ctx);
        custom_to_mpfr_mat(B_mpfr,    B,    ldb, ctx);
        custom_to_mpfr_mat(C_in_mpfr, C_in, ldc, ctx);

        mpfr_syr2k_ref(C_ref, uplo, trans_ref, n, k,
                        mpfr_alpha.get(), A_mpfr, B_mpfr,
                        mpfr_beta.get(), C_in_mpfr);

        ErrorResult err = compute_error_matrix_triangle(C_ref, C_out, ldc, uplo, ctx);
        SentinelResult sr = check_matrix_sentinels(C_out, n, n, ldc, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c trans=%c", uplo, trans);
        report_result("SYR2K", params_str, err, &sr, format);

        std::free(A); std::free(B); std::free(C_in);
        std::free(C_out); std::free(alpha); std::free(beta);
    }}
}
