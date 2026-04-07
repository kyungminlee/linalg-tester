/* gemm.cpp -- BLAS Level 3 GEMM accuracy tester */

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

/* Fortran-ABI function pointer (hidden char lengths at the end) */
extern "C" typedef void (*gemm_fn_t)(
    const char *transa, const char *transb,
    const int  *m,      const int  *n,      const int  *k,
    const void *alpha,
    const void *A,      const int  *lda,
    const void *B,      const int  *ldb,
    const void *beta,
    void       *C,      const int  *ldc,
    std::size_t transa_len, std::size_t transb_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: C_ref = alpha * op(A) * op(B) + beta * C_in        */
/* ------------------------------------------------------------------ */

static void mpfr_gemm_ref(MpfrMatrix &C_ref,
                           char transa, char transb,
                           int m, int n, int k,
                           mpfr_t alpha,
                           const MpfrMatrix &A,
                           const MpfrMatrix &B,
                           mpfr_t beta,
                           const MpfrMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar acc(prec), alpha_acc(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
            for (int p = 0; p < k; ++p) {
                const mpfr_t &a_ip = (transa == 'N')
                    ? A.at(i, p) : A.at(p, i);
                const mpfr_t &b_pj = (transb == 'N')
                    ? B.at(p, j) : B.at(j, p);
                mpfr_fma(acc.get(), a_ip, b_pj, acc.get(), MPFR_RNDN);
            }
            mpfr_mul(alpha_acc.get(), alpha, acc.get(), MPFR_RNDN);
            mpfr_mul(beta_c.get(),   beta,  C_in.at(i, j), MPFR_RNDN);
            mpfr_add(C_ref.at(i, j), alpha_acc.get(), beta_c.get(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_gemm(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<gemm_fn_t>(load_sym(lib, sym));

    int m = params.m, n = params.n, k = params.k;
    mpfr_prec_t prec = ctx.prec;

    for (char ta : {'N', 'T', 'C'}) {
        for (char tb : {'N', 'T', 'C'}) {
            unsigned seed_A  = params.seed;
            unsigned seed_B  = params.seed + 1;
            unsigned seed_C  = params.seed + 2;
            unsigned seed_ab = params.seed + 3;

            int rows_A = (ta == 'N') ? m : k;
            int cols_A = (ta == 'N') ? k : m;
            int rows_B = (tb == 'N') ? k : n;
            int cols_B = (tb == 'N') ? n : k;

            int lda = rows_A + params.ld_pad;
            int ldb = rows_B + params.ld_pad;
            int ldc = m      + params.ld_pad;

            void *A     = gen_random_array(lda * cols_A, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
            void *B     = gen_random_array(ldb * cols_B, ctx.typesize, ctx.from_mpfr, prec, &seed_B);
            void *C_in  = gen_random_array(ldc * n,      ctx.typesize, ctx.from_mpfr, prec, &seed_C);
            void *alpha = gen_random_array(1,            ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
            void *beta  = gen_random_array(1,            ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

            void *C_out = std::malloc(static_cast<std::size_t>(ldc) * n * ctx.typesize);
            std::memcpy(C_out, C_in, static_cast<std::size_t>(ldc) * n * ctx.typesize);

            fn(&ta, &tb, &m, &n, &k,
               alpha, A, &lda, B, &ldb, beta, C_out, &ldc,
               (std::size_t)1, (std::size_t)1);

            /* Treat 'C' same as 'T' for real types in MPFR reference */
            char ta_ref = (std::toupper(static_cast<unsigned char>(ta)) == 'C') ? 'T' : std::toupper(static_cast<unsigned char>(ta));
            char tb_ref = (std::toupper(static_cast<unsigned char>(tb)) == 'C') ? 'T' : std::toupper(static_cast<unsigned char>(tb));

            MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
            ctx.to_mpfr(mpfr_alpha.get(), alpha);
            ctx.to_mpfr(mpfr_beta.get(),  beta);

            MpfrMatrix A_mpfr(rows_A, cols_A, prec);
            MpfrMatrix B_mpfr(rows_B, cols_B, prec);
            MpfrMatrix C_in_mpfr(m, n, prec);
            MpfrMatrix C_ref(m, n, prec);

            custom_to_mpfr_mat(A_mpfr,    A,    lda, ctx);
            custom_to_mpfr_mat(B_mpfr,    B,    ldb, ctx);
            custom_to_mpfr_mat(C_in_mpfr, C_in, ldc, ctx);

            mpfr_gemm_ref(C_ref, ta_ref, tb_ref, m, n, k,
                          mpfr_alpha.get(), A_mpfr, B_mpfr,
                          mpfr_beta.get(), C_in_mpfr);

            ErrorResult err = compute_error_matrix(C_ref, C_out, ldc, ctx);

            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "transa=%c transb=%c", ta, tb);
            report_result("GEMM", params_str, err, format);

            std::free(A); std::free(B); std::free(C_in);
            std::free(C_out); std::free(alpha); std::free(beta);
        }
    }
}
