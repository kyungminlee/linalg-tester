/* trmm.cpp -- BLAS Level 3 TRMM accuracy tester */

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

/* Fortran-ABI function pointer (same signature as TRSM) */
extern "C" typedef void (*trmm_fn_t)(
    const char *side,   const char *uplo,
    const char *transa, const char *diag,
    const int  *m,      const int  *n,
    const void *alpha,
    const void *A,      const int  *lda,
    void       *B,      const int  *ldb,
    std::size_t side_len,   std::size_t uplo_len,
    std::size_t transa_len, std::size_t diag_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: B = alpha * op(A) * B  (side='L')                   */
/*                 B = alpha * B * op(A)  (side='R')                   */
/*   A triangular (uplo, diag)                                         */
/* ------------------------------------------------------------------ */

static void mpfr_trmm_ref(MpfrMatrix &B_ref,
                           char side, char uplo, char transa, char diag,
                           int m, int n,
                           mpfr_t alpha,
                           const MpfrMatrix &A,
                           const MpfrMatrix &B_in)
{
    int ka = (side == 'L') ? m : n;
    mpfr_prec_t prec = mpfr_get_prec(alpha);

    /* Build working matrix Aw: extract triangle, zero other half,
       set unit diagonal if diag=='U'. */
    MpfrMatrix Aw(ka, ka, prec);
    for (int j = 0; j < ka; ++j) {
        for (int i = 0; i < ka; ++i) {
            if ((uplo == 'U' && j >= i) || (uplo == 'L' && j <= i))
                mpfr_set(Aw.at(i, j), A.at(i, j), MPFR_RNDN);
            else
                mpfr_set_d(Aw.at(i, j), 0.0, MPFR_RNDN);

            if (diag == 'U' && i == j)
                mpfr_set_d(Aw.at(i, j), 1.0, MPFR_RNDN);
        }
    }

    /* If transa != 'N', transpose Aw */
    if (transa != 'N') {
        MpfrMatrix Awt(ka, ka, prec);
        for (int j = 0; j < ka; ++j)
            for (int i = 0; i < ka; ++i)
                mpfr_set(Awt.at(i, j), Aw.at(j, i), MPFR_RNDN);
        for (int j = 0; j < ka; ++j)
            for (int i = 0; i < ka; ++i)
                mpfr_set(Aw.at(i, j), Awt.at(i, j), MPFR_RNDN);
    }

    MpfrScalar acc(prec), alpha_acc(prec);

    if (side == 'L') {
        /* B_ref(i,j) = alpha * sum_p Aw(i,p) * B_in(p,j), p=0..m-1 */
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
                for (int p = 0; p < m; ++p)
                    mpfr_fma(acc.get(), Aw.at(i, p), B_in.at(p, j),
                             acc.get(), MPFR_RNDN);
                mpfr_mul(B_ref.at(i, j), alpha, acc.get(), MPFR_RNDN);
            }
        }
    } else {
        /* B_ref(i,j) = alpha * sum_p B_in(i,p) * Aw(p,j), p=0..n-1 */
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
                for (int p = 0; p < n; ++p)
                    mpfr_fma(acc.get(), B_in.at(i, p), Aw.at(p, j),
                             acc.get(), MPFR_RNDN);
                mpfr_mul(B_ref.at(i, j), alpha, acc.get(), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_trmm(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<trmm_fn_t>(load_sym(lib, sym));

    int m = params.m, n = params.n;
    mpfr_prec_t prec = ctx.prec;

    for (char side : {'L', 'R'}) {
    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T', 'C'}) {
    for (char diag : {'N', 'U'}) {
        int ka = (side == 'L') ? m : n;

        int lda = ka + params.ld_pad;
        int ldb = m  + params.ld_pad;

        unsigned seed_A  = params.seed + 10;
        unsigned seed_B  = params.seed + 11;
        unsigned seed_al = params.seed + 12;

        void *A     = gen_triangular_array(ka, uplo, diag,
                                           ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *B_in  = gen_random_array(ldb * n, ctx.typesize, ctx.from_mpfr, prec, &seed_B);
        void *alpha = gen_random_array(1,       ctx.typesize, ctx.from_mpfr, prec, &seed_al);

        unsigned sentinel_seed = 0xDEAD0001;
        void *B_out = alloc_with_sentinel(ldb * n, ctx.typesize, sentinel_seed);
        copy_matrix_active(B_out, B_in, m, n, ldb, ctx.typesize);

        fn(&side, &uplo, &trans, &diag,
           &m, &n, alpha, A, &lda, B_out, &ldb,
           (std::size_t)1, (std::size_t)1, (std::size_t)1, (std::size_t)1);

        char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
                         ? 'T' : std::toupper(static_cast<unsigned char>(trans));

        MpfrScalar mpfr_alpha(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);

        MpfrMatrix A_mpfr(ka, ka, prec);
        MpfrMatrix B_in_mpfr(m, n, prec);
        MpfrMatrix B_ref(m, n, prec);

        custom_to_mpfr_mat(A_mpfr,    A,    lda, ctx);
        custom_to_mpfr_mat(B_in_mpfr, B_in, ldb, ctx);

        mpfr_trmm_ref(B_ref, side, uplo, trans_ref, diag,
                       m, n, mpfr_alpha.get(), A_mpfr, B_in_mpfr);

        ErrorResult err = compute_error_matrix(B_ref, B_out, ldb, ctx);
        SentinelResult sr = check_matrix_sentinels(B_out, m, n, ldb, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "side=%c uplo=%c trans=%c diag=%c", side, uplo, trans, diag);
        report_result("TRMM", params_str, err, &sr, format);

        std::free(A); std::free(B_in); std::free(B_out); std::free(alpha);
    }}}}
}
