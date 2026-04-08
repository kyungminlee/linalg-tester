/* trsm.cpp -- BLAS Level 3 TRSM accuracy tester */

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
extern "C" typedef void (*trsm_fn_t)(
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
/* MPFR reference: solve op(A)*X = alpha*B or X*op(A) = alpha*B        */
/* ------------------------------------------------------------------ */

static void mpfr_trsm_ref(MpfrMatrix &X,
                           char side, char uplo, char transa, char diag,
                           int m, int n,
                           mpfr_t alpha,
                           const MpfrMatrix &A)
{
    int ka = (side == 'L') ? m : n;
    mpfr_prec_t prec = mpfr_get_prec(alpha);

    /* Scale X by alpha */
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            mpfr_mul(X.at(i, j), alpha, X.at(i, j), MPFR_RNDN);

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

    bool solve_upper = ((uplo == 'U') != (transa != 'N'));

    MpfrScalar tmp(prec);

    if (side == 'L') {
        for (int j = 0; j < n; ++j) {
            if (solve_upper) {
                for (int i = ka - 1; i >= 0; --i) {
                    for (int p = i + 1; p < ka; ++p) {
                        mpfr_mul(tmp.get(), Aw.at(i, p), X.at(p, j), MPFR_RNDN);
                        mpfr_sub(X.at(i, j), X.at(i, j), tmp.get(), MPFR_RNDN);
                    }
                    mpfr_div(X.at(i, j), X.at(i, j), Aw.at(i, i), MPFR_RNDN);
                }
            } else {
                for (int i = 0; i < ka; ++i) {
                    for (int p = 0; p < i; ++p) {
                        mpfr_mul(tmp.get(), Aw.at(i, p), X.at(p, j), MPFR_RNDN);
                        mpfr_sub(X.at(i, j), X.at(i, j), tmp.get(), MPFR_RNDN);
                    }
                    mpfr_div(X.at(i, j), X.at(i, j), Aw.at(i, i), MPFR_RNDN);
                }
            }
        }
    } else {
        for (int i = 0; i < m; ++i) {
            if (solve_upper) {
                for (int j = 0; j < ka; ++j) {
                    for (int p = 0; p < j; ++p) {
                        mpfr_mul(tmp.get(), X.at(i, p), Aw.at(p, j), MPFR_RNDN);
                        mpfr_sub(X.at(i, j), X.at(i, j), tmp.get(), MPFR_RNDN);
                    }
                    mpfr_div(X.at(i, j), X.at(i, j), Aw.at(j, j), MPFR_RNDN);
                }
            } else {
                for (int j = ka - 1; j >= 0; --j) {
                    for (int p = j + 1; p < ka; ++p) {
                        mpfr_mul(tmp.get(), X.at(i, p), Aw.at(p, j), MPFR_RNDN);
                        mpfr_sub(X.at(i, j), X.at(i, j), tmp.get(), MPFR_RNDN);
                    }
                    mpfr_div(X.at(i, j), X.at(i, j), Aw.at(j, j), MPFR_RNDN);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_trsm(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<trsm_fn_t>(load_sym(lib, sym));

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
        void *X_out = alloc_with_sentinel(ldb * n, ctx.typesize, sentinel_seed);
        copy_matrix_active(X_out, B_in, m, n, ldb, ctx.typesize);

        fn(&side, &uplo, &trans, &diag,
           &m, &n, alpha, A, &lda, X_out, &ldb,
           (std::size_t)1, (std::size_t)1, (std::size_t)1, (std::size_t)1);

        char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
                         ? 'T' : std::toupper(static_cast<unsigned char>(trans));

        MpfrScalar mpfr_alpha(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);

        MpfrMatrix A_mpfr(ka, ka, prec);
        MpfrMatrix X_ref(m, n, prec);

        custom_to_mpfr_mat(A_mpfr, A,    lda, ctx);
        custom_to_mpfr_mat(X_ref,  B_in, ldb, ctx);

        mpfr_trsm_ref(X_ref, side, uplo, trans_ref, diag,
                       m, n, mpfr_alpha.get(), A_mpfr);

        ErrorResult err = compute_error_matrix(X_ref, X_out, ldb, ctx);
        SentinelResult sr = check_matrix_sentinels(X_out, m, n, ldb, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "side=%c uplo=%c trans=%c diag=%c", side, uplo, trans, diag);
        report_result("TRSM", params_str, err, &sr, format);

        std::free(A); std::free(B_in); std::free(X_out); std::free(alpha);
    }}}}
}
