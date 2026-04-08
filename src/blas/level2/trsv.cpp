/* trsv.cpp -- BLAS Level 2 TRSV accuracy tester */

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

/* Fortran-ABI function pointer (3 hidden char lengths) */
extern "C" typedef void (*trsv_fn_t)(
    const char *uplo,  const char *trans, const char *diag,
    const int  *n,
    const void *A,     const int  *lda,
    void       *x,     const int  *incx,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: solve op(A) * x = b for x                          */
/*   Forward or backward substitution depending on uplo/trans          */
/* ------------------------------------------------------------------ */

static void mpfr_trsv_ref(MpfrMatrix &x_ref,
                           char uplo, char trans, char diag,
                           int n,
                           const MpfrMatrix &A,
                           const MpfrMatrix &b)
{
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar acc(prec), tmp(prec);

    /* Copy b into x_ref */
    for (int i = 0; i < n; ++i)
        mpfr_set(x_ref.at(i, 0), b.at(i, 0), MPFR_RNDN);

    /* Determine iteration order:
       op(A) is lower-triangular => forward substitution (i = 0..n-1)
       op(A) is upper-triangular => backward substitution (i = n-1..0)

       op(A) lower when: (uplo='L' and trans='N') or (uplo='U' and trans!='N')
       op(A) upper when: (uplo='U' and trans='N') or (uplo='L' and trans!='N')
    */
    bool op_lower = (uplo == 'L' && trans == 'N') ||
                    (uplo == 'U' && trans != 'N');

    if (op_lower) {
        /* Forward substitution */
        for (int i = 0; i < n; ++i) {
            mpfr_set(acc.get(), x_ref.at(i, 0), MPFR_RNDN);
            for (int j = 0; j < i; ++j) {
                int row = (trans == 'N') ? i : j;
                int col = (trans == 'N') ? j : i;
                mpfr_mul(tmp.get(), A.at(row, col), x_ref.at(j, 0), MPFR_RNDN);
                mpfr_sub(acc.get(), acc.get(), tmp.get(), MPFR_RNDN);
            }
            if (diag == 'N') {
                mpfr_div(x_ref.at(i, 0), acc.get(), A.at(i, i), MPFR_RNDN);
            } else {
                mpfr_set(x_ref.at(i, 0), acc.get(), MPFR_RNDN);
            }
        }
    } else {
        /* Backward substitution */
        for (int i = n - 1; i >= 0; --i) {
            mpfr_set(acc.get(), x_ref.at(i, 0), MPFR_RNDN);
            for (int j = i + 1; j < n; ++j) {
                int row = (trans == 'N') ? i : j;
                int col = (trans == 'N') ? j : i;
                mpfr_mul(tmp.get(), A.at(row, col), x_ref.at(j, 0), MPFR_RNDN);
                mpfr_sub(acc.get(), acc.get(), tmp.get(), MPFR_RNDN);
            }
            if (diag == 'N') {
                mpfr_div(x_ref.at(i, 0), acc.get(), A.at(i, i), MPFR_RNDN);
            } else {
                mpfr_set(x_ref.at(i, 0), acc.get(), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_trsv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<trsv_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
        for (char trans : {'N', 'T', 'C'}) {
            for (char diag : {'N', 'U'}) {
                int lda = n + params.ld_pad;
                int abs_incx = (incx < 0) ? -incx : incx;
                int x_alloc = 1 + (n - 1) * abs_incx;

                unsigned seed_A = params.seed;
                unsigned seed_x = params.seed + 1;

                /* Generate diagonal-dominant triangular A for stability */
                void *A    = gen_triangular_array(n, uplo, diag, ctx.typesize,
                                                  ctx.from_mpfr, prec, &seed_A);
                void *x_in = gen_random_array(x_alloc, ctx.typesize,
                                              ctx.from_mpfr, prec, &seed_x);

                /* x starts as b; TRSV solves in-place */
                unsigned sentinel_seed = 0xDEAD0001;
                void *x_work = alloc_with_sentinel(x_alloc, ctx.typesize, sentinel_seed);
                copy_vector_active(x_work, x_in, n, incx, ctx.typesize);

                fn(&uplo, &trans, &diag, &n, A, &lda, x_work, &incx,
                   (std::size_t)1, (std::size_t)1, (std::size_t)1);

                /* Treat 'C' same as 'T' for real types */
                char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
                    ? 'T' : std::toupper(static_cast<unsigned char>(trans));

                MpfrMatrix A_mpfr(n, n, prec);
                MpfrMatrix b_mpfr(n, 1, prec);
                MpfrMatrix x_ref(n, 1, prec);

                custom_to_mpfr_mat(A_mpfr, A, lda, ctx);
                custom_to_mpfr_vec(b_mpfr, x_in, incx, ctx);

                mpfr_trsv_ref(x_ref, uplo, trans_ref, diag, n,
                              A_mpfr, b_mpfr);

                ErrorResult err = compute_error_vector(x_ref, x_work, incx, ctx);
                SentinelResult sr = check_vector_sentinels(x_work, n, incx, ctx.typesize, sentinel_seed);

                char params_str[128];
                std::snprintf(params_str, sizeof(params_str),
                              "uplo=%c trans=%c diag=%c n=%d incx=%d",
                              uplo, trans, diag, n, incx);
                report_result("TRSV", params_str, err, &sr, format);

                std::free(A); std::free(x_in); std::free(x_work);
            }
        }
    }
}
