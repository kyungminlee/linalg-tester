/* trmv.cpp -- BLAS Level 2 TRMV accuracy tester */

#include "../level2.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer (3 hidden char lengths) */
extern "C" typedef void (*trmv_fn_t)(
    const char *uplo,  const char *trans, const char *diag,
    const int  *n,
    const void *A,     const int  *lda,
    void       *x,     const int  *incx,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: y = op(Aw) * x_in                                   */
/*   Aw = extracted triangle with unit diagonal if diag='U'            */
/* ------------------------------------------------------------------ */

static void mpfr_trmv_ref(MpfrMatrix &y_ref,
                           char uplo, char trans, char diag,
                           int n,
                           const MpfrMatrix &A,
                           const MpfrMatrix &x_in)
{
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar acc(prec);

    for (int i = 0; i < n; ++i) {
        mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j) {
            /* Determine the effective A(row, col) with triangle + diag */
            int row = (trans == 'N') ? i : j;
            int col = (trans == 'N') ? j : i;

            bool in_triangle;
            if (uplo == 'U')
                in_triangle = (row <= col);
            else
                in_triangle = (row >= col);

            if (row == col) {
                if (diag == 'U') {
                    /* unit diagonal: A_eff = 1 */
                    mpfr_add(acc.get(), acc.get(), x_in.at(j, 0), MPFR_RNDN);
                } else {
                    mpfr_fma(acc.get(), A.at(row, col), x_in.at(j, 0),
                             acc.get(), MPFR_RNDN);
                }
            } else if (in_triangle) {
                mpfr_fma(acc.get(), A.at(row, col), x_in.at(j, 0),
                         acc.get(), MPFR_RNDN);
            }
            /* else: zero, skip */
        }
        mpfr_set(y_ref.at(i, 0), acc.get(), MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_trmv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<trmv_fn_t>(load_sym(lib, sym));

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

                void *A    = gen_triangular_array(n, uplo, diag, ctx.typesize,
                                                  ctx.from_mpfr, prec, &seed_A);
                void *x_in = gen_random_array(x_alloc, ctx.typesize,
                                              ctx.from_mpfr, prec, &seed_x);

                /* Save x_in before in-place call */
                void *x_work = std::malloc(static_cast<std::size_t>(x_alloc) * ctx.typesize);
                std::memcpy(x_work, x_in, static_cast<std::size_t>(x_alloc) * ctx.typesize);

                fn(&uplo, &trans, &diag, &n, A, &lda, x_work, &incx,
                   (std::size_t)1, (std::size_t)1, (std::size_t)1);

                /* Treat 'C' same as 'T' for real types */
                char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
                    ? 'T' : std::toupper(static_cast<unsigned char>(trans));

                MpfrMatrix A_mpfr(n, n, prec);
                MpfrMatrix x_in_mpfr(n, 1, prec);
                MpfrMatrix y_ref(n, 1, prec);

                custom_to_mpfr_mat(A_mpfr, A, lda, ctx);
                custom_to_mpfr_vec(x_in_mpfr, x_in, incx, ctx);

                mpfr_trmv_ref(y_ref, uplo, trans_ref, diag, n,
                              A_mpfr, x_in_mpfr);

                ErrorResult err = compute_error_vector(y_ref, x_work, incx, ctx);

                char params_str[128];
                std::snprintf(params_str, sizeof(params_str),
                              "uplo=%c trans=%c diag=%c n=%d incx=%d",
                              uplo, trans, diag, n, incx);
                report_result("TRMV", params_str, err, format);

                std::free(A); std::free(x_in); std::free(x_work);
            }
        }
    }
}
