/* tpmv.cpp -- BLAS Level 2 TPMV accuracy tester */

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

/* Fortran-ABI function pointer */
extern "C" typedef void (*tpmv_fn_t)(
    const char *uplo, const char *trans, const char *diag,
    const int  *n,
    const void *AP,
    void       *x,    const int  *incx,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len
);

/* ------------------------------------------------------------------ */
/* Expand packed triangular storage to full MpfrMatrix                 */
/* ------------------------------------------------------------------ */

static void expand_packed_tri_to_full(MpfrMatrix &full,
                                       const void *AP,
                                       int n, char uplo, char diag,
                                       const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(AP);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            mpfr_set_d(full.at(i, j), 0.0, MPFR_RNDN);

    if (uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                std::size_t idx = static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(j) * (j + 1) / 2;
                ctx.to_mpfr(full.at(i, j), p + idx * ctx.typesize);
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                std::size_t idx = static_cast<std::size_t>(i)
                                + static_cast<std::size_t>(j)
                                  * (2 * n - j - 1) / 2;
                ctx.to_mpfr(full.at(i, j), p + idx * ctx.typesize);
            }
        }
    }

    if (diag == 'U') {
        for (int i = 0; i < n; ++i)
            mpfr_set_d(full.at(i, i), 1.0, MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* MPFR reference: y = op(A) * x                                      */
/* ------------------------------------------------------------------ */

static void mpfr_matvec(MpfrMatrix &y,
                          char trans, int n,
                          const MpfrMatrix &A,
                          const MpfrMatrix &x)
{
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));

    for (int i = 0; i < n; ++i) {
        mpfr_set_d(y.at(i, 0), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j) {
            const mpfr_t &a_ij = (trans == 'N')
                ? A.at(i, j) : A.at(j, i);
            mpfr_fma(y.at(i, 0), a_ij, x.at(j, 0),
                     y.at(i, 0), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_tpmv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<tpmv_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T', 'C'}) {
    for (char diag : {'N', 'U'}) {
        int abs_incx = (incx < 0) ? -incx : incx;
        int x_alloc = 1 + (n - 1) * abs_incx;

        unsigned seed_AP = params.seed;
        unsigned seed_x  = params.seed + 1;

        void *AP   = gen_packed_triangular_array(n, uplo, diag,
                                                  ctx.typesize, ctx.from_mpfr, prec, &seed_AP);
        void *x_in = gen_random_array(x_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_x);

        unsigned sentinel_seed = 0xDEAD0001;
        void *x_out = alloc_with_sentinel(x_alloc, ctx.typesize, sentinel_seed);
        copy_vector_active(x_out, x_in, n, incx, ctx.typesize);

        fn(&uplo, &trans, &diag, &n,
           AP, x_out, &incx,
           (std::size_t)1, (std::size_t)1, (std::size_t)1);

        char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
            ? 'T' : std::toupper(static_cast<unsigned char>(trans));

        MpfrMatrix A_full(n, n, prec);
        expand_packed_tri_to_full(A_full, AP, n, uplo, diag, ctx);

        MpfrMatrix x_mpfr(n, 1, prec);
        MpfrMatrix y_ref(n, 1, prec);

        custom_to_mpfr_vec(x_mpfr, x_in, incx, ctx);
        mpfr_matvec(y_ref, trans_ref, n, A_full, x_mpfr);

        ErrorResult err = compute_error_vector(y_ref, x_out, incx, ctx);
        SentinelResult sr = check_vector_sentinels(x_out, n, incx, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c trans=%c diag=%c n=%d",
                      uplo, trans, diag, n);
        report_result("TPMV", params_str, err, &sr, format);

        std::free(AP); std::free(x_in); std::free(x_out);
    }}}
}
