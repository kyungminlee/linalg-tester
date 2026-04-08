/* tbmv.cpp -- BLAS Level 2 TBMV accuracy tester */

#include "../level2.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer */
extern "C" typedef void (*tbmv_fn_t)(
    const char *uplo, const char *trans, const char *diag,
    const int  *n,    const int  *k,
    const void *AB,   const int  *ldab,
    void       *x,    const int  *incx,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len
);

/* ------------------------------------------------------------------ */
/* Expand triangular banded storage to full MpfrMatrix                 */
/* ------------------------------------------------------------------ */

static void expand_triband_to_full(MpfrMatrix &full,
                                    const void *AB, int ldab,
                                    int n, int k, char uplo, char diag,
                                    const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(AB);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            mpfr_set_d(full.at(i, j), 0.0, MPFR_RNDN);

    if (uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            int i_lo = std::max(0, j - k);
            for (int i = i_lo; i <= j; ++i) {
                std::size_t idx = static_cast<std::size_t>(k + i - j)
                                + static_cast<std::size_t>(j) * ldab;
                ctx.to_mpfr(full.at(i, j), p + idx * ctx.typesize);
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            int i_hi = std::min(n - 1, j + k);
            for (int i = j; i <= i_hi; ++i) {
                std::size_t idx = static_cast<std::size_t>(i - j)
                                + static_cast<std::size_t>(j) * ldab;
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

void test_tbmv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<tbmv_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int k = params.kl;
    int incx = params.incx;
    mpfr_prec_t prec = ctx.prec;

    int ldab = k + 1;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T', 'C'}) {
    for (char diag : {'N', 'U'}) {
        int abs_incx = (incx < 0) ? -incx : incx;
        int x_alloc = 1 + (n - 1) * abs_incx;

        unsigned seed_AB = params.seed;
        unsigned seed_x  = params.seed + 1;

        void *AB   = gen_triangular_band_array(n, k, uplo, diag,
                                                ctx.typesize, ctx.from_mpfr, prec, &seed_AB);
        void *x_in = gen_random_array(x_alloc, ctx.typesize, ctx.from_mpfr, prec, &seed_x);

        unsigned sentinel_seed = 0xDEAD0001;
        void *x_out = alloc_with_sentinel(x_alloc, ctx.typesize, sentinel_seed);
        copy_vector_active(x_out, x_in, n, incx, ctx.typesize);

        fn(&uplo, &trans, &diag, &n, &k,
           AB, &ldab, x_out, &incx,
           (std::size_t)1, (std::size_t)1, (std::size_t)1);

        char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
            ? 'T' : std::toupper(static_cast<unsigned char>(trans));

        MpfrMatrix A_full(n, n, prec);
        expand_triband_to_full(A_full, AB, ldab, n, k, uplo, diag, ctx);

        MpfrMatrix x_mpfr(n, 1, prec);
        MpfrMatrix y_ref(n, 1, prec);

        custom_to_mpfr_vec(x_mpfr, x_in, incx, ctx);
        mpfr_matvec(y_ref, trans_ref, n, A_full, x_mpfr);

        ErrorResult err = compute_error_vector(y_ref, x_out, incx, ctx);
        SentinelResult sr = check_vector_sentinels(x_out, n, incx, ctx.typesize, sentinel_seed);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c trans=%c diag=%c n=%d k=%d",
                      uplo, trans, diag, n, k);
        report_result("TBMV", params_str, err, &sr, format);

        std::free(AB); std::free(x_in); std::free(x_out);
    }}}
}
