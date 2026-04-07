/* tpsv.cpp -- BLAS Level 2 TPSV accuracy tester */

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

/* Fortran-ABI function pointer (same as TPMV) */
extern "C" typedef void (*tpsv_fn_t)(
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
/* MPFR reference: solve op(A)*x = b via forward/backward subst       */
/* ------------------------------------------------------------------ */

static void mpfr_trisolve(MpfrMatrix &x_ref,
                            char trans, int n,
                            const MpfrMatrix &A)
{
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar tmp(prec);

    auto a_elem = [&](int i, int j) -> const mpfr_t & {
        return (trans == 'N') ? A.at(i, j) : A.at(j, i);
    };

    /* Determine effective triangle shape. For n > 1, check if element
       (1, 0) of the effective matrix is zero => upper triangular. */
    bool effective_upper = true;
    for (int i = 1; i < n && effective_upper; ++i) {
        if (mpfr_zero_p(a_elem(i, 0)) == 0)
            effective_upper = false;
    }

    if (effective_upper) {
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                mpfr_mul(tmp.get(), a_elem(i, j), x_ref.at(j, 0), MPFR_RNDN);
                mpfr_sub(x_ref.at(i, 0), x_ref.at(i, 0), tmp.get(), MPFR_RNDN);
            }
            mpfr_div(x_ref.at(i, 0), x_ref.at(i, 0), a_elem(i, i), MPFR_RNDN);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                mpfr_mul(tmp.get(), a_elem(i, j), x_ref.at(j, 0), MPFR_RNDN);
                mpfr_sub(x_ref.at(i, 0), x_ref.at(i, 0), tmp.get(), MPFR_RNDN);
            }
            mpfr_div(x_ref.at(i, 0), x_ref.at(i, 0), a_elem(i, i), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_tpsv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<tpsv_fn_t>(load_sym(lib, sym));

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

        void *x_out = std::malloc(static_cast<std::size_t>(x_alloc) * ctx.typesize);
        std::memcpy(x_out, x_in, static_cast<std::size_t>(x_alloc) * ctx.typesize);

        fn(&uplo, &trans, &diag, &n,
           AP, x_out, &incx,
           (std::size_t)1, (std::size_t)1, (std::size_t)1);

        char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
            ? 'T' : std::toupper(static_cast<unsigned char>(trans));

        MpfrMatrix A_full(n, n, prec);
        expand_packed_tri_to_full(A_full, AP, n, uplo, diag, ctx);

        MpfrMatrix x_ref(n, 1, prec);
        custom_to_mpfr_vec(x_ref, x_in, incx, ctx);

        mpfr_trisolve(x_ref, trans_ref, n, A_full);

        ErrorResult err = compute_error_vector(x_ref, x_out, incx, ctx);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c trans=%c diag=%c n=%d",
                      uplo, trans, diag, n);
        report_result("TPSV", params_str, err, format);

        std::free(AP); std::free(x_in); std::free(x_out);
    }}}
}
