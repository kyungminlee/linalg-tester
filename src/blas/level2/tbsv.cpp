/* tbsv.cpp -- BLAS Level 2 TBSV accuracy tester */

#include "../level2.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer (same as TBMV) */
extern "C" typedef void (*tbsv_fn_t)(
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
/* MPFR reference: solve op(A)*x = b via forward/backward subst       */
/* ------------------------------------------------------------------ */

static void mpfr_trisolve(MpfrMatrix &x_ref,
                            char trans, int n,
                            const MpfrMatrix &A)
{
    /* x_ref already contains b on entry; solution overwrites it. */
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar tmp(prec);

    /* After applying trans, determine the effective triangle shape. */
    /* If trans='N': upper stays upper, lower stays lower.
       If trans='T': upper becomes lower, lower becomes upper.
       We work with the transposed access pattern. */

    /* For trans='N' + upper, or trans='T' + lower: backward substitution.
       For trans='N' + lower, or trans='T' + upper: forward substitution.
       We determine this by checking if the effective matrix is upper. */

    /* Helper lambda for element access with transpose */
    auto a_elem = [&](int i, int j) -> const mpfr_t & {
        return (trans == 'N') ? A.at(i, j) : A.at(j, i);
    };

    /* Check if the effective matrix (after transpose) is upper triangular.
       Original upper + no-trans = upper effective.
       Original upper + trans = lower effective.
       But we don't have 'uplo' here -- we have the full expanded matrix
       which already has the triangle set. After transpose, the nonzero
       pattern flips. We detect by checking A(0,n-1) vs A(n-1,0). */

    /* Actually, we should just detect from the matrix structure.
       Simpler: check if the upper-right is nonzero (upper) or
       lower-left is nonzero (lower). But with unit diag, we can't
       easily tell. Instead, let's check if a_elem(0, n-1) is nonzero
       for n > 1. For n=1, it doesn't matter. */

    /* Most robust approach: try to determine from the original matrix.
       For the effective system (trans applied), if diagonal and above
       are the nonzero part, do backward subst; if diagonal and below,
       do forward subst. We check: is the effective matrix upper
       triangular? i.e., is a_elem(i, j) == 0 for i > j? */

    bool effective_upper = true;
    for (int i = 1; i < n && effective_upper; ++i) {
        if (mpfr_zero_p(a_elem(i, 0)) == 0)
            effective_upper = false;
    }

    if (effective_upper) {
        /* backward substitution */
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                mpfr_mul(tmp.get(), a_elem(i, j), x_ref.at(j, 0), MPFR_RNDN);
                mpfr_sub(x_ref.at(i, 0), x_ref.at(i, 0), tmp.get(), MPFR_RNDN);
            }
            mpfr_div(x_ref.at(i, 0), x_ref.at(i, 0), a_elem(i, i), MPFR_RNDN);
        }
    } else {
        /* forward substitution */
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

void test_tbsv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<tbsv_fn_t>(load_sym(lib, sym));

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

        void *x_out = std::malloc(static_cast<std::size_t>(x_alloc) * ctx.typesize);
        std::memcpy(x_out, x_in, static_cast<std::size_t>(x_alloc) * ctx.typesize);

        fn(&uplo, &trans, &diag, &n, &k,
           AB, &ldab, x_out, &incx,
           (std::size_t)1, (std::size_t)1, (std::size_t)1);

        char trans_ref = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
            ? 'T' : std::toupper(static_cast<unsigned char>(trans));

        MpfrMatrix A_full(n, n, prec);
        expand_triband_to_full(A_full, AB, ldab, n, k, uplo, diag, ctx);

        MpfrMatrix x_ref(n, 1, prec);
        custom_to_mpfr_vec(x_ref, x_in, incx, ctx);

        mpfr_trisolve(x_ref, trans_ref, n, A_full);

        ErrorResult err = compute_error_vector(x_ref, x_out, incx, ctx);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c trans=%c diag=%c n=%d k=%d",
                      uplo, trans, diag, n, k);
        report_result("TBSV", params_str, err, format);

        std::free(AB); std::free(x_in); std::free(x_out);
    }}}
}
