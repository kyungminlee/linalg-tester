/* ptrsv.cpp -- PBLAS Level 2 PTRSV accuracy tester */

#include "../pblas.h"
#include "../pblas_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer for PDTRSV (hidden char lengths at end) */
extern "C" typedef void (*ptrsv_fn_t)(
    const char *uplo,   const char *trans,  const char *diag,
    const int  *n,
    const void *A,      const int  *ia,     const int  *ja,     const int  *desca,
    void       *x,      const int  *ix,     const int  *jx,     const int  *descx,
    const int  *incx,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: solve op(A)*x = b (in-place, b stored in x)        */
/* Forward/backward substitution on the full triangular system.        */
/* ------------------------------------------------------------------ */

static void mpfr_ptrsv_ref(MpfrMatrix &x_ref,
                            char uplo, char trans, char diag,
                            int n,
                            const MpfrMatrix &A,
                            const MpfrMatrix &b_in)
{
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));

    /* Build full triangular matrix */
    MpfrMatrix Aw(n, n, prec);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if ((uplo == 'U' && j >= i) || (uplo == 'L' && j <= i))
                mpfr_set(Aw.at(i, j), A.at(i, j), MPFR_RNDN);
            else
                mpfr_set_d(Aw.at(i, j), 0.0, MPFR_RNDN);
            if (diag == 'U' && i == j)
                mpfr_set_d(Aw.at(i, j), 1.0, MPFR_RNDN);
        }
    }

    /* Transpose if needed */
    if (trans != 'N') {
        MpfrMatrix Awt(n, n, prec);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                mpfr_set(Awt.at(i, j), Aw.at(j, i), MPFR_RNDN);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                mpfr_set(Aw.at(i, j), Awt.at(i, j), MPFR_RNDN);
    }

    /* Copy b_in to x_ref */
    for (int i = 0; i < n; ++i)
        mpfr_set(x_ref.at(i, 0), b_in.at(i, 0), MPFR_RNDN);

    /* Forward/backward substitution */
    bool solve_upper = ((uplo == 'U') != (trans != 'N'));
    MpfrScalar tmp(prec);

    if (solve_upper) {
        /* Back substitution (upper triangular after transpose consideration) */
        for (int i = n - 1; i >= 0; --i) {
            for (int p = i + 1; p < n; ++p) {
                mpfr_mul(tmp.get(), Aw.at(i, p), x_ref.at(p, 0), MPFR_RNDN);
                mpfr_sub(x_ref.at(i, 0), x_ref.at(i, 0), tmp.get(), MPFR_RNDN);
            }
            mpfr_div(x_ref.at(i, 0), x_ref.at(i, 0), Aw.at(i, i), MPFR_RNDN);
        }
    } else {
        /* Forward substitution (lower triangular after transpose consideration) */
        for (int i = 0; i < n; ++i) {
            for (int p = 0; p < i; ++p) {
                mpfr_mul(tmp.get(), Aw.at(i, p), x_ref.at(p, 0), MPFR_RNDN);
                mpfr_sub(x_ref.at(i, 0), x_ref.at(i, 0), tmp.get(), MPFR_RNDN);
            }
            mpfr_div(x_ref.at(i, 0), x_ref.at(i, 0), Aw.at(i, i), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_ptrsv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.n;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PTRSV", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<ptrsv_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PTRSV", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T'}) {
    for (char diag : {'N', 'U'}) {
        unsigned seed_A = params.seed + 10;
        unsigned seed_x = params.seed + 11;

        /* Generate global data */
        void *A_g = gen_triangular_array(n, uplo, diag,
                                         ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *x_g = gen_random_array(n, ctx.typesize, ctx.from_mpfr, prec, &seed_x);

        /* Local dimensions for matrix A (n x n) */
        int loc_rA = pc.local_rows(n);
        int loc_cA = pc.local_cols(n);
        int lld_a  = std::max(1, loc_rA);

        /* Local dimensions for vector x (n x 1) */
        int loc_rx = pc.local_rows(n);
        int loc_cx = pc.local_cols(1);
        int lld_x  = std::max(1, loc_rx);

        /* Allocate local storage */
        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD000Au;
        void *x_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_x) * std::max(1, loc_cx),
                                           ctx.typesize, sentinel_seed);

        /* Scatter global to local */
        scatter_global_to_local(A_loc, lld_a, A_g, n,
                                n, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);
        scatter_global_to_local(x_loc, lld_x, x_g, n,
                                n, 1, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);

        /* Create descriptors */
        int desc_a[9], desc_x[9];
        pc.make_desc(desc_a, n, n, lld_a);
        pc.make_desc(desc_x, n, 1, lld_x);

        /* Call PBLAS (in-place: solve op(A)*x = b, b stored in x) */
        int one = 1;
        int incx = 1;
        fn(&uplo, &trans, &diag, &n,
           A_loc, &one, &one, desc_a,
           x_loc, &one, &one, desc_x, &incx,
           (std::size_t)1, (std::size_t)1, (std::size_t)1);

        /* MPFR reference on saved global x (the original RHS) */
        MpfrMatrix A_mpfr(n, n, prec);
        MpfrMatrix b_in_mpfr(n, 1, prec);
        MpfrMatrix x_ref(n, 1, prec);

        custom_to_mpfr_mat(A_mpfr, A_g, n, ctx);
        custom_to_mpfr_mat(b_in_mpfr, x_g, n, ctx);

        mpfr_ptrsv_ref(x_ref, uplo, trans, diag, n, A_mpfr, b_in_mpfr);

        /* Extract local reference and compare */
        MpfrMatrix x_local_ref(loc_rx, std::max(1, loc_cx), prec);
        extract_local_mpfr(x_local_ref, x_ref, n, 1, pc.mb, pc.nb,
                           pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

        ErrorResult err = compute_error_matrix(x_local_ref, x_loc, lld_x, ctx);
        SentinelResult sr = check_matrix_sentinels(x_loc, loc_rx, std::max(1, loc_cx), lld_x,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "uplo=%c trans=%c diag=%c grid=%dx%d",
                          uplo, trans, diag, pc.bc.nprow, pc.bc.npcol);
            report_result("PTRSV", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(x_g);
        std::free(A_loc); std::free(x_loc);
    }}}

    pc.finalize();
}
