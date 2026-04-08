/* pher.cpp -- PBLAS Level 2 PHER accuracy tester (complex-only) */

#include "../pblas.h"
#include "../pblas_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer for PCHER/PZHER (hidden char length at end) */
extern "C" typedef void (*pher_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *x,      const int  *ix,     const int  *jx,     const int  *descx,
    const int  *incx,
    void       *A,      const int  *ia,     const int  *ja,     const int  *desca,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: A_ref[i,j] = alpha * x[i] * conj(x[j]) + A_in[i,j]*/
/*   alpha is REAL. Only the uplo triangle is updated.                 */
/* ------------------------------------------------------------------ */

static void mpfr_pher_ref(MpfrComplexMatrix &A_ref,
                           char uplo, int n,
                           mpfr_t alpha,
                           const MpfrComplexMatrix &x,
                           const MpfrComplexMatrix &A_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrComplexScalar xixjc(prec), tmp(prec), xjc(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            bool in_triangle = (uplo == 'U') ? (i <= j) : (i >= j);
            if (in_triangle) {
                /* xjc = conj(x[j]) */
                mpfr_complex_conj(xjc.re(), xjc.im(),
                                  x.re(j, 0), x.im(j, 0), MPFR_RNDN);
                /* xixjc = x[i] * conj(x[j]) */
                mpfr_complex_mul(xixjc.re(), xixjc.im(),
                                 x.re(i, 0), x.im(i, 0),
                                 xjc.re(), xjc.im(), MPFR_RNDN);
                /* tmp = alpha * xixjc (alpha is real) */
                mpfr_complex_mul_real(tmp.re(), tmp.im(),
                                      xixjc.re(), xixjc.im(),
                                      alpha, MPFR_RNDN);
                /* A_ref[i,j] = tmp + A_in[i,j] */
                mpfr_complex_add(A_ref.re(i, j), A_ref.im(i, j),
                                 tmp.re(), tmp.im(),
                                 A_in.re(i, j), A_in.im(i, j), MPFR_RNDN);
            } else {
                /* Copy original value unchanged */
                mpfr_set(A_ref.re(i, j), A_in.re(i, j), MPFR_RNDN);
                mpfr_set(A_ref.im(i, j), A_in.im(i, j), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pher(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "PHER requires --complex\n");
        return;
    }

    int mb = params.mb > 0 ? params.mb : params.n;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PHER", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pher_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PHER", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n;
    mpfr_prec_t prec = ctx.prec;

    /* alpha is REAL for PHER */
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_x  = params.seed;
        unsigned seed_A  = params.seed + 1;
        unsigned seed_al = params.seed + 2;

        /* Generate global data */
        void *x_g   = gen_random_complex_array(n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
        void *A_g   = gen_hermitian_array(n, uplo, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *alpha = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_al);

        /* Local dimensions for vector x (n x 1) */
        int loc_rx = pc.local_rows(n);
        int loc_cx = pc.local_cols(1);
        int lld_x  = std::max(1, loc_rx);

        /* Local dimensions for matrix A (n x n) */
        int loc_rA = pc.local_rows(n);
        int loc_cA = pc.local_cols(n);
        int lld_a  = std::max(1, loc_rA) + params.ld_pad;

        /* Allocate local storage */
        void *x_loc = std::calloc(static_cast<std::size_t>(lld_x) * std::max(1, loc_cx), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0021u;
        void *A_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA),
                                           ctx.typesize, sentinel_seed);

        /* Scatter global to local */
        scatter_global_to_local(x_loc, lld_x, x_g, n,
                                n, 1, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);
        scatter_global_to_local(A_loc, lld_a, A_g, n,
                                n, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);

        /* Create descriptors */
        int desc_x[9], desc_a[9];
        pc.make_desc(desc_x, n, 1, lld_x);
        pc.make_desc(desc_a, n, n, lld_a);

        /* Call PBLAS */
        int one = 1;
        int incx = 1;
        fn(&uplo, &n,
           alpha,
           x_loc, &one, &one, desc_x, &incx,
           A_loc, &one, &one, desc_a,
           (std::size_t)1);

        /* MPFR reference on global data */
        MpfrScalar mpfr_alpha(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);

        MpfrComplexMatrix x_mpfr(n, 1, prec);
        MpfrComplexMatrix A_in_mpfr(n, n, prec);
        MpfrComplexMatrix A_ref(n, n, prec);

        custom_to_mpfr_complex_mat(x_mpfr, x_g, n, ctx);
        custom_to_mpfr_complex_mat(A_in_mpfr, A_g, n, ctx);

        mpfr_pher_ref(A_ref, uplo, n, mpfr_alpha.get(), x_mpfr, A_in_mpfr);

        /* Build local reference: uplo triangle from A_ref, opposite from A_in */
        MpfrComplexMatrix A_local_ref(loc_rA, loc_cA, prec);
        extract_local_mpfr_complex_herm(A_local_ref, A_ref, A_in_mpfr,
                                         n, pc.mb, pc.nb,
                                         pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                         uplo);

        ErrorResult err = compute_error_complex_matrix(A_local_ref, A_loc, lld_a, ctx);
        SentinelResult sr = check_matrix_sentinels(A_loc, loc_rA, loc_cA, lld_a,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "uplo=%c grid=%dx%d", uplo,
                          pc.bc.nprow, pc.bc.npcol);
            report_result("PHER", params_str, err, &sr, format);
        }

        std::free(x_g); std::free(A_g);
        std::free(x_loc); std::free(A_loc);
        std::free(alpha);
    }

    pc.finalize();
}
