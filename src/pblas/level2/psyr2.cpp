/* psyr2.cpp -- PBLAS Level 2 PSYR2 accuracy tester */

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

/* Fortran-ABI function pointer for PDSYR2 (hidden char length at end) */
extern "C" typedef void (*psyr2_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *x,      const int  *ix,     const int  *jx,     const int  *descx,
    const int  *incx,
    const void *y,      const int  *iy,     const int  *jy,     const int  *descy,
    const int  *incy,
    void       *A,      const int  *ia,     const int  *ja,     const int  *desca,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: A_ref = alpha * (x*y^T + y*x^T) + A_in (full n x n) */
/* ------------------------------------------------------------------ */

static void mpfr_psyr2_ref(MpfrMatrix &A_ref,
                            int n,
                            mpfr_t alpha,
                            const MpfrMatrix &x,
                            const MpfrMatrix &y,
                            const MpfrMatrix &A_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar xy(prec), yx(prec), sum(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            mpfr_mul(xy.get(), x.at(i, 0), y.at(j, 0), MPFR_RNDN);
            mpfr_mul(yx.get(), y.at(i, 0), x.at(j, 0), MPFR_RNDN);
            mpfr_add(sum.get(), xy.get(), yx.get(), MPFR_RNDN);
            mpfr_fma(A_ref.at(i, j), alpha, sum.get(), A_in.at(i, j), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_psyr2(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.n;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYR2", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<psyr2_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYR2", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_x  = params.seed;
        unsigned seed_y  = params.seed + 1;
        unsigned seed_A  = params.seed + 2;
        unsigned seed_al = params.seed + 3;

        /* Generate global data */
        void *x_g   = gen_random_array(n, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
        void *y_g   = gen_random_array(n, ctx.typesize, ctx.from_mpfr, prec, &seed_y);
        void *A_g   = gen_random_array(n * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *alpha = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_al);

        /* Local dimensions for vectors x, y (n x 1) */
        int loc_rx = pc.local_rows(n);
        int loc_cx = pc.local_cols(1);
        int lld_x  = std::max(1, loc_rx);

        int loc_ry = pc.local_rows(n);
        int loc_cy = pc.local_cols(1);
        int lld_y  = std::max(1, loc_ry);

        /* Local dimensions for matrix A (n x n) */
        int loc_rA = pc.local_rows(n);
        int loc_cA = pc.local_cols(n);
        int lld_a  = std::max(1, loc_rA) + params.ld_pad;

        /* Allocate local storage */
        void *x_loc = std::calloc(static_cast<std::size_t>(lld_x) * std::max(1, loc_cx), ctx.typesize);
        void *y_loc = std::calloc(static_cast<std::size_t>(lld_y) * std::max(1, loc_cy), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD000Du;
        void *A_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA),
                                           ctx.typesize, sentinel_seed);

        /* Scatter global to local */
        scatter_global_to_local(x_loc, lld_x, x_g, n,
                                n, 1, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);
        scatter_global_to_local(y_loc, lld_y, y_g, n,
                                n, 1, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);
        scatter_global_to_local(A_loc, lld_a, A_g, n,
                                n, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);

        /* Create descriptors */
        int desc_x[9], desc_y[9], desc_a[9];
        pc.make_desc(desc_x, n, 1, lld_x);
        pc.make_desc(desc_y, n, 1, lld_y);
        pc.make_desc(desc_a, n, n, lld_a);

        /* Call PBLAS */
        int one = 1;
        int incx = 1, incy = 1;
        fn(&uplo, &n,
           alpha,
           x_loc, &one, &one, desc_x, &incx,
           y_loc, &one, &one, desc_y, &incy,
           A_loc, &one, &one, desc_a,
           (std::size_t)1);

        /* MPFR reference on global data */
        MpfrScalar mpfr_alpha(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);

        MpfrMatrix x_mpfr(n, 1, prec);
        MpfrMatrix y_mpfr(n, 1, prec);
        MpfrMatrix A_in_mpfr(n, n, prec);
        MpfrMatrix A_ref(n, n, prec);

        custom_to_mpfr_mat(x_mpfr, x_g, n, ctx);
        custom_to_mpfr_mat(y_mpfr, y_g, n, ctx);
        custom_to_mpfr_mat(A_in_mpfr, A_g, n, ctx);

        mpfr_psyr2_ref(A_ref, n, mpfr_alpha.get(), x_mpfr, y_mpfr, A_in_mpfr);

        /* Build local reference: uplo triangle from A_ref, opposite from A_in */
        MpfrMatrix A_local_ref(loc_rA, loc_cA, prec);
        extract_local_mpfr_sym(A_local_ref, A_ref, A_in_mpfr,
                               n, pc.mb, pc.nb,
                               pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                               uplo);

        ErrorResult err = compute_error_matrix(A_local_ref, A_loc, lld_a, ctx);
        SentinelResult sr = check_matrix_sentinels(A_loc, loc_rA, loc_cA, lld_a,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "uplo=%c grid=%dx%d", uplo,
                          pc.bc.nprow, pc.bc.npcol);
            report_result("PSYR2", params_str, err, &sr, format);
        }

        std::free(x_g); std::free(y_g); std::free(A_g);
        std::free(x_loc); std::free(y_loc); std::free(A_loc);
        std::free(alpha);
    }

    pc.finalize();
}
