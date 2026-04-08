/* pgerc.cpp -- PBLAS Level 2 PGERC accuracy tester (complex-only) */

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

/* Fortran-ABI function pointer for PCGERC/PZGERC (no char params, no hidden lengths) */
extern "C" typedef void (*pgerc_fn_t)(
    const int  *m,      const int  *n,
    const void *alpha,
    const void *x,      const int  *ix,     const int  *jx,     const int  *descx,
    const int  *incx,
    const void *y,      const int  *iy,     const int  *jy,     const int  *descy,
    const int  *incy,
    void       *A,      const int  *ia,     const int  *ja,     const int  *desca
);

/* ------------------------------------------------------------------ */
/* MPFR reference: A_ref = alpha * x * conj(y)^T + A_in                */
/* ------------------------------------------------------------------ */

static void mpfr_pgerc_ref(MpfrComplexMatrix &A_ref,
                            int m, int n,
                            const MpfrComplexScalar &alpha,
                            const MpfrComplexMatrix &x,
                            const MpfrComplexMatrix &y,
                            const MpfrComplexMatrix &A_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha.re());
    MpfrComplexScalar tmp(prec), yc(prec);

    for (int j = 0; j < n; ++j) {
        /* Conjugate y[j] once per column */
        mpfr_complex_conj(yc.re(), yc.im(),
                          y.re(j, 0), y.im(j, 0), MPFR_RNDN);
        for (int i = 0; i < m; ++i) {
            /* tmp = x[i] * conj(y[j]) */
            mpfr_complex_mul(tmp.re(), tmp.im(),
                             x.re(i, 0), x.im(i, 0),
                             yc.re(), yc.im(), MPFR_RNDN);
            /* A_ref[i,j] = alpha * tmp + A_in[i,j] */
            mpfr_complex_mul(tmp.re(), tmp.im(),
                             alpha.re(), alpha.im(),
                             tmp.re(), tmp.im(), MPFR_RNDN);
            mpfr_complex_add(A_ref.re(i, j), A_ref.im(i, j),
                             tmp.re(), tmp.im(),
                             A_in.re(i, j), A_in.im(i, j), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pgerc(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "PGERC requires --complex\n");
        return;
    }

    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PGERC", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pgerc_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PGERC", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int m = params.m, n = params.n;
    mpfr_prec_t prec = ctx.prec;

    {
        unsigned seed_x  = params.seed;
        unsigned seed_y  = params.seed + 1;
        unsigned seed_A  = params.seed + 2;
        unsigned seed_al = params.seed + 3;

        /* Generate global data */
        void *x_g   = gen_random_complex_array(m, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
        void *y_g   = gen_random_complex_array(n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_y);
        void *A_g   = gen_random_complex_array(m * n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *alpha = gen_random_complex_array(1, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_al);

        /* Local dimensions for vectors */
        int loc_rx = pc.local_rows(m);
        int loc_cx = pc.local_cols(1);
        int lld_x  = std::max(1, loc_rx);

        int loc_ry = pc.local_rows(n);
        int loc_cy = pc.local_cols(1);
        int lld_y  = std::max(1, loc_ry);

        /* Local dimensions for matrix A (m x n) */
        int loc_rA = pc.local_rows(m);
        int loc_cA = pc.local_cols(n);
        int lld_a  = std::max(1, loc_rA) + params.ld_pad;

        /* Allocate local storage */
        void *x_loc = std::calloc(static_cast<std::size_t>(lld_x) * std::max(1, loc_cx), ctx.typesize);
        void *y_loc = std::calloc(static_cast<std::size_t>(lld_y) * std::max(1, loc_cy), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0020u;
        void *A_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA),
                                           ctx.typesize, sentinel_seed);

        /* Scatter global to local */
        scatter_global_to_local(x_loc, lld_x, x_g, m,
                                m, 1, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);
        scatter_global_to_local(y_loc, lld_y, y_g, n,
                                n, 1, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);
        scatter_global_to_local(A_loc, lld_a, A_g, m,
                                m, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);

        /* Create descriptors */
        int desc_x[9], desc_y[9], desc_a[9];
        pc.make_desc(desc_x, m, 1, lld_x);
        pc.make_desc(desc_y, n, 1, lld_y);
        pc.make_desc(desc_a, m, n, lld_a);

        /* Call PBLAS */
        int one = 1;
        int incx = 1, incy = 1;
        fn(&m, &n,
           alpha,
           x_loc, &one, &one, desc_x, &incx,
           y_loc, &one, &one, desc_y, &incy,
           A_loc, &one, &one, desc_a);

        /* MPFR reference on global data */
        MpfrComplexScalar mpfr_alpha(prec);
        ctx.to_mpfr_complex(mpfr_alpha.re(), mpfr_alpha.im(), alpha);

        MpfrComplexMatrix x_mpfr(m, 1, prec);
        MpfrComplexMatrix y_mpfr(n, 1, prec);
        MpfrComplexMatrix A_in_mpfr(m, n, prec);
        MpfrComplexMatrix A_ref(m, n, prec);

        custom_to_mpfr_complex_mat(x_mpfr, x_g, m, ctx);
        custom_to_mpfr_complex_mat(y_mpfr, y_g, n, ctx);
        custom_to_mpfr_complex_mat(A_in_mpfr, A_g, m, ctx);

        mpfr_pgerc_ref(A_ref, m, n, mpfr_alpha, x_mpfr, y_mpfr, A_in_mpfr);

        /* Extract local reference and compare */
        MpfrComplexMatrix A_local_ref(loc_rA, loc_cA, prec);
        extract_local_mpfr_complex(A_local_ref, A_ref, m, n, pc.mb, pc.nb,
                                   pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

        ErrorResult err = compute_error_complex_matrix(A_local_ref, A_loc, lld_a, ctx);
        SentinelResult sr = check_matrix_sentinels(A_loc, loc_rA, loc_cA, lld_a,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "grid=%dx%d", pc.bc.nprow, pc.bc.npcol);
            report_result("PGERC", params_str, err, &sr, format);
        }

        std::free(x_g); std::free(y_g); std::free(A_g);
        std::free(x_loc); std::free(y_loc); std::free(A_loc);
        std::free(alpha);
    }

    pc.finalize();
}
