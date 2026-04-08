/* psymv.cpp -- PBLAS Level 2 PSYMV accuracy tester */

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

/* Fortran-ABI function pointer for PDSYMV (hidden char length at end) */
extern "C" typedef void (*psymv_fn_t)(
    const char *uplo,
    const int  *n,
    const void *alpha,
    const void *A,      const int  *ia,     const int  *ja,     const int  *desca,
    const void *x,      const int  *ix,     const int  *jx,     const int  *descx,
    const int  *incx,
    const void *beta,
    void       *y,      const int  *iy,     const int  *jy,     const int  *descy,
    const int  *incy,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: y_ref = alpha * A_full * x + beta * y_in            */
/* ------------------------------------------------------------------ */

static void mpfr_psymv_ref(MpfrMatrix &y_ref,
                            char uplo, int n,
                            mpfr_t alpha,
                            const MpfrMatrix &A,
                            const MpfrMatrix &x,
                            mpfr_t beta,
                            const MpfrMatrix &y_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);

    /* Expand symmetric matrix */
    MpfrMatrix Af(n, n, prec);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            if ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j))
                mpfr_set(Af.at(i, j), A.at(i, j), MPFR_RNDN);
            else
                mpfr_set(Af.at(i, j), A.at(j, i), MPFR_RNDN);
        }

    MpfrScalar acc(prec), alpha_acc(prec), beta_y(prec);

    for (int i = 0; i < n; ++i) {
        mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
        for (int p = 0; p < n; ++p)
            mpfr_fma(acc.get(), Af.at(i, p), x.at(p, 0), acc.get(), MPFR_RNDN);
        mpfr_mul(alpha_acc.get(), alpha, acc.get(), MPFR_RNDN);
        mpfr_mul(beta_y.get(), beta, y_in.at(i, 0), MPFR_RNDN);
        mpfr_add(y_ref.at(i, 0), alpha_acc.get(), beta_y.get(), MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_psymv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.n;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYMV", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<psymv_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYMV", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A  = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_y  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        /* Generate global data */
        void *A_g   = gen_symmetric_array(n, uplo, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *x_g   = gen_random_array(n, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
        void *y_g   = gen_random_array(n, ctx.typesize, ctx.from_mpfr, prec, &seed_y);
        void *alpha = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        /* Local dimensions for matrix A (n x n) */
        int loc_rA = pc.local_rows(n);
        int loc_cA = pc.local_cols(n);
        int lld_a  = std::max(1, loc_rA);

        /* Local dimensions for vectors x, y (n x 1) */
        int loc_rx = pc.local_rows(n);
        int loc_cx = pc.local_cols(1);
        int lld_x  = std::max(1, loc_rx);

        int loc_ry = pc.local_rows(n);
        int loc_cy = pc.local_cols(1);
        int lld_y  = std::max(1, loc_ry);

        /* Allocate local storage */
        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA), ctx.typesize);
        void *x_loc = std::calloc(static_cast<std::size_t>(lld_x) * std::max(1, loc_cx), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0008u;
        void *y_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_y) * std::max(1, loc_cy),
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
        scatter_global_to_local(y_loc, lld_y, y_g, n,
                                n, 1, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);

        /* Create descriptors */
        int desc_a[9], desc_x[9], desc_y[9];
        pc.make_desc(desc_a, n, n, lld_a);
        pc.make_desc(desc_x, n, 1, lld_x);
        pc.make_desc(desc_y, n, 1, lld_y);

        /* Call PBLAS */
        int one = 1;
        int incx = 1, incy = 1;
        fn(&uplo, &n,
           alpha,
           A_loc, &one, &one, desc_a,
           x_loc, &one, &one, desc_x, &incx,
           beta,
           y_loc, &one, &one, desc_y, &incy,
           (std::size_t)1);

        /* MPFR reference on global data */
        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_mpfr(n, n, prec);
        MpfrMatrix x_mpfr(n, 1, prec);
        MpfrMatrix y_in_mpfr(n, 1, prec);
        MpfrMatrix y_ref(n, 1, prec);

        custom_to_mpfr_mat(A_mpfr, A_g, n, ctx);
        custom_to_mpfr_mat(x_mpfr, x_g, n, ctx);
        custom_to_mpfr_mat(y_in_mpfr, y_g, n, ctx);

        mpfr_psymv_ref(y_ref, uplo, n,
                        mpfr_alpha.get(), A_mpfr, x_mpfr,
                        mpfr_beta.get(), y_in_mpfr);

        /* Extract local reference and compare */
        MpfrMatrix y_local_ref(loc_ry, std::max(1, loc_cy), prec);
        extract_local_mpfr(y_local_ref, y_ref, n, 1, pc.mb, pc.nb,
                           pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

        ErrorResult err = compute_error_matrix(y_local_ref, y_loc, lld_y, ctx);
        SentinelResult sr = check_matrix_sentinels(y_loc, loc_ry, std::max(1, loc_cy), lld_y,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "uplo=%c grid=%dx%d", uplo,
                          pc.bc.nprow, pc.bc.npcol);
            report_result("PSYMV", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(x_g); std::free(y_g);
        std::free(A_loc); std::free(x_loc); std::free(y_loc);
        std::free(alpha); std::free(beta);
    }

    pc.finalize();
}
