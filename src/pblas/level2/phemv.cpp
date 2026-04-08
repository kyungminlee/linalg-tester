/* phemv.cpp -- PBLAS Level 2 PHEMV accuracy tester (complex-only) */

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

/* Fortran-ABI function pointer for PCHEMV/PZHEMV (hidden char length at end) */
extern "C" typedef void (*phemv_fn_t)(
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
/* MPFR reference: y_ref = alpha * A * x + beta * y_in                 */
/*   A is Hermitian; expand from uplo triangle with conjugation        */
/* ------------------------------------------------------------------ */

static void mpfr_phemv_ref(MpfrComplexMatrix &y_ref,
                            char uplo, int n,
                            const MpfrComplexScalar &alpha,
                            const MpfrComplexMatrix &A,
                            const MpfrComplexMatrix &x,
                            const MpfrComplexScalar &beta,
                            const MpfrComplexMatrix &y_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha.re());
    MpfrComplexScalar acc(prec), tmp(prec), a_ij(prec);

    for (int i = 0; i < n; ++i) {
        mpfr_set_d(acc.re(), 0.0, MPFR_RNDN);
        mpfr_set_d(acc.im(), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j) {
            /* Access A symmetrically with conjugation */
            if ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j)) {
                mpfr_set(a_ij.re(), A.re(i, j), MPFR_RNDN);
                mpfr_set(a_ij.im(), A.im(i, j), MPFR_RNDN);
            } else {
                /* Mirror with conjugation: A(i,j) = conj(A(j,i)) */
                mpfr_complex_conj(a_ij.re(), a_ij.im(),
                                  A.re(j, i), A.im(j, i), MPFR_RNDN);
            }
            mpfr_complex_fma(acc.re(), acc.im(),
                             a_ij.re(), a_ij.im(),
                             x.re(j, 0), x.im(j, 0), MPFR_RNDN);
        }
        /* tmp = alpha * acc */
        mpfr_complex_mul(tmp.re(), tmp.im(),
                         alpha.re(), alpha.im(),
                         acc.re(), acc.im(), MPFR_RNDN);
        /* acc = beta * y_in[i] */
        mpfr_complex_mul(acc.re(), acc.im(),
                         beta.re(), beta.im(),
                         y_in.re(i, 0), y_in.im(i, 0), MPFR_RNDN);
        /* y_ref[i] = tmp + acc */
        mpfr_complex_add(y_ref.re(i, 0), y_ref.im(i, 0),
                         tmp.re(), tmp.im(),
                         acc.re(), acc.im(), MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_phemv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "PHEMV requires --complex\n");
        return;
    }

    int mb = params.mb > 0 ? params.mb : params.n;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PHEMV", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<phemv_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PHEMV", "error=symbol_not_found", br, format);
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
        void *A_g   = gen_hermitian_array(n, uplo, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *x_g   = gen_random_complex_array(n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
        void *y_g   = gen_random_complex_array(n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_y);
        void *alpha = gen_random_complex_array(1, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_ab);
        void *beta  = gen_random_complex_array(1, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_ab);

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

        unsigned sentinel_seed = 0xFBAD001Eu;
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
        MpfrComplexScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr_complex(mpfr_alpha.re(), mpfr_alpha.im(), alpha);
        ctx.to_mpfr_complex(mpfr_beta.re(),  mpfr_beta.im(),  beta);

        MpfrComplexMatrix A_mpfr(n, n, prec);
        MpfrComplexMatrix x_mpfr(n, 1, prec);
        MpfrComplexMatrix y_in_mpfr(n, 1, prec);
        MpfrComplexMatrix y_ref(n, 1, prec);

        custom_to_mpfr_complex_mat(A_mpfr, A_g, n, ctx);
        custom_to_mpfr_complex_mat(x_mpfr, x_g, n, ctx);
        custom_to_mpfr_complex_mat(y_in_mpfr, y_g, n, ctx);

        mpfr_phemv_ref(y_ref, uplo, n, mpfr_alpha, A_mpfr, x_mpfr,
                        mpfr_beta, y_in_mpfr);

        /* Extract local reference and compare */
        MpfrComplexMatrix y_local_ref(loc_ry, std::max(1, loc_cy), prec);
        extract_local_mpfr_complex(y_local_ref, y_ref, n, 1, pc.mb, pc.nb,
                                   pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

        ErrorResult err = compute_error_complex_matrix(y_local_ref, y_loc, lld_y, ctx);
        SentinelResult sr = check_matrix_sentinels(y_loc, loc_ry, std::max(1, loc_cy), lld_y,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "uplo=%c grid=%dx%d", uplo,
                          pc.bc.nprow, pc.bc.npcol);
            report_result("PHEMV", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(x_g); std::free(y_g);
        std::free(A_loc); std::free(x_loc); std::free(y_loc);
        std::free(alpha); std::free(beta);
    }

    pc.finalize();
}
