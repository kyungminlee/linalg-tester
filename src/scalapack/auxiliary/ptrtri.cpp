/* ptrtri.cpp -- ScaLAPACK PDTRTRI accuracy tester */

#include "../scalapack.h"
#include "../scalapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*ptrtri_fn_t)(const char *, const char *, const int *,
                                        void *, const int *, const int *, const int *,
                                        int *, std::size_t, std::size_t);

/* ------------------------------------------------------------------ */
/* MPFR triangular inverse: solve A*X = I by triangular substitution   */
/* ------------------------------------------------------------------ */

static void mpfr_tri_inverse(MpfrMatrix &Ainv, const MpfrMatrix &A, char uplo)
{
    int n = A.rows();
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    MpfrScalar tmp(prec);

    /* Initialize Ainv = I */
    mpfr_mat_set_identity(Ainv);

    if (uplo == 'U') {
        /* Solve A*X = I column by column (backward substitution) */
        for (int col = 0; col < n; ++col) {
            for (int k = n - 1; k >= 0; --k) {
                for (int i = k + 1; i < n; ++i) {
                    mpfr_mul(tmp.get(), A.at(k, i), Ainv.at(i, col), MPFR_RNDN);
                    mpfr_sub(Ainv.at(k, col), Ainv.at(k, col), tmp.get(), MPFR_RNDN);
                }
                mpfr_div(Ainv.at(k, col), Ainv.at(k, col), A.at(k, k), MPFR_RNDN);
            }
        }
        /* Zero lower triangle of result */
        for (int j = 0; j < n; ++j)
            for (int i = j + 1; i < n; ++i)
                mpfr_set_d(Ainv.at(i, j), 0.0, MPFR_RNDN);
    } else {
        /* Solve A*X = I column by column (forward substitution) */
        for (int col = 0; col < n; ++col) {
            for (int k = 0; k < n; ++k) {
                for (int i = 0; i < k; ++i) {
                    mpfr_mul(tmp.get(), A.at(k, i), Ainv.at(i, col), MPFR_RNDN);
                    mpfr_sub(Ainv.at(k, col), Ainv.at(k, col), tmp.get(), MPFR_RNDN);
                }
                mpfr_div(Ainv.at(k, col), Ainv.at(k, col), A.at(k, k), MPFR_RNDN);
            }
        }
        /* Zero upper triangle of result */
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < j; ++i)
                mpfr_set_d(Ainv.at(i, j), 0.0, MPFR_RNDN);
    }
}

void test_ptrtri(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PTRTRI", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<ptrtri_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PTRTRI", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.m;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;
        char diag = 'N';

        /* Generate triangular A */
        void *A_g = gen_triangular_array(n, uplo, diag, ctx.typesize,
                                          ctx.from_mpfr, prec, &seed_A);

        /* Local dimensions */
        int loc_rA = pc.local_rows(n);
        int loc_cA = pc.local_cols(n);
        int lld_a = std::max(1, loc_rA);

        /* Allocate and scatter */
        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA), ctx.typesize);
        scatter_global_to_local(A_loc, lld_a, A_g, n,
                                n, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);

        /* Descriptor */
        int desc_a[9];
        pc.make_desc(desc_a, n, n, lld_a);

        /* Call PDTRTRI */
        int one = 1;
        int info = 0;
        fn(&uplo, &diag, &n, A_loc, &one, &one, desc_a, &info,
           (std::size_t)1, (std::size_t)1);

        /* MPFR reference: compute triangular inverse */
        MpfrMatrix A_mpfr(n, n, prec);
        custom_to_mpfr_mat(A_mpfr, A_g, n, ctx);

        MpfrMatrix Ainv_ref(n, n, prec);
        mpfr_tri_inverse(Ainv_ref, A_mpfr, uplo);

        /* Extract local portion of reference and compare */
        MpfrMatrix Ainv_local_ref(loc_rA, loc_cA, prec);
        extract_local_mpfr(Ainv_local_ref, Ainv_ref, n, n, pc.mb, pc.nb,
                           pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

        ErrorResult err = compute_error_matrix(Ainv_local_ref, A_loc, lld_a, ctx);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "uplo=%c diag=%c n=%d grid=%dx%d", uplo, diag, n,
                          pc.bc.nprow, pc.bc.npcol);
            LapackResult lr = {err.max_relative / eps, -1.0, 0};
            report_lapack_result("PTRTRI", params_str, lr, format);
        }

        std::free(A_g);
        std::free(A_loc);
    }

    pc.finalize();
}
