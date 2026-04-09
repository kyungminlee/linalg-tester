/* cppotrs.cpp -- ScaLAPACK CPPOTRS (complex PPOTRS) accuracy tester */

#include "../scalapack.h"
#include "../scalapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*ppotrf_fn_t)(const char *, const int *,
                                        void *, const int *, const int *, const int *,
                                        int *, std::size_t);
extern "C" typedef void (*ppotrs_fn_t)(const char *, const int *, const int *,
                                        const void *, const int *, const int *, const int *,
                                        void *, const int *, const int *, const int *,
                                        int *, std::size_t);

void test_cppotrs(const TesterCtx &ctx, void *lib, const char *sym,
                  const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPPOTRS", "error=init_failed", br, format);
        return;
    }

    /* Load PZPOTRS */
    auto *fn = reinterpret_cast<ppotrs_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPPOTRS", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    /* Derive PZPOTRF symbol */
    std::string trf_sym(sym);
    auto pos = trf_sym.find("potrs");
    if (pos != std::string::npos) trf_sym.replace(pos, 5, "potrf");
    auto *fn_trf = reinterpret_cast<ppotrf_fn_t>(try_load_sym(lib, trf_sym.c_str()));
    if (!fn_trf) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPPOTRS", "error=potrf_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.m;
    int nrhs = params.n;
    mpfr_prec_t prec = ctx.prec;


    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;
        unsigned seed_B = params.seed + 1;

        /* Generate HPD A and random complex B */
        void *A_g = gen_hermitian_positive_definite_array(n, ctx.typesize,
                        ctx.from_mpfr_complex, ctx.to_mpfr_complex, prec, &seed_A);
        void *B_g = gen_random_complex_array(n * nrhs, ctx.typesize,
                        ctx.from_mpfr_complex, prec, &seed_B);

        /* Local dimensions */
        int loc_rA = pc.local_rows(n);
        int loc_cA = pc.local_cols(n);
        int loc_rB = pc.local_rows(n);
        int loc_cB = pc.local_cols(nrhs);

        int lld_a = std::max(1, loc_rA);
        int lld_b = std::max(1, loc_rB);

        /* Allocate local */
        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA), ctx.typesize);
        void *B_loc = std::calloc(static_cast<std::size_t>(lld_b) * std::max(1, loc_cB), ctx.typesize);

        /* Scatter */
        scatter_global_to_local(A_loc, lld_a, A_g, n,
                                n, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);
        scatter_global_to_local(B_loc, lld_b, B_g, n,
                                n, nrhs, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);

        /* Descriptors */
        int desc_a[9], desc_b[9];
        pc.make_desc(desc_a, n, n, lld_a);
        pc.make_desc(desc_b, n, nrhs, lld_b);

        /* Call PZPOTRF */
        int one = 1;
        int info = 0;
        fn_trf(&uplo, &n, A_loc, &one, &one, desc_a, &info, (std::size_t)1);

        /* Call PZPOTRS */
        fn(&uplo, &n, &nrhs, A_loc, &one, &one, desc_a,
           B_loc, &one, &one, desc_b, &info, (std::size_t)1);

        /* MPFR originals for residual computation */
        MpfrComplexMatrix A_orig(n, n, prec);
        MpfrComplexMatrix B_orig(n, nrhs, prec);

        custom_to_mpfr_complex_mat(A_orig, A_g, n, ctx);
        custom_to_mpfr_complex_mat(B_orig, B_g, n, ctx);

        /* Compute solve residual ||AX-B||/(||A||*||X||*n*eps) */
        double residual = scalapack_gather_solve_residual_complex(A_orig, B_orig, B_loc, lld_b, n, nrhs, pc, ctx);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "uplo=%c n=%d nrhs=%d grid=%dx%d", uplo, n, nrhs,
                          pc.bc.nprow, pc.bc.npcol);
            LapackResult lr = {residual, -1.0, 0};
            report_lapack_result("CPPOTRS", params_str, lr, format);
        }

        std::free(A_g); std::free(B_g);
        std::free(A_loc); std::free(B_loc);
    }

    pc.finalize();
}
