/* cpgesv.cpp -- ScaLAPACK CPGESV accuracy tester
   (Complex general linear solve via LU factorization) */

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
#include <string>

/* Fortran-ABI function pointer */
extern "C" typedef void (*pgesv_fn_t)(
    const int *n, const int *nrhs,
    void *a, const int *ia, const int *ja, const int *desca,
    int *ipiv,
    void *b, const int *ib, const int *jb, const int *descb,
    int *info);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_cpgesv(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPGESV", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pgesv_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPGESV", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n;
    int nrhs = std::min(params.k, 4);
    if (nrhs < 1) nrhs = 1;
    mpfr_prec_t prec = ctx.prec;


    unsigned seed_A = params.seed;
    unsigned seed_B = params.seed + 1;

    /* Generate global complex matrices: A(n x n), B(n x nrhs) */
    void *A_g = gen_random_complex_array(n * n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
    void *B_g = gen_random_complex_array(n * nrhs, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_B);

    /* MPFR originals */
    MpfrComplexMatrix A_orig(n, n, prec);
    MpfrComplexMatrix B_orig(n, nrhs, prec);
    custom_to_mpfr_complex_mat(A_orig, A_g, n, ctx);
    custom_to_mpfr_complex_mat(B_orig, B_g, n, ctx);

    /* Local dimensions */
    int loc_m    = pc.local_rows(n);
    int loc_n    = pc.local_cols(n);
    int loc_nrhs = pc.local_cols(nrhs);
    int lld_a    = std::max(1, loc_m);
    int lld_b    = std::max(1, loc_m);

    /* Allocate local matrices */
    void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n), ctx.typesize);
    void *B_loc = std::calloc(static_cast<std::size_t>(lld_b) * std::max(1, loc_nrhs), ctx.typesize);

    /* Scatter global to local */
    scatter_global_to_local(A_loc, lld_a, A_g, n,
                            n, n, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);
    scatter_global_to_local(B_loc, lld_b, B_g, n,
                            n, nrhs, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);

    /* Create descriptors */
    int desc_a[9], desc_b[9];
    pc.make_desc(desc_a, n, n, lld_a);
    pc.make_desc(desc_b, n, nrhs, lld_b);

    /* Distributed pivot array */
    int ipiv_len = loc_m + pc.mb;
    int *ipiv = new int[ipiv_len]();

    /* Call PGESV: factor A and solve A*X = B in one call */
    int one = 1, info = 0;
    fn(&n, &nrhs,
       A_loc, &one, &one, desc_a, ipiv,
       B_loc, &one, &one, desc_b,
       &info);

    /* Compute solve residual ||AX-B||/(||A||*||X||*n*eps) */
    double residual = scalapack_gather_solve_residual_complex(A_orig, B_orig, B_loc, lld_b,
                                                               n, nrhs, pc, ctx);

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "n=%d nrhs=%d grid=%dx%d",
                      n, nrhs, pc.bc.nprow, pc.bc.npcol);
        LapackResult lr;
        lr.residual = residual;
        lr.orthogonality = -1.0;
        lr.info = info;
        report_lapack_result("CPGESV", params_str, lr, format);
    }

    /* Cleanup */
    std::free(A_g);
    std::free(B_g);
    std::free(A_loc);
    std::free(B_loc);
    delete[] ipiv;

    pc.finalize();
}
