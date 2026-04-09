/* pgetrf.cpp -- ScaLAPACK PGETRF accuracy tester
   (LU factorization with partial pivoting, verified via solve) */

#include "../scalapack.h"
#include "../scalapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/generators.h"
#include "../../core/report.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

/* Fortran-ABI function pointers */
extern "C" typedef void (*pgetrf_fn_t)(
    const int *m, const int *n, void *a,
    const int *ia, const int *ja, const int *desca,
    int *ipiv, int *info);

extern "C" typedef void (*pgetrs_fn_t)(
    const char *trans, const int *n, const int *nrhs,
    const void *a, const int *ia, const int *ja, const int *desca,
    const int *ipiv,
    void *b, const int *ib, const int *jb, const int *descb,
    int *info, std::size_t);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pgetrf(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PGETRF", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pgetrf_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PGETRF", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    /* Derive PGETRS symbol from PGETRF symbol */
    std::string getrs_sym(sym);
    auto pos = getrs_sym.find("getrf");
    if (pos != std::string::npos) getrs_sym.replace(pos, 5, "getrs");
    auto *fn_getrs = reinterpret_cast<pgetrs_fn_t>(try_load_sym(lib, getrs_sym.c_str()));
    if (!fn_getrs) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PGETRF", "error=pgetrs_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n;
    int nrhs = std::min(params.k, 4);
    if (nrhs < 1) nrhs = 1;
    mpfr_prec_t prec = ctx.prec;


    unsigned seed_A = params.seed;
    unsigned seed_B = params.seed + 1;

    /* Generate global matrices: A(n x n), B(n x nrhs) */
    void *A_g = gen_random_array(n * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
    void *B_g = gen_random_array(n * nrhs, ctx.typesize, ctx.from_mpfr, prec, &seed_B);

    /* MPFR originals */
    MpfrMatrix A_orig(n, n, prec);
    MpfrMatrix B_orig(n, nrhs, prec);
    custom_to_mpfr_mat(A_orig, A_g, n, ctx);
    custom_to_mpfr_mat(B_orig, B_g, n, ctx);

    /* Local dimensions */
    int loc_m  = pc.local_rows(n);
    int loc_n  = pc.local_cols(n);
    int loc_nrhs = pc.local_cols(nrhs);
    int lld_a  = std::max(1, loc_m);
    int lld_b  = std::max(1, loc_m);

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

    /* Call PGETRF */
    int one = 1, info = 0;
    fn(&n, &n, A_loc, &one, &one, desc_a, ipiv, &info);

    int info_getrs = 0;
    if (info == 0) {
        /* Call PGETRS: solve A*X = B using factored A */
        char trans = 'N';
        fn_getrs(&trans, &n, &nrhs,
                 A_loc, &one, &one, desc_a, ipiv,
                 B_loc, &one, &one, desc_b,
                 &info_getrs, (std::size_t)1);
    }

    /* Compute solve residual ||AX-B||/(||A||*||X||*n*eps) */
    double residual = scalapack_gather_solve_residual(A_orig, B_orig, B_loc, lld_b,
                                                       n, nrhs, pc, ctx);

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "n=%d nrhs=%d grid=%dx%d",
                      n, nrhs, pc.bc.nprow, pc.bc.npcol);
        LapackResult lr;
        lr.residual = residual;
        lr.orthogonality = -1.0;
        lr.info = (info != 0) ? info : info_getrs;
        report_lapack_result("PGETRF", params_str, lr, format);
    }

    /* Cleanup */
    std::free(A_g);
    std::free(B_g);
    std::free(A_loc);
    std::free(B_loc);
    delete[] ipiv;

    pc.finalize();
}
