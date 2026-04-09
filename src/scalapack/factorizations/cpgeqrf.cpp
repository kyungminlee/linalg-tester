/* cpgeqrf.cpp -- ScaLAPACK CPGEQRF accuracy tester
   (Complex QR factorization, verified via ||A - Q*R|| residual) */

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

/* Fortran-ABI function pointers */
extern "C" typedef void (*pgeqrf_fn_t)(
    const int *m, const int *n, void *a,
    const int *ia, const int *ja, const int *desca,
    void *tau, void *work, const int *lwork, int *info);

extern "C" typedef void (*pungqr_fn_t)(
    const int *m, const int *n, const int *k, void *a,
    const int *ia, const int *ja, const int *desca,
    const void *tau, void *work, const int *lwork, int *info);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_cpgeqrf(const TesterCtx &ctx, void *lib, const char *sym,
                  const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPGEQRF", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pgeqrf_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPGEQRF", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    /* Derive PDUNGQR symbol (e.g. "pcgeqrf_" -> "pcungqr_") */
    std::string ungqr_sym(sym);
    auto pos = ungqr_sym.find("geqrf");
    if (pos != std::string::npos) ungqr_sym.replace(pos, 5, "ungqr");
    auto *fn_ungqr = reinterpret_cast<pungqr_fn_t>(try_load_sym(lib, ungqr_sym.c_str()));

    /* Load PDGEMM (complex variant) for residual computation */
    sc_pgemm_fn_t fn_gemm = load_pgemm(lib, sym);

    if (!fn_ungqr || !fn_gemm) {
        BlacsResult br = {false, -1.0, 0};
        const char *msg = !fn_ungqr ? "error=pungqr_not_found"
                                     : "error=pgemm_not_found";
        report_blacs_result("CPGEQRF", msg, br, format);
        pc.finalize();
        return;
    }

    int m = params.m, n = params.n;
    int mn = std::min(m, n);
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    unsigned seed_A = params.seed;

    /* Generate global complex m-by-n matrix A */
    void *A_g = gen_random_complex_array(m * n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);

    /* MPFR original */
    MpfrComplexMatrix A_orig(m, n, prec);
    custom_to_mpfr_complex_mat(A_orig, A_g, m, ctx);

    /* Local dimensions */
    int loc_m   = pc.local_rows(m);
    int loc_n   = pc.local_cols(n);
    int loc_mn  = pc.local_cols(mn);
    int lld_a   = std::max(1, loc_m);

    /* Allocate local A */
    void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n), ctx.typesize);

    /* Scatter global A to local */
    scatter_global_to_local(A_loc, lld_a, A_g, m,
                            m, n, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);

    /* Create descriptor for A */
    int desc_a[9];
    pc.make_desc(desc_a, m, n, lld_a);

    /* TAU: length LOCc(JA + min(m,n) - 1) */
    int tau_len = pc.local_cols(mn) + pc.nb;
    void *tau = std::calloc(std::max(1, tau_len), ctx.typesize);

    /* Workspace query for PGEQRF */
    int lwork = -1, info = 0;
    void *work_query = std::malloc(ctx.typesize);
    int one = 1;
    fn(&m, &n, A_loc, &one, &one, desc_a, tau, work_query, &lwork, &info);
    lwork = query_lwork(work_query, ctx);
    if (lwork < 1) lwork = 1;
    std::free(work_query);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);

    /* Call PGEQRF */
    info = 0;
    fn(&m, &n, A_loc, &one, &one, desc_a, tau, work, &lwork, &info);
    std::free(work);

    if (info != 0) {
        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "m=%d n=%d grid=%dx%d", m, n, pc.bc.nprow, pc.bc.npcol);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("CPGEQRF", params_str, lr, format);
        }
        std::free(A_g); std::free(A_loc); std::free(tau);
        pc.finalize();
        return;
    }

    /* Save R from the upper triangle of factored A.
       R is min(m,n)-by-n; stored in the upper triangle of A_loc.
       We need to copy it before PDUNGQR overwrites A_loc with Q. */
    int loc_mn_rows = pc.local_rows(mn);
    int lld_r = std::max(1, loc_mn_rows);
    void *R_loc = std::calloc(static_cast<std::size_t>(lld_r) * std::max(1, loc_n), ctx.typesize);

    /* Extract upper triangle: for each local element (il, jl),
       map to global (ig, jg), copy if ig <= jg */
    for (int jl = 0; jl < loc_n; ++jl) {
        int jg = indxl2g(jl, pc.nb, pc.bc.mycol, 0, pc.bc.npcol);
        for (int il = 0; il < loc_mn_rows; ++il) {
            int ig = indxl2g(il, pc.mb, pc.bc.myrow, 0, pc.bc.nprow);
            if (ig <= jg && ig < mn) {
                std::memcpy(
                    static_cast<char *>(R_loc) +
                        (static_cast<std::size_t>(jl) * lld_r + il) * ctx.typesize,
                    static_cast<const char *>(A_loc) +
                        (static_cast<std::size_t>(jl) * lld_a + il) * ctx.typesize,
                    ctx.typesize);
            }
        }
    }

    /* Call PDUNGQR to get Q (m-by-mn) from the factored A_loc.
       Q overwrites A_loc (first mn columns). */
    int lwork_ungqr = -1, info_ungqr = 0;
    work_query = std::malloc(ctx.typesize);
    fn_ungqr(&m, &mn, &mn, A_loc, &one, &one, desc_a,
             tau, work_query, &lwork_ungqr, &info_ungqr);
    lwork_ungqr = query_lwork(work_query, ctx);
    if (lwork_ungqr < 1) lwork_ungqr = 1;
    std::free(work_query);

    work = std::malloc(static_cast<std::size_t>(lwork_ungqr) * ctx.typesize);
    info_ungqr = 0;
    fn_ungqr(&m, &mn, &mn, A_loc, &one, &one, desc_a,
             tau, work, &lwork_ungqr, &info_ungqr);
    std::free(work);
    std::free(tau);

    /* Q is now in A_loc (m-by-mn local portion).
       R is in R_loc (mn-by-n local portion).
       Compute C = Q * R using PDGEMM, compare with original A. */

    /* Allocate local C (m-by-n) */
    int lld_c = std::max(1, loc_m);
    void *C_loc = std::calloc(static_cast<std::size_t>(lld_c) * std::max(1, loc_n), ctx.typesize);

    /* Descriptors: Q is m-by-mn, R is mn-by-n, C is m-by-n */
    int desc_q[9], desc_r[9], desc_c[9];
    pc.make_desc(desc_q, m, mn, lld_a);    /* Q stored in A_loc */
    pc.make_desc(desc_r, mn, n, lld_r);
    pc.make_desc(desc_c, m, n, lld_c);

    /* alpha=1, beta=0: C = 1*Q*R + 0*C */
    MpfrScalar mpfr_one(prec), mpfr_zero(prec);
    mpfr_set_d(mpfr_one.get(), 1.0, MPFR_RNDN);
    mpfr_set_d(mpfr_zero.get(), 0.0, MPFR_RNDN);

    void *alpha = std::malloc(ctx.typesize);
    void *beta  = std::malloc(ctx.typesize);
    ctx.from_mpfr(alpha, mpfr_one.get(), MPFR_RNDN);
    ctx.from_mpfr(beta,  mpfr_zero.get(), MPFR_RNDN);

    char no = 'N';
    fn_gemm(&no, &no, &m, &n, &mn,
            alpha,
            A_loc, &one, &one, desc_q,
            R_loc, &one, &one, desc_r,
            beta,
            C_loc, &one, &one, desc_c,
            (std::size_t)1, (std::size_t)1);

    /* Compare local C_loc vs local A_orig (element-by-element) */
    MpfrComplexMatrix A_local_ref(loc_m, loc_n, prec);
    extract_local_mpfr_complex(A_local_ref, A_orig, m, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

    ErrorResult err = compute_error_complex_matrix(A_local_ref, C_loc, lld_c, ctx);

    /* Residual scaled like LAPACK: max_relative / eps */
    double residual = err.max_relative / eps;

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "m=%d n=%d grid=%dx%d",
                      m, n, pc.bc.nprow, pc.bc.npcol);
        LapackResult lr;
        lr.residual = residual;
        lr.orthogonality = -1.0;  /* Would need Q^H*Q check */
        lr.info = (info != 0) ? info : info_ungqr;
        report_lapack_result("CPGEQRF", params_str, lr, format);
    }

    /* Cleanup */
    std::free(A_g);
    std::free(A_loc);
    std::free(R_loc);
    std::free(C_loc);
    std::free(alpha);
    std::free(beta);

    pc.finalize();
}
