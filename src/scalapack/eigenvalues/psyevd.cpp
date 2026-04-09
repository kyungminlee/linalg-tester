/* psyevd.cpp -- ScaLAPACK PDSYEVD accuracy tester
   (symmetric eigenvalue, divide & conquer) */

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

extern "C" typedef void (*psyevd_fn_t)(
    const char *jobz, const char *uplo, const int *n,
    void *a, const int *ia, const int *ja, const int *desca,
    void *w,
    void *z, const int *iz, const int *jz, const int *descz,
    void *work, const int *lwork,
    int *iwork, const int *liwork,
    int *info,
    std::size_t, std::size_t);

void test_psyevd(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYEVD", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<psyevd_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYEVD", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    /* Load PDGEMM for residual computation */
    sc_pgemm_fn_t pgemm = load_pgemm(lib, sym);
    if (!pgemm) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYEVD", "error=pgemm_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    unsigned seed_A = params.seed;

    /* Generate symmetric positive definite A */
    void *A_g = gen_positive_definite_array(n, ctx.typesize, ctx.from_mpfr,
                                            ctx.to_mpfr, prec, &seed_A);

    /* MPFR copy for norm */
    MpfrMatrix A_mpfr(n, n, prec);
    custom_to_mpfr_mat(A_mpfr, A_g, n, ctx);
    double norm_A = mpfr_mat_norm1(A_mpfr);

    /* Local dimensions */
    int loc_m = pc.local_rows(n);
    int loc_n = pc.local_cols(n);
    int lld = std::max(1, loc_m);

    std::size_t loc_size = static_cast<std::size_t>(lld) * std::max(1, loc_n) * ctx.typesize;
    void *A_loc      = std::calloc(loc_size, 1);
    void *A_copy_loc = std::calloc(loc_size, 1);
    void *Z_loc      = std::calloc(loc_size, 1);
    void *C_loc      = std::calloc(loc_size, 1);

    void *W_buf = std::calloc(static_cast<std::size_t>(n), ctx.typesize);

    /* Scatter A */
    scatter_global_to_local(A_loc, lld, A_g, n,
                            n, n, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);
    std::memcpy(A_copy_loc, A_loc, loc_size);

    /* Descriptors */
    int descA[9], descA2[9], descZ[9], descC[9];
    pc.make_desc(descA,  n, n, lld);
    pc.make_desc(descA2, n, n, lld);
    pc.make_desc(descZ,  n, n, lld);
    pc.make_desc(descC,  n, n, lld);

    int one = 1, info = 0;

    /* Workspace query: LWORK=-1, LIWORK=-1 */
    int lwork_q = -1, liwork_q = -1;
    void *work_query_buf = std::malloc(ctx.typesize);
    int iwork_query = 0;
    fn("V", "L", &n, A_loc, &one, &one, descA, W_buf,
       Z_loc, &one, &one, descZ,
       work_query_buf, &lwork_q,
       &iwork_query, &liwork_q,
       &info,
       (std::size_t)1, (std::size_t)1);
    int lwork = query_lwork(work_query_buf, ctx);
    int liwork = iwork_query;
    std::free(work_query_buf);

    /* Ensure generous workspace. ScaLAPACK PDSYEVD has known workspace
       underestimation issues. Use a large safe minimum. */
    int min_lwork = std::max(lwork, 1 + 6 * n + 2 * loc_m * loc_n + n * n);
    lwork = min_lwork;
    int min_liwork = std::max(liwork, 7 * n + 8 * pc.bc.npcol + 2);
    liwork = min_liwork;

    /* Restore A_loc */
    std::memcpy(A_loc, A_copy_loc, loc_size);

    /* Allocate workspace */
    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    int *iwork = new int[liwork]();
    info = 0;
    fn("V", "L", &n, A_loc, &one, &one, descA, W_buf,
       Z_loc, &one, &one, descZ,
       work, &lwork,
       iwork, &liwork,
       &info,
       (std::size_t)1, (std::size_t)1);
    std::free(work);
    delete[] iwork;

    if (info != 0) {
        if (pc.bc.is_root()) {
            char ps[128];
            std::snprintf(ps, sizeof(ps), "n=%d grid=%dx%d",
                          n, pc.bc.nprow, pc.bc.npcol);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("PSYEVD", ps, lr, format);
        }
    } else {
        /* Compute C = A_copy * Z via PDGEMM */
        MpfrScalar one_mpfr(prec), zero_mpfr(prec);
        mpfr_set_d(one_mpfr.get(), 1.0, MPFR_RNDN);
        mpfr_set_d(zero_mpfr.get(), 0.0, MPFR_RNDN);
        char alpha_buf[64], beta_buf[64];
        ctx.from_mpfr(alpha_buf, one_mpfr.get(), MPFR_RNDN);
        ctx.from_mpfr(beta_buf, zero_mpfr.get(), MPFR_RNDN);

        pgemm("N", "N", &n, &n, &n,
              alpha_buf,
              A_copy_loc, &one, &one, descA2,
              Z_loc, &one, &one, descZ,
              beta_buf,
              C_loc, &one, &one, descC,
              (std::size_t)1, (std::size_t)1);

        /* Convert local C (AZ) and Z to MPFR */
        MpfrMatrix AZ_mpfr(loc_m, loc_n, prec);
        custom_to_mpfr_mat(AZ_mpfr, C_loc, lld, ctx);

        MpfrMatrix Z_mpfr(loc_m, loc_n, prec);
        custom_to_mpfr_mat(Z_mpfr, Z_loc, lld, ctx);

        /* Scale Z columns by eigenvalues: ZW[i,jl] = Z[i,jl] * W[jg] */
        MpfrScalar w_val(prec);
        for (int jl = 0; jl < loc_n; ++jl) {
            int jg = indxl2g(jl, pc.nb, pc.bc.mycol, 0, pc.bc.npcol);
            ctx.to_mpfr(w_val.get(),
                        static_cast<const char *>(W_buf) + jg * ctx.typesize);
            for (int il = 0; il < loc_m; ++il)
                mpfr_mul(Z_mpfr.at(il, jl), Z_mpfr.at(il, jl),
                         w_val.get(), MPFR_RNDN);
        }

        /* Residual: ||AZ - ZW|| / (||A|| * n * eps) */
        MpfrMatrix resid(loc_m, loc_n, prec);
        mpfr_mat_sub(resid, AZ_mpfr, Z_mpfr);
        double norm_resid = mpfr_mat_norm1(resid);
        double residual = (norm_A > 0.0) ? norm_resid / (norm_A * n * eps) : 0.0;

        if (pc.bc.is_root()) {
            char ps[128];
            std::snprintf(ps, sizeof(ps), "n=%d grid=%dx%d",
                          n, pc.bc.nprow, pc.bc.npcol);
            LapackResult lr = {residual, -1.0, info};
            report_lapack_result("PSYEVD", ps, lr, format);
        }
    }

    /* Cleanup */
    std::free(A_g);
    std::free(A_loc);
    std::free(A_copy_loc);
    std::free(Z_loc);
    std::free(C_loc);
    std::free(W_buf);

    pc.finalize();
}
