/* pgesvd.cpp -- ScaLAPACK PDGESVD accuracy tester (SVD) */

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

extern "C" typedef void (*pgesvd_fn_t)(
    const char *jobu, const char *jobvt,
    const int *m, const int *n,
    void *a, const int *ia, const int *ja, const int *desca,
    void *s,
    void *u, const int *iu, const int *ju, const int *descu,
    void *vt, const int *ivt, const int *jvt, const int *descvt,
    void *work, const int *lwork, int *info,
    std::size_t, std::size_t);

void test_pgesvd(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PGESVD", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pgesvd_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PGESVD", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    /* Load PDGEMM for residual computation */
    sc_pgemm_fn_t pgemm = load_pgemm(lib, sym);
    if (!pgemm) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PGESVD", "error=pgemm_not_found", br, format);
        pc.finalize();
        return;
    }

    int m = params.m, n = params.n;
    int mn = std::min(m, n);
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    unsigned seed_A = params.seed;

    /* Generate general random A (m x n) */
    void *A_g = gen_random_array(m * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);

    /* MPFR copy for norm and verification */
    MpfrMatrix A_mpfr(m, n, prec);
    custom_to_mpfr_mat(A_mpfr, A_g, m, ctx);
    double norm_A = mpfr_mat_norm1(A_mpfr);

    /* Local dimensions for A (m x n) */
    int loc_m_a = pc.local_rows(m);
    int loc_n_a = pc.local_cols(n);
    int lld_a = std::max(1, loc_m_a);

    /* Local dimensions for U (m x m) */
    int loc_m_u = pc.local_rows(m);
    int loc_n_u = pc.local_cols(m);
    int lld_u = std::max(1, loc_m_u);

    /* Local dimensions for VT (n x n) */
    int loc_m_vt = pc.local_rows(n);
    int loc_n_vt = pc.local_cols(n);
    int lld_vt = std::max(1, loc_m_vt);

    /* Allocate local matrices */
    void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n_a),
                              ctx.typesize);
    void *A_copy_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n_a),
                                   ctx.typesize);
    void *U_loc = std::calloc(static_cast<std::size_t>(lld_u) * std::max(1, loc_n_u),
                              ctx.typesize);
    void *VT_loc = std::calloc(static_cast<std::size_t>(lld_vt) * std::max(1, loc_n_vt),
                               ctx.typesize);

    /* Singular values (replicated, size min(m,n)) */
    void *S_buf = std::calloc(static_cast<std::size_t>(mn), ctx.typesize);

    /* Scatter A */
    scatter_global_to_local(A_loc, lld_a, A_g, m,
                            m, n, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);
    std::size_t a_loc_bytes = static_cast<std::size_t>(lld_a) * std::max(1, loc_n_a) * ctx.typesize;
    std::memcpy(A_copy_loc, A_loc, a_loc_bytes);

    /* Descriptors */
    int descA[9], descA2[9], descU[9], descVT[9];
    pc.make_desc(descA,  m, n, lld_a);
    pc.make_desc(descA2, m, n, lld_a);
    pc.make_desc(descU,  m, m, lld_u);
    pc.make_desc(descVT, n, n, lld_vt);

    int one = 1, info = 0;

    /* Workspace query */
    int lwork_q = -1;
    void *work_query_buf = std::malloc(ctx.typesize);
    fn("V", "V", &m, &n,
       A_loc, &one, &one, descA, S_buf,
       U_loc, &one, &one, descU,
       VT_loc, &one, &one, descVT,
       work_query_buf, &lwork_q, &info,
       (std::size_t)1, (std::size_t)1);
    int lwork = query_lwork(work_query_buf, ctx);
    std::free(work_query_buf);

    /* Restore A_loc */
    std::memcpy(A_loc, A_copy_loc, a_loc_bytes);

    /* Call PDGESVD */
    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    info = 0;
    fn("V", "V", &m, &n,
       A_loc, &one, &one, descA, S_buf,
       U_loc, &one, &one, descU,
       VT_loc, &one, &one, descVT,
       work, &lwork, &info,
       (std::size_t)1, (std::size_t)1);
    std::free(work);

    if (info != 0) {
        if (pc.bc.is_root()) {
            char ps[128];
            std::snprintf(ps, sizeof(ps), "m=%d n=%d grid=%dx%d",
                          m, n, pc.bc.nprow, pc.bc.npcol);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("PGESVD", ps, lr, format);
        }
    } else {
        /* Verification: ||A - U*diag(S)*VT|| / (||A|| * max(m,n) * eps)
           Step 1: Scale U columns by S values -> US
           Step 2: PDGEMM: C = US * VT
           Step 3: Compare C vs A_copy */

        /* US: local dimensions m x mn.
           We need a distributed US matrix. For simplicity, modify U_loc in-place
           (scale first mn columns by S values).
           Then use PDGEMM: C(m,n) = U_scaled(m,m) * VT(n,n)
           But dimensions don't match for general GEMM...

           Actually: A = U(:,1:mn) * diag(S) * VT(1:mn,:)
           So we need US(m, mn) * VT_top(mn, n).

           For PDGEMM with ScaLAPACK, we can use sub-matrix indexing.
           US: first mn columns of U, scaled by S.
           PDGEMM('N', 'N', m, n, mn, 1, US, 1, 1, descUS, VT, 1, 1, descVT, 0, C, 1, 1, descC)
        */

        /* Create descriptor for US (m x mn) */
        int loc_n_us = pc.local_cols(mn);
        int lld_us = std::max(1, loc_m_u);
        int descUS[9];
        pc.make_desc(descUS, m, mn, lld_us);

        /* Copy first mn columns of U into US_loc and scale */
        void *US_loc = std::calloc(static_cast<std::size_t>(lld_us) * std::max(1, loc_n_us),
                                   ctx.typesize);
        /* Extract first mn columns of U_loc (same local storage layout) */
        for (int jl = 0; jl < loc_n_us; ++jl) {
            int jg = indxl2g(jl, pc.nb, pc.bc.mycol, 0, pc.bc.npcol);
            if (jg < mn) {
                std::memcpy(
                    static_cast<char *>(US_loc) + static_cast<std::size_t>(jl) * lld_us * ctx.typesize,
                    static_cast<const char *>(U_loc) + static_cast<std::size_t>(jl) * lld_u * ctx.typesize,
                    static_cast<std::size_t>(loc_m_u) * ctx.typesize);
            }
        }

        /* Scale US columns by S: done in MPFR */
        MpfrMatrix US_mpfr(loc_m_u, loc_n_us, prec);
        custom_to_mpfr_mat(US_mpfr, US_loc, lld_us, ctx);

        MpfrScalar s_val(prec);
        for (int jl = 0; jl < loc_n_us; ++jl) {
            int jg = indxl2g(jl, pc.nb, pc.bc.mycol, 0, pc.bc.npcol);
            if (jg < mn) {
                ctx.to_mpfr(s_val.get(),
                            static_cast<const char *>(S_buf) + jg * ctx.typesize);
                for (int il = 0; il < loc_m_u; ++il)
                    mpfr_mul(US_mpfr.at(il, jl), US_mpfr.at(il, jl),
                             s_val.get(), MPFR_RNDN);
            }
        }

        /* Write scaled US back to custom format for PDGEMM */
        for (int jl = 0; jl < loc_n_us; ++jl) {
            for (int il = 0; il < loc_m_u; ++il) {
                ctx.from_mpfr(
                    static_cast<char *>(US_loc) +
                        (static_cast<std::size_t>(jl) * lld_us + il) * ctx.typesize,
                    US_mpfr.at(il, jl), MPFR_RNDN);
            }
        }

        /* Allocate C (m x n), same layout as A */
        void *C_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n_a),
                                  ctx.typesize);
        int descC[9];
        pc.make_desc(descC, m, n, lld_a);

        /* PDGEMM: C = 1*US*VT + 0*C */
        MpfrScalar one_mpfr(prec), zero_mpfr(prec);
        mpfr_set_d(one_mpfr.get(), 1.0, MPFR_RNDN);
        mpfr_set_d(zero_mpfr.get(), 0.0, MPFR_RNDN);
        char alpha_buf[64], beta_buf[64];
        ctx.from_mpfr(alpha_buf, one_mpfr.get(), MPFR_RNDN);
        ctx.from_mpfr(beta_buf, zero_mpfr.get(), MPFR_RNDN);

        pgemm("N", "N", &m, &n, &mn,
              alpha_buf,
              US_loc, &one, &one, descUS,
              VT_loc, &one, &one, descVT,
              beta_buf,
              C_loc, &one, &one, descC,
              (std::size_t)1, (std::size_t)1);

        /* Compare local C vs local A_copy: residual */
        MpfrMatrix C_mpfr(loc_m_a, loc_n_a, prec);
        custom_to_mpfr_mat(C_mpfr, C_loc, lld_a, ctx);

        MpfrMatrix A_loc_mpfr(loc_m_a, loc_n_a, prec);
        custom_to_mpfr_mat(A_loc_mpfr, A_copy_loc, lld_a, ctx);

        MpfrMatrix resid(loc_m_a, loc_n_a, prec);
        mpfr_mat_sub(resid, A_loc_mpfr, C_mpfr);
        double norm_resid = mpfr_mat_norm1(resid);
        double residual = (norm_A > 0.0)
            ? norm_resid / (norm_A * std::max(m, n) * eps) : 0.0;

        if (pc.bc.is_root()) {
            char ps[128];
            std::snprintf(ps, sizeof(ps), "m=%d n=%d grid=%dx%d",
                          m, n, pc.bc.nprow, pc.bc.npcol);
            LapackResult lr = {residual, -1.0, info};
            report_lapack_result("PGESVD", ps, lr, format);
        }

        std::free(US_loc);
        std::free(C_loc);
    }

    /* Cleanup */
    std::free(A_g);
    std::free(A_loc);
    std::free(A_copy_loc);
    std::free(U_loc);
    std::free(VT_loc);
    std::free(S_buf);

    pc.finalize();
}
