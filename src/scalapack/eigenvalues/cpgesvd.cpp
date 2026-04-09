/* cpgesvd.cpp -- ScaLAPACK PZGESVD accuracy tester (complex SVD) */

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

extern "C" typedef void (*cpgesvd_fn_t)(
    const char *jobu, const char *jobvt,
    const int *m, const int *n,
    void *a, const int *ia, const int *ja, const int *desca,
    void *s,
    void *u, const int *iu, const int *ju, const int *descu,
    void *vt, const int *ivt, const int *jvt, const int *descvt,
    void *work, const int *lwork, void *rwork,
    int *info,
    std::size_t, std::size_t);

void test_cpgesvd(const TesterCtx &ctx, void *lib, const char *sym,
                  const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPGESVD", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<cpgesvd_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPGESVD", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    /* Load PZGEMM for residual computation */
    sc_pgemm_fn_t pgemm = load_pgemm(lib, sym);
    if (!pgemm) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPGESVD", "error=pgemm_not_found", br, format);
        pc.finalize();
        return;
    }

    int m = params.m, n = params.n;
    int mn = std::min(m, n);
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    std::size_t real_size = ctx.typesize / 2;

    unsigned seed_A = params.seed;

    /* Generate general random complex A (m x n) */
    void *A_g = gen_random_complex_array(m * n, ctx.typesize,
                    ctx.from_mpfr_complex, prec, &seed_A);

    /* MPFR copy for norm and verification */
    MpfrComplexMatrix A_mpfr(m, n, prec);
    custom_to_mpfr_complex_mat(A_mpfr, A_g, m, ctx);
    double norm_A = mpfr_complex_mat_norm1(A_mpfr);

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

    /* Allocate local matrices (complex) */
    void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n_a),
                              ctx.typesize);
    void *A_copy_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n_a),
                                   ctx.typesize);
    void *U_loc = std::calloc(static_cast<std::size_t>(lld_u) * std::max(1, loc_n_u),
                              ctx.typesize);
    void *VT_loc = std::calloc(static_cast<std::size_t>(lld_vt) * std::max(1, loc_n_vt),
                               ctx.typesize);

    /* Singular values (real, replicated, size min(m,n)) */
    void *S_buf = std::calloc(static_cast<std::size_t>(mn), real_size);

    /* RWORK: 1 + 4*max(m,n) real elements */
    std::size_t rwork_size = static_cast<std::size_t>(1 + 4 * std::max(m, n));
    void *rwork_buf = std::calloc(rwork_size, real_size);

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
    char work_query_buf[128];
    fn("V", "V", &m, &n,
       A_loc, &one, &one, descA, S_buf,
       U_loc, &one, &one, descU,
       VT_loc, &one, &one, descVT,
       work_query_buf, &lwork_q, rwork_buf,
       &info,
       (std::size_t)1, (std::size_t)1);
    int lwork = query_lwork_complex(work_query_buf, ctx);

    /* Restore A_loc */
    std::memcpy(A_loc, A_copy_loc, a_loc_bytes);

    /* Call PZGESVD */
    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    info = 0;
    fn("V", "V", &m, &n,
       A_loc, &one, &one, descA, S_buf,
       U_loc, &one, &one, descU,
       VT_loc, &one, &one, descVT,
       work, &lwork, rwork_buf,
       &info,
       (std::size_t)1, (std::size_t)1);
    std::free(work);

    if (info != 0) {
        if (pc.bc.is_root()) {
            char ps[128];
            std::snprintf(ps, sizeof(ps), "m=%d n=%d grid=%dx%d",
                          m, n, pc.bc.nprow, pc.bc.npcol);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("CPGESVD", ps, lr, format);
        }
    } else {
        /* Verification: ||A - U*diag(S)*VT|| / (||A|| * max(m,n) * eps)
           Step 1: Copy first mn columns of U, scale by S (real) values
           Step 2: PZGEMM: C = US * VT
           Step 3: Compare C vs A_copy */

        /* Create descriptor for US (m x mn) */
        int loc_n_us = pc.local_cols(mn);
        int lld_us = std::max(1, loc_m_u);
        int descUS[9];
        pc.make_desc(descUS, m, mn, lld_us);

        /* Copy first mn columns of U into US_loc and scale */
        void *US_loc = std::calloc(static_cast<std::size_t>(lld_us) * std::max(1, loc_n_us),
                                   ctx.typesize);
        for (int jl = 0; jl < loc_n_us; ++jl) {
            int jg = indxl2g(jl, pc.nb, pc.bc.mycol, 0, pc.bc.npcol);
            if (jg < mn) {
                std::memcpy(
                    static_cast<char *>(US_loc) + static_cast<std::size_t>(jl) * lld_us * ctx.typesize,
                    static_cast<const char *>(U_loc) + static_cast<std::size_t>(jl) * lld_u * ctx.typesize,
                    static_cast<std::size_t>(loc_m_u) * ctx.typesize);
            }
        }

        /* Scale US columns by S (real): done in MPFR */
        MpfrComplexMatrix US_mpfr(loc_m_u, loc_n_us, prec);
        custom_to_mpfr_complex_mat(US_mpfr, US_loc, lld_us, ctx);

        MpfrScalar s_val(prec);
        for (int jl = 0; jl < loc_n_us; ++jl) {
            int jg = indxl2g(jl, pc.nb, pc.bc.mycol, 0, pc.bc.npcol);
            if (jg < mn) {
                ctx.to_mpfr(s_val.get(),
                            static_cast<const char *>(S_buf) + jg * real_size);
                for (int il = 0; il < loc_m_u; ++il) {
                    mpfr_mul(US_mpfr.re(il, jl), US_mpfr.re(il, jl),
                             s_val.get(), MPFR_RNDN);
                    mpfr_mul(US_mpfr.im(il, jl), US_mpfr.im(il, jl),
                             s_val.get(), MPFR_RNDN);
                }
            }
        }

        /* Write scaled US back to custom format for PZGEMM */
        for (int jl = 0; jl < loc_n_us; ++jl) {
            for (int il = 0; il < loc_m_u; ++il) {
                ctx.from_mpfr_complex(
                    static_cast<char *>(US_loc) +
                        (static_cast<std::size_t>(jl) * lld_us + il) * ctx.typesize,
                    US_mpfr.re(il, jl), US_mpfr.im(il, jl), MPFR_RNDN);
            }
        }

        /* Allocate C (m x n), same layout as A */
        void *C_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n_a),
                                  ctx.typesize);
        int descC[9];
        pc.make_desc(descC, m, n, lld_a);

        /* PZGEMM: C = 1*US*VT + 0*C */
        MpfrScalar one_re(prec), zero_re(prec);
        mpfr_set_d(one_re.get(), 1.0, MPFR_RNDN);
        mpfr_set_d(zero_re.get(), 0.0, MPFR_RNDN);
        char alpha_buf[128], beta_buf[128];
        ctx.from_mpfr_complex(alpha_buf, one_re.get(), zero_re.get(), MPFR_RNDN);
        ctx.from_mpfr_complex(beta_buf, zero_re.get(), zero_re.get(), MPFR_RNDN);

        pgemm("N", "N", &m, &n, &mn,
              alpha_buf,
              US_loc, &one, &one, descUS,
              VT_loc, &one, &one, descVT,
              beta_buf,
              C_loc, &one, &one, descC,
              (std::size_t)1, (std::size_t)1);

        /* Compare local C vs local A_copy: residual */
        MpfrComplexMatrix C_mpfr(loc_m_a, loc_n_a, prec);
        custom_to_mpfr_complex_mat(C_mpfr, C_loc, lld_a, ctx);

        MpfrComplexMatrix A_loc_mpfr(loc_m_a, loc_n_a, prec);
        custom_to_mpfr_complex_mat(A_loc_mpfr, A_copy_loc, lld_a, ctx);

        MpfrComplexMatrix resid(loc_m_a, loc_n_a, prec);
        mpfr_complex_mat_sub(resid, A_loc_mpfr, C_mpfr);
        double norm_resid = mpfr_complex_mat_norm1(resid);
        double residual = (norm_A > 0.0)
            ? norm_resid / (norm_A * std::max(m, n) * eps) : 0.0;

        if (pc.bc.is_root()) {
            char ps[128];
            std::snprintf(ps, sizeof(ps), "m=%d n=%d grid=%dx%d",
                          m, n, pc.bc.nprow, pc.bc.npcol);
            LapackResult lr = {residual, -1.0, info};
            report_lapack_result("CPGESVD", ps, lr, format);
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
    std::free(rwork_buf);

    pc.finalize();
}
