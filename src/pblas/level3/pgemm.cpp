/* pgemm.cpp -- PBLAS Level 3 PGEMM accuracy tester */

#include "../pblas.h"
#include "../pblas_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer for PDGEMM (hidden char lengths at the end) */
extern "C" typedef void (*pgemm_fn_t)(
    const char *transa, const char *transb,
    const int  *m,      const int  *n,      const int  *k,
    const void *alpha,
    const void *A,      const int  *ia,     const int  *ja,     const int  *desca,
    const void *B,      const int  *ib,     const int  *jb,     const int  *descb,
    const void *beta,
    void       *C,      const int  *ic,     const int  *jc,     const int  *descc,
    std::size_t transa_len, std::size_t transb_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: C_ref = alpha * op(A) * op(B) + beta * C_in        */
/* ------------------------------------------------------------------ */

static void mpfr_pgemm_ref(MpfrMatrix &C_ref,
                            char transa, char transb,
                            int m, int n, int k,
                            mpfr_t alpha,
                            const MpfrMatrix &A,
                            const MpfrMatrix &B,
                            mpfr_t beta,
                            const MpfrMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar acc(prec), alpha_acc(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
            for (int p = 0; p < k; ++p) {
                const mpfr_t &a_ip = (transa == 'N')
                    ? A.at(i, p) : A.at(p, i);
                const mpfr_t &b_pj = (transb == 'N')
                    ? B.at(p, j) : B.at(j, p);
                mpfr_fma(acc.get(), a_ip, b_pj, acc.get(), MPFR_RNDN);
            }
            mpfr_mul(alpha_acc.get(), alpha, acc.get(), MPFR_RNDN);
            mpfr_mul(beta_c.get(),   beta,  C_in.at(i, j), MPFR_RNDN);
            mpfr_add(C_ref.at(i, j), alpha_acc.get(), beta_c.get(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pgemm(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PGEMM", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pgemm_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PGEMM", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int m = params.m, n = params.n, k = params.k;
    mpfr_prec_t prec = ctx.prec;

    for (char ta : {'N', 'T'}) {
    for (char tb : {'N', 'T'}) {
        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        int rows_A = (ta == 'N') ? m : k;
        int cols_A = (ta == 'N') ? k : m;
        int rows_B = (tb == 'N') ? k : n;
        int cols_B = (tb == 'N') ? n : k;

        /* Generate global matrices */
        void *A_g = gen_random_array(rows_A * cols_A, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *B_g = gen_random_array(rows_B * cols_B, ctx.typesize, ctx.from_mpfr, prec, &seed_B);
        void *C_g = gen_random_array(m * n,           ctx.typesize, ctx.from_mpfr, prec, &seed_C);
        void *alpha = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        /* Local dimensions */
        int loc_rA = pc.local_rows(rows_A);
        int loc_cA = pc.local_cols(cols_A);
        int loc_rB = pc.local_rows(rows_B);
        int loc_cB = pc.local_cols(cols_B);
        int loc_m  = pc.local_rows(m);
        int loc_n  = pc.local_cols(n);

        int lld_a = std::max(1, loc_rA);
        int lld_b = std::max(1, loc_rB);
        int lld_c = std::max(1, loc_m) + params.ld_pad;

        /* Allocate local matrices */
        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA), ctx.typesize);
        void *B_loc = std::calloc(static_cast<std::size_t>(lld_b) * std::max(1, loc_cB), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0001u;
        void *C_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_c) * std::max(1, loc_n),
                                           ctx.typesize, sentinel_seed);

        /* Scatter global to local */
        scatter_global_to_local(A_loc, lld_a, A_g, rows_A,
                                rows_A, cols_A, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);
        scatter_global_to_local(B_loc, lld_b, B_g, rows_B,
                                rows_B, cols_B, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);
        scatter_global_to_local(C_loc, lld_c, C_g, m,
                                m, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);

        /* Create descriptors */
        int desc_a[9], desc_b[9], desc_c[9];
        pc.make_desc(desc_a, rows_A, cols_A, lld_a);
        pc.make_desc(desc_b, rows_B, cols_B, lld_b);
        pc.make_desc(desc_c, m, n, lld_c);

        /* Call PBLAS */
        int one = 1;
        fn(&ta, &tb, &m, &n, &k,
           alpha,
           A_loc, &one, &one, desc_a,
           B_loc, &one, &one, desc_b,
           beta,
           C_loc, &one, &one, desc_c,
           (std::size_t)1, (std::size_t)1);

        /* MPFR reference on global matrices */
        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_mpfr(rows_A, cols_A, prec);
        MpfrMatrix B_mpfr(rows_B, cols_B, prec);
        MpfrMatrix C_in_mpfr(m, n, prec);
        MpfrMatrix C_ref(m, n, prec);

        custom_to_mpfr_mat(A_mpfr, A_g, rows_A, ctx);
        custom_to_mpfr_mat(B_mpfr, B_g, rows_B, ctx);
        custom_to_mpfr_mat(C_in_mpfr, C_g, m, ctx);

        mpfr_pgemm_ref(C_ref, ta, tb, m, n, k,
                        mpfr_alpha.get(), A_mpfr, B_mpfr,
                        mpfr_beta.get(), C_in_mpfr);

        /* Extract local reference and compare */
        MpfrMatrix C_local_ref(loc_m, loc_n, prec);
        extract_local_mpfr(C_local_ref, C_ref, m, n, pc.mb, pc.nb,
                           pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

        ErrorResult err = compute_error_matrix(C_local_ref, C_loc, lld_c, ctx);
        SentinelResult sr = check_matrix_sentinels(C_loc, loc_m, loc_n, lld_c,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "transa=%c transb=%c grid=%dx%d", ta, tb,
                          pc.bc.nprow, pc.bc.npcol);
            report_result("PGEMM", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(B_g); std::free(C_g);
        std::free(A_loc); std::free(B_loc); std::free(C_loc);
        std::free(alpha); std::free(beta);
    }}

    pc.finalize();
}
