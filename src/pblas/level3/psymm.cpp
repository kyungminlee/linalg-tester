/* psymm.cpp -- PBLAS Level 3 PSYMM accuracy tester */

#include "../pblas.h"
#include "../pblas_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer for PDSYMM */
extern "C" typedef void (*psymm_fn_t)(
    const char *side,  const char *uplo,
    const int  *m,     const int  *n,
    const void *alpha,
    const void *A,     const int  *ia,  const int  *ja,  const int  *desca,
    const void *B,     const int  *ib,  const int  *jb,  const int  *descb,
    const void *beta,
    void       *C,     const int  *ic,  const int  *jc,  const int  *descc,
    std::size_t side_len, std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference                                                       */
/* ------------------------------------------------------------------ */

static void mpfr_psymm_ref(MpfrMatrix &C_ref,
                            char side, char uplo,
                            int m, int n,
                            mpfr_t alpha,
                            const MpfrMatrix &A,
                            const MpfrMatrix &B,
                            mpfr_t beta,
                            const MpfrMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    int ka = (side == 'L') ? m : n;

    MpfrMatrix Af(ka, ka, prec);
    for (int j = 0; j < ka; ++j)
        for (int i = 0; i < ka; ++i) {
            if ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j))
                mpfr_set(Af.at(i, j), A.at(i, j), MPFR_RNDN);
            else
                mpfr_set(Af.at(i, j), A.at(j, i), MPFR_RNDN);
        }

    MpfrScalar acc(prec), alpha_acc(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
            if (side == 'L') {
                for (int p = 0; p < m; ++p)
                    mpfr_fma(acc.get(), Af.at(i, p), B.at(p, j), acc.get(), MPFR_RNDN);
            } else {
                for (int p = 0; p < n; ++p)
                    mpfr_fma(acc.get(), B.at(i, p), Af.at(p, j), acc.get(), MPFR_RNDN);
            }
            mpfr_mul(alpha_acc.get(), alpha, acc.get(), MPFR_RNDN);
            mpfr_mul(beta_c.get(), beta, C_in.at(i, j), MPFR_RNDN);
            mpfr_add(C_ref.at(i, j), alpha_acc.get(), beta_c.get(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_psymm(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYMM", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<psymm_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYMM", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int m = params.m, n = params.n;
    mpfr_prec_t prec = ctx.prec;

    for (char side : {'L', 'R'}) {
    for (char uplo : {'U', 'L'}) {
        int ka = (side == 'L') ? m : n;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        void *A_g   = gen_symmetric_array(ka, uplo, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *B_g   = gen_random_array(m * n, ctx.typesize, ctx.from_mpfr, prec, &seed_B);
        void *C_g   = gen_random_array(m * n, ctx.typesize, ctx.from_mpfr, prec, &seed_C);
        void *alpha = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        int loc_rA = pc.local_rows(ka);
        int loc_cA = pc.local_cols(ka);
        int loc_m  = pc.local_rows(m);
        int loc_n  = pc.local_cols(n);

        int lld_a = std::max(1, loc_rA);
        int lld_b = std::max(1, loc_m);
        int lld_c = std::max(1, loc_m) + params.ld_pad;

        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA), ctx.typesize);
        void *B_loc = std::calloc(static_cast<std::size_t>(lld_b) * std::max(1, loc_n), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0003u;
        void *C_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_c) * std::max(1, loc_n),
                                           ctx.typesize, sentinel_seed);

        scatter_global_to_local(A_loc, lld_a, A_g, ka, ka, ka, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);
        scatter_global_to_local(B_loc, lld_b, B_g, m, m, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);
        scatter_global_to_local(C_loc, lld_c, C_g, m, m, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);

        int desc_a[9], desc_b[9], desc_c[9];
        pc.make_desc(desc_a, ka, ka, lld_a);
        pc.make_desc(desc_b, m, n, lld_b);
        pc.make_desc(desc_c, m, n, lld_c);

        int one = 1;
        fn(&side, &uplo, &m, &n,
           alpha,
           A_loc, &one, &one, desc_a,
           B_loc, &one, &one, desc_b,
           beta,
           C_loc, &one, &one, desc_c,
           (std::size_t)1, (std::size_t)1);

        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_mpfr(ka, ka, prec);
        MpfrMatrix B_mpfr(m, n, prec);
        MpfrMatrix C_in_mpfr(m, n, prec);
        MpfrMatrix C_ref(m, n, prec);

        custom_to_mpfr_mat(A_mpfr, A_g, ka, ctx);
        custom_to_mpfr_mat(B_mpfr, B_g, m, ctx);
        custom_to_mpfr_mat(C_in_mpfr, C_g, m, ctx);

        mpfr_psymm_ref(C_ref, side, uplo, m, n,
                        mpfr_alpha.get(), A_mpfr, B_mpfr,
                        mpfr_beta.get(), C_in_mpfr);

        MpfrMatrix C_local_ref(loc_m, loc_n, prec);
        extract_local_mpfr(C_local_ref, C_ref, m, n, pc.mb, pc.nb,
                           pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

        ErrorResult err = compute_error_matrix(C_local_ref, C_loc, lld_c, ctx);
        SentinelResult sr = check_matrix_sentinels(C_loc, loc_m, loc_n, lld_c,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "side=%c uplo=%c grid=%dx%d", side, uplo,
                          pc.bc.nprow, pc.bc.npcol);
            report_result("PSYMM", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(B_g); std::free(C_g);
        std::free(A_loc); std::free(B_loc); std::free(C_loc);
        std::free(alpha); std::free(beta);
    }}

    pc.finalize();
}
