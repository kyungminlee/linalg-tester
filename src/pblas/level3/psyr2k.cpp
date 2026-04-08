/* psyr2k.cpp -- PBLAS Level 3 PSYR2K accuracy tester */

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

/* Fortran-ABI function pointer for PDSYR2K */
extern "C" typedef void (*psyr2k_fn_t)(
    const char *uplo,  const char *trans,
    const int  *n,     const int  *k,
    const void *alpha,
    const void *A,     const int  *ia,  const int  *ja,  const int  *desca,
    const void *B,     const int  *ib,  const int  *jb,  const int  *descb,
    const void *beta,
    void       *C,     const int  *ic,  const int  *jc,  const int  *descc,
    std::size_t uplo_len, std::size_t trans_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference                                                       */
/* ------------------------------------------------------------------ */

static void mpfr_psyr2k_ref(MpfrMatrix &C_ref,
                             char uplo, char trans,
                             int n, int k,
                             mpfr_t alpha,
                             const MpfrMatrix &A,
                             const MpfrMatrix &B,
                             mpfr_t beta,
                             const MpfrMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar ab_acc(prec), ba_acc(prec), sum(prec);
    MpfrScalar alpha_sum(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            mpfr_set_d(ab_acc.get(), 0.0, MPFR_RNDN);
            mpfr_set_d(ba_acc.get(), 0.0, MPFR_RNDN);
            for (int p = 0; p < k; ++p) {
                if (trans == 'N') {
                    mpfr_fma(ab_acc.get(), A.at(i, p), B.at(j, p), ab_acc.get(), MPFR_RNDN);
                    mpfr_fma(ba_acc.get(), B.at(i, p), A.at(j, p), ba_acc.get(), MPFR_RNDN);
                } else {
                    mpfr_fma(ab_acc.get(), A.at(p, i), B.at(p, j), ab_acc.get(), MPFR_RNDN);
                    mpfr_fma(ba_acc.get(), B.at(p, i), A.at(p, j), ba_acc.get(), MPFR_RNDN);
                }
            }
            mpfr_add(sum.get(), ab_acc.get(), ba_acc.get(), MPFR_RNDN);
            mpfr_mul(alpha_sum.get(), alpha, sum.get(), MPFR_RNDN);
            mpfr_mul(beta_c.get(), beta, C_in.at(i, j), MPFR_RNDN);
            mpfr_add(C_ref.at(i, j), alpha_sum.get(), beta_c.get(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_psyr2k(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.n;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYR2K", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<psyr2k_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYR2K", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n, k = params.k;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T'}) {
        int rows_AB = (trans == 'N') ? n : k;
        int cols_AB = (trans == 'N') ? k : n;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        void *A_g   = gen_random_array(rows_AB * cols_AB, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *B_g   = gen_random_array(rows_AB * cols_AB, ctx.typesize, ctx.from_mpfr, prec, &seed_B);
        void *C_g   = gen_random_array(n * n, ctx.typesize, ctx.from_mpfr, prec, &seed_C);
        void *alpha = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        int loc_rAB = pc.local_rows(rows_AB);
        int loc_cAB = pc.local_cols(cols_AB);
        int loc_n   = pc.local_rows(n);
        int loc_nc  = pc.local_cols(n);

        int lld_a = std::max(1, loc_rAB);
        int lld_b = std::max(1, loc_rAB);
        int lld_c = std::max(1, loc_n) + params.ld_pad;

        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cAB), ctx.typesize);
        void *B_loc = std::calloc(static_cast<std::size_t>(lld_b) * std::max(1, loc_cAB), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0005u;
        void *C_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_c) * std::max(1, loc_nc),
                                           ctx.typesize, sentinel_seed);

        scatter_global_to_local(A_loc, lld_a, A_g, rows_AB, rows_AB, cols_AB, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);
        scatter_global_to_local(B_loc, lld_b, B_g, rows_AB, rows_AB, cols_AB, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);
        scatter_global_to_local(C_loc, lld_c, C_g, n, n, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);

        int desc_a[9], desc_b[9], desc_c[9];
        pc.make_desc(desc_a, rows_AB, cols_AB, lld_a);
        pc.make_desc(desc_b, rows_AB, cols_AB, lld_b);
        pc.make_desc(desc_c, n, n, lld_c);

        int one = 1;
        fn(&uplo, &trans, &n, &k,
           alpha,
           A_loc, &one, &one, desc_a,
           B_loc, &one, &one, desc_b,
           beta,
           C_loc, &one, &one, desc_c,
           (std::size_t)1, (std::size_t)1);

        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_mpfr(rows_AB, cols_AB, prec);
        MpfrMatrix B_mpfr(rows_AB, cols_AB, prec);
        MpfrMatrix C_in_mpfr(n, n, prec);
        MpfrMatrix C_ref(n, n, prec);

        custom_to_mpfr_mat(A_mpfr, A_g, rows_AB, ctx);
        custom_to_mpfr_mat(B_mpfr, B_g, rows_AB, ctx);
        custom_to_mpfr_mat(C_in_mpfr, C_g, n, ctx);

        mpfr_psyr2k_ref(C_ref, uplo, trans, n, k,
                         mpfr_alpha.get(), A_mpfr, B_mpfr,
                         mpfr_beta.get(), C_in_mpfr);

        MpfrMatrix C_local_ref(loc_n, loc_nc, prec);
        extract_local_mpfr_sym(C_local_ref, C_ref, C_in_mpfr,
                               n, pc.mb, pc.nb,
                               pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                               uplo);

        ErrorResult err = compute_error_matrix(C_local_ref, C_loc, lld_c, ctx);
        SentinelResult sr = check_matrix_sentinels(C_loc, loc_n, loc_nc, lld_c,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "uplo=%c trans=%c grid=%dx%d", uplo, trans,
                          pc.bc.nprow, pc.bc.npcol);
            report_result("PSYR2K", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(B_g); std::free(C_g);
        std::free(A_loc); std::free(B_loc); std::free(C_loc);
        std::free(alpha); std::free(beta);
    }}

    pc.finalize();
}
