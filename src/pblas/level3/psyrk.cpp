/* psyrk.cpp -- PBLAS Level 3 PSYRK accuracy tester */

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

/* Fortran-ABI function pointer for PDSYRK */
extern "C" typedef void (*psyrk_fn_t)(
    const char *uplo,  const char *trans,
    const int  *n,     const int  *k,
    const void *alpha,
    const void *A,     const int  *ia,  const int  *ja,  const int  *desca,
    const void *beta,
    void       *C,     const int  *ic,  const int  *jc,  const int  *descc,
    std::size_t uplo_len, std::size_t trans_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference                                                       */
/* ------------------------------------------------------------------ */

static void mpfr_psyrk_ref(MpfrMatrix &C_ref,
                            char uplo, char trans,
                            int n, int k,
                            mpfr_t alpha,
                            const MpfrMatrix &A,
                            mpfr_t beta,
                            const MpfrMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrScalar acc(prec), alpha_acc(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
            for (int p = 0; p < k; ++p) {
                if (trans == 'N')
                    mpfr_fma(acc.get(), A.at(i, p), A.at(j, p), acc.get(), MPFR_RNDN);
                else
                    mpfr_fma(acc.get(), A.at(p, i), A.at(p, j), acc.get(), MPFR_RNDN);
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

void test_psyrk(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.n;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYRK", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<psyrk_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSYRK", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n, k = params.k;
    mpfr_prec_t prec = ctx.prec;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T'}) {
        int rows_A = (trans == 'N') ? n : k;
        int cols_A = (trans == 'N') ? k : n;

        unsigned seed_A  = params.seed;
        unsigned seed_C  = params.seed + 1;
        unsigned seed_ab = params.seed + 2;

        void *A_g   = gen_random_array(rows_A * cols_A, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *C_g   = gen_random_array(n * n, ctx.typesize, ctx.from_mpfr, prec, &seed_C);
        void *alpha = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_ab);
        void *beta  = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_ab);

        int loc_rA = pc.local_rows(rows_A);
        int loc_cA = pc.local_cols(cols_A);
        int loc_n  = pc.local_rows(n);
        int loc_nc = pc.local_cols(n);

        int lld_a = std::max(1, loc_rA);
        int lld_c = std::max(1, loc_n) + params.ld_pad;

        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0004u;
        void *C_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_c) * std::max(1, loc_nc),
                                           ctx.typesize, sentinel_seed);

        scatter_global_to_local(A_loc, lld_a, A_g, rows_A, rows_A, cols_A, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);
        scatter_global_to_local(C_loc, lld_c, C_g, n, n, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);

        int desc_a[9], desc_c[9];
        pc.make_desc(desc_a, rows_A, cols_A, lld_a);
        pc.make_desc(desc_c, n, n, lld_c);

        int one = 1;
        fn(&uplo, &trans, &n, &k,
           alpha,
           A_loc, &one, &one, desc_a,
           beta,
           C_loc, &one, &one, desc_c,
           (std::size_t)1, (std::size_t)1);

        /* MPFR reference */
        MpfrScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);
        ctx.to_mpfr(mpfr_beta.get(),  beta);

        MpfrMatrix A_mpfr(rows_A, cols_A, prec);
        MpfrMatrix C_in_mpfr(n, n, prec);
        MpfrMatrix C_ref(n, n, prec);

        custom_to_mpfr_mat(A_mpfr, A_g, rows_A, ctx);
        custom_to_mpfr_mat(C_in_mpfr, C_g, n, ctx);

        mpfr_psyrk_ref(C_ref, uplo, trans, n, k,
                        mpfr_alpha.get(), A_mpfr,
                        mpfr_beta.get(), C_in_mpfr);

        /* Build local reference: uplo triangle from C_ref, opposite from C_in */
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
            report_result("PSYRK", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(C_g);
        std::free(A_loc); std::free(C_loc);
        std::free(alpha); std::free(beta);
    }}

    pc.finalize();
}
