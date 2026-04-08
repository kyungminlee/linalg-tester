/* ptrmm.cpp -- PBLAS Level 3 PTRMM accuracy tester */

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

/* Fortran-ABI function pointer for PDTRMM */
extern "C" typedef void (*ptrmm_fn_t)(
    const char *side,   const char *uplo,
    const char *transa, const char *diag,
    const int  *m,      const int  *n,
    const void *alpha,
    const void *A,      const int  *ia,  const int  *ja,  const int  *desca,
    void       *B,      const int  *ib,  const int  *jb,  const int  *descb,
    std::size_t side_len,   std::size_t uplo_len,
    std::size_t transa_len, std::size_t diag_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference: B = alpha * op(A) * B  (side='L')                   */
/*                 B = alpha * B * op(A)  (side='R')                   */
/* ------------------------------------------------------------------ */

static void mpfr_ptrmm_ref(MpfrMatrix &B_ref,
                            char side, char uplo, char transa, char diag,
                            int m, int n,
                            mpfr_t alpha,
                            const MpfrMatrix &A,
                            const MpfrMatrix &B_in)
{
    int ka = (side == 'L') ? m : n;
    mpfr_prec_t prec = mpfr_get_prec(alpha);

    MpfrMatrix Aw(ka, ka, prec);
    for (int j = 0; j < ka; ++j) {
        for (int i = 0; i < ka; ++i) {
            if ((uplo == 'U' && j >= i) || (uplo == 'L' && j <= i))
                mpfr_set(Aw.at(i, j), A.at(i, j), MPFR_RNDN);
            else
                mpfr_set_d(Aw.at(i, j), 0.0, MPFR_RNDN);
            if (diag == 'U' && i == j)
                mpfr_set_d(Aw.at(i, j), 1.0, MPFR_RNDN);
        }
    }

    if (transa != 'N') {
        MpfrMatrix Awt(ka, ka, prec);
        for (int j = 0; j < ka; ++j)
            for (int i = 0; i < ka; ++i)
                mpfr_set(Awt.at(i, j), Aw.at(j, i), MPFR_RNDN);
        for (int j = 0; j < ka; ++j)
            for (int i = 0; i < ka; ++i)
                mpfr_set(Aw.at(i, j), Awt.at(i, j), MPFR_RNDN);
    }

    MpfrScalar acc(prec);

    if (side == 'L') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
                for (int p = 0; p < m; ++p)
                    mpfr_fma(acc.get(), Aw.at(i, p), B_in.at(p, j),
                             acc.get(), MPFR_RNDN);
                mpfr_mul(B_ref.at(i, j), alpha, acc.get(), MPFR_RNDN);
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                mpfr_set_d(acc.get(), 0.0, MPFR_RNDN);
                for (int p = 0; p < n; ++p)
                    mpfr_fma(acc.get(), B_in.at(i, p), Aw.at(p, j),
                             acc.get(), MPFR_RNDN);
                mpfr_mul(B_ref.at(i, j), alpha, acc.get(), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_ptrmm(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PTRMM", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<ptrmm_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PTRMM", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int m = params.m, n = params.n;
    mpfr_prec_t prec = ctx.prec;

    for (char side : {'L', 'R'}) {
    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T'}) {
    for (char diag : {'N', 'U'}) {
        int ka = (side == 'L') ? m : n;

        unsigned seed_A  = params.seed + 10;
        unsigned seed_B  = params.seed + 11;
        unsigned seed_al = params.seed + 12;

        void *A_g   = gen_triangular_array(ka, uplo, diag,
                                           ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *B_g   = gen_random_array(m * n, ctx.typesize, ctx.from_mpfr, prec, &seed_B);
        void *alpha = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_al);

        int loc_rA = pc.local_rows(ka);
        int loc_cA = pc.local_cols(ka);
        int loc_m  = pc.local_rows(m);
        int loc_n  = pc.local_cols(n);

        int lld_a = std::max(1, loc_rA);
        int lld_b = std::max(1, loc_m) + params.ld_pad;

        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0006u;
        void *B_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_b) * std::max(1, loc_n),
                                           ctx.typesize, sentinel_seed);

        scatter_global_to_local(A_loc, lld_a, A_g, ka, ka, ka, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);
        scatter_global_to_local(B_loc, lld_b, B_g, m, m, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);

        int desc_a[9], desc_b[9];
        pc.make_desc(desc_a, ka, ka, lld_a);
        pc.make_desc(desc_b, m, n, lld_b);

        int one = 1;
        fn(&side, &uplo, &trans, &diag, &m, &n,
           alpha,
           A_loc, &one, &one, desc_a,
           B_loc, &one, &one, desc_b,
           (std::size_t)1, (std::size_t)1, (std::size_t)1, (std::size_t)1);

        /* MPFR reference */
        MpfrScalar mpfr_alpha(prec);
        ctx.to_mpfr(mpfr_alpha.get(), alpha);

        MpfrMatrix A_mpfr(ka, ka, prec);
        MpfrMatrix B_in_mpfr(m, n, prec);
        MpfrMatrix B_ref(m, n, prec);

        custom_to_mpfr_mat(A_mpfr, A_g, ka, ctx);
        custom_to_mpfr_mat(B_in_mpfr, B_g, m, ctx);

        mpfr_ptrmm_ref(B_ref, side, uplo, trans, diag,
                        m, n, mpfr_alpha.get(), A_mpfr, B_in_mpfr);

        MpfrMatrix B_local_ref(loc_m, loc_n, prec);
        extract_local_mpfr(B_local_ref, B_ref, m, n, pc.mb, pc.nb,
                           pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

        ErrorResult err = compute_error_matrix(B_local_ref, B_loc, lld_b, ctx);
        SentinelResult sr = check_matrix_sentinels(B_loc, loc_m, loc_n, lld_b,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "side=%c uplo=%c trans=%c diag=%c grid=%dx%d",
                          side, uplo, trans, diag, pc.bc.nprow, pc.bc.npcol);
            report_result("PTRMM", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(B_g); std::free(alpha);
        std::free(A_loc); std::free(B_loc);
    }}}}

    pc.finalize();
}
