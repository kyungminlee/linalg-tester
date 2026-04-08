/* pher2k.cpp -- PBLAS Level 3 PHER2K accuracy tester (complex-only) */

#include "../pblas.h"
#include "../pblas_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer for PZHER2K */
extern "C" typedef void (*pher2k_fn_t)(
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
/* MPFR reference:                                                      */
/*   trans='N': C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C           */
/*              A,B are n-by-k                                          */
/*   trans='C': C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C           */
/*              A,B are k-by-n                                          */
/* alpha is complex, beta is REAL. Only uplo triangle is meaningful.    */
/* ------------------------------------------------------------------ */

static void mpfr_pher2k_ref(MpfrComplexMatrix &C_ref,
                             char uplo, char trans,
                             int n, int k,
                             const MpfrComplexScalar &alpha,
                             const MpfrComplexMatrix &A,
                             const MpfrComplexMatrix &B,
                             mpfr_t beta,
                             const MpfrComplexMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha.re());

    MpfrComplexScalar conj_alpha(prec);
    mpfr_complex_conj(conj_alpha.re(), conj_alpha.im(),
                       alpha.re(), alpha.im(), MPFR_RNDN);

    MpfrComplexScalar ab_acc(prec), ba_acc(prec);
    MpfrComplexScalar term1(prec), term2(prec), sum12(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            mpfr_set_d(ab_acc.re(), 0.0, MPFR_RNDN);
            mpfr_set_d(ab_acc.im(), 0.0, MPFR_RNDN);
            mpfr_set_d(ba_acc.re(), 0.0, MPFR_RNDN);
            mpfr_set_d(ba_acc.im(), 0.0, MPFR_RNDN);

            for (int p = 0; p < k; ++p) {
                if (trans == 'N') {
                    /* ab_acc += A(i,p) * conj(B(j,p)) */
                    MpfrComplexScalar conj_bjp(prec);
                    mpfr_complex_conj(conj_bjp.re(), conj_bjp.im(),
                                       B.re(j, p), B.im(j, p), MPFR_RNDN);
                    mpfr_complex_fma(ab_acc.re(), ab_acc.im(),
                                      A.re(i, p), A.im(i, p),
                                      conj_bjp.re(), conj_bjp.im(),
                                      MPFR_RNDN);
                    /* ba_acc += B(i,p) * conj(A(j,p)) */
                    MpfrComplexScalar conj_ajp(prec);
                    mpfr_complex_conj(conj_ajp.re(), conj_ajp.im(),
                                       A.re(j, p), A.im(j, p), MPFR_RNDN);
                    mpfr_complex_fma(ba_acc.re(), ba_acc.im(),
                                      B.re(i, p), B.im(i, p),
                                      conj_ajp.re(), conj_ajp.im(),
                                      MPFR_RNDN);
                } else {
                    /* ab_acc += conj(A(p,i)) * B(p,j) */
                    MpfrComplexScalar conj_api(prec);
                    mpfr_complex_conj(conj_api.re(), conj_api.im(),
                                       A.re(p, i), A.im(p, i), MPFR_RNDN);
                    mpfr_complex_fma(ab_acc.re(), ab_acc.im(),
                                      conj_api.re(), conj_api.im(),
                                      B.re(p, j), B.im(p, j),
                                      MPFR_RNDN);
                    /* ba_acc += conj(B(p,i)) * A(p,j) */
                    MpfrComplexScalar conj_bpi(prec);
                    mpfr_complex_conj(conj_bpi.re(), conj_bpi.im(),
                                       B.re(p, i), B.im(p, i), MPFR_RNDN);
                    mpfr_complex_fma(ba_acc.re(), ba_acc.im(),
                                      conj_bpi.re(), conj_bpi.im(),
                                      A.re(p, j), A.im(p, j),
                                      MPFR_RNDN);
                }
            }

            /* term1 = alpha * ab_acc */
            mpfr_complex_mul(term1.re(), term1.im(),
                              alpha.re(), alpha.im(),
                              ab_acc.re(), ab_acc.im(), MPFR_RNDN);
            /* term2 = conj(alpha) * ba_acc */
            mpfr_complex_mul(term2.re(), term2.im(),
                              conj_alpha.re(), conj_alpha.im(),
                              ba_acc.re(), ba_acc.im(), MPFR_RNDN);
            /* sum12 = term1 + term2 */
            mpfr_complex_add(sum12.re(), sum12.im(),
                              term1.re(), term1.im(),
                              term2.re(), term2.im(), MPFR_RNDN);
            /* beta_c = beta * C_in(i,j)  (beta is real) */
            mpfr_complex_mul_real(beta_c.re(), beta_c.im(),
                                   C_in.re(i, j), C_in.im(i, j),
                                   beta, MPFR_RNDN);
            /* C_ref(i,j) = sum12 + beta_c */
            mpfr_complex_add(C_ref.re(i, j), C_ref.im(i, j),
                              sum12.re(), sum12.im(),
                              beta_c.re(), beta_c.im(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pher2k(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        fprintf(stderr, "PHER2K requires --complex\n");
        return;
    }

    int mb = params.mb > 0 ? params.mb : params.n;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PHER2K", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pher2k_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PHER2K", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n, k = params.k;
    mpfr_prec_t prec = ctx.prec;
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'C'}) {
        int rows_AB = (trans == 'N') ? n : k;
        int cols_AB = (trans == 'N') ? k : n;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_al = params.seed + 3;
        unsigned seed_be = params.seed + 4;

        void *A_g   = gen_random_complex_array(rows_AB * cols_AB, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *B_g   = gen_random_complex_array(rows_AB * cols_AB, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_B);
        void *C_g   = gen_hermitian_array(n, uplo, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_C);
        void *alpha = gen_random_complex_array(1, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_al);
        void *beta  = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_be);

        int loc_rAB = pc.local_rows(rows_AB);
        int loc_cAB = pc.local_cols(cols_AB);
        int loc_n   = pc.local_rows(n);
        int loc_nc  = pc.local_cols(n);

        int lld_a = std::max(1, loc_rAB);
        int lld_b = std::max(1, loc_rAB);
        int lld_c = std::max(1, loc_n) + params.ld_pad;

        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cAB), ctx.typesize);
        void *B_loc = std::calloc(static_cast<std::size_t>(lld_b) * std::max(1, loc_cAB), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0022u;
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

        /* MPFR reference */
        MpfrComplexScalar mpfr_alpha(prec);
        ctx.to_mpfr_complex(mpfr_alpha.re(), mpfr_alpha.im(), alpha);

        MpfrScalar mpfr_beta(prec);
        ctx.to_mpfr(mpfr_beta.get(), beta);

        MpfrComplexMatrix A_mpfr(rows_AB, cols_AB, prec);
        MpfrComplexMatrix B_mpfr(rows_AB, cols_AB, prec);
        MpfrComplexMatrix C_in_mpfr(n, n, prec);
        MpfrComplexMatrix C_ref(n, n, prec);

        custom_to_mpfr_complex_mat(A_mpfr,    A_g, rows_AB, ctx);
        custom_to_mpfr_complex_mat(B_mpfr,    B_g, rows_AB, ctx);
        custom_to_mpfr_complex_mat(C_in_mpfr, C_g, n,       ctx);

        mpfr_pher2k_ref(C_ref, uplo, trans, n, k,
                         mpfr_alpha, A_mpfr, B_mpfr,
                         mpfr_beta.get(), C_in_mpfr);

        /* Build local reference: uplo triangle from C_ref, opposite from C_in */
        MpfrComplexMatrix C_local_ref(loc_n, loc_nc, prec);
        extract_local_mpfr_complex_herm(C_local_ref, C_ref, C_in_mpfr,
                                         n, pc.mb, pc.nb,
                                         pc.bc.myrow, pc.bc.mycol,
                                         pc.bc.nprow, pc.bc.npcol, uplo);

        ErrorResult err = compute_error_complex_matrix(C_local_ref, C_loc, lld_c, ctx);
        SentinelResult sr = check_matrix_sentinels(C_loc, loc_n, loc_nc, lld_c,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "uplo=%c trans=%c grid=%dx%d", uplo, trans,
                          pc.bc.nprow, pc.bc.npcol);
            report_result("PHER2K", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(B_g); std::free(C_g);
        std::free(A_loc); std::free(B_loc); std::free(C_loc);
        std::free(alpha); std::free(beta);
    }}

    pc.finalize();
}
