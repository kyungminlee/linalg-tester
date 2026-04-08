/* pherk.cpp -- PBLAS Level 3 PHERK accuracy tester (complex-only) */

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

/* Fortran-ABI function pointer for PZHERK */
extern "C" typedef void (*pherk_fn_t)(
    const char *uplo,  const char *trans,
    const int  *n,     const int  *k,
    const void *alpha,
    const void *A,     const int  *ia,  const int  *ja,  const int  *desca,
    const void *beta,
    void       *C,     const int  *ic,  const int  *jc,  const int  *descc,
    std::size_t uplo_len, std::size_t trans_len
);

/* ------------------------------------------------------------------ */
/* MPFR reference:                                                      */
/*   trans='N': C = alpha * A * A^H + beta * C,  A is n-by-k           */
/*   trans='C': C = alpha * A^H * A + beta * C,  A is k-by-n           */
/* alpha and beta are REAL scalars. Only uplo triangle is meaningful.   */
/* ------------------------------------------------------------------ */

static void mpfr_pherk_ref(MpfrComplexMatrix &C_ref,
                            char uplo, char trans,
                            int n, int k,
                            mpfr_t alpha,
                            const MpfrComplexMatrix &A,
                            mpfr_t beta,
                            const MpfrComplexMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha);
    MpfrComplexScalar acc(prec), beta_c(prec);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            mpfr_set_d(acc.re(), 0.0, MPFR_RNDN);
            mpfr_set_d(acc.im(), 0.0, MPFR_RNDN);

            for (int p = 0; p < k; ++p) {
                if (trans == 'N') {
                    /* A(i,p) * conj(A(j,p)) */
                    MpfrComplexScalar conj_ajp(prec);
                    mpfr_complex_conj(conj_ajp.re(), conj_ajp.im(),
                                       A.re(j, p), A.im(j, p), MPFR_RNDN);
                    mpfr_complex_fma(acc.re(), acc.im(),
                                      A.re(i, p), A.im(i, p),
                                      conj_ajp.re(), conj_ajp.im(),
                                      MPFR_RNDN);
                } else {
                    /* conj(A(p,i)) * A(p,j) */
                    MpfrComplexScalar conj_api(prec);
                    mpfr_complex_conj(conj_api.re(), conj_api.im(),
                                       A.re(p, i), A.im(p, i), MPFR_RNDN);
                    mpfr_complex_fma(acc.re(), acc.im(),
                                      conj_api.re(), conj_api.im(),
                                      A.re(p, j), A.im(p, j),
                                      MPFR_RNDN);
                }
            }

            /* alpha * acc  (alpha is real) */
            MpfrComplexScalar alpha_acc(prec);
            mpfr_complex_mul_real(alpha_acc.re(), alpha_acc.im(),
                                   acc.re(), acc.im(), alpha, MPFR_RNDN);
            /* beta * C_in(i,j)  (beta is real) */
            mpfr_complex_mul_real(beta_c.re(), beta_c.im(),
                                   C_in.re(i, j), C_in.im(i, j),
                                   beta, MPFR_RNDN);
            /* C_ref(i,j) = alpha*acc + beta*C_in */
            mpfr_complex_add(C_ref.re(i, j), C_ref.im(i, j),
                              alpha_acc.re(), alpha_acc.im(),
                              beta_c.re(), beta_c.im(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pherk(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        fprintf(stderr, "PHERK requires --complex\n");
        return;
    }

    int mb = params.mb > 0 ? params.mb : params.n;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PHERK", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pherk_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PHERK", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.n, k = params.k;
    mpfr_prec_t prec = ctx.prec;
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'C'}) {
        int rows_A = (trans == 'N') ? n : k;
        int cols_A = (trans == 'N') ? k : n;

        unsigned seed_A  = params.seed;
        unsigned seed_C  = params.seed + 1;
        unsigned seed_al = params.seed + 2;
        unsigned seed_be = params.seed + 3;

        void *A_g   = gen_random_complex_array(rows_A * cols_A, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *C_g   = gen_hermitian_array(n, uplo, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_C);
        void *alpha = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_al);
        void *beta  = gen_random_array(1, real_typesize, ctx.from_mpfr, prec, &seed_be);

        int loc_rA = pc.local_rows(rows_A);
        int loc_cA = pc.local_cols(cols_A);
        int loc_n  = pc.local_rows(n);
        int loc_nc = pc.local_cols(n);

        int lld_a = std::max(1, loc_rA);
        int lld_c = std::max(1, loc_n) + params.ld_pad;

        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0021u;
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

        MpfrComplexMatrix A_mpfr(rows_A, cols_A, prec);
        MpfrComplexMatrix C_in_mpfr(n, n, prec);
        MpfrComplexMatrix C_ref(n, n, prec);

        custom_to_mpfr_complex_mat(A_mpfr,    A_g, rows_A, ctx);
        custom_to_mpfr_complex_mat(C_in_mpfr, C_g, n,      ctx);

        mpfr_pherk_ref(C_ref, uplo, trans, n, k,
                        mpfr_alpha.get(), A_mpfr,
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
            report_result("PHERK", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(C_g);
        std::free(A_loc); std::free(C_loc);
        std::free(alpha); std::free(beta);
    }}

    pc.finalize();
}
