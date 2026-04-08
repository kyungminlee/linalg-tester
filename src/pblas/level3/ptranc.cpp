/* ptranc.cpp -- PBLAS Level 3 PZTRANC accuracy tester (complex conjugate transpose) */

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

/* Fortran-ABI function pointer for PZTRANC (no char args, no hidden lengths) */
extern "C" typedef void (*ptranc_fn_t)(
    const int  *m,     const int  *n,
    const void *alpha,
    const void *A,     const int  *ia,  const int  *ja,  const int  *desca,
    const void *beta,
    void       *C,     const int  *ic,  const int  *jc,  const int  *descc
);

/* ------------------------------------------------------------------ */
/* MPFR reference: C_ref(j,i) = alpha * conj(A(i,j)) + beta * C_in(j,i) */
/*   A is m-by-n, C is n-by-m                                          */
/* ------------------------------------------------------------------ */

static void mpfr_ptranc_ref(MpfrComplexMatrix &C_ref,
                             int m, int n,
                             const MpfrComplexScalar &alpha,
                             const MpfrComplexMatrix &A,
                             const MpfrComplexScalar &beta,
                             const MpfrComplexMatrix &C_in)
{
    mpfr_prec_t prec = mpfr_get_prec(alpha.re());
    MpfrComplexScalar conj_a(prec), alpha_term(prec), beta_term(prec);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            /* conj(A(i,j)) */
            mpfr_complex_conj(conj_a.re(), conj_a.im(),
                               A.re(i, j), A.im(i, j), MPFR_RNDN);
            /* alpha * conj(A(i,j)) */
            mpfr_complex_mul(alpha_term.re(), alpha_term.im(),
                              alpha.re(), alpha.im(),
                              conj_a.re(), conj_a.im(), MPFR_RNDN);
            /* beta * C_in(j,i) */
            mpfr_complex_mul(beta_term.re(), beta_term.im(),
                              beta.re(), beta.im(),
                              C_in.re(j, i), C_in.im(j, i), MPFR_RNDN);
            /* C_ref(j,i) = alpha_term + beta_term */
            mpfr_complex_add(C_ref.re(j, i), C_ref.im(j, i),
                              alpha_term.re(), alpha_term.im(),
                              beta_term.re(), beta_term.im(), MPFR_RNDN);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_ptranc(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        fprintf(stderr, "PTRANC requires --complex\n");
        return;
    }

    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PTRANC", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<ptranc_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PTRANC", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int m = params.m, n = params.n;
    mpfr_prec_t prec = ctx.prec;

    {
        unsigned seed_A  = params.seed;
        unsigned seed_C  = params.seed + 1;
        unsigned seed_ab = params.seed + 2;

        /* A is m-by-n, C is n-by-m */
        void *A_g   = gen_random_complex_array(m * n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *C_g   = gen_random_complex_array(n * m, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_C);
        void *alpha = gen_random_complex_array(1,     ctx.typesize, ctx.from_mpfr_complex, prec, &seed_ab);
        void *beta  = gen_random_complex_array(1,     ctx.typesize, ctx.from_mpfr_complex, prec, &seed_ab);

        /* Local dims for A (m-by-n) */
        int loc_rA = pc.local_rows(m);
        int loc_cA = pc.local_cols(n);
        /* Local dims for C (n-by-m) */
        int loc_rC = pc.local_rows(n);
        int loc_cC = pc.local_cols(m);

        int lld_a = std::max(1, loc_rA);
        int lld_c = std::max(1, loc_rC) + params.ld_pad;

        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_cA), ctx.typesize);

        unsigned sentinel_seed = 0xFBAD0024u;
        void *C_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_c) * std::max(1, loc_cC),
                                           ctx.typesize, sentinel_seed);

        scatter_global_to_local(A_loc, lld_a, A_g, m, m, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);
        scatter_global_to_local(C_loc, lld_c, C_g, n, n, m, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol, ctx.typesize);

        int desc_a[9], desc_c[9];
        pc.make_desc(desc_a, m, n, lld_a);
        pc.make_desc(desc_c, n, m, lld_c);

        int one = 1;
        fn(&m, &n,
           alpha,
           A_loc, &one, &one, desc_a,
           beta,
           C_loc, &one, &one, desc_c);

        /* MPFR reference */
        MpfrComplexScalar mpfr_alpha(prec), mpfr_beta(prec);
        ctx.to_mpfr_complex(mpfr_alpha.re(), mpfr_alpha.im(), alpha);
        ctx.to_mpfr_complex(mpfr_beta.re(),  mpfr_beta.im(),  beta);

        MpfrComplexMatrix A_mpfr(m, n, prec);
        MpfrComplexMatrix C_in_mpfr(n, m, prec);
        MpfrComplexMatrix C_ref(n, m, prec);

        custom_to_mpfr_complex_mat(A_mpfr,    A_g, m, ctx);
        custom_to_mpfr_complex_mat(C_in_mpfr, C_g, n, ctx);

        mpfr_ptranc_ref(C_ref, m, n,
                         mpfr_alpha, A_mpfr,
                         mpfr_beta, C_in_mpfr);

        /* Extract local reference and compare */
        MpfrComplexMatrix C_local_ref(loc_rC, loc_cC, prec);
        extract_local_mpfr_complex(C_local_ref, C_ref, n, m, pc.mb, pc.nb,
                                    pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

        ErrorResult err = compute_error_complex_matrix(C_local_ref, C_loc, lld_c, ctx);
        SentinelResult sr = check_matrix_sentinels(C_loc, loc_rC, loc_cC, lld_c,
                                                    ctx.typesize, sentinel_seed);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "grid=%dx%d", pc.bc.nprow, pc.bc.npcol);
            report_result("PTRANC", params_str, err, &sr, format);
        }

        std::free(A_g); std::free(C_g);
        std::free(A_loc); std::free(C_loc);
        std::free(alpha); std::free(beta);
    }

    pc.finalize();
}
