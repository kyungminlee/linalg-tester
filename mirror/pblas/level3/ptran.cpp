/* ptran.cpp -- Mirror tester for PBLAS Level 3 PTRAN (real transpose) */

#include "../mirror_pblas_common.h"

/* No character params -- no hidden lengths */
extern "C" typedef void (*ptran_fn_t)(
    const int *m, const int *n,
    const void *alpha,
    const void *A, const int *ia, const int *ja, const int *desca,
    const void *beta,
    void *C, const int *ic, const int *jc, const int *descc);

void mirror_test_ptran(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_A = params.seed;
    unsigned seed_C = params.seed + 1;

    /* A is m x n, C is n x m.  C = alpha * A^T + beta * C */
    MpfrMatrix A_mpfr(m, n, prec);
    MpfrMatrix C_mpfr(n, m, prec);

    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);
    gen_mpfr_random_matrix(C_mpfr, prec, &seed_C);

    /* Use alpha=1, beta=0 for a pure transpose */
    MpfrScalar alpha_mpfr(prec), beta_mpfr(prec);
    mpfr_set_d(alpha_mpfr.get(), 1.0, MPFR_RNDN);
    mpfr_set_d(beta_mpfr.get(), 0.0, MPFR_RNDN);

    auto run_side = [&](const MirrorSide &s, MpfrMatrix &result) {
        MirrorPblasCtx mpc;
        if (!mpc.init(s, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         s.label.c_str());
            return;
        }

        int loc_rA = mpc.local_rows(m);
        int loc_rC = mpc.local_rows(n);
        int lld_a = std::max(1, loc_rA);
        int lld_c = std::max(1, loc_rC);

        void *A_loc = scatter_mpfr_to_local(A_mpfr, m, n,
                                             lld_a, mpc, s.ctx);
        void *C_loc = scatter_mpfr_to_local(C_mpfr, n, m,
                                             lld_c, mpc, s.ctx);
        void *alpha_n = mpfr_scalar_to_native(alpha_mpfr, s.ctx);
        void *beta_n  = mpfr_scalar_to_native(beta_mpfr, s.ctx);

        int desc_a[9], desc_c[9];
        mpc.make_desc(desc_a, m, n, lld_a);
        mpc.make_desc(desc_c, n, m, lld_c);

        int one = 1;
        auto *fn = reinterpret_cast<ptran_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&m, &n,
           alpha_n,
           A_loc, &one, &one, desc_a,
           beta_n,
           C_loc, &one, &one, desc_c);

        gather_local_to_mpfr(result, C_loc, lld_c, n, m, mpc, s.ctx);

        std::free(A_loc);
        std::free(C_loc);
        std::free(alpha_n);
        std::free(beta_n);
        mpc.finalize();
    };

    MpfrMatrix res_a(n, m, prec); run_side(a, res_a);
    MpfrMatrix res_b(n, m, prec); run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    char ps[128];
    std::snprintf(ps, sizeof(ps), "m=%d n=%d", m, n);
    mirror_report_result("PTRAN", ps, err, nullptr, nullptr, config);
}
