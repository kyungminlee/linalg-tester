/* psymm.cpp -- Mirror tester for PBLAS Level 3 PSYMM */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*psymm_fn_t)(
    const char *side, const char *uplo,
    const int *m, const int *n,
    const void *alpha,
    const void *A, const int *ia, const int *ja, const int *desca,
    const void *B, const int *ib, const int *jb, const int *descb,
    const void *beta,
    void *C, const int *ic, const int *jc, const int *descc,
    std::size_t, std::size_t);

void mirror_test_psymm(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char side : {'L', 'R'}) {
    for (char uplo : {'U', 'L'}) {
        int ka = (side == 'L') ? m : n;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        MpfrMatrix A_mpfr(ka, ka, prec);
        MpfrMatrix B_mpfr(m, n, prec);
        MpfrMatrix C_mpfr(m, n, prec);
        MpfrScalar alpha_mpfr(prec), beta_mpfr(prec);

        gen_mpfr_symmetric_matrix(A_mpfr, uplo, prec, &seed_A);
        gen_mpfr_random_matrix(B_mpfr, prec, &seed_B);
        gen_mpfr_random_matrix(C_mpfr, prec, &seed_C);
        gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_ab);
        gen_mpfr_random_scalar(beta_mpfr, prec, &seed_ab);

        auto run_side = [&](const MirrorSide &s, MpfrMatrix &result) {
            MirrorPblasCtx mpc;
            if (!mpc.init(s, mb, nb)) {
                std::fprintf(stderr, "BLACS init failed for side %s\n",
                             s.label.c_str());
                return;
            }

            int loc_rA = mpc.local_rows(ka);
            int loc_rB = mpc.local_rows(m);
            int loc_rC = mpc.local_rows(m);
            int lld_a = std::max(1, loc_rA);
            int lld_b = std::max(1, loc_rB);
            int lld_c = std::max(1, loc_rC);

            void *A_loc = scatter_mpfr_to_local(A_mpfr, ka, ka,
                                                 lld_a, mpc, s.ctx);
            void *B_loc = scatter_mpfr_to_local(B_mpfr, m, n,
                                                 lld_b, mpc, s.ctx);
            void *C_loc = scatter_mpfr_to_local(C_mpfr, m, n,
                                                 lld_c, mpc, s.ctx);
            void *alpha_n = mpfr_scalar_to_native(alpha_mpfr, s.ctx);
            void *beta_n  = mpfr_scalar_to_native(beta_mpfr, s.ctx);

            int desc_a[9], desc_b[9], desc_c[9];
            mpc.make_desc(desc_a, ka, ka, lld_a);
            mpc.make_desc(desc_b, m, n, lld_b);
            mpc.make_desc(desc_c, m, n, lld_c);

            int one = 1;
            auto *fn = reinterpret_cast<psymm_fn_t>(
                load_sym(s.lib, s.sym.c_str()));
            fn(&side, &uplo, &m, &n,
               alpha_n,
               A_loc, &one, &one, desc_a,
               B_loc, &one, &one, desc_b,
               beta_n,
               C_loc, &one, &one, desc_c,
               (std::size_t)1, (std::size_t)1);

            gather_local_to_mpfr(result, C_loc, lld_c, m, n, mpc, s.ctx);

            std::free(A_loc);
            std::free(B_loc);
            std::free(C_loc);
            std::free(alpha_n);
            std::free(beta_n);
            mpc.finalize();
        };

        MpfrMatrix res_a(m, n, prec); run_side(a, res_a);
        MpfrMatrix res_b(m, n, prec); run_side(b, res_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

        char ps[128];
        std::snprintf(ps, sizeof(ps), "side=%c uplo=%c", side, uplo);
        mirror_report_result("PSYMM", ps, err, nullptr, nullptr, config);
    }}
}
