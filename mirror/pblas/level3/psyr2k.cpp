/* psyr2k.cpp -- Mirror tester for PBLAS Level 3 PSYR2K */

#include "../mirror_pblas_common.h"

#include <cctype>

extern "C" typedef void (*psyr2k_fn_t)(
    const char *uplo, const char *trans,
    const int *n, const int *k,
    const void *alpha,
    const void *A, const int *ia, const int *ja, const int *desca,
    const void *B, const int *ib, const int *jb, const int *descb,
    const void *beta,
    void *C, const int *ic, const int *jc, const int *descc,
    std::size_t, std::size_t);

void mirror_test_psyr2k(const MirrorSide &a, const MirrorSide &b,
                          const TestParams &params, const MirrorConfig &config)
{
    int n = params.n, k = params.k;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T', 'C'}) {
        char trans_eff = (std::toupper(static_cast<unsigned char>(trans)) == 'C')
                         ? 'T' : std::toupper(static_cast<unsigned char>(trans));

        int rows_AB = (trans_eff == 'N') ? n : k;
        int cols_AB = (trans_eff == 'N') ? k : n;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        MpfrMatrix A_mpfr(rows_AB, cols_AB, prec);
        MpfrMatrix B_mpfr(rows_AB, cols_AB, prec);
        MpfrMatrix C_mpfr(n, n, prec);
        MpfrScalar alpha_mpfr(prec), beta_mpfr(prec);

        gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);
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

            int loc_rAB = mpc.local_rows(rows_AB);
            int loc_rC  = mpc.local_rows(n);
            int lld_ab = std::max(1, loc_rAB);
            int lld_c  = std::max(1, loc_rC);

            void *A_loc = scatter_mpfr_to_local(A_mpfr, rows_AB, cols_AB,
                                                 lld_ab, mpc, s.ctx);
            void *B_loc = scatter_mpfr_to_local(B_mpfr, rows_AB, cols_AB,
                                                 lld_ab, mpc, s.ctx);
            void *C_loc = scatter_mpfr_to_local(C_mpfr, n, n,
                                                 lld_c, mpc, s.ctx);
            void *alpha_n = mpfr_scalar_to_native(alpha_mpfr, s.ctx);
            void *beta_n  = mpfr_scalar_to_native(beta_mpfr, s.ctx);

            int desc_a[9], desc_b[9], desc_c[9];
            mpc.make_desc(desc_a, rows_AB, cols_AB, lld_ab);
            mpc.make_desc(desc_b, rows_AB, cols_AB, lld_ab);
            mpc.make_desc(desc_c, n, n, lld_c);

            int one = 1;
            auto *fn = reinterpret_cast<psyr2k_fn_t>(
                load_sym(s.lib, s.sym.c_str()));
            fn(&uplo, &trans, &n, &k,
               alpha_n,
               A_loc, &one, &one, desc_a,
               B_loc, &one, &one, desc_b,
               beta_n,
               C_loc, &one, &one, desc_c,
               (std::size_t)1, (std::size_t)1);

            gather_local_to_mpfr(result, C_loc, lld_c, n, n, mpc, s.ctx);

            std::free(A_loc);
            std::free(B_loc);
            std::free(C_loc);
            std::free(alpha_n);
            std::free(beta_n);
            mpc.finalize();
        };

        MpfrMatrix res_a(n, n, prec); run_side(a, res_a);
        MpfrMatrix res_b(n, n, prec); run_side(b, res_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

        char ps[128];
        std::snprintf(ps, sizeof(ps), "uplo=%c trans=%c", uplo, trans);
        mirror_report_result("PSYR2K", ps, err, nullptr, nullptr, config);
    }}
}
