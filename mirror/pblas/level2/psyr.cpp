/* psyr.cpp -- Mirror tester for PBLAS Level 2 PSYR (symmetric rank-1) */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*psyr_fn_t)(
    const char *uplo,
    const int *n,
    const void *alpha,
    const void *x, const int *ix, const int *jx, const int *descx,
    const int *incx,
    void *A, const int *ia, const int *ja, const int *desca,
    std::size_t);

void mirror_test_psyr(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A  = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_al = params.seed + 2;

        MpfrMatrix A_mpfr(n, n, prec);
        MpfrMatrix x_mpfr(n, 1, prec);
        MpfrScalar alpha_mpfr(prec);

        gen_mpfr_symmetric_matrix(A_mpfr, uplo, prec, &seed_A);
        gen_mpfr_random_vector(x_mpfr, prec, &seed_x);
        gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_al);

        auto run_side = [&](const MirrorSide &s, MpfrMatrix &result) {
            MirrorPblasCtx mpc;
            if (!mpc.init(s, mb, nb)) {
                std::fprintf(stderr, "BLACS init failed for side %s\n",
                             s.label.c_str());
                return;
            }

            int loc_rA = mpc.local_rows(n);
            int loc_rx = mpc.local_rows(n);
            int lld_a = std::max(1, loc_rA);
            int lld_x = std::max(1, loc_rx);

            void *A_loc = scatter_mpfr_to_local(A_mpfr, n, n,
                                                 lld_a, mpc, s.ctx);
            void *x_loc = scatter_mpfr_to_local(x_mpfr, n, 1,
                                                 lld_x, mpc, s.ctx);
            void *alpha_n = mpfr_scalar_to_native(alpha_mpfr, s.ctx);

            int desc_a[9], desc_x[9];
            mpc.make_desc(desc_a, n, n, lld_a);
            mpc.make_desc(desc_x, n, 1, lld_x);

            int one = 1;
            auto *fn = reinterpret_cast<psyr_fn_t>(
                load_sym(s.lib, s.sym.c_str()));
            fn(&uplo, &n,
               alpha_n,
               x_loc, &one, &one, desc_x, &one,
               A_loc, &one, &one, desc_a,
               (std::size_t)1);

            gather_local_to_mpfr(result, A_loc, lld_a, n, n, mpc, s.ctx);

            std::free(A_loc);
            std::free(x_loc);
            std::free(alpha_n);
            mpc.finalize();
        };

        MpfrMatrix res_a(n, n, prec); run_side(a, res_a);
        MpfrMatrix res_b(n, n, prec); run_side(b, res_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

        char ps[128];
        std::snprintf(ps, sizeof(ps), "uplo=%c n=%d", uplo, n);
        mirror_report_result("PSYR", ps, err, nullptr, nullptr, config);
    }
}
