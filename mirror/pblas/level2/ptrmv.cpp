/* ptrmv.cpp -- Mirror tester for PBLAS Level 2 PTRMV */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*ptrmv_fn_t)(
    const char *uplo, const char *trans, const char *diag,
    const int *n,
    const void *A, const int *ia, const int *ja, const int *desca,
    void *x, const int *ix, const int *jx, const int *descx,
    const int *incx,
    std::size_t, std::size_t, std::size_t);

void mirror_test_ptrmv(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T', 'C'}) {
    for (char diag : {'N', 'U'}) {
        unsigned seed_A = params.seed;
        unsigned seed_x = params.seed + 1;

        MpfrMatrix A_mpfr(n, n, prec);
        MpfrMatrix x_mpfr(n, 1, prec);

        gen_mpfr_triangular_matrix(A_mpfr, uplo, diag, prec, &seed_A);
        gen_mpfr_random_vector(x_mpfr, prec, &seed_x);

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

            int desc_a[9], desc_x[9];
            mpc.make_desc(desc_a, n, n, lld_a);
            mpc.make_desc(desc_x, n, 1, lld_x);

            int one = 1;
            auto *fn = reinterpret_cast<ptrmv_fn_t>(
                load_sym(s.lib, s.sym.c_str()));
            fn(&uplo, &trans, &diag, &n,
               A_loc, &one, &one, desc_a,
               x_loc, &one, &one, desc_x, &one,
               (std::size_t)1, (std::size_t)1, (std::size_t)1);

            gather_local_to_mpfr(result, x_loc, lld_x, n, 1, mpc, s.ctx);

            std::free(A_loc);
            std::free(x_loc);
            mpc.finalize();
        };

        MpfrMatrix res_a(n, 1, prec); run_side(a, res_a);
        MpfrMatrix res_b(n, 1, prec); run_side(b, res_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

        char ps[128];
        std::snprintf(ps, sizeof(ps), "uplo=%c trans=%c diag=%c n=%d",
                      uplo, trans, diag, n);
        mirror_report_result("PTRMV", ps, err, nullptr, nullptr, config);
    }}}
}
