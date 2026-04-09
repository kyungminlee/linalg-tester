/* pger.cpp -- Mirror tester for PBLAS Level 2 PGER (rank-1 update) */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*pger_fn_t)(
    const int *m, const int *n,
    const void *alpha,
    const void *x, const int *ix, const int *jx, const int *descx,
    const int *incx,
    const void *y, const int *iy, const int *jy, const int *descy,
    const int *incy,
    void *A, const int *ia, const int *ja, const int *desca);

void mirror_test_pger(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_A  = params.seed;
    unsigned seed_x  = params.seed + 1;
    unsigned seed_y  = params.seed + 2;
    unsigned seed_al = params.seed + 3;

    MpfrMatrix A_mpfr(m, n, prec);
    MpfrMatrix x_mpfr(m, 1, prec);
    MpfrMatrix y_mpfr(n, 1, prec);
    MpfrScalar alpha_mpfr(prec);

    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);
    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_vector(y_mpfr, prec, &seed_y);
    gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_al);

    auto run_side = [&](const MirrorSide &s, MpfrMatrix &result) {
        MirrorPblasCtx mpc;
        if (!mpc.init(s, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         s.label.c_str());
            return;
        }

        int loc_rA = mpc.local_rows(m);
        int loc_rx = mpc.local_rows(m);
        int loc_ry = mpc.local_rows(n);
        int lld_a = std::max(1, loc_rA);
        int lld_x = std::max(1, loc_rx);
        int lld_y = std::max(1, loc_ry);

        void *A_loc = scatter_mpfr_to_local(A_mpfr, m, n,
                                             lld_a, mpc, s.ctx);
        void *x_loc = scatter_mpfr_to_local(x_mpfr, m, 1,
                                             lld_x, mpc, s.ctx);
        void *y_loc = scatter_mpfr_to_local(y_mpfr, n, 1,
                                             lld_y, mpc, s.ctx);
        void *alpha_n = mpfr_scalar_to_native(alpha_mpfr, s.ctx);

        int desc_a[9], desc_x[9], desc_y[9];
        mpc.make_desc(desc_a, m, n, lld_a);
        mpc.make_desc(desc_x, m, 1, lld_x);
        mpc.make_desc(desc_y, n, 1, lld_y);

        int one = 1;
        auto *fn = reinterpret_cast<pger_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&m, &n,
           alpha_n,
           x_loc, &one, &one, desc_x, &one,
           y_loc, &one, &one, desc_y, &one,
           A_loc, &one, &one, desc_a);

        gather_local_to_mpfr(result, A_loc, lld_a, m, n, mpc, s.ctx);

        std::free(A_loc);
        std::free(x_loc);
        std::free(y_loc);
        std::free(alpha_n);
        mpc.finalize();
    };

    MpfrMatrix res_a(m, n, prec); run_side(a, res_a);
    MpfrMatrix res_b(m, n, prec); run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    char ps[128];
    std::snprintf(ps, sizeof(ps), "m=%d n=%d", m, n);
    mirror_report_result("PGER", ps, err, nullptr, nullptr, config);
}
