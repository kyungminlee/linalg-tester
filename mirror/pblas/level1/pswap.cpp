/* pswap.cpp -- Mirror tester for PBLAS Level 1 PSWAP (vector swap) */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*pswap_fn_t)(
    const int *n,
    void *x, const int *ix, const int *jx, const int *descx,
    const int *incx,
    void *y, const int *iy, const int *jy, const int *descy,
    const int *incy);

void mirror_test_pswap(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;
    unsigned seed_y = params.seed + 1;

    MpfrMatrix x_mpfr(n, 1, prec);
    MpfrMatrix y_mpfr(n, 1, prec);

    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_vector(y_mpfr, prec, &seed_y);

    auto run_side = [&](const MirrorSide &s, MpfrMatrix &res_x,
                        MpfrMatrix &res_y) {
        MirrorPblasCtx mpc;
        if (!mpc.init(s, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         s.label.c_str());
            return;
        }

        int loc_rx = mpc.local_rows(n);
        int loc_ry = mpc.local_rows(n);
        int lld_x = std::max(1, loc_rx);
        int lld_y = std::max(1, loc_ry);

        void *x_loc = scatter_mpfr_to_local(x_mpfr, n, 1, lld_x, mpc, s.ctx);
        void *y_loc = scatter_mpfr_to_local(y_mpfr, n, 1, lld_y, mpc, s.ctx);

        int desc_x[9], desc_y[9];
        mpc.make_desc(desc_x, n, 1, lld_x);
        mpc.make_desc(desc_y, n, 1, lld_y);

        int one = 1;
        auto *fn = reinterpret_cast<pswap_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&n,
           x_loc, &one, &one, desc_x, &one,
           y_loc, &one, &one, desc_y, &one);

        gather_local_to_mpfr(res_x, x_loc, lld_x, n, 1, mpc, s.ctx);
        gather_local_to_mpfr(res_y, y_loc, lld_y, n, 1, mpc, s.ctx);

        std::free(x_loc);
        std::free(y_loc);
        mpc.finalize();
    };

    MpfrMatrix rx_a(n, 1, prec), ry_a(n, 1, prec);
    MpfrMatrix rx_b(n, 1, prec), ry_b(n, 1, prec);
    run_side(a, rx_a, ry_a);
    run_side(b, rx_b, ry_b);

    /* Compare x after swap */
    {
        const MpfrMatrix &ref = (config.reference == "a") ? rx_a : rx_b;
        const MpfrMatrix &tst = (config.reference == "a") ? rx_b : rx_a;
        ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);
        char ps[128];
        std::snprintf(ps, sizeof(ps), "n=%d output=x", n);
        mirror_report_result("PSWAP", ps, err, nullptr, nullptr, config);
    }
    /* Compare y after swap */
    {
        const MpfrMatrix &ref = (config.reference == "a") ? ry_a : ry_b;
        const MpfrMatrix &tst = (config.reference == "a") ? ry_b : ry_a;
        ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);
        char ps[128];
        std::snprintf(ps, sizeof(ps), "n=%d output=y", n);
        mirror_report_result("PSWAP", ps, err, nullptr, nullptr, config);
    }
}
