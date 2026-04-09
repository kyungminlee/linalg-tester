/* pcopy.cpp -- Mirror tester for PBLAS Level 1 PCOPY (vector copy) */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*pcopy_fn_t)(
    const int *n,
    const void *x, const int *ix, const int *jx, const int *descx,
    const int *incx,
    void *y, const int *iy, const int *jy, const int *descy,
    const int *incy);

void mirror_test_pcopy(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;

    MpfrMatrix x_mpfr(n, 1, prec);
    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);

    auto run_side = [&](const MirrorSide &s, MpfrMatrix &result) {
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
        void *y_loc = std::calloc(
            static_cast<std::size_t>(lld_y) * std::max(1, mpc.local_cols(1)),
            s.ctx.typesize);

        int desc_x[9], desc_y[9];
        mpc.make_desc(desc_x, n, 1, lld_x);
        mpc.make_desc(desc_y, n, 1, lld_y);

        int one = 1;
        auto *fn = reinterpret_cast<pcopy_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&n,
           x_loc, &one, &one, desc_x, &one,
           y_loc, &one, &one, desc_y, &one);

        gather_local_to_mpfr(result, y_loc, lld_y, n, 1, mpc, s.ctx);

        std::free(x_loc);
        std::free(y_loc);
        mpc.finalize();
    };

    MpfrMatrix res_a(n, 1, prec); run_side(a, res_a);
    MpfrMatrix res_b(n, 1, prec); run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

    char ps[128];
    std::snprintf(ps, sizeof(ps), "n=%d", n);
    mirror_report_result("PCOPY", ps, err, nullptr, nullptr, config);
}
