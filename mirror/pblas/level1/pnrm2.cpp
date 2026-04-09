/* pnrm2.cpp -- Mirror tester for PBLAS Level 1 PNRM2 (2-norm) */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*pnrm2_fn_t)(
    const int *n,
    void *norm2,
    const void *x, const int *ix, const int *jx, const int *descx,
    const int *incx);

void mirror_test_pnrm2(const MirrorSide &a, const MirrorSide &b,
                          const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;

    MpfrMatrix x_mpfr(n, 1, prec);
    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);

    auto run_side = [&](const MirrorSide &s, MpfrScalar &result) {
        MirrorPblasCtx mpc;
        if (!mpc.init(s, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         s.label.c_str());
            return;
        }

        int loc_rx = mpc.local_rows(n);
        int lld_x = std::max(1, loc_rx);

        void *x_loc = scatter_mpfr_to_local(x_mpfr, n, 1, lld_x, mpc, s.ctx);
        void *norm_buf = std::calloc(1, s.ctx.typesize);

        int desc_x[9];
        mpc.make_desc(desc_x, n, 1, lld_x);

        int one = 1;
        auto *fn = reinterpret_cast<pnrm2_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&n, norm_buf, x_loc, &one, &one, desc_x, &one);

        s.ctx.to_mpfr(result.get(), norm_buf);

        std::free(x_loc);
        std::free(norm_buf);
        mpc.finalize();
    };

    MpfrScalar res_a(prec); run_side(a, res_a);
    MpfrScalar res_b(prec); run_side(b, res_b);

    const MpfrScalar &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrScalar &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_scalar(ref, tst, prec);

    char ps[128];
    std::snprintf(ps, sizeof(ps), "n=%d", n);
    mirror_report_result("PNRM2", ps, err, nullptr, nullptr, config);
}
