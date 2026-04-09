/* pscal.cpp -- Mirror tester for PBLAS Level 1 PSCAL (vector scale) */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*pscal_fn_t)(
    const int *n,
    const void *alpha,
    void *x, const int *ix, const int *jx, const int *descx,
    const int *incx);

void mirror_test_pscal(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x  = params.seed;
    unsigned seed_al = params.seed + 1;

    MpfrMatrix x_mpfr(n, 1, prec);
    MpfrScalar alpha_mpfr(prec);

    gen_mpfr_random_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_al);

    auto run_side = [&](const MirrorSide &s, MpfrMatrix &result) {
        MirrorPblasCtx mpc;
        if (!mpc.init(s, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         s.label.c_str());
            return;
        }

        int loc_rx = mpc.local_rows(n);
        int lld_x = std::max(1, loc_rx);

        void *x_loc = scatter_mpfr_to_local(x_mpfr, n, 1, lld_x, mpc, s.ctx);
        void *alpha_n = mpfr_scalar_to_native(alpha_mpfr, s.ctx);

        int desc_x[9];
        mpc.make_desc(desc_x, n, 1, lld_x);

        int one = 1;
        auto *fn = reinterpret_cast<pscal_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&n, alpha_n, x_loc, &one, &one, desc_x, &one);

        gather_local_to_mpfr(result, x_loc, lld_x, n, 1, mpc, s.ctx);

        std::free(x_loc);
        std::free(alpha_n);
        mpc.finalize();
    };

    MpfrMatrix res_a(n, 1, prec); run_side(a, res_a);
    MpfrMatrix res_b(n, 1, prec); run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

    char ps[128];
    std::snprintf(ps, sizeof(ps), "n=%d", n);
    mirror_report_result("PSCAL", ps, err, nullptr, nullptr, config);
}
