/* pdot.cpp -- Mirror tester for PBLAS Level 1 PDOT (dot product) */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*pdot_fn_t)(
    const int *n,
    void *dot,
    const void *x, const int *ix, const int *jx, const int *descx,
    const int *incx,
    const void *y, const int *iy, const int *jy, const int *descy,
    const int *incy);

void mirror_test_pdot(const MirrorSide &a, const MirrorSide &b,
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

    auto run_side = [&](const MirrorSide &s, MpfrScalar &result) {
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
        void *dot_buf = std::calloc(1, s.ctx.typesize);

        int desc_x[9], desc_y[9];
        mpc.make_desc(desc_x, n, 1, lld_x);
        mpc.make_desc(desc_y, n, 1, lld_y);

        int one = 1;
        auto *fn = reinterpret_cast<pdot_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&n, dot_buf,
           x_loc, &one, &one, desc_x, &one,
           y_loc, &one, &one, desc_y, &one);

        s.ctx.to_mpfr(result.get(), dot_buf);

        std::free(x_loc);
        std::free(y_loc);
        std::free(dot_buf);
        mpc.finalize();
    };

    MpfrScalar res_a(prec); run_side(a, res_a);
    MpfrScalar res_b(prec); run_side(b, res_b);

    const MpfrScalar &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrScalar &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_scalar(ref, tst, prec);

    char ps[128];
    std::snprintf(ps, sizeof(ps), "n=%d", n);
    mirror_report_result("PDOT", ps, err, nullptr, nullptr, config);
}
