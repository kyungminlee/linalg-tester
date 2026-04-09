/* pdotc.cpp -- Mirror tester for PBLAS Level 1 PDOTC (complex conjugated dot) */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*pdotc_fn_t)(
    const int *n,
    void *dotc,
    const void *x, const int *ix, const int *jx, const int *descx,
    const int *incx,
    const void *y, const int *iy, const int *jy, const int *descy,
    const int *incy);

void mirror_test_pdotc(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_x = params.seed;
    unsigned seed_y = params.seed + 1;

    MpfrComplexMatrix x_mpfr(n, 1, prec);
    MpfrComplexMatrix y_mpfr(n, 1, prec);

    gen_mpfr_random_complex_vector(x_mpfr, prec, &seed_x);
    gen_mpfr_random_complex_vector(y_mpfr, prec, &seed_y);

    auto run_side = [&](const MirrorSide &s, MpfrComplexScalar &result) {
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

        void *x_loc = scatter_mpfr_complex_to_local(x_mpfr, n, 1,
                                                     lld_x, mpc, s.ctx);
        void *y_loc = scatter_mpfr_complex_to_local(y_mpfr, n, 1,
                                                     lld_y, mpc, s.ctx);
        void *dot_buf = std::calloc(1, s.ctx.typesize);

        int desc_x[9], desc_y[9];
        mpc.make_desc(desc_x, n, 1, lld_x);
        mpc.make_desc(desc_y, n, 1, lld_y);

        int one = 1;
        auto *fn = reinterpret_cast<pdotc_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&n, dot_buf,
           x_loc, &one, &one, desc_x, &one,
           y_loc, &one, &one, desc_y, &one);

        s.ctx.to_mpfr_complex(result.re(), result.im(), dot_buf);

        std::free(x_loc);
        std::free(y_loc);
        std::free(dot_buf);
        mpc.finalize();
    };

    MpfrComplexScalar res_a(prec); run_side(a, res_a);
    MpfrComplexScalar res_b(prec); run_side(b, res_b);

    const MpfrComplexScalar &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrComplexScalar &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_complex_scalar(ref, tst, prec);

    char ps[128];
    std::snprintf(ps, sizeof(ps), "n=%d", n);
    mirror_report_result("PDOTC", ps, err, nullptr, nullptr, config);
}
