/* cpgemr2d.cpp -- Mirror tester for ScaLAPACK complex PGEMR2D (distributed matrix redistribution) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*cpgemr2d_fn_t)(
    const int *m, const int *n,
    const void *A, const int *ia, const int *ja, const int *desca,
    void *B, const int *ib, const int *jb, const int *descb,
    const int *ictxt
);

void mirror_test_cpgemr2d(const MirrorSide &a, const MirrorSide &b,
                            const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int mb1 = 4, nb1 = 4;   /* source block sizes */
    int mb2 = 3, nb2 = 3;   /* destination block sizes */
    mpfr_prec_t prec = config.prec;

    unsigned seed_A = params.seed;

    MpfrComplexMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &s, MpfrComplexMatrix &result) {
        MirrorPblasCtx mpc;
        if (!mpc.init(s, mb1, nb1)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         s.label.c_str());
            return;
        }

        int loc_rA = mpc.local_rows(m);
        int lld_a = std::max(1, loc_rA);
        int lld_b = std::max(1, m);

        void *A_loc = scatter_mpfr_complex_to_local(A_mpfr, m, n,
                                                      lld_a, mpc, s.ctx);
        void *B_loc = std::calloc(
            static_cast<std::size_t>(lld_b) * std::max(1, n), s.ctx.typesize);

        /* Source descriptor with (mb1, nb1) */
        int desc_a[9] = {1, mpc.pc.bc.ictxt, m, n, mb1, nb1, 0, 0, lld_a};
        /* Destination descriptor with (mb2, nb2) on same context */
        int desc_b[9] = {1, mpc.pc.bc.ictxt, m, n, mb2, nb2, 0, 0, lld_b};

        int one = 1;
        int ictxt = mpc.pc.bc.ictxt;

        auto *fn = reinterpret_cast<cpgemr2d_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&m, &n,
           A_loc, &one, &one, desc_a,
           B_loc, &one, &one, desc_b,
           &ictxt);

        gather_local_complex_to_mpfr(result, B_loc, lld_b, m, n, mpc, s.ctx);

        std::free(A_loc);
        std::free(B_loc);
        mpc.finalize();
    };

    MpfrComplexMatrix res_a(m, n, prec);
    MpfrComplexMatrix res_b(m, n, prec);
    run_side(a, res_a);
    run_side(b, res_b);

    const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "m=%d n=%d", m, n);
    mirror_report_result("CPGEMR2D", params_str, err,
                          nullptr, nullptr, config);
}
