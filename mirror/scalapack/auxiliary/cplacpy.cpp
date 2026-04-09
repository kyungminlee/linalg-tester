/* cplacpy.cpp -- Mirror tester for ScaLAPACK complex PLACPY (distributed matrix copy) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*placpy_fn_t)(
    const char *uplo, const int *m, const int *n,
    const void *A, const int *ia, const int *ja, const int *desca,
    void *B, const int *ib, const int *jb, const int *descb,
    std::size_t);

void mirror_test_cplacpy(const MirrorSide &a, const MirrorSide &b,
                          const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;
    char uplo = 'A';

    unsigned seed_A = params.seed;

    MpfrComplexMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &side, MpfrComplexMatrix &result) {
        MirrorPblasCtx mpc;
        if (!mpc.init(side, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         side.label.c_str());
            return;
        }

        int loc_rA = mpc.local_rows(m);
        int loc_rB = mpc.local_rows(m);
        int lld_a = std::max(1, loc_rA);
        int lld_b = std::max(1, loc_rB);

        void *A_loc = scatter_mpfr_complex_to_local(A_mpfr, m, n, lld_a, mpc, side.ctx);

        int loc_nB = mpc.local_cols(n);
        void *B_loc = std::calloc(
            static_cast<std::size_t>(lld_b) * std::max(1, loc_nB), side.ctx.typesize);

        int desc_a[9], desc_b[9];
        mpc.make_desc(desc_a, m, n, lld_a);
        mpc.make_desc(desc_b, m, n, lld_b);

        int one = 1;

        auto *fn = reinterpret_cast<placpy_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&uplo, &m, &n,
           A_loc, &one, &one, desc_a,
           B_loc, &one, &one, desc_b,
           (std::size_t)1);

        gather_local_complex_to_mpfr(result, B_loc, lld_b, m, n, mpc, side.ctx);

        std::free(A_loc);
        std::free(B_loc);
        mpc.finalize();
    };

    MpfrComplexMatrix res_a(m, n, prec); run_side(a, res_a);
    MpfrComplexMatrix res_b(m, n, prec); run_side(b, res_b);

    const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

    mirror_report_result("CPLACPY", "uplo=A", err, nullptr, nullptr, config);
}
