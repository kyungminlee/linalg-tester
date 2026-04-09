/* pgesv.cpp -- Mirror tester for ScaLAPACK PGESV (distributed general solve) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*pgesv_fn_t)(
    const int *n, const int *nrhs,
    void *A, const int *ia, const int *ja, const int *desca,
    int *ipiv,
    void *B, const int *ib, const int *jb, const int *descb,
    int *info);

void mirror_test_pgesv(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int nrhs = std::min(n, 4);
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_A = params.seed, seed_B = params.seed + 1;
    MpfrMatrix A_mpfr(n, n, prec);
    MpfrMatrix B_mpfr(n, nrhs, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);
    gen_mpfr_random_matrix(B_mpfr, prec, &seed_B);

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
        MirrorPblasCtx mpc;
        if (!mpc.init(side, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         side.label.c_str());
            return;
        }

        int loc_m = mpc.local_rows(n);
        int loc_nrhs = mpc.local_cols(nrhs);
        int lld_a = std::max(1, loc_m);
        int lld_b = std::max(1, loc_m);

        void *A_loc = scatter_mpfr_to_local(A_mpfr, n, n, lld_a, mpc, side.ctx);
        void *B_loc = scatter_mpfr_to_local(B_mpfr, n, nrhs, lld_b, mpc, side.ctx);

        int desc_a[9], desc_b[9];
        mpc.make_desc(desc_a, n, n, lld_a);
        mpc.make_desc(desc_b, n, nrhs, lld_b);

        int ipiv_len = loc_m + mb;
        int *ipiv = static_cast<int *>(std::calloc(ipiv_len, sizeof(int)));
        int info = 0, one = 1;

        auto *fn = reinterpret_cast<pgesv_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&n, &nrhs, A_loc, &one, &one, desc_a, ipiv,
           B_loc, &one, &one, desc_b, &info);

        gather_local_to_mpfr(result, B_loc, lld_b, n, nrhs, mpc, side.ctx);

        std::free(A_loc);
        std::free(B_loc);
        std::free(ipiv);
        mpc.finalize();
    };

    MpfrMatrix res_a(n, nrhs, prec); run_side(a, res_a);
    MpfrMatrix res_b(n, nrhs, prec); run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    char ps[128];
    std::snprintf(ps, sizeof(ps), "n=%d nrhs=%d ref=%s", n, nrhs,
                  (config.reference == "a") ? "A" : "B");
    mirror_report_result("PGESV", ps, err, nullptr, nullptr, config);
}
