/* pgetrf.cpp -- Mirror tester for ScaLAPACK PGETRF (distributed LU factorization) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*pgetrf_fn_t)(
    const int *m, const int *n, void *A, const int *ia, const int *ja,
    const int *desca, int *ipiv, int *info);

void mirror_test_pgetrf(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_A = params.seed;
    MpfrMatrix A_mpfr(n, n, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

    bool ipiv_match = true;

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result,
                        std::vector<int> &ipiv_out) {
        MirrorPblasCtx mpc;
        if (!mpc.init(side, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         side.label.c_str());
            return;
        }

        int loc_m = mpc.local_rows(n);
        int lld_a = std::max(1, loc_m);

        void *A_loc = scatter_mpfr_to_local(A_mpfr, n, n, lld_a, mpc, side.ctx);

        int desc_a[9];
        mpc.make_desc(desc_a, n, n, lld_a);

        int ipiv_len = loc_m + mb;
        int *ipiv = static_cast<int *>(std::calloc(ipiv_len, sizeof(int)));
        int info = 0, one = 1;

        auto *fn = reinterpret_cast<pgetrf_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&n, &n, A_loc, &one, &one, desc_a, ipiv, &info);

        gather_local_to_mpfr(result, A_loc, lld_a, n, n, mpc, side.ctx);

        ipiv_out.assign(ipiv, ipiv + ipiv_len);

        std::free(A_loc);
        std::free(ipiv);
        mpc.finalize();
    };

    MpfrMatrix res_a(n, n, prec);
    MpfrMatrix res_b(n, n, prec);
    std::vector<int> ipiv_a, ipiv_b;
    run_side(a, res_a, ipiv_a);
    run_side(b, res_b, ipiv_b);

    ipiv_match = (ipiv_a == ipiv_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    char ps[128];
    std::snprintf(ps, sizeof(ps), "n=%d ipiv_match=%s ref=%s",
                  n, ipiv_match ? "yes" : "NO",
                  (config.reference == "a") ? "A" : "B");
    mirror_report_result("PGETRF", ps, err, nullptr, nullptr, config);
}
