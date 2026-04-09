/* ppotrf.cpp -- Mirror tester for ScaLAPACK PPOTRF (distributed Cholesky factorization) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*ppotrf_fn_t)(
    const char *uplo, const int *n, void *A, const int *ia, const int *ja,
    const int *desca, int *info, std::size_t uplo_len);

void mirror_test_ppotrf(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;
        MpfrMatrix A_mpfr(n, n, prec);
        gen_mpfr_positive_definite_matrix(A_mpfr, prec, &seed_A);

        auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
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

            int info = 0, one = 1;

            auto *fn = reinterpret_cast<ppotrf_fn_t>(
                load_sym(side.lib, side.sym.c_str()));
            fn(&uplo, &n, A_loc, &one, &one, desc_a, &info, (std::size_t)1);

            gather_local_to_mpfr(result, A_loc, lld_a, n, n, mpc, side.ctx);

            std::free(A_loc);
            mpc.finalize();
        };

        MpfrMatrix res_a(n, n, prec); run_side(a, res_a);
        MpfrMatrix res_b(n, n, prec); run_side(b, res_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

        char ps[128];
        std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
        mirror_report_result("PPOTRF", ps, err, nullptr, nullptr, config);
    }
}
