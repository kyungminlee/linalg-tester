/* cptrtri.cpp -- Mirror tester for ScaLAPACK complex PTRTRI (distributed triangular inverse) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*ptrtri_fn_t)(
    const char *uplo, const char *diag, const int *n,
    void *A, const int *ia, const int *ja, const int *desca,
    int *info, std::size_t, std::size_t);

void mirror_test_cptrtri(const MirrorSide &a, const MirrorSide &b,
                          const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;
    char diag = 'N';

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        MpfrComplexMatrix A_mpfr(n, n, prec);
        gen_mpfr_triangular_complex_matrix(A_mpfr, uplo, diag, prec, &seed_A);

        auto run_side = [&](const MirrorSide &side, MpfrComplexMatrix &result) {
            MirrorPblasCtx mpc;
            if (!mpc.init(side, mb, nb)) {
                std::fprintf(stderr, "BLACS init failed for side %s\n",
                             side.label.c_str());
                return;
            }

            int loc_rA = mpc.local_rows(n);
            int lld_a = std::max(1, loc_rA);

            void *A_loc = scatter_mpfr_complex_to_local(A_mpfr, n, n, lld_a, mpc, side.ctx);

            int desc_a[9];
            mpc.make_desc(desc_a, n, n, lld_a);

            int one = 1;
            int info = 0;

            auto *fn = reinterpret_cast<ptrtri_fn_t>(
                load_sym(side.lib, side.sym.c_str()));
            fn(&uplo, &diag, &n, A_loc, &one, &one, desc_a,
               &info, (std::size_t)1, (std::size_t)1);

            gather_local_complex_to_mpfr(result, A_loc, lld_a, n, n, mpc, side.ctx);

            std::free(A_loc);
            mpc.finalize();
        };

        MpfrComplexMatrix res_a(n, n, prec); run_side(a, res_a);
        MpfrComplexMatrix res_b(n, n, prec); run_side(b, res_b);

        const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

        char ps[128];
        std::snprintf(ps, sizeof(ps), "uplo=%c diag=%c", uplo, diag);
        mirror_report_result("CPTRTRI", ps, err, nullptr, nullptr, config);
    }
}
