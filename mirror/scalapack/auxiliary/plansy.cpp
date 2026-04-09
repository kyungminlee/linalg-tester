/* plansy.cpp -- Mirror tester for ScaLAPACK PLANSY (distributed symmetric matrix norm) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef float (*plansy_s_fn_t)(
    const char *norm, const char *uplo, const int *n,
    const void *A, const int *ia, const int *ja, const int *desca,
    void *work, std::size_t, std::size_t);

extern "C" typedef double (*plansy_d_fn_t)(
    const char *norm, const char *uplo, const int *n,
    const void *A, const int *ia, const int *ja, const int *desca,
    void *work, std::size_t, std::size_t);

extern "C" typedef long double (*plansy_l_fn_t)(
    const char *norm, const char *uplo, const int *n,
    const void *A, const int *ia, const int *ja, const int *desca,
    void *work, std::size_t, std::size_t);

void mirror_test_plansy(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char norm : {'M', '1', 'I', 'F'}) {
        for (char uplo : {'U', 'L'}) {
            unsigned seed_A = params.seed;

            MpfrMatrix A_mpfr(n, n, prec);
            gen_mpfr_symmetric_matrix(A_mpfr, uplo, prec, &seed_A);

            auto run_side = [&](const MirrorSide &side, MpfrScalar &result) {
                MirrorPblasCtx mpc;
                if (!mpc.init(side, mb, nb)) {
                    std::fprintf(stderr, "BLACS init failed for side %s\n",
                                 side.label.c_str());
                    return;
                }

                int loc_rA = mpc.local_rows(n);
                int loc_cA = mpc.local_cols(n);
                int lld_a = std::max(1, loc_rA);

                void *A_loc = scatter_mpfr_to_local(A_mpfr, n, n, lld_a, mpc, side.ctx);

                int desc_a[9];
                mpc.make_desc(desc_a, n, n, lld_a);

                int work_size = std::max(loc_rA, loc_cA) + 1;
                void *work = std::calloc(work_size, side.ctx.typesize);

                int one = 1;

                if (side.ctx.typesize == 4) {
                    auto *fn = reinterpret_cast<plansy_s_fn_t>(
                        load_sym(side.lib, side.sym.c_str()));
                    float val = fn(&norm, &uplo, &n,
                                   A_loc, &one, &one, desc_a,
                                   work, (std::size_t)1, (std::size_t)1);
                    mpfr_set_flt(result.get(), val, MPFR_RNDN);
                } else if (side.ctx.typesize == 8) {
                    auto *fn = reinterpret_cast<plansy_d_fn_t>(
                        load_sym(side.lib, side.sym.c_str()));
                    double val = fn(&norm, &uplo, &n,
                                    A_loc, &one, &one, desc_a,
                                    work, (std::size_t)1, (std::size_t)1);
                    mpfr_set_d(result.get(), val, MPFR_RNDN);
                } else {
                    auto *fn = reinterpret_cast<plansy_l_fn_t>(
                        load_sym(side.lib, side.sym.c_str()));
                    long double val = fn(&norm, &uplo, &n,
                                         A_loc, &one, &one, desc_a,
                                         work, (std::size_t)1, (std::size_t)1);
                    mpfr_set_ld(result.get(), val, MPFR_RNDN);
                }

                std::free(A_loc);
                std::free(work);
                mpc.finalize();
            };

            MpfrScalar res_a(prec); run_side(a, res_a);
            MpfrScalar res_b(prec); run_side(b, res_b);

            const MpfrScalar &ref = (config.reference == "a") ? res_a : res_b;
            const MpfrScalar &tst = (config.reference == "a") ? res_b : res_a;
            ErrorResult err = compute_error_mpfr_scalar(ref, tst, prec);

            char ps[128];
            std::snprintf(ps, sizeof(ps), "norm=%c uplo=%c", norm, uplo);
            mirror_report_result("PLANSY", ps, err, nullptr, nullptr, config);
        }
    }
}
